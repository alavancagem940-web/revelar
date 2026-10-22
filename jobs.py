"""Fila de processamento em threading.Thread — a UI nunca bloqueia."""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from export_ios import (
    FFmpegMissing,
    MediaError,
    export_photo,
    find_ffmpeg,
    human_os_error,
    next_export_path,
)
from photo_engine import enhance_photo, is_photo, is_video, load_image
from video_engine import enhance_and_export_video, enhance_frame, first_frame

# Força profissional interna — sem slider na UI.
AUTO_INTENSITY = 0.72


@dataclass
class JobRequest:
    kind: str
    source: Path
    intensity: float = AUTO_INTENSITY
    output_dir: Optional[Path] = None
    image_format: str = "jpeg"
    rgb: Optional[np.ndarray] = None
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])


@dataclass
class UiEvent:
    type: str
    job_id: str = ""
    percent: float = 0.0
    status: str = ""
    error: str = ""
    path: Optional[Path] = None
    token: int = 0
    preview_rgb: Optional[np.ndarray] = None
    output_path: Optional[Path] = None


class JobController:
    """Um worker sequencial + threads extras só para pré-visualização."""

    def __init__(self, events: queue.Queue) -> None:
        self.events = events
        self._pending: list[JobRequest] = []
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._wakeup = threading.Event()
        self._alive = True
        self._busy = False
        self._current_id: Optional[str] = None
        self._results: dict[str, Any] = {}
        self._worker = threading.Thread(
            target=self._loop, name="revelar-worker", daemon=True
        )
        self._worker.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def enqueue(self, request: JobRequest) -> str:
        with self._lock:
            self._pending.append(request)
        self._wakeup.set()
        self.events.put(
            UiEvent(
                type="queued",
                job_id=request.job_id,
                path=request.source,
                status="na fila",
            )
        )
        return request.job_id

    def cancel_current(self) -> None:
        self._cancel.set()
        with self._lock:
            job_id = self._current_id
        if job_id:
            self.events.put(
                UiEvent(type="progress", job_id=job_id, status="cancelando…", percent=-1)
            )

    def take_result(self, job_id: str) -> Any:
        with self._lock:
            return self._results.pop(job_id, None)

    def request_preview(self, path: Path, token: int) -> None:
        thread = threading.Thread(
            target=self._preview_worker,
            args=(Path(path), token),
            name="revelar-preview",
            daemon=True,
        )
        thread.start()

    def shutdown(self) -> None:
        self._alive = False
        self._cancel.set()
        self._wakeup.set()

    def _store(self, job_id: str, value: Any) -> None:
        with self._lock:
            self._results[job_id] = value

    def _loop(self) -> None:
        while self._alive:
            job = self._pop()
            if job is None:
                self._wakeup.wait(timeout=0.25)
                self._wakeup.clear()
                continue
            self._cancel.clear()
            with self._lock:
                self._busy = True
                self._current_id = job.job_id
            try:
                self._execute(job)
            except InterruptedError:
                self.events.put(
                    UiEvent(
                        type="cancelled",
                        job_id=job.job_id,
                        path=job.source,
                        status="Cancelado.",
                    )
                )
            except Exception as exc:
                self.events.put(
                    UiEvent(
                        type="error",
                        job_id=job.job_id,
                        path=job.source,
                        error=_human_error(exc),
                        status=_human_error(exc),
                    )
                )
            finally:
                with self._lock:
                    self._busy = False
                    self._current_id = None

    def _pop(self) -> Optional[JobRequest]:
        with self._lock:
            if not self._pending:
                return None
            return self._pending.pop(0)

    def _progress(self, job: JobRequest, percent: float, status: str) -> None:
        if self._cancel.is_set():
            raise InterruptedError("cancelado")
        self.events.put(
            UiEvent(
                type="progress",
                job_id=job.job_id,
                path=job.source,
                percent=max(0.0, min(100.0, percent)),
                status=status,
            )
        )

    def _execute(self, job: JobRequest) -> None:
        kind = job.kind
        src = Path(job.source)
        self._progress(job, 1.0, "lendo")

        if kind == "photo_enhance":
            rgb = job.rgb if job.rgb is not None else load_image(src)
            out = enhance_photo(
                rgb,
                job.intensity,
                cancel=self._cancel,
                progress=lambda p, s: self._progress(job, p, s),
            )
            self._store(job.job_id, out)
            self.events.put(
                UiEvent(
                    type="done",
                    job_id=job.job_id,
                    path=src,
                    status="concluído",
                    percent=100.0,
                    preview_rgb=_thumb(out),
                )
            )
            return

        if kind == "photo_export":
            rgb = job.rgb
            if rgb is None:
                rgb = load_image(src)
                rgb = enhance_photo(
                    rgb,
                    job.intensity,
                    cancel=self._cancel,
                    progress=lambda p, s: self._progress(job, p * 0.7, s),
                )
            else:
                self._progress(job, 40.0, "exportando")
            folder = Path(job.output_dir or Path.home() / "Revelar")
            ext = ".png" if job.image_format == "png" else ".jpg"
            dest = next_export_path(folder, "foto", ext)
            self._progress(job, 80.0, "exportando")
            export_photo(rgb, dest, fmt=job.image_format)
            self._store(job.job_id, rgb)
            self.events.put(
                UiEvent(
                    type="done",
                    job_id=job.job_id,
                    path=src,
                    status=f"salvo: {dest.name}",
                    percent=100.0,
                    output_path=dest,
                    preview_rgb=_thumb(rgb),
                )
            )
            return

        if kind == "video_preview":
            frame = first_frame(src)
            out = enhance_frame(frame, job.intensity)
            self._store(job.job_id, out)
            self.events.put(
                UiEvent(
                    type="done",
                    job_id=job.job_id,
                    path=src,
                    status="prévia do quadro",
                    percent=100.0,
                    preview_rgb=_thumb(out),
                )
            )
            return

        if kind == "video_export":
            if not find_ffmpeg():
                raise FFmpegMissing()
            folder = Path(job.output_dir or Path.home() / "Revelar")
            dest = next_export_path(folder, "video", ".mp4")
            last_frame: list[np.ndarray] = []

            def live(frame: np.ndarray) -> None:
                thumb = _thumb(frame, 480)
                last_frame[:] = [thumb]
                self.events.put(
                    UiEvent(
                        type="live_frame",
                        job_id=job.job_id,
                        path=src,
                        preview_rgb=thumb,
                    )
                )

            enhance_and_export_video(
                src,
                dest,
                intensity=job.intensity if job.intensity else AUTO_INTENSITY,
                cancel=self._cancel,
                progress=lambda p, s: self._progress(job, p, s),
                live_preview=live,
            )
            if last_frame:
                self._store(job.job_id, last_frame[0])
            self.events.put(
                UiEvent(
                    type="done",
                    job_id=job.job_id,
                    path=src,
                    status=f"salvo: {dest.name}",
                    percent=100.0,
                    output_path=dest,
                    preview_rgb=last_frame[0] if last_frame else None,
                )
            )
            return

        raise MediaError(f"tipo de job desconhecido: {kind}")

    def _preview_worker(self, path: Path, token: int) -> None:
        try:
            if is_photo(path):
                rgb = load_image(path)
            elif is_video(path):
                rgb = first_frame(path)
            else:
                raise MediaError(f"formato não suportado: {path.suffix}")
            self.events.put(
                UiEvent(
                    type="original_preview",
                    token=token,
                    path=path,
                    preview_rgb=_thumb(rgb, 900),
                    status="lendo",
                )
            )
        except Exception as exc:
            self.events.put(
                UiEvent(
                    type="original_preview",
                    token=token,
                    path=path,
                    error=_human_error(exc),
                    status=_human_error(exc),
                )
            )


def _thumb(rgb: np.ndarray, max_side: int = 720) -> np.ndarray:
    from photo_engine import downscale_rgb

    return downscale_rgb(rgb, max_side=max_side)


def _human_error(exc: BaseException) -> str:
    if isinstance(exc, InterruptedError):
        return "Cancelado."
    nice = human_os_error(exc)
    if nice:
        return nice
    if isinstance(exc, FFmpegMissing):
        return str(exc)
    if isinstance(exc, MediaError):
        text = str(exc)
        if not text.lower().startswith("erro"):
            return f"Erro: {text}"
        return text
    text = str(exc).strip() or exc.__class__.__name__
    lowered = text.lower()
    if "ffmpeg" in lowered:
        return (
            "Erro: ffmpeg não encontrado ou falhou. "
            "Instale o ffmpeg para exportar vídeo H.264 + AAC."
        )
    if "corrupt" in lowered or "invalid" in lowered:
        return f"Erro: arquivo corrompido ({text})"
    return f"Erro: {text}"
