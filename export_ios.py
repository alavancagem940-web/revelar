"""Exportação nativa para o iPhone: JPEG/PNG com sRGB e MP4 H.264 + AAC."""

from __future__ import annotations

import errno
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from PIL import Image, ImageCms

JPEG_QUALITY = 94
PHOTO_PREFIX = "foto"
VIDEO_PREFIX = "video"

H264_VIDEO_ARGS = [
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-pix_fmt",
    "yuv420p",
    "-profile:v",
    "high",
    "-level",
    "4.1",
    "-tag:v",
    "avc1",
]
AAC_AUDIO_ARGS = [
    "-c:a",
    "aac",
    "-b:a",
    "160k",
    "-ac",
    "2",
    "-ar",
    "44100",
]
FASTSTART_ARGS = ["-movflags", "+faststart"]


class FFmpegMissing(RuntimeError):
    """ffmpeg ausente — o vídeo H.264/AAC não pode ser exportado."""

    def __init__(self) -> None:
        super().__init__(
            "ffmpeg não encontrado. Instale o ffmpeg para exportar vídeo H.264 + AAC "
            "(ex.: sudo apt install ffmpeg no Ubuntu, brew install ffmpeg no macOS)."
        )


class MediaError(RuntimeError):
    """Arquivo ilegível, codec ausente ou mídia corrompida."""


def find_ffmpeg() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        return None
    return None


def require_ffmpeg() -> str:
    exe = find_ffmpeg()
    if not exe:
        raise FFmpegMissing()
    return exe


def srgb_icc_bytes() -> Optional[bytes]:
    try:
        profile = ImageCms.createProfile("sRGB")
        return ImageCms.ImageCmsProfile(profile).tobytes()
    except Exception:
        return None


def even_size(width: int, height: int) -> tuple[int, int]:
    return width - (width % 2), height - (height % 2)


def crop_even(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    nw, nh = even_size(w, h)
    if nw == w and nh == h:
        return rgb
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    return rgb[y0 : y0 + nh, x0 : x0 + nw]


def target_video_bitrate(width: int, height: int, fps: float) -> int:
    """Bitrate-alvo em bits/s: qualidade alta sem arquivo inchado (iPhone)."""
    fps = max(12.0, min(float(fps or 30.0), 60.0))
    pixels = max(1, width * height) * fps
    bitrate = int(pixels * 0.08)
    return max(800_000, min(bitrate, 12_000_000))


def next_export_path(folder: Path, kind: str, ext: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    prefix = PHOTO_PREFIX if kind == "foto" else VIDEO_PREFIX
    if not ext.startswith("."):
        ext = f".{ext}"
    n = 1
    while True:
        candidate = folder / f"revelar-{prefix}-{n:03d}{ext}"
        if not candidate.exists():
            return candidate
        n += 1


def human_os_error(exc: BaseException) -> Optional[str]:
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "Erro: disco cheio. Liberte espaço e tente de novo."
    return None


def export_photo(
    rgb: np.ndarray,
    dest: Path,
    fmt: str = "jpeg",
    quality: int = JPEG_QUALITY,
) -> Path:
    """Grava JPEG de alta qualidade ou PNG, com perfil sRGB amigável ao Fotos do iOS."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise MediaError("imagem inválida para exportação.")
    image = Image.fromarray(np.clip(rgb[..., :3], 0, 255).astype(np.uint8), mode="RGB")
    icc = srgb_icc_bytes()
    fmt = fmt.lower().strip()
    save_kwargs: dict = {}
    if icc:
        save_kwargs["icc_profile"] = icc
    try:
        if fmt == "png":
            image.save(dest, format="PNG", optimize=True, **save_kwargs)
        else:
            image.save(
                dest,
                format="JPEG",
                quality=int(quality),
                optimize=True,
                progressive=True,
                **save_kwargs,
            )
    except OSError as exc:
        nice = human_os_error(exc)
        raise MediaError(nice or f"falha ao gravar a foto: {exc}") from exc
    return dest


def _drain_stderr(proc: subprocess.Popen, bucket: list[bytes]) -> None:
    try:
        if proc.stderr is None:
            return
        for line in iter(proc.stderr.readline, b""):
            bucket.append(line)
            if len(bucket) > 400:
                del bucket[:200]
    except Exception:
        return


def _ffmpeg_error_text(bucket: list[bytes], fallback: str) -> str:
    text = b"".join(bucket[-40:]).decode("utf-8", errors="replace")
    lowered = text.lower()
    if "no space" in lowered or "no espaço" in lowered:
        return "Erro: disco cheio. Liberte espaço e tente de novo."
    if "invalid data" in lowered or "corrupt" in lowered:
        return "Erro: arquivo corrompido ou codec não suportado."
    snippet = text.strip().splitlines()
    tail = snippet[-1] if snippet else fallback
    return f"Erro no ffmpeg: {tail}"


def encode_h264_aac(
    frames_rgb: Iterable[np.ndarray],
    width: int,
    height: int,
    fps: float,
    dest: Path,
    audio_source: Optional[Path] = None,
    bitrate: Optional[int] = None,
    cancel: Optional[threading.Event] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    total_frames: Optional[int] = None,
) -> Path:
    """Um passe de encode: H.264 yuv420p + AAC, +faststart, tag avc1."""
    ffmpeg = require_ffmpeg()
    width, height = even_size(width, height)
    if width < 2 or height < 2:
        raise MediaError("vídeo pequeno demais para H.264 (yuv420p).")
    fps = float(fps if fps and fps > 1 else 30.0)
    bitrate = int(bitrate or target_video_bitrate(width, height, fps))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    with tempfile.TemporaryDirectory(prefix="revelar-") as tmp:
        tmp_dir = Path(tmp)
        video_only = tmp_dir / "video.mp4"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            *H264_VIDEO_ARGS,
            "-crf",
            "20",
            "-maxrate",
            str(bitrate),
            "-bufsize",
            str(int(bitrate * 2)),
            str(video_only),
        ]
        err: list[bytes] = []
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=10 * 1024 * 1024,
        )
        drain = threading.Thread(target=_drain_stderr, args=(proc, err), daemon=True)
        drain.start()
        written = 0
        try:
            assert proc.stdin is not None
            for frame in frames_rgb:
                if cancelled():
                    raise InterruptedError("cancelado")
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = np.ascontiguousarray(
                        Image.fromarray(frame[..., :3].astype(np.uint8)).resize(
                            (width, height), Image.Resampling.LANCZOS
                        )
                    )
                rgb = np.ascontiguousarray(frame[..., :3].astype(np.uint8))
                proc.stdin.write(rgb.tobytes())
                written += 1
                if progress and total_frames:
                    # encode ocupa a segunda metade do job (50–95%)
                    frac = min(1.0, written / max(1, total_frames))
                    progress(50.0 + 45.0 * frac, "exportando H.264")
            proc.stdin.close()
            while proc.poll() is None:
                if cancelled():
                    proc.kill()
                    raise InterruptedError("cancelado")
                time.sleep(0.15)
            code = proc.returncode
        except InterruptedError:
            try:
                proc.kill()
            except Exception:
                pass
            raise
        except BrokenPipeError as exc:
            proc.wait(timeout=30)
            raise MediaError(_ffmpeg_error_text(err, str(exc))) from exc
        except OSError as exc:
            nice = human_os_error(exc)
            raise MediaError(nice or f"falha ao gravar o vídeo: {exc}") from exc
        finally:
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            drain.join(timeout=2)

        if cancelled():
            raise InterruptedError("cancelado")
        if code != 0:
            raise MediaError(_ffmpeg_error_text(err, "falha ao codificar H.264"))
        if written == 0:
            raise MediaError("nenhum quadro para exportar.")
        if not video_only.exists() or video_only.stat().st_size < 32:
            raise MediaError(_ffmpeg_error_text(err, "arquivo H.264 vazio"))

        if progress:
            progress(96.0, "otimizando bitrate")
        mux_audio_h264(
            video_only,
            audio_source if audio_source and Path(audio_source).exists() else None,
            dest,
            ffmpeg,
            cancel,
        )
        if progress:
            progress(100.0, "concluído")
        return dest


def mux_audio_h264(
    video_mp4: Path,
    audio_source: Optional[Path],
    dest: Path,
    ffmpeg: Optional[str] = None,
    cancel: Optional[threading.Event] = None,
) -> Path:
    ffmpeg = ffmpeg or require_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    has_audio = False
    if audio_source is not None:
        has_audio = _has_audio_stream(ffmpeg, Path(audio_source))
    if has_audio:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_mp4),
            "-i",
            str(audio_source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            *AAC_AUDIO_ARGS,
            *FASTSTART_ARGS,
            "-shortest",
            str(dest),
        ]
    else:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_mp4),
            "-c:v",
            "copy",
            "-an",
            *FASTSTART_ARGS,
            str(dest),
        ]
    err: list[bytes] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    drain = threading.Thread(target=_drain_stderr, args=(proc, err), daemon=True)
    drain.start()
    while proc.poll() is None:
        if cancel is not None and cancel.is_set():
            proc.kill()
            raise InterruptedError("cancelado")
        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            continue
    drain.join(timeout=2)
    if proc.returncode != 0:
        raise MediaError(_ffmpeg_error_text(err, "falha ao muxar AAC / faststart"))
    if not dest.exists() or dest.stat().st_size < 32:
        raise MediaError("exportação vazia — verifique o espaço em disco.")
    return dest


def _has_audio_stream(ffmpeg: str, source: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    str(source),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return "audio" in (result.stdout or "").lower()
        except Exception:
            pass
    # Fallback: ffmpeg -i imprime streams no stderr
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(source)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        blob = (result.stderr or "") + (result.stdout or "")
        return "Audio:" in blob or "audio:" in blob.lower()
    except Exception:
        return False
