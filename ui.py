"""Interface Revelar — CustomTkinter, modo escuro, 100% automático."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from export_ios import find_ffmpeg
from jobs import AUTO_INTENSITY, JobController, JobRequest, UiEvent
from photo_engine import is_photo, is_video

APP_TITLE = "Revelar"
BG = "#121212"
PANEL = "#1c1c1c"
PANEL_2 = "#242424"
TEXT = "#EDE6D6"
MUTED = "#9A907E"
ERROR = "#E07070"

FILETYPES = [
    (
        "Fotos e vídeos",
        "*.jpg *.jpeg *.png *.heic *.heif *.mp4 *.mov "
        "*.JPG *.JPEG *.PNG *.HEIC *.HEIF *.MP4 *.MOV",
    ),
    ("Fotos", "*.jpg *.jpeg *.png *.heic *.heif *.JPG *.JPEG *.PNG *.HEIC *.HEIF"),
    ("Vídeos", "*.mp4 *.mov *.MP4 *.MOV"),
    ("Todos os arquivos", "*.*"),
]


@dataclass
class QueueItem:
    path: Path
    kind: str
    status: str = "na fila"
    processed_rgb: Optional[np.ndarray] = None
    original_preview: Optional[np.ndarray] = None
    output_path: Optional[Path] = None
    error: str = ""
    job_id: Optional[str] = None
    started: bool = False


def _fit_pil(rgb: np.ndarray, max_w: int, max_h: int) -> Image.Image:
    image = Image.fromarray(np.clip(rgb[..., :3], 0, 255).astype(np.uint8), mode="RGB")
    max_w, max_h = max(8, int(max_w)), max(8, int(max_h))
    image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return image


def _caption_image(width: int, height: int, text: str) -> Image.Image:
    width, height = max(8, width), max(8, height)
    image = Image.new("RGB", (width, height), (22, 22, 22))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((18, height // 2 - 8), text, fill=(140, 132, 118), font=font)
    return image


class RevelarApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x800")
        self.minsize(960, 620)
        self.configure(fg_color=BG)

        self.events: queue.Queue[UiEvent] = queue.Queue()
        self.jobs = JobController(self.events)
        self.items: list[QueueItem] = []
        self.selected: Optional[QueueItem] = None
        self._preview_token = 0
        self._img_orig: Optional[ctk.CTkImage] = None
        self._img_res: Optional[ctk.CTkImage] = None
        self._queue_buttons: list[ctk.CTkButton] = []
        self.output_dir = Path.home() / "Revelar"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._resize_after: Optional[str] = None

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Configure>", self._on_resize)
        self.after(50, self._poll)
        self.after(200, self._load_samples)
        self._set_status(self._idle_status(), 0.0)

    def _idle_status(self) -> str:
        if find_ffmpeg():
            return "Automático. Vídeo: MP4 H.264 + AAC · foto: JPEG sRGB."
        return (
            "Automático. ffmpeg não encontrado — fotos ok; "
            "instale o ffmpeg para exportar vídeo H.264 + AAC."
        )

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._header()
        self._body()
        self._footer()

    def _header(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=72)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            bar,
            text="Revelar",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=TEXT,
        )
        title.grid(row=0, column=0, padx=(22, 12), pady=(12, 0), sticky="w")
        subtitle = ctk.CTkLabel(
            bar,
            text="100% automático · antes e depois · arquivos prontos para o iPhone",
            font=ctk.CTkFont(size=13),
            text_color=MUTED,
        )
        subtitle.grid(row=1, column=0, padx=(22, 12), pady=(0, 12), sticky="w")

        out_wrap = ctk.CTkFrame(bar, fg_color="transparent")
        out_wrap.grid(row=0, column=1, rowspan=2, padx=16, pady=12, sticky="e")
        ctk.CTkLabel(
            out_wrap, text="Pasta de saída", text_color=MUTED, font=ctk.CTkFont(size=12)
        ).pack(anchor="e")
        row = ctk.CTkFrame(out_wrap, fg_color="transparent")
        row.pack(fill="x")
        self.out_entry = ctk.CTkEntry(row, width=340, text_color=TEXT)
        self.out_entry.insert(0, str(self.output_dir))
        self.out_entry.pack(side="left", padx=(0, 8))
        ctk.CTkButton(row, text="Escolher pasta", width=140, command=self._choose_output).pack(
            side="left"
        )

    def _body(self) -> None:
        body = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew", padx=14, pady=10)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self._build_queue(body)
        self._build_preview(body)

    def _build_queue(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(parent, fg_color=PANEL, width=280, corner_radius=12)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        panel.grid_rowconfigure(2, weight=1)
        panel.grid_propagate(False)

        ctk.CTkLabel(
            panel, text="Fila", font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT
        ).grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")

        btns = ctk.CTkFrame(panel, fg_color="transparent")
        btns.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
        ctk.CTkButton(btns, text="Abrir arquivos", command=self._open_files).pack(
            fill="x", pady=(0, 6)
        )
        ctk.CTkButton(
            btns,
            text="Remover da fila",
            fg_color=PANEL_2,
            hover_color="#333",
            command=self._remove_selected,
        ).pack(fill="x")

        self.queue_scroll = ctk.CTkScrollableFrame(panel, fg_color=PANEL_2, corner_radius=8)
        self.queue_scroll.grid(row=2, column=0, padx=10, pady=(4, 12), sticky="nsew")
        self._render_queue()

    def _build_preview(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=12)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_columnconfigure(1, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(panel, text="Antes", text_color=MUTED).grid(row=0, column=0, pady=(12, 0))
        ctk.CTkLabel(panel, text="Depois", text_color=MUTED).grid(row=0, column=1, pady=(12, 0))

        self.lbl_orig = ctk.CTkLabel(panel, text="", fg_color="#141414", corner_radius=8)
        self.lbl_orig.grid(row=1, column=0, padx=(12, 6), pady=10, sticky="nsew")
        self.lbl_res = ctk.CTkLabel(panel, text="", fg_color="#141414", corner_radius=8)
        self.lbl_res.grid(row=1, column=1, padx=(6, 12), pady=10, sticky="nsew")
        self._paint_placeholders()

        self.preview_hint = ctk.CTkLabel(
            panel,
            text="Abra JPG, PNG, HEIC ou MP4/MOV. O tratamento começa sozinho.",
            text_color=MUTED,
            font=ctk.CTkFont(size=12),
        )
        self.preview_hint.grid(row=2, column=0, columnspan=2, padx=12, pady=(0, 4))
        self.export_hint = ctk.CTkLabel(
            panel,
            text=f"Pasta de saída: {self.output_dir}",
            text_color=TEXT,
            font=ctk.CTkFont(size=13),
        )
        self.export_hint.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 12))

    def _footer(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=64)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        self.btn_cancel = ctk.CTkButton(
            bar,
            text="Cancelar",
            width=120,
            fg_color="#5a2e2e",
            hover_color="#7a3a3a",
            command=self._cancel,
            state="disabled",
        )
        self.btn_cancel.grid(row=0, column=0, rowspan=2, padx=16, pady=12)

        self.progress = ctk.CTkProgressBar(bar, height=12)
        self.progress.grid(row=0, column=1, sticky="ew", padx=(0, 16), pady=(14, 2))
        self.progress.set(0)
        self.status = ctk.CTkLabel(bar, text="Pronto.", text_color=TEXT, anchor="w")
        self.status.grid(row=1, column=1, sticky="ew", padx=(0, 16), pady=(0, 10))

    def _paint_placeholders(self) -> None:
        w, h = self._preview_box()
        orig = _caption_image(w, h, "Sem pré-visualização")
        res = _caption_image(w, h, "Revelando automaticamente…")
        self._img_orig = ctk.CTkImage(light_image=orig, dark_image=orig, size=(w, h))
        self._img_res = ctk.CTkImage(light_image=res, dark_image=res, size=(w, h))
        self.lbl_orig.configure(image=self._img_orig, text="")
        self.lbl_res.configure(image=self._img_res, text="")

    def _preview_box(self) -> tuple[int, int]:
        width = max(320, (int(self.winfo_width() or 1280) - 340) // 2)
        height = max(220, int(self.winfo_height() or 800) - 280)
        return min(width, 720), min(height, 560)

    def _set_status(self, text: str, percent: Optional[float] = None, error: bool = False) -> None:
        self.status.configure(text=text, text_color=ERROR if error else TEXT)
        if percent is not None and percent >= 0:
            self.progress.set(max(0.0, min(1.0, percent / 100.0)))
        self.btn_cancel.configure(state="normal" if self.jobs.busy else "disabled")

    def _output_dir(self) -> Path:
        raw = self.out_entry.get().strip() or str(Path.home() / "Revelar")
        path = Path(raw).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self.output_dir = path
        return path

    def _choose_output(self) -> None:
        chosen = ctk.filedialog.askdirectory(
            title="Pasta de saída", initialdir=str(self._output_dir())
        )
        if chosen:
            self.out_entry.delete(0, "end")
            self.out_entry.insert(0, chosen)
            self.output_dir = Path(chosen)
            self._refresh_export_hint()

    def _refresh_export_hint(self) -> None:
        item = self.selected
        if item and item.output_path:
            self.export_hint.configure(text=f"Exportado: {item.output_path}")
        else:
            self.export_hint.configure(text=f"Pasta de saída: {self._output_dir()}")

    def _open_files(self) -> None:
        paths = ctk.filedialog.askopenfilenames(title="Abrir fotos ou vídeos", filetypes=FILETYPES)
        if not paths:
            return
        self._add_paths([Path(p) for p in paths])

    def _add_paths(self, paths: list[Path], select_first: bool = True) -> None:
        added: list[QueueItem] = []
        known = {item.path.resolve() for item in self.items}
        for path in paths:
            path = Path(path)
            if not path.is_file():
                continue
            if path.resolve() in known:
                continue
            if is_photo(path):
                kind = "foto"
            elif is_video(path):
                kind = "video"
            else:
                continue
            item = QueueItem(path=path, kind=kind)
            self.items.append(item)
            added.append(item)
            known.add(path.resolve())
        self._render_queue()
        for item in added:
            self._start_auto(item)
        if added and select_first:
            self._select(added[0])

    def _start_auto(self, item: QueueItem) -> None:
        if item.started or item.output_path is not None:
            return
        folder = self._output_dir()
        if item.kind == "foto":
            job = JobRequest(
                kind="photo_export",
                source=item.path,
                intensity=AUTO_INTENSITY,
                output_dir=folder,
                image_format="jpeg",
            )
        else:
            job = JobRequest(
                kind="video_export",
                source=item.path,
                intensity=AUTO_INTENSITY,
                output_dir=folder,
            )
        item.job_id = self.jobs.enqueue(job)
        item.started = True
        item.status = "na fila"
        item.error = ""
        self._render_queue()
        self.btn_cancel.configure(state="normal")

    def _remove_selected(self) -> None:
        if self.selected is None:
            return
        self.items = [item for item in self.items if item is not self.selected]
        self.selected = self.items[0] if self.items else None
        self._render_queue()
        if self.selected:
            self._select(self.selected)
        else:
            self._paint_placeholders()
            self.preview_hint.configure(
                text="Abra JPG, PNG, HEIC ou MP4/MOV. O tratamento começa sozinho."
            )
            self._refresh_export_hint()

    def _render_queue(self) -> None:
        for child in self.queue_scroll.winfo_children():
            child.destroy()
        self._queue_buttons.clear()
        if not self.items:
            ctk.CTkLabel(
                self.queue_scroll,
                text="Nenhum arquivo na fila.\nAbra JPG, PNG, HEIC,\nMP4 ou MOV.",
                text_color=MUTED,
                justify="left",
            ).pack(anchor="w", padx=8, pady=12)
            return
        for item in self.items:
            selected = item is self.selected
            color = "#3a3328" if selected else PANEL
            label = f"{item.path.name}\n{item.kind} · {item.status}"
            btn = ctk.CTkButton(
                self.queue_scroll,
                text=label,
                anchor="w",
                fg_color=color,
                hover_color="#4a4336",
                text_color=ERROR if item.error else TEXT,
                command=lambda current=item: self._select(current),
                height=52,
            )
            btn.pack(fill="x", pady=3, padx=2)
            self._queue_buttons.append(btn)

    def _select(self, item: QueueItem) -> None:
        self.selected = item
        self._render_queue()
        self._preview_token += 1
        token = self._preview_token
        if item.original_preview is not None:
            self._show_original(item.original_preview)
            self._refresh_previews()
        else:
            self._paint_placeholders()
            self.preview_hint.configure(text=f"Lendo {item.path.name}…")
        self._refresh_export_hint()
        self.jobs.request_preview(item.path, token)

    def _cancel(self) -> None:
        self.jobs.cancel_current()
        self._set_status("cancelando…")

    def _show_original(self, rgb: np.ndarray) -> None:
        w, h = self._preview_box()
        pil = _fit_pil(rgb, w, h)
        self._img_orig = ctk.CTkImage(
            light_image=pil, dark_image=pil, size=(pil.width, pil.height)
        )
        self.lbl_orig.configure(image=self._img_orig, text="")

    def _show_result(self, rgb: Optional[np.ndarray], empty: str) -> None:
        w, h = self._preview_box()
        if rgb is None:
            pil = _caption_image(w, h, empty)
            size = (w, h)
        else:
            pil = _fit_pil(rgb, w, h)
            size = (pil.width, pil.height)
        self._img_res = ctk.CTkImage(light_image=pil, dark_image=pil, size=size)
        self.lbl_res.configure(image=self._img_res, text="")

    def _refresh_previews(self) -> None:
        item = self.selected
        if item is None:
            self._paint_placeholders()
            self._refresh_export_hint()
            return
        if item.original_preview is not None:
            self._show_original(item.original_preview)
        if item.processed_rgb is not None:
            self._show_result(item.processed_rgb, "")
            extra = f" · {item.output_path.name}" if item.output_path else ""
            self.preview_hint.configure(text=f"{item.path.name}{extra}")
        else:
            self._show_result(None, "Revelando automaticamente…")
            self.preview_hint.configure(text=f"Tratando {item.path.name} em segundo plano.")
        self._refresh_export_hint()

    def _item_by_job(self, event: UiEvent) -> Optional[QueueItem]:
        if event.job_id:
            for item in self.items:
                if item.job_id == event.job_id:
                    return item
        if event.path:
            for item in self.items:
                if item.path == event.path:
                    return item
        return None

    def _poll(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                self._handle(event)
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _handle(self, event: UiEvent) -> None:
        if event.type == "original_preview":
            if event.token != self._preview_token:
                return
            item = None
            if event.path:
                for candidate in self.items:
                    if candidate.path == event.path:
                        item = candidate
                        break
            if event.error:
                if item:
                    item.error = event.error
                    item.status = "erro"
                    self._render_queue()
                self._set_status(event.error, error=True)
                self.preview_hint.configure(text=event.error)
                return
            if item is not None and event.preview_rgb is not None:
                item.original_preview = event.preview_rgb
                if item is self.selected:
                    self._refresh_previews()
            return

        item = self._item_by_job(event)
        if event.type == "progress":
            if item:
                item.status = event.status or item.status
                self._render_queue()
            label = event.status or "processando"
            if item:
                label = f"{label} — {item.path.name}"
            if event.percent >= 0:
                label = f"{label}  ({event.percent:.0f}%)"
            self._set_status(label, event.percent if event.percent >= 0 else None)
            return

        if event.type == "live_frame":
            if item is not None:
                if event.preview_rgb is not None:
                    item.processed_rgb = event.preview_rgb
                if item is self.selected and event.preview_rgb is not None:
                    self._show_result(event.preview_rgb, "")
            return

        if event.type == "queued":
            self.btn_cancel.configure(state="normal")
            return

        if event.type == "done":
            result = self.jobs.take_result(event.job_id)
            if item:
                if isinstance(result, np.ndarray):
                    item.processed_rgb = result
                elif event.preview_rgb is not None and item.processed_rgb is None:
                    item.processed_rgb = event.preview_rgb
                if event.output_path:
                    item.output_path = event.output_path
                item.status = "pronto"
                item.error = ""
                self._render_queue()
                if item is self.selected:
                    self._refresh_previews()
            msg = event.status or "concluído"
            if event.output_path:
                msg = f"Salvo para o iPhone: {event.output_path}"
            self._set_status(msg, 100.0)
            if not self.jobs.busy:
                self.btn_cancel.configure(state="disabled")
            return

        if event.type == "cancelled":
            if item:
                item.status = "cancelado"
                item.started = False
                self._render_queue()
            self._set_status("Cancelado.", 0)
            if not self.jobs.busy:
                self.btn_cancel.configure(state="disabled")
            return

        if event.type == "error":
            if item:
                item.status = "erro"
                item.error = event.error
                self._render_queue()
            self._set_status(event.error or "Erro.", 0, error=True)
            if not self.jobs.busy:
                self.btn_cancel.configure(state="disabled")
            return

    def _load_samples(self) -> None:
        folder = Path(__file__).resolve().parent / "amostras"
        if not folder.is_dir():
            return
        files = sorted(
            [
                path
                for path in folder.iterdir()
                if path.is_file() and (is_photo(path) or is_video(path))
            ]
        )
        if files and not self.items:
            self._add_paths(files, select_first=True)
            self.preview_hint.configure(
                text="Amostras na fila — o tratamento automático já começou."
            )

    def _on_resize(self, _event) -> None:
        if self._resize_after:
            try:
                self.after_cancel(self._resize_after)
            except Exception:
                pass
        self._resize_after = self.after(180, self._refresh_previews)

    def _on_close(self) -> None:
        self.jobs.shutdown()
        self.destroy()


def run_app() -> None:
    ctk.set_appearance_mode("dark")
    try:
        ctk.set_default_color_theme("gold")
    except Exception:
        ctk.set_default_color_theme("dark-blue")
    app = RevelarApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()
