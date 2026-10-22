"""Aprimoramento de vídeo: equalização de cor, estabilização leve e encode iPhone."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from moviepy import VideoFileClip

from export_ios import (
    FFmpegMissing,
    MediaError,
    crop_even,
    encode_h264_aac,
    find_ffmpeg,
    require_ffmpeg,
    target_video_bitrate,
)
from photo_engine import _auto_exposure, _clahe, _unsharp, _white_balance

ProgressFn = Callable[[float, str], None]


def probe_video(path: Path | str) -> dict:
    path = Path(path)
    require_ffmpeg()
    try:
        with VideoFileClip(str(path)) as clip:
            fps = float(clip.fps or 30.0)
            width, height = clip.size
            duration = float(clip.duration or 0.0)
            has_audio = clip.audio is not None
    except FFmpegMissing:
        raise
    except Exception as exc:
        raise MediaError(
            f"arquivo de vídeo corrompido ou codec ausente: {path.name} ({exc})"
        ) from exc
    return {
        "fps": fps,
        "width": int(width),
        "height": int(height),
        "duration": duration,
        "has_audio": has_audio,
        "frames": max(1, int(round(duration * fps))),
    }


def first_frame(path: Path | str) -> np.ndarray:
    path = Path(path)
    try:
        with VideoFileClip(str(path)) as clip:
            frame = clip.get_frame(0)
    except Exception as exc:
        raise MediaError(
            f"não foi possível ler o vídeo {path.name}: {exc}"
        ) from exc
    rgb = np.ascontiguousarray(frame[..., :3].astype(np.uint8))
    if rgb.shape[0] < 2 or rgb.shape[1] < 2:
        raise MediaError(f"vídeo inválido: {path.name}")
    return rgb


def enhance_frame(
    rgb: np.ndarray,
    intensity: float = 0.65,
    gains: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Equalização de um quadro (RGB). Sem denoise pesado — adequado a vídeo."""
    intensity = float(np.clip(intensity, 0.0, 1.0))
    original = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    if intensity <= 0.01:
        return original
    if gains is not None:
        working = _apply_gains(original, gains, intensity)
        bgr = np.ascontiguousarray(working[..., ::-1])
    else:
        bgr = np.ascontiguousarray(original[..., ::-1])
        bgr = _white_balance(bgr, intensity)
    bgr = _auto_exposure(bgr, intensity * 0.85)
    bgr = _clahe(bgr, intensity * 0.75)
    bgr = _unsharp(bgr, intensity * 0.55)
    processed = np.ascontiguousarray(bgr[..., ::-1])
    if intensity < 0.999:
        blended = (
            original.astype(np.float32) * (1.0 - intensity)
            + processed.astype(np.float32) * intensity
        )
        processed = np.clip(blended, 0, 255).astype(np.uint8)
    return processed


def _sample_wb_gains(path: Path, samples: int = 8) -> Optional[np.ndarray]:
    """Ganhos de gray-world médios para equalização estável (sem flicker)."""
    try:
        with VideoFileClip(str(path)) as clip:
            duration = float(clip.duration or 0.0)
            if duration <= 0:
                return None
            times = np.linspace(0, max(0.0, duration - 1.0 / max(clip.fps or 30, 1)), samples)
            means = []
            for t in times:
                frame = clip.get_frame(float(t))
                bgr = frame[..., ::-1].astype(np.float32)
                means.append(bgr.reshape(-1, 3).mean(axis=0))
            avg = np.mean(np.stack(means, axis=0), axis=0)
            gray = float(avg.mean())
            return gray / np.maximum(avg, 1.0)
    except Exception:
        return None


def _apply_gains(rgb: np.ndarray, gains: np.ndarray, strength: float) -> np.ndarray:
    work = rgb.astype(np.float32)
    # MoviePy/OpenCV: RGB vs BGR — ganhos calculados em BGR (B,G,R)
    # Converte para RGB (R,G,B) = gains[[2,1,0]]
    rgb_gains = gains[[2, 1, 0]]
    scale = 1.0 + (rgb_gains - 1.0) * strength
    work *= scale
    return np.clip(work, 0, 255).astype(np.uint8)


def _stabilize_shifts(path: Path, cancel: Optional[threading.Event], progress: Optional[ProgressFn]) -> np.ndarray:
    """Deslocamento por quadro (dx, dy) via phase correlation — estabilização leve."""
    with VideoFileClip(str(path)) as clip:
        fps = float(clip.fps or 30.0)
        n = max(1, int(round(float(clip.duration or 0.0) * fps)))
        prev = None
        shifts = []
        for index, frame in enumerate(clip.iter_frames()):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelado")
            gray = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            if prev is None:
                shifts.append((0.0, 0.0))
            else:
                delta, _ = cv2.phaseCorrelate(
                    np.float32(prev),
                    np.float32(small),
                )
                shifts.append((float(delta[0]) * 4.0, float(delta[1]) * 4.0))
            prev = small
            if progress and n:
                progress(8.0 + 22.0 * min(1.0, (index + 1) / n), "estabilização")
    return np.array(shifts, dtype=np.float32)


def _smooth_corrections(shifts: np.ndarray, window: int, max_px: float) -> np.ndarray:
    if len(shifts) == 0:
        return shifts
    window = int(window)
    if window < 3 or len(shifts) < 3:
        return np.zeros_like(shifts)
    if window % 2 == 0:
        window += 1
    window = min(window, len(shifts) if len(shifts) % 2 == 1 else max(3, len(shifts) - 1))
    cum = np.cumsum(shifts, axis=0)
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.empty_like(cum)
    for axis in (0, 1):
        padded = np.pad(cum[:, axis], (pad, pad), mode="edge")
        smoothed[:, axis] = np.convolve(padded, kernel, mode="valid")
    corr = smoothed - cum
    return np.clip(corr, -max_px, max_px)


def _warp_rgb(rgb: np.ndarray, dx: float, dy: float, crop: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    warped = cv2.warpAffine(
        rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    crop = float(np.clip(crop, 0.0, 0.12))
    if crop <= 0.001:
        return crop_even(warped)
    x0 = int(w * crop)
    y0 = int(h * crop)
    x1 = w - x0
    y1 = h - y0
    if x1 - x0 < 8 or y1 - y0 < 8:
        return crop_even(warped)
    cropped = warped[y0:y1, x0:x1]
    return crop_even(
        cv2.resize(cropped, (w - (w % 2), h - (h % 2)), interpolation=cv2.INTER_LINEAR)
    )


def enhance_and_export_video(
    source: Path | str,
    dest: Path | str,
    intensity: float = 0.65,
    cancel: Optional[threading.Event] = None,
    progress: Optional[ProgressFn] = None,
    live_preview: Optional[Callable[[np.ndarray], None]] = None,
) -> Path:
    """Um passe de export iPhone-ready: equaliza, estabiliza leve, H.264 + AAC."""
    source = Path(source)
    dest = Path(dest)
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if progress:
        progress(2.0, "lendo")
    if not find_ffmpeg():
        raise FFmpegMissing()
    info = probe_video(source)
    fps = info["fps"]
    width, height = int(info["width"]), int(info["height"])
    width, height = width - (width % 2), height - (height % 2)
    nframes = int(info["frames"])

    if progress:
        progress(6.0, "equalização de cor")
    gains = _sample_wb_gains(source) if intensity > 0.02 else None

    corrections = None
    if intensity > 0.08:
        if progress:
            progress(8.0, "estabilização")
        try:
            shifts = _stabilize_shifts(source, cancel, progress)
            window = 9 + int(22 * intensity)
            max_px = 0.08 * max(width, height)
            corrections = _smooth_corrections(shifts, window, max_px)
        except InterruptedError:
            raise
        except Exception:
            corrections = None

    bitrate = target_video_bitrate(width, height, fps)
    if progress:
        progress(32.0, "otimizando bitrate")

    crop = 0.012 + 0.028 * intensity if corrections is not None else 0.0

    def frame_iter():
        with VideoFileClip(str(source)) as clip:
            for index, frame in enumerate(clip.iter_frames()):
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("cancelado")
                rgb = np.ascontiguousarray(frame[..., :3].astype(np.uint8))
                if intensity > 0.01:
                    rgb = enhance_frame(rgb, intensity, gains=gains)
                if corrections is not None and index < len(corrections):
                    dx, dy = float(corrections[index, 0]), float(corrections[index, 1])
                    rgb = _warp_rgb(rgb, dx, dy, crop)
                else:
                    rgb = crop_even(rgb)
                if rgb.shape[1] != width or rgb.shape[0] != height:
                    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                    rgb = crop_even(rgb)
                    if rgb.shape[1] != width or rgb.shape[0] != height:
                        canvas = np.zeros((height, width, 3), dtype=np.uint8)
                        hh, ww = min(height, rgb.shape[0]), min(width, rgb.shape[1])
                        canvas[:hh, :ww] = rgb[:hh, :ww]
                        rgb = canvas
                if progress and nframes:
                    frac = min(1.0, (index + 1) / nframes)
                    progress(32.0 + 18.0 * frac, "equalização de cor")
                if live_preview is not None and index % 6 == 0:
                    live_preview(rgb)
                yield rgb

    audio_source = source if info.get("has_audio") else None
    if progress:
        progress(50.0, "exportando H.264")
    return encode_h264_aac(
        frame_iter(),
        width,
        height,
        fps,
        dest,
        audio_source=audio_source,
        bitrate=bitrate,
        cancel=cancel,
        progress=progress,
        total_frames=nframes,
    )
