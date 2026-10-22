"""Aprimoramento automático de fotos: exposição, CLAHE, WB, nitidez e ruído."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from export_ios import MediaError

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}

_heif_tried = False
_heif_ok = False

ProgressFn = Callable[[float, str], None]


def _ensure_heif() -> bool:
    global _heif_tried, _heif_ok
    if _heif_tried:
        return _heif_ok
    _heif_tried = True
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        _heif_ok = True
    except Exception:
        _heif_ok = False
    return _heif_ok


def is_photo(path: Path) -> bool:
    return Path(path).suffix.lower() in PHOTO_EXTENSIONS


def is_video(path: Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _check(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise InterruptedError("cancelado")


def _report(progress: Optional[ProgressFn], percent: float, status: str) -> None:
    if progress is not None:
        progress(percent, status)


def load_image(path: Path | str) -> np.ndarray:
    """Carrega JPG/PNG/HEIC como RGB uint8, respeitando orientação EXIF."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        if not _ensure_heif():
            raise MediaError(
                "não foi possível abrir HEIC. Instale pillow-heif "
                "(pip install pillow-heif) e a libheif do sistema."
            )
    else:
        _ensure_heif()
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise MediaError(
            f"arquivo corrompido ou formato não suportado: {path.name}"
        ) from exc
    except OSError as exc:
        raise MediaError(f"falha ao ler {path.name}: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
        raise MediaError(f"imagem inválida: {path.name}")
    return rgb


def downscale_rgb(rgb: np.ndarray, max_side: int = 720) -> np.ndarray:
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return rgb
    scale = max_side / float(longest)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[..., ::-1])


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bgr[..., ::-1])


def _white_balance(bgr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.01:
        return bgr
    work = bgr.astype(np.float32)
    means = work.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    scale = gray / np.maximum(means, 1.0)
    scale = 1.0 + (scale - 1.0) * strength
    work *= scale
    return np.clip(work, 0, 255).astype(np.uint8)


def _auto_exposure(bgr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.01:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mean = float(v.mean())
    if mean < 1.0:
        return bgr
    target = 118.0
    ratio = float(np.clip(target / mean, 0.55, 2.15))
    ratio = 1.0 + (ratio - 1.0) * strength
    gamma = 1.0 / max(ratio, 0.15)
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = cv2.LUT(v, lut)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _clahe(bgr: np.ndarray, strength: float) -> np.ndarray:
    clip = 1.0 + 3.2 * max(0.0, strength)
    h, w = bgr.shape[:2]
    tile = 8 if min(h, w) >= 64 else 2
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def _unsharp(bgr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.01:
        return bgr
    amount = 0.35 + 0.85 * strength
    radius = 1.2 + 0.6 * strength
    blur = cv2.GaussianBlur(bgr, (0, 0), radius)
    sharp = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
    return sharp


def _denoise(bgr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.05:
        return bgr
    h, w = bgr.shape[:2]
    if max(h, w) >= 1280:
        sigma = 18.0 + 55.0 * strength
        return cv2.bilateralFilter(bgr, 5, sigma, sigma)
    h_param = 2.0 + 7.5 * strength
    return cv2.fastNlMeansDenoisingColored(bgr, None, h_param, h_param, 7, 21)


def enhance_photo(
    rgb: np.ndarray,
    intensity: float = 0.65,
    cancel: Optional[threading.Event] = None,
    progress: Optional[ProgressFn] = None,
) -> np.ndarray:
    """Pipeline profissional. `intensity` em 0..1 mistura o resultado com o original."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise MediaError("imagem inválida.")
    intensity = float(np.clip(intensity, 0.0, 1.0))
    _check(cancel)
    _report(progress, 8.0, "lendo")
    original = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    bgr = _rgb_to_bgr(original)

    _check(cancel)
    _report(progress, 22.0, "redução de ruído")
    bgr = _denoise(bgr, intensity)

    _check(cancel)
    _report(progress, 40.0, "balanço de branco")
    bgr = _white_balance(bgr, intensity)

    _check(cancel)
    _report(progress, 58.0, "exposição")
    bgr = _auto_exposure(bgr, intensity)

    _check(cancel)
    _report(progress, 74.0, "CLAHE")
    bgr = _clahe(bgr, intensity)

    _check(cancel)
    _report(progress, 88.0, "nitidez")
    bgr = _unsharp(bgr, intensity)

    processed = _bgr_to_rgb(bgr)
    if intensity < 0.999:
        alpha = intensity
        blended = (
            original.astype(np.float32) * (1.0 - alpha)
            + processed.astype(np.float32) * alpha
        )
        processed = np.clip(blended, 0, 255).astype(np.uint8)
    _report(progress, 100.0, "concluído")
    return processed
