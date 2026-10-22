"""Aprimoramento automático de fotos: recuperação + punch em resolução nativa."""

from __future__ import annotations

import threading
from dataclasses import dataclass
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
    """Carrega JPG/PNG/HEIC como RGB uint8, resolução nativa, orientação EXIF."""
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
    except OSError as orig_err:
        raise MediaError(f"falha ao ler {path.name}: {orig_err}") from orig_err
    if rgb.ndim != 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
        raise MediaError(f"imagem inválida: {path.name}")
    return rgb


def downscale_rgb(rgb: np.ndarray, max_side: int = 720) -> np.ndarray:
    """Só para pré-visualização na UI — o pipeline de foto não usa isto."""
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
    """Gray-world moderado — usado também pelo vídeo."""
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
    """Exposição HSV suave — fallback do vídeo, não o punch da foto."""
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
    """CLAHE no L — versão mais branda para vídeo."""
    clip = 1.2 + 1.4 * max(0.0, min(1.0, strength))
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
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def _levels_percentile(
    channel: np.ndarray,
    low_p: float = 0.7,
    high_p: float = 99.45,
    floor: float = 7.0,
    ceiling: float = 249.0,
) -> np.ndarray:
    """Estica pretos/brancos por percentil, sem mandar sombra para 0."""
    sample = channel
    if channel.size > 4_000_000:
        step = int(np.sqrt(channel.size / 1_000_000))
        sample = channel[::step, ::step]
    lo = float(np.percentile(sample, low_p))
    hi = float(np.percentile(sample, high_p))
    if hi - lo < 12.0:
        return channel
    scale = (ceiling - floor) / (hi - lo)
    out = (channel.astype(np.float32) - lo) * scale + floor
    return np.clip(out, 0, 255).astype(np.uint8)


def _clahe_l(l_ch: np.ndarray) -> np.ndarray:
    """CLAHE no L, clip baixo para não posterizar; mistura com o original."""
    h, w = l_ch.shape[:2]
    side = min(h, w)
    if side < 64:
        tile = 2
    elif side >= 2400:
        tile = 12
    elif side >= 1200:
        tile = 10
    else:
        tile = 8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
    local = clahe.apply(l_ch)
    # 65% local / 35% global evita halo e banding
    mixed = 0.65 * local.astype(np.float32) + 0.35 * l_ch.astype(np.float32)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _hf_unsharp_l(
    l_ch: np.ndarray,
    sigma: float,
    amount: float,
    threshold: float = 2.0,
) -> np.ndarray:
    """Unsharp de alta frequência só no L (sem franja de cor)."""
    blur = cv2.GaussianBlur(l_ch, (0, 0), sigma)
    high = l_ch.astype(np.float32) - blur.astype(np.float32)
    if threshold > 0:
        high = np.where(np.abs(high) >= threshold, high, 0.0)
    out = l_ch.astype(np.float32) + amount * high
    return np.clip(out, 0, 255).astype(np.uint8)


def _s_curve_l(l_ch: np.ndarray, amount: float = 0.16) -> np.ndarray:
    """Contraste em S: pretos mais profundos, cinza lavado some, extremos protegidos."""
    x = l_ch.astype(np.float32) / 255.0
    # empurra meios sem clipar sombras finas
    mid = (x - 0.5) * (1.0 - np.abs(2.0 * x - 1.0))
    y = np.clip(x + amount * mid, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


@dataclass(frozen=True)
class RecoveryPlan:
    deblock: bool
    upscale: bool
    scale: float
    reason: str


def _blockiness(gray: np.ndarray) -> float:
    """Razão de descontinuidade na grelha 8×8 típica de JPEG."""
    g = gray.astype(np.float32)
    if g.shape[1] < 16 or g.shape[0] < 16:
        return 1.0
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    off = float(dx.mean() + dy.mean()) + 1e-6
    on = float(dx[:, 7::8].mean() + dy[7::8, :].mean())
    return on / off


def _laplacian_var(gray: np.ndarray) -> float:
    if max(gray.shape) > 1600:
        scale = 1600 / max(gray.shape)
        gray = cv2.resize(
            gray,
            (max(8, int(gray.shape[1] * scale)), max(8, int(gray.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_recovery(rgb: np.ndarray) -> RecoveryPlan:
    """Decide deblock/Lanczos. Nunca reduz um original nítido em resolução cheia."""
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    gray = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2GRAY)
    lap = _laplacian_var(gray)
    block = _blockiness(gray)
    small = long_side < 1400
    soft = lap < 70.0
    # originais nítidos em resolução cheia: só punch, sem deblock/upscale
    if long_side >= 1600 and lap >= 80.0 and block < 1.12:
        return RecoveryPlan(False, False, 1.0, "original nítido — sem upscale")

    compressed = block >= 1.15
    deblock = (small and (compressed or soft)) or block >= 1.25

    scale = 1.0
    target = 1920.0
    if small or (soft and long_side < 1920):
        scale = min(3.0, target / float(long_side))
        if long_side * scale > 2560:
            scale = 2560.0 / float(long_side)
        if scale < 1.06:
            scale = 1.0
    # jamais downscale
    scale = max(1.0, scale)
    upscale = scale > 1.0
    if deblock and upscale:
        reason = "JPEG/compactado + Lanczos"
    elif deblock:
        reason = "desbloqueio de compressão"
    elif upscale:
        reason = "Lanczos (fonte pequena ou mole)"
    else:
        reason = "sem recuperação"
    return RecoveryPlan(deblock, upscale, float(scale), reason)


def _deblock_compression(bgr: np.ndarray) -> np.ndarray:
    """Suaviza artefato de bloco/mosquito sem apagar bordas."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y_ch, cr, cb = cv2.split(ycc)
    cr = cv2.medianBlur(cr, 3)
    cb = cv2.medianBlur(cb, 3)
    y_ch = cv2.bilateralFilter(y_ch, 7, 22, 7)
    cleaned = cv2.cvtColor(cv2.merge((y_ch, cr, cb)), cv2.COLOR_YCrCb2BGR)
    h, w = cleaned.shape[:2]
    if max(h, w) <= 1600:
        cleaned = cv2.fastNlMeansDenoisingColored(cleaned, None, 3.5, 3.5, 7, 15)
    return cleaned


def _lanczos_upscale(bgr: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.01:
        return bgr
    h, w = bgr.shape[:2]
    nw = max(2, int(round(w * scale)))
    nh = max(2, int(round(h * scale)))
    nw -= nw % 2
    nh -= nh % 2
    if nw <= w and nh <= h:
        return bgr
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


def _chroma_pop(a_ch: np.ndarray, b_ch: np.ndarray, factor: float = 1.07) -> tuple[np.ndarray, np.ndarray]:
    a = (a_ch.astype(np.float32) - 128.0) * factor + 128.0
    b = (b_ch.astype(np.float32) - 128.0) * factor + 128.0
    return np.clip(a, 0, 255).astype(np.uint8), np.clip(b, 0, 255).astype(np.uint8)


def enhance_photo(
    rgb: np.ndarray,
    intensity: float = 1.0,
    cancel: Optional[threading.Event] = None,
    progress: Optional[ProgressFn] = None,
) -> np.ndarray:
    """Recuperação (se preciso) + punch LAB. Nunca reduz original nítido."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise MediaError("imagem inválida.")
    _check(cancel)
    _report(progress, 4.0, "lendo")
    original = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    bgr = _rgb_to_bgr(original)

    plan = assess_recovery(original)
    if plan.deblock:
        _check(cancel)
        _report(progress, 10.0, "desbloqueando")
        bgr = _deblock_compression(bgr)
    if plan.upscale:
        _check(cancel)
        _report(progress, 16.0, "Lanczos")
        bgr = _lanczos_upscale(bgr, plan.scale)

    _check(cancel)
    _report(progress, 22.0, "balanço de branco")
    bgr = _white_balance(bgr, 0.5)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    _check(cancel)
    _report(progress, 38.0, "pretos e brancos")
    l_ch = _levels_percentile(l_ch)

    _check(cancel)
    _report(progress, 58.0, "CLAHE")
    l_ch = _clahe_l(l_ch)

    _check(cancel)
    _report(progress, 74.0, "nitidez")
    # alta frequência (fios, dobras) + clareza local leve
    l_ch = _hf_unsharp_l(l_ch, sigma=0.75, amount=1.28, threshold=2.0)
    l_ch = _hf_unsharp_l(l_ch, sigma=2.1, amount=0.32, threshold=1.0)

    _check(cancel)
    _report(progress, 88.0, "contraste")
    l_ch = _s_curve_l(l_ch, amount=0.15)
    # não esmagar o último detalhe de sombra no preto
    floor = float(np.percentile(l_ch, 0.35))
    if floor < 5.0:
        lifted = l_ch.astype(np.float32)
        crushed = lifted < 6.0
        lifted[crushed] = 6.0 + lifted[crushed] * 0.4
        l_ch = np.clip(lifted, 0, 255).astype(np.uint8)
    a_ch, b_ch = _chroma_pop(a_ch, b_ch, 1.08)

    processed = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    processed = _bgr_to_rgb(processed)

    # intensity só existe por compatibilidade interna — a UI não tem slider.
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity < 0.999 and processed.shape == original.shape:
        blended = (
            original.astype(np.float32) * (1.0 - intensity)
            + processed.astype(np.float32) * intensity
        )
        processed = np.clip(blended, 0, 255).astype(np.uint8)

    _report(progress, 100.0, "concluído")
    return processed
