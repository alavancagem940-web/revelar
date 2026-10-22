"""Aprimoramento automático de fotos: detalhe tipo celular, sem HDR/contraste no retrato."""

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


def _clahe_l(l_ch: np.ndarray, mix: float = 0.5, clip: float = 1.6) -> np.ndarray:
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
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    local = clahe.apply(l_ch)
    mix = float(np.clip(mix, 0.0, 1.0))
    mixed = mix * local.astype(np.float32) + (1.0 - mix) * l_ch.astype(np.float32)
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


def _structure_mask(gray: np.ndarray) -> np.ndarray:
    """0 = pele/fundo lisos, 1 = cabelo, tecido, bordas."""
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 1.4)
    local = np.sqrt(np.maximum(cv2.GaussianBlur((g - blur) ** 2, (0, 0), 1.4), 0.0))
    mask = np.clip((local - 2.8) / 9.0, 0.0, 1.0)
    edges = cv2.Canny(gray, 35, 110).astype(np.float32) / 255.0
    mask = np.maximum(mask, cv2.GaussianBlur(edges, (0, 0), 0.8) * 0.85)
    return cv2.GaussianBlur(mask, (0, 0), 1.1)


def _deepen_structured_blacks(
    l_ch: np.ndarray, mask: np.ndarray, amount: float
) -> np.ndarray:
    """Aprofunda sombra em cabelo/tecido; não granula fundo liso."""
    if amount <= 0.01:
        return l_ch
    x = l_ch.astype(np.float32)
    shadow = np.clip((78.0 - x) / 78.0, 0.0, 1.0)
    weight = shadow * (0.25 + 0.75 * mask.astype(np.float32))
    return np.clip(x - (15.0 * amount) * weight, 0.0, 255.0)


def _masked_micro_contrast(
    l_ch: np.ndarray, mask: np.ndarray, amount: float
) -> np.ndarray:
    """Micro-textura só na máscara (cabelo/tecido). Pele e fundo ficam lisos."""
    if amount <= 0.01:
        return l_ch
    high = _hf_unsharp_l(l_ch, sigma=0.75, amount=0.48 * amount, threshold=2.4)
    clarity = _hf_unsharp_l(high, sigma=2.2, amount=0.20 * amount, threshold=4.0)
    mix = mask.astype(np.float32)
    out = (1.0 - mix) * l_ch.astype(np.float32) + mix * clarity.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


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


def _smooth_fraction(gray: np.ndarray) -> float:
    """Fração de área lisa (pele, fundo de estúdio)."""
    g = gray.astype(np.float32)
    if max(g.shape) > 1200:
        scale = 1200 / max(g.shape)
        g = cv2.resize(
            g,
            (max(8, int(g.shape[1] * scale)), max(8, int(g.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    mean = cv2.blur(g, (7, 7))
    var = cv2.blur((g - mean) ** 2, (7, 7))
    local = np.sqrt(np.maximum(var, 0.0))
    return float((local < 6.5).mean())


def _l_contrast(rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)
    return float(np.percentile(l_ch, 95) - np.percentile(l_ch, 5))


PHONE_LONG_SIDE = 1600.0


def assess_recovery(rgb: np.ndarray) -> RecoveryPlan:
    """Fonte pequena (anexo de chat) → upscale limpo a ~1600. Deblock só se for mole."""
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    gray = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2GRAY)
    lap = _laplacian_var(gray)
    small = long_side < 1400
    soft = lap < 70.0

    if long_side >= 1400 and lap >= 70.0:
        return RecoveryPlan(False, False, 1.0, "já em tamanho de celular")

    deblock = bool(small and soft)

    scale = 1.0
    if small:
        scale = min(4.8, PHONE_LONG_SIDE / float(long_side))
        if long_side * scale > 1800:
            scale = 1800.0 / float(long_side)
        if scale < 1.12:
            scale = 1.0
    scale = max(1.0, scale)
    upscale = scale > 1.0
    if deblock and upscale:
        reason = "JPEG mole + upscale limpo"
    elif deblock:
        reason = "desbloqueio leve"
    elif upscale:
        reason = "upscale limpo tipo celular"
    else:
        reason = "sem recuperação"
    return RecoveryPlan(deblock, upscale, float(scale), reason)


def _underexposed(rgb: np.ndarray) -> bool:
    lab = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2LAB)
    mean_l = float(lab[:, :, 0].mean())
    return mean_l < 58.0 and _l_contrast(rgb) < 95.0


def _iterative_backproject_l(
    l_ch: np.ndarray, nw: int, nh: int, iters: int = 8
) -> np.ndarray:
    """Reconstrói L em alta res para ficar fiel ao original quando rebaixado."""
    h, w = l_ch.shape[:2]
    src = l_ch.astype(np.float32)
    hr = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    for _ in range(iters):
        down = cv2.resize(hr, (w, h), interpolation=cv2.INTER_AREA)
        err_up = cv2.resize(src - down, (nw, nh), interpolation=cv2.INTER_CUBIC)
        hr = np.clip(hr + 0.9 * err_up, 0.0, 255.0)
    return hr


def _protect_smooth(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fundo e pele lisos; cabelo/tecido ficam com o detalhe reconstruído."""
    bg = cv2.GaussianBlur(bgr, (0, 0), 2.2)
    skin = cv2.bilateralFilter(bgr, 7, 14, 5)
    m = mask.astype(np.float32)[..., None]
    mix_bg = np.clip(1.0 - m * 4.0, 0.0, 1.0)
    mix_st = np.clip((m - 0.22) / 0.55, 0.0, 1.0)
    mix_skin = np.clip(1.0 - mix_bg - mix_st, 0.0, 1.0)
    out = (
        bg.astype(np.float32) * mix_bg
        + skin.astype(np.float32) * mix_skin
        + bgr.astype(np.float32) * mix_st
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def _clean_phone_upscale(bgr: np.ndarray, scale: float) -> np.ndarray:
    """Upscale tipo celular: backprojection no L, chroma Lanczos, sem CLAHE."""
    if scale <= 1.01:
        return bgr
    h, w = bgr.shape[:2]
    nw = max(2, int(round(w * scale)))
    nh = max(2, int(round(h * scale)))
    nw -= nw % 2
    nh -= nh % 2
    if nw <= w and nh <= h:
        return bgr
    rgb = _bgr_to_rgb(bgr)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask_n = _structure_mask(gray)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_hr = _iterative_backproject_l(l_ch, nw, nh)
    a_hr = cv2.resize(a_ch, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    b_hr = cv2.resize(b_ch, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    rgb_hr = cv2.cvtColor(
        cv2.merge((np.clip(l_hr, 0, 255).astype(np.uint8), a_hr, b_hr)),
        cv2.COLOR_LAB2RGB,
    )
    mask = cv2.resize(mask_n, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), 1.1) * 1.25, 0.0, 1.0)
    out_bgr = _protect_smooth(_rgb_to_bgr(rgb_hr), mask)
    lab = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    sharp = _hf_unsharp_l(l_ch, sigma=0.8, amount=0.58, threshold=1.4)
    l_ch = np.clip(
        (1.0 - mask) * l_ch.astype(np.float32) + mask * sharp.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def _deblock_compression(bgr: np.ndarray) -> np.ndarray:
    """Desbloqueio leve: chroma e mosquito, sem NLMeans que lava detalhe."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y_ch, cr, cb = cv2.split(ycc)
    cr = cv2.medianBlur(cr, 3)
    cb = cv2.medianBlur(cb, 3)
    y_ch = cv2.bilateralFilter(y_ch, 5, 14, 5)
    return cv2.cvtColor(cv2.merge((y_ch, cr, cb)), cv2.COLOR_YCrCb2BGR)


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
    """Detalhe de celular: upscale limpo se a fonte for pequena. Sem HDR no retrato."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise MediaError("imagem inválida.")
    _check(cancel)
    _report(progress, 4.0, "lendo")
    original = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    bgr = _rgb_to_bgr(original)

    plan = assess_recovery(original)
    already_phone = max(original.shape[:2]) >= 1400 and not plan.upscale

    if plan.deblock:
        _check(cancel)
        _report(progress, 10.0, "desbloqueando")
        bgr = _deblock_compression(bgr)
    if plan.upscale:
        _check(cancel)
        _report(progress, 18.0, "detalhe tipo celular")
        bgr = _clean_phone_upscale(bgr, plan.scale)

    if already_phone:
        # foto já no nível da foto 1: não piorar com curva/CLAHE
        _report(progress, 70.0, "preservar")
        processed = _bgr_to_rgb(bgr)
    elif _underexposed(original):
        _check(cancel)
        _report(progress, 40.0, "exposição")
        bgr = _white_balance(bgr, 0.18)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = _levels_percentile(
            l_ch, low_p=1.0, high_p=99.2, floor=8.0, ceiling=248.0
        )
        processed = _bgr_to_rgb(
            cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        )
    else:
        _report(progress, 80.0, "limpo")
        processed = _bgr_to_rgb(bgr)

    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity < 0.999 and processed.shape == original.shape:
        blended = (
            original.astype(np.float32) * (1.0 - intensity)
            + processed.astype(np.float32) * intensity
        )
        processed = np.clip(blended, 0, 255).astype(np.uint8)

    _report(progress, 100.0, "concluído")
    return processed
