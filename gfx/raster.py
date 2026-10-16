"""
Procedural asset rasterizer (numpy).

Images are float32 premultiplied RGBA arrays shaped (H, W, 4) in 0..1.
Shapes are built as signed-distance fields (negative inside), turned into
antialiased coverage masks, then painted with colors, gradients or textures.
Everything here runs once at asset-build time, never per frame.
"""
import math
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

RGB = Tuple[float, float, float]


def rgb01(c) -> np.ndarray:
    """(r, g, b) in 0-255 -> float array 0..1."""
    return np.asarray(c, np.float32)[:3] / 255.0


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- canvases

def blank(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 4), np.float32)


def solid(w: int, h: int, color, a: float = 1.0) -> np.ndarray:
    img = np.empty((h, w, 4), np.float32)
    img[..., :3] = rgb01(color) * a
    img[..., 3] = a
    return img


def grid(w: int, h: int, x0: float = 0.0, y0: float = 0.0):
    """Pixel-center coordinate grids (xx, yy)."""
    ys = np.arange(h, dtype=np.float32) + 0.5 + y0
    xs = np.arange(w, dtype=np.float32) + 0.5 + x0
    return np.meshgrid(xs, ys)


def paint(img: np.ndarray, mask: np.ndarray, color, a: float = 1.0, op: str = "over"):
    """Paint a coverage mask (H, W) or per-pixel color (H, W, 3) onto img in place."""
    cov = np.clip(mask, 0.0, 1.0)[..., None] * a
    col = color if isinstance(color, np.ndarray) and color.ndim == 3 else rgb01(color)
    src_rgb = col * cov
    if op == "over":
        img[..., :3] = src_rgb + img[..., :3] * (1 - cov)
        img[..., 3:] = cov + img[..., 3:] * (1 - cov)
    elif op == "add":
        img[..., :3] = np.minimum(img[..., :3] + src_rgb, 1.0)
        img[..., 3:] = np.minimum(img[..., 3:] + cov, 1.0)
    elif op == "screen":
        img[..., :3] = 1 - (1 - img[..., :3]) * (1 - src_rgb)
        img[..., 3:] = cov + img[..., 3:] * (1 - cov)
    elif op == "multiply":
        # multiply toward color by coverage (keeps alpha)
        img[..., :3] = img[..., :3] * (1 - cov + col * cov)
    elif op == "erase":
        img *= (1 - cov)
    else:
        raise ValueError(op)
    return img


def composite(dst: np.ndarray, src: np.ndarray, x: int = 0, y: int = 0, op: str = "over",
              alpha: float = 1.0):
    """Composite premultiplied src onto dst at (x, y), clipped."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return dst
    s = src[y0 - y:y1 - y, x0 - x:x1 - x] * alpha
    d = dst[y0:y1, x0:x1]
    if op == "over":
        d[:] = s + d * (1 - s[..., 3:])
    elif op == "add":
        d[:] = np.minimum(d + s, 1.0)
    elif op == "screen":
        d[..., :3] = 1 - (1 - d[..., :3]) * (1 - s[..., :3])
        d[..., 3:] = s[..., 3:] + d[..., 3:] * (1 - s[..., 3:])
    else:
        raise ValueError(op)
    return dst


def tint(mask: np.ndarray, color, a: float = 1.0) -> np.ndarray:
    """Coverage mask -> premultiplied RGBA image of one color."""
    cov = np.clip(mask, 0, 1).astype(np.float32) * a
    img = np.empty(mask.shape + (4,), np.float32)
    img[..., :3] = rgb01(color) * cov[..., None]
    img[..., 3] = cov
    return img


def with_alpha(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Straight RGB (H, W, 3) + alpha (H, W) -> premultiplied RGBA."""
    a = np.clip(alpha, 0, 1).astype(np.float32)
    return np.concatenate([rgb * a[..., None], a[..., None]], axis=-1).astype(np.float32)


def opaque(rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([np.clip(rgb, 0, 1), np.ones(rgb.shape[:2] + (1,), np.float32)], axis=-1)


# ---------------------------------------------------------------- distance fields

def sd_circle(xx, yy, cx, cy, r):
    return np.hypot(xx - cx, yy - cy) - r


def sd_segment(xx, yy, ax, ay, bx, by, r=0.0):
    pax, pay = xx - ax, yy - ay
    bax, bay = bx - ax, by - ay
    denom = bax * bax + bay * bay
    h = np.clip((pax * bax + pay * bay) / denom, 0.0, 1.0) if denom > 0 else 0.0
    return np.hypot(pax - bax * h, pay - bay * h) - r


def sd_tapered(xx, yy, ax, ay, bx, by, ra, rb):
    """Capsule whose radius goes from ra at A to rb at B (approximate, good for strokes)."""
    pax, pay = xx - ax, yy - ay
    bax, bay = bx - ax, by - ay
    denom = bax * bax + bay * bay
    h = np.clip((pax * bax + pay * bay) / denom, 0.0, 1.0) if denom > 0 else np.zeros_like(xx)
    return np.hypot(pax - bax * h, pay - bay * h) - (ra + (rb - ra) * h)


def sd_box(xx, yy, cx, cy, hw, hh, r=0.0):
    qx = np.abs(xx - cx) - hw + r
    qy = np.abs(yy - cy) - hh + r
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def sd_polygon(xx, yy, pts: Sequence[Tuple[float, float]]):
    """Signed distance to a simple polygon (even-odd inside test)."""
    p = np.asarray(pts, np.float32)
    d = np.full(xx.shape, np.inf, np.float32)
    inside = np.zeros(xx.shape, bool)
    n = len(p)
    for i in range(n):
        ax, ay = p[i]
        bx, by = p[(i + 1) % n]
        d = np.minimum(d, sd_segment(xx, yy, ax, ay, bx, by))
        cond = (ay > yy) != (by > yy)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (bx - ax) * (yy - ay) / (by - ay) + ax
        inside ^= cond & (xx < xint)
    return np.where(inside, -d, d)


def coverage(sd, aa: float = 1.0):
    """Antialiased coverage from a signed distance (aa = edge width in px)."""
    return np.clip(0.5 - sd / aa, 0.0, 1.0)


def glow(sd, radius: float, power: float = 2.0):
    """Soft falloff outside a shape: 1 at the edge, ~0 at `radius`."""
    t = np.clip(1.0 - np.maximum(sd, 0) / radius, 0.0, 1.0)
    return t ** power


def glow_exp(sd, radius: float):
    return np.exp(-np.maximum(sd, 0) / radius)


# ---------------------------------------------------------------- gradients

def gradient_stops(t: np.ndarray, stops: Sequence[Tuple[float, Sequence[float]]]) -> np.ndarray:
    """Map t (any shape) through color stops [(pos, (r,g,b)), ...] -> (..., 3) float."""
    pos = np.array([s[0] for s in stops], np.float32)
    cols = np.array([rgb01(s[1]) for s in stops], np.float32)
    t = np.clip(t, pos[0], pos[-1])
    out = np.empty(t.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, pos, cols[:, c])
    return out


def linear_gradient(w, h, stops, angle_deg: float = 90.0):
    """angle 90 = top to bottom."""
    xx, yy = grid(w, h)
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    proj = (xx - w / 2) * dx + (yy - h / 2) * dy
    span = abs(w / 2 * dx) + abs(h / 2 * dy)
    t = (proj / span + 1) / 2
    return gradient_stops(t, stops)


def radial_t(w, h, cx, cy, r, ry: Optional[float] = None):
    xx, yy = grid(w, h)
    ry = ry or r
    return np.sqrt(((xx - cx) / r) ** 2 + ((yy - cy) / ry) ** 2)


# ---------------------------------------------------------------- noise

def _hash2(ix, iy, seed):
    h = (ix * 374761393 + iy * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def value_noise(xx, yy, scale: float, seed: int = 0):
    """Smooth value noise in ~0..1 sampled at (xx, yy) / scale."""
    x = xx / scale
    y = yy / scale
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    v00 = _hash2(x0, y0, seed)
    v10 = _hash2(x0 + 1, y0, seed)
    v01 = _hash2(x0, y0 + 1, seed)
    v11 = _hash2(x0 + 1, y0 + 1, seed)
    return lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy)


def fbm(xx, yy, scale: float, octaves: int = 5, seed: int = 0, gain: float = 0.5,
        lacunarity: float = 2.0):
    total = np.zeros_like(xx, dtype=np.float32)
    amp, norm, s = 1.0, 0.0, scale
    for o in range(octaves):
        total += value_noise(xx, yy, s, seed + o * 101) * amp
        norm += amp
        amp *= gain
        s /= lacunarity
    return total / norm


def fbm1d(n: int, scale: float, octaves: int = 5, seed: int = 0, gain: float = 0.5):
    xs = np.arange(n, dtype=np.float32)
    return fbm(xs[None, :], np.zeros((1, n), np.float32) + seed * 13.7, scale, octaves, seed, gain)[0]


def white_noise(w, h, seed: int = 0):
    return np.random.default_rng(seed).random((h, w), dtype=np.float32)


# ---------------------------------------------------------------- filters

def box_blur(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    if r <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float32)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def blur(a: np.ndarray, radius: float, passes: int = 3) -> np.ndarray:
    """Approximate gaussian blur (works on (H,W) or (H,W,C))."""
    r = max(1, int(round(radius / math.sqrt(passes))))
    out = a.astype(np.float32)
    for _ in range(passes):
        out = box_blur(out, r, 0)
        out = box_blur(out, r, 1)
    return out


def bloom(img: np.ndarray, radius: float, strength: float = 1.0) -> np.ndarray:
    """Add a blurred copy of img onto itself (additive)."""
    out = img + blur(img, radius) * strength
    return np.clip(out, 0, 1)


def pad(img: np.ndarray, p: int) -> np.ndarray:
    return np.pad(img, ((p, p), (p, p), (0, 0)))


# ---------------------------------------------------------------- shape helpers

def ridge(w: int, base_y: float, amp: float, scale: float, seed: int, octaves: int = 6,
          sharp: float = 0.0) -> np.ndarray:
    """1D mountain-ridge heights (y per column). sharp>0 makes peaky ridged noise."""
    n = fbm1d(w, scale, octaves, seed)
    if sharp > 0:
        n = lerp(n, 1 - np.abs(n * 2 - 1), sharp)
    return base_y - n * amp


def below_curve(w: int, h: int, ys: np.ndarray, aa: float = 1.0) -> np.ndarray:
    """Coverage of everything below a per-column curve."""
    _, yy = grid(w, h)
    return np.clip((yy - ys[None, :]) / aa + 0.5, 0, 1)


def scatter(rng: np.random.Generator, n: int, w: float, h: float):
    return rng.random(n) * w, rng.random(n) * h
