"""
Silhouette theme: cinematic, painterly silhouettes (think Limbo / Alto / Gris / Ori).

Battlefield  -> "Ember Dusk": a sunset sky over layered ranges with atmospheric perspective,
                a low sun with god-rays, a ruined arch and a lone windswept tree, reeds, embers.
Final Dest.  -> "Moonlit Blue Hour": a huge pale moon, fading stars, a sea of cloud, floating
                rock islands with hanging vines, falling petals.

Fighters are near-black silhouettes rim-lit in their character color from the sun / moon.
"""
import math
from typing import Optional, Tuple

import numpy as np

from gfx import raster as R
from gfx import xrender as X
from gfx.cache import cached
from gfx.context import Shape
from gfx.text import Face, FontStyle

from . import register
from .base import DODGE_COLORS, FighterLook, Theme

RGB = Tuple[int, int, int]

PAD_X, PAD_Y = 48, 28             # parallax margin baked around background layers

# Ember Dusk (Battlefield)
SUN_X, SUN_Y, SUN_R = 640, 436, 66
DUSK_HZ = 500
INK_DUSK = (17, 8, 16)
RIM_DUSK = (255, 198, 132)
HAZE_DUSK = (236, 120, 92)

# Moonlit Blue Hour (Final Destination)
MOON_X, MOON_Y, MOON_R = 905, 160, 62
NIGHT_HZ = 528
INK_NIGHT = (8, 11, 22)
RIM_NIGHT = (190, 224, 255)
HAZE_NIGHT = (150, 174, 214)

IVORY = (246, 238, 222)

CHAR_COLORS = {
    "RUSH": (255, 96, 66),      # ember red
    "ZONE": (92, 214, 255),     # glacier cyan
    "GRAB": (255, 196, 72),     # gold
    "FLEX": (96, 232, 160),     # jade
}
DEFAULT_COLORS = [(255, 136, 100), (132, 176, 255)]


def _mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


# =============================================================================
# GENERIC HELPERS (asset build time only)
# =============================================================================

def _screen_grid(W, H):
    xx, yy = R.grid(W, H)
    return xx - PAD_X, yy - PAD_Y


def _profile(n: int, scale: float, seed: int, octaves: int = 6, sharp: float = 0.0) -> np.ndarray:
    """1D noise profile normalized to 0..1 (sharp > 0 gives peaky ridges)."""
    p = R.fbm1d(n, scale, octaves, seed)
    p = (p - p.min()) / max(1e-6, float(p.max() - p.min()))
    if sharp > 0:
        p = R.lerp(p, 1 - np.abs(p * 2 - 1), sharp)
        p = (p - p.min()) / max(1e-6, float(p.max() - p.min()))
    return p.astype(np.float32)


def _capsules(mask: np.ndarray, segs, ox: float = 0.0, oy: float = 0.0):
    """Max-composite tapered capsules (ax, ay, bx, by, ra, rb) into a coverage mask in place.
    Segment coords are given in a space where the mask's top-left is (ox, oy)."""
    H, W = mask.shape
    for ax, ay, bx, by, ra, rb in segs:
        ax, ay, bx, by = ax - ox, ay - oy, bx - ox, by - oy
        r = max(ra, rb) + 2
        x0, x1 = int(max(0, min(ax, bx) - r)), int(min(W, max(ax, bx) + r + 1))
        y0, y1 = int(max(0, min(ay, by) - r)), int(min(H, max(ay, by) + r + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        xx, yy = R.grid(x1 - x0, y1 - y0, x0, y0)
        d = R.sd_tapered(xx, yy, ax, ay, bx, by, ra, rb)
        np.maximum(mask[y0:y1, x0:x1], R.coverage(d), out=mask[y0:y1, x0:x1])


def _blobs(mask: np.ndarray, circles, noise_amp: float = 0.0, seed: int = 0, ox: float = 0.0, oy: float = 0.0,
           squash: float = 1.0, noise_scale: float = 5.0):
    """Union of discs (cx, cy, r) with an optional noisy (leafy) edge, max-composited into mask."""
    if not circles:
        return
    H, W = mask.shape
    if noise_amp <= 0:
        for (cx, cy, r) in circles:
            cx, cy = cx - ox, cy - oy
            x0, x1 = int(max(0, cx - r - 2)), int(min(W, cx + r + 3))
            y0, y1 = int(max(0, cy - r * squash - 2)), int(min(H, cy + r * squash + 3))
            if x0 >= x1 or y0 >= y1:
                continue
            xx, yy = R.grid(x1 - x0, y1 - y0, x0, y0)
            c = R.coverage(np.hypot(xx - cx, (yy - cy) / squash) - r)
            np.maximum(mask[y0:y1, x0:x1], c, out=mask[y0:y1, x0:x1])
        return
    xs = [c[0] - ox for c in circles]
    ys = [c[1] - oy for c in circles]
    rm = max(c[2] for c in circles) + noise_amp + 3
    x0, x1 = int(max(0, min(xs) - rm)), int(min(W, max(xs) + rm + 1))
    y0, y1 = int(max(0, min(ys) - rm)), int(min(H, max(ys) + rm + 1))
    if x0 >= x1 or y0 >= y1:
        return
    xx, yy = R.grid(x1 - x0, y1 - y0, x0, y0)
    d = np.full(xx.shape, 1e6, np.float32)
    for (cx, cy, r) in circles:
        d = np.minimum(d, np.hypot(xx - (cx - ox), (yy - (cy - oy)) / squash) - r)
    if noise_amp > 0:
        n = R.fbm(xx, yy, noise_scale, 3, seed=seed)
        d = d + (n - 0.5) * 2 * noise_amp
    np.maximum(mask[y0:y1, x0:x1], R.coverage(d), out=mask[y0:y1, x0:x1])


def _curve(rng, x, y, ang, length, steps, bend, gravity=0.0):
    """Points along a gently bending stroke starting at (x, y)."""
    pts = [(x, y)]
    step = length / steps
    for _ in range(steps):
        ang += rng.normal(0, bend) + gravity
        x += math.cos(ang) * step
        y += math.sin(ang) * step
        pts.append((x, y))
    return pts, ang


def _stroke(pts, w0, w1):
    n = len(pts) - 1
    segs = []
    for i in range(n):
        ra = (w0 + (w1 - w0) * i / n) / 2
        rb = (w0 + (w1 - w0) * (i + 1) / n) / 2
        segs.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], ra, rb))
    return segs


def _tree(rng, x, y, scale, lean):
    """A windswept tree: tapered branch capsules streaming downwind + flattened leaf clumps."""
    segs, leaves = [], []
    target = -0.32          # branches get pulled toward this angle (up and to the right)

    def branch(px, py, ang, length, rad, depth):
        r0 = rad
        for i in range(3):
            ang += rng.normal(0, 0.08) + (target - ang) * 0.10 * lean * (1.0 if depth < 4 else 0.35)
            l = length / 3
            nx, ny = px + math.cos(ang) * l, py + math.sin(ang) * l
            r1 = rad * (1 - 0.32 * (i + 1) / 3)
            segs.append((px, py, nx, ny, r0, r1))
            px, py, r0 = nx, ny, r1
        if depth <= 1:
            leaves.append((px + rng.normal(0, 2), py + rng.normal(0, 1.5), (8 + rng.random() * 7) * scale))
        if depth == 0:
            return
        for k, spread in enumerate((-0.55, 0.45)):
            branch(px, py, ang + spread + rng.normal(0, 0.12), length * rng.uniform(0.66, 0.82), r0 * 0.72,
                   depth - 1)

    branch(x, y, -math.pi / 2 + 0.05, 52 * scale, 5.2 * scale, 4)
    for k in (-1, 1):
        segs.append((x, y - 5 * scale, x + k * 11 * scale, y + 4 * scale, 3.2 * scale, 1.2 * scale))
    return segs, leaves


def _stars(img, sx, sy, rng, n, y_max, fade_to, avoid=None):
    H, W = img.shape[:2]
    for _ in range(n):
        x = rng.random() * W - PAD_X
        y = rng.random() * y_max
        if avoid is not None and math.hypot(x - avoid[0], y - avoid[1]) < avoid[2]:
            continue
        size = rng.random() ** 3 * 1.5 + 0.45
        b = (0.3 + rng.random() * 0.7) * max(0.0, 1 - y / fade_to) ** 1.3
        if b <= 0.02:
            continue
        px, py = x + PAD_X, y + PAD_Y
        x0, y0 = int(px) - 4, int(py) - 4
        gx, gy = R.grid(9, 9, x0, y0)
        d = np.hypot(gx - px, gy - py)
        a = (R.coverage(d - size * 0.5) + np.exp(-d / (size * 1.3)) * 0.35) * b
        col = (215 + rng.integers(0, 40), 222 + rng.integers(0, 33), 255)
        R.composite(img, R.tint(np.clip(a, 0, 1), col), x0, y0, op="add")


def _paint_ridge(img, sx, top, col, fog_col, fog_px, rim_col=None, rim=0.0, light_x=SUN_X, fog_amt=1.0):
    """Mountain range below the per-column curve `top` (screen y), hazier toward its base.
    Only the rows the range can touch are processed."""
    H, W = img.shape[:2]
    y0 = int(max(0, min(H - 1, top.min() + PAD_Y - 10)))
    sub = img[y0:]
    _, yy = R.grid(W, H - y0, 0, y0)
    depth = (yy - PAD_Y) - top[None, :]
    cov = np.clip(depth + 0.5, 0, 1)
    fog = (np.clip(depth / fog_px, 0, 1) ** 0.9) * fog_amt
    color = R.gradient_stops(fog, [(0, col), (1, fog_col)])
    R.paint(sub, cov, color)
    if rim_col is not None and rim > 0:
        slope = np.gradient(top)
        x = sx[0]
        facing = np.clip(np.sign(light_x - x) * slope * 1.1 + 0.2, 0, 1)[None, :]
        line = np.exp(-np.maximum(depth, 0) / 1.2) * cov
        halo = np.exp(np.minimum(depth, 0) / 4.0) * (depth < 0) * 0.3
        R.paint(sub, np.clip((line + halo) * facing * rim, 0, 1), rim_col, op="screen")


def _rows(H, y_lo, y_hi):
    """Canvas row slice covering screen rows y_lo..y_hi."""
    return slice(int(max(0, min(H, y_lo + PAD_Y))), int(max(0, min(H, y_hi + PAD_Y))))


def _vgrad(W, H, stops) -> np.ndarray:
    """Vertical gradient keyed on screen y, computed once per row and broadcast."""
    ys = np.arange(H, dtype=np.float32) + 0.5 - PAD_Y
    col = R.gradient_stops(ys, stops)
    return np.ascontiguousarray(np.broadcast_to(col[:, None, :], (H, W, 3)))


def _haze(img, sy, y, up, down, col, a):
    rs = _rows(img.shape[0], y - up * 6, y + down * 6)
    d = sy[rs] - y
    m = np.where(d < 0, np.exp(d / up), np.exp(-d / down)) * a
    R.paint(img[rs], m, col, op="screen")


def _screen_layers(img, layers):
    """Several screen-blended color washes on an opaque image in one pass."""
    keep = np.ones(img.shape[:2] + (3,), np.float32)
    for m, c in layers:
        keep *= 1 - m[..., None] * R.rgb01(c)
    img[..., :3] = 1 - (1 - img[..., :3]) * keep


def _vignette(img, sx, sy, w, h, strength, col):
    v = np.sqrt(((sx - w / 2) / (w * 0.62)) ** 2 + ((sy - h / 2) / (h * 0.72)) ** 2)
    R.paint(img, np.clip(v - 0.62, 0, 1) ** 1.5 * strength, col)


# =============================================================================
# SPRITES
# =============================================================================

def _glow_sprite(size: int, power: float = 2.2) -> np.ndarray:
    xx, yy = R.grid(size, size)
    d = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    return R.tint(np.clip(1 - d, 0, 1) ** power, (255, 255, 255))


def _ember_sprite(size: int) -> np.ndarray:
    xx, yy = R.grid(size, size)
    d = np.hypot(xx - size / 2, yy - size / 2)
    a = R.coverage(d - size * 0.09) * 0.9 + np.exp(-d / (size * 0.16)) * 0.6
    a *= np.clip(1 - d / (size / 2), 0, 1)
    return R.tint(np.clip(a, 0, 1), (255, 255, 255))


def _streak_sprite(w: int, h: int) -> np.ndarray:
    """Anamorphic light streak: long soft horizontal line."""
    xx, yy = R.grid(w, h)
    dx = np.abs(xx - w / 2) / (w / 2)
    dy = np.abs(yy - h / 2) / (h / 2)
    a = np.clip(1 - dx, 0, 1) ** 2.2 * (np.exp(-(dy * 3.2) ** 2) * 0.8 + np.clip(1 - dy, 0, 1) ** 3 * 0.2)
    return R.tint(np.clip(a, 0, 1), (255, 255, 255))


def _ring_sprite(size: int, width: float) -> np.ndarray:
    xx, yy = R.grid(size, size)
    d = np.abs(np.hypot(xx - size / 2, yy - size / 2) - (size / 2 - width * 3))
    a = np.exp(-(d / width) ** 2) + np.exp(-(d / (width * 3.5)) ** 2) * 0.25
    return R.tint(np.clip(a, 0, 1), (255, 255, 255))


def _glint_sprite(size: int) -> np.ndarray:
    """Tiny four-point star."""
    xx, yy = R.grid(size, size)
    dx, dy = np.abs(xx - size / 2) / (size / 2), np.abs(yy - size / 2) / (size / 2)
    core = np.exp(-(np.hypot(dx, dy) / 0.14) ** 2)
    star = np.clip(1 - dx, 0, 1) ** 4 * np.exp(-(dy / 0.06) ** 2) + np.clip(1 - dy, 0, 1) ** 4 * np.exp(-(dx / 0.06) ** 2)
    halo = np.clip(1 - np.hypot(dx, dy), 0, 1) ** 3 * 0.3
    return R.tint(np.clip(core + star * 0.8 + halo, 0, 1), (255, 255, 255))


def _puff_sprite(size: int) -> np.ndarray:
    """Soft, slightly billowy dust puff."""
    xx, yy = R.grid(size, size)
    d = np.hypot(xx - size / 2, (yy - size / 2) * 1.15) / (size / 2)
    n = R.fbm(xx, yy, size / 3.0, 4, seed=17)
    a = np.clip(1 - d + (n - 0.5) * 0.7, 0, 1) ** 1.6
    return R.tint(np.clip(a, 0, 1), (255, 255, 255))


def _bubble_sprite(size: int) -> np.ndarray:
    """Shield: translucent light bubble with fresnel rim and a soft highlight."""
    xx, yy = R.grid(size, size)
    c = size / 2
    r = size / 2 - 6
    d = np.hypot(xx - c, yy - c)
    inside = R.coverage(d - r)
    fres = np.clip(d / r, 0, 1) ** 4
    rim = np.exp(-((d - r) / 1.3) ** 2) + np.exp(-np.abs(d - r) / 4.0) * 0.35
    hl = np.exp(-(np.hypot(xx - (c - r * 0.38), yy - (c - r * 0.42)) / (r * 0.22)) ** 2) * 0.45
    a = inside * (0.05 + fres * 0.45 + hl) + rim
    return R.tint(np.clip(a, 0, 1), (255, 255, 255))


def _petal_sheet(n: int = 8, s: int = 16) -> np.ndarray:
    """Falling petal in n flutter phases, side by side (pale moonlit pink)."""
    img = R.blank(n * s, s)
    for k in range(n):
        xx, yy = R.grid(s, s)
        th = k * math.pi / n
        squash = 0.3 + 0.7 * abs(math.cos(k * 2 * math.pi / n))
        x, y = xx - s / 2, yy - s / 2
        u = (x * math.cos(th) + y * math.sin(th)) / 5.2
        v = (-x * math.sin(th) + y * math.cos(th)) / (2.6 * squash + 0.4)
        d = np.sqrt(u * u + v * v) - 1 + u * 0.18
        a = R.coverage(d * 3.5) * (0.85 + 0.15 * np.clip(-u, 0, 1))
        col = R.gradient_stops(np.clip(u * 0.5 + 0.5, 0, 1), [(0, (252, 236, 246)), (1, (214, 182, 214))])
        img[:, k * s:(k + 1) * s] = R.with_alpha(col, a)
    return img


def _bird_sheet(n: int = 4, s: int = 18) -> np.ndarray:
    """A small bird in n wing-beat phases, side by side (dark, hazy)."""
    img = R.blank(n * s, s)
    tips = [-5.0, -1.5, 3.0, -1.0]
    for k in range(n):
        m = np.zeros((s, s), np.float32)
        cx, cy = s / 2, s / 2 + 1
        ty = tips[k % len(tips)]
        segs = [(cx, cy, cx - 4, cy + ty * 0.35 - 1, 1.1, 0.8), (cx - 4, cy + ty * 0.35 - 1, cx - 8, cy + ty, 0.8, 0.25),
                (cx, cy, cx + 4, cy + ty * 0.35 - 1, 1.1, 0.8), (cx + 4, cy + ty * 0.35 - 1, cx + 8, cy + ty, 0.8, 0.25),
                (cx - 1.5, cy + 0.3, cx + 2.5, cy + 0.1, 1.2, 0.9)]
        _capsules(m, segs)
        img[:, k * s:(k + 1) * s] = R.tint(m, (44, 18, 40))
    return img


def _hud_fade(w: int, h: int, right: bool) -> np.ndarray:
    xx, yy = R.grid(w, h)
    cx = w if right else 0
    d = np.sqrt(((xx - cx) / w) ** 2 + ((yy - h) / h) ** 2)
    a = np.clip(1 - d, 0, 1) ** 1.5 * 0.78
    return R.tint(a, (6, 3, 9))


def _panel_strip() -> np.ndarray:
    """Vertical gradient used (stretched) as the translucent panel body."""
    h = 128
    img = R.blank(8, h)
    t = (np.arange(h, dtype=np.float32) + 0.5) / h
    a = 0.66 + 0.2 * t
    col = R.gradient_stops(t, [(0, (30, 18, 36)), (0.5, (16, 9, 20)), (1, (8, 4, 10))])
    img[..., :3] = (col * a[:, None])[:, None, :]
    img[..., 3] = a[:, None]
    return img


def _mist_sprite(w: int, h: int, seed: int) -> np.ndarray:
    """A long, soft wisp of low mist (edges fade so instances can drift and overlap)."""
    xx, yy = R.grid(w, h)
    n = R.fbm(xx * 0.5, yy * 1.6, 70, 5, seed=seed)
    prof = np.exp(-((yy - h * 0.55) / (h * 0.28)) ** 2)
    edge = np.clip(np.minimum(xx, w - xx) / (w * 0.22), 0, 1) ** 1.5
    a = np.clip((n - 0.32) * 1.9, 0, 1) * prof * edge
    return R.tint(np.clip(a, 0, 1), (255, 255, 255))


def _reed_sprite(seed: int, w: int = 110, h: int = 270) -> np.ndarray:
    """A clump of cattails and long leaves, anchored at bottom center."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), np.float32)
    bx, by = w / 2, h + 2
    segs = []
    for i in range(9):
        ang = -math.pi / 2 + rng.normal(0, 0.22)
        length = h * rng.uniform(0.35, 0.8)
        pts, _ = _curve(rng, bx + rng.normal(0, 5), by, ang, length, 6, 0.05,
                        gravity=0.035 * (1 if ang > -math.pi / 2 else -1))
        segs += _stroke(pts, rng.uniform(4.5, 6.5), 0.6)
    heads = []
    for i in range(3):
        ang = -math.pi / 2 + rng.normal(0, 0.1)
        length = h * rng.uniform(0.72, 0.96)
        pts, a2 = _curve(rng, bx + rng.normal(0, 4), by, ang, length, 6, 0.025)
        segs += _stroke(pts, 2.6, 1.6)
        tx, ty = pts[-1]
        hx, hy = tx - math.cos(a2) * 14, ty - math.sin(a2) * 14
        heads.append((tx, ty, hx, hy, a2))
        segs.append((tx, ty, tx + math.cos(a2) * 12, ty + math.sin(a2) * 12, 0.9, 0.3))
    _capsules(mask, segs)
    for tx, ty, hx, hy, a2 in heads:
        _capsules(mask, [(tx, ty, hx - math.cos(a2) * 8, hy - math.sin(a2) * 8, 3.6, 3.4)])
    return R.tint(mask, INK_DUSK)


# =============================================================================
# EMBER DUSK (Battlefield)
# =============================================================================

def _dusk_sky(W: int, H: int) -> np.ndarray:
    """Sky, sun, god-rays and lit clouds (opaque, padded coordinates)."""
    sx, sy = _screen_grid(W, H)
    sky = _vgrad(W, H, [
        (-30, (12, 40, 62)), (70, (22, 40, 80)), (170, (56, 42, 106)), (260, (116, 54, 116)),
        (340, (186, 80, 104)), (410, (232, 120, 84)), (465, (236, 106, 66)), (DUSK_HZ, (218, 76, 58)),
        (560, (160, 50, 56)), (H, (70, 22, 38))])
    img = R.opaque(sky)
    rng = np.random.default_rng(3)
    _stars(img, sx, sy, rng, 70, 170, 190)

    e = np.hypot((sx - SUN_X) / 1.9, sy - SUN_Y)
    d = np.hypot(sx - SUN_X, sy - SUN_Y)
    _screen_layers(img, [(np.exp(-e / 320.0) * 0.55, (255, 132, 70)), (np.exp(-e / 120.0) * 0.55, (255, 186, 106))])

    # god-rays: angular noise fanning out of the sun, only above the horizon
    rs = _rows(H, -PAD_Y, DUSK_HZ + 12)
    ang = np.arctan2(sy[rs] - SUN_Y, sx[rs] - SUN_X)
    n = R.value_noise(ang * 10 + 40, np.zeros_like(ang) + 3.3, 1.0, seed=5) * 0.6 + \
        R.value_noise(ang * 23 + 9, np.zeros_like(ang) + 7.1, 1.0, seed=8) * 0.4
    rays = np.clip((n - 0.45) * 2.0, 0, 1) ** 1.6
    fall = np.exp(-d[rs] / 320.0) * np.clip((d[rs] - SUN_R) / 90, 0, 1) * \
        R.smoothstep(DUSK_HZ + 10, DUSK_HZ - 60, sy[rs])
    R.paint(img[rs], rays * fall * 0.13, (255, 206, 150), op="screen")

    # cloud wisps lit from below
    rs = _rows(H, 56, 424)
    csx, csy = sx[rs], sy[rs]
    cn = R.fbm(csx * 0.42 + 900, csy * 3.0, 150, 5, seed=4)
    band = R.smoothstep(60, 170, csy) * R.smoothstep(420, 300, csy)
    dens = np.clip((cn - 0.5) * 3.6, 0, 1) * band
    below = np.roll(dens, -4, axis=0)
    under = np.clip((dens - below) * 2.2, 0, 1)
    body = R.gradient_stops(csy, [(60, (56, 40, 92)), (240, (108, 52, 104)), (420, (178, 76, 92))])
    R.paint(img[rs], dens * 0.75, body)
    warm = R.gradient_stops(np.clip(e[rs] / 520.0, 0, 1), [(0, (255, 212, 150)), (0.5, (255, 140, 110)),
                                                          (1, (200, 92, 120))])
    R.paint(img[rs], np.clip(under * 0.9 + dens * 0.18, 0, 1) * band, warm, op="screen")

    # the sun itself, soft disc with limb darkening and a tight bloom
    rs = _rows(H, SUN_Y - SUN_R * 6, SUN_Y + SUN_R * 6)
    R.paint(img[rs], np.exp(-np.maximum(d[rs] - SUN_R, 0) / 24.0) * 0.9, (255, 222, 160), op="screen")
    rs = _rows(H, SUN_Y - SUN_R - 3, SUN_Y + SUN_R + 3)
    limb = np.clip(d[rs] / SUN_R, 0, 1)
    disc = R.gradient_stops(limb, [(0, (255, 248, 222)), (0.75, (255, 232, 180)), (1, (255, 200, 130))])
    R.paint(img[rs], R.coverage(d[rs] - SUN_R, 1.5), disc)
    return img


def _dusk_far(W: int, H: int) -> np.ndarray:
    img = _dusk_sky(W, H)
    sx, sy = _screen_grid(W, H)
    x = sx[0]
    ax = np.abs(x - SUN_X)
    # three distant ranges: lighter, hazier and closer to the sky color the farther they are
    p1 = _profile(W, 210, 21, 6, sharp=0.55)
    env1 = R.smoothstep(60, 560, ax)
    top1 = 452 - env1 * 150 - p1 * 70 + 26
    _paint_ridge(img, sx, top1, (186, 92, 116), (240, 140, 100), 90, RIM_DUSK, 0.35)
    _haze(img, sy, DUSK_HZ - 26, 18, 30, HAZE_DUSK, 0.32)

    p2 = _profile(W, 150, 33, 6, sharp=0.6)
    env2 = R.smoothstep(140, 600, ax)
    top2 = 478 - env2 * 120 - p2 * 55 + 22
    _paint_ridge(img, sx, top2, (138, 58, 98), (226, 112, 94), 110, RIM_DUSK, 0.5)
    _haze(img, sy, DUSK_HZ - 2, 14, 30, HAZE_DUSK, 0.3)

    p3 = _profile(W, 120, 47, 6, sharp=0.5)
    env3 = R.smoothstep(220, 620, ax)
    top3 = 505 - env3 * 95 - p3 * 45 + 18
    # a plateau for the ruined arch on the right
    arch_x = 1092
    wgt = np.exp(-((x - arch_x) / 70.0) ** 4)
    plateau = 430.0
    top3 = top3 * (1 - wgt) + plateau * wgt
    _paint_ridge(img, sx, top3, (86, 34, 72), (196, 86, 82), 120, RIM_DUSK, 0.65)
    _ruin(img, arch_x + PAD_X, plateau + PAD_Y + 1, (84, 33, 70), RIM_DUSK)
    _haze(img, sy, DUSK_HZ + 24, 16, 40, HAZE_DUSK, 0.32)
    _vignette(img, sx, sy, W - 2 * PAD_X, H - 2 * PAD_Y, 0.5, (20, 6, 22))
    return np.clip(img, 0, 1)


def _ruin(img, cx, base, col, rim_col):
    """Ruined temple arch with a lone broken column, silhouetted on a far plateau."""
    x0, y0 = int(cx - 110), int(base - 120)
    w, h = 220, 124
    xx, yy = R.grid(w, h, x0, y0)
    big = 1e6
    sd = R.sd_box(xx, yy, cx, base - 4, 64, 5)                              # stylobate
    sd = np.minimum(sd, R.sd_box(xx, yy, cx, base - 11, 54, 3))
    for px in (-34, 34):                                                     # pillars
        top = 72 if px < 0 else 60
        p = R.sd_box(xx, yy, cx + px, base - 14 - top / 2, 7, top / 2)
        sd = np.minimum(sd, p)
        sd = np.minimum(sd, R.sd_box(xx, yy, cx + px, base - 14 - top, 10, 3))  # capital
    # arch: upper half ring between the pillars, broken on the right
    acx, acy, rm = cx, base - 86, 34
    ring = np.abs(np.hypot(xx - acx, yy - acy) - rm) - 6
    ring = np.where(yy <= acy + 1, ring, big)
    ang = np.degrees(np.arctan2(yy - acy, xx - acx))
    ring = np.where((ang > -52) & (ang < 10), big, ring)                    # missing chunk
    sd = np.minimum(sd, ring)
    sd = np.minimum(sd, R.sd_box(xx, yy, cx + 34, base - 77, 8, 2.5))
    # broken tall column to the right, fallen block to the left
    sd = np.minimum(sd, R.sd_box(xx, yy, cx + 82, base - 40, 6, 40))
    sd = np.minimum(sd, R.sd_polygon(xx, yy, [(cx + 75, base - 79), (cx + 89, base - 86), (cx + 89, base - 78),
                                              (cx + 75, base - 76)]))
    sd = np.minimum(sd, R.sd_box(xx, yy, cx - 78, base - 6, 13, 6, 1.5))
    cov = R.coverage(sd)
    R.composite(img, R.tint(cov, col), x0, y0)
    # faint rim along the sun-facing (left) edges
    edge = np.exp(-np.abs(sd) / 0.9) * (sd > -2) * np.clip(-np.gradient(sd, axis=1) * 2, 0, 1)
    R.composite(img, R.tint(np.clip(edge * 0.45, 0, 1), rim_col), x0, y0, op="add")


def _dusk_mid(W: int, H: int) -> np.ndarray:
    """Near ridge with the lone tree, valley mist and dark foothills (transparent above)."""
    img = R.blank(W, H)
    sx, sy = _screen_grid(W, H)
    x = sx[0]
    ax = np.abs(x - SUN_X)
    p4 = _profile(W, 170, 59, 6, sharp=0.45)
    env4 = R.smoothstep(330, 660, ax)
    tilt = np.where(x < SUN_X, 1.0, 0.8)
    top4 = 590 - env4 * 175 * tilt - p4 * 40 * (0.3 + env4)
    tree_x = 96
    ti = int(tree_x + PAD_X)
    _paint_ridge(img, sx, top4, (48, 18, 42), (120, 46, 66), 180, RIM_DUSK, 0.9)
    rng = np.random.default_rng(12)
    segs, leaves = _tree(rng, tree_x + PAD_X, float(top4[ti]) + PAD_Y + 4, 1.3, lean=1.0)
    mask = np.zeros((H, W), np.float32)
    _capsules(mask, segs)
    _blobs(mask, leaves, noise_amp=3.0, seed=6, squash=0.42, noise_scale=3.5)
    R.paint(img, mask, (40, 14, 36))
    # warm light catching the tree's sun-facing side
    glow = np.clip(mask - np.roll(mask, -2, axis=1), 0, 1)
    R.paint(img, glow * 0.4, RIM_DUSK, op="screen")

    # valley fog filling the lowlands below the platforms, lit by the sun
    rs = _rows(H, 556, H)
    fog = R.smoothstep(560, 645, sy[rs])
    fcol = R.gradient_stops(sy[rs], [(580, (232, 120, 98)), (650, (198, 92, 88)), (740, (132, 52, 72))])
    R.paint(img[rs], fog * 0.86, fcol)
    # foothills sinking into the fog
    p5 = _profile(W, 110, 71, 5, sharp=0.3)
    top5 = 662 - R.smoothstep(250, 680, ax) * 62 - p5 * 26
    _paint_ridge(img, sx, top5, (70, 26, 52), (170, 72, 82), 46, RIM_DUSK, 0.5, fog_amt=0.85)
    rs = _rows(H, 686, H)
    R.paint(img[rs], R.smoothstep(690, 760, sy[rs]) * 0.55, (120, 44, 66))
    return np.clip(img, 0, 1)


def _dusk_fg(W: int, H: int) -> np.ndarray:
    """Grass and reed silhouettes along the bottom edge (tall at the corners, low in the center)."""
    rng = np.random.default_rng(91)
    band = 180
    oy = H - band
    mask = np.zeros((band, W), np.float32)
    segs = []
    x = 0.0
    while x < W:
        sxp = x - PAD_X
        dc = abs(sxp - 640)
        if dc > 560:
            hmax = 120
        elif dc > 400:
            hmax = 26 + (dc - 400) / 160 * 70
        else:
            hmax = 14 + 8 * rng.random()
        h = hmax * rng.uniform(0.35, 1.0)
        ang = -math.pi / 2 + rng.normal(0.12, 0.18)
        pts, _ = _curve(rng, x, H + 2, ang, h, 4, 0.06, gravity=0.04)
        segs += _stroke(pts, rng.uniform(2.6, 4.2), 0.5)
        x += rng.uniform(1.5, 4.5) if dc > 400 else rng.uniform(3, 7)
    _capsules(mask, segs, 0, oy)
    # ground strip to anchor the blades
    _, yy = R.grid(W, band, 0, oy)
    mask = np.maximum(mask, np.clip((yy - (H - 6)) / 3, 0, 1))
    img = R.blank(W, H)
    img[oy:] = R.tint(mask, INK_DUSK)
    return img


def _dusk_slab(w: int, h: int):
    """Main Battlefield platform: an ancient masonry slab on an inverted rock mass, with roots."""
    GX, GT, GB = 26, 18, 128
    W, H = w + 2 * GX, GT + h + GB
    xx, yy = R.grid(W, H)
    lx, ly = xx - GX, yy - GT
    col_x = np.arange(W, dtype=np.float32) + 0.5 - GX
    t = np.clip(col_x / w, 0, 1)
    slab = R.sd_box(lx, ly, w / 2, h / 2, w / 2, h / 2, 2.5)
    env = np.clip(np.sin(np.pi * t), 0, 1) ** 0.6 * (col_x > 0) * (col_x < w)
    prof = _profile(W, 46, 7, 5, sharp=0.6)
    fine = _profile(W, 9, 8, 3)
    bottom = h - 4 + env * (30 + 62 * prof + 8 * fine)
    inset = 22 + 10 * np.sin(t * 9)
    side = np.clip(np.minimum(col_x - inset * (1 - env), w - inset * (1 - env) - col_x), 0, 1)
    under = np.clip(bottom[None, :] - ly + 0.5, 0, 1) * (ly > h - 8) * side[None, :]
    body = np.maximum(R.coverage(slab), under)

    img = R.blank(W, H)
    depth = np.clip(ly / (h + 90), 0, 1)
    col = R.gradient_stops(depth, [(0, (34, 16, 28)), (0.12, INK_DUSK), (0.7, (24, 10, 22)), (1, (60, 24, 40))])
    R.paint(img, body, col)
    # masonry: a lit cornice, a shadow groove, then courses of blocks fading into darkness
    sc = R.coverage(slab)
    near = np.exp(-np.maximum(ly - 8, 0) / 22.0) * sc
    R.paint(img, ((ly > 0) & (ly < 7)) * sc * 0.55, (66, 32, 44))
    R.paint(img, ((ly >= 7) & (ly < 9.5)) * sc, (8, 3, 8))
    course = (ly > 28) & (ly < 29.6)
    off = np.where(ly < 29, 0, 26)
    vj = (np.mod(lx + off, 52) < 1.4) & (ly > 9.5)
    R.paint(img, (course | vj) * near * 0.85, (62, 28, 40))
    R.paint(img, (np.mod(lx + off - 1.4, 52) < 1.2) * (ly > 9.5) * near * 0.5, (6, 2, 6))
    tex = R.fbm(xx, yy, 6, 3, seed=3)
    R.paint(img, np.clip((tex - 0.5) * 1.6, 0, 1) * near * 0.35, (110, 54, 56), op="screen")
    # rim light: a thin bright lip, a warm falloff down the face and a faint glow above
    inx = np.clip(np.minimum(lx, w - lx) + 0.5, 0, 1)
    lip = np.clip(1.8 - np.abs(ly - 0.6), 0, 1) * inx
    R.paint(img, np.exp(-np.maximum(ly, 0) / 5.0) * R.coverage(slab) * 0.35, RIM_DUSK, op="screen")
    R.paint(img, lip, (255, 222, 170))
    above = np.exp(np.minimum(ly, 0) / 4.0) * (ly < 0) * inx * 0.35
    R.paint(img, above, RIM_DUSK, op="add")
    # roots hanging from the underside
    rng = np.random.default_rng(w)
    mask = np.zeros((H, W), np.float32)
    segs = []
    for _ in range(26):
        cx = rng.uniform(0.08, 0.92) * w
        ci = int(np.clip(cx + GX, 0, W - 1))
        y0 = bottom[ci] - 3
        if y0 < h:
            continue
        pts, _ = _curve(rng, cx + GX, y0 + GT, math.pi / 2 + rng.normal(0, 0.3), rng.uniform(14, 62), 6, 0.22)
        segs += _stroke(pts, rng.uniform(2.0, 3.4), 0.5)
    _capsules(mask, segs)
    R.paint(img, mask, (30, 12, 26))
    return np.clip(img, 0, 1), GX, GT


def _ledge(w: int, h: int, kind: str):
    """Soft platform: a slim stone ledge with a small rocky underside and hanging roots."""
    GX, GT, GB = 16, 14, 60
    W, H = w + 2 * GX, GT + h + GB
    xx, yy = R.grid(W, H)
    lx, ly = xx - GX, yy - GT
    night = kind == "night"
    ink = INK_NIGHT if night else INK_DUSK
    rim_col = RIM_NIGHT if night else RIM_DUSK
    col_x = np.arange(W, dtype=np.float32) + 0.5 - GX
    t = np.clip(col_x / w, 0, 1)
    slab = R.sd_box(lx, ly, w / 2, 4.5, w / 2, 4.5, 4)
    env = np.clip(np.sin(np.pi * t), 0, 1) ** 0.8 * (col_x > 6) * (col_x < w - 6)
    prof = _profile(W, 22, w + 5, 4, sharp=0.6)
    bottom = 7 + env * (6 + 16 * prof)
    under = np.clip(bottom[None, :] - ly + 0.5, 0, 1) * (ly > 3)
    body = np.maximum(R.coverage(slab), under * (env[None, :] > 0))
    img = R.blank(W, H)
    lift = (40, 20, 32) if not night else (22, 30, 52)
    col = R.gradient_stops(np.clip(ly / 30, 0, 1), [(0, lift), (0.25, ink), (1, ink)])
    R.paint(img, body, col)
    tex = R.fbm(xx, yy, 5, 3, seed=w)
    near = np.exp(-np.maximum(ly, 0) / 5.0) * R.coverage(slab)
    R.paint(img, np.clip((tex - 0.45) * 1.5, 0, 1) * near * 0.4, rim_col, op="screen")
    inx = np.clip(np.minimum(lx, w - lx) - 1, 0, 1)
    R.paint(img, np.clip(1.6 - np.abs(ly - 0.5), 0, 1) * inx, _mix(rim_col, (255, 255, 255), 0.35))
    R.paint(img, np.exp(np.minimum(ly, 0) / 3.5) * (ly < 0) * inx * 0.3, rim_col, op="add")
    rng = np.random.default_rng(w * 3 + (7 if night else 0))
    mask = np.zeros((H, W), np.float32)
    segs, leaves = [], []
    for _ in range(max(3, w // 40)):
        cx = rng.uniform(0.15, 0.85) * w
        y0 = bottom[int(cx + GX)] - 2
        pts, _ = _curve(rng, cx + GX, y0 + GT, math.pi / 2 + rng.normal(0, 0.25), rng.uniform(10, 38), 5, 0.25)
        segs += _stroke(pts, 2.0, 0.5)
        if night:
            for (px, py) in pts[1::2]:
                leaves.append((px + rng.normal(0, 2), py, 1.8))
    _capsules(mask, segs)
    _blobs(mask, leaves)
    R.paint(img, mask, ink)
    return np.clip(img, 0, 1), GX, GT


# =============================================================================
# MOONLIT BLUE HOUR (Final Destination)
# =============================================================================

def _night_far(W: int, H: int) -> np.ndarray:
    sx, sy = _screen_grid(W, H)
    x = sx[0]
    sky = _vgrad(W, H, [
        (-30, (5, 8, 24)), (110, (12, 20, 52)), (250, (30, 40, 92)), (370, (62, 76, 136)),
        (460, (104, 124, 178)), (NIGHT_HZ, (156, 180, 218)), (NIGHT_HZ + 30, (128, 150, 198)),
        (H, (64, 78, 128))])
    img = R.opaque(sky)
    rng = np.random.default_rng(41)
    _stars(img, sx, sy, rng, 380, 470, 470, avoid=(MOON_X, MOON_Y, MOON_R * 2.2))

    d = np.hypot(sx - MOON_X, sy - MOON_Y)
    _screen_layers(img, [(np.exp(-d / 260.0) * 0.42, (96, 120, 196)),
                         (np.exp(-np.maximum(d - MOON_R, 0) / 34.0) * 0.55, (196, 212, 245))])
    R.paint(img, np.exp(-((d - MOON_R * 3.3) / 10.0) ** 2) * 0.04, (190, 210, 250), op="add")
    # thin cirrus streaks, silvered near the moon
    rs = _rows(H, 66, 334)
    cn = R.fbm(sx[rs] * 0.35 + 300, sy[rs] * 3.4, 140, 5, seed=12)
    band = R.smoothstep(70, 150, sy[rs]) * R.smoothstep(330, 230, sy[rs])
    dens = np.clip((cn - 0.56) * 3.2, 0, 1) * band
    lit = R.gradient_stops(np.clip(d[rs] / 520, 0, 1), [(0, (214, 226, 250)), (1, (92, 106, 160))])
    R.paint(img[rs], dens * 0.55, lit, op="screen")

    # moon: limb-darkened disc with soft maria (computed in its bounding box only)
    bx0, by0 = int(MOON_X + PAD_X - MOON_R - 3), int(MOON_Y + PAD_Y - MOON_R - 3)
    bs = int(MOON_R * 2 + 6)
    mx, my = R.grid(bs, bs, bx0, by0)
    md = np.hypot(mx - PAD_X - MOON_X, my - PAD_Y - MOON_Y)
    disc = R.coverage(md - MOON_R, 1.4)
    mu = np.sqrt(np.clip(1 - (md / MOON_R) ** 2, 0, 1))
    maria = R.fbm(mx, my, 28, 4, seed=9)
    tone = np.clip(0.75 + mu * 0.25 - np.clip((maria - 0.5) * 1.4, 0, 0.3) * 0.5, 0, 1)
    mcol = R.gradient_stops(tone, [(0.55, (150, 166, 204)), (0.8, (214, 222, 242)), (1, (246, 248, 255))])
    R.paint(img[by0:by0 + bs, bx0:bx0 + bs], disc, mcol)

    # distant hazy ranges along the horizon and two tiny far islands
    p0 = _profile(W, 140, 61, 6, sharp=0.6)
    top0 = NIGHT_HZ - 8 - p0 * 46 * (0.4 + 0.6 * R.smoothstep(120, 600, np.abs(x - 640)))
    _paint_ridge(img, sx, top0, (88, 104, 158), (150, 174, 214), 50, RIM_NIGHT, 0.3, light_x=MOON_X)
    for (ix, iy, iw, seed) in ((262, 318, 58, 3), (1150, 262, 40, 4), (470, 210, 26, 5)):
        spr, gx, gt = _rock_island(iw, 5, iw * 0.55, seed, (44, 54, 102), (66, 80, 132), RIM_NIGHT, 0.22,
                                   grass=False, vines=False)
        R.composite(img, spr, int(ix - gx + PAD_X), int(iy - gt + PAD_Y))
    _haze(img, sy, NIGHT_HZ - 4, 22, 20, HAZE_NIGHT, 0.45)

    # sea of cloud below the horizon, tops lit by the moon
    for k, (base, amp, col, top_col) in enumerate((
            (NIGHT_HZ + 22, 22, (104, 122, 176), (182, 198, 232)),
            (NIGHT_HZ + 70, 30, (78, 94, 150), (160, 178, 222)),
            (NIGHT_HZ + 138, 36, (56, 68, 118), (132, 150, 204)))):
        bil = _profile(W, 38 + k * 14, 80 + k, 4)
        bil = 1 - (1 - bil) ** 2
        top = base - bil * amp
        rs = _rows(H, base - amp - 2, H)
        depth = sy[rs] - top[None, :]
        cov = np.clip(depth + 0.5, 0, 1)
        shade = np.clip(depth / (amp * 1.4), 0, 1)
        ccol = R.gradient_stops(shade, [(0, top_col), (0.35, col), (1, _mix(col, (30, 36, 70), 0.3))])
        R.paint(img[rs], cov, ccol)
        _haze(img, sy, base - amp * 0.4, 12, 26, HAZE_NIGHT, 0.18)
    _vignette(img, sx, sy, W - 2 * PAD_X, H - 2 * PAD_Y, 0.55, (4, 6, 18))
    return np.clip(img, 0, 1)


def _rock_island(w: float, top_h: float, depth: float, seed: int, ink, fog_col, rim_col, rim_a: float,
                 grass: bool = True, vines: bool = True, flat: bool = False, pad: int = 0):
    """A floating rock island: flat-ish top, jagged underside tapering to points, hanging vines.
    Returns (rgba, gx, gt): the island's top-left surface point inside the sprite."""
    rng = np.random.default_rng(seed)
    w = int(w)
    vine_len = depth * 0.9 if vines else 0
    GX, GT = 14 + pad, 12 + pad
    W, H = w + 2 * GX, int(GT + top_h + depth + vine_len + 16)
    xx, yy = R.grid(W, H)
    lx, ly = xx - GX, yy - GT
    col_x = np.arange(W, dtype=np.float32) + 0.5 - GX
    t = np.clip(col_x / w, 0, 1)
    inside = (col_x > 0) & (col_x < w)
    if flat:
        top = np.zeros(W, np.float32)
        top = top + np.where(inside, 0, 50)
        rnd = np.clip(np.minimum(col_x, w - col_x), 0, 6)
        top = top + (6 - rnd) ** 2 / 6 * inside      # round the corners off below the surface
    else:
        bump = _profile(W, max(8, w / 6), seed + 1, 3)
        top = (bump - 0.5) * 3 + (1 - np.clip(np.sin(np.pi * t), 0, 1) ** 0.3) * 8
        top = np.where(inside, top, 50)
    env = np.clip(np.sin(np.pi * t), 0, 1) ** 0.7 * inside
    jag = _profile(W, max(5, w / 16), seed + 2, 5, sharp=0.9)
    spike = _profile(W, max(4, w / 40), seed + 4, 3, sharp=0.95) ** 2
    big = _profile(W, max(20, w / 3), seed + 3, 3)
    thick = top_h * (0.4 + 0.6 * np.clip(np.sin(np.pi * t), 0, 1) ** 0.35) if flat else top_h
    bottom = thick + env * depth * (0.28 + 0.4 * big + 0.3 * jag ** 1.5 + 0.22 * spike)
    cov = np.clip(ly - top[None, :] + 0.5, 0, 1) * np.clip(bottom[None, :] - ly + 0.5, 0, 1) * inside[None, :]
    img = R.blank(W, H)
    dz = np.clip(ly / max(1, top_h + depth), 0, 1)
    col = R.gradient_stops(dz, [(0, _mix(ink, rim_col, 0.12)), (0.08, ink), (0.6, _mix(ink, fog_col, 0.25)),
                                (1, _mix(ink, fog_col, 0.55))])
    R.paint(img, cov, col)
    # strata: crisp near the rim, ghosting out with depth; cool moonlight grazing the right end
    near = np.exp(-np.maximum(ly - top[None, :], 0) / 12.0) * cov
    tex = R.fbm(xx * 0.4, yy * 2.2, 10, 3, seed=seed)
    R.paint(img, np.clip((tex - 0.5) * 1.8, 0, 1) * near * 0.35 * rim_a, rim_col, op="screen")
    strata = np.clip((R.fbm(xx * 0.15, yy * 1.4, 14, 3, seed=seed + 9) - 0.6) * 4, 0, 1)
    R.paint(img, strata * cov * np.exp(-np.maximum(ly, 0) / 70.0) * 0.12 * rim_a, rim_col, op="screen")
    endl = np.exp(-np.maximum(w - lx, 0) / (8 + w * 0.04)) * cov * np.exp(-np.maximum(ly, 0) / 60.0)
    R.paint(img, endl * 0.22 * rim_a, rim_col, op="screen")
    if flat:
        # rock facets: ridged noise gives crack lines; the moon side catches a little light
        rn = R.fbm(xx * 0.5, yy * 0.22, 26, 4, seed=seed + 21)
        crack = (1 - np.abs(rn * 2 - 1)) ** 10
        fade = np.exp(-np.maximum(ly - 8, 0) / 90.0) * cov * (ly > 6)
        R.paint(img, crack * fade * 0.55, _mix(ink, (0, 0, 0), 0.6))
        facet = np.clip((R.fbm(xx * 0.3, yy * 0.15, 40, 3, seed=seed + 5) - 0.5) * 3, 0, 1)
        R.paint(img, facet * fade * (0.3 + 0.7 * t[None, :]) * 0.10, rim_col, op="screen")
    # rim light along the top, brighter toward the moon side (right)
    side = 0.55 + 0.45 * t[None, :]
    dt = ly - top[None, :]
    lip = np.clip(1.7 - np.abs(dt - 0.6), 0, 1) * inside[None, :] * side
    R.paint(img, lip * rim_a, _mix(rim_col, (255, 255, 255), 0.3))
    R.paint(img, np.exp(-np.maximum(dt, 0) / 4.0) * cov * 0.3 * rim_a * side, rim_col, op="screen")
    R.paint(img, np.exp(np.minimum(dt, 0) / 3.5) * (dt < 0) * inside[None, :] * 0.28 * rim_a * side, rim_col,
            op="add")
    mask = np.zeros((H, W), np.float32)
    segs, leaves = [], []
    if grass:
        for _ in range(int(w / 3)):
            cx = rng.uniform(0.02, 0.98) * w
            ci = int(np.clip(cx + GX, 0, W - 1))
            if flat and 10 < cx < w - 10:
                continue
            h = rng.uniform(2, 7) * (1.4 if cx < 0.1 * w or cx > 0.9 * w else 1)
            pts, _ = _curve(rng, cx + GX, top[ci] + GT + 1.5, -math.pi / 2 + rng.normal(0.15, 0.3), h, 2, 0.1)
            segs += _stroke(pts, 1.6, 0.4)
    if vines:
        for _ in range(max(2, int(w / 22))):
            cx = rng.uniform(0.08, 0.92) * w
            ci = int(np.clip(cx + GX, 0, W - 1))
            y0 = bottom[ci] - 4
            if y0 < top_h + 4:
                continue
            L = rng.uniform(0.25, 1.0) * vine_len
            pts, _ = _curve(rng, cx + GX, y0 + GT, math.pi / 2 + rng.normal(0, 0.12), L, 8, 0.12)
            segs += _stroke(pts, 1.8, 0.6)
            for j, (px, py) in enumerate(pts[1:]):
                s = 1 if j % 2 else -1
                leaves.append((px + s * 2.2, py, 1.5 + rng.random() * 1.0))
    _capsules(mask, segs)
    _blobs(mask, leaves)
    R.paint(img, mask, _mix(ink, fog_col, 0.15))
    return np.clip(img, 0, 1), GX, GT


def _night_mid(W: int, H: int) -> np.ndarray:
    """Floating rock islands with hanging vines, kept to the sides of the playfield."""
    img = R.blank(W, H)
    for (ix, iy, iw, depth, seed, ink, fog) in (
            (60, 236, 150, 90, 11, (30, 38, 72), (90, 108, 160)),
            (1150, 392, 120, 76, 13, (26, 32, 64), (86, 104, 156)),
            (368, 116, 54, 38, 15, (52, 64, 110), (110, 128, 178)),
            (1196, 120, 70, 46, 17, (44, 56, 100), (104, 122, 172))):
        spr, gx, gt = _rock_island(iw, 10, depth, seed, ink, fog, RIM_NIGHT, 0.85)
        R.composite(img, spr, int(ix - gx + PAD_X), int(iy - gt + PAD_Y))
    return img


def _night_fg(W: int, H: int) -> np.ndarray:
    """A blossoming branch reaching in from the top-left corner."""
    rng = np.random.default_rng(77)
    bw, bh = 330, 190
    mask = np.zeros((bh, bw), np.float32)
    bloom = np.zeros((bh, bw), np.float32)
    segs, flowers = [], []

    def twig(x, y, ang, length, w0, depth):
        pts, a2 = _curve(rng, x, y, ang, length, 5, 0.12, gravity=0.02)
        segs.extend(_stroke(pts, w0, max(0.8, w0 * 0.45)))
        for (px, py) in pts[2:]:
            if rng.random() < 0.55:
                flowers.append((px + rng.normal(0, 3), py + rng.normal(0, 3), rng.uniform(1.8, 3.4)))
        if depth > 0:
            for k in range(2):
                i = rng.integers(1, len(pts) - 1)
                twig(pts[i][0], pts[i][1], a2 + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9),
                     length * rng.uniform(0.4, 0.6), w0 * 0.55, depth - 1)
        flowers.append((pts[-1][0], pts[-1][1], 3.2))

    twig(-10, 34, 0.18, 300, 11, 3)
    twig(-10, 10, 0.5, 120, 6, 1)
    _capsules(mask, segs)
    _blobs(bloom, flowers, noise_amp=0.8, seed=4)
    img = R.blank(W, H)
    sub = R.blank(bw, bh)
    R.paint(sub, mask, INK_NIGHT)
    xx, yy = R.grid(bw, bh)
    lit = np.clip(0.4 + (xx / bw) * 0.6, 0, 1)
    fcol = R.gradient_stops(lit, [(0, (150, 118, 156)), (1, (236, 206, 230))])
    R.paint(sub, bloom * 0.95, fcol)
    img[:bh, :bw] = sub
    return img


# =============================================================================
# THEME
# =============================================================================

@register
class SilhouetteTheme(Theme):
    name = "silhouette"
    title = "Silhouette"

    palette = {
        "bg": (14, 8, 16), "panel": (20, 12, 24), "panel_edge": (150, 118, 120),
        "text": IVORY, "text_dim": (170, 150, 150), "accent": (255, 196, 120),
        "accent2": (150, 210, 236), "good": (120, 224, 170), "warn": (255, 182, 92),
        "bad": (240, 92, 72), "p1": (255, 120, 88), "p2": (124, 186, 255),
    }

    def build(self):
        g = self.g
        self.fonts["ui"] = Face(g, FontStyle(weight=0.85, tracking=2.0, glow=5))
        self.fonts["title"] = Face(g, FontStyle(weight=0.62, tracking=3.3, glow=11))
        self.fonts["small"] = Face(g, FontStyle(weight=0.95, tracking=1.2, glow=0))
        self.fonts["digits"] = Face(g, FontStyle(weight=0.72, tracking=0.9, glow=9))
        self.fonts["ui"].prewarm(11, 13, 15, 18, 20)
        self.fonts["small"].prewarm(11)
        self.fonts["digits"].prewarm(32, 48, 56)

        def up(key, fn, deps):
            return g.upload(cached("sil-" + key, fn, deps))

        self.spr_glow = up("glow128", lambda: _glow_sprite(128), [_glow_sprite])
        self.spr_glow_soft = up("glow256s", lambda: _glow_sprite(256, 3.0), [_glow_sprite])
        self.spr_streak = up("streak", lambda: _streak_sprite(256, 24), [_streak_sprite])
        self.spr_ring = up("ring160", lambda: _ring_sprite(160, 2.0), [_ring_sprite])
        self.spr_glint = up("glint24", lambda: _glint_sprite(24), [_glint_sprite])
        self.spr_puff = up("puff64", lambda: _puff_sprite(64), [_puff_sprite])
        self.spr_bubble = up("bubble128", lambda: _bubble_sprite(128), [_bubble_sprite])
        self.spr_embers = [up(f"ember{s}", (lambda s=s: _ember_sprite(s)), [_ember_sprite]) for s in (8, 12, 18)]
        self.spr_eye = up("eye12", lambda: _ember_sprite(12), [_ember_sprite])
        self.spr_petals = up("petals", lambda: _petal_sheet(8, 16), [_petal_sheet])
        self.spr_birds = up("birds", lambda: _bird_sheet(4, 18), [_bird_sheet, _capsules])
        self.spr_mist = [up(f"mist{k}", (lambda k=k: _mist_sprite(900, 120, 30 + k)), [_mist_sprite])
                         for k in range(2)]
        self.spr_reeds = [up(f"reeds{k}", (lambda k=k: _reed_sprite(50 + k)), [_reed_sprite, _curve, _stroke])
                          for k in range(2)]
        self.hud_fade = [up(f"hudfade{k}", (lambda k=k: _hud_fade(540, 200, bool(k))), [_hud_fade])
                         for k in range(2)]
        self.panel_strip = up("panelstrip", _panel_strip, [_panel_strip])
        self._kind = "dusk"
        self._victory_frame = None
        rng = np.random.default_rng(5)
        # embers: (x0, speed, phase, sway, size index, layer)
        self.embers = [(rng.random() * self.w, 0.25 + rng.random() * 0.55, rng.random() * 2000,
                        8 + rng.random() * 26, int(rng.integers(0, 2)), 0) for _ in range(22)]
        self.embers += [(rng.random() * self.w, 0.7 + rng.random() * 0.6, rng.random() * 2000,
                         16 + rng.random() * 30, int(rng.integers(1, 3)), 1) for _ in range(7)]
        # petals / snow motes: (x0, speed, phase, sway, frame offset, kind)
        self.petals = [(rng.random() * (self.w + 300), 0.45 + rng.random() * 0.5, rng.random() * 2000,
                        10 + rng.random() * 26, int(rng.integers(0, 8)), int(rng.random() < 0.55))
                       for _ in range(26)]

    def reset_match(self):
        super().reset_match()
        self._victory_frame = None

    def build_stage(self, stage) -> dict:
        g = self.g
        W, H = self.w + 2 * PAD_X, self.h + 2 * PAD_Y
        kind = "night" if "final" in stage.name.lower() else "dusk"
        a = {"kind": kind, "pad": (PAD_X, PAD_Y)}
        if kind == "dusk":
            deps = [_dusk_far, _dusk_sky, _paint_ridge, _profile, _ruin, _stars, _haze, _vignette]
            far = cached(f"sil-dusk-far-{W}x{H}", lambda: _dusk_far(W, H), deps)
            mid = cached(f"sil-dusk-mid-{W}x{H}", lambda: _dusk_mid(W, H), deps + [_dusk_mid, _tree, _blobs,
                                                                                      _capsules])
            fg = cached(f"sil-dusk-fg-{W}x{H}", lambda: _dusk_fg(W, H), [_dusk_fg, _curve, _stroke, _capsules])
            a["_blur_src"] = (f"sil-dusk-blur-{W}x{H}", far, mid, fg,
                              deps + [_dusk_mid, _dusk_fg, SilhouetteTheme._blurred])
        else:
            deps = [_night_far, _paint_ridge, _profile, _stars, _haze, _vignette, _rock_island]
            far = cached(f"sil-night-far-{W}x{H}", lambda: _night_far(W, H), deps)
            mid = cached(f"sil-night-mid-{W}x{H}", lambda: _night_mid(W, H), deps + [_night_mid, _blobs, _capsules])
            fg = cached(f"sil-night-fg-{W}x{H}", lambda: _night_fg(W, H), [_night_fg, _curve, _stroke, _blobs])
        a["far"] = g.upload(far)
        a["mid"] = g.upload_cropped(mid)
        a["fg"] = g.upload_cropped(fg)
        plats = {}
        for p in [stage.main_platform] + stage.platforms:
            key = (int(p.width), int(p.height), not p.passthrough)
            if key in plats:
                continue
            w, h, solid = key
            if solid and kind == "dusk":
                fn = (lambda w=w, h=h: _dusk_slab(w, h))
                deps_p = [_dusk_slab, _profile, _curve, _stroke, _capsules]
            elif solid:
                fn = (lambda w=w, h=h: _rock_island(w, h, 150, 101, INK_NIGHT, (40, 52, 96), RIM_NIGHT, 1.0,
                                                    grass=True, vines=True, flat=True, pad=8))
                deps_p = [_rock_island, _profile, _curve, _stroke, _capsules, _blobs]
            else:
                fn = (lambda w=w, h=h: _ledge(w, h, kind))
                deps_p = [_ledge, _profile, _curve, _stroke, _capsules, _blobs]
            res = {}

            def build_img(fn=fn, res=res):
                img, gx, gt = fn()
                res["off"] = (gx, gt)
                return img
            img = cached(f"sil-plat-{kind}-{w}x{h}-{int(solid)}", build_img, deps_p)
            off = res.get("off")
            if off is None:      # loaded from cache: offsets are deterministic, recompute cheaply
                off = self._plat_offsets(kind, solid)
            plats[key] = (g.upload(img), off[0], off[1])
        a["plats"] = plats
        return a

    @staticmethod
    def _plat_offsets(kind, solid):
        if solid and kind == "dusk":
            return 26, 18
        if solid:
            return 14 + 8, 12 + 8
        return 16, 14

    @staticmethod
    def _blurred(far, mid, fg) -> np.ndarray:
        img = far.astype(np.float32) / 255.0
        for layer in (mid, fg):
            R.composite(img, layer.astype(np.float32) / 255.0, 0, 0)
        img = R.blur(img, 9)
        img[..., :3] *= 0.62
        img[..., 3] = 1.0
        return img

    # ------------------------------------------------------------------ colors & light

    def character_color(self, fighter) -> RGB:
        ct = fighter.stats.character_type
        if ct in CHAR_COLORS:
            return CHAR_COLORS[ct]
        return DEFAULT_COLORS[fighter.player_id % 2]

    def color_for_name(self, name: Optional[str], idx: int = 0) -> RGB:
        return CHAR_COLORS.get(name or "", DEFAULT_COLORS[idx % 2])

    def _light(self):
        return (MOON_X, MOON_Y) if self._kind == "night" else (SUN_X, SUN_Y)

    def _ink(self):
        return INK_NIGHT if self._kind == "night" else INK_DUSK

    def _rim(self):
        return RIM_NIGHT if self._kind == "night" else RIM_DUSK

    # ------------------------------------------------------------------ stage

    def draw_background(self, game_state, stage, camera, shake):
        a = self.stage_assets(stage)
        self._kind = a["kind"]
        g = self.g
        cx, cy = camera
        f = self.fx.frame
        g.blit(a["far"], -PAD_X - cx * 0.12 + shake[0] * 0.2, -PAD_Y - cy * 0.12 + shake[1] * 0.2, op=X.OP_SRC)
        if self._kind == "dusk":
            self._birds(f, cx * 0.2)
            self._embers(f, 0, cx * 0.3)
        mid, mx, my = a["mid"]
        g.blit(mid, mx - PAD_X - cx * 0.4 + shake[0] * 0.5, my - PAD_Y - cy * 0.4 + shake[1] * 0.5)
        self._mist(f, cx * 0.6, 596 if self._kind == "dusk" else 560,
                   (214, 104, 96) if self._kind == "dusk" else (176, 196, 232), 0.30)

    def _embers(self, frame, layer, cam_x, alpha=1.0):
        g = self.g
        h = self.h + 60
        for (x0, sp, ph, sway, si, lay) in self.embers:
            if lay != layer:
                continue
            t = frame * sp + ph
            y = self.h + 30 - (t % h)
            x = (x0 + math.sin(t * 0.021) * sway + t * 0.18 - cam_x) % (self.w + 40) - 20
            life = (t % h) / h
            flick = 0.65 + 0.35 * math.sin(t * 0.37 + x0)
            a = flick * min(1.0, life * 5) * (1 - life) ** 0.6 * alpha
            col = (255, 176, 92) if si < 2 else (255, 204, 130)
            g.stamp(self.spr_embers[si], x, y, col, min(1.0, a * 1.1))

    def _birds(self, frame, cam_x):
        """A small flock crossing the dusk sky now and then, wings beating then gliding."""
        g = self.g
        span = self.w + 900
        for i, (dx, dy, ph) in enumerate(((0, 0, 0), (26, -9, 3), (44, 7, 6), (70, -4, 9), (96, 12, 2))):
            t = frame * 0.55 + i * 3 + 300
            x = self.w + 120 - (t % span) + dx - cam_x
            if x < -30 or x > self.w + 30:
                continue
            y = 178 + dy + math.sin(t * 0.02 + i) * 4 - (t % span) * 0.04
            beat = (frame // 4 + ph) % 14
            k = beat % 4 if beat < 8 else 1
            g.blit(self.spr_birds, x - 9, y - 9, 0.85, sx=k * 18, sy=0, w=18, h=18)

    def _petals(self, frame, cam_x, alpha=1.0):
        g = self.g
        h = self.h + 40
        img = self.spr_petals
        for (x0, sp, ph, sway, fo, is_petal) in self.petals:
            t = frame * sp + ph
            y = (t % h) - 20
            x = (x0 - t * 0.35 + math.sin(t * 0.03) * sway - cam_x) % (self.w + 60) - 30
            fade = min(1.0, (t % h) / 60) * alpha
            if is_petal:
                k = (int(t * 0.12) + fo) % 8
                g.blit(img, x - 8, y - 8, 0.85 * fade, sx=k * 16, sy=0, w=16, h=16)
            else:
                g.stamp(self.spr_embers[0], x, y, (220, 232, 255), 0.55 * fade)

    def _mist(self, frame, cam_x, y, col, a):
        g = self.g
        for k, spr in enumerate(self.spr_mist):
            span = self.w + spr.w
            x = (frame * (0.22 + 0.13 * k) + k * 700 - cam_x) % span - spr.w
            g.stamp(spr, x + spr.w / 2, y + k * 26, col, a * (1 - 0.25 * k), op=X.OP_OVER)

    def draw_platform(self, plat, stage, shake):
        a = self.stage_assets(stage)
        e = a["plats"].get((int(plat.width), int(plat.height), not plat.passthrough))
        if e is None:
            return super().draw_platform(plat, stage, shake)
        img, gx, gt = e
        self.g.blit(img, plat.x - gx + shake[0], plat.y - gt + shake[1])

    def draw_foreground(self, game_state, stage, camera, shake):
        a = self.stage_assets(stage)
        g = self.g
        f = self.fx.frame
        cx, cy = camera
        if self._kind == "dusk":
            self._embers(f, 1, cx * 0.9)
            self._reeds(f, cx * 0.85, cy * 0.3)
        else:
            self._mist(f + 400, cx * 1.0, 688, (150, 172, 214), 0.26)
            self._petals(f, cx * 0.7)
        fg, fx, fy = a["fg"]
        if fg is not None:
            g.blit(fg, fx - PAD_X - cx * 0.85, fy - PAD_Y - cy * 0.3 + shake[1] * 0.4)
        for ko in self.fx.ko_flashes:
            t = (f - ko["frame"]) / 40.0
            col = self.character_color(game_state.fighters[ko["pid"]])
            g.fill_rect(0, 0, self.w, self.h, _mix(col, (255, 230, 200), 0.5), 0.22 * (1 - t) ** 2, op=X.OP_ADD)

    def _reeds(self, frame, cam_x, cam_y):
        g = self.g
        for k, (x, img) in enumerate(((54, self.spr_reeds[0]), (self.w - 46, self.spr_reeds[1]))):
            sway = math.sin(frame * 0.021 + k * 2.1) * 0.035 + math.sin(frame * 0.053 + k) * 0.012
            g.blit_xform(img, x - cam_x, self.h + 6 - cam_y, 1.0, 1.0, sway, anchor=(0.5, 1.0))

    # ------------------------------------------------------------------ fighters

    def _body(self, sk, grow: float = 0.0, dx: float = 0.0, dy: float = 0.0, s: float = 1.0) -> Shape:
        """The silhouette: tapered limbs, a solid torso, a filled head and little foot wedges."""
        sh = Shape()
        f = sk.facing
        hx0, hy0 = sk.hip
        hip = (hx0 + dx, hy0 + dy)
        neck = (sk.neck[0] + dx, sk.neck[1] + dy)
        g2 = grow
        for knee, foot in ((sk.back_knee, sk.back_foot), (sk.front_knee, sk.front_foot)):
            kx, ky = knee[0] + dx, knee[1] + dy
            fx, fy = foot[0] + dx, foot[1] + dy
            sh.segment(hip[0], hip[1], kx, ky, (9.6 + g2) * s, (8.0 + g2) * s)
            sh.segment(kx, ky, fx, fy, (8.0 + g2) * s, (5.0 + g2) * s)
            sh.segment(fx, fy, fx + f * 7.0 * s, fy + 0.6 * s, (5.0 + g2) * s, (2.4 + g2) * s)
        for elbow, hand in ((sk.back_elbow, sk.back_hand), (sk.front_elbow, sk.front_hand)):
            ex, ey = elbow[0] + dx, elbow[1] + dy
            hx, hy = hand[0] + dx, hand[1] + dy
            sh.segment(neck[0], neck[1], ex, ey, (8.4 + g2) * s, (6.6 + g2) * s)
            sh.segment(ex, ey, hx, hy, (6.6 + g2) * s, (3.4 + g2) * s)
            sh.circle(hx, hy, (2.9 + g2 * 0.5) * s)
        sh.segment(neck[0], neck[1], hip[0], hip[1], (12.5 + g2) * s, (9.6 + g2) * s)
        hdx, hdy = sk.head[0] + dx, sk.head[1] + dy
        sh.segment(neck[0], neck[1], hdx, hdy, (5.2 + g2) * s, (5.2 + g2) * s)
        sh.circle(hdx, hdy, sk.head_r + g2 * 0.5 * s)
        return sh

    def _light_dir(self, x, y, light=None):
        lx, ly = light or self._light()
        dx, dy = lx - x, ly - y
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return 0.0, -1.0
        return dx / d, dy / d

    def _silhouette(self, sk, rim_col, body_col, a: float, s: float = 1.0, glow_a: float = 0.2,
                    light=None, eye: bool = True, eye_col=None):
        g = self.g
        cx, cy = sk.center
        ux, uy = self._light_dir(cx, cy, light)
        if glow_a > 0:
            g.stamp(self.spr_glow, cx + ux * 6 * s, cy - 6 * s + uy * 6 * s, rim_col, glow_a * a,
                    scale_x=1.15 * s, scale_y=1.55 * s)
        off = 2.6 * s
        g.fill_shape(self._body(sk, 1.6, ux * off, uy * off, s), rim_col, a)
        g.fill_shape(self._body(sk, 0.0, 0.0, 0.0, s), body_col, a)
        if eye:
            hx, hy = sk.head
            g.stamp(self.spr_eye, hx + sk.facing * sk.head_r * 0.45, hy - 1.5 * s, eye_col or rim_col, a)

    def draw_fighter_aura(self, fighter, sk, look: FighterLook):
        if look.armor:
            cx, cy = sk.center
            pulse = 0.6 + 0.4 * math.sin(self.fx.frame * 0.35)
            self.g.stamp(self.spr_glow_soft, cx, cy - 8, (255, 186, 70), 0.6 * pulse, scale_x=0.85, scale_y=1.05)

    def draw_fighter(self, fighter, sk, look: FighterLook):
        a = 0.4 if look.blink else 1.0
        col = (255, 200, 90) if look.armor else look.color
        rim = _mix(col, (255, 255, 255), 0.28)
        body = self._ink()
        if look.flash > 0:
            rim = _mix(rim, (255, 255, 255), look.flash)
            body = _mix(body, (255, 244, 230), look.flash * 0.75)
        if look.armor:
            pulse = 0.65 + 0.35 * math.sin(self.fx.frame * 0.35)
            self.g.fill_shape(self._body(sk, 7.0), (255, 190, 80), 0.22 * pulse * a, op=X.OP_ADD)
        self._silhouette(sk, rim, body, a, glow_a=0.22, eye_col=_mix(col, (255, 255, 255), 0.6))
        if look.shield_stun:
            hx, hy = sk.head
            f = self.fx.frame
            for k in range(3):
                ang = f * 0.15 + k * 2.09
                self.g.stamp(self.spr_glint, hx + math.cos(ang) * 18, hy - 16 + math.sin(ang) * 5,
                             (255, 226, 150), 0.9)

    def draw_afterimage(self, fighter, sk, dodge_type, alpha: float):
        col = _mix(DODGE_COLORS.get(dodge_type, (170, 190, 240)), (150, 186, 240), 0.55)
        self.g.fill_shape(self._body(sk, 0.0), col, alpha * 0.42)

    def draw_knockback_trail(self, fighter, pts):
        col = self.character_color(fighter)
        n = len(pts)
        s_glow, s_core = Shape(), Shape()
        for i in range(n - 1):
            t0, t1 = i / n, (i + 1) / n
            s_glow.segment(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 4 + 30 * t0, 4 + 30 * t1)
            s_core.segment(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 1 + 7 * t0, 1 + 7 * t1)
        self.g.fill_shape(s_glow, col, 0.24, op=X.OP_ADD)
        self.g.fill_shape(s_core, _mix(col, (255, 255, 255), 0.6), 0.85, op=X.OP_ADD)

    def draw_shield(self, fighter, sk):
        hp = max(0.0, fighter.shield_health) / 100.0
        r = 26 + 26 * hp
        base = _mix(self.character_color(fighter), (255, 255, 255), 0.45)
        col = base if hp > 0.5 else (255, 196, 110) if hp > 0.25 else (255, 100, 80)
        cx, cy = sk.center
        cy -= 4
        flick = 0.9 + 0.1 * math.sin(self.fx.frame * 0.7)
        sc = (r * 2 + 12) / self.spr_bubble.w
        self.g.stamp(self.spr_glow_soft, cx, cy, col, 0.12, scale_x=sc * 0.75)
        self.g.stamp(self.spr_bubble, cx, cy, col, 0.6 * flick, scale_x=sc)

    def draw_counter(self, fighter, sk):
        cx, cy = sk.center
        f = self.fx.frame
        for k in range(3):
            ph = ((f * 0.035) + k / 3.0) % 1.0
            rr = 28 + 40 * ph
            self.g.stamp(self.spr_ring, cx, cy, (170, 206, 255), 0.75 * (1 - ph) ** 1.3,
                         scale_x=rr * 2 / (self.spr_ring.w - 12))
        self.g.stamp(self.spr_glow, cx, cy, (140, 180, 255), 0.18, scale_x=0.8)

    def draw_reflector(self, fighter, sk):
        cx, cy = sk.center
        f = self.fx.frame
        rot = f * 0.05
        pts = [(cx + math.cos(i * math.pi / 3 + rot) * 50, cy + math.sin(i * math.pi / 3 + rot) * 50)
               for i in range(6)]
        col = (150, 236, 255)
        self.g.polygon(pts, col, 0.07, op=X.OP_ADD)
        self.g.polyline(pts, 7, col, 0.14, closed=True, op=X.OP_ADD)
        self.g.polyline(pts, 1.6, (226, 250, 255), 0.95, closed=True, op=X.OP_ADD)
        for i, (px, py) in enumerate(pts):
            tw = 0.6 + 0.4 * math.sin(f * 0.3 + i * 1.3)
            self.g.stamp(self.spr_glint, px, py, (220, 248, 255), tw)

    # ------------------------------------------------------------------ projectiles & effects

    def draw_projectile(self, p, trail, shake):
        g = self.g
        col = tuple(int(c) for c in p.color)
        lite = _mix(col, (255, 250, 240), 0.55)
        x, y = p.position.x + shake[0], p.position.y + shake[1]
        n = len(trail)
        if n > 1:
            s = Shape()
            for i in range(n - 1):
                t0, t1 = i / n, (i + 1) / n
                s.segment(trail[i][0] + shake[0], trail[i][1] + shake[1], trail[i + 1][0] + shake[0],
                          trail[i + 1][1] + shake[1], p.height * 0.9 * t0, p.height * 0.9 * t1)
            g.fill_shape(s, col, 0.28, op=X.OP_ADD)
        if p.width > p.height * 1.8:
            g.stamp(self.spr_glow_soft, x, y, col, 0.35, scale_x=p.width * 2.0 / 256, scale_y=p.height * 4 / 256)
            g.stamp(self.spr_streak, x, y, col, 0.95, scale_x=p.width * 1.9 / 256, scale_y=p.height * 1.6 / 24)
            g.stamp(self.spr_streak, x, y, (255, 252, 244), 0.95, scale_x=p.width * 1.4 / 256,
                    scale_y=p.height * 0.55 / 24)
        else:
            pulse = 1 + 0.1 * math.sin(self.fx.frame * 0.5)
            g.stamp(self.spr_glow_soft, x, y, col, 0.45, scale_x=p.width * 3.2 * pulse / 256)
            g.stamp(self.spr_glow, x, y, lite, 0.85, scale_x=p.width * 1.25 / 128)
            g.circle(x, y, p.width * 0.2, (255, 252, 244))

    _TIER = {(255, 255, 255): (255, 238, 210), (255, 255, 100): (255, 212, 110),
             (255, 180, 80): (255, 150, 62), (255, 100, 100): (240, 70, 52)}

    def draw_effect(self, e, shake):
        from renderer import EffectType
        g = self.g
        x, y = e.x + shake[0], e.y + shake[1]
        t = e.progress
        if e.effect_type == EffectType.HIT_SPARK:
            col = self._TIER.get(tuple(e.color), tuple(e.color))
            k = 0.65 + 0.45 * min(1.6, e.intensity)
            fade = (1 - t) ** 1.4
            g.stamp(self.spr_glow_soft, x, y, col, 0.55 * fade, scale_x=(0.35 + 0.5 * t) * k)
            g.stamp(self.spr_glow, x, y, (255, 246, 226), fade, scale_x=(0.45 + 0.3 * t) * k)
            g.stamp(self.spr_streak, x, y, _mix(col, (255, 255, 255), 0.5), 0.9 * fade,
                    scale_x=(0.8 + 1.6 * t) * k, scale_y=0.7 * (1 - t * 0.5))
            base = math.radians(e.angle)
            seed = int(e.x * 7 + e.y * 13) & 1023
            s = Shape()
            for i in range(9):
                h1 = ((seed * 31 + i * 97) % 101) / 101.0
                h2 = ((seed * 17 + i * 53) % 89) / 89.0
                ang = base + (h1 - 0.5) * 2.6 if i < 6 else h1 * 6.283
                v = (40 + 80 * h2) * k
                r = v * _ease_out(t)
                px = x + math.cos(ang) * r
                py = y - math.sin(ang) * r + 34 * t * t
                tl = (6 + 10 * h2) * (1 - t) + 1
                s.segment(px - math.cos(ang) * tl, py + math.sin(ang) * tl, px, py, 0.6, 3.4 * (1 - t) + 0.9)
            g.fill_shape(s, _mix(col, (255, 255, 255), 0.4), 1 - t * t, op=X.OP_ADD)
        elif e.effect_type == EffectType.LANDING_DUST:
            fy = y + 40
            dust = (150, 176, 220) if self._kind == "night" else (236, 150, 104)
            for side in (-1, 1):
                for j, sc in enumerate((1.0, 0.7)):
                    tt = min(1.0, t * (1.0 + 0.4 * j))
                    px = x + side * (8 + (40 if j == 0 else 22) * _ease_out(tt))
                    py = fy - 7 - (9 + 6 * j) * tt
                    sz = (0.3 + 0.5 * tt) * sc
                    g.stamp(self.spr_puff, px, py, dust, 0.5 * (1 - tt) ** 1.3, scale_x=sz * 1.4, scale_y=sz)
            g.stamp(self.spr_streak, x, fy, self._rim(), 0.55 * (1 - t), scale_x=0.25 + 0.3 * t, scale_y=0.3)
        elif e.effect_type == EffectType.REFLECTOR_HEX:
            r = 24 + 50 * _ease_out(t)
            pts = [(x + math.cos(i * math.pi / 3 + 0.5) * r, y + math.sin(i * math.pi / 3 + 0.5) * r)
                   for i in range(6)]
            g.polyline(pts, 6, (150, 236, 255), 0.18 * (1 - t), closed=True, op=X.OP_ADD)
            g.polyline(pts, 1.8, (226, 250, 255), 1 - t, closed=True, op=X.OP_ADD)
            g.stamp(self.spr_glow, x, y, (200, 246, 255), 0.8 * (1 - t), scale_x=0.6)
            g.stamp(self.spr_streak, x, y, (220, 248, 255), 0.7 * (1 - t), scale_x=0.9, scale_y=0.5)
        elif e.effect_type == EffectType.FRAME_ADVANTAGE:
            adv = int(e.intensity)
            label = f"+{adv}" if adv > 0 else str(adv)
            col = _mix(tuple(e.color), IVORY, 0.35)
            self.text(x, y - 30 - 20 * t, label, 20, col, align="center", alpha=1 - t * t,
                      glow=tuple(e.color), glow_alpha=0.7 * (1 - t))

    # ------------------------------------------------------------------ HUD

    def _damage_color(self, dmg: float) -> RGB:
        stops = [(0, IVORY), (40, (255, 216, 150)), (80, (255, 156, 84)), (120, (240, 82, 58)),
                 (180, (196, 34, 44))]
        for (d0, c0), (d1, c1) in zip(stops, stops[1:]):
            if dmg <= d1:
                return _mix(c0, c1, (dmg - d0) / (d1 - d0))
        return stops[-1][1]

    def _diamond(self, x, y, r, col, a=1.0, op=X.OP_OVER):
        self.g.polygon([(x, y - r), (x + r * 0.8, y), (x, y + r), (x - r * 0.8, y)], col, a, op)

    def draw_hud(self, game_state):
        g = self.g
        f = self.fx.frame
        w, h = self.w, self.h
        if not game_state.match_over:
            self._victory_frame = None
        g.blit(self.hud_fade[0], 0, h - self.hud_fade[0].h)
        g.blit(self.hud_fade[1], w - self.hud_fade[1].w, h - self.hud_fade[1].h)
        for i, fighter in enumerate(game_state.fighters[:2]):
            col = self.character_color(fighter)
            right = i == 1
            sgn = -1 if right else 1
            x0 = w - 46 if right else 46
            al = "right" if right else "left"
            name_col = _mix(col, IVORY, 0.45)
            name = fighter.stats.name.upper()
            nw = self.text(x0, h - 104, name, 13, name_col, align=al, tracking=3.4)
            # stocks: small diamonds after the name
            for s in range(max(0, fighter.stocks)):
                sx = x0 + sgn * (nw + 18 + s * 14)
                self._diamond(sx, h - 109, 4.2, col, 0.95)
            g.fill_rect(x0 if not right else x0 - 34, h - 94, 34, 1, col, 0.8)
            # damage with a pop
            dmg = fighter.damage_percent
            anim = game_state.damage_anim_frames[i] if i < len(game_state.damage_anim_frames) else 0
            k = min(1.0, anim / 20.0)
            size = 56 if k > 0.4 else 48
            dcol = self._damage_color(dmg)
            if k > 0:
                dcol = _mix(dcol, (255, 255, 255), 0.5 * k)
            jit = math.sin(f * 2.1) * 2.5 * k
            digits = f"{int(dmg)}"
            base_y = h - 40
            pct_w = self.measure("%", 20)
            if right:
                self.text(x0, base_y, "%", 20, dcol, align="right", alpha=0.8)
                self.face("digits").draw(digits, x0 - pct_w - 6 + jit, base_y, size, dcol, align="right",
                                         glow_color=dcol, glow_alpha=0.35 + 0.5 * k)
            else:
                dw = self.face("digits").draw(digits, x0 + jit, base_y, size, dcol, align="left",
                                              glow_color=dcol, glow_alpha=0.35 + 0.5 * k)
                self.text(x0 + dw + 6, base_y, "%", 20, dcol, alpha=0.8)
            # hairline shield meter
            hp = max(0.0, fighter.shield_health) / 100.0
            bw = 120
            bx = x0 - bw if right else x0
            g.fill_rect(bx, h - 26, bw, 1, IVORY, 0.18)
            if hp > 0:
                sc = _mix(col, IVORY, 0.5) if hp > 0.5 else (255, 190, 100) if hp > 0.25 else (255, 96, 76)
                fw = bw * hp
                g.fill_rect(bx + (bw - fw if right else 0), h - 26.5, fw, 2, sc, 0.9)
            combo = game_state.combo_counts[i] if i < len(game_state.combo_counts) else 0
            if combo >= 2:
                self.text(x0, h - 136, f"{combo} HIT COMBO", 13, _mix(col, (255, 255, 255), 0.65), align=al,
                          glow=col, glow_alpha=0.9, tracking=3.4)
        t = game_state.get_time_display()
        if t:
            low = game_state.time_remaining < 600
            col = (255, 110, 80) if low and (f // 15) % 2 == 0 else IVORY
            tw = self.face("digits").draw(t, w / 2, 54, 32, col, align="center", glow_color=col, glow_alpha=0.4)
            for sgn in (-1, 1):
                g.stamp(self.spr_streak, w / 2 + sgn * (tw / 2 + 52), 38, col, 0.55, scale_x=0.32, scale_y=0.12)
                self._diamond(w / 2 + sgn * (tw / 2 + 16), 38, 2.6, col, 0.7)

    def draw_victory(self, game_state):
        g = self.g
        f = self.fx.frame
        w, h = self.w, self.h
        if self._victory_frame is None:
            self._victory_frame = f
        t = f - self._victory_frame
        e = _ease_out(t / 32.0)
        g.fill_rect(0, 0, w, h, (6, 3, 10), 0.4 * e)
        bar = 92 * e
        g.fill_rect(0, 0, w, bar, (2, 1, 4))
        g.fill_rect(0, h - bar, w, bar, (2, 1, 4))
        if game_state.winner is not None:
            fighter = game_state.fighters[game_state.winner]
            col = self.character_color(fighter)
            txt = fighter.stats.name.upper()
            cap = "VICTORY"
        else:
            col, txt, cap = IVORY, "DRAW", "NO CONTEST"
        a = _ease_out((t - 10) / 26.0)
        if a <= 0:
            return
        cy = h / 2
        g.stamp(self.spr_glow_soft, w / 2, cy, col, 0.22 * a, scale_x=3.6, scale_y=0.9)
        g.stamp(self.spr_streak, w / 2, cy + 4, _mix(col, (255, 255, 255), 0.5), 0.6 * a,
                scale_x=w / 256 * (0.4 + 0.6 * a), scale_y=0.35)
        g.stamp(self.spr_glint, w / 2 + (a - 0.5) * 420, cy - 2, (255, 250, 240), 0.9 * a, scale_x=2.4, scale_y=1.2)
        g.stamp(self.spr_glow_soft, w / 2, cy - 10, (6, 3, 10), 0.55 * a, scale_x=2.6, scale_y=0.75, op=X.OP_OVER)
        self._halo_text(w / 2, cy, txt, 64, role="title", a=0.5 * a)
        self.text(w / 2, cy - 58, cap, 15, _mix(col, IVORY, 0.6), align="center", valign="middle", alpha=a,
                  tracking=6)
        self.text(w / 2, cy, txt, 64, _mix(col, (255, 255, 255), 0.6), role="title", align="center",
                  valign="middle", glow=col, glow_alpha=0.9 * a, alpha=a)
        g.fill_rect(w / 2 - 140 * a, cy + 50, 280 * a, 1, col, 0.6 * a)
        self._diamond(w / 2, cy + 50.5, 4, col, a)

    # ------------------------------------------------------------------ UI kit

    def ui_backdrop(self, frame: int, variant: str = "menu"):
        from stages import battlefield
        a = self.stage_assets(battlefield())
        self._kind = "dusk"
        g = self.g
        sway = math.sin(frame * 0.003) * 14
        if variant == "menu":
            g.blit(a["far"], -PAD_X + sway * 0.3, -PAD_Y, op=X.OP_SRC)
            self._embers(frame, 0, -sway)
            mid, mx, my = a["mid"]
            g.blit(mid, mx - PAD_X + sway * 0.7, my - PAD_Y)
            self._mist(frame, -sway, 596, (214, 104, 96), 0.3)
            fg, fx, fy = a["fg"]
            g.blit(fg, fx - PAD_X + sway, fy - PAD_Y)
            self._reeds(frame, -sway, 0)
            self._embers(frame, 1, -sway)
            g.stamp(self.spr_glow_soft, self.w / 2, self.h * 0.64, (12, 4, 14), 0.62, scale_x=2.2, scale_y=2.2,
                    op=X.OP_OVER)
        else:
            if "blur" not in a:      # built on first use: only menus need it
                key, far, mid, fg, deps = a.pop("_blur_src")
                a["blur"] = g.upload(cached(key, lambda: self._blurred(far, mid, fg), deps))
            g.blit(a["blur"], -PAD_X + sway * 0.3, -PAD_Y, op=X.OP_SRC)
            self._embers(frame, 0, -sway, alpha=0.6)

    def ui_panel(self, x, y, w, h, emphasis: float = 0.0, accent: Optional[RGB] = None):
        g = self.g
        strip = self.panel_strip
        g.blit_xform(strip, x + w / 2, y + h / 2, w / strip.w, h / strip.h)
        edge = accent or self.palette["panel_edge"]
        ea = 0.22 + 0.4 * emphasis
        g.fill_rect(x, y, w, 1, edge, ea)
        g.fill_rect(x, y + h - 1, w, 1, edge, ea * 0.7)
        g.fill_rect(x, y + 1, 1, h - 2, edge, ea * 0.7)
        g.fill_rect(x + w - 1, y + 1, 1, h - 2, edge, ea * 0.7)
        if emphasis > 0:
            L = min(16, h / 3, w / 4)
            for (cx, cy, sx, sy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
                g.fill_rect(cx if sx > 0 else cx - L, cy - (0 if sy > 0 else 2) - 0.5, L, 2, edge, emphasis)
                g.fill_rect(cx - (0 if sx > 0 else 2), cy if sy > 0 else cy - L, 2, L, edge, emphasis)
            g.stamp(self.spr_streak, x + w / 2, y, edge, 0.5 * emphasis, scale_x=w / 256 * 0.9, scale_y=0.35)

    def _halo_text(self, x, y, s, size, role="ui", align="center", a=0.55):
        """A soft dark halo under text so thin letters stay legible over bright sky."""
        self.face(role).draw(str(s), x, y, size, (8, 4, 10), 0.01, align, "middle", glow_color=(8, 4, 10),
                             glow_alpha=a, glow_op=X.OP_OVER)

    def ui_button(self, x, y, w, h, label, selected: bool, frame: int, accent: Optional[RGB] = None,
                  sublabel: Optional[str] = None, size: int = 20):
        g = self.g
        acc = accent or self.palette["accent"]
        cx = x + w / 2
        has_sub = bool(sublabel) and selected
        cy = y + h * 0.36 if has_sub else y + h / 2
        if selected:
            pulse = 0.8 + 0.2 * math.sin(frame * 0.12)
            g.stamp(self.spr_glow_soft, cx, y + h / 2, (10, 4, 12), 0.5, scale_x=w / 256 * 1.3,
                    scale_y=h / 256 * 1.8, op=X.OP_OVER)
            g.stamp(self.spr_glow_soft, cx, cy, acc, 0.10 * pulse, scale_x=w / 256 * 1.2, scale_y=h / 256 * 1.6)
            lw = self.measure(label, size)
            self._halo_text(cx, cy, label, size, a=0.5)
            self.text(cx, cy, label, size, (255, 250, 240), align="center", valign="middle", glow=acc,
                      glow_alpha=0.75 * pulse)
            for sgn in (-1, 1):
                self._diamond(cx + sgn * (lw / 2 + 22), cy, 4.5, acc, pulse)
            if has_sub:
                sub_size = max(12, int(size * 0.58))
                self._halo_text(cx, y + h * 0.78, sublabel, sub_size, a=0.6)
                self.text(cx, y + h * 0.78, sublabel, sub_size, _mix(self.palette["text_dim"], IVORY, 0.4),
                          align="center", valign="middle")
            else:
                uy = min(y + h - 4, cy + size * 0.5 + 8)
                g.fill_rect(cx - lw / 2 - 10, uy, lw + 20, 1, acc, 0.75)
                g.stamp(self.spr_streak, cx, uy + 0.5, acc, 0.8, scale_x=(lw + 90) / 256 * 1.6, scale_y=0.2)
        else:
            self._halo_text(cx, cy, label, size, a=0.6)
            self.text(cx, cy, label, size, _mix(self.palette["text_dim"], IVORY, 0.45), align="center",
                      valign="middle", alpha=0.92)

    def ui_title(self, x, y, s, size=48, color=None, frame: int = 0, align="center"):
        col = color or self.palette["accent"]
        face = self.face("title")
        main = _mix(col, (255, 250, 240), 0.55)
        face.draw(s, x, y, size, (8, 4, 10), 0.01, align, "middle", glow_color=(10, 4, 14), glow_alpha=0.35,
                  glow_op=X.OP_OVER)
        width = face.draw(s, x, y, size, main, align=align, valign="middle", glow_color=col, glow_alpha=0.85)
        left = x - width / 2 if align == "center" else x - width if align == "right" else x
        # slow light sweep: brighten the glyphs under a moving band
        at = face.atlas(size)
        span = width + 500
        p = (frame * 2.4) % (span + 400) - 250
        pen = left
        for ch in s:
            gl = (at.glyphs.get(ch) or at.glyphs["?"])[0]
            mid = pen + gl.adv / 2
            boost = math.exp(-((mid - left - p) / 70.0) ** 2)
            if boost > 0.04 and ch != " ":
                face.draw(ch, pen, y, size, (255, 255, 255), alpha=0.55 * boost, valign="middle", op=X.OP_ADD)
            pen += gl.adv + at.tracking_px
        if size <= 52 and align == "center":
            for sgn in (-1, 1):
                self.g.stamp(self.spr_streak, x + sgn * (width / 2 + 70), y, col, 0.6, scale_x=0.38, scale_y=0.1)
                self._diamond(x + sgn * (width / 2 + 26), y, 3.2, col, 0.8)

    def ui_bar(self, x, y, w, h, frac: float, color: RGB, bg_alpha: float = 0.35):
        g = self.g
        cy = y + h / 2
        g.fill_rect(x, cy - 0.5, w, 1, IVORY, 0.12 + bg_alpha * 0.3)
        if frac > 0:
            fw = w * min(1.0, frac)
            th = max(2.0, min(3.0, h * 0.35))
            g.fill_rect(x, cy - th / 2, fw, th, color, 0.95)
            g.stamp(self.spr_glow, x + fw, cy, color, 0.35, scale_x=0.22)
            self._diamond(x + fw, cy, max(3.0, h * 0.42), _mix(color, (255, 255, 255), 0.4))

    def ui_toggle(self, x, y, on: bool, frame: int):
        g = self.g
        w, h = 46, 22
        cy = y + h / 2
        col = self.palette["good"] if on else self.palette["text_dim"]
        g.fill_rect(x + 4, cy - 0.5, w - 8, 1, col, 0.55 if on else 0.3)
        kx = x + w - 9 if on else x + 9
        if on:
            g.stamp(self.spr_glow, kx, cy, col, 0.45, scale_x=0.3)
            self._diamond(kx, cy, 7, col)
        else:
            pts = [(kx, cy - 7), (kx + 5.6, cy), (kx, cy + 7), (kx - 5.6, cy)]
            g.polyline(pts, 1.3, col, 0.8, closed=True)

    def ui_arrow(self, x, y, direction: int, color: RGB, size: float = 10, alpha: float = 1.0):
        d = direction
        pts = [(x - d * size * 0.35, y - size), (x + d * size * 0.45, y), (x - d * size * 0.35, y + size)]
        self.g.polyline(pts, max(1.4, size * 0.2), color, alpha)

    def ui_burst(self, x, y, text: str, t: float, color: Optional[RGB] = None, size: int = 120):
        g = self.g
        col = color or self.palette["accent"]
        pop = 1.0 + max(0.0, 0.2 - t) * 2.0
        alpha = 1.0 if t < 0.7 else max(0.0, 1 - (t - 0.7) / 0.3)
        self.face("title").draw(text, x, y, int(size * pop), (8, 4, 10), 0.01, "center", "middle",
                                glow_color=(8, 4, 10), glow_alpha=0.5 * alpha, glow_op=X.OP_OVER)
        g.stamp(self.spr_glow_soft, x, y, col, 0.25 * alpha, scale_x=3.0, scale_y=1.1)
        g.stamp(self.spr_streak, x, y, _mix(col, (255, 255, 255), 0.5), 0.7 * alpha,
                scale_x=self.w / 256 * min(1.0, 0.3 + t * 4), scale_y=0.4)
        self.text(x, y, text, int(size * pop), _mix(col, (255, 252, 244), 0.6), role="title", align="center",
                  valign="middle", alpha=alpha, glow=col, glow_alpha=alpha)

    def ui_stage_thumb(self, stage_name: str, x, y, w, h, frame: int, selected: bool = False):
        from stages import get_stage
        stage = get_stage(stage_name)
        a = self.stage_assets(stage)
        g = self.g
        kx, ky = w / self.w, h / self.h
        far = a["far"]
        g.blit_xform(far, x + w / 2, y + h / 2, kx, ky)
        for key in ("mid", "fg"):
            img, mx, my = a[key]
            if img is not None:
                g.blit_xform(img, x + (mx - PAD_X + img.w / 2) * kx, y + (my - PAD_Y + img.h / 2) * ky, kx, ky)
        for p in [stage.main_platform] + stage.platforms:
            e = a["plats"].get((int(p.width), int(p.height), not p.passthrough))
            if e is None:
                continue
            img, gx, gt = e
            g.blit_xform(img, x + (p.x - gx + img.w / 2) * kx, y + (p.y - gt + img.h / 2) * ky, kx, ky)
        edge = self.palette["accent"] if selected else self.palette["panel_edge"]
        ea = 0.9 if selected else 0.35
        g.fill_rect(x, y, w, 1, edge, ea)
        g.fill_rect(x, y + h - 1, w, 1, edge, ea)
        g.fill_rect(x, y, 1, h, edge, ea)
        g.fill_rect(x + w - 1, y, 1, h, edge, ea)

    def draw_figure_static(self, sk, color: RGB, scale: float = 1.0):
        rim = _mix(color, (255, 255, 255), 0.28)
        self._silhouette(sk, rim, INK_DUSK, 1.0, s=scale, glow_a=0.28, light=(SUN_X, SUN_Y),
                         eye_col=_mix(color, (255, 255, 255), 0.6))
