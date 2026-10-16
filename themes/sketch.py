"""
Sketchbook theme: the whole game is drawn in a school notebook.

Battlefield  -> ruled notebook page: pencil mountains and hills, drifting puffy clouds,
                a doodled sun, margin doodles, coffee ring, tape; torn legal-pad strips
                taped on as the soft platforms.
Final Dest.  -> graph-paper page in blue ballpoint: a floating island with a hatched rocky
                underside, ringed planet, rocket with a loop-the-loop trail, star doodles.

Fighters are ink stick figures with pressure-tapered, boiling strokes (re-jittered every
few frames, like hand-drawn animation), colored in with a highlighter that misses the lines.
Everything static is generated once with numpy (pencil = coverage x paper tooth, hatching =
oriented line fields clipped to regions) and disk-cached; per-frame work is sprite blits plus
a couple of tessellated shapes per fighter.
"""
import hashlib
import inspect
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from gfx import raster as R
from gfx import xrender as X
from gfx.cache import cached
from gfx.context import Shape
from gfx.font import glyph_strokes
from gfx.text import Face, FontStyle, _render_glyph

from . import register
from .base import FighterLook, Theme

RGB = Tuple[int, int, int]

PAD_X, PAD_Y = 24, 16            # margin baked around the page for screen shake

PAPER = (247, 242, 229)
PAPER_HI = (253, 251, 245)
INK = (26, 30, 52)
GRAPHITE = (78, 80, 90)
GRAPHITE_TXT = (96, 98, 108)
BALLPOINT = (34, 58, 156)
RED_PEN = (204, 38, 44)
GREEN_PEN = (34, 132, 64)
RULE_BLUE = (120, 162, 218)
GRID_BLUE = (112, 160, 214)
MARGIN_RED = (226, 92, 100)
COFFEE = (166, 112, 62)
HILITE = (255, 232, 60)

CHAR_COLORS = {
    "RUSH": (255, 66, 96),
    "ZONE": (36, 168, 255),
    "GRAB": (255, 146, 24),
    "FLEX": (64, 206, 72),
}
DEFAULT_COLORS = [(255, 84, 168), (56, 150, 255)]


def _mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _on_paper(c, paper=PAPER):
    """Color a translucent marker shows on paper (multiply), for drawing with plain OVER."""
    return tuple(int(paper[i] * c[i] / 255) for i in range(3))


def _hilite(c) -> RGB:
    """Highlighter tone for a color: dark inks get lightened so labels stay readable on top."""
    luma = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
    return _on_paper(_mix(c, (255, 255, 255), 0.62 if luma < 120 else 0.4 if luma < 170 else 0.15))


def _stable(obj) -> str:
    """Process-independent short hash (Python's hash() of str is randomized per run)."""
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]


def _h32(*vals) -> int:
    h = 2166136261
    for v in vals:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return h


# =============================================================================
# RASTER HELPERS (asset generation only)
# =============================================================================

def _resample(pts, step: float) -> np.ndarray:
    p = np.asarray(pts, np.float64)
    if len(p) < 2:
        return p
    seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    if L < 1e-6:
        return p[:1]
    n = max(2, int(L / step) + 1)
    t = np.linspace(0, L, n)
    return np.stack([np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])], 1)


def _wobble(pts, amp: float, seed: int, step: float = 3.0, wl: float = 40.0, over: float = 0.0) -> np.ndarray:
    """Resample a polyline and push it around with smooth noise: a hand-drawn line."""
    p = _resample(pts, step)
    n = len(p)
    if n < 2:
        return p
    rng = np.random.default_rng(seed)
    d = np.gradient(p, axis=0)
    norm = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    nx, ny = -d[:, 1] / norm, d[:, 0] / norm
    s = np.arange(n) * step
    off = np.zeros(n)
    for w, a in ((wl, 1.0), (wl * 0.37, 0.45), (wl * 0.13, 0.18)):
        off += a * np.sin(s * 2 * math.pi / w + rng.random() * 6.28)
    off *= amp / 1.6
    q = p + np.stack([nx * off, ny * off], 1)
    if over > 0:
        t0 = q[0] - q[min(2, n - 1)]
        t1 = q[-1] - q[max(0, n - 3)]
        t0 /= max(1e-9, np.hypot(*t0))
        t1 /= max(1e-9, np.hypot(*t1))
        q = np.vstack([q[0] + t0 * over * (0.4 + 0.8 * rng.random()), q,
                       q[-1] + t1 * over * (0.6 + 0.8 * rng.random())])
    return q


def _stroke(mask, pts, w0: float, w1: Optional[float] = None, taper: float = 0.0, alpha: float = 1.0,
            x0: float = 0.0, y0: float = 0.0):
    """Max-accumulate a (tapered) polyline into a coverage mask whose pixel (0,0) is at (x0, y0)."""
    p = np.asarray(pts, np.float64)
    n = len(p)
    if n < 2:
        return
    w1 = w0 if w1 is None else w1
    seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = cum / max(cum[-1], 1e-9)
    w = w0 + (w1 - w0) * t
    if taper > 0:
        w = w * (1 - taper * np.abs(2 * t - 1) ** 3)
    H, W = mask.shape
    for i in range(n - 1):
        ax, ay = p[i]
        bx, by = p[i + 1]
        r = max(w[i], w[i + 1]) / 2 + 1.5
        xa = int(max(0, math.floor(min(ax, bx) - r - x0)))
        xb = int(min(W, math.ceil(max(ax, bx) + r - x0)))
        ya = int(max(0, math.floor(min(ay, by) - r - y0)))
        yb = int(min(H, math.ceil(max(ay, by) + r - y0)))
        if xa >= xb or ya >= yb:
            continue
        xx, yy = R.grid(xb - xa, yb - ya, xa + x0, ya + y0)
        d = R.sd_tapered(xx, yy, ax, ay, bx, by, max(0.3, w[i] / 2), max(0.3, w[i + 1] / 2))
        win = mask[ya:yb, xa:xb]
        np.maximum(win, R.coverage(d) * alpha, out=win)


def _loop_pts(cx, cy, rx, ry, seed, turns=1.12, jit=0.05, n=0, a0=None):
    """A hand-drawn circle: slightly more than one turn so the pen overlaps where it started."""
    rng = np.random.default_rng(seed)
    n = n or max(16, int((rx + ry) * turns * 0.9))
    a0 = rng.random() * 6.28 if a0 is None else a0
    a = a0 + np.linspace(0, turns * 2 * math.pi, n)
    ph = rng.random(2) * 6.28
    k = 1 + jit * np.sin(2 * a + ph[0]) + jit * 0.5 * np.sin(3 * a + ph[1])
    k = k * (1 + (rng.random() - 0.5) * jit * (a - a0) / (2 * math.pi))   # spiral drift
    return np.stack([cx + np.cos(a) * rx * k, cy + np.sin(a) * ry * k], 1)


def _hatch(xx, yy, ang: float, spacing: float, width: float, seed: int, length: float = 46.0,
           gap: float = 0.1, wob: float = 0.5) -> np.ndarray:
    """Oriented field of short pencil strokes (each with soft ends and its own pressure)."""
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    u = xx * ca + yy * sa
    v = -xx * sa + yy * ca
    k = np.floor(u / spacing).astype(np.int64)
    zero = np.zeros_like(k)
    off = (R._hash2(k, zero, seed) - 0.5) * spacing * 0.3
    center = (k + 0.5) * spacing + off + np.sin(v * 0.045 + k * 1.3) * wob
    d = np.abs(u - center)
    sh = R._hash2(k, zero + 1, seed) * length
    sv = (v + sh) / length
    j = np.floor(sv).astype(np.int64)
    fr = sv - j
    keep = R._hash2(k, j + 7, seed) > gap
    inten = 0.6 + 0.4 * R._hash2(k, j + 13, seed)
    prof = np.clip(np.minimum(fr, 1 - fr) * length / 7.0, 0, 1)
    w = width * (0.6 + 0.4 * prof)
    return np.clip(w / 2 - d + 0.5, 0, 1) * keep * inten * prof


_MEMO: Dict[tuple, np.ndarray] = {}


def _tooth(W, H, seed):
    """Paper tooth (grain that breaks up pencil strokes). Memoized; treat the result as read-only."""
    key = ("tooth", W, H, seed)
    t = _MEMO.get(key)
    if t is None:
        xx, yy = R.grid(W, H)
        t = R.value_noise(xx, yy, 1.15, seed) * 0.7 + R.value_noise(xx, yy, 2.7, seed + 1) * 0.3
        t = _MEMO[key] = np.clip(0.42 + t * 0.85, 0, 1).astype(np.float32)
    return t


def _hatch_box(shape, box, *args, **kw) -> np.ndarray:
    """_hatch evaluated only inside box=(x0, y0, x1, y1); zeros elsewhere."""
    H, W = shape
    x0, y0 = int(max(0, box[0])), int(max(0, box[1]))
    x1, y1 = int(min(W, box[2])), int(min(H, box[3]))
    out = np.zeros(shape, np.float32)
    if x0 < x1 and y0 < y1:
        xx, yy = R.grid(x1 - x0, y1 - y0, x0, y0)
        out[y0:y1, x0:x1] = _hatch(xx, yy, *args, **kw)
    return out


def _curve(mask, ys, width: float, alpha: float = 1.0, keep=None, seed: int = 0, wob: float = 0.0):
    """Stroke a per-column curve y(x) in one vectorized pass (long ridge/contour lines)."""
    H, W = mask.shape
    ys = np.asarray(ys, np.float32)
    if wob > 0:
        ys = ys + (R.fbm1d(W, 30, 3, seed=seed) - 0.5) * 2 * wob
    slope = np.gradient(ys)
    y0 = int(max(0, np.floor(ys.min() - width - 3)))
    y1 = int(min(H, np.ceil(ys.max() + width + 4)))
    if y0 >= y1:
        return
    yy = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
    d = np.abs(yy - ys[None, :]) / np.sqrt(1 + slope ** 2)[None, :]
    ww = width * (0.85 + 0.3 * R.fbm1d(W, 50, 2, seed=seed + 1))[None, :]
    cov = np.clip(ww / 2 - d + 0.5, 0, 1) * alpha
    if keep is not None:
        cov = cov * keep[None, :]
    win = mask[y0:y1]
    np.maximum(win, cov, out=win)


def _mul(rgb, cov, color, a: float = 1.0):
    """Multiply-blend a colored mark onto an opaque RGB canvas (pencil/marker on paper)."""
    c = R.rgb01(color)
    rgb *= 1 - (np.clip(cov, 0, 1) * a)[..., None] * (1 - c)


def _over(rgb, cov, color, a: float = 1.0):
    m = (np.clip(cov, 0, 1) * a)[..., None]
    rgb[:] = rgb * (1 - m) + R.rgb01(color) * m


def _np_text(mask, s: str, x: float, y: float, size: float, style: FontStyle, seed: int,
             alpha: float = 1.0) -> float:
    """Rasterize stroke-font text into a mask (baseline at y). Returns the advance width."""
    rng = np.random.default_rng(seed)
    sk = size / 10.0
    pen = x
    H, W = mask.shape
    for ch in s:
        adv, strokes, _ = glyph_strokes(ch)
        if strokes:
            m, ox, oy = _render_glyph(strokes, size, style, 2, rng)
            px, py = int(round(pen + ox)), int(round(y + oy))
            h, w = m.shape
            xa, ya = max(0, px), max(0, py)
            xb, yb = min(W, px + w), min(H, py + h)
            if xa < xb and ya < yb:
                win = mask[ya:yb, xa:xb]
                np.maximum(win, m[ya - py:yb - py, xa - px:xb - px] * alpha, out=win)
        pen += adv * sk + style.tracking * sk
    return pen - x - style.tracking * sk


def _text_width(s: str, size: float, style: FontStyle) -> float:
    sk = size / 10.0
    return sum(glyph_strokes(ch)[0] * sk + style.tracking * sk for ch in s) - style.tracking * sk


PENCIL_HAND = FontStyle(weight=1.3, tracking=1.6, jitter=0.35, variants=1, taper=0.5, seed=4)
BALL_HAND = FontStyle(weight=1.15, tracking=2.2, jitter=0.2, variants=1, taper=0.25, seed=9)
DIGIT_HAND = FontStyle(weight=2.3, tracking=0.9, jitter=0.28, variants=1, taper=0.25, slant=0.06, seed=3)
HEAVY_HAND = FontStyle(weight=2.5, tracking=1.3, jitter=0.3, variants=1, taper=0.3, slant=0.08, seed=13)
TITLE_HAND = FontStyle(weight=1.9, tracking=1.8, jitter=0.42, variants=1, taper=0.55, seed=21)
TITLE_BIG = FontStyle(weight=1.35, tracking=1.8, jitter=0.36, variants=1, taper=0.5, seed=22)


def _paper(W, H, seed, base=PAPER) -> np.ndarray:
    """Warm paper: tone blotches, tooth grain, faint fibers, vignette. Returns opaque RGB float (a copy)."""
    key = ("paper", W, H, seed, base)
    if key in _MEMO:
        return _MEMO[key].copy()
    xx, yy = R.grid(W, H)
    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = R.rgb01(base)
    k = 4
    lx, ly = R.grid((W + k - 1) // k, (H + k - 1) // k)
    low = R.fbm(lx * k, ly * k, 230, 4, seed=seed)
    blot = np.repeat(np.repeat(low, k, 0), k, 1)[:H, :W]
    blot = R.box_blur(R.box_blur(blot, 2, 0), 2, 1)
    grain = R.value_noise(xx, yy, 1.4, seed + 5)
    rgb *= (0.968 + 0.045 * blot + 0.022 * (grain - 0.5))[..., None]
    fib = np.zeros((H, W), np.float32)
    rng = np.random.default_rng(seed + 11)
    for _ in range(int(W * H / 7000)):
        x, y = rng.random() * W, rng.random() * H
        L = 6 + rng.random() * 24
        a = rng.random() * math.pi
        pts = [(x, y), (x + math.cos(a) * L * 0.5 + rng.normal() * 2, y + math.sin(a) * L * 0.5 + rng.normal() * 2),
               (x + math.cos(a) * L, y + math.sin(a) * L)]
        _stroke(fib, _resample(pts, 3), 0.7, alpha=0.3 + 0.5 * rng.random())
    rgb *= (1 - fib * 0.07)[..., None]
    v = np.sqrt(((xx - W / 2) / (W * 0.6)) ** 2 + ((yy - H / 2) / (H * 0.6)) ** 2)
    vig = np.clip(v - 0.62, 0, 1) ** 1.5 * 0.85
    rgb *= (1 - vig[..., None] * (1 - R.rgb01((200, 176, 138))))
    _MEMO[key] = rgb
    return rgb.copy()


# ---------------------------------------------------------------- doodle primitives (numpy)

def _star_pts(cx, cy, r, rot=0.0, points=5, inner=0.45):
    out = []
    for i in range(points * 2 + 1):
        a = rot - math.pi / 2 + i * math.pi / points
        rr = r if i % 2 == 0 else r * inner
        out.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    return out


def _doodle_star(mask, cx, cy, r, seed, w=1.5, fill=None, rot=None):
    rng = np.random.default_rng(seed)
    rot = rng.random() * 0.6 - 0.3 if rot is None else rot
    pts = _star_pts(cx, cy, r, rot)
    _stroke(mask, _wobble(pts, r * 0.05, seed, step=2, wl=r * 1.5, over=r * 0.15), w, taper=0.3)
    if fill is not None:
        # scribble fill: zigzag across the star, clipped
        H, W = fill.shape
        x0, y0 = int(max(0, cx - r - 2)), int(max(0, cy - r - 2))
        x1, y1 = int(min(W, cx + r + 3)), int(min(H, cy + r + 3))
        if x0 < x1 and y0 < y1:
            xx, yy = R.grid(x1 - x0, y1 - y0, x0, y0)
            inside = R.coverage(R.sd_polygon(xx, yy, pts[:-1]) + 0.5)
            h = _hatch(xx, yy, 35, 2.2, 1.3, seed, length=r * 3, gap=0.0, wob=0.3)
            win = fill[y0:y1, x0:x1]
            np.maximum(win, h * inside, out=win)


def _doodle_spiral(mask, cx, cy, r, seed, turns=3.2, w=1.3):
    a = np.linspace(0, turns * 2 * math.pi, int(turns * 30))
    rr = np.linspace(1.5, r, len(a))
    pts = np.stack([cx + np.cos(a) * rr, cy + np.sin(a) * rr * 0.9], 1)
    _stroke(mask, _wobble(pts, 0.6, seed, step=2, wl=18), w, taper=0.4)


def _doodle_arrow(mask, x0, y0, x1, y1, seed, w=1.6, bend=14):
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1
    nx, ny = -dy / L, dx / L
    pts = [(x0 + dx * t + nx * bend * math.sin(math.pi * t), y0 + dy * t + ny * bend * math.sin(math.pi * t))
           for t in np.linspace(0, 1, 16)]
    _stroke(mask, _wobble(pts, 0.7, seed, step=3, wl=30), w, taper=0.3)
    ex, ey = pts[-1]
    px, py = pts[-3]
    a = math.atan2(ey - py, ex - px)
    for s in (-1, 1):
        aa = a + math.pi + s * 0.5
        _stroke(mask, [(ex, ey), (ex + math.cos(aa) * 11, ey + math.sin(aa) * 11)], w * 1.1, w * 0.6)


def _doodle_smiley(mask, cx, cy, r, seed, w=1.5):
    _stroke(mask, _wobble(_loop_pts(cx, cy, r, r, seed), 0.4, seed, step=2, wl=20), w, taper=0.5)
    for s in (-1, 1):
        _stroke(mask, [(cx + s * r * 0.35, cy - r * 0.3), (cx + s * r * 0.35, cy - r * 0.12)], w * 1.3)
    a = np.linspace(0.2 * math.pi, 0.8 * math.pi, 10)
    _stroke(mask, np.stack([cx + np.cos(a) * r * 0.55, cy + np.sin(a) * r * 0.5], 1), w, taper=0.4)


def _doodle_tape(rgb, cx, cy, w, h, ang, seed, tint=(250, 246, 222)):
    """Translucent tape strip with jagged torn ends, multiplied onto an opaque canvas."""
    H, W = rgb.shape[:2]
    rr = int(math.hypot(w, h) / 2 + 6)
    x0, y0 = int(max(0, cx - rr)), int(max(0, cy - rr))
    x1, y1 = int(min(W, cx + rr)), int(min(H, cy + rr))
    if x0 >= x1 or y0 >= y1:
        return
    xx, yy = R.grid(x1 - x0, y1 - y0, x0, y0)
    c, s = math.cos(ang), math.sin(ang)
    u = (xx - cx) * c + (yy - cy) * s
    v = -(xx - cx) * s + (yy - cy) * c
    jag = (R._hash2(np.floor(v / 2.2).astype(np.int64), (u > 0).astype(np.int64), seed) - 0.5) * 3.5
    inside = np.clip(w / 2 + jag - np.abs(u) + 0.5, 0, 1) * np.clip(h / 2 - np.abs(v) + 0.5, 0, 1)
    sub = rgb[y0:y1, x0:x1]
    sub[:] = sub * (1 - inside[..., None] * 0.42) + R.rgb01(tint) * inside[..., None] * 0.42
    sheen = inside * np.exp(-((v + h * 0.18) / (h * 0.12)) ** 2) * 0.22
    sub[:] = sub * (1 - sheen[..., None]) + sheen[..., None]
    edge = inside * np.clip(1.3 - np.abs(np.abs(v) - h / 2), 0, 1)
    _mul(sub, edge, (150, 140, 110), 0.35)


def _pine(mask, x, y, h, seed, w=1.4):
    _stroke(mask, _wobble([(x, y), (x, y - h * 0.3)], 0.3, seed, step=2), w)
    tiers = 3
    for i in range(tiers):
        ty = y - h * 0.22 - i * h * 0.24
        tw = h * (0.34 - i * 0.07)
        tip = ty - h * 0.36
        pts = [(x - tw, ty), (x, tip), (x + tw, ty), (x - tw * 0.9, ty + 0.5)]
        _stroke(mask, _wobble(pts, 0.5, seed + i, step=2, wl=12), w, taper=0.2)


def _lollipop(mask, fill, x, y, h, seed, w=1.4):
    _stroke(mask, _wobble([(x, y), (x + 1, y - h * 0.55)], 0.4, seed, step=2), w)
    r = h * 0.28
    cy = y - h * 0.55 - r * 0.8
    _stroke(mask, _wobble(_loop_pts(x, cy, r, r * 0.9, seed), 0.6, seed, step=2, wl=14), w, taper=0.4)
    a = np.linspace(0, 5 * 2 * math.pi, 120)
    rr = r * (0.2 + 0.75 * np.abs(np.sin(a * 0.21)))
    pts = np.stack([x + np.cos(a) * rr, cy + np.sin(a) * rr * 0.9], 1)
    _stroke(fill, _wobble(pts, 0.8, seed + 1, step=2, wl=10), 1.8, alpha=0.8)


def _rgba_from_layers(W, H, layers, tooth=None) -> np.ndarray:
    """Composite (mask, color, alpha, pencil?) layers over transparent -> premultiplied RGBA."""
    img = R.blank(W, H)
    for mask, color, a, pencil in layers:
        m = mask * (tooth if (pencil and tooth is not None) else 1.0)
        R.paint(img, m, color, a=a)
    return img


# =============================================================================
# PAGES
# =============================================================================

def _ruled(rgb, W, H, top=112, sp=34, margin_x=104, holes=True, seed=0):
    ox, oy = PAD_X, PAD_Y
    xx, yy = R.grid(W, H)
    rel = yy - (oy + top)
    d = np.abs(rel - np.round(rel / sp) * sp)
    lines = np.clip(1.25 - d, 0, 1) * (rel > -sp / 2)
    dens = (0.72 + 0.4 * R.fbm1d(W, 80, 3, seed=seed + 3))[None, :]
    _mul(rgb, lines * dens, RULE_BLUE, 0.62)
    mx = ox + margin_x
    m = np.clip(1.3 - np.abs(xx - mx), 0, 1) + np.clip(0.9 - np.abs(xx - mx - 4.5), 0, 1) * 0.35
    _mul(rgb, m * (0.8 + 0.2 * R.fbm1d(H, 60, 3, seed=seed + 4)[:, None]), MARGIN_RED, 0.8)
    if holes:
        for hy in (150, 360, 570):
            cx, cy, r = ox + 44, oy + hy, 14.5
            x0, y0 = int(cx - r - 6), int(cy - r - 6)
            sx, sy = R.grid(int(2 * r + 12), int(2 * r + 12), x0, y0)
            dd = np.hypot(sx - cx, sy - cy)
            inside = R.coverage(dd - r)
            shade = np.clip((sy - (cy - r)) / (2 * r), 0, 1)
            desk = R.gradient_stops(shade, [(0, (92, 80, 70)), (0.5, (150, 132, 112)), (1, (164, 146, 124))])
            sub = rgb[y0:y0 + sx.shape[0], x0:x0 + sx.shape[1]]
            sub[:] = sub * (1 - inside[..., None]) + desk * inside[..., None]
            rim = np.exp(-((dd - r - 1.2) / 1.1) ** 2) * (sy > cy - 2)
            _mul(sub, rim, (150, 140, 120), 0.45)


def _bf_far(W: int, H: int) -> np.ndarray:
    """Battlefield page: ruled paper + pencil landscape + margin doodles (opaque)."""
    ox, oy = PAD_X, PAD_Y
    rgb = _paper(W, H, 3)
    _ruled(rgb, W, H, seed=3)
    xx, yy = R.grid(W, H)
    tooth = _tooth(W, H, 21)
    SH = (H, W)
    G = np.zeros((H, W), np.float32)      # graphite strokes
    S = np.zeros((H, W), np.float32)      # soft graphite tone
    Y = np.zeros((H, W), np.float32)      # yellow colored pencil
    O = np.zeros((H, W), np.float32)      # orange colored pencil
    GR = np.zeros((H, W), np.float32)     # green colored pencil
    B = np.zeros((H, W), np.float32)      # blue ballpoint
    RP = np.zeros((H, W), np.float32)     # red pen
    cols = np.arange(W, dtype=np.float32)
    xs = cols - ox

    # --- distant mountains: asymmetric peaks with angular ridges, shaded left of each crest line
    peaks = [(40, 330, 120, 150), (238, 262, 150, 190), (452, 344, 110, 120), (642, 236, 170, 200),
             (868, 300, 150, 140), (1060, 254, 170, 160), (1250, 318, 150, 120)]
    base = 540.0
    env = np.full(W, 1e9, np.float32)
    owner = np.zeros(W, np.int64)
    for i, (ax, ay, hl, hr) in enumerate(peaks):
        hw = np.where(xs < ax, hl, hr)
        v = ay + np.abs(xs - ax) * (base - ay) / hw
        better = v < env
        owner[better] = i
        env = np.minimum(env, v)
    rng = np.random.default_rng(4)
    kx = np.arange(-40, W + 40, 22, dtype=np.float32)
    ky = rng.normal(0, 5.5, len(kx))
    rough = np.interp(cols, kx, ky) + (R.fbm1d(W, 14, 2, seed=4) - 0.5) * 4
    for (ax, ay, hl, hr) in peaks:      # keep apexes crisp
        rough *= 1 - np.exp(-((xs - ax) / 10) ** 2)
    ridge = env + rough + oy
    below = R.below_curve(W, H, ridge)
    pk = np.array(peaks, np.float32)
    apex_x = pk[owner, 0] + ox
    apex_y = pk[owner, 1] + oy
    crest = apex_x[None, :] - np.clip(yy - apex_y[None, :], 0, None) * 0.42 + np.sin(yy / 17.0) * 3
    snow_y = apex_y + 28 + 5 * np.abs(((xs / 9.0) % 2) - 1)
    depth = yy - ridge[None, :]
    shade = below * (xx < crest) * np.clip(1 - (yy - apex_y[None, :]) / 230, 0, 1) * (yy > snow_y[None, :])
    G_m = _hatch_box(SH, (0, oy + 200, W, oy + 566), 62, 3.8, 1.1, 7, length=34, gap=0.04) * np.clip(shade * 2.2, 0, 1)
    G_m = np.maximum(G_m, _hatch_box(SH, (0, oy + 200, W, oy + 566), -30, 5.5, 0.9, 8, length=26) *
                     R.smoothstep(0.45, 0.75, shade) * (depth < 70) * 0.6)
    lit = below * (xx >= crest) * R.smoothstep(150, 230, yy - apex_y[None, :])
    G_m = np.maximum(G_m, _hatch_box(SH, (0, oy + 200, W, oy + 566), 75, 7.0, 0.8, 9, length=18, gap=0.5) * lit * 0.4)
    _curve(G_m, ridge, 1.6, 0.9, seed=5, wob=0.5)
    for i, (ax, ay, hl, hr) in enumerate(peaks):
        L = (base - ay) * 0.62
        cpts = [(ax + ox - k * 0.42 * L / 6 + math.sin(k * 1.7 + i) * 2.5, ay + oy + 3 + k * L / 6) for k in range(7)]
        _stroke(G_m, _wobble(cpts, 0.8, 30 + i, step=3, wl=22), 1.3, taper=0.6, alpha=0.75)
        zz = []
        for j, x in enumerate(np.arange(ax - 60, ax + 61, 6)):
            col = int(x + ox)
            if 0 <= col < W:
                yv = snow_y[col]
                if yv > ridge[col] + 2:
                    zz.append((x + ox, yv))
        if len(zz) > 2:
            _stroke(G_m, zz, 1.0, alpha=0.65)

    # --- rolling hills in front (erase the mountains behind them)
    hill = 482 + 24 * np.sin(xs / 150 + 0.6) + 15 * np.sin(xs / 71 + 2.1) + (R.fbm1d(W, 120, 3, seed=9) - 0.5) * 18
    hill_bg = hill + oy
    hmask = R.below_curve(W, H, hill_bg)
    G_m *= (1 - hmask)
    G = np.maximum(G, G_m * 0.7)
    _curve(G, hill_bg, 1.9, 0.9, seed=12, wob=0.8)
    hill2 = 560 + 22 * np.sin(xs / 120 + 2.6) + (R.fbm1d(W, 90, 3, seed=19) - 0.5) * 14 + oy
    side = (np.abs(xs - 640) > 330).astype(np.float32)
    h2 = R.below_curve(W, H, hill2) * side[None, :]
    G *= (1 - h2 * 0.9)
    _curve(G, hill2, 1.8, 0.85, keep=side, seed=13, wob=0.7)
    S = np.maximum(S, hmask * np.clip((yy - hill_bg[None, :]) / 260, 0, 1) * 0.5)
    S = np.maximum(S, h2 * np.clip((yy - hill2[None, :]) / 200, 0, 1) * 0.6)
    rng = np.random.default_rng(41)
    for _ in range(70):
        x = rng.random() * W
        col = int(x)
        y = hill_bg[col] + 4 + rng.random() * 50
        hh = 3 + rng.random() * 3
        _stroke(G, [(x - 2, y), (x - 1, y - hh)], 0.9, alpha=0.6)
        _stroke(G, [(x + 1, y), (x + 2.5, y - hh * 0.8)], 0.9, alpha=0.6)
    for i, (tx, kind, th) in enumerate([(142, "pine", 60), (188, "lolli", 50), (230, "pine", 42),
                                        (1062, "lolli", 44), (1112, "pine", 58), (1166, "pine", 46),
                                        (1222, "lolli", 54)]):
        x = tx + ox
        y = (hill2[int(x)] if side[int(x)] else hill_bg[int(x)]) + 10
        if kind == "pine":
            _pine(G, x, y, th, 50 + i)
            # green scribble inside the pine
            pts = []
            for k in range(14):
                yy0 = y - th * 0.25 - k * th * 0.045
                half = th * 0.3 * (1 - k / 15)
                pts += [(x - half, yy0), (x + half, yy0 - 1.5)]
            _stroke(GR, _wobble(pts, 0.5, 70 + i, step=2, wl=9), 1.6, alpha=0.75)
        else:
            _lollipop(G, GR, x, y, th, 60 + i)

    # --- sun with spiral scribble fill, closed happy eyes and a smile (rays are a rotating sprite)
    sx, sy, sr = 1162 + ox, 88 + oy, 40
    a = np.linspace(0, 7 * 2 * math.pi, 420)
    rr = np.linspace(2, sr - 3, len(a))
    pts = np.stack([sx + np.cos(a) * rr, sy + np.sin(a) * rr], 1)
    _stroke(Y, _wobble(pts, 1.2, 77, step=2, wl=11), 2.6, alpha=0.95)
    crescent = np.clip((np.hypot(xx - sx, yy - sy) - sr * 0.45) / (sr * 0.5), 0, 1) * \
        R.coverage(np.hypot(xx - sx, yy - sy) - sr + 2) * ((xx - sx) + (yy - sy) * 0.6 > 4)
    O = np.maximum(O, _hatch_box(SH, (sx - sr - 2, sy - sr - 2, sx + sr + 3, sy + sr + 3), 50, 3.2, 1.4, 78,
                                 length=30, gap=0.05) * crescent)
    _stroke(G, _wobble(_loop_pts(sx, sy, sr, sr, 79), 0.6, 79, step=2, wl=24), 1.9, taper=0.5)
    for s_ in (-1, 1):
        aa = np.linspace(math.pi * 1.15, math.pi * 1.85, 8)
        _stroke(G, np.stack([sx + s_ * 13 + np.cos(aa) * 6, sy - 4 + np.sin(aa) * 5 + 5], 1), 1.7, taper=0.4)
    aa = np.linspace(0.15 * math.pi, 0.85 * math.pi, 12)
    _stroke(G, np.stack([sx + np.cos(aa) * 16, sy + 6 + np.sin(aa) * 11], 1), 1.8, taper=0.5)

    # --- header: title, underline, page number
    _np_text(G, "Stick Fight!!", 132 + ox, 74 + oy, 30, PENCIL_HAND, 5, alpha=0.9)
    _stroke(G, _wobble([(128 + ox, 86 + oy), (352 + ox, 84 + oy)], 1.0, 6, step=3, wl=60, over=6), 1.6, taper=0.4)
    _stroke(G, _wobble([(150 + ox, 92 + oy), (330 + ox, 91 + oy)], 1.0, 7, step=3, wl=60, over=6), 1.2, taper=0.5,
            alpha=0.7)
    _np_text(B, "round 1  vs.", 392 + ox, 76 + oy, 17, BALL_HAND, 8, alpha=0.9)
    _doodle_star(B, 520 + ox, 68 + oy, 9, 9, w=1.3)

    # --- margin doodles
    _doodle_spiral(G, 56 + ox, 58 + oy, 16, 11)
    _doodle_star(B, 52 + ox, 252 + oy, 14, 12, w=1.5, fill=B)
    _doodle_star(B, 70 + ox, 296 + oy, 7, 13, w=1.2)
    _doodle_smiley(G, 54 + ox, 456 + oy, 14, 14)
    _doodle_arrow(RP, 30 + ox, 515 + oy, 86 + ox, 500 + oy, 15, w=1.6, bend=-8)
    _np_text(RP, "go!", 34 + ox, 540 + oy, 14, BALL_HAND, 16)
    # paper airplane + dashed loop trail
    px, py = 236 + ox, 168 + oy
    plane = [(px, py), (px - 40, py + 6), (px - 30, py + 22), (px, py)]
    _stroke(B, _wobble(plane, 0.4, 20, step=2, wl=20, over=3), 1.5)
    _stroke(B, _wobble([(px, py), (px - 34, py + 12), (px - 28, py + 26)], 0.4, 21, step=2, wl=20), 1.3)
    t = np.linspace(0, 1, 260)
    trail = np.stack([px - 44 - 120 * t + 26 * np.sin(t * 2 * math.pi * 1.0 + 0.5) * np.sin(math.pi * t),
                      py + 16 + 40 * t - 30 * np.sin(t * math.pi * 2.0) * np.sin(math.pi * t)], 1)
    tr = _wobble(trail, 0.8, 22, step=2, wl=40)
    for i in range(0, len(tr) - 4, 7):
        _stroke(B, tr[i:i + 4], 1.2, alpha=0.8)

    # --- coffee ring on the right edge, tape on the top-left corner
    cx, cy, cr = 1198 + ox, 392 + oy, 50
    bx0, by0 = int(cx - cr - 12), int(cy - cr - 30)
    bx1, by1 = min(W, int(cx + cr + 40)), int(cy + cr + 12)
    wx, wy = R.grid(bx1 - bx0, by1 - by0, bx0, by0)
    d = np.hypot(wx - cx, wy - cy) + (R.fbm(wx, wy, 18, 3, seed=31) - 0.5) * 5
    ring = np.exp(-((d - cr) / 1.6) ** 2) * 0.75 + np.exp(-((d - cr + 3) / 4) ** 2) * 0.25 + \
        R.coverage(d - cr) * 0.08
    ring *= 0.6 + 0.5 * R.fbm(wx, wy, 40, 3, seed=32)
    d2 = np.hypot(wx - cx - 20, wy - cy + 14) + (R.fbm(wx, wy, 18, 3, seed=33) - 0.5) * 5
    ring2 = np.exp(-((d2 - cr * 0.96) / 1.4) ** 2) * 0.5 * (wx > cx - 10)
    _mul(rgb[by0:by1, bx0:bx1], np.clip(ring + ring2, 0, 1), COFFEE, 0.55)

    # --- composite the pencil layers
    _mul(rgb, S * (0.7 + 0.3 * tooth), GRAPHITE, 0.22)
    _mul(rgb, Y * tooth, (250, 200, 40), 0.9)
    _mul(rgb, O * tooth, (240, 130, 40), 0.75)
    _mul(rgb, GR * tooth, (110, 180, 80), 0.6)
    _mul(rgb, G * tooth, GRAPHITE, 0.95)
    _mul(rgb, B * (0.85 + 0.15 * tooth), BALLPOINT, 0.85)
    _mul(rgb, RP * (0.85 + 0.15 * tooth), RED_PEN, 0.85)
    _doodle_tape(rgb, 30 + ox, 10 + oy, 120, 34, -0.62, 40)
    _doodle_tape(rgb, 1250 + ox, 712 + oy, 110, 32, -0.55, 41)
    return R.opaque(np.clip(rgb, 0, 1))


def _fd_far(W: int, H: int) -> np.ndarray:
    """Final Destination page: graph paper with a blue-ballpoint space scene (opaque)."""
    ox, oy = PAD_X, PAD_Y
    rgb = _paper(W, H, 8, base=(247, 245, 237))
    xx, yy = R.grid(W, H)
    tooth = _tooth(W, H, 21)
    SH = (H, W)
    gx = xx - (ox + 200)
    gy = yy - (oy + 550)
    dmx = np.abs(gx - np.round(gx / 16) * 16)
    dmy = np.abs(gy - np.round(gy / 16) * 16)
    minor = np.maximum(np.clip(0.95 - dmx, 0, 1), np.clip(0.95 - dmy, 0, 1))
    dMx = np.abs(gx - np.round(gx / 80) * 80)
    dMy = np.abs(gy - np.round(gy / 80) * 80)
    major = np.maximum(np.clip(1.35 - dMx, 0, 1), np.clip(1.35 - dMy, 0, 1))
    _mul(rgb, minor, GRID_BLUE, 0.3)
    _mul(rgb, major, GRID_BLUE, 0.5)
    B = np.zeros((H, W), np.float32)
    BL = np.zeros((H, W), np.float32)    # lighter ballpoint (hatching)
    RP = np.zeros((H, W), np.float32)
    Y = np.zeros((H, W), np.float32)

    # spiral binding along the top
    for x in range(ox + 22, W - 10, 36):
        cx, cy = x, oy + 18
        x0, y0 = cx - 12, 0
        sx, sy = R.grid(24, cy + 12, x0, y0)
        hole = R.coverage(np.hypot((sx - cx) / 5.5, (sy - cy) / 4.5) - 1, 0.25)
        sub = rgb[y0:y0 + sx.shape[0], x0:x0 + sx.shape[1]]
        _over(sub, hole, (70, 68, 76), 1.0)
        wire = R.coverage(R.sd_segment(sx, sy, cx - 1, cy + 1, cx + 5, -4, 1.9))
        shine = R.coverage(R.sd_segment(sx, sy, cx - 0.2, cy, cx + 4.6, -4, 0.6))
        _over(sub, wire, (120, 122, 132), 1.0)
        _over(sub, shine, (220, 222, 230), 0.8)

    # header box
    _np_text(B, "FINAL DESTINATION", 46 + ox, 78 + oy, 18, BALL_HAND, 3)
    w = _text_width("FINAL DESTINATION", 18, BALL_HAND)
    box = [(36 + ox, 50 + oy), (56 + w + ox, 49 + oy), (57 + w + ox, 89 + oy), (35 + ox, 90 + oy), (36 + ox, 48 + oy)]
    _stroke(B, _wobble(box, 0.6, 4, step=3, wl=40, over=5), 1.4)
    _np_text(B, "p.42", 1200 + ox, 78 + oy, 15, BALL_HAND, 5, alpha=0.8)

    # ringed planet
    pcx, pcy, pr = 1060 + ox, 178 + oy, 62
    dd = np.hypot(xx - pcx, yy - pcy)
    body = R.coverage(dd - pr)
    ang = math.radians(-16)
    ux = (xx - pcx) * math.cos(ang) + (yy - pcy) * math.sin(ang)
    uy = -(xx - pcx) * math.sin(ang) + (yy - pcy) * math.cos(ang)
    lit = np.clip(((xx - pcx) * 0.7 - (yy - pcy) * 0.7) / pr, -1, 1)
    shade_p = body * np.clip((0.25 - lit) * 1.4, 0, 1)
    pbox = (pcx - pr * 2.4, pcy - pr * 1.4, pcx + pr * 2.4, pcy + pr * 1.4)
    BL = np.maximum(BL, _hatch_box(SH, pbox, 35, 3.4, 1.1, 51, length=60, gap=0.02) * np.clip(shade_p * 2, 0, 1))
    BL = np.maximum(BL, _hatch_box(SH, pbox, -40, 3.8, 1.0, 52, length=60, gap=0.02) *
                    R.smoothstep(0.45, 0.8, shade_p))
    for k, off in enumerate((-30, -12, 8, 26, 42)):
        a = np.linspace(-1, 1, 60)
        half = math.sqrt(max(0, pr ** 2 - off ** 2)) - 2
        px_ = a * half
        py_ = off + 6 * (1 - a ** 2)
        c, s = math.cos(-ang), math.sin(-ang)
        pts = np.stack([pcx + px_ * c - py_ * s, pcy + px_ * s + py_ * c], 1)
        _stroke(B, _wobble(pts, 0.5, 53 + k, step=2, wl=30), 1.1, taper=0.6, alpha=0.8)
    _stroke(B, _wobble(_loop_pts(pcx, pcy, pr, pr, 54), 0.5, 54, step=2, wl=30), 1.8, taper=0.4)
    _stroke(B, _wobble(_loop_pts(pcx, pcy, pr + 1.5, pr + 1.5, 55, turns=0.6), 0.6, 55, step=2, wl=30), 1.0,
            taper=0.6, alpha=0.6)
    for rx, ry, w_ in ((pr * 1.85, pr * 0.42, 1.8), (pr * 2.15, pr * 0.52, 1.5)):
        a = np.linspace(0, 2 * math.pi, 200)
        ex, ey = np.cos(a) * rx, np.sin(a) * ry
        c, s = math.cos(-ang), math.sin(-ang)
        X_ = pcx + ex * c - ey * s
        Y_ = pcy + ex * s + ey * c
        hidden = (np.hypot(X_ - pcx, Y_ - pcy) < pr + 1) & (ey < 0)
        run = []
        for i in range(len(a)):
            if not hidden[i]:
                run.append((X_[i], Y_[i]))
            elif len(run) > 2:
                _stroke(B, run, w_, taper=0.2)
                run = []
            else:
                run = []
        if len(run) > 2:
            _stroke(B, run, w_, taper=0.2)
    ringzone = ((ux / (pr * 1.85)) ** 2 + (uy / (pr * 0.42)) ** 2 > 1) & \
        ((ux / (pr * 2.15)) ** 2 + (uy / (pr * 0.52)) ** 2 < 1) & ~((dd < pr) & (uy < 0))
    BL = np.maximum(BL, _hatch_box(SH, pbox, -70, 3.0, 0.9,
                                   56, length=14, gap=0.3) * ringzone * 0.45)
    # moon
    mcx, mcy, mr = 930 + ox, 92 + oy, 15
    _stroke(B, _wobble(_loop_pts(mcx, mcy, mr, mr, 57), 0.4, 57, step=2, wl=20), 1.5, taper=0.4)
    for (cx_, cy_, cr_) in ((-5, -3, 3.5), (5, 4, 2.5), (2, -7, 1.8)):
        _stroke(B, _loop_pts(mcx + cx_, mcy + cy_, cr_, cr_, 58, turns=1.0), 1.0)

    # comet
    cx, cy = 470 + ox, 150 + oy
    _doodle_star(B, cx, cy, 10, 60, w=1.5, fill=BL)
    for k in range(4):
        _stroke(B, _wobble([(cx + 10 + k * 2, cy - 6 + k * 5), (cx + 70 + k * 10, cy - 30 + k * 4)], 0.5, 61 + k,
                           step=3, wl=30), 1.2, 0.5, alpha=0.8)

    # rocket loop trail (the rocket itself is a bobbing sprite)
    rx0, ry0 = 205 + ox, 196 + oy
    t = np.linspace(0, 1, 300)
    trail = np.stack([rx0 - 30 - 150 * t + 34 * np.sin(t * 2 * math.pi) * np.sin(math.pi * t),
                      ry0 + 30 + 150 * t - 40 * np.sin(t * 2 * math.pi + 0.4) * np.sin(math.pi * t)], 1)
    tr = _wobble(trail, 0.8, 62, step=2, wl=40)
    for i in range(0, len(tr) - 4, 8):
        _stroke(B, tr[i:i + 4], 1.3, alpha=0.8)

    # little UFO with a dashed tractor beam
    ux0, uy0 = 96 + ox, 440 + oy
    _stroke(B, _wobble(_loop_pts(ux0, uy0, 30, 8, 63), 0.4, 63, step=2, wl=20), 1.6, taper=0.4)
    a = np.linspace(math.pi, 2 * math.pi, 16)
    _stroke(B, _wobble(np.stack([ux0 + np.cos(a) * 13, uy0 - 4 + np.sin(a) * 13], 1), 0.3, 64, step=2), 1.5)
    for k in (-1, 0, 1):
        _stroke(RP, _loop_pts(ux0 + k * 14, uy0 + 1, 1.8, 1.8, 65 + k, turns=1.0), 1.4)
    for s_ in (-1, 1):
        pts = _resample([(ux0 + s_ * 10, uy0 + 9), (ux0 + s_ * 30, uy0 + 70)], 2)
        for i in range(0, len(pts) - 2, 5):
            _stroke(BL, pts[i:i + 3], 1.1)

    # constellation
    stars = [(1128, 318), (1168, 300), (1206, 336), (1240, 314), (1196, 392), (1150, 410)]
    for (ax, ay), (bx, by) in zip(stars, stars[1:]):
        pts = _resample([(ax + ox, ay + oy), (bx + ox, by + oy)], 2)
        for i in range(0, len(pts) - 2, 5):
            _stroke(BL, pts[i:i + 3], 1.0)
    for i, (ax, ay) in enumerate(stars):
        _doodle_star(B, ax + ox, ay + oy, 6, 70 + i, w=1.3, fill=BL if i % 2 else None)

    # scattered stars: 5-point outlines, scribble-filled, asterisks, sparkles, dots
    rng = np.random.default_rng(81)
    placed = []
    keep_out = [(1060, 178, 150), (205, 196, 70), (96, 440, 50), (470, 150, 40), (640, 50, 90)]
    tries = 0
    while len(placed) < 34 and tries < 600:
        tries += 1
        x, y = 30 + rng.random() * 1220, 110 + rng.random() * 400
        if any(math.hypot(x - kx, y - ky) < kr for kx, ky, kr in keep_out):
            continue
        if any(math.hypot(x - px_, y - py_) < 70 for px_, py_ in placed):
            continue
        placed.append((x, y))
        kind = rng.integers(0, 5)
        X0, Y0 = x + ox, y + oy
        s = int(rng.integers(0, 1 << 30))
        if kind == 0:
            _doodle_star(B, X0, Y0, 5 + rng.random() * 5, s, w=1.2)
        elif kind == 1:
            _doodle_star(RP if rng.random() < 0.35 else B, X0, Y0, 6 + rng.random() * 4, s, w=1.3, fill=BL)
        elif kind == 2:
            for k in range(3):
                a = k * math.pi / 3 + rng.random() * 0.3
                r_ = 4 + rng.random() * 2
                ca, sa = math.cos(a) * r_, math.sin(a) * r_
                _stroke(B, [(X0 - ca, Y0 - sa), (X0 + ca, Y0 + sa)], 1.1)
        elif kind == 3:
            r_ = 6 + rng.random() * 4
            q = r_ * 0.2
            pts = [(X0, Y0 - r_), (X0 + q, Y0 - q), (X0 + r_, Y0), (X0 + q, Y0 + q), (X0, Y0 + r_), (X0 - q, Y0 + q),
                   (X0 - r_, Y0), (X0 - q, Y0 - q), (X0, Y0 - r_)]
            _stroke(Y if rng.random() < 0.5 else B, _wobble(pts, 0.3, s, step=1.5, wl=10), 1.2)
        else:
            for k in range(3):
                _stroke(B, _loop_pts(X0 + rng.normal() * 6, Y0 + rng.normal() * 6, 1.0, 1.0, s + k, turns=1.0), 1.4)

    _mul(rgb, BL * (0.8 + 0.2 * tooth), BALLPOINT, 0.55)
    _mul(rgb, Y * tooth, (240, 170, 30), 0.9)
    _mul(rgb, B, BALLPOINT, 0.9)
    _mul(rgb, RP, RED_PEN, 0.85)
    _doodle_tape(rgb, 1262 + ox, 690 + oy, 100, 30, 0.7, 44, tint=(230, 245, 250))
    return R.opaque(np.clip(rgb, 0, 1))


def _menu_page(W: int, H: int) -> np.ndarray:
    """Ruled page for menus: doodles crowd the edges and leave the middle free."""
    ox, oy = PAD_X, PAD_Y
    rgb = _paper(W, H, 3)
    _ruled(rgb, W, H, seed=12)
    tooth = _tooth(W, H, 21)
    SH = (H, W)
    G = np.zeros((H, W), np.float32)
    B = np.zeros((H, W), np.float32)
    BF = np.zeros((H, W), np.float32)
    RP = np.zeros((H, W), np.float32)
    Y = np.zeros((H, W), np.float32)
    _doodle_spiral(G, 56 + ox, 60 + oy, 18, 1)
    _doodle_star(B, 52 + ox, 255 + oy, 15, 2, w=1.5, fill=BF)
    _doodle_smiley(G, 54 + ox, 460 + oy, 15, 3)
    _doodle_star(RP, 66 + ox, 660 + oy, 11, 4, w=1.4)
    # tic-tac-toe, bottom right
    gx0, gy0, s = 1150 + ox, 600 + oy, 26
    for k in (1, 2):
        _stroke(G, _wobble([(gx0 + k * s, gy0 - 4), (gx0 + k * s + 2, gy0 + 3 * s + 4)], 0.6, 10 + k, step=2), 1.4)
        _stroke(G, _wobble([(gx0 - 4, gy0 + k * s), (gx0 + 3 * s + 4, gy0 + k * s - 1)], 0.6, 20 + k, step=2), 1.4)
    for (i, j, o) in ((0, 0, 1), (1, 1, 0), (2, 0, 0), (2, 2, 1), (0, 2, 1)):
        cx, cy = gx0 + i * s + s / 2, gy0 + j * s + s / 2
        if o:
            _stroke(RP, _loop_pts(cx, cy, 8, 8, 30 + i * 3 + j), 1.6, taper=0.4)
        else:
            _stroke(B, [(cx - 7, cy - 7), (cx + 7, cy + 7)], 1.7, taper=0.3)
            _stroke(B, [(cx + 7, cy - 7), (cx - 7, cy + 7)], 1.7, taper=0.3)
    # 3D cube, top right
    cx, cy, s = 1180 + ox, 120 + oy, 34
    f = [(cx, cy), (cx + s, cy), (cx + s, cy + s), (cx, cy + s), (cx, cy)]
    b = [(x + 14, y - 12) for x, y in f]
    _stroke(G, _wobble(f, 0.5, 40, step=2, wl=30, over=3), 1.5)
    _stroke(G, _wobble(b, 0.5, 41, step=2, wl=30, over=3), 1.2, alpha=0.8)
    for (p, q) in zip(f[:4], b[:4]):
        _stroke(G, _wobble([p, q], 0.3, 42 + int(p[0]), step=2), 1.2)
    # lightning bolt
    bolt = [(1060 + ox, 60 + oy), (1046 + ox, 92 + oy), (1060 + ox, 90 + oy), (1048 + ox, 124 + oy),
            (1074 + ox, 82 + oy), (1060 + ox, 84 + oy), (1070 + ox, 60 + oy), (1060 + ox, 60 + oy)]
    _stroke(G, _wobble(bolt, 0.4, 50, step=2, wl=20), 1.5)
    xx, yy = R.grid(W, H)
    inside = R.coverage(R.sd_polygon(xx, yy, bolt[:-1]))
    Y = np.maximum(Y, _hatch_box(SH, (1040 + ox, 55 + oy, 1080 + ox, 130 + oy), 30, 2.6, 1.5, 51, gap=0.0) * inside)
    # two tiny stick figures sword-fighting, bottom left
    for k, (fx, face) in enumerate(((160, 1), (214, -1))):
        X0, Y0 = fx + ox, 650 + oy
        _stroke(G, _loop_pts(X0, Y0 - 30, 6, 6, 60 + k), 1.4)
        _stroke(G, [(X0, Y0 - 24), (X0, Y0 - 6)], 1.5)
        _stroke(G, [(X0, Y0 - 6), (X0 - 6, Y0 + 8)], 1.5)
        _stroke(G, [(X0, Y0 - 6), (X0 + 6, Y0 + 8)], 1.5)
        _stroke(G, [(X0, Y0 - 20), (X0 + face * 10, Y0 - 14)], 1.4)
        _stroke(G, [(X0 + face * 10, Y0 - 14), (X0 + face * 26, Y0 - 30)], 1.6, alpha=0.9)
    _np_text(RP, "clang!", 168 + ox, 600 + oy, 13, BALL_HAND, 61)
    # header scribble
    _np_text(G, "doodles...", 132 + ox, 74 + oy, 22, PENCIL_HAND, 7, alpha=0.7)
    _mul(rgb, Y * tooth, (250, 200, 40), 0.85)
    _mul(rgb, G * tooth, GRAPHITE, 0.9)
    _mul(rgb, B, BALLPOINT, 0.85)
    _mul(rgb, BF * tooth, BALLPOINT, 0.5)
    _mul(rgb, RP, RED_PEN, 0.85)
    _doodle_tape(rgb, 30 + ox, 10 + oy, 120, 34, -0.62, 70)
    _doodle_tape(rgb, 640 + ox, 6 + oy, 130, 32, 0.04, 71, tint=(255, 220, 230))
    return R.opaque(np.clip(rgb, 0, 1))


# =============================================================================
# SCENE SPRITES
# =============================================================================

def _cloud_sprite(w: int, h: int, seed: int) -> np.ndarray:
    M = 12
    W, H = w + 2 * M, h + 2 * M
    xx, yy = R.grid(W, H)
    rng = np.random.default_rng(seed)
    yb = M + h * 0.86
    n = 4 + int(rng.integers(0, 2))
    sd = np.full((H, W), 1e9, np.float32)
    bumps = []
    for i in range(n):
        t = (i + 0.5) / n
        cx = M + w * (0.12 + 0.76 * t)
        r = h * (0.32 + 0.22 * math.sin(math.pi * t) + rng.random() * 0.1)
        cy = yb - r * (0.55 + 0.35 * math.sin(math.pi * t))
        bumps.append((cx, cy, r))
        sd = np.minimum(sd, np.hypot(xx - cx, yy - cy) - r)
    sd = np.maximum(sd, yy - yb)
    sdn = sd + (R.fbm(xx, yy, 7, 2, seed=seed) - 0.5) * 2.4
    G = np.clip(1.0 - np.abs(sdn) + 0.4, 0, 1) * (1 - 0.35 * (np.abs(yy - yb) < 1.5))
    inside = R.coverage(sd)
    G = np.maximum(G, _hatch(xx, yy, 52, 4.0, 1.0, seed + 1, length=20, gap=0.05) * inside *
                   R.smoothstep(yb - 22, yb - 4, yy) * 0.75)
    for cx, cy, r in bumps[1:-1]:
        a = np.linspace(math.pi * 0.95, math.pi * 1.45, 8)
        pts = np.stack([cx + np.cos(a) * r * 0.6, cy + 4 + np.sin(a) * r * 0.6], 1)
        _stroke(G, pts, 1.1, taper=0.6, alpha=0.6)
    tooth = _tooth(W, H, seed + 3)
    return _rgba_from_layers(W, H, [(G, GRAPHITE, 0.85, True)], tooth)


def _sun_rays(seed: int) -> np.ndarray:
    S = 200
    G = np.zeros((S, S), np.float32)
    O = np.zeros((S, S), np.float32)
    c = S / 2
    for i in range(14):
        a = i * 2 * math.pi / 14 + 0.1
        r0, r1 = 50, (78 if i % 2 == 0 else 66)
        p0 = (c + math.cos(a) * r0, c + math.sin(a) * r0)
        p1 = (c + math.cos(a) * r1, c + math.sin(a) * r1)
        pts = _wobble([p0, p1], 0.8, seed + i, step=2, wl=14)
        _stroke(O, pts + 1.2, 3.4, 1.4, alpha=0.9)
        _stroke(G, pts, 1.8, 0.8, taper=0.2)
    tooth = _tooth(S, S, seed)
    return _rgba_from_layers(S, S, [(O, _on_paper((245, 150, 40)), 0.8, True), (G, GRAPHITE, 0.9, True)], tooth)


def _rocket_sprite(seed: int) -> np.ndarray:
    S = 110
    B = np.zeros((S, S), np.float32)
    F = np.zeros((S, S), np.float32)
    c = S / 2
    ang = math.radians(-38)
    ca, sa = math.cos(ang), math.sin(ang)

    def P(u, v):
        return (c + u * ca - v * sa, c + u * sa + v * ca)
    body = [P(-26, -10), P(14, -11), P(30, 0), P(14, 11), P(-26, 10), P(-26, -10)]
    _stroke(B, _wobble(body, 0.4, seed, step=2, wl=24, over=3), 1.6)
    _stroke(B, _wobble([P(-26, -10), P(-36, -20), P(-30, -8)], 0.3, seed + 1, step=2), 1.4)
    _stroke(B, _wobble([P(-26, 10), P(-36, 20), P(-30, 8)], 0.3, seed + 2, step=2), 1.4)
    wx, wy = P(4, 0)
    _stroke(B, _loop_pts(wx, wy, 5.5, 5.5, seed + 3), 1.4)
    _stroke(B, [P(16, -10), P(17, 10)], 1.0, alpha=0.7)
    xx, yy = R.grid(S, S)
    inside = R.coverage(R.sd_polygon(xx, yy, body[:-1]))
    F = np.maximum(F, _hatch(xx, yy, 52, 3.2, 1.0, seed + 4, gap=0.05) * inside *
                   (((xx - c) * -sa + (yy - c) * ca) > 2))
    return _rgba_from_layers(S, S, [(np.clip(inside, 0, 1), PAPER_HI, 1.0, False), (F, BALLPOINT, 0.5, False),
                                    (B, BALLPOINT, 0.95, False)])


def _flame_sprite(seed: int) -> np.ndarray:
    S = 64
    Rm = np.zeros((S, S), np.float32)
    Ym = np.zeros((S, S), np.float32)
    rng = np.random.default_rng(seed)
    c = S / 2
    ang = math.radians(-38) + math.pi
    for k in range(5):
        a = ang + (k - 2) * 0.22 + rng.normal() * 0.05
        L = 16 + rng.random() * 12 - abs(k - 2) * 3
        pts = [(c, c), (c + math.cos(a) * L * 0.5 + rng.normal() * 2, c + math.sin(a) * L * 0.5 + rng.normal() * 2),
               (c + math.cos(a) * L, c + math.sin(a) * L)]
        _stroke(Rm if k % 2 == 0 else Ym, _resample(pts, 2), 2.2, 0.6)
    return _rgba_from_layers(S, S, [(Ym, (240, 170, 30), 0.9, False), (Rm, RED_PEN, 0.9, False)])


def _rock_profile(W: int, x0: float, w: float, y1: float, under: float, seed: int) -> np.ndarray:
    """Angular (piecewise-linear) underside silhouette: bottom y per column."""
    rng = np.random.default_rng(seed)
    cols = np.arange(W, dtype=np.float32)
    kx = [x0 + 2.0]
    x = x0 + 2.0
    while x < x0 + w - 40:
        x += 24 + rng.random() * 42
        kx.append(min(x, x0 + w - 2))
    kx[-1] = x0 + w - 2
    kx = np.array(kx, np.float32)
    t = (kx - x0) / w
    prof = np.clip(1 - np.abs(2 * t - 1) ** 1.5, 0, 1)
    depth = under * prof ** 0.9 * (0.5 + 0.5 * rng.random(len(kx)))
    depth += under * 0.32 * np.exp(-((t - 0.5 - rng.normal() * 0.04) / 0.1) ** 2)
    depth[0] = depth[-1] = 0
    return y1 + np.interp(cols, kx, depth, left=0, right=0)


def _facets(xx, yy, region, seed: int, n: int, box):
    """Voronoi rock facets inside a region: (cell index map, facet edge coverage)."""
    rng = np.random.default_rng(seed)
    bx0, by0, bx1, by1 = box
    px = rng.uniform(bx0, bx1, n)
    py = rng.uniform(by0, by1, n)
    xw = xx + 5 * np.sin(yy / 11.0) + 3 * np.sin(yy / 4.3)
    yw = yy + 5 * np.sin(xx / 15.0)
    d1 = np.full(xx.shape, 1e9, np.float32)
    d2 = np.full(xx.shape, 1e9, np.float32)
    idx = np.zeros(xx.shape, np.int64)
    for k in range(n):
        d = np.hypot(xw - px[k], (yw - py[k]) * 1.25)
        closer = d < d1
        d2 = np.where(closer, d1, np.minimum(d2, d))
        idx = np.where(closer, k, idx)
        d1 = np.minimum(d1, d)
    edge = np.clip(1.0 - (d2 - d1) * 1.15, 0, 1) * region
    return idx, edge, px, py


def _rock_hatch(xx, yy, idx, region, cx_cells, seed, left_x, right_x, top_y, depth_px, spacing=4.6):
    """Each facet gets one of four hatch directions and its own darkness (darker left and near the top)."""
    n = len(cx_cells)
    rng = np.random.default_rng(seed)
    bucket = rng.integers(0, 4, n)
    dark = 0.35 + 0.4 * rng.random(n)
    tl = np.clip(1 - (cx_cells - left_x) / max(1.0, right_x - left_x), 0, 1)
    dark = np.clip(dark + 0.3 * tl, 0, 1)
    out = np.zeros(xx.shape, np.float32)
    for b, ang in enumerate((28, 62, 118, 152)):
        sel = (bucket[idx] == b)
        if sel.any():
            out = np.maximum(out, _hatch(xx, yy, ang, spacing, 1.0, seed + b, length=40, gap=0.06) * sel)
    out = out * dark[idx] * region
    near = 1 - R.smoothstep(3, depth_px, yy - top_y)
    out = np.maximum(out, _hatch(xx, yy, -12, spacing * 0.9, 1.0, seed + 9, length=40, gap=0.03) * region * near * 0.9)
    cross = (dark[idx] > 0.72) * region
    out = np.maximum(out, _hatch(xx, yy, 90 + 40, spacing * 1.1, 0.9, seed + 11, length=26) * cross * 0.7)
    return out


def _plat_bf_main(w: int, h: int) -> np.ndarray:
    """Pencil slab with a faceted, hatched rocky underside and grass tufts. Top edge at y=M."""
    M, UNDER = 34, 112
    W, H = w + 2 * M, M + h + UNDER + 16
    xx, yy = R.grid(W, H)
    x0, y0, x1, y1 = M, M, M + w, M + h
    cols = np.arange(W, dtype=np.float32)
    bot = _rock_profile(W, x0 + 6, w - 12, y1, UNDER, 3)
    under = np.clip(bot[None, :] - yy + 0.5, 0, 1) * (yy > y1 - 2)
    slab = R.coverage(R.sd_box(xx, yy, (x0 + x1) / 2, (y0 + y1) / 2, w / 2, h / 2))
    body = np.maximum(slab, under)
    G = np.zeros((H, W), np.float32)
    GR = np.zeros((H, W), np.float32)
    depth_s = np.clip((yy - y0) / h, 0, 1)
    G = np.maximum(G, _hatch(xx, yy, 45, 4.2, 1.1, 11, length=34, gap=0.04) * slab * (0.55 + 0.4 * depth_s))
    G = np.maximum(G, _hatch(xx, yy, -45, 5.0, 0.95, 12, length=28) * slab * R.smoothstep(0.45, 1.0, depth_s) * 0.7)
    idx, edge, pcx, pcy = _facets(xx, yy, under, 13, 26, (x0, y1, x1, y1 + UNDER))
    G = np.maximum(G, _rock_hatch(xx, yy, idx, under, pcx, 14, x0, x1, y1, 26))
    G = np.maximum(G, edge * (yy > y1 + 3) * 0.85)
    rng = np.random.default_rng(16)
    for k in range(2):
        yv = y0 + 17 + k * 15
        xa = x0 + 20
        while xa < x1 - 40:
            L = 60 + rng.random() * 140
            xb = min(x1 - 12, xa + L)
            _stroke(G, _wobble([(xa, yv + rng.normal()), (xb, yv + rng.normal())], 0.8, int(xa) + k, step=3, wl=40),
                    1.2, taper=0.6, alpha=0.8)
            xa = xb + 20 + rng.random() * 60
    for k in range(int(w / 70)):
        cx = x0 + 30 + k * 70 + rng.random() * 30
        _stroke(G, _wobble([(cx, y0 + 4), (cx + rng.normal() * 3, y0 + 4 + 8 + rng.random() * 10)], 0.6, 700 + k,
                           step=2), 1.0, taper=0.5, alpha=0.6)
    for k in range(6):
        cx = x0 + 80 + rng.random() * (w - 160)
        cyb = bot[int(cx)]
        if cyb < y1 + 20:
            continue
        L = 10 + rng.random() * 18
        pts = [(cx, cyb - 2), (cx + 3, cyb + L * 0.5), (cx - 2, cyb + L)]
        _stroke(G, _wobble(pts, 1.0, 200 + k, step=2, wl=10), 1.1, 0.5, alpha=0.85)
    for p in range(2):
        a = 1.0 if p == 0 else 0.55
        j = 1.0 if p == 0 else -1.2
        _stroke(G, _wobble([(x0 - 9, y1 + j), (x1 + 8, y1 - j)], 0.6, 300 + p, step=3, wl=60), 2.1, taper=0.4, alpha=a)
        _stroke(G, _wobble([(x0 + j, y0 - 6), (x0 - j, y1 + 8)], 0.5, 310 + p, step=3, wl=40), 2.1, taper=0.4, alpha=a)
        _stroke(G, _wobble([(x1 - j, y0 - 7), (x1 + j, y1 + 9)], 0.5, 320 + p, step=3, wl=40), 2.1, taper=0.4, alpha=a)
    _curve(G, bot, 2.2, keep=((cols > x0 + 5) & (cols < x1 - 5)).astype(np.float32), seed=330, wob=0.4)
    # the walkable top: a firm dark pencil line, exactly at y0
    _stroke(G, [(x0 - 10, y0 + 0.4), (x1 + 10, y0 - 0.4)], 3.0, taper=0.3)
    _stroke(G, _wobble([(x0 - 4, y0 + 1.8), (x1 + 4, y0 + 1.4)], 0.5, 340, step=3, wl=80), 1.2, taper=0.5, alpha=0.6)
    for k in range(int(w / 44)):
        gx = x0 + 14 + k * 44 + rng.random() * 26
        if gx > x1 - 10:
            break
        for s_ in (-1, 0, 1):
            hgt = 4 + rng.random() * 3.5
            _stroke(GR, [(gx + s_ * 2.5, y0), (gx + s_ * 4.5, y0 - hgt)], 1.6, 0.6)
            _stroke(G, [(gx + s_ * 2.5, y0), (gx + s_ * 4.5, y0 - hgt)], 0.9, 0.4, alpha=0.7)
    tooth = _tooth(W, H, 17)
    return _rgba_from_layers(W, H, [(body, (240, 235, 221), 1.0, False), (GR, _on_paper((110, 180, 80)), 0.85, True),
                                    (G, (66, 68, 78), 1.0, True)], tooth)


def _plat_fd_main(w: int, h: int) -> np.ndarray:
    """Ballpoint floating island: cross-hatched slab, faceted rocky underside, floating chunks."""
    M, UNDER = 50, 112
    W, H = w + 2 * M, M + h + UNDER + 20
    xx, yy = R.grid(W, H)
    x0, y0, x1, y1 = M, M, M + w, M + h
    cols = np.arange(W, dtype=np.float32)
    bot = _rock_profile(W, x0 + 4, w - 8, y1, UNDER, 5)
    under = np.clip(bot[None, :] - yy + 0.5, 0, 1) * (yy > y1 - 2)
    slab = R.coverage(R.sd_box(xx, yy, (x0 + x1) / 2, (y0 + y1) / 2, w / 2, h / 2))
    chunks = [(x0 + 100, y1 + 70, 13), (x1 - 96, y1 + 62, 16), (x0 + 150, y1 + 100, 7), (x1 - 160, y1 + 104, 9)]
    csd = np.full((H, W), 1e9, np.float32)
    chunk_pts = []
    for (cx, cy, r) in chunks:
        pts = [(cx + math.cos(a) * r * (0.8 + 0.35 * math.sin(a * 3 + cx)), cy + math.sin(a) * r * 0.75)
               for a in np.linspace(0, 2 * math.pi, 8)[:-1]]
        chunk_pts.append(pts)
        csd = np.minimum(csd, R.sd_polygon(xx, yy, pts))
    chunk = R.coverage(csd) * (1 - under)
    body = np.maximum(np.maximum(slab, under), chunk)
    B = np.zeros((H, W), np.float32)
    BL = np.zeros((H, W), np.float32)
    depth_s = np.clip((yy - y0) / h, 0, 1)
    BL = np.maximum(BL, _hatch(xx, yy, 40, 3.6, 1.1, 21, length=50, gap=0.03) * slab * (0.6 + 0.4 * depth_s))
    BL = np.maximum(BL, _hatch(xx, yy, -40, 4.2, 0.95, 22, length=50, gap=0.03) * slab *
                    R.smoothstep(0.35, 1, depth_s) * 0.85)
    idx, edge, pcx, pcy = _facets(xx, yy, under, 23, 30, (x0, y1, x1, y1 + UNDER))
    BL = np.maximum(BL, _rock_hatch(xx, yy, idx, under, pcx, 24, x0, x1, y1, 32, spacing=4.0))
    B = np.maximum(B, edge * (yy > y1 + 3) * 0.9)
    BL = np.maximum(BL, _hatch(xx, yy, 60, 3.2, 1.0, 41) * chunk * 0.85)
    for p in range(2):
        a = 1.0 if p == 0 else 0.55
        j = 1.0 if p == 0 else -1.2
        _stroke(B, _wobble([(x0 - 10, y1 + j), (x1 + 9, y1 - j)], 0.5, 400 + p, step=3, wl=60), 1.9, taper=0.4, alpha=a)
        _stroke(B, _wobble([(x0 + j, y0 - 6), (x0 - j, y1 + 8)], 0.5, 410 + p, step=3, wl=40), 1.9, taper=0.4, alpha=a)
        _stroke(B, _wobble([(x1 - j, y0 - 7), (x1 + j, y1 + 9)], 0.5, 420 + p, step=3, wl=40), 1.9, taper=0.4, alpha=a)
    _curve(B, bot, 1.9, keep=((cols > x0 + 3) & (cols < x1 - 3)).astype(np.float32), seed=430, wob=0.3)
    for pts, (cx, cy, r) in zip(chunk_pts, chunks):
        _stroke(B, _wobble(pts + [pts[0], pts[1]], 0.4, int(cx), step=2, wl=14), 1.5)
        _stroke(B, [(cx - r - 12, cy + 3), (cx - r - 4, cy + 3)], 1.0, alpha=0.6)
        _stroke(B, [(cx + r + 4, cy - 2), (cx + r + 14, cy - 2)], 1.0, alpha=0.6)
    rng = np.random.default_rng(44)
    for k in range(6):
        cx = x0 + 120 + rng.random() * (w - 240)
        cyb = bot[int(cx)]
        if cyb < y1 + 30:
            continue
        L = 14 + rng.random() * 26
        pts = [(cx, cyb - 2), (cx + 4, cyb + L * 0.4), (cx - 3, cyb + L * 0.75), (cx + 1, cyb + L)]
        _stroke(B, _wobble(pts, 1.0, 500 + k, step=2, wl=10), 1.1, 0.5)
        _stroke(B, _loop_pts(cx + 1, cyb + L + 2, 1.6, 1.6, 510 + k, turns=1.0), 1.0)
    _stroke(B, [(x0 - 10, y0 + 0.4), (x1 + 10, y0 - 0.4)], 3.0, taper=0.3)
    _stroke(B, _wobble([(x0 - 4, y0 + 1.8), (x1 + 4, y0 + 1.4)], 0.5, 440, step=3, wl=80), 1.1, taper=0.5, alpha=0.6)
    return _rgba_from_layers(W, H, [(body, (250, 249, 243), 1.0, False), (BL, BALLPOINT, 0.55, False),
                                    (B, BALLPOINT, 0.95, False)])


def _plat_strip(w: int, seed: int) -> np.ndarray:
    """Soft platform: a torn strip of legal-pad paper, taped on. Top edge at y=M."""
    M, SH = 16, 19
    W, H = w + 2 * M, SH + 2 * M + 6
    xx, yy = R.grid(W, H)
    x0, y0, x1 = M, M, M + w
    rng = np.random.default_rng(seed)
    cols = np.arange(W, dtype=np.float32)
    tear = (R.fbm1d(W, 9, 3, seed=seed) - 0.5) * 6 + (R._hash2(cols.astype(np.int64) // 2, np.zeros(W, np.int64),
                                                                seed) - 0.5) * 2.2
    yb = y0 + SH + tear
    inside = np.clip(yb[None, :] - yy + 0.5, 0, 1) * R.coverage(-(yy - y0) + 0.5 - 0.5) * \
        np.clip(xx - x0 + 0.5, 0, 1) * np.clip(x1 - xx + 0.5, 0, 1)
    img = R.blank(W, H)
    shadow = R.blur(np.roll(np.roll(inside, 4, axis=0), 3, axis=1), 3.0)
    R.paint(img, shadow, (70, 55, 30), a=0.28)
    paper = R.gradient_stops(np.clip((yy - y0) / SH, 0, 1), [(0, (255, 240, 150)), (1, (246, 222, 120))])
    paper *= (0.97 + 0.05 * R.fbm(xx, yy, 6, 2, seed=seed + 1))[..., None]
    R.paint(img, inside, paper)
    fiber = inside * np.clip(1 - (yb[None, :] - yy) / 2.6, 0, 1)
    R.paint(img, fiber, (255, 253, 240), a=0.85)
    R.paint(img, inside * np.clip(0.9 - np.abs(yy - (y0 + 10.5)), 0, 1), (150, 180, 220), a=0.6)
    G = np.zeros((H, W), np.float32)
    _stroke(G, [(x0 - 3, y0 + 0.6), (x1 + 3, y0 + 0.2)], 2.6, taper=0.25)
    _stroke(G, _wobble([(x0 + 2, y0 + 2.2), (x1 - 2, y0 + 1.8)], 0.4, seed + 3, step=3, wl=60), 1.0, taper=0.5,
            alpha=0.5)
    _stroke(G, _wobble([(x0 + 0.5, y0), (x0 - 0.5, y0 + SH)], 0.3, seed + 4, step=2), 1.2, alpha=0.5)
    _stroke(G, _wobble([(x1 - 0.5, y0), (x1 + 0.5, y0 + SH)], 0.3, seed + 5, step=2), 1.2, alpha=0.5)
    tooth = _tooth(W, H, seed + 6)
    R.paint(img, G * tooth, GRAPHITE, a=0.95)
    for k, ex in enumerate((x0 + 8, x1 - 8)):
        ang = (-0.35 if k == 0 else 0.3) + rng.normal() * 0.05
        tw, th = 18, 34
        c, s = math.cos(ang), math.sin(ang)
        cy = y0 + 6
        u = (xx - ex) * c + (yy - cy) * s
        v = -(xx - ex) * s + (yy - cy) * c
        jag = (R._hash2(np.floor(u / 2.0).astype(np.int64), (v > 0).astype(np.int64), seed + k) - 0.5) * 3
        tape = np.clip(tw / 2 - np.abs(u) + 0.5, 0, 1) * np.clip(th / 2 + jag - np.abs(v) + 0.5, 0, 1)
        R.paint(img, tape, (250, 247, 230), a=0.62)
        R.paint(img, tape * np.exp(-((u + 3) / 2.5) ** 2), (255, 255, 255), a=0.35)
        edge = tape * np.clip(1.2 - np.abs(np.abs(u) - tw / 2), 0, 1)
        R.paint(img, edge, (140, 130, 100), a=0.45)
    return img


# =============================================================================
# FX SPRITES (white alpha masks unless noted)
# =============================================================================

def _mask_rgba(m) -> np.ndarray:
    return R.tint(np.clip(m, 0, 1), (255, 255, 255))


def _bubble_mask(seed: int) -> np.ndarray:
    S = 160
    m = np.zeros((S, S), np.float32)
    c, r = S / 2, S / 2 - 8
    xx, yy = R.grid(S, S)
    d = np.hypot(xx - c, yy - c)
    _stroke(m, _wobble(_loop_pts(c, c, r, r, seed, turns=1.15), 1.0, seed, step=3, wl=40), 3.2, taper=0.6)
    _stroke(m, _wobble(_loop_pts(c, c, r - 3, r - 2, seed + 1, turns=0.7), 1.2, seed + 1, step=3, wl=40), 1.8,
            taper=0.7, alpha=0.7)
    cres = R.coverage(d - r + 2) * (np.hypot(xx - c + r * 0.35, yy - c + r * 0.35) > r * 1.05)
    m = np.maximum(m, _hatch(xx, yy, 45, 5.0, 1.6, seed + 2, length=40, gap=0.05) * cres * 0.8)
    for k, (a0, a1, rr) in enumerate(((3.6, 4.4, 0.72), (3.45, 3.62, 0.72))):
        a = np.linspace(a0, a1, 10)
        _stroke(m, np.stack([c + np.cos(a) * r * rr, c + np.sin(a) * r * rr], 1), 3.0, taper=0.7)
    m = np.maximum(m, R.coverage(d - r) * 0.14)
    return _mask_rgba(m)


def _scribble_ring(seed: int) -> np.ndarray:
    S = 160
    m = np.zeros((S, S), np.float32)
    c, r = S / 2, S / 2 - 10
    for k in range(3):
        pts = _loop_pts(c, c, r - k * 4, r - k * 3.5, seed + k, turns=1.1, jit=0.07)
        _stroke(m, _wobble(pts, 1.4, seed + k, step=3, wl=30), 2.6 - k * 0.5, taper=0.6, alpha=1 - k * 0.2)
    return _mask_rgba(m)


def _scribble_ball(seed: int) -> np.ndarray:
    S = 64
    m = np.zeros((S, S), np.float32)
    rng = np.random.default_rng(seed)
    c, r = S / 2, S / 2 - 6
    pts = []
    for k in range(9):
        rr = r * (0.3 + 0.65 * rng.random())
        cx, cy = c + rng.normal() * r * 0.18, c + rng.normal() * r * 0.18
        pts.append(_loop_pts(cx, cy, rr, rr * (0.8 + 0.3 * rng.random()), seed + k, turns=1.0))
    _stroke(m, _wobble(np.vstack(pts), 0.6, seed, step=2, wl=12), 2.0, alpha=0.85)
    return _mask_rgba(m)


def _ink_ring(seed: int) -> np.ndarray:
    S = 64
    m = np.zeros((S, S), np.float32)
    c, r = S / 2, S / 2 - 6
    _stroke(m, _wobble(_loop_pts(c, c, r, r, seed, turns=1.15), 0.8, seed, step=2, wl=20), 3.0, taper=0.6)
    return _mask_rgba(m)


def _small_star(seed: int) -> np.ndarray:
    S = 64
    m = np.zeros((S, S), np.float32)
    rng = np.random.default_rng(seed)
    c = S / 2
    n = 6 + int(rng.integers(0, 3))
    for k in range(n):
        a = k * 2 * math.pi / n + rng.normal() * 0.15
        r0, r1 = 7 + rng.random() * 3, 18 + rng.random() * 10
        _stroke(m, [(c + math.cos(a) * r0, c + math.sin(a) * r0), (c + math.cos(a) * r1, c + math.sin(a) * r1)],
                3.2, 0.8)
    _stroke(m, _wobble(_star_pts(c, c, 7, rng.random(), 4, 0.4), 0.3, seed, step=1.5, wl=8), 1.8)
    return _mask_rgba(m)


def _splat(seed: int) -> np.ndarray:
    S = 128
    xx, yy = R.grid(S, S)
    c = S / 2
    rng = np.random.default_rng(seed)
    ang = np.arctan2(yy - c, xx - c)
    rad = 22 + 6 * np.sin(ang * 5 + rng.random() * 6) + 4 * np.sin(ang * 9 + rng.random() * 6)
    sd = np.hypot(xx - c, yy - c) - rad
    for k in range(11):
        a = rng.random() * 6.28
        dist = 30 + rng.random() * 26
        r = 2 + rng.random() * 4.5
        sd = np.minimum(sd, np.hypot(xx - c - math.cos(a) * dist, yy - c - math.sin(a) * dist) - r)
        if rng.random() < 0.6:
            sd = np.minimum(sd, R.sd_segment(xx, yy, c + math.cos(a) * 18, c + math.sin(a) * 18,
                                             c + math.cos(a) * (dist - r), c + math.sin(a) * (dist - r), r * 0.45))
    return _mask_rgba(R.coverage(sd))


def _puff(seed: int) -> np.ndarray:
    S = 64
    m = np.zeros((S, S), np.float32)
    rng = np.random.default_rng(seed)
    c = S / 2
    pts = []
    n = 5
    for k in range(n + 1):
        a = math.pi + k * math.pi / n
        rr = 15 + rng.random() * 3
        cx, cy = c + math.cos(a) * rr, c + 6 + math.sin(a) * rr * 0.8
        b = np.linspace(a - 1.2, a + 1.2, 10)
        pts.append(np.stack([cx + np.cos(b) * 7, cy + np.sin(b) * 6], 1))
    _stroke(m, _wobble(np.vstack(pts), 0.5, seed, step=2, wl=12), 2.0, taper=0.4)
    _stroke(m, _wobble([(c - 18, c + 10), (c + 18, c + 10)], 0.6, seed + 1, step=2), 1.6, taper=0.6)
    a = np.linspace(0, 1.6 * math.pi, 14)
    _stroke(m, np.stack([c - 2 + np.cos(a) * 4, c + 2 + np.sin(a) * 3], 1), 1.4, taper=0.6, alpha=0.8)
    return _mask_rgba(m)


def _smudge(seed: int) -> np.ndarray:
    W, H = 128, 32
    xx, yy = R.grid(W, H)
    e = np.sqrt(((xx - W / 2) / (W * 0.45)) ** 2 + ((yy - H / 2) / (H * 0.32)) ** 2)
    streak = 0.75 + 0.25 * R.value_noise(xx * 0.15, yy * 3, 1.0, seed)
    return _mask_rgba(np.clip(1 - e, 0, 1) ** 1.3 * streak)


def _blob(seed: int) -> np.ndarray:
    S = 128
    xx, yy = R.grid(S, S)
    d = np.hypot(xx - S / 2, yy - S / 2) / (S / 2)
    n = R.fbm(xx, yy, 14, 3, seed=seed)
    h = _hatch(xx, yy, 40, 4.0, 2.4, seed, length=30, gap=0.0, wob=1.0)
    a = np.clip(1 - d, 0, 1) ** 0.9 * (0.55 + 0.45 * h) * (0.7 + 0.5 * n)
    return _mask_rgba(np.clip(a, 0, 1))


def _burst(word: str, fill, seed: int, R0: float) -> np.ndarray:
    """Comic hit burst: jagged star, halftone dots, ink outline, hand-lettered sound effect (RGBA)."""
    S = int(R0 * 2 + 70)
    xx, yy = R.grid(S, S)
    c = S / 2
    rng = np.random.default_rng(seed)
    n = 12 + int(rng.integers(0, 4))
    a0 = rng.random() * 6.28
    outer, inner = [], []
    for i in range(n * 2):
        a = a0 + i * math.pi / n + rng.normal() * 0.06
        if i % 2 == 0:
            rr = R0 * (0.88 + 0.26 * rng.random())
        else:
            rr = R0 * (0.58 + 0.1 * rng.random())
        outer.append((c + math.cos(a) * rr, c + math.sin(a) * rr * 0.86))
        inner.append((c + math.cos(a) * rr * 0.66, c + math.sin(a) * rr * 0.66 * 0.86))
    sd = R.sd_polygon(xx, yy, outer)
    sdi = R.sd_polygon(xx, yy, inner)
    img = R.blank(S, S)
    body = R.coverage(sd)
    R.paint(img, body, fill)
    a = math.radians(20)
    u = (xx * math.cos(a) + yy * math.sin(a)) / 7.0
    v = (-xx * math.sin(a) + yy * math.cos(a)) / 7.0
    dd = np.hypot(u - np.round(u), v - np.round(v)) * 7.0
    rad = np.clip(np.hypot(xx - c, yy - c) / R0, 0, 1)
    dots = R.coverage(dd - (0.6 + 2.6 * rad)) * body * R.coverage(-sdi - 2)
    R.paint(img, dots, _mix(fill, (120, 20, 20), 0.35), a=0.6)
    R.paint(img, R.coverage(sdi), _mix(fill, (255, 255, 240), 0.65))
    m = np.zeros((S, S), np.float32)
    _stroke(m, _wobble(outer + [outer[0], outer[1]], 0.6, seed, step=2, wl=24), 4.2, taper=0.15)
    for i in range(0, n * 2, 2):
        x_, y_ = outer[i]
        dx, dy = x_ - c, y_ - c
        L = math.hypot(dx, dy) or 1
        if rng.random() < 0.6:
            _stroke(m, [(x_ + dx / L * 8, y_ + dy / L * 8), (x_ + dx / L * (18 + rng.random() * 14),
                                                             y_ + dy / L * (18 + rng.random() * 14))], 3.0, 0.8)
    R.paint(img, m, INK)
    size = R0 * (0.62 if len(word) <= 4 else 0.5)
    tw = _text_width(word, size, HEAVY_HAND)
    if tw > R0 * 1.12:
        size *= R0 * 1.12 / tw
        tw = R0 * 1.12
    tm = np.zeros((S, S), np.float32)
    _np_text(tm, word, c - tw / 2, c + size / 2, size, HEAVY_HAND, seed)
    halo = np.clip(R.blur(tm, 2.2) * 5, 0, 1)
    R.paint(img, halo, (255, 255, 250))
    R.paint(img, tm, INK)
    return img


def _crown(seed: int) -> np.ndarray:
    W, H = 120, 90
    xx, yy = R.grid(W, H)
    pts = [(14, 74), (8, 22), (36, 50), (60, 12), (84, 50), (112, 22), (106, 74), (14, 74)]
    inside = R.coverage(R.sd_polygon(xx, yy, pts[:-1]))
    Y = _hatch(xx, yy, 35, 2.6, 1.8, seed, gap=0.0) * inside
    G = np.zeros((H, W), np.float32)
    _stroke(G, _wobble(pts, 0.6, seed, step=2, wl=30, over=4), 2.4, taper=0.2)
    _stroke(G, _wobble([(16, 62), (104, 62)], 0.6, seed + 1, step=2), 1.6)
    for (x, y) in ((8, 18), (60, 8), (112, 18)):
        _stroke(G, _loop_pts(x, y, 4, 4, seed + x), 1.8)
    tooth = _tooth(W, H, seed)
    return _rgba_from_layers(W, H, [(Y, _on_paper((255, 200, 30)), 0.95, True), (G, INK, 1.0, False)], tooth)


def _squiggle(seed: int) -> np.ndarray:
    S = 24
    m = np.zeros((S, S), np.float32)
    rng = np.random.default_rng(seed)
    kind = seed % 3
    if kind == 0:
        t = np.linspace(-8, 8, 16)
        pts = np.stack([S / 2 + t, S / 2 + 3 * np.sin(t * 0.9)], 1)
        _stroke(m, pts, 2.2)
    elif kind == 1:
        _stroke(m, _star_pts(S / 2, S / 2, 7, rng.random(), 5, 0.45), 1.6)
    else:
        _stroke(m, _loop_pts(S / 2, S / 2, 5, 5, seed, turns=1.1), 2.0)
    return _mask_rgba(m)


def _marker_tex(color, seed: int) -> np.ndarray:
    """Seamless 128px highlighter tile: diagonal nib streaks where passes overlap, uneven ink."""
    S = 128
    xx, yy = R.grid(S, S)
    tau = 2 * math.pi / S
    rng = np.random.default_rng(seed)
    a = np.zeros((S, S), np.float32)
    for (fx, fy, amp) in ((10, -6, 0.06), (21, -13, 0.03), (3, 2, 0.07), (-2, 5, 0.06), (1, -4, 0.06),
                          (5, 3, 0.05), (7, 9, 0.03)):
        a += amp * np.sin(tau * (fx * xx + fy * yy) + rng.random() * 6.28)
    streak = np.abs(np.sin(tau * (10 * xx - 6 * yy) * 0.5 + 1.0)) ** 10 * 0.08
    alpha = np.clip(0.66 + a + streak, 0.42, 0.95)
    return R.tint(alpha, _on_paper(color))


def _sticky(w: int, h: int, color, angle_deg: float, seed: int) -> np.ndarray:
    G = 14
    W, H = w + 2 * G, h + 2 * G
    xx, yy = R.grid(W, H)
    a = math.radians(angle_deg)
    u = (xx - W / 2) * math.cos(a) + (yy - H / 2) * math.sin(a)
    v = -(xx - W / 2) * math.sin(a) + (yy - H / 2) * math.cos(a)
    sd = R.sd_box(u, v, 0, 0, w / 2, h / 2, 1.5)
    img = R.blank(W, H)
    sh = R.blur(R.coverage(R.sd_box(u - 3, v - 5, 0, 0, w / 2 - 2, h / 2, 2)), 4.0)
    R.paint(img, sh, (60, 45, 20), a=0.32)
    base = _mix(color, (255, 250, 232), 0.7)
    t = (v + h / 2) / h
    col = R.gradient_stops(t, [(0, _mix(base, (255, 255, 255), 0.1)), (0.18, base),
                               (1, _mix(base, (120, 90, 50), 0.1))])
    corner = np.exp(-np.hypot(u - w / 2, v - h / 2) / 26.0) * 0.18
    col *= (1 - corner[..., None])
    col *= (0.975 + 0.04 * R.fbm(xx, yy, 5, 2, seed=seed))[..., None]
    R.paint(img, R.coverage(sd), col)
    R.paint(img, R.coverage(sd) * (np.abs(v + h / 2 - 16) < 0.6), (0, 0, 0), a=0.05)
    R.paint(img, np.clip(1 - np.abs(sd + 0.5), 0, 1), (90, 70, 40), a=0.18)
    return img


def _stock_head(color, seed: int) -> np.ndarray:
    S = 24
    img = R.blank(S, S)
    xx, yy = R.grid(S, S)
    R.paint(img, R.coverage(np.hypot(xx - S / 2 - 1.5, yy - S / 2 - 1) - 7.5), _on_paper(color), a=0.85)
    m = np.zeros((S, S), np.float32)
    _stroke(m, _wobble(_loop_pts(S / 2, S / 2, 7, 7, seed, turns=1.15), 0.3, seed, step=1.5, wl=10), 1.8, taper=0.6)
    for s_ in (-1, 1):
        _stroke(m, _loop_pts(S / 2 + s_ * 2.6, S / 2 - 1, 0.6, 0.6, seed + s_, turns=1.0), 1.4)
    a = np.linspace(0.25 * math.pi, 0.75 * math.pi, 6)
    _stroke(m, np.stack([S / 2 + np.cos(a) * 3.4, S / 2 + 0.5 + np.sin(a) * 2.6], 1), 1.1)
    R.paint(img, m, INK)
    return img


def _bar_frame(w: int, h: int, seed: int) -> np.ndarray:
    W, H = w + 16, h + 16
    m = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = 8, 8, 8 + w, 8 + h
    _stroke(m, _wobble([(x0 - 4, y0), (x1 + 3, y0 - 0.5)], 0.4, seed, step=2, wl=40), 1.5, taper=0.4)
    _stroke(m, _wobble([(x0 - 3, y1), (x1 + 4, y1 + 0.5)], 0.4, seed + 1, step=2, wl=40), 1.5, taper=0.4)
    _stroke(m, _wobble([(x0, y0 - 3), (x0 + 0.5, y1 + 3)], 0.3, seed + 2, step=2), 1.5, taper=0.4)
    _stroke(m, _wobble([(x1, y0 - 3), (x1 - 0.5, y1 + 3)], 0.3, seed + 3, step=2), 1.5, taper=0.4)
    return _mask_rgba(m)


def _timer_tag(seed: int) -> np.ndarray:
    W, H = 120, 58
    xx, yy = R.grid(W, H)
    img = R.blank(W, H)
    tear = (R.fbm1d(W, 6, 2, seed=seed) - 0.5) * 4
    inside = np.clip(xx - 10, 0, 1) * np.clip(W - 10 - xx, 0, 1) * np.clip(yy - 12, 0, 1) * \
        np.clip((H - 8 + tear[None, :]) - yy, 0, 1)
    R.paint(img, R.blur(np.roll(np.roll(inside, 3, 0), 2, 1), 2.5), (60, 45, 20), a=0.25)
    R.paint(img, inside, (252, 250, 243))
    R.paint(img, inside * np.clip(1 - ((H - 8 + tear[None, :]) - yy) / 2, 0, 1), (255, 255, 255), a=0.8)
    tape = np.clip(1 - np.abs(yy - 12) / 7, 0, 1) ** 0.3 * (np.abs(xx - W / 2) < 26)
    R.paint(img, tape, (250, 246, 220), a=0.55)
    return img


# =============================================================================
# THEME
# =============================================================================

def _glyph_sheet(chars: str, size: int, style: FontStyle, variants: int, cell_w: int, cell_h: int, pad: int,
                 base: float) -> np.ndarray:
    """One row of fixed-size cells per variant; each glyph's pen origin sits at (pad, base) in its cell."""
    sheet = np.zeros((cell_h * variants, cell_w * len(chars)), np.float32)
    for v in range(variants):
        for i, ch in enumerate(chars):
            cell = np.zeros((cell_h, cell_w), np.float32)
            _np_text(cell, ch, pad, base, size, style, _h32(ord(ch), v, size, style.seed))
            sheet[v * cell_h:(v + 1) * cell_h, i * cell_w:(i + 1) * cell_w] = cell
    return _mask_rgba(sheet)


class _GlyphStamps:
    """
    Hand-lettered glyph sprites for a small character set (HUD digits), drawn as tinted stamps.
    Cheaper to build than a full atlas per size, and they scale smoothly for the damage pop.
    """

    def __init__(self, g, chars: str, style: FontStyle, sizes, variants: int = 2):
        self.g = g
        self.chars = chars
        self.style = style
        self.sets = []
        for size in sizes:
            pad = int(size * 0.35) + 3
            cell_w = int(size * 0.85) + 2 * pad
            cell_h = int(size * 1.5) + 2 * pad
            base = pad + size * 1.15
            arr = _c(f"glyphs-{size}-{variants}-{_stable((chars, style))}",
                     lambda: _glyph_sheet(chars, size, style, variants, cell_w, cell_h, pad, base), _glyph_sheet)
            imgs = [[g.upload(np.ascontiguousarray(arr[v * cell_h:(v + 1) * cell_h, i * cell_w:(i + 1) * cell_w]))
                     for i in range(len(chars))] for v in range(variants)]
            self.sets.append((size, imgs, pad, base, cell_w, cell_h))

    def _set(self, size):
        return min(self.sets, key=lambda st: abs(st[0] - size) + (0 if st[0] >= size else 0.5))

    def measure(self, text: str, size: float) -> float:
        sk = size / 10.0
        return sum(glyph_strokes(ch)[0] * sk + self.style.tracking * sk for ch in text) - self.style.tracking * sk

    def draw(self, text: str, x: float, y: float, size: float, color, alpha: float = 1.0, align: str = "left"):
        base_size, imgs, pad, base, cw, ch = self._set(size)
        k = size / base_size
        width = self.measure(text, size)
        if align == "center":
            x -= width / 2
        elif align == "right":
            x -= width
        sk = size / 10.0
        pen = x
        nv = len(imgs)
        for i, c in enumerate(text):
            j = self.chars.find(c)
            if j >= 0:
                img = imgs[(i * 7 + ord(c)) % nv][j]
                self.g.stamp(img, pen + (cw / 2 - pad) * k, y + (ch / 2 - base) * k, color, alpha, scale_x=k,
                             op=X.OP_OVER)
            pen += glyph_strokes(c)[0] * sk + self.style.tracking * sk
        return width


_CORE_FUNCS = None
_SRC_MEMO: Dict[object, bytes] = {}


def _srchash(*funcs) -> str:
    """Per-asset cache tag from the source of the generator and the shared helpers it uses."""
    global _CORE_FUNCS
    if _CORE_FUNCS is None:
        _CORE_FUNCS = (_resample, _wobble, _stroke, _loop_pts, _hatch, _hatch_box, _curve, _tooth, _mul, _over,
                       _np_text, _rgba_from_layers, _mask_rgba, _mix, _on_paper)
    h = hashlib.sha1()
    for f in tuple(funcs) + _CORE_FUNCS:
        src = _SRC_MEMO.get(f)
        if src is None:
            try:
                src = inspect.getsource(f).encode()
            except (OSError, TypeError):
                src = repr(f).encode()
            _SRC_MEMO[f] = src
        h.update(src)
    h.update(repr((PAPER, INK, GRAPHITE, BALLPOINT, RED_PEN, PAD_X, PAD_Y)).encode())
    return h.hexdigest()[:8]


def _c(key: str, builder, *funcs):
    return cached(f"sketch-{key}-{_srchash(*funcs)}", builder, [])


@register
class SketchTheme(Theme):
    name = "sketch"
    title = "Sketchbook"

    palette = {
        "bg": PAPER, "panel": PAPER_HI, "panel_edge": GRAPHITE,
        "text": INK, "text_dim": GRAPHITE_TXT, "accent": RED_PEN,
        "accent2": BALLPOINT, "good": GREEN_PEN, "warn": (226, 128, 20),
        "bad": RED_PEN, "p1": DEFAULT_COLORS[0], "p2": DEFAULT_COLORS[1],
    }

    # ------------------------------------------------------------------ build

    def build(self):
        g = self.g
        self.fonts["ui"] = Face(g, FontStyle(weight=1.3, tracking=1.5, jitter=0.3, variants=3, taper=0.5, seed=11))
        self.fonts["title"] = Face(g, FontStyle(weight=1.8, tracking=1.9, jitter=0.4, variants=2, taper=0.55,
                                                seed=5))
        self.fonts["small"] = Face(g, FontStyle(weight=1.35, tracking=1.3, jitter=0.12, variants=2, taper=0.3,
                                                seed=19))
        self.fonts["digits"] = Face(g, FontStyle(weight=2.3, tracking=0.9, jitter=0.28, variants=2, taper=0.25,
                                                 slant=0.06, seed=3))
        self.fonts["ui"].prewarm(18, 22)
        self.fonts["small"].prewarm(11, 13)
        self.digits = _GlyphStamps(g, "0123456789%:-+", DIGIT_HAND, (30, 52), variants=2)

        def up(key, fn, *args):
            return g.upload(_c(key, lambda: fn(*args), fn))

        self.spr_bubble = [up(f"bubble{i}", _bubble_mask, 10 + i) for i in range(3)]
        self.spr_sring = [up(f"sring{i}", _scribble_ring, 20 + i) for i in range(3)]
        self.spr_ball = [up(f"ball{i}", _scribble_ball, 30 + i) for i in range(2)]
        self.spr_inkring = [up(f"inkring{i}", _ink_ring, 40 + i) for i in range(3)]
        self.spr_star = [up(f"sstar{i}", _small_star, 50 + i) for i in range(3)]
        self.spr_splat = [up(f"splat{i}", _splat, 60 + i) for i in range(2)]
        self.spr_puff = [up(f"puff{i}", _puff, 70 + i) for i in range(2)]
        self.spr_smudge = up("smudge", _smudge, 80)
        self.spr_blob = up("blob", _blob, 81)
        self.spr_crown = up("crown", _crown, 82)
        self.spr_squiggle = [up(f"squig{i}", _squiggle, 90 + i) for i in range(3)]
        self.spr_bar = up("barframe", _bar_frame, 90, 14, 85)
        self.spr_timer = up("timertag", _timer_tag, 86)
        words = {1: ("pow!", "bif!"), 2: ("BAM!", "WHAM!"), 3: ("KRAK!", "KA-POW!")}
        fills = {1: (255, 226, 70), 2: (255, 160, 50), 3: (240, 70, 60)}
        radii = {1: 50, 2: 60, 3: 72}
        self.spr_burst = {}
        for tier in (1, 2, 3):
            for k, wd in enumerate(words[tier]):
                self.spr_burst[(tier, k)] = up(f"burst-{tier}-{k}", _burst, wd, fills[tier], 200 + tier * 7 + k,
                                               radii[tier])
        self.spr_burst_blank = up("burst-blank", _burst, "", (255, 226, 70), 233, 80)
        self._hud_cards: Dict[tuple, object] = {}
        self._heads: Dict[tuple, object] = {}
        self._jcache: Dict[tuple, list] = {}
        self._words: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._edges: Dict[tuple, list] = {}
        self._ui_frame = 0
        self.menu_page = None
        self._mtex: Dict[tuple, object] = {}

    def build_stage(self, stage) -> dict:
        g = self.g
        W, H = self.w + 2 * PAD_X, self.h + 2 * PAD_Y
        kind = "space" if "final" in stage.name.lower() else "ruled"
        a = {"kind": kind, "pad": (PAD_X, PAD_Y), "mid": None, "arrays": {}}
        if kind == "ruled":
            a["far"] = self._up_keep(a, "far", _c(f"bf-far-{W}x{H}", lambda: _bf_far(W, H), _bf_far, _paper, _ruled,
                                                  _doodle_star, _doodle_spiral, _doodle_arrow, _doodle_smiley,
                                                  _doodle_tape, _pine, _lollipop, _star_pts, _text_width))
            self_clouds = []
            for i, (cw, ch) in enumerate(((170, 74), (200, 86), (150, 66))):
                self_clouds.append([g.upload(_c(f"cloud{i}-{v}", lambda cw=cw, ch=ch, s=i * 10 + v:
                                                _cloud_sprite(cw, ch, 300 + s), _cloud_sprite)) for v in range(2)])
            a["clouds"] = self_clouds
            a["rays"] = g.upload(_c("sunrays", lambda: _sun_rays(5), _sun_rays))
        else:
            a["far"] = self._up_keep(a, "far", _c(f"fd-far-{W}x{H}", lambda: _fd_far(W, H), _fd_far, _paper,
                                                  _doodle_star, _star_pts, _text_width))
            a["rocket"] = g.upload(_c("rocket", lambda: _rocket_sprite(7), _rocket_sprite))
            a["flame"] = [g.upload(_c(f"flame{i}", lambda i=i: _flame_sprite(8 + i), _flame_sprite)) for i in range(3)]
        plats = {}
        for p in [stage.main_platform] + stage.platforms:
            key = (int(p.width), int(p.height), not p.passthrough)
            if key in plats:
                continue
            w, h, solid = key
            if solid and kind == "ruled":
                img = self._up_keep(a, key, _c(f"plat-bf-{w}x{h}", lambda: _plat_bf_main(w, h), _plat_bf_main,
                                                _rock_profile, _facets, _rock_hatch))
                plats[key] = (img, 34, 34)
            elif solid:
                img = self._up_keep(a, key, _c(f"plat-fd-{w}x{h}", lambda: _plat_fd_main(w, h), _plat_fd_main,
                                                _rock_profile, _facets, _rock_hatch))
                plats[key] = (img, 50, 50)
            else:
                img = self._up_keep(a, key, _c(f"plat-strip-{w}", lambda: _plat_strip(w, 500 + w), _plat_strip))
                plats[key] = (img, 16, 16)
        a["plats"] = plats
        return a

    def _up_keep(self, a, key, arr):
        a["arrays"][key] = arr
        return self.g.upload(arr)

    def _thumb(self, stage, a):
        """320x180 area-averaged snapshot of the page with its platforms (for stage select)."""
        img = a.get("thumb")
        if img is not None:
            return img
        arrs = a["arrays"]
        pages = arrs["far"][PAD_Y:PAD_Y + self.h, PAD_X:PAD_X + self.w].astype(np.float32) / 255.0
        for p in [stage.main_platform] + stage.platforms:
            key = (int(p.width), int(p.height), not p.passthrough)
            ent = a["plats"].get(key)
            if ent is None or key not in arrs:
                continue
            _, mx, my = ent
            R.composite(pages, arrs[key].astype(np.float32) / 255.0, int(p.x - mx), int(p.y - my))
        k = 4
        h4, w4 = self.h // k, self.w // k
        small = pages[:h4 * k, :w4 * k].reshape(h4, k, w4, k, 4).mean(axis=(1, 3))
        img = a["thumb"] = self.g.upload(small)
        a["arrays"] = {}
        return img

    # ------------------------------------------------------------------ colors

    def character_color(self, fighter) -> RGB:
        ct = fighter.stats.character_type
        if ct in CHAR_COLORS:
            return CHAR_COLORS[ct]
        return DEFAULT_COLORS[fighter.player_id % 2]

    def color_for_name(self, name: Optional[str], idx: int = 0) -> RGB:
        return CHAR_COLORS.get(name or "", DEFAULT_COLORS[idx % 2])

    def _marker(self, s_mark: Shape, color, alpha: float = 1.0):
        """Fill a marker shape with the streaky highlighter texture (solid fallback when faded)."""
        if alpha < 0.99:
            self.g.fill_shape(s_mark, _on_paper(color), 0.72 * alpha)
            return
        tex = self._mtex.get(color)
        if tex is None:
            tex = self._mtex[color] = self.g.upload(_c(f"mtex-{color}", lambda: _marker_tex(color, 5), _marker_tex))
            self.g.set_repeat(tex)
        self.g.fill_shape(s_mark, color, 1.0, src=tex)

    # ------------------------------------------------------------------ jitter

    def _jit(self, key: int, tick: int, n: int = 72) -> List[float]:
        k = (key, tick)
        v = self._jcache.get(k)
        if v is None:
            if len(self._jcache) > 96:
                self._jcache.clear()
            v = self._jcache[k] = (np.random.default_rng(_h32(key, tick)).random(n) * 2 - 1).tolist()
        return v

    # ------------------------------------------------------------------ stage

    def draw_background(self, game_state, stage, camera, shake):
        a = self.stage_assets(stage)
        g = self.g
        f = self.fx.frame
        sx, sy = shake
        g.blit(a["far"], -PAD_X + sx, -PAD_Y + sy, op=X.OP_SRC)
        self._scene_life(a, f, sx, sy)

    def _scene_life(self, a, f, sx, sy):
        g = self.g
        if a["kind"] == "ruled":
            rays = a["rays"]
            g.blit_xform(rays, 1162 + sx, 88 + sy, 1.0, angle=f * 0.004)
            for i, (bx, by) in enumerate(((420, 150), (800, 108), (1000, 222))):
                imgs = a["clouds"][i]
                img = imgs[(f // 18 + i) % 2]
                x = bx + math.sin(f * 0.0035 + i * 2.1) * 26
                g.blit(img, x - img.w / 2 + sx, by - img.h / 2 + sy)
            s = Shape()
            for i in range(3):
                bx = (230 + f * 0.35 + i * 46) % 1500 - 110
                by = 205 + i * 16 + math.sin(f * 0.02 + i * 2) * 8
                flap = math.sin(f * 0.22 + i * 1.3)
                wy = 3 + 4 * flap
                sz = 7 - i
                s.segment(bx - sz, by - wy, bx, by, 0.6, 1.7)
                s.segment(bx, by, bx + sz, by - wy, 1.7, 0.6)
            g.fill_shape(s, GRAPHITE, 0.85)
        else:
            bob = math.sin(f * 0.05) * 4
            rx, ry = 205 + sx, 196 + bob + sy
            fl = a["flame"][(f // 4) % 3]
            g.blit(fl, rx - 30 - fl.w / 2, ry + 24 - fl.h / 2)
            g.blit(a["rocket"], rx - a["rocket"].w / 2, ry - a["rocket"].h / 2)
            cyc, ph = divmod(f, 420)
            if ph < 46:
                t = ph / 46.0
                hx = _h32(cyc, 9)
                x0, y0 = 300 + hx % 600, 90 + (hx >> 10) % 120
                ex, ey = x0 + 260 * t, y0 + 90 * t
                s = Shape()
                for k in range(5):
                    a0, a1 = max(0.0, t - 0.32 + k * 0.064), max(0.0, t - 0.32 + k * 0.064 + 0.04)
                    s.segment(x0 + 260 * a0 + sx, y0 + 90 * a0 + sy, x0 + 260 * a1 + sx, y0 + 90 * a1 + sy,
                              0.6 + k * 0.4, 0.9 + k * 0.4)
                g.fill_shape(s, BALLPOINT, 0.8 * (1 - t))
                g.stamp(self.spr_star[cyc % 3], ex + sx, ey + sy, (230, 150, 20), 1 - t * t, scale_x=0.32,
                        angle=f * 0.2, op=X.OP_OVER)
            for i, (x, y) in enumerate(((330, 260), (760, 205), (590, 330), (1210, 520))):
                tw = 0.5 + 0.5 * math.sin(f * 0.07 + i * 1.9)
                g.stamp(self.spr_star[i % 3], x + sx, y + sy, BALLPOINT, 0.55 * tw, scale_x=0.25 + 0.12 * tw,
                        angle=i, op=X.OP_OVER)

    def draw_platform(self, plat, stage, shake):
        a = self.stage_assets(stage)
        ent = a["plats"].get((int(plat.width), int(plat.height), not plat.passthrough))
        if ent is None:
            return super().draw_platform(plat, stage, shake)
        img, mx, my = ent
        self.g.blit(img, plat.x - mx + shake[0], plat.y - my + shake[1])

    def draw_foreground(self, game_state, stage, camera, shake):
        f = self.fx.frame
        for ko in self.fx.ko_flashes:
            t = (f - ko["frame"]) / 40.0
            self.g.fill_rect(0, 0, self.w, self.h, (255, 255, 255), 0.45 * (1 - t) ** 2)

    # ------------------------------------------------------------------ fighters

    def _taper_line(self, s: Shape, pts, w0: float, w1: float):
        n = len(pts)
        cum = [0.0]
        for i in range(n - 1):
            cum.append(cum[-1] + math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
        L = cum[-1] or 1.0
        ws = []
        for c in cum:
            t = c / L
            w = w0 + (w1 - w0) * t
            if t < 0.1:
                w *= 0.55 + 4.5 * t
            ws.append(w)
        ws[-1] *= 0.35
        for i in range(n - 1):
            s.segment(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], ws[i], ws[i + 1], caps=False)
        for i in range(n - 1):
            s.circle(pts[i][0], pts[i][1], ws[i] / 2, 8)

    def _figure(self, sk, J, scale: float = 1.0, ink_w: float = 3.5, eye: str = "dot", marker_w: float = 11.0):
        """Ink + marker shapes for a skeleton with jitter table J (one boil tick)."""
        s_ink, s_mark = Shape(), Shape()
        ji = 2
        amp = 1.4 * scale
        mox = (2.0 + J[0] * 1.6) * scale
        moy = (1.8 + J[1] * 1.3) * scale
        for _, pts in sk.chains():
            s_mark.polyline([(x + mox, y + moy) for x, y in pts], marker_w * scale)
            dense = [(pts[0][0] + J[ji] * 0.7 * scale, pts[0][1] + J[ji + 1] * 0.7 * scale)]
            ji += 2
            for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                dx, dy = bx - ax, by - ay
                L = math.hypot(dx, dy) or 1.0
                nx, ny = -dy / L, dx / L
                bend = J[ji] * amp * min(1.0, L / 26)
                ji += 1
                for t, k in ((0.34, 0.87), (0.67, 0.87)):
                    dense.append((ax + dx * t + nx * bend * k, ay + dy * t + ny * bend * k))
                dense.append((bx + J[ji] * 0.6 * scale, by + J[ji + 1] * 0.6 * scale))
                ji += 2
            (px, py), (ex, ey) = dense[-2], dense[-1]
            d = math.hypot(ex - px, ey - py) or 1.0
            o = (2.0 + 2.5 * abs(J[ji])) * scale
            ji += 1
            dense.append((ex + (ex - px) / d * o, ey + (ey - py) / d * o))
            self._taper_line(s_ink, dense, ink_w * 1.15 * scale, ink_w * 0.6 * scale)
        hx, hy = sk.head
        r = sk.head_r
        s_mark.circle(hx + mox, hy + moy, r + 1.0 * scale)
        a0 = J[ji] * math.pi
        turns = 1.12 + 0.06 * J[ji + 1]
        ph = J[ji + 2] * 3
        n = 22
        prev = None
        for i in range(n):
            t = i / (n - 1)
            ang = a0 + t * turns * 2 * math.pi
            rr = r * (1 + 0.05 * math.sin(2 * ang + ph)) + t * 0.8 * scale
            p = (hx + math.cos(ang) * rr, hy + math.sin(ang) * rr * 0.97)
            w = ink_w * scale * (0.35 + 0.7 * math.sin(math.pi * t) ** 0.5)
            if prev is not None:
                s_ink.segment(prev[0][0], prev[0][1], p[0], p[1], prev[1], w, caps=False)
                s_ink.circle(p[0], p[1], w / 2, 8)
            prev = (p, w)
        f = sk.facing
        ex, ey = hx + f * r * 0.42, hy - r * 0.12
        if eye == "x":
            e = 2.6 * scale
            s_ink.segment(ex - e, ey - e, ex + e, ey + e, 1.8 * scale)
            s_ink.segment(ex - e, ey + e, ex + e, ey - e, 1.8 * scale)
        else:
            s_ink.circle(ex, ey, 2.0 * scale, 10)
        return s_ink, s_mark

    def draw_fighter_aura(self, fighter, sk, look: FighterLook):
        g = self.g
        sx, sy = self.r.shake_offset
        if fighter.grounded and not look.blink:
            fy = fighter.position.y + 40 + sy
            g.stamp(self.spr_smudge, fighter.position.x + sx, fy + 1, GRAPHITE, 0.3, scale_x=0.55, scale_y=0.5,
                    op=X.OP_OVER)
        if look.armor:
            cx, cy = sk.center
            f = self.fx.frame
            pulse = 0.75 + 0.25 * math.sin(f * 0.4)
            g.stamp(self.spr_blob, cx, cy - 10, _on_paper((255, 206, 60)), 0.5 * pulse, scale_x=1.0, scale_y=1.3,
                    angle=f * 0.05, op=X.OP_OVER)
            s = Shape()
            for _, pts in sk.chains():
                s.polyline(pts, 22)
            s.circle(sk.head[0], sk.head[1], sk.head_r + 10)
            g.fill_shape(s, _on_paper((255, 190, 30)), 0.6 * pulse)
            for k in range(3):
                ang = f * 0.09 + k * 2.09
                g.stamp(self.spr_star[k], cx + math.cos(ang) * 34, cy - 10 + math.sin(ang) * 44, (200, 130, 10),
                        0.9, scale_x=0.3, angle=f * 0.1, op=X.OP_OVER)

    def draw_fighter(self, fighter, sk, look: FighterLook):
        g = self.g
        f = self.fx.frame
        pid = fighter.player_id
        J = self._jit(pid + 1, f // 4)
        a = 0.3 if look.blink else look.alpha
        ink = INK
        if look.flash > 0:
            ink = _mix(INK, RED_PEN, look.flash)
        s_ink, s_mark = self._figure(sk, J, eye="x" if look.hitstun else "dot")
        if look.flash > 0:
            g.fill_shape(s_mark, _mix(_on_paper(look.color), (255, 120, 120), look.flash * 0.5), 0.72 * a)
        else:
            self._marker(s_mark, look.color, a)
        g.fill_shape(s_ink, ink, a)
        if look.hitstun or look.shield_stun:
            hx, hy = sk.head
            s = Shape()
            for side in (-1, 1):
                cx = hx + side * (sk.head_r + 9)
                for k in range(2):
                    ox = side * k * 5
                    s.segment(cx + ox, hy - 7 + J[k + side + 3] * 1.5, cx + ox + side * 2, hy + 5, 1.8, 0.6)
            g.fill_shape(s, ink, 0.85 * a)
        if look.shield_stun:
            hx, hy = sk.head
            for k in range(3):
                ang = f * 0.15 + k * 2.09
                g.stamp(self.spr_star[k], hx + math.cos(ang) * 20, hy - 18 + math.sin(ang) * 6, (230, 150, 20), 0.95,
                        scale_x=0.32, angle=f * 0.1, op=X.OP_OVER)

    def draw_afterimage(self, fighter, sk, dodge_type, alpha: float):
        """Eraser smudge: a gray ghost where the figure was drawn and rubbed out."""
        s, s2 = Shape(), Shape()
        for _, pts in sk.chains():
            s.polyline(pts, 9)
            s2.polyline([(x + 2.5, y + 1) for x, y in pts], 2.2)
        s.circle(sk.head[0], sk.head[1], sk.head_r + 4)
        s2.ring(sk.head[0] + 2.5, sk.head[1] + 1, sk.head_r, 2.0)
        self.g.fill_shape(s, (196, 190, 178), alpha * 0.55)
        self.g.fill_shape(s2, GRAPHITE, alpha * 0.32)

    def draw_knockback_trail(self, fighter, pts):
        n = len(pts)
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 8:
            return
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        J = self._jit(1000 + fighter.player_id, self.fx.frame // 3, 24)
        s = Shape()
        for k in range(6):
            off = (k - 2.5) * 10 + J[k] * 3
            start = 0.1 + 0.3 * abs(J[k + 6])
            end = 0.82 + 0.1 * J[k + 12]
            ax, ay = x0 + dx * start + nx * off, y0 + dy * start + ny * off
            bx, by = x0 + dx * end + nx * off - ux * 14, y0 + dy * end + ny * off - uy * 14
            s.segment(ax, ay, bx, by, 0.6, 2.6 if abs(off) < 20 else 1.8)
        self.g.fill_shape(s, INK, 0.65 * min(1.0, n / 6))
        col = _on_paper(self.character_color(fighter))
        m = Shape()
        m.segment(x0 + dx * 0.3, y0 + dy * 0.3, x1 - ux * 10, y1 - uy * 10, 3, 16)
        self.g.fill_shape(m, col, 0.35)

    def draw_shield(self, fighter, sk):
        hp = max(0.0, fighter.shield_health) / 100.0
        r = 28 + 26 * hp
        col = BALLPOINT if hp > 0.5 else (214, 120, 20) if hp > 0.25 else RED_PEN
        cx, cy = sk.center
        cy -= 4
        f = self.fx.frame
        spr = self.spr_bubble[(f // 5) % 3]
        sc = (r * 2) / (spr.w - 16)
        self.g.stamp(self.spr_blob, cx, cy, _on_paper(_mix(col, (255, 255, 255), 0.55)), 0.35, scale_x=sc * 1.05,
                     op=X.OP_OVER)
        self.g.stamp(spr, cx, cy, col, 0.9, scale_x=sc, op=X.OP_OVER)

    def draw_counter(self, fighter, sk):
        cx, cy = sk.center
        f = self.fx.frame
        for k in range(2):
            spr = self.spr_sring[(f // 4 + k) % 3]
            rr = 48 + 6 * math.sin(f * 0.5 + k * 3) + k * 8
            self.g.stamp(spr, cx, cy, BALLPOINT, 0.85 - k * 0.25, scale_x=rr * 2 / (spr.w - 20),
                         angle=f * (0.08 if k else -0.06), op=X.OP_OVER)

    def _hex_pts(self, cx, cy, r, rot, J, amp=2.0):
        pts = []
        for i in range(7):
            a = rot + i * math.pi / 3
            j = J[i % 6]
            pts.append((cx + math.cos(a) * (r + j * amp), cy + math.sin(a) * (r + j * amp)))
        return pts

    def _sketch_poly(self, s: Shape, pts, w, J, over=4.0):
        """Closed polygon drawn as separate overshooting edge strokes."""
        n = len(pts) - 1
        for i in range(n):
            (ax, ay), (bx, by) = pts[i], pts[i + 1]
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy) or 1
            ux, uy = dx / L, dy / L
            o1 = over * (0.4 + 0.6 * abs(J[(i + 3) % len(J)]))
            o2 = over * (0.4 + 0.6 * abs(J[(i + 5) % len(J)]))
            s.segment(ax - ux * o1, ay - uy * o1, bx + ux * o2, by + uy * o2, w, w * 0.8)

    def draw_reflector(self, fighter, sk):
        cx, cy = sk.center
        f = self.fx.frame
        J = self._jit(2000 + fighter.player_id, f // 3, 12)
        pts = self._hex_pts(cx, cy, 52, f * 0.03, J)
        self.g.polygon(pts[:-1], _on_paper((120, 230, 235)), 0.22)
        s = Shape()
        self._sketch_poly(s, pts, 2.6, J)
        pts2 = self._hex_pts(cx + 1.5, cy - 1, 49, f * 0.03 + 0.02, J[6:] + J[:6], 1.5)
        self._sketch_poly(s, pts2, 1.3, J[3:] + J[:3])
        self.g.fill_shape(s, (16, 112, 128), 0.9)

    # ------------------------------------------------------------------ projectiles & effects

    def draw_projectile(self, p, trail, shake):
        g = self.g
        col = tuple(int(c) for c in p.color)
        pc = _mix(col, INK, 0.3)
        x, y = p.position.x + shake[0], p.position.y + shake[1]
        f = self.fx.frame
        n = len(trail)
        beam = p.width > p.height * 1.8
        if n > 2 and not beam:
            s = Shape()
            for i in range(0, n - 2, 2):
                (ax, ay), (bx, by) = trail[i], trail[i + 1]
                t = (i + 1) / n
                s.segment(ax + shake[0], ay + shake[1], bx + shake[0], by + shake[1], 1.2 + 2.4 * t)
            for k, i in enumerate((n // 3, (2 * n) // 3)):
                tx, ty = trail[i]
                s.ring(tx + shake[0], ty + shake[1] + (k * 2 - 1) * 5, 2.5 + k * 1.5, 1.4, 10)
            g.fill_shape(s, INK, 0.7)
        if beam:
            hw, hh = p.width / 2 + 6, p.height / 2 + 4
            J = self._jit(id(p) & 0xFFFF, f // 3, 24)
            d = -1 if p.velocity.x > 0 else 1
            m = Shape()
            m.segment(x + d * (hw + 40), y + 1, x - d * hw, y + 1, hh * 0.6, hh * 2.4)
            g.fill_shape(m, _on_paper(col), 0.55)
            m = Shape()
            k = 10
            pts = []
            for i in range(k + 1):
                px_ = x - hw + 2 * hw * i / k
                pts.append((px_, y + (hh if i % 2 else -hh) * (0.75 + 0.2 * J[i])))
            m.polyline(pts, 3.4)
            g.fill_shape(m, _on_paper(pc), 0.95)
            s = Shape()
            box = [(x - hw, y - hh + J[0]), (x + hw, y - hh + J[1]), (x + hw - d * 9, y + J[2]),
                   (x + hw, y + hh + J[3]), (x - hw, y + hh + J[4]), (x - hw + d * 9, y + J[5]),
                   (x - hw, y - hh + J[0])]
            if d > 0:
                box = [(2 * x - bx, by) for bx, by in box]
            self._sketch_poly(s, box, 2.2, J, 5)
            for i in range(4):
                yy = y + (i - 1.5) * hh * 0.6
                L = 26 + 22 * abs(J[i + 6])
                s.segment(x + d * (hw + 6), yy, x + d * (hw + 6 + L), yy, 2.0, 0.5)
            g.fill_shape(s, INK, 0.92)
        else:
            sc = p.width / 30.0
            pulse = 1 + 0.08 * math.sin(f * 0.5)
            g.stamp(self.spr_blob, x, y, _on_paper(col), 0.8, scale_x=sc * 0.42 * pulse, op=X.OP_OVER)
            g.stamp(self.spr_ball[(f // 3) % 2], x, y, pc, 0.9, scale_x=sc * 0.95, angle=f * 0.25, op=X.OP_OVER)
            g.stamp(self.spr_inkring[(f // 4) % 3], x, y, INK, 0.95, scale_x=sc * 0.95 * pulse, op=X.OP_OVER)

    @staticmethod
    def _tier(color) -> int:
        r, gg, b = color
        if r > 240 and gg > 240 and b > 240:
            return 0
        if gg > 220:
            return 1
        if gg > 140:
            return 2
        return 3

    def draw_effect(self, e, shake):
        from renderer import EffectType
        g = self.g
        x, y = e.x + shake[0], e.y + shake[1]
        t = e.progress
        if e.effect_type == EffectType.HIT_SPARK:
            tier = self._tier(e.color)
            var = _h32(int(e.x * 3), int(e.y * 7), e.duration) % 2
            alpha = 1.0 if t < 0.6 else max(0.0, (1 - t) / 0.4)
            if tier == 0:
                sc = 0.55 + 0.6 * min(1.0, t * 3)
                g.stamp(self.spr_star[var], x, y, INK, alpha, scale_x=sc, angle=math.radians(e.angle), op=X.OP_OVER)
                return
            k = 0.8 + 0.25 * min(1.5, e.intensity)
            if t < 0.25:
                pop = 0.5 + 0.7 * (t / 0.25)
            elif t < 0.45:
                pop = 1.2 - 0.2 * ((t - 0.25) / 0.2)
            else:
                pop = 1.0
            ang = math.radians(e.angle)
            if tier >= 2:
                spl = self.spr_splat[var]
                g.stamp(spl, x + math.cos(ang) * 26, y - math.sin(ang) * 26, INK, 0.9 * alpha,
                        scale_x=(0.5 + 0.4 * min(1, t * 4)) * k, angle=var * 1.7 + e.angle, op=X.OP_OVER)
            img = self.spr_burst[(tier, var)]
            tilt = ((_h32(int(e.x), int(e.y)) % 100) / 100 - 0.5) * 0.5
            g.blit_xform(img, x, y, pop * k, angle=tilt, alpha=alpha)
        elif e.effect_type == EffectType.LANDING_DUST:
            fy = y + 40
            a = min(1.0, (1 - t) * 1.4)
            for side in (-1, 1):
                spr = self.spr_puff[(side + 1) // 2]
                sc = 0.75 + 0.5 * t
                g.stamp(spr, x + side * (18 + 36 * t), fy - 12 - 9 * t, (64, 66, 76), a,
                        scale_x=side * sc, scale_y=sc, op=X.OP_OVER)
            s = Shape()
            for side in (-1, 1):
                for k in range(2):
                    bx = x + side * (30 + 50 * t + k * 8)
                    s.segment(bx, fy - 2 - k * 6, bx + side * (10 - k * 3), fy - 2 - k * 6 - 2, 1.6, 0.5)
            g.fill_shape(s, GRAPHITE, a)
        elif e.effect_type == EffectType.REFLECTOR_HEX:
            J = self._jit(3000, int(e.x) // 3, 12)
            r = 26 + 54 * t
            s = Shape()
            self._sketch_poly(s, self._hex_pts(x, y, r, t * 0.5, J), 2.6 * (1 - t) + 0.8, J)
            g.fill_shape(s, (16, 112, 128), 1 - t)
            for k in range(3):
                a = k * 2.1 + t * 2
                g.stamp(self.spr_star[k], x + math.cos(a) * r * 0.8, y + math.sin(a) * r * 0.8, (16, 112, 128),
                        1 - t, scale_x=0.3, op=X.OP_OVER)
        elif e.effect_type == EffectType.FRAME_ADVANTAGE:
            adv = int(e.intensity)
            label = f"+{adv}" if adv > 0 else str(adv)
            col = GREEN_PEN if adv > 0 else RED_PEN if adv < 0 else GRAPHITE
            self.text(x, y - 30 - 20 * t, label, 22, col, align="center", alpha=1 - t * t)

    # ------------------------------------------------------------------ HUD

    def _damage_color(self, dmg: float) -> RGB:
        stops = [(0, INK), (50, (70, 30, 50)), (100, (150, 26, 36)), (150, (206, 20, 30))]
        for (d0, c0), (d1, c1) in zip(stops, stops[1:]):
            if dmg <= d1:
                return _mix(c0, c1, (dmg - d0) / (d1 - d0))
        return stops[-1][1]

    def _card(self, col, right: bool, cw: int, ch: int):
        key = (col, right, cw, ch)
        img = self._hud_cards.get(key)
        if img is None:
            ang = 1.2 if right else -1.2
            img = self._hud_cards[key] = self.g.upload(
                _c(f"sticky-{col}-{int(right)}-{cw}x{ch}", lambda: _sticky(cw, ch, col, ang, 7 + int(right)), _sticky))
        return img

    def _head(self, col):
        img = self._heads.get(col)
        if img is None:
            img = self._heads[col] = self.g.upload(_c(f"head-{col}", lambda: _stock_head(col, 3), _stock_head))
        return img

    def _scribble_fill(self, s: Shape, x, y, w, h, frac, J, step=4.5, width=2.6):
        """Zigzag 'colored in' fill from x to x + w*frac."""
        if frac <= 0:
            return
        ex = x + w * min(1.0, frac)
        px = x + 1
        top = True
        prev = (px, y + h - 1)
        i = 0
        while px < ex:
            px = min(ex, px + step)
            pt = (px + J[i % len(J)] * 0.8, (y + 1.5) if top else (y + h - 1.5))
            s.segment(prev[0], prev[1], pt[0], pt[1], width, caps=False)
            s.circle(pt[0], pt[1], width / 2, 6)
            prev = pt
            top = not top
            i += 1

    def draw_hud(self, game_state):
        g = self.g
        f = self.fx.frame
        for i, fighter in enumerate(game_state.fighters[:2]):
            col = self.character_color(fighter)
            right = i == 1
            cw, ch = 260, 96
            x = self.w - 30 - cw if right else 30
            y = self.h - ch - 20
            card = self._card(col, right, cw, ch)
            g.blit(card, x - 14, y - 14)
            name = fighter.stats.name
            self.text(x + 16, y + 27, name, 18, INK)
            img = self._head(col)
            ns = max(0, fighter.stocks)
            if ns <= 4:
                for s in range(ns):
                    g.blit(img, x + 16 + s * 22, y + 37)
            else:
                g.blit(img, x + 16, y + 37)
                self.text(x + 44, y + 55, f"x{ns}", 18, INK)
            hp = max(0.0, fighter.shield_health) / 100.0
            bx, by, bw, bh = x + 18, y + ch - 24, 90, 14
            J = self._jit(4000 + i, f // 6, 16)
            if hp > 0:
                sc = col if hp > 0.5 else (240, 150, 30) if hp > 0.25 else (235, 60, 60)
                s = Shape()
                self._scribble_fill(s, bx, by, bw, bh, hp, J)
                g.fill_shape(s, _on_paper(sc), 0.85)
            g.stamp(self.spr_bar, bx + bw / 2, by + bh / 2, INK, 0.85, op=X.OP_OVER)
            dmg = fighter.damage_percent
            anim = game_state.damage_anim_frames[i] if i < len(game_state.damage_anim_frames) else 0
            pop = 1.0 + 0.16 * (anim / 20.0) ** 2
            jig = math.sin(f * 2.3) * 3 * anim / 20.0
            dcol = self._damage_color(dmg)
            size = int(48 * pop)
            tx = x + cw - 14 + jig
            ty = y + ch - 22
            dig = self.digits
            if dmg >= 100:
                wdt = dig.measure(f"{int(dmg)}%", size)
                s = Shape()
                s.segment(tx - wdt - 2, ty + 7, tx + 4, ty + 5, 2.4, 1.0)
                s.segment(tx - wdt + 6, ty + 11, tx - 4, ty + 10, 1.6, 0.6)
                g.fill_shape(s, RED_PEN, 0.85)
            dig.draw(f"{int(dmg)}%", tx, ty, size, dcol, align="right")
            combo = game_state.combo_counts[i] if i < len(game_state.combo_counts) else 0
            if combo >= 2:
                cx = x + cw / 2
                wob = math.sin(f * 0.3) * 1.5
                wdt = self.text(cx, y - 18 + wob, f"{combo} hits!", 22, RED_PEN, align="center")
                s = Shape()
                Jc = self._jit(5000 + i, f // 4, 8)
                s.segment(cx - wdt / 2 - 4, y - 10 + Jc[0], cx + wdt / 2 + 6, y - 11 + Jc[1], 2.2, 0.8)
                g.fill_shape(s, RED_PEN, 0.8)
        t = game_state.get_time_display()
        if t:
            low = game_state.time_remaining < 600
            col = RED_PEN if low and (f // 15) % 2 == 0 else INK
            tag = self.spr_timer
            g.blit(tag, self.w / 2 - tag.w / 2, 18)
            self.digits.draw(t, self.w / 2, 18 + 46, 30, col, align="center")

    # ------------------------------------------------------------------ text sprites

    def _word(self, text: str, size: int, ink=INK, marker=None, style=TITLE_HAND, double: bool = True):
        """Big hand-lettered text rendered once with numpy (marker shadow + double ink pass)."""
        key = (text, size, ink, marker, style.seed, double)
        ent = self._words.get(key)
        if ent is not None:
            self._words.move_to_end(key)
            return ent
        if len(self._words) > 48:
            _, old = self._words.popitem(last=False)
            self.g.free(old[0])
        pad = int(size * 0.3) + 6
        tw = _text_width(text, size, style)
        W, H = int(tw + 2 * pad + size * 0.2), int(size * 1.5 + 2 * pad)
        base = pad + size * 1.1

        def build():
            m1 = np.zeros((H, W), np.float32)
            _np_text(m1, text, pad, base, size, style, _h32(len(text), size, style.seed))
            img = R.blank(W, H)
            if marker is not None:
                sh = int(max(2, min(5, size * 0.055)))
                mm = np.clip(R.blur(np.roll(np.roll(m1, sh, 0), sh, 1), min(2.5, size * 0.035)) * 3.0, 0, 1)
                R.paint(img, mm, _on_paper(marker), a=0.9)
            if double:
                m2 = np.zeros((H, W), np.float32)
                st2 = FontStyle(weight=style.weight * 0.55, tracking=style.tracking, jitter=style.jitter * 1.3,
                                variants=1, taper=style.taper, slant=style.slant, seed=style.seed + 1)
                off = min(3.0, size * 0.035)
                _np_text(m2, text, pad + off, base - off, size, st2, _h32(size, len(text), 99))
                R.paint(img, m2, ink, a=0.55)
            R.paint(img, m1, ink)
            return img
        arr = _c(f"word-{_stable(key)}-{size}", build, _np_text)
        img = self.g.upload(arr)
        ent = self._words[key] = (img, pad, base, tw)
        return ent

    def _draw_word(self, x, y, text, size, ink=INK, marker=None, align="center", valign="middle", alpha=1.0,
                   scale=1.0, angle=0.0, style=TITLE_HAND):
        if style is TITLE_HAND and size >= 70:
            style = TITLE_BIG
        img, pad, base, tw = self._word(text, size, ink, marker, style)
        if align == "center":
            ax = (pad + tw / 2) / img.w
        elif align == "right":
            ax = (pad + tw) / img.w
        else:
            ax = pad / img.w
        ay_px = base - size / 2 if valign == "middle" else base - size if valign == "top" else base
        if abs(scale - 1) < 1e-3 and abs(angle) < 1e-4:
            self.g.blit(img, x - ax * img.w, y - ay_px, alpha)
        else:
            self.g.blit_xform(img, x, y, scale, angle=angle, alpha=alpha, anchor=(ax, ay_px / img.h))
        return tw * scale

    # ------------------------------------------------------------------ victory

    def draw_victory(self, game_state):
        g = self.g
        f = self.fx.frame
        g.fill_rect(0, 0, self.w, self.h, PAPER, 0.62)
        if game_state.winner is not None:
            fighter = game_state.fighters[game_state.winner]
            col = self.character_color(fighter)
            txt = f"{fighter.stats.name} WINS!"
        else:
            col, txt = (160, 160, 170), "DRAW!"
        cx, cy = self.w / 2, self.h / 2
        img, pad, base, tw = self._word(txt, 72, INK, col, TITLE_BIG)
        J = self._jit(6000, f // 6, 16)
        sw = tw + 60
        g.polygon([(cx - sw / 2, cy - 30 + J[0] * 2), (cx + sw / 2, cy - 36 + J[1] * 2),
                   (cx + sw / 2 + 8, cy + 34 + J[2] * 2), (cx - sw / 2 + 4, cy + 30 + J[3] * 2)],
                  _on_paper(HILITE), 0.75)
        self._draw_word(cx, cy, txt, 72, INK, col)
        s = Shape()
        for k in range(2):
            yy = cy + 52 + k * 10
            pts = [(cx - tw / 2 - 10 + k * 20 + i * (tw + 20 - k * 40) / 8, yy + J[i + k] * 2.2) for i in range(9)]
            self._taper_line(s, pts, 5.0 - k * 1.5, 2.4 - k * 0.6)
        g.fill_shape(s, RED_PEN, 0.9)
        if game_state.winner is not None:
            bob = math.sin(f * 0.12) * 4
            g.blit_xform(self.spr_crown, cx, cy - 92 + bob, 1.0, angle=math.sin(f * 0.07) * 0.08)
        confetti = [(255, 66, 96), (36, 168, 255), (255, 200, 30), (64, 206, 72), (190, 90, 230)]
        for k in range(44):
            hx = _h32(k, 77)
            px = (hx % 1280)
            speed = 1.2 + (hx >> 12) % 100 / 60
            py = ((hx >> 4) % 760 + f * speed) % 780 - 40
            px += math.sin(f * 0.05 + k) * 14
            c = confetti[k % len(confetti)]
            g.stamp(self.spr_squiggle[k % 3], px, py, _mix(c, INK, 0.15), 0.95, scale_x=1.5,
                    angle=f * 0.06 * (1 if k % 2 else -1) + k, op=X.OP_OVER)

    # ------------------------------------------------------------------ UI kit

    def _menu_assets(self):
        if self.menu_page is None:
            W, H = self.w + 2 * PAD_X, self.h + 2 * PAD_Y
            self.menu_page = self.g.upload(_c(f"menu-page-{W}x{H}", lambda: _menu_page(W, H), _menu_page, _paper,
                                              _ruled, _doodle_star, _doodle_spiral, _doodle_smiley, _doodle_tape,
                                              _star_pts))
        return self.menu_page

    def ui_scrim(self, alpha: float):
        """Wash with paper color instead of black, so the page doesn't turn gray."""
        self.g.fill_rect(0, 0, self.w, self.h, (246, 240, 226), alpha * 1.3)

    def ui_backdrop(self, frame: int, variant: str = "menu"):
        g = self.g
        self._ui_frame = frame
        g.blit(self._menu_assets(), -PAD_X, -PAD_Y, op=X.OP_SRC)
        # a doodled stick figure in the corner that waves (line boil + arm swing)
        from poses import POSE_IDLE, skeleton_from_pose
        sk = skeleton_from_pose(POSE_IDLE, 0, 0, -1)
        wave = math.sin(frame * 0.12)
        sk.front_elbow = (sk.neck[0] - 14, sk.neck[1] - 12)
        sk.front_hand = (sk.neck[0] - 18 + wave * 8, sk.neck[1] - 34)
        sk = sk.transformed(scale=0.72, pivot=(0, 0), dx=1240, dy=548)
        self.draw_figure_static(sk, DEFAULT_COLORS[1], 0.75)
        # boiling margin stars
        for i, (x, y) in enumerate(((60, 320), (1220, 260), (1150, 420))):
            g.stamp(self.spr_star[(frame // 8 + i) % 3], x, y, BALLPOINT if i else RED_PEN, 0.8, scale_x=0.38,
                    angle=i * 0.7, op=X.OP_OVER)
        if variant != "menu":
            g.fill_rect(0, 0, self.w, self.h, PAPER, 0.5)

    def _torn_edge(self, w, h):
        key = (int(w), int(h))
        pts = self._edges.get(key)
        if pts is None:
            rng = np.random.default_rng(_h32(int(w), int(h)))
            pts = []
            n = max(4, int(w / 16))
            for i in range(n + 1):
                pts.append((w * i / n, rng.normal() * 0.6))
            m = max(3, int(h / 30))
            for i in range(1, m):
                pts.append((w + rng.normal() * 0.7, h * i / m))
            n2 = max(6, int(w / 7))
            for i in range(n2, -1, -1):
                pts.append((w * i / n2, h + (rng.random() - 0.5) * 4.5))
            for i in range(m - 1, 0, -1):
                pts.append((rng.normal() * 0.7, h * i / m))
            self._edges[key] = pts
        return pts

    def _wobbly_rect(self, x, y, w, h, J, amp=1.2):
        return [(x + J[0] * amp, y + J[1] * amp), (x + w + J[2] * amp, y + J[3] * amp),
                (x + w + J[4] * amp, y + h + J[5] * amp), (x + J[6] * amp, y + h + J[7] * amp),
                (x + J[0] * amp, y + J[1] * amp)]

    def _tape(self, cx, cy, w, h, ang, color, alpha=0.55):
        c, s = math.cos(ang), math.sin(ang)
        pts = []
        for i in range(5):
            v = -h / 2 + h * i / 4
            pts.append((-w / 2 + (1.5 if i % 2 else -1.5), v))
        for i in range(5):
            v = h / 2 - h * i / 4
            pts.append((w / 2 + (1.5 if i % 2 else -1.5), v))
        self.g.polygon([(cx + u * c - v * s, cy + u * s + v * c) for u, v in pts], color, alpha)

    def ui_panel(self, x, y, w, h, emphasis: float = 0.0, accent: Optional[RGB] = None):
        g = self.g
        edge = self._torn_edge(w, h)
        pts = [(x + px, y + py) for px, py in edge]
        g.polygon([(px + 4, py + 5) for px, py in pts], (70, 56, 32), 0.16)
        g.polygon(pts, PAPER_HI)
        if h > 90 and w > 120:
            s = Shape()
            yy = y + 30
            while yy < y + h - 10:
                s.segment(x + 6, yy, x + w - 6, yy, 1.0, caps=False)
                yy += 26
            g.fill_shape(s, RULE_BLUE, 0.28)
        tc = _mix(accent, (255, 255, 255), 0.35) if accent else (236, 226, 190)
        self._tape(x + 14, y + 4, 52, 18, -0.6, tc)
        self._tape(x + w - 14, y + 4, 52, 18, 0.6, tc)
        if emphasis > 0:
            J = self._jit(_h32(int(x), int(y), int(w)), self._ui_frame // 6, 12)
            s = Shape()
            self._sketch_poly(s, self._wobbly_rect(x - 5, y - 5, w + 10, h + 10, J, 2.0), 2.2, J, 8)
            g.fill_shape(s, accent or RED_PEN, 0.85 * min(1.0, emphasis))

    def ui_button(self, x, y, w, h, label, selected: bool, frame: int, accent: Optional[RGB] = None,
                  sublabel: Optional[str] = None, size: int = 20):
        g = self.g
        self._ui_frame = frame
        seed = _h32(int(x), int(y), int(w), int(h))
        J = self._jit(seed, frame // 6 if selected else 0, 16)
        box = self._wobbly_rect(x, y, w, h, J, 1.4)
        g.polygon(box[:-1], PAPER_HI, 0.97)
        cy = y + h / 2 if not (sublabel and selected) else y + h * 0.40
        if selected:
            hl = _hilite(accent) if accent else _on_paper(HILITE)
            lw = min(w - 10, self.measure(label, size) + 26)
            sk = J[8] * 2
            g.polygon([(x + w / 2 - lw / 2 + sk, cy - size * 0.55), (x + w / 2 + lw / 2 + 4, cy - size * 0.6 + J[9]),
                       (x + w / 2 + lw / 2 - sk, cy + size * 0.55), (x + w / 2 - lw / 2 - 4, cy + size * 0.6 + J[10])],
                      hl, 0.7)
        s = Shape()
        self._sketch_poly(s, box, 2.0 if selected else 1.5, J, 6)
        g.fill_shape(s, INK if selected else GRAPHITE, 0.95 if selected else 0.75)
        col = INK if selected else GRAPHITE_TXT
        self.text(x + w / 2, cy, label, size, col, align="center", valign="middle")
        if selected:
            s = Shape()
            rx, ry = w / 2 + 9, h / 2 + 7
            n = 34
            a0 = 2.4 + J[11] * 0.3
            pts = []
            for i in range(n):
                t = i / (n - 1)
                a = a0 + t * 2 * math.pi * 1.1
                k = 1 + 0.03 * math.sin(a * 2 + J[12] * 3) + t * 0.04
                # squircle so it hugs the box corners instead of a big ellipse
                ca, sa = math.cos(a), math.sin(a)
                ex = math.copysign(abs(ca) ** 0.55, ca)
                ey = math.copysign(abs(sa) ** 0.55, sa)
                pts.append((x + w / 2 + ex * rx * k, y + h / 2 + ey * ry * k))
            self._taper_line(s, pts, 2.6, 1.6)
            g.fill_shape(s, RED_PEN, 0.85)
            if sublabel:
                self.text(x + w / 2, y + h * 0.76, sublabel, max(11, int(size * 0.55)), GRAPHITE_TXT, role="small",
                          align="center", valign="middle")

    def ui_title(self, x, y, s, size=48, color=None, frame: int = 0, align="center"):
        col = color or self.palette["accent"]
        if size < 26:
            self.text(x + 2, y + 2, s, size, _on_paper(col), role="title", align=align, valign="middle", alpha=0.8)
            self.text(x, y, s, size, INK, role="title", align=align, valign="middle")
            return
        wob = math.sin(frame * 0.05) * 0.012
        self._draw_word(x, y, s, int(size), INK, col, align=align, angle=wob)

    def ui_bar(self, x, y, w, h, frac: float, color: RGB, bg_alpha: float = 0.35):
        g = self.g
        J = self._jit(_h32(int(x), int(y), int(w)), self._ui_frame // 8, 16)
        if frac > 0:
            s = Shape()
            self._scribble_fill(s, x, y, w, h, frac, J, step=max(3.0, h * 0.4), width=max(2.0, h * 0.22))
            g.fill_shape(s, _on_paper(color), 0.9)
        s = Shape()
        self._sketch_poly(s, self._wobbly_rect(x, y, w, h, J, 0.8), 1.6, J, 4)
        g.fill_shape(s, INK, 0.85)

    def ui_toggle(self, x, y, on: bool, frame: int):
        g = self.g
        J = self._jit(_h32(int(x), int(y), 7), frame // 8 if on else 0, 16)
        bx, by, bs = x + 2, y, 22
        g.polygon(self._wobbly_rect(bx, by, bs, bs, J, 0.8)[:-1], PAPER_HI)
        s = Shape()
        self._sketch_poly(s, self._wobbly_rect(bx, by, bs, bs, J, 0.8), 1.7, J, 4)
        g.fill_shape(s, INK, 0.9)
        if on:
            s = Shape()
            self._taper_line(s, [(bx + 3, by + 11), (bx + 9, by + 19 + J[9])], 3.6, 3.0)
            self._taper_line(s, [(bx + 9, by + 19 + J[9]), (bx + 17, by + 6), (bx + 29 + J[10] * 2, by - 7)], 3.4, 1.6)
            g.fill_shape(s, RED_PEN)
        self.text(bx + bs + 7, by + bs / 2, "on" if on else "off", 13, INK if on else GRAPHITE_TXT, role="small",
                  valign="middle")

    def ui_arrow(self, x, y, direction: int, color: RGB, size: float = 10, alpha: float = 1.0):
        d = direction
        J = self._jit(_h32(int(x), int(y), d), self._ui_frame // 8, 8)
        pts = [(x + d * size, y + J[0]), (x - d * size * 0.6, y - size + J[1]), (x - d * size * 0.45, y + J[2] * 0.5),
               (x - d * size * 0.6, y + size + J[3])]
        self.g.polygon(pts, _on_paper(color), 0.85 * alpha)
        s = Shape()
        self._sketch_poly(s, pts + [pts[0]], max(1.4, size * 0.16), J, size * 0.25)
        self.g.fill_shape(s, INK, 0.9 * alpha)

    def ui_burst(self, x, y, text: str, t: float, color: Optional[RGB] = None, size: int = 120):
        col = color or self.palette["accent"]
        g = self.g
        if t < 0.18:
            pop = 0.6 + 0.6 * (t / 0.18)
        elif t < 0.35:
            pop = 1.2 - 0.2 * ((t - 0.18) / 0.17)
        else:
            pop = 1.0
        alpha = 1.0 if t < 0.7 else max(0.0, 1 - (t - 0.7) / 0.3)
        sz = int(min(112, size))
        k = size / sz
        tilt = ((_h32(len(text), ord(text[0]) if text else 0) % 100) / 100 - 0.5) * 0.16
        if not text.strip().isdigit():
            bw = _text_width(text, sz, TITLE_BIG if sz >= 70 else TITLE_HAND)
            sc = max(1.0, bw * 1.08 / 100.0, sz * 1.25 / 100.0) * pop * k
            g.blit_xform(self.spr_burst_blank, x, y, sc, sc * 0.72, angle=-tilt, alpha=alpha * 0.95)
        else:
            spr = self.spr_inkring[int(text.strip()[0]) % 3]
            g.stamp(spr, x, y, col, alpha * 0.9, scale_x=size * 1.3 / spr.w * pop, angle=t * 2, op=X.OP_OVER)
        self._draw_word(x, y, text, sz, INK, col, alpha=alpha, scale=pop * k, angle=tilt)

    def ui_stage_thumb(self, stage_name: str, x, y, w, h, frame: int, selected: bool = False):
        """A polaroid-ish snapshot of the page, taped down; the selected one gets circled in red pen."""
        from stages import get_stage
        g = self.g
        stage = get_stage(stage_name)
        a = self.stage_assets(stage)
        g.fill_rect(x - 6 + 4, y - 6 + 5, w + 12, h + 22, (70, 56, 32), 0.18)
        g.fill_rect(x - 6, y - 6, w + 12, h + 22, PAPER_HI)
        th = self._thumb(stage, a)
        g.blit_xform(th, x + w / 2, y + h / 2, w / th.w, h / th.h)
        s = Shape()
        J = self._jit(_h32(int(x), int(y)), frame // 6 if selected else 0, 12)
        self._sketch_poly(s, self._wobbly_rect(x, y, w, h, J, 0.6), 1.3, J, 3)
        g.fill_shape(s, GRAPHITE, 0.8)
        self._tape(x + w / 2, y - 6, 60, 18, J[4] * 0.08, (236, 226, 190))
        if selected:
            s = Shape()
            self._sketch_poly(s, self._wobbly_rect(x - 10, y - 10, w + 20, h + 32, J, 2.5), 2.6, J, 10)
            g.fill_shape(s, RED_PEN, 0.9)

    def draw_figure_static(self, sk, color: RGB, scale: float = 1.0):
        J = self._jit(_h32(*color), self._ui_frame // 5)
        s_ink, s_mark = self._figure(sk, J, scale=scale)
        self._marker(s_mark, tuple(int(c) for c in color))
        self.g.fill_shape(s_ink, INK)
