"""
Neon Arcade theme: glowing light-tube fighters on synthwave stages.

Battlefield  -> "Sunset Grid": striped retro sun, wireframe mountains, scrolling floor grid.
Final Dest.  -> "Cyber Void": nebula, wireframe planet with ring, drifting shards.
"""
import math
from typing import Dict, Optional, Tuple

import numpy as np

from engine import FighterState
from gfx import raster as R
from gfx import xrender as X
from gfx.cache import cached
from gfx.context import Shape
from gfx.text import Face, FontStyle

from . import register
from .base import DODGE_COLORS, FighterLook, Theme

RGB = Tuple[int, int, int]

PAD_X, PAD_Y = 48, 28            # parallax margin baked around background layers
HORIZON = 470                    # Sunset Grid horizon line (screen y)

CHAR_COLORS = {
    "RUSH": (255, 52, 120),
    "ZONE": (40, 224, 255),
    "GRAB": (255, 168, 36),
    "FLEX": (110, 255, 100),
}
DEFAULT_COLORS = [(255, 70, 160), (70, 200, 255)]


def _mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# =============================================================================
# SPRITES
# =============================================================================

def _glow_sprite(size: int, power: float = 2.2) -> np.ndarray:
    xx, yy = R.grid(size, size)
    d = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    a = np.clip(1 - d, 0, 1) ** power
    return R.tint(a, (255, 255, 255))


def _flare_sprite(size: int) -> np.ndarray:
    xx, yy = R.grid(size, size)
    cx = cy = size / 2
    dx, dy = np.abs(xx - cx) / (size / 2), np.abs(yy - cy) / (size / 2)
    core = np.clip(1 - np.hypot(dx, dy) * 3.2, 0, 1) ** 1.5
    star = np.clip(1 - dx * 1.0, 0, 1) ** 6 * np.clip(1 - dy * 14, 0, 1) + \
        np.clip(1 - dy * 1.0, 0, 1) ** 6 * np.clip(1 - dx * 14, 0, 1)
    halo = np.clip(1 - np.hypot(dx, dy), 0, 1) ** 3 * 0.35
    return R.tint(np.clip(core + star + halo, 0, 1), (255, 255, 255))


def _ring_sprite(size: int, width: float) -> np.ndarray:
    xx, yy = R.grid(size, size)
    d = np.abs(np.hypot(xx - size / 2, yy - size / 2) - (size / 2 - width * 2.5))
    a = np.exp(-(d / width) ** 2) + np.exp(-(d / (width * 3)) ** 2) * 0.3
    return R.tint(np.clip(a, 0, 1), (255, 255, 255))


def _hex_sprite(size: int) -> np.ndarray:
    """Honeycomb shield texture inside a circle."""
    xx, yy = R.grid(size, size)
    cx = cy = size / 2
    r = size / 2 - 2
    s = size / 9.0
    q = (xx - cx) / (s * 1.5)
    rr = (yy - cy) / (s * math.sqrt(3)) - q / 2
    # distance to hex cell edges via cube rounding
    x3, z3 = q, rr
    y3 = -x3 - z3
    rx, ry, rz = np.round(x3), np.round(y3), np.round(z3)
    dx, dy, dz = np.abs(rx - x3), np.abs(ry - y3), np.abs(rz - z3)
    edge = np.maximum(np.maximum(dx, dy), dz)
    lines = np.clip((edge - 0.40) / 0.08, 0, 1)
    dist = np.hypot(xx - cx, yy - cy)
    inside = R.coverage(dist - r)
    fres = np.clip(dist / r, 0, 1) ** 3
    a = inside * (lines * 0.55 + fres * 0.6 + 0.06)
    return R.tint(np.clip(a, 0, 1), (255, 255, 255))


def _tube_sprite(w: int, h: int) -> np.ndarray:
    """Glow profile across a tube, with rounded soft ends (stretched along x per segment)."""
    xx, yy = R.grid(w, h)
    end = 13.0
    dx = np.maximum(np.abs(xx - w / 2) - (w / 2 - end), 0)
    d = np.hypot(dx, yy - h / 2)
    a = np.exp(-(d / 4.2) ** 2) * 0.55 + np.exp(-d / 6.5) * 0.45
    return R.tint(np.clip(a, 0, 1), (255, 255, 255))


def _hud_card(w: int, h: int, color) -> np.ndarray:
    G = 10
    W, H = w + 2 * G, h + 2 * G
    xx, yy = R.grid(W, H)
    sd = R.sd_box(xx, yy, W / 2, H / 2, w / 2, h / 2, 14)
    img = R.blank(W, H)
    R.paint(img, R.coverage(sd), R.gradient_stops((yy - G) / h, [(0, (16, 6, 40)), (1, (6, 2, 18))]), a=0.8)
    R.paint(img, R.coverage(sd + 2) * np.exp(-((yy - G) / 18)) * 0.10, (255, 255, 255), op="add")
    edge = np.abs(sd)
    R.paint(img, np.exp(-(edge / 5.0) ** 2) * 0.25 + np.exp(-edge / 9) * 0.10, color, op="add")
    R.paint(img, R.coverage(edge - 0.8), color, a=0.95)
    return np.clip(img, 0, 1)


def _streak_sprite(w: int, h: int) -> np.ndarray:
    xx, yy = R.grid(w, h)
    dx = np.abs(xx - w / 2) / (w / 2)
    dy = np.abs(yy - h / 2) / (h / 2)
    a = np.clip(1 - dx, 0, 1) ** 1.5 * np.clip(1 - dy, 0, 1) ** 2
    return R.tint(a, (255, 255, 255))


# =============================================================================
# BACKGROUNDS
# =============================================================================

def _stars(img, rng, n, y_max, w, h, fade_to=None):
    xs = rng.random(n) * w
    ys = rng.random(n) * y_max
    for x, y in zip(xs, ys):
        size = rng.random() ** 3 * 1.6 + 0.5
        b = 0.35 + rng.random() * 0.65
        if fade_to:
            b *= np.clip(1 - y / fade_to, 0, 1) ** 0.7
        x0, y0 = int(x) - 4, int(y) - 4
        sx, sy = R.grid(9, 9, x0, y0)
        d = np.hypot(sx - x, sy - y)
        a = (R.coverage(d - size * 0.5) + np.exp(-d / (size * 1.4)) * 0.4) * b
        tintc = (210 + rng.integers(0, 45), 210 + rng.integers(0, 45), 255)
        R.composite(img, R.tint(np.clip(a, 0, 1), tintc), x0, y0, op="add")


def _sunset_far(W: int, H: int) -> np.ndarray:
    """Sky, stars, sun and floor grid (opaque). Coordinates include the parallax pad."""
    oy = PAD_Y
    hz = HORIZON + oy
    sky = R.linear_gradient(W, H, [
        (0.00, (4, 3, 18)), (0.30, (14, 6, 44)), (0.52, (52, 12, 92)),
        ((hz - 40) / H, (150, 30, 130)), (hz / H, (255, 90, 150)), (hz / H + 0.002, (14, 4, 34)),
        (1.00, (6, 2, 18))])
    img = R.opaque(sky)
    rng = np.random.default_rng(11)
    _stars(img, rng, 260, hz - 120, W, H, fade_to=hz - 60)

    # horizon haze
    xx, yy = R.grid(W, H)
    haze = np.exp(-np.abs(yy - hz) / 26.0) * 0.55
    R.paint(img, haze, (255, 80, 170), op="screen")

    # retro sun
    cx, cy, r = W / 2, hz - 14, 196
    d = np.hypot(xx - cx, yy - cy)
    sun_glow = np.exp(-np.maximum(d - r, 0) / 70.0) * 0.55 * (yy < hz)
    R.paint(img, sun_glow, (255, 70, 140), op="screen")
    local = (yy - (cy - r * 0.05)) / r
    bands = np.zeros_like(xx, bool)
    for i in range(7):
        p, th = 0.08 + i * 0.13, 0.026 + i * 0.02
        bands |= np.abs(local - p) < th / 2
    sun_cov = R.coverage(d - r) * (~bands) * (yy < hz - 1)
    sun_col = R.gradient_stops((yy - (cy - r)) / (1.15 * r),
                               [(0.0, (255, 245, 140)), (0.45, (255, 170, 60)), (0.8, (255, 70, 120)),
                                (1.0, (230, 40, 160))])
    R.paint(img, sun_cov, sun_col)

    # floor: vertical grid lines converging on the vanishing point
    floor = yy > hz
    vx, vy = cx, hz
    lines = np.zeros_like(xx)
    for k in range(-24, 25):
        bx = cx + k * 90
        dist = R.sd_segment(xx, yy, vx, vy, bx * 3 - 2 * cx, H + (H - vy) * 2)
        depth = np.clip((yy - hz) / (H - hz), 0, 1)
        lines = np.maximum(lines, np.exp(-(dist / (0.55 + depth * 1.3)) ** 2) * (0.15 + depth * 0.45))
    floor_glow = np.exp(-(yy - hz) / 90.0)
    R.paint(img, lines * floor, (255, 60, 200), op="add")
    R.paint(img, floor_glow * floor * 0.18, (120, 40, 200), op="add")
    R.paint(img, R.coverage(np.abs(yy - hz) - 0.9), (255, 160, 220))
    return np.clip(img, 0, 1)


def _sunset_mountains(W: int, H: int) -> np.ndarray:
    hz = HORIZON + PAD_Y
    img = R.blank(W, H)
    xx, yy = R.grid(W, H)
    for seed, base, amp, col_fill, col_edge, shift in (
            (5, hz, 150, (26, 8, 58), (60, 220, 255), 0.0),
            (9, hz, 95, (16, 4, 36), (255, 60, 190), 0.0)):
        ridge = R.ridge(W, base, amp, 260, seed, octaves=5, sharp=0.7)
        # keep the middle low so the sun shows
        mid = np.exp(-((np.arange(W) - W / 2) / (W * 0.2)) ** 2)
        ridge = ridge + mid * amp * 0.85
        ridge = np.minimum(ridge, base - 2)
        cov = R.below_curve(W, H, ridge) * (yy <= hz)
        fill = R.gradient_stops((yy - ridge[None, :]) / max(1, amp), [(0, col_fill), (1, (8, 2, 20))])
        R.paint(img, cov, fill)
        # contour wireframe
        for k in range(1, 6):
            off = ridge + (base - ridge) * (k / 6.0)
            line = np.exp(-((yy - off[None, :]) / 0.8) ** 2) * cov * (0.28 - k * 0.035)
            R.paint(img, line, col_edge, op="add")
        slope = np.abs(np.gradient(ridge))[None, :]
        edge_d = np.abs(yy - ridge[None, :]) / np.sqrt(1 + slope ** 2)
        edge = (np.exp(-(edge_d / 1.1) ** 2) + np.exp(-edge_d / 6.0) * 0.35) * (yy <= hz)
        R.paint(img, np.clip(edge, 0, 1), col_edge, op="add")
    return img


def _void_far(W: int, H: int) -> np.ndarray:
    xx, yy = R.grid(W, H)
    sky = R.linear_gradient(W, H, [(0.0, (2, 2, 12)), (0.55, (8, 6, 30)), (1.0, (14, 4, 30))])
    img = R.opaque(sky)
    n1 = R.fbm(xx, yy, 340, 6, seed=3)
    n2 = R.fbm(xx + 500, yy, 220, 5, seed=8)
    neb = np.clip((n1 - 0.42) * 2.4, 0, 1) ** 1.6
    neb2 = np.clip((n2 - 0.5) * 2.6, 0, 1) ** 2
    R.paint(img, neb * 0.55, (120, 40, 210), op="add")
    R.paint(img, neb2 * 0.45, (20, 160, 220), op="add")
    R.paint(img, (neb * neb2) * 0.6, (255, 90, 200), op="add")
    rng = np.random.default_rng(23)
    _stars(img, rng, 420, H, W, H)
    # wireframe planet
    pcx, pcy, pr = W * 0.78, H * 0.26, 150
    d = np.hypot(xx - pcx, yy - pcy)
    body = R.coverage(d - pr)
    shade = np.clip(1 - ((xx - pcx + pr * 0.5) ** 2 + (yy - pcy + pr * 0.5) ** 2) ** 0.5 / (pr * 2.2), 0, 1)
    R.paint(img, body, R.gradient_stops(shade, [(0, (6, 4, 22)), (1, (40, 16, 80))]))
    with np.errstate(invalid="ignore"):
        lat = np.arcsin(np.clip((yy - pcy) / pr, -1, 1))
        lon = np.arcsin(np.clip((xx - pcx) / (pr * np.cos(lat) + 1e-3), -1, 1))
    lat_lines = np.exp(-((np.abs(((lat / (math.pi / 10)) % 1) - 0.5) - 0.5) * pr * 0.33) ** 2 / 1.2)
    lon_lines = np.exp(-((np.abs(((lon / (math.pi / 12)) % 1) - 0.5) - 0.5) * pr * 0.25) ** 2 / 1.2)
    grid = np.clip(lat_lines + lon_lines, 0, 1) * body * (0.25 + shade * 0.75)
    R.paint(img, grid, (60, 220, 255), op="add")
    rim = np.exp(-np.abs(d - pr) / 2.5) + np.exp(-np.maximum(d - pr, 0) / 30.0) * 0.4
    R.paint(img, np.clip(rim, 0, 1), (90, 200, 255), op="add")
    # planetary ring (ellipse), front half drawn over the body
    ang = math.radians(-14)
    ux = (xx - pcx) * math.cos(ang) - (yy - pcy) * math.sin(ang)
    uy = (xx - pcx) * math.sin(ang) + (yy - pcy) * math.cos(ang)
    e = np.sqrt((ux / (pr * 1.9)) ** 2 + (uy / (pr * 0.42)) ** 2)
    ring = np.exp(-((e - 1) * 60) ** 2) + np.exp(-((e - 1.08) * 70) ** 2) * 0.6
    front = (uy > 0) | (d > pr)
    R.paint(img, np.clip(ring, 0, 1) * front, (255, 70, 200), op="add")
    # distant horizon glow bottom
    R.paint(img, np.exp(-(H - yy) / 120.0) * 0.35, (90, 30, 160), op="add")
    return np.clip(img, 0, 1)


def _void_shards(W: int, H: int) -> np.ndarray:
    img = R.blank(W, H)
    rng = np.random.default_rng(31)
    for i in range(16):
        cx, cy = rng.random() * W, 60 + rng.random() * (H * 0.7)
        if 360 < cx - PAD_X < 920 and cy > 380:
            continue
        s = 6 + rng.random() * 18
        a0 = rng.random() * math.pi * 2
        pts = [(cx + math.cos(a0 + k * 2.1 + rng.random() * 0.5) * s,
                cy + math.sin(a0 + k * 2.1 + rng.random() * 0.5) * s) for k in range(3)]
        x0, y0 = int(cx - s - 12), int(cy - s - 12)
        sz = int(2 * s + 24)
        xx, yy = R.grid(sz, sz, x0, y0)
        sd = R.sd_polygon(xx, yy, pts)
        col = (60, 220, 255) if i % 2 else (255, 80, 200)
        R.composite(img, R.tint(R.coverage(sd) * 0.55, (20, 10, 40)), x0, y0)
        edge = np.exp(-(np.abs(sd) / 0.9) ** 2) + np.exp(-np.maximum(sd, 0) / 5.0) * 0.3
        R.composite(img, R.tint(np.clip(edge, 0, 1) * 0.9, col), x0, y0, op="add")
    return img


def _platform_sprite(w: int, h: int, solid: bool, style: str) -> np.ndarray:
    """Neon platform with glow padding (G px on every side)."""
    G = 36
    W, H = w + 2 * G, h + 2 * G + (40 if solid else 16)
    xx, yy = R.grid(W, H)
    img = R.blank(W, H)
    x0, y0 = G, G
    edge_col = (70, 230, 255)
    under_col = (255, 60, 190)
    if solid:
        if style == "void":
            # trapezoid hull: wide top, narrower bottom
            pts = [(x0, y0), (x0 + w, y0), (x0 + w - 60, y0 + h + 26), (x0 + 60, y0 + h + 26)]
            sd = R.sd_polygon(xx, yy, pts)
            depth = (yy - y0) / (h + 26)
        else:
            sd = R.sd_box(xx, yy, x0 + w / 2, y0 + h / 2, w / 2, h / 2, 3)
            depth = (yy - y0) / h
        body = R.coverage(sd)
        fill = R.gradient_stops(depth, [(0, (34, 12, 70)), (0.25, (18, 8, 46)), (1, (8, 3, 22))])
        R.paint(img, R.blur(np.exp(-np.maximum(sd, 0) / 14.0) * (yy > y0 + 4), 3) * 0.55,
                under_col, op="add")
        R.paint(img, body * 0.94, fill)
        # circuit traces / LED strips on the face
        rng = np.random.default_rng(w)
        for k in range(14):
            ty = y0 + 12 + rng.random() * (h - 18 if style != "void" else h + 6)
            tx0 = x0 + 20 + rng.random() * (w - 140)
            tx1 = tx0 + 30 + rng.random() * 90
            d = R.sd_segment(xx, yy, tx0, ty, tx1, ty)
            c = edge_col if k % 3 else under_col
            R.paint(img, np.exp(-(d / 0.7) ** 2) * body * 0.55, c, op="add")
            R.paint(img, R.coverage(np.hypot(xx - tx1, yy - ty) - 1.6) * body, c, op="add")
        for k in range(1, 8):
            vx = x0 + w * k / 8
            d = np.abs(xx - vx)
            R.paint(img, np.exp(-(d / 0.6) ** 2) * body * (yy > y0 + 4) * 0.12, edge_col, op="add")
    else:
        sd = R.sd_box(xx, yy, x0 + w / 2, y0 + h / 2 + 1, w / 2, h / 2 - 1, 4)
        body = R.coverage(sd)
        R.paint(img, R.blur(np.exp(-np.maximum(sd, 0) / 8.0), 2) * (yy > y0 + h / 2) * 0.7, under_col, op="add")
        R.paint(img, body * 0.85, (24, 10, 52))
    # the glowing top edge tube (full width)
    d = R.sd_segment(xx, yy, x0 + 2, y0 + 1, x0 + w - 2, y0 + 1)
    R.paint(img, np.exp(-np.maximum(d, 0) / 9.0) * 0.55, edge_col, op="add")
    R.paint(img, np.exp(-(np.maximum(d, 0) / 2.4) ** 2), edge_col, op="add")
    R.paint(img, R.coverage(d - 1.1), (235, 255, 255))
    return np.clip(img, 0, 1)


def _overlay(W: int, H: int) -> np.ndarray:
    """Vignette + faint scanlines (drawn over the world, under the HUD)."""
    xx, yy = R.grid(W, H)
    v = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.68)) ** 2)
    vig = np.clip(v - 0.55, 0, 1) ** 1.6 * 0.85
    scan = ((yy.astype(np.int32) % 3) == 0) * 0.07
    a = np.clip(vig + scan, 0, 1)
    return R.tint(a, (4, 0, 12))


# =============================================================================
# THEME
# =============================================================================

@register
class NeonTheme(Theme):
    name = "neon"
    title = "Neon Arcade"

    palette = {
        "bg": (6, 4, 20), "panel": (14, 8, 34), "panel_edge": (90, 70, 170),
        "text": (240, 240, 255), "text_dim": (150, 140, 200), "accent": (255, 60, 190),
        "accent2": (50, 220, 255), "good": (90, 255, 170), "warn": (255, 210, 60),
        "bad": (255, 60, 100), "p1": (255, 60, 150), "p2": (60, 200, 255),
    }

    def build(self):
        g = self.g
        self.fonts["ui"] = Face(g, FontStyle(weight=1.05, tracking=2.4, glow=7))
        self.fonts["title"] = Face(g, FontStyle(weight=1.2, tracking=2.8, glow=12))
        self.fonts["small"] = Face(g, FontStyle(weight=1.1, tracking=1.0, glow=0))
        self.fonts["digits"] = Face(g, FontStyle(weight=1.35, tracking=1.4, glow=10, slant=0.12))
        self.fonts["ui"].prewarm(11, 15, 18, 20)
        self.fonts["small"].prewarm(11)
        self.fonts["digits"].prewarm(34, 44, 48, 52, 56, 60)
        dep = [_glow_sprite]
        self.spr_glow = g.upload(cached("neon-glow128", lambda: _glow_sprite(128), dep))
        self.spr_glow_soft = g.upload(cached("neon-glow256s", lambda: _glow_sprite(256, 3.2), dep))
        self.spr_flare = g.upload(cached("neon-flare160", lambda: _flare_sprite(160), [_flare_sprite]))
        self.spr_ring = g.upload(cached("neon-ring160", lambda: _ring_sprite(160, 2.2), [_ring_sprite]))
        self.spr_hex = g.upload(cached("neon-hex160", lambda: _hex_sprite(160), [_hex_sprite]))
        self.spr_streak = g.upload(cached("neon-streak", lambda: _streak_sprite(128, 24), [_streak_sprite]))
        self.spr_tube = g.upload(cached("neon-tube", lambda: _tube_sprite(64, 28), [_tube_sprite]))
        self._hud_cards = {}
        self.overlay = g.upload(cached(f"neon-overlay-{self.w}x{self.h}", lambda: _overlay(self.w, self.h),
                                       [_overlay]))
        rng = np.random.default_rng(5)
        self.twinkles = [(rng.random() * self.w, rng.random() * (HORIZON - 160), rng.random() * 6.28,
                          0.03 + rng.random() * 0.08) for _ in range(28)]

    def build_stage(self, stage) -> dict:
        g = self.g
        W, H = self.w + 2 * PAD_X, self.h + 2 * PAD_Y
        kind = "void" if "final" in stage.name.lower() else "sunset"
        if kind == "sunset":
            far = g.upload(cached(f"neon-sunset-far-{W}x{H}", lambda: _sunset_far(W, H), [_sunset_far, _stars]))
            mid = g.upload_cropped(cached(f"neon-sunset-mtn-{W}x{H}", lambda: _sunset_mountains(W, H),
                                          [_sunset_mountains]))
        else:
            far = g.upload(cached(f"neon-void-far-{W}x{H}", lambda: _void_far(W, H), [_void_far, _stars]))
            mid = g.upload_cropped(cached(f"neon-void-shards-{W}x{H}", lambda: _void_shards(W, H),
                                          [_void_shards]))
        plats = {}
        for p in [stage.main_platform] + stage.platforms:
            key = (int(p.width), int(p.height), not p.passthrough)
            if key not in plats:
                w, h, solid = key
                plats[key] = g.upload(cached(f"neon-plat-{kind}-{w}x{h}-{int(solid)}",
                                             lambda: _platform_sprite(w, h, solid, kind), [_platform_sprite]))
        return {"kind": kind, "far": far, "mid": mid, "plats": plats}

    # ------------------------------------------------------------------ colors

    def character_color(self, fighter) -> RGB:
        ct = fighter.stats.character_type
        if ct in CHAR_COLORS:
            return CHAR_COLORS[ct]
        return DEFAULT_COLORS[fighter.player_id % 2]

    def color_for_name(self, name: Optional[str], idx: int = 0) -> RGB:
        return CHAR_COLORS.get(name or "", DEFAULT_COLORS[idx % 2])

    # ------------------------------------------------------------------ stage

    def draw_background(self, game_state, stage, camera, shake):
        a = self.stage_assets(stage)
        g = self.g
        cx, cy = camera
        far_k, mid_k = 0.15, 0.45
        g.blit(a["far"], -PAD_X - cx * far_k + shake[0] * 0.2, -PAD_Y - cy * far_k + shake[1] * 0.2,
               op=X.OP_SRC)
        f = self.fx.frame
        if a["kind"] == "sunset":
            self._floor_lines(f, shake)
        for x, y, ph, sp in self.twinkles:
            tw = 0.5 + 0.5 * math.sin(f * sp + ph)
            if tw > 0.55:
                g.stamp(self.spr_flare, x - cx * far_k, y - cy * far_k, (200, 220, 255), (tw - 0.55) * 1.6,
                        scale_x=0.12 + tw * 0.06)
        mid, mx, my = a["mid"]
        g.blit(mid, mx - PAD_X - cx * mid_k + shake[0] * 0.5, my - PAD_Y - cy * mid_k + shake[1] * 0.5)

    def _floor_lines(self, frame, shake):
        """Horizontal grid lines scrolling toward the viewer."""
        g = self.g
        n = 14
        phase = (frame * 0.012) % 1.0
        s = Shape()
        s_glow = Shape()
        for k in range(n):
            t = (k + phase) / n
            y = HORIZON + (self.h + 30 - HORIZON) * t ** 2.1 + shake[1] * 0.3
            w = 0.6 + t * 1.8
            s.segment(-10, y, self.w + 10, y, w, caps=False)
            s_glow.segment(-10, y, self.w + 10, y, w * 5, caps=False)
        g.fill_shape(s_glow, (255, 50, 190), 0.06, op=X.OP_ADD)
        g.fill_shape(s, (255, 90, 210), 0.32, op=X.OP_ADD)

    def draw_platform(self, plat, stage, shake):
        a = self.stage_assets(stage)
        img = a["plats"].get((int(plat.width), int(plat.height), not plat.passthrough))
        if img is not None:
            self.g.blit(img, plat.x - 36 + shake[0], plat.y - 36 + shake[1])
        # a light pulse running along the edge
        f = self.fx.frame
        px = plat.x + ((f * 4 + plat.x * 3) % (plat.width + 200)) - 100
        if plat.x <= px <= plat.x + plat.width:
            self.g.stamp(self.spr_streak, px + shake[0], plat.y + 1 + shake[1], (120, 240, 255), 0.8,
                         scale_x=0.9, scale_y=0.5)

    def draw_foreground(self, game_state, stage, camera, shake):
        f = self.fx.frame
        for ko in self.fx.ko_flashes:
            t = (f - ko["frame"]) / 40.0
            fighter = game_state.fighters[ko["pid"]]
            col = self.character_color(fighter)
            self.g.fill_rect(0, 0, self.w, self.h, col, 0.25 * (1 - t) ** 2, op=X.OP_ADD)
        self.g.blit(self.overlay, 0, 0)

    # ------------------------------------------------------------------ fighters

    def _tube_shape(self, sk, width: float, head_ring: bool = True) -> Shape:
        s = Shape()
        for _, pts in sk.chains():
            s.polyline(pts, width)
        hx, hy = sk.head
        if head_ring:
            s.ring(hx, hy, sk.head_r, width)
        else:
            s.circle(hx, hy, sk.head_r + width / 2)
        return s

    def _tube_glow(self, sk, col, a: float = 1.0, scale: float = 1.0):
        """Soft glow along every limb: one stretched streak sprite per segment."""
        g = self.g
        spr = self.spr_tube
        for _, pts in sk.chains():
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 1:
                    continue
                g.stamp(spr, (x1 + x2) / 2, (y1 + y2) / 2, col, 0.55 * a,
                        scale_x=(length + 26 * scale) / spr.w, scale_y=scale,
                        angle=math.atan2(y2 - y1, x2 - x1))
        g.stamp(self.spr_glow, sk.head[0], sk.head[1], col, 0.55 * a,
                scale_x=(sk.head_r * 2 + 30 * scale) / self.spr_glow.w)

    def draw_fighter_aura(self, fighter, sk, look: FighterLook):
        cx, cy = sk.center
        col = look.color
        if look.armor:
            pulse = 0.6 + 0.4 * math.sin(self.fx.frame * 0.4)
            self.g.stamp(self.spr_glow_soft, cx, cy - 8, (255, 190, 40), 0.55 * pulse, scale_x=0.85, scale_y=1.05)
        self.g.stamp(self.spr_glow_soft, cx, cy - 6, col, 0.16 if not look.blink else 0.06,
                     scale_x=0.55, scale_y=0.75)

    def draw_fighter(self, fighter, sk, look: FighterLook):
        g = self.g
        col = look.color
        a = 0.35 if look.blink else 1.0
        if look.flash > 0:
            col = _mix(col, (255, 255, 255), look.flash)
        core = _mix(col, (255, 255, 255), 0.6)
        # dark backing so the tube reads against bright scenery
        g.fill_shape(self._tube_shape(sk, 14, head_ring=False), (6, 2, 18), 0.55 * a)
        self._tube_glow(sk, col, a)
        g.fill_shape(self._tube_shape(sk, 4.4), core, a)
        hx, hy = sk.head
        g.circle(hx, hy, sk.head_r - 2, col, 0.18 * a, op=X.OP_ADD)
        # eye: a bright dot looking where we face
        g.circle(hx + sk.facing * sk.head_r * 0.35, hy - 1.5, 2.0, (255, 255, 255), a)
        if look.shield_stun:
            f = self.fx.frame
            for k in range(3):
                ang = f * 0.15 + k * 2.09
                g.stamp(self.spr_flare, hx + math.cos(ang) * 18, hy - 16 + math.sin(ang) * 5,
                        (255, 230, 90), 0.9, scale_x=0.12)

    def draw_afterimage(self, fighter, sk, dodge_type, alpha: float):
        col = DODGE_COLORS.get(dodge_type, (200, 200, 255))
        s = self._tube_shape(sk, 6)
        self.g.fill_shape(s, col, alpha * 0.45, op=X.OP_ADD)
        sk2 = sk.transformed(dx=3)
        self.g.fill_shape(self._tube_shape(sk2, 2), (255, 60, 200), alpha * 0.4, op=X.OP_ADD)

    def draw_knockback_trail(self, fighter, pts):
        col = self.character_color(fighter)
        n = len(pts)
        s_glow, s_core = Shape(), Shape()
        for i in range(n - 1):
            t0, t1 = i / n, (i + 1) / n
            s_glow.segment(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 4 + 26 * t0, 4 + 26 * t1)
            s_core.segment(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 1 + 5 * t0, 1 + 5 * t1)
        self.g.fill_shape(s_glow, col, 0.18, op=X.OP_ADD)
        self.g.fill_shape(s_core, _mix(col, (255, 255, 255), 0.5), 0.6, op=X.OP_ADD)

    def draw_shield(self, fighter, sk):
        hp = max(0.0, fighter.shield_health) / 100.0
        r = 26 + 26 * hp
        col = (80, 255, 190) if hp > 0.5 else (255, 210, 60) if hp > 0.25 else (255, 60, 100)
        cx, cy = sk.center
        cy -= 4
        flick = 0.85 + 0.15 * math.sin(self.fx.frame * 0.9)
        scale = (r * 2) / self.spr_hex.w
        self.g.stamp(self.spr_hex, cx, cy, col, 0.42 * flick, scale_x=scale)
        self.g.stamp(self.spr_glow_soft, cx, cy, col, 0.10, scale_x=scale * 0.9)

    def draw_counter(self, fighter, sk):
        cx, cy = sk.center
        f = self.fx.frame
        for k in range(2):
            rr = 46 + 6 * math.sin(f * 0.5 + k * 3)
            self.g.stamp(self.spr_ring, cx, cy, (90, 150, 255), 0.7, scale_x=rr * 2 / self.spr_ring.w)
        # electric arcs
        rng = np.random.default_rng(f // 2)
        for _ in range(3):
            a0 = rng.random() * 6.28
            pts = []
            for j in range(5):
                a = a0 + j * 0.18
                rad = 44 + rng.random() * 10 - 5
                pts.append((cx + math.cos(a) * rad, cy + math.sin(a) * rad))
            self.g.polyline(pts, 1.6, (180, 210, 255), 0.9, op=X.OP_ADD)

    def draw_reflector(self, fighter, sk):
        cx, cy = sk.center
        f = self.fx.frame
        pts = [(cx + math.cos(i * math.pi / 3 + f * 0.08) * 52, cy + math.sin(i * math.pi / 3 + f * 0.08) * 52)
               for i in range(6)]
        self.g.polygon(pts, (60, 240, 255), 0.10, op=X.OP_ADD)
        self.g.polyline(pts, 9, (60, 240, 255), 0.25, closed=True, op=X.OP_ADD)
        self.g.polyline(pts, 2.4, (200, 255, 255), 0.95, closed=True, op=X.OP_ADD)

    # ------------------------------------------------------------------ projectiles & effects

    def draw_projectile(self, p, trail, shake):
        g = self.g
        col = tuple(p.color)
        x, y = p.position.x + shake[0], p.position.y + shake[1]
        n = len(trail)
        elongated = p.width > p.height * 1.8
        for i, (tx, ty) in enumerate(trail[:-1]):
            t = (i + 1) / n
            g.stamp(self.spr_glow, tx + shake[0], ty + shake[1], col, 0.35 * t,
                    scale_x=(p.width / 60) * (0.5 + t), scale_y=(p.height / 60) * (0.5 + t))
        if elongated:
            g.stamp(self.spr_streak, x, y, col, 0.9, scale_x=p.width / 50, scale_y=p.height / 10)
            g.stamp(self.spr_streak, x, y, (255, 255, 255), 0.9, scale_x=p.width / 80, scale_y=p.height / 26)
        else:
            pulse = 1 + 0.12 * math.sin(self.fx.frame * 0.6)
            g.stamp(self.spr_glow, x, y, col, 0.9, scale_x=p.width / 36 * pulse)
            g.stamp(self.spr_flare, x, y, _mix(col, (255, 255, 255), 0.5), 0.9, scale_x=p.width / 70)
            g.circle(x, y, p.width * 0.22, (255, 255, 255))

    def draw_effect(self, e, shake):
        from renderer import EffectType
        g = self.g
        x, y = e.x + shake[0], e.y + shake[1]
        t = e.progress
        if e.effect_type == EffectType.HIT_SPARK:
            col = e.color if e.color != (255, 255, 255) else (180, 240, 255)
            k = 0.6 + 0.5 * min(1.5, e.intensity)
            g.stamp(self.spr_flare, x, y, (255, 255, 255), (1 - t) ** 1.5, scale_x=(0.6 + t * 0.9) * k,
                    angle=math.radians(e.angle) * 0.3)
            g.stamp(self.spr_glow_soft, x, y, col, 0.6 * (1 - t), scale_x=(0.3 + t * 0.6) * k)
            g.stamp(self.spr_ring, x, y, col, 0.9 * (1 - t), scale_x=(0.15 + t * 0.9) * k)
            n = 7
            s = Shape()
            for i in range(n):
                ang = math.radians(e.angle) + (i - n / 2) * 0.45 + (i * 1.7) % 0.6
                r0, r1 = (10 + 50 * t) * k, (24 + 90 * t ** 0.7) * k
                s.segment(x + math.cos(ang) * r0, y - math.sin(ang) * r0,
                          x + math.cos(ang) * r1, y - math.sin(ang) * r1, 3.2 * (1 - t) + 0.6, 0.4)
            g.fill_shape(s, _mix(col, (255, 255, 255), 0.4), 1 - t, op=X.OP_ADD)
        elif e.effect_type == EffectType.LANDING_DUST:
            fy = y + 40
            for k in range(2):
                tt = min(1.0, t * (1.2 + k * 0.5))
                g.stamp(self.spr_ring, x, fy, (90, 220, 255), 0.7 * (1 - tt), scale_x=(0.2 + tt * 0.8),
                        scale_y=(0.05 + tt * 0.16))
            for side in (-1, 1):
                g.stamp(self.spr_glow, x + side * (10 + 40 * t), fy - 3 - 6 * t, (255, 90, 210), 0.5 * (1 - t),
                        scale_x=0.18)
        elif e.effect_type == EffectType.REFLECTOR_HEX:
            pts = [(x + math.cos(i * math.pi / 3) * (24 + 50 * t), y + math.sin(i * math.pi / 3) * (24 + 50 * t))
                   for i in range(6)]
            g.polyline(pts, 3, (120, 255, 255), 1 - t, closed=True, op=X.OP_ADD)
            g.stamp(self.spr_flare, x, y, (200, 255, 255), 1 - t, scale_x=0.8)
        elif e.effect_type == EffectType.FRAME_ADVANTAGE:
            adv = int(e.intensity)
            label = f"+{adv}" if adv > 0 else str(adv)
            self.text(x, y - 30 - 20 * t, label, 20, e.color, align="center", alpha=1 - t * t,
                      glow=e.color, glow_alpha=0.8 * (1 - t))

    # ------------------------------------------------------------------ HUD

    def _damage_color(self, dmg: float) -> RGB:
        stops = [(0, (240, 250, 255)), (40, (255, 240, 120)), (80, (255, 160, 60)), (120, (255, 70, 90)),
                 (180, (255, 30, 160))]
        for (d0, c0), (d1, c1) in zip(stops, stops[1:]):
            if dmg <= d1:
                return _mix(c0, c1, (dmg - d0) / (d1 - d0))
        return stops[-1][1]

    def draw_hud(self, game_state):
        g = self.g
        f = self.fx.frame
        for i, fighter in enumerate(game_state.fighters[:2]):
            col = self.character_color(fighter)
            right = i == 1
            cw, ch = 250, 92
            x = self.w - 30 - cw if right else 30
            y = self.h - ch - 22
            card = self._hud_cards.get(col)
            if card is None:
                card = self._hud_cards[col] = g.upload(_hud_card(cw, ch, col))
            g.blit(card, x - 10, y - 10)
            name = fighter.stats.name.upper()
            self.text(x + 18, y + 26, name, 15, _mix(col, (255, 255, 255), 0.3), glow=col, glow_alpha=0.6)
            # damage, with a pop when it changes
            dmg = fighter.damage_percent
            anim = game_state.damage_anim_frames[i] if i < len(game_state.damage_anim_frames) else 0
            pop = 1.0 + 0.35 * (anim / 20.0) ** 2
            shake_x = (math.sin(f * 2.3) * 3 * anim / 20.0)
            dcol = self._damage_color(dmg)
            size = int(44 * pop)
            self.face("digits").draw(f"{int(dmg)}%", x + cw - 18 + shake_x, y + ch - 18, size, dcol,
                                     align="right", glow_color=dcol, glow_alpha=0.75)
            # stocks: little neon heads
            for s in range(max(0, fighter.stocks)):
                sx = x + 26 + s * 22
                sy = y + 50
                g.ring(sx, sy, 6.5, 2.2, col, 0.95)
                g.stamp(self.spr_glow, sx, sy, col, 0.35, scale_x=0.22)
            # shield tube
            hp = max(0.0, fighter.shield_health) / 100.0
            bx, by, bw = x + 18, y + ch - 16, 92
            g.line(bx, by, bx + bw, by, 4, (40, 30, 80), 0.9)
            if hp > 0:
                sc = (80, 255, 190) if hp > 0.5 else (255, 210, 60) if hp > 0.25 else (255, 60, 100)
                g.line(bx, by, bx + bw * hp, by, 3, sc, 1.0)
                g.line(bx, by, bx + bw * hp, by, 9, sc, 0.15, op=X.OP_ADD)
            # combo counter
            combo = game_state.combo_counts[i] if i < len(game_state.combo_counts) else 0
            if combo >= 2:
                cx = x + cw / 2
                self.text(cx, y - 18, f"{combo} HIT COMBO", 18, (255, 255, 255), align="center",
                          glow=(255, 60, 190), glow_alpha=1.0)
        t = game_state.get_time_display()
        if t:
            low = game_state.time_remaining < 600
            col = (255, 80, 110) if low and (f // 15) % 2 == 0 else (230, 240, 255)
            self.face("digits").draw(t, self.w / 2, 54, 34, col, align="center", glow_color=(255, 60, 190),
                                     glow_alpha=0.8)

    def draw_victory(self, game_state):
        g = self.g
        f = self.fx.frame
        g.fill_rect(0, 0, self.w, self.h, (6, 0, 20), 0.55)
        if game_state.winner is not None:
            fighter = game_state.fighters[game_state.winner]
            col = self.character_color(fighter)
            txt = f"{fighter.stats.name.upper()} WINS"
        else:
            col, txt = (255, 255, 255), "DRAW"
        cy = self.h / 2
        for k in range(3):
            yy = cy + (k - 1) * 46
            g.stamp(self.spr_streak, self.w / 2, yy, col, 0.35, scale_x=self.w / 90, scale_y=0.3)
        flick = 1.0 if (f % 47) > 3 else 0.55
        self.text(self.w / 2, cy, txt, 64, _mix(col, (255, 255, 255), 0.55), role="title", align="center",
                  valign="middle", glow=col, glow_alpha=flick, alpha=flick)

    # ------------------------------------------------------------------ UI kit

    def ui_backdrop(self, frame: int, variant: str = "menu"):
        from stages import battlefield
        a = self.stage_assets(battlefield())
        g = self.g
        sway = math.sin(frame * 0.004) * 20
        g.blit(a["far"], -PAD_X + sway * 0.3, -PAD_Y, op=X.OP_SRC)
        self.fx.frame = frame
        self._floor_lines(frame, (0, 0))
        for x, y, ph, sp in self.twinkles:
            tw = 0.5 + 0.5 * math.sin(frame * sp + ph)
            if tw > 0.55:
                g.stamp(self.spr_flare, x, y, (200, 220, 255), (tw - 0.55) * 1.6, scale_x=0.12 + tw * 0.06)
        mid, mx, my = a["mid"]
        g.blit(mid, mx - PAD_X + sway, my - PAD_Y)
        if variant != "menu":
            g.fill_rect(0, 0, self.w, self.h, (6, 2, 20), 0.45)
        g.blit(self.overlay, 0, 0)

    def ui_panel(self, x, y, w, h, emphasis: float = 0.0, accent: Optional[RGB] = None):
        g = self.g
        acc = accent or self.palette["panel_edge"]
        g.round_rect(x, y, w, h, 12, (10, 4, 28), 0.78)
        if emphasis > 0:
            g.round_rect_outline(x, y, w, h, 12, 10, acc, 0.10 * emphasis, op=X.OP_ADD)
            g.round_rect_outline(x, y, w, h, 12, 4, acc, 0.25 * emphasis, op=X.OP_ADD)
        g.round_rect_outline(x, y, w, h, 12, 1.4 + emphasis * 0.6, acc, 0.55 + 0.45 * emphasis)

    def ui_button(self, x, y, w, h, label, selected: bool, frame: int, accent: Optional[RGB] = None,
                  sublabel: Optional[str] = None, size: int = 20):
        acc = accent or self.palette["accent"]
        pulse = 0.75 + 0.25 * math.sin(frame * 0.15) if selected else 0.0
        self.ui_panel(x, y, w, h, pulse, acc if selected else None)
        col = (255, 255, 255) if selected else self.palette["text_dim"]
        cy = y + h / 2 if not sublabel or not selected else y + h * 0.40
        self.text(x + w / 2, cy, label, size, col, align="center", valign="middle",
                  glow=acc if selected else None, glow_alpha=0.9 if selected else 0.0)
        if sublabel and selected:
            self.text(x + w / 2, y + h * 0.76, sublabel, max(11, int(size * 0.55)), self.palette["text_dim"],
                      align="center", valign="middle")

    def ui_title(self, x, y, s, size=48, color=None, frame: int = 0, align="center"):
        col = color or self.palette["accent"]
        flick = 1.0
        if frame % 173 in (3, 4, 9):
            flick = 0.4
        self.text(x, y, s, size, _mix(col, (255, 255, 255), 0.55), role="title", align=align, valign="middle",
                  glow=col, glow_alpha=1.0 * flick, alpha=0.6 + 0.4 * flick)

    def ui_bar(self, x, y, w, h, frac: float, color: RGB, bg_alpha: float = 0.35):
        g = self.g
        g.line(x + h / 2, y + h / 2, x + w - h / 2, y + h / 2, h, (40, 30, 80), 0.9)
        if frac > 0:
            ex = x + h / 2 + (w - h) * min(1.0, frac)
            g.line(x + h / 2, y + h / 2, ex, y + h / 2, h * 0.7, color)
            g.line(x + h / 2, y + h / 2, ex, y + h / 2, h * 2.2, color, 0.15, op=X.OP_ADD)

    def draw_figure_static(self, sk, color: RGB, scale: float = 1.0):
        look = FighterLook(color=color)
        g = self.g
        core = _mix(color, (255, 255, 255), 0.6)
        g.fill_shape(self._tube_shape(sk, 14 * scale, head_ring=False), (6, 2, 18), 0.55)
        self._tube_glow(sk, color, 1.0, scale)
        g.fill_shape(self._tube_shape(sk, 4.4 * scale), core)
        g.circle(sk.head[0], sk.head[1], sk.head_r - 2 * scale, color, 0.18, op=X.OP_ADD)
