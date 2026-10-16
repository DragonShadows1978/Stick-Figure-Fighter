"""
Text rendering: stroke-font glyphs rasterized into per-size atlases, drawn as
colored masks through XRender (any color/alpha, nothing re-rendered per string).
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import raster as R
from . import xrender as X
from .context import Gfx, Image
from .font import glyph_strokes


@dataclass(frozen=True)
class FontStyle:
    weight: float = 1.1          # stroke width in font units (cap height = 10 units)
    slant: float = 0.0           # x shear per unit of height (0.2 = italic)
    tracking: float = 1.6        # extra space between glyphs, font units
    jitter: float = 0.0          # hand-drawn wobble amplitude, font units
    variants: int = 1            # glyph variants (for jittered styles)
    taper: float = 0.0           # 0..1 pressure taper at stroke ends
    double: float = 0.0          # alpha of a second, offset pass (sketchy double line)
    glow: float = 0.0            # glow radius in px at size 32 (scales with size)
    seed: int = 7


@dataclass
class _Glyph:
    x: int
    y: int
    w: int
    h: int
    ox: float     # offset from pen x to glyph image left
    oy: float     # offset from baseline to glyph image top (negative = above)
    adv: float


def _resample(stroke, step: float):
    out = []
    for i in range(len(stroke) - 1):
        (ax, ay), (bx, by) = stroke[i], stroke[i + 1]
        n = max(1, int(math.hypot(bx - ax, by - ay) / step))
        for k in range(n):
            t = k / n
            out.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    out.append(stroke[-1])
    return out


def _wobble(stroke, amp: float, rng: np.random.Generator):
    if amp <= 0 or len(stroke) < 2:
        return stroke
    pts = _resample(stroke, 0.6)
    n = len(pts)
    ph1, ph2 = rng.random(2) * 6.28
    f1, f2 = 0.15 + rng.random() * 0.2, 0.4 + rng.random() * 0.3
    ox, oy = (rng.random(2) - 0.5) * amp * 1.5
    out = []
    for i, (x, y) in enumerate(pts):
        t = i / max(1, n - 1)
        out.append((x + ox * (1 - t) + math.sin(i * f1 + ph1) * amp,
                    y + oy * t + math.cos(i * f2 + ph2) * amp))
    # overshoot the ends a little, like a quick pen stroke
    if len(out) >= 2 and math.hypot(out[-1][0] - out[0][0], out[-1][1] - out[0][1]) > 1.0:
        (x0, y0), (x1, y1) = out[0], out[1]
        d = math.hypot(x1 - x0, y1 - y0) or 1
        out.insert(0, (x0 - (x1 - x0) / d * amp * 0.8, y0 - (y1 - y0) / d * amp * 0.8))
        (xa, ya), (xb, yb) = out[-2], out[-1]
        d = math.hypot(xb - xa, yb - ya) or 1
        out.append((xb + (xb - xa) / d * amp * 0.8, yb + (yb - ya) / d * amp * 0.8))
    return out


def _render_glyph(strokes, px: float, style: FontStyle, pad: int, rng) -> Tuple[np.ndarray, float, float]:
    """Coverage mask for one glyph. Returns (mask, left_offset_px, top_offset_from_baseline_px)."""
    if not strokes:
        return np.zeros((1, 1), np.float32), 0.0, 0.0
    sk = px / 10.0
    w_px = max(1.0, style.weight * sk)

    def xf(pt):
        x, y = pt
        return x * sk + (10 - y) * sk * style.slant, y * sk

    passes = [(strokes, 1.0)]
    if style.double > 0:
        passes.append((strokes, style.double))
    rendered = []
    for strokes_p, alpha in passes:
        for s in strokes_p:
            pts = [xf(p) for p in _wobble(s, style.jitter, rng)] if style.jitter else [xf(p) for p in s]
            rendered.append((pts, alpha))
    xs = [p[0] for pts, _ in rendered for p in pts]
    ys = [p[1] for pts, _ in rendered for p in pts]
    x0, x1 = min(xs) - w_px - pad, max(xs) + w_px + pad
    y0, y1 = min(ys) - w_px - pad, max(ys) + w_px + pad
    W, H = int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))
    xx, yy = R.grid(W, H, x0, y0)
    mask = np.zeros((H, W), np.float32)
    for pts, alpha in rendered:
        d = np.full((H, W), np.inf, np.float32)
        n = len(pts)
        for i in range(max(1, n - 1)):
            ax, ay = pts[i]
            bx, by = pts[min(i + 1, n - 1)]
            if style.taper > 0 and n > 2:
                t0, t1 = i / (n - 1), (i + 1) / (n - 1)
                prof = lambda t: 1 - style.taper * (abs(2 * t - 1) ** 3)
                d = np.minimum(d, R.sd_tapered(xx, yy, ax, ay, bx, by,
                                               w_px / 2 * prof(t0), w_px / 2 * prof(t1)))
            else:
                d = np.minimum(d, R.sd_segment(xx, yy, ax, ay, bx, by, w_px / 2))
        mask = np.maximum(mask, R.coverage(d) * alpha)
    return mask, x0, y0 - 10 * sk


SIZE_LADDER = (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 44,
               48, 52, 56, 60, 64, 72, 80, 88, 96, 112, 128)


def ladder_size(size: float) -> int:
    """Snap to the nearest atlas size so animated text never triggers new atlases mid-match."""
    return min(SIZE_LADDER, key=lambda s: abs(s - size))


def _atlas_cache_path(style: "FontStyle", size: int) -> str:
    import os
    from . import font as _font
    from .cache import CACHE_DIR, source_hash
    tag = source_hash(_font, __import__(__name__, fromlist=["x"]))
    key = "_".join(f"{v}" for v in (style.weight, style.slant, style.tracking, style.jitter, style.variants,
                                     style.taper, style.double, style.glow, style.seed))
    return os.path.join(CACHE_DIR, f"font-{key}-{size}-{tag}.npz")


class GlyphAtlas:
    """All glyphs for one (style, size), plus an optional blurred glow atlas."""

    CHARS = (" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
             ".,:;!?'\"-+=_/\\|%()[]{}<>^*#&~@←→↑↓♦⊕♪")

    def __init__(self, gfx: Gfx, style: FontStyle, size: int):
        self.size = size
        self.style = style
        sk = size / 10.0
        glow_px = style.glow * size / 32.0
        self.tracking_px = style.tracking * sk
        self.cap_px = size
        if self._load(gfx, glow_px):
            return
        pad = int(math.ceil(glow_px * 2.2)) + 2
        rng = np.random.default_rng(style.seed)
        entries: Dict[str, List[Tuple[np.ndarray, float, float, float]]] = {}
        for ch in self.CHARS:
            adv, strokes, _ = glyph_strokes(ch)
            vs = []
            for v in range(style.variants):
                m, ox, oy = _render_glyph(strokes, size, style, pad, rng)
                vs.append((m, ox, oy, adv * sk))
            entries[ch] = vs
        # shelf-pack
        atlas_w = 1024 if size < 48 else 2048
        x = y = row_h = 0
        placed = []
        for ch, vs in entries.items():
            for m, ox, oy, adv in vs:
                h, w = m.shape
                if x + w > atlas_w:
                    x, y, row_h = 0, y + row_h + 1, 0
                placed.append((ch, x, y, m, ox, oy, adv))
                x += w + 1
                row_h = max(row_h, h)
        atlas_h = y + row_h + 1
        mask = np.zeros((atlas_h, atlas_w), np.float32)
        self.glyphs: Dict[str, List[_Glyph]] = {}
        for ch, gx, gy, m, ox, oy, adv in placed:
            h, w = m.shape
            mask[gy:gy + h, gx:gx + w] = m
            self.glyphs.setdefault(ch, []).append(_Glyph(gx, gy, w, h, ox, oy, adv))
        glow = np.clip(R.blur(mask, glow_px) * 1.6, 0, 1) if glow_px > 0 else None
        self._save(mask, glow)
        self._upload(gfx, mask, glow)

    def _upload(self, gfx, mask, glow):
        self.mask_img = gfx.upload_alpha(mask)
        self.glow_img: Optional[Image] = gfx.upload_alpha(glow) if glow is not None else None

    def _save(self, mask, glow):
        import os
        rows = [(ord(ch), g.x, g.y, g.w, g.h, g.ox, g.oy, g.adv) for ch, vs in self.glyphs.items() for g in vs]
        try:
            path = _atlas_cache_path(self.style, self.size)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            arrays = {"mask": (mask * 255 + 0.5).astype(np.uint8), "rows": np.array(rows, np.float64)}
            if glow is not None:
                arrays["glow"] = (glow * 255 + 0.5).astype(np.uint8)
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)
        except OSError:
            pass

    def _load(self, gfx, glow_px) -> bool:
        try:
            with np.load(_atlas_cache_path(self.style, self.size)) as data:
                mask = data["mask"]
                glow = data["glow"] if "glow" in data else None
                rows = data["rows"]
        except (OSError, KeyError, ValueError):
            return False
        self.glyphs = {}
        for code, x, y, w, h, ox, oy, adv in rows:
            self.glyphs.setdefault(chr(int(code)), []).append(
                _Glyph(int(x), int(y), int(w), int(h), float(ox), float(oy), float(adv)))
        self._upload(gfx, mask, glow if glow_px > 0 else None)
        return True


class Face:
    """A drawable font: style + size, atlases built lazily per size."""

    def __init__(self, gfx: Gfx, style: FontStyle = FontStyle()):
        self.gfx = gfx
        self.style = style
        self._atlases: Dict[int, GlyphAtlas] = {}

    def atlas(self, size: int) -> GlyphAtlas:
        size = ladder_size(size)
        a = self._atlases.get(size)
        if a is None:
            a = self._atlases[size] = GlyphAtlas(self.gfx, self.style, size)
        return a

    def prewarm(self, *sizes):
        for s in sizes:
            self.atlas(s)

    def measure(self, text: str, size: int) -> float:
        a = self.atlas(size)
        total = 0.0
        for ch in text:
            g = a.glyphs.get(ch) or a.glyphs["?"]
            total += g[0].adv + a.tracking_px
        return max(0.0, total - a.tracking_px)

    def draw(self, text: str, x: float, y: float, size: int, color, alpha: float = 1.0,
             align: str = "left", valign: str = "baseline", glow_color=None,
             glow_alpha: float = 0.0, op: int = X.OP_OVER, glow_op: int = X.OP_ADD,
             tracking: Optional[float] = None) -> float:
        """Draw text; returns its width. valign: baseline | top | middle | bottom."""
        if not text or alpha <= 0:
            return 0.0
        a = self.atlas(size)
        track = a.tracking_px if tracking is None else tracking * size / 10.0
        width = sum((a.glyphs.get(c) or a.glyphs["?"])[0].adv + track for c in text) - track
        if align == "center":
            x -= width / 2
        elif align == "right":
            x -= width
        if valign == "top":
            y += a.cap_px
        elif valign == "middle":
            y += a.cap_px / 2
        g = self.gfx
        d = g.display
        layers = []
        if glow_alpha > 0 and a.glow_img is not None:
            layers.append((a.glow_img, g.solid(glow_color or color, min(1.0, glow_alpha)), glow_op))
        layers.append((a.mask_img, g.solid(color, alpha), op))
        for img, src, layer_op in layers:
            pen = x
            for i, ch in enumerate(text):
                vs = a.glyphs.get(ch) or a.glyphs["?"]
                gl = vs[(i * 7 + ord(ch)) % len(vs)]
                if gl.w > 1:
                    X.XRenderComposite(d, layer_op, src, img.pic, g.dst, 0, 0, gl.x, gl.y,
                                       int(round(pen + gl.ox)), int(round(y + gl.oy)), gl.w, gl.h)
                pen += gl.adv + track
        return width
