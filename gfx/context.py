"""
Gfx: an alpha-compositing, antialiased drawing context on top of XRender.

Everything draws into the renderer's back-buffer pixmap (or an offscreen
Image while a `with gfx.target(img):` block is active). Colors are (r, g, b)
0-255 with a separate alpha 0-1.

Shapes are tessellated into triangles client-side and sent in one
XRenderCompositeTriangles call per shape group, so overlapping pieces of the
same shape (e.g. limb capsules meeting at a joint) don't double their alpha.
"""
import math
from collections import OrderedDict
from contextlib import contextmanager
from ctypes import byref, cast, c_void_p
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import xrender as X

RGB = Tuple[int, int, int]


class Image:
    """An ARGB32 picture living on the X server."""

    __slots__ = ("pic", "pixmap", "w", "h", "filtered")

    def __init__(self, pic: int, pixmap: int, w: int, h: int):
        self.pic = pic
        self.pixmap = pixmap
        self.w = w
        self.h = h
        self.filtered = False  # bilinear filter is only enabled once the image gets transformed


def to_bgra8(rgba: np.ndarray) -> np.ndarray:
    """Premultiplied float RGBA (H, W, 4) in 0..1 -> contiguous premultiplied BGRA uint8."""
    if rgba.dtype == np.uint8:
        out = rgba[..., ::-1][..., [1, 2, 3, 0]]
    else:
        out = (np.clip(rgba[..., [2, 1, 0, 3]], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(out)


def _segments_for(r: float) -> int:
    return max(10, min(72, int(r * 1.3) + 8))


_UNIT: dict = {}


def _unit(n: int):
    u = _UNIT.get(n)
    if u is None:
        a = np.arange(n + 1) * (2 * math.pi / n)
        u = _UNIT[n] = (np.cos(a), np.sin(a))
    return u


def circle_points(cx: float, cy: float, r: float, segments: int = 0) -> List[Tuple[float, float]]:
    n = segments or _segments_for(r)
    c, s = _unit(n)
    return list(zip((cx + c[:-1] * r).tolist(), (cy + s[:-1] * r).tolist()))


class Shape:
    """Triangle accumulator; fill everything at once with Gfx.fill_shape."""

    __slots__ = ("tris", "chunks")

    def __init__(self):
        self.tris: List[float] = []
        self.chunks: List[np.ndarray] = []   # (k, 6) float arrays from vectorized primitives

    def tri(self, x1, y1, x2, y2, x3, y3):
        self.tris.extend((x1, y1, x2, y2, x3, y3))
        return self

    def fan(self, cx: float, cy: float, pts: Sequence[Tuple[float, float]]):
        n = len(pts)
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            self.tris.extend((cx, cy, ax, ay, bx, by))
        return self

    def circle(self, cx: float, cy: float, r: float, segments: int = 0):
        if r > 0:
            n = segments or _segments_for(r)
            c, s = _unit(n)
            x = cx + c * r
            y = cy + s * r
            arr = np.empty((n, 6))
            arr[:, 0] = cx
            arr[:, 1] = cy
            arr[:, 2] = x[:-1]
            arr[:, 3] = y[:-1]
            arr[:, 4] = x[1:]
            arr[:, 5] = y[1:]
            self.chunks.append(arr)
        return self

    def quad(self, x1, y1, x2, y2, x3, y3, x4, y4):
        self.tris.extend((x1, y1, x2, y2, x3, y3, x1, y1, x3, y3, x4, y4))
        return self

    def rect(self, x, y, w, h):
        return self.quad(x, y, x + w, y, x + w, y + h, x, y + h)

    def segment(self, x1, y1, x2, y2, w1: float, w2: Optional[float] = None, caps: bool = True):
        """Thick line from width w1 to w2 (tapered), with round caps."""
        if w2 is None:
            w2 = w1
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            if caps:
                self.circle(x1, y1, max(w1, w2) / 2)
            return self
        nx, ny = -dy / length, dx / length
        h1, h2 = w1 / 2, w2 / 2
        self.quad(x1 + nx * h1, y1 + ny * h1, x2 + nx * h2, y2 + ny * h2,
                  x2 - nx * h2, y2 - ny * h2, x1 - nx * h1, y1 - ny * h1)
        if caps:
            self.circle(x1, y1, h1)
            self.circle(x2, y2, h2)
        return self

    def polyline(self, pts: Sequence[Tuple[float, float]], width: float, closed: bool = False,
                 joints: bool = True):
        """Thick polyline: capless segments plus one round cap per vertex."""
        n = len(pts)
        if n == 1:
            return self.circle(pts[0][0], pts[0][1], width / 2)
        last = n if closed else n - 1
        for i in range(last):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            self.segment(ax, ay, bx, by, width, caps=False)
        if joints:
            seg = max(8, min(40, int(width * 0.9) + 6))
            for x, y in pts:
                self.circle(x, y, width / 2, seg)
        return self

    def ring(self, cx, cy, r, width, segments: int = 0):
        ro, ri = r + width / 2, max(0.0, r - width / 2)
        n = segments or _segments_for(ro)
        c, s = _unit(n)
        ox, oy = cx + c * ro, cy + s * ro
        ix, iy = cx + c * ri, cy + s * ri
        arr = np.empty((2 * n, 6))
        arr[0::2] = np.stack([ox[:-1], oy[:-1], ox[1:], oy[1:], ix[1:], iy[1:]], axis=1)
        arr[1::2] = np.stack([ox[:-1], oy[:-1], ix[1:], iy[1:], ix[:-1], iy[:-1]], axis=1)
        self.chunks.append(arr)
        return self

    def stroke_closed(self, pts: Sequence[Tuple[float, float]], width: float):
        """Outline a closed convex-ish path with mitred joins (no per-vertex caps)."""
        p = np.asarray(pts, np.float64)
        n = len(p)
        if n < 3:
            return self
        prev = np.roll(p, 1, axis=0)
        nxt = np.roll(p, -1, axis=0)
        d1 = p - prev
        d2 = nxt - p
        d1 /= np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-9)
        d2 /= np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-9)
        n1 = np.stack([-d1[:, 1], d1[:, 0]], axis=1)
        n2 = np.stack([-d2[:, 1], d2[:, 0]], axis=1)
        m = n1 + n2
        m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        scale = 1.0 / np.maximum(np.sum(m * n1, axis=1, keepdims=True), 0.3)
        off = m * scale * (width / 2)
        outer, inner = p + off, p - off
        o2, i2 = np.roll(outer, -1, axis=0), np.roll(inner, -1, axis=0)
        arr = np.empty((2 * n, 6))
        arr[0::2] = np.concatenate([outer, o2, i2], axis=1)
        arr[1::2] = np.concatenate([outer, i2, inner], axis=1)
        self.chunks.append(arr)
        return self

    def arc(self, cx, cy, r, width, start: float, end: float, segments: int = 0):
        """Thick arc between angles (radians), butt ends."""
        n = segments or max(6, int(abs(end - start) * r / 6) + 4)
        ro, ri = r + width / 2, max(0.0, r - width / 2)
        prev = None
        for i in range(n + 1):
            a = start + (end - start) * i / n
            c, s = math.cos(a), math.sin(a)
            cur = (cx + c * ro, cy + s * ro, cx + c * ri, cy + s * ri)
            if prev:
                self.quad(prev[0], prev[1], cur[0], cur[1], cur[2], cur[3], prev[2], prev[3])
            prev = cur
        return self


class Gfx:
    """XRender drawing context. Check `Gfx.create()` returning None for no-XRender fallback."""

    SOLID_CACHE_SIZE = 512

    def __init__(self, display, screen: int, drawable: int, width: int, height: int):
        self.display = display
        self.width = width
        self.height = height
        self.visual = X.XDefaultVisual(display, screen)
        self.fmt_argb = X.XRenderFindStandardFormat(display, X.PICT_STANDARD_ARGB32)
        self.fmt_a8 = X.XRenderFindStandardFormat(display, X.PICT_STANDARD_A8)
        vis_fmt = X.XRenderFindVisualFormat(display, self.visual)
        attrs = X.XRenderPictureAttributes()
        self.screen_pic = X.XRenderCreatePicture(display, drawable, vis_fmt, 0, byref(attrs))
        self.dst = self.screen_pic
        self._drawable = drawable
        self._solids: "OrderedDict[tuple, int]" = OrderedDict()
        self._images: List[Image] = []
        self._gc32 = None

    @classmethod
    def create(cls, display, screen: int, drawable: int, width: int, height: int) -> Optional["Gfx"]:
        if not X.has_render(display):
            return None
        return cls(display, screen, drawable, width, height)

    # ------------------------------------------------------------------ colors

    def solid(self, rgb: RGB, a: float = 1.0) -> int:
        """Cached solid-fill source picture (premultiplied)."""
        r, g, b = rgb
        ai = max(0, min(255, int(a * 255 + 0.5)))
        key = (int(r), int(g), int(b), ai)
        pic = self._solids.get(key)
        if pic is not None:
            self._solids.move_to_end(key)
            return pic
        af = ai / 255.0
        col = X.XRenderColor(int(r * 257 * af), int(g * 257 * af), int(b * 257 * af), ai * 257)
        pic = X.XRenderCreateSolidFill(self.display, byref(col))
        self._solids[key] = pic
        if len(self._solids) > self.SOLID_CACHE_SIZE:
            _, old = self._solids.popitem(last=False)
            X.XRenderFreePicture(self.display, old)
        return pic

    # ------------------------------------------------------------------ images

    def upload(self, rgba: np.ndarray) -> Image:
        """Upload premultiplied RGBA (float 0..1 or uint8) as a server-side Image."""
        data = to_bgra8(rgba)
        h, w = data.shape[:2]
        pixmap = X.XCreatePixmap(self.display, self._drawable, w, h, 32)
        if self._gc32 is None:
            self._gc32 = X.XCreateGC(self.display, pixmap, 0, None)
        ximg = X.XCreateImage(self.display, self.visual, 32, X.ZPIXMAP, 0,
                              data.ctypes.data_as(c_void_p), w, h, 32, w * 4)
        ximg.contents.byte_order = X.LSB_FIRST
        X.XPutImage(self.display, pixmap, self._gc32, ximg, 0, 0, 0, 0, w, h)
        ximg.contents.data = None
        X.XFree(cast(ximg, c_void_p))
        attrs = X.XRenderPictureAttributes()
        pic = X.XRenderCreatePicture(self.display, pixmap, self.fmt_argb, 0, byref(attrs))
        img = Image(pic, pixmap, w, h)
        self._images.append(img)
        return img

    def upload_cropped(self, rgba: np.ndarray, margin: int = 1):
        """Upload only the non-transparent bounding box. Returns (image, x_offset, y_offset)."""
        alpha = rgba[..., 3]
        ys, xs = np.nonzero(alpha > (0 if alpha.dtype == np.uint8 else 1e-4))
        if len(xs) == 0:
            return None, 0, 0
        x0, x1 = max(0, xs.min() - margin), min(rgba.shape[1], xs.max() + 1 + margin)
        y0, y1 = max(0, ys.min() - margin), min(rgba.shape[0], ys.max() + 1 + margin)
        return self.upload(np.ascontiguousarray(rgba[y0:y1, x0:x1])), int(x0), int(y0)

    def upload_alpha(self, mask) -> Image:
        """Upload a coverage mask (float 0..1 or uint8) as a white premultiplied image."""
        m = mask if mask.dtype == np.uint8 else (np.clip(mask, 0, 1) * 255 + 0.5).astype(np.uint8)
        return self.upload(np.repeat(m[..., None], 4, axis=2))

    def offscreen(self, w: int, h: int) -> Image:
        """Blank transparent ARGB image usable as a drawing target."""
        return self.upload(np.zeros((h, w, 4), np.uint8))

    def free(self, img: Image):
        if img.pic:
            X.XRenderFreePicture(self.display, img.pic)
            X.XFreePixmap(self.display, img.pixmap)
            img.pic = 0
        if img in self._images:
            self._images.remove(img)

    def _ensure_filter(self, img: Image):
        if not img.filtered:
            X.XRenderSetPictureFilter(self.display, img.pic, b"bilinear", None, 0)
            img.filtered = True

    def set_repeat(self, img: Image, repeat: int = X.REPEAT_NORMAL):
        attrs = X.XRenderPictureAttributes()
        attrs.repeat = repeat
        X.XRenderChangePicture(self.display, img.pic, X.CP_REPEAT, byref(attrs))

    @contextmanager
    def target(self, img: Image):
        """Redirect drawing into an offscreen Image."""
        prev = self.dst
        self.dst = img.pic
        try:
            yield
        finally:
            self.dst = prev

    def clear(self, img: Optional[Image] = None):
        """Make an image (or the current target) fully transparent."""
        pic = img.pic if img else self.dst
        w = img.w if img else self.width
        h = img.h if img else self.height
        col = X.XRenderColor(0, 0, 0, 0)
        X.XRenderFillRectangle(self.display, X.OP_SRC, pic, byref(col), 0, 0, w, h)

    # ------------------------------------------------------------------ blits

    def blit(self, img: Image, x: float, y: float, alpha: float = 1.0, op: int = X.OP_OVER,
             sx: int = 0, sy: int = 0, w: Optional[int] = None, h: Optional[int] = None):
        if img is None or alpha <= 0.0:
            return
        w = img.w - sx if w is None else w
        h = img.h - sy if h is None else h
        mask = 0 if alpha >= 0.999 else self.solid((0, 0, 0), alpha)
        X.XRenderComposite(self.display, op, img.pic, mask, self.dst,
                           sx, sy, 0, 0, int(round(x)), int(round(y)), w, h)

    def blit_region(self, img: Image, sx: float, sy: float, sw: float, sh: float,
                    dx: float, dy: float, dw: float, dh: float, alpha: float = 1.0, op: int = X.OP_OVER):
        """Scale the source rectangle (sx, sy, sw, sh) of img into the destination rectangle."""
        if img is None or alpha <= 0 or dw <= 0 or dh <= 0:
            return
        t = X.XTransform()
        vals = ((sw / dw, 0.0, sx), (0.0, sh / dh, sy), (0.0, 0.0, 1.0))
        for i in range(3):
            for j in range(3):
                t.matrix[i][j] = X.fixed(vals[i][j])
        self._ensure_filter(img)
        X.XRenderSetPictureTransform(self.display, img.pic, byref(t))
        mask = 0 if alpha >= 0.999 else self.solid((0, 0, 0), alpha)
        X.XRenderComposite(self.display, op, img.pic, mask, self.dst, 0, 0, 0, 0,
                           int(round(dx)), int(round(dy)), int(round(dw)), int(round(dh)))
        X.XRenderSetPictureTransform(self.display, img.pic, byref(_IDENTITY))

    def blit_centered(self, img: Image, cx: float, cy: float, alpha: float = 1.0,
                      op: int = X.OP_OVER):
        self.blit(img, cx - img.w / 2, cy - img.h / 2, alpha, op)

    def blit_xform(self, img: Image, cx: float, cy: float, scale_x: float = 1.0,
                   scale_y: Optional[float] = None, angle: float = 0.0, alpha: float = 1.0,
                   op: int = X.OP_OVER, anchor: Tuple[float, float] = (0.5, 0.5)):
        """Blit with scale/rotation (radians) around `anchor` (fraction of image) placed at (cx, cy)."""
        if img is None or alpha <= 0.0:
            return
        if scale_y is None:
            scale_y = scale_x
        if abs(scale_x) < 1e-4 or abs(scale_y) < 1e-4:
            return
        ax, ay = img.w * anchor[0], img.h * anchor[1]
        c, s = math.cos(angle), math.sin(angle)
        # Bounding box of the transformed image in destination space
        corners = []
        for px, py in ((0, 0), (img.w, 0), (img.w, img.h), (0, img.h)):
            ux, uy = (px - ax) * scale_x, (py - ay) * scale_y
            corners.append((cx + ux * c - uy * s, cy + ux * s + uy * c))
        bx = math.floor(min(p[0] for p in corners))
        by = math.floor(min(p[1] for p in corners))
        bw = math.ceil(max(p[0] for p in corners)) - bx
        bh = math.ceil(max(p[1] for p in corners)) - by
        # Map bbox-local destination coords back into image coords:
        # img = S^-1 R^-1 (local + b - c) + a
        ox, oy = bx - cx, by - cy
        m00, m01 = c / scale_x, s / scale_x
        m10, m11 = -s / scale_y, c / scale_y
        m02 = m00 * ox + m01 * oy + ax
        m12 = m10 * ox + m11 * oy + ay
        t = X.XTransform()
        vals = ((m00, m01, m02), (m10, m11, m12), (0.0, 0.0, 1.0))
        for i in range(3):
            for j in range(3):
                t.matrix[i][j] = X.fixed(vals[i][j])
        self._ensure_filter(img)
        X.XRenderSetPictureTransform(self.display, img.pic, byref(t))
        mask = 0 if alpha >= 0.999 else self.solid((0, 0, 0), alpha)
        X.XRenderComposite(self.display, op, img.pic, mask, self.dst, 0, 0, 0, 0, bx, by, bw, bh)
        ident = X.XTransform()
        for i in range(3):
            ident.matrix[i][i] = X.fixed(1.0)
        X.XRenderSetPictureTransform(self.display, img.pic, byref(ident))

    def stamp(self, mask: Image, cx: float, cy: float, rgb: RGB, a: float = 1.0,
              scale_x: float = 1.0, scale_y: Optional[float] = None, angle: float = 0.0,
              op: int = X.OP_ADD):
        """
        Draw a solid color through an image's alpha (a tinted sprite), centered at (cx, cy),
        optionally scaled/rotated. Default op is additive: ideal for glows and sparks.
        """
        if mask is None or a <= 0.0:
            return
        if scale_y is None:
            scale_y = scale_x
        src = self.solid(rgb, min(1.0, a))
        if abs(scale_x - 1) < 1e-3 and abs(scale_y - 1) < 1e-3 and abs(angle) < 1e-4:
            X.XRenderComposite(self.display, op, src, mask.pic, self.dst, 0, 0, 0, 0,
                               int(round(cx - mask.w / 2)), int(round(cy - mask.h / 2)), mask.w, mask.h)
            return
        if abs(scale_x) < 1e-3 or abs(scale_y) < 1e-3:
            return
        c, s = math.cos(angle), math.sin(angle)
        hw, hh = mask.w / 2, mask.h / 2
        xs, ys = [], []
        for px, py in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
            ux, uy = px * scale_x, py * scale_y
            xs.append(cx + ux * c - uy * s)
            ys.append(cy + ux * s + uy * c)
        bx, by = math.floor(min(xs)), math.floor(min(ys))
        bw, bh = math.ceil(max(xs)) - bx, math.ceil(max(ys)) - by
        ox, oy = bx - cx, by - cy
        m00, m01 = c / scale_x, s / scale_x
        m10, m11 = -s / scale_y, c / scale_y
        t = X.XTransform()
        vals = ((m00, m01, m00 * ox + m01 * oy + hw), (m10, m11, m10 * ox + m11 * oy + hh), (0.0, 0.0, 1.0))
        for i in range(3):
            for j in range(3):
                t.matrix[i][j] = X.fixed(vals[i][j])
        self._ensure_filter(mask)
        X.XRenderSetPictureTransform(self.display, mask.pic, byref(t))
        X.XRenderComposite(self.display, op, src, mask.pic, self.dst, 0, 0, 0, 0, bx, by, bw, bh)
        X.XRenderSetPictureTransform(self.display, mask.pic, byref(_IDENTITY))

    # ------------------------------------------------------------------ shapes

    def fill_rect(self, x: float, y: float, w: float, h: float, rgb: RGB, a: float = 1.0,
                  op: int = X.OP_OVER):
        if w <= 0 or h <= 0 or a <= 0:
            return
        af = max(0.0, min(1.0, a))
        r, g, b = rgb
        col = X.XRenderColor(int(r * 257 * af), int(g * 257 * af), int(b * 257 * af), int(af * 65535))
        X.XRenderFillRectangle(self.display, op, self.dst, byref(col),
                               int(round(x)), int(round(y)), int(round(w)), int(round(h)))

    def fill_shape(self, shape: Shape, rgb: RGB, a: float = 1.0, op: int = X.OP_OVER,
                   src: Optional[Image] = None, src_x: int = 0, src_y: int = 0):
        """Fill accumulated triangles with a solid color, or with an image as the source."""
        parts = shape.chunks
        if shape.tris:
            parts = parts + [np.asarray(shape.tris, dtype=np.float64).reshape(-1, 6)]
        if not parts or a <= 0:
            return
        allt = parts[0] if len(parts) == 1 else np.concatenate(parts)
        n = len(allt)
        if n == 0:
            return
        F = 65536.0
        arr = (X.XTriangle * n)()
        flat = (allt.reshape(-1) * F).astype(np.int32)
        # XTriangle is 6 contiguous int32s
        np.ctypeslib.as_array(cast(arr, X.POINTER(X.c_int)), shape=(n * 6,))[:] = flat
        source = src.pic if src is not None else self.solid(rgb, a)
        X.XRenderCompositeTriangles(self.display, op, source, self.dst, self.fmt_a8,
                                    src_x, src_y, arr, n)

    def polygon(self, pts: Sequence[Tuple[float, float]], rgb: RGB, a: float = 1.0,
                op: int = X.OP_OVER, winding: bool = False):
        """Fill an arbitrary (possibly concave) polygon, antialiased."""
        n = len(pts)
        if n < 3 or a <= 0:
            return
        arr = (X.XPointDouble * n)(*[X.XPointDouble(float(x), float(y)) for x, y in pts])
        X.XRenderCompositeDoublePoly(self.display, op, self.solid(rgb, a), self.dst, self.fmt_a8,
                                     0, 0, 0, 0, arr, n, 1 if winding else 0)

    def circle(self, cx, cy, r, rgb: RGB, a: float = 1.0, op: int = X.OP_OVER):
        self.fill_shape(Shape().circle(cx, cy, r), rgb, a, op)

    def ring(self, cx, cy, r, width, rgb: RGB, a: float = 1.0, op: int = X.OP_OVER):
        self.fill_shape(Shape().ring(cx, cy, r, width), rgb, a, op)

    def line(self, x1, y1, x2, y2, width: float, rgb: RGB, a: float = 1.0, op: int = X.OP_OVER):
        self.fill_shape(Shape().segment(x1, y1, x2, y2, width), rgb, a, op)

    def polyline(self, pts, width: float, rgb: RGB, a: float = 1.0, closed: bool = False,
                 op: int = X.OP_OVER):
        self.fill_shape(Shape().polyline(pts, width, closed), rgb, a, op)

    def round_rect(self, x, y, w, h, radius, rgb: RGB, a: float = 1.0, op: int = X.OP_OVER):
        self.polygon(round_rect_points(x, y, w, h, radius), rgb, a, op)

    def round_rect_outline(self, x, y, w, h, radius, width, rgb: RGB, a: float = 1.0,
                           op: int = X.OP_OVER):
        self.fill_shape(Shape().stroke_closed(round_rect_points(x, y, w, h, radius), width), rgb, a, op)

    def flush(self):
        X.XFlush(self.display)


_IDENTITY = X.XTransform()
for _i in range(3):
    _IDENTITY.matrix[_i][_i] = X.fixed(1.0)


def round_rect_points(x, y, w, h, r, per_corner: int = 6) -> List[Tuple[float, float]]:
    r = max(0.0, min(r, w / 2, h / 2))
    if r <= 0.5:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    pts = []
    for cx, cy, a0 in ((x + w - r, y + r, -math.pi / 2), (x + w - r, y + h - r, 0.0),
                       (x + r, y + h - r, math.pi / 2), (x + r, y + r, math.pi)):
        for i in range(per_corner + 1):
            a = a0 + (math.pi / 2) * i / per_corner
            pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    return pts
