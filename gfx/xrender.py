"""
ctypes bindings for the bits of Xlib + XRender the gfx layer needs.

XRender gives us server-side alpha compositing and antialiased polygons,
which core X11 drawing can't do.
"""
import ctypes
import ctypes.util
from ctypes import (POINTER, Structure, byref, c_char_p, c_double, c_int, c_long,
                    c_short, c_uint, c_ulong, c_ushort, c_void_p)

Display = c_void_p
Drawable = c_ulong
Pixmap = c_ulong
Picture = c_ulong
GC = c_void_p
Atom = c_ulong
XFixed = c_int

# Picture ops
OP_CLEAR = 0
OP_SRC = 1
OP_OVER = 3
OP_ADD = 12
# Screen-ish blend modes (Render 0.11+)
OP_SCREEN = 0x0e
OP_MULTIPLY = 0x0d

PICT_STANDARD_ARGB32 = 0
PICT_STANDARD_A8 = 2

CP_REPEAT = 1 << 0
REPEAT_NONE = 0
REPEAT_NORMAL = 1
REPEAT_PAD = 2

ZPIXMAP = 2
LSB_FIRST = 0


class XRenderColor(Structure):
    _fields_ = [("red", c_ushort), ("green", c_ushort), ("blue", c_ushort), ("alpha", c_ushort)]


class XPointFixed(Structure):
    _fields_ = [("x", XFixed), ("y", XFixed)]


class XTriangle(Structure):
    _fields_ = [("p1", XPointFixed), ("p2", XPointFixed), ("p3", XPointFixed)]


class XPointDouble(Structure):
    _fields_ = [("x", c_double), ("y", c_double)]


class XTransform(Structure):
    _fields_ = [("matrix", (XFixed * 3) * 3)]


class XRenderPictureAttributes(Structure):
    _fields_ = [
        ("repeat", c_int),
        ("alpha_map", Picture),
        ("alpha_x_origin", c_int),
        ("alpha_y_origin", c_int),
        ("clip_x_origin", c_int),
        ("clip_y_origin", c_int),
        ("clip_mask", Pixmap),
        ("graphics_exposures", c_int),
        ("subwindow_mode", c_int),
        ("poly_edge", c_int),
        ("poly_mode", c_int),
        ("dither", Atom),
        ("component_alpha", c_int),
    ]


class XImage(Structure):
    # Leading fields only; we never touch the function table.
    _fields_ = [
        ("width", c_int), ("height", c_int), ("xoffset", c_int), ("format", c_int),
        ("data", c_void_p), ("byte_order", c_int), ("bitmap_unit", c_int),
        ("bitmap_bit_order", c_int), ("bitmap_pad", c_int), ("depth", c_int),
        ("bytes_per_line", c_int), ("bits_per_pixel", c_int),
    ]


def _load(name, fallback):
    path = ctypes.util.find_library(name) or fallback
    return ctypes.CDLL(path)


try:
    x11 = _load("X11", "libX11.so.6")
except OSError:  # pragma: no cover - no X11 at all
    x11 = None

try:
    xr = _load("Xrender", "libXrender.so.1")
except OSError:
    xr = None


def _sig(lib, name, argtypes, restype):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


if x11 is not None:
    XDefaultVisual = _sig(x11, "XDefaultVisual", [Display, c_int], c_void_p)
    XCreatePixmap = _sig(x11, "XCreatePixmap", [Display, Drawable, c_uint, c_uint, c_uint], Pixmap)
    XFreePixmap = _sig(x11, "XFreePixmap", [Display, Pixmap], c_int)
    XCreateGC = _sig(x11, "XCreateGC", [Display, Drawable, c_ulong, c_void_p], GC)
    XFreeGC = _sig(x11, "XFreeGC", [Display, GC], c_int)
    XCreateImage = _sig(x11, "XCreateImage",
                        [Display, c_void_p, c_uint, c_int, c_int, c_void_p, c_uint, c_uint, c_int, c_int],
                        POINTER(XImage))
    XPutImage = _sig(x11, "XPutImage",
                     [Display, Drawable, GC, POINTER(XImage), c_int, c_int, c_int, c_int, c_uint, c_uint],
                     c_int)
    XFree = _sig(x11, "XFree", [c_void_p], c_int)
    XFlush = _sig(x11, "XFlush", [Display], c_int)

if xr is not None:
    XRenderQueryExtension = _sig(xr, "XRenderQueryExtension", [Display, POINTER(c_int), POINTER(c_int)], c_int)
    XRenderQueryVersion = _sig(xr, "XRenderQueryVersion", [Display, POINTER(c_int), POINTER(c_int)], c_int)
    XRenderFindVisualFormat = _sig(xr, "XRenderFindVisualFormat", [Display, c_void_p], c_void_p)
    XRenderFindStandardFormat = _sig(xr, "XRenderFindStandardFormat", [Display, c_int], c_void_p)
    XRenderCreatePicture = _sig(xr, "XRenderCreatePicture",
                                [Display, Drawable, c_void_p, c_ulong, POINTER(XRenderPictureAttributes)],
                                Picture)
    XRenderFreePicture = _sig(xr, "XRenderFreePicture", [Display, Picture], None)
    XRenderCreateSolidFill = _sig(xr, "XRenderCreateSolidFill", [Display, POINTER(XRenderColor)], Picture)
    XRenderComposite = _sig(xr, "XRenderComposite",
                            [Display, c_int, Picture, Picture, Picture,
                             c_int, c_int, c_int, c_int, c_int, c_int, c_uint, c_uint], None)
    XRenderFillRectangle = _sig(xr, "XRenderFillRectangle",
                                [Display, c_int, Picture, POINTER(XRenderColor), c_int, c_int, c_uint, c_uint],
                                None)
    XRenderCompositeTriangles = _sig(xr, "XRenderCompositeTriangles",
                                     [Display, c_int, Picture, Picture, c_void_p, c_int, c_int,
                                      POINTER(XTriangle), c_int], None)
    XRenderCompositeDoublePoly = _sig(xr, "XRenderCompositeDoublePoly",
                                      [Display, c_int, Picture, Picture, c_void_p, c_int, c_int, c_int, c_int,
                                       POINTER(XPointDouble), c_int, c_int], None)
    XRenderSetPictureTransform = _sig(xr, "XRenderSetPictureTransform",
                                      [Display, Picture, POINTER(XTransform)], None)
    XRenderSetPictureFilter = _sig(xr, "XRenderSetPictureFilter",
                                   [Display, Picture, c_char_p, c_void_p, c_int], None)
    XRenderChangePicture = _sig(xr, "XRenderChangePicture",
                                [Display, Picture, c_ulong, POINTER(XRenderPictureAttributes)], None)


def has_render(display) -> bool:
    """True if the X server supports Render >= 0.10 (needed for solid fills)."""
    if x11 is None or xr is None or not display:
        return False
    ev, err = c_int(), c_int()
    if not XRenderQueryExtension(display, byref(ev), byref(err)):
        return False
    major, minor = c_int(), c_int()
    if not XRenderQueryVersion(display, byref(major), byref(minor)):
        return False
    return (major.value, minor.value) >= (0, 10)


def fixed(v: float) -> int:
    return int(v * 65536.0)
