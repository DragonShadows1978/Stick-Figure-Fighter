"""Read pixels back from a drawable and write PNGs (dev tooling and tests; no PIL needed)."""
import struct
import zlib
from ctypes import POINTER, c_int, c_uint, c_ulong, c_void_p, cast, string_at

import numpy as np

from . import xrender as X

if X.x11 is not None:
    XGetImage = X._sig(X.x11, "XGetImage",
                       [X.Display, X.Drawable, c_int, c_int, c_uint, c_uint, c_ulong, c_int],
                       POINTER(X.XImage))


def read_rgb(display, drawable, w: int, h: int) -> np.ndarray:
    """(h, w, 3) uint8 RGB from a 24/32-bit drawable."""
    img = XGetImage(display, drawable, 0, 0, w, h, 0xFFFFFFFF, X.ZPIXMAP)
    if not img:
        raise RuntimeError("XGetImage failed")
    im = img.contents
    raw = string_at(im.data, im.bytes_per_line * h)
    arr = np.frombuffer(raw, np.uint8).reshape(h, im.bytes_per_line)[:, :w * 4].reshape(h, w, 4)
    rgb = arr[..., [2, 1, 0]].copy()
    X.XFree(im.data)
    X.XFree(cast(img, c_void_p))
    return rgb


def write_png(path: str, rgb: np.ndarray):
    h, w = rgb.shape[:2]
    channels = rgb.shape[2] if rgb.ndim == 3 else 1
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)
