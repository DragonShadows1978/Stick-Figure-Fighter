"""
Offscreen rendering sandbox: open an X display, make a back buffer, hand a
Gfx to a draw function, save the result as PNG. Used to iterate on art.

    python3 tools/gfx_sandbox.py out.png        # draws the built-in test card
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gfx import xrender as X  # noqa: E402
from gfx.capture import read_rgb, write_png  # noqa: E402
from gfx.context import Gfx, Shape  # noqa: E402

W, H = 1280, 720


class Sandbox:
    def __init__(self, display_name=None, w=W, h=H):
        self.w, self.h = w, h
        name = (display_name or os.environ.get("DISPLAY", ":99")).encode()
        X.x11.XOpenDisplay.argtypes = [X.c_char_p]
        X.x11.XOpenDisplay.restype = X.Display
        self.dpy = X.x11.XOpenDisplay(name)
        if not self.dpy:
            raise RuntimeError("cannot open display")
        X.x11.XDefaultScreen.argtypes = [X.Display]
        self.screen = X.x11.XDefaultScreen(self.dpy)
        X.x11.XRootWindow.argtypes = [X.Display, X.c_int]
        X.x11.XRootWindow.restype = X.c_ulong
        X.x11.XDefaultDepth.argtypes = [X.Display, X.c_int]
        root = X.x11.XRootWindow(self.dpy, self.screen)
        depth = X.x11.XDefaultDepth(self.dpy, self.screen)
        self.buffer = X.XCreatePixmap(self.dpy, root, w, h, depth)
        self.gfx = Gfx.create(self.dpy, self.screen, self.buffer, w, h)
        assert self.gfx, "XRender unavailable"

    def save(self, path):
        X.x11.XSync.argtypes = [X.Display, X.c_int]
        X.x11.XSync(self.dpy, 0)
        write_png(path, read_rgb(self.dpy, self.buffer, self.w, self.h))


def test_card(g: Gfx):
    g.fill_rect(0, 0, W, H, (16, 18, 28))
    for i in range(12):
        g.circle(80 + i * 100, 100, 10 + i * 3, (255, 80 + i * 14, 120), 0.4 + i * 0.05)
    s = Shape()
    s.segment(100, 300, 400, 380, 18).segment(400, 380, 520, 260, 18, 4)
    g.fill_shape(s, (80, 220, 255), 0.6)
    g.polygon([(700, 250), (900, 300), (820, 450), (760, 330), (650, 420)], (255, 210, 60), 0.8)
    g.ring(1100, 340, 70, 10, (120, 255, 140), 0.9)
    g.round_rect(100, 500, 300, 120, 24, (60, 70, 120), 0.9)
    g.round_rect_outline(100, 500, 300, 120, 24, 3, (180, 200, 255))
    g.fill_rect(500, 500, 300, 120, (255, 255, 255), 0.25, op=X.OP_ADD)


if __name__ == "__main__":
    sb = Sandbox()
    t = time.perf_counter()
    test_card(sb.gfx)
    sb.save(sys.argv[1] if len(sys.argv) > 1 else "sandbox.png")
    print("ok %.1fms" % ((time.perf_counter() - t) * 1000))
