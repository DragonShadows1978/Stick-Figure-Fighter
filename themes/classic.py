"""Classic theme: the original core-X11 look. Also the fallback without XRender."""
from .base import Theme


class ClassicTheme(Theme):
    name = "classic"
    title = "Classic"
    classic = True

    def build(self):
        pass
