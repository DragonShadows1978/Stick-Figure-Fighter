"""
Visual themes. Each theme is a complete art direction (stages, fighters,
effects, HUD, menus) drawn through the XRender gfx layer; "classic" is the
original core-X11 renderer and the fallback when XRender is unavailable.
"""
from typing import Dict, List, Type

from .base import Theme
from .classic import ClassicTheme

_REGISTRY: Dict[str, Type[Theme]] = {"classic": ClassicTheme}


def register(cls: Type[Theme]):
    _REGISTRY[cls.name] = cls
    return cls


def _load_builtin():
    # Imported lazily so a broken theme can't take the game down at import time.
    for mod in ("neon", "sketch", "silhouette"):
        try:
            __import__(f"themes.{mod}")
        except ImportError:
            pass


def available() -> List[str]:
    _load_builtin()
    order = ["neon", "sketch", "silhouette", "classic"]
    return [n for n in order if n in _REGISTRY] + [n for n in _REGISTRY if n not in order]


def create(name: str, renderer) -> Theme:
    _load_builtin()
    cls = _REGISTRY.get(name)
    if cls is None or (renderer.gfx is None and not cls.classic):
        cls = ClassicTheme
    return cls(renderer)


def title(name: str) -> str:
    _load_builtin()
    cls = _REGISTRY.get(name)
    return cls.title if cls else name
