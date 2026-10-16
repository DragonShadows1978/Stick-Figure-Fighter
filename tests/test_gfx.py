"""
Graphics tests.

Pure tests (always run): rasterizer math, font coverage, tessellation.
Smoke tests (need an X display with XRender, e.g. Xvfb on $DISPLAY): every
theme renders a match frame, every menu screen and every in-game screen
without exceptions, and produces a non-blank image.
"""
import os
import random

import numpy as np
import pytest

from gfx import raster as R
from gfx.font import GLYPHS, glyph_strokes
from gfx.text import GlyphAtlas, SIZE_LADDER, ladder_size


# ---------------------------------------------------------------- pure

def test_circle_sdf_and_coverage():
    xx, yy = R.grid(21, 21)
    d = R.sd_circle(xx, yy, 10.5, 10.5, 5)
    cov = R.coverage(d)
    assert cov[10, 10] == 1.0 and cov[0, 0] == 0.0
    # area of the AA disc is close to pi r^2
    assert abs(cov.sum() - np.pi * 25) < 2.0


def test_polygon_sdf_inside_outside():
    xx, yy = R.grid(40, 40)
    d = R.sd_polygon(xx, yy, [(5, 5), (35, 5), (35, 35), (5, 35)])
    assert d[20, 20] < 0 and d[1, 1] > 0


def test_blur_preserves_mass():
    img = np.zeros((64, 64), np.float32)
    img[32, 32] = 100.0
    out = R.blur(img, 4)
    assert abs(out.sum() - 100.0) < 1.0
    assert out[32, 32] < 100.0


def test_paint_over_is_premultiplied():
    img = R.blank(4, 4)
    R.paint(img, np.ones((4, 4)), (255, 0, 0), a=0.5)
    assert np.allclose(img[0, 0], [0.5, 0, 0, 0.5])


def test_every_atlas_char_has_a_glyph():
    for ch in GlyphAtlas.CHARS:
        if ch.islower():
            assert ch.upper() in GLYPHS
        else:
            assert ch in GLYPHS, ch
        adv, strokes, _ = glyph_strokes(ch)
        assert adv >= 0


def test_size_ladder_snaps():
    assert ladder_size(43) == 44
    assert all(ladder_size(s) == s for s in SIZE_LADDER)


def test_shape_tessellation():
    from gfx.context import Shape
    s = Shape().circle(0, 0, 10, segments=12).segment(0, 0, 10, 0, 4, caps=False).ring(0, 0, 5, 2, segments=8)
    tris = sum(len(c) for c in s.chunks) + len(s.tris) // 6
    assert tris == 12 + 2 + 16


# ---------------------------------------------------------------- X smoke tests

def _display_ok():
    if not os.environ.get("DISPLAY"):
        return False
    try:
        from gfx import xrender as X
        X.x11.XOpenDisplay.argtypes = [X.c_char_p]
        X.x11.XOpenDisplay.restype = X.Display
        d = X.x11.XOpenDisplay(os.environ["DISPLAY"].encode())
        return bool(d) and X.has_render(d)
    except Exception:
        return False


needs_x = pytest.mark.skipif(not _display_ok(), reason="needs an X display with XRender")


def _themes():
    import themes
    return [n for n in themes.available() if n != "classic"]


@pytest.fixture(scope="module")
def renderer():
    from renderer import X11Renderer
    r = X11Renderer(display_name=os.environ.get("DISPLAY"), theme="classic")
    yield r
    r.close()


def _grab(r):
    from gfx.capture import read_rgb
    from renderer import _x11
    _x11.XSync(r.display, 0)
    return read_rgb(r.display, r.buffer, r.width, r.height)


@needs_x
@pytest.mark.parametrize("theme", _themes() if _display_ok() else [])
def test_theme_renders_everything(renderer, theme):
    import screens
    from engine import GameLoop, InputState
    from main import CountdownState, MatchResults
    from match_setup import create_game
    from menu_system import MatchSettings, MenuRenderer, MenuState, MenuSystem, SettingsSubmenu

    r = renderer
    r.set_theme(theme)
    assert r.theme.name == theme
    rng = random.Random(0)
    for stage in ("battlefield", "final_destination"):
        game = create_game(stage, "ZONE", "FLEX")
        loop = GameLoop(game, headless=True)
        for frame in range(90):
            for f in game.fighters:
                f.input = InputState(left=rng.random() < 0.3, right=rng.random() < 0.3,
                                     attack=rng.random() < 0.1, special=rng.random() < 0.05,
                                     shield=rng.random() < 0.05, jump=rng.random() < 0.05)
            loop.step()
            if frame == 40:
                r.add_hit_effect(x=600, y=400, knockback_magnitude=12, angle=45, damage=25)
                r.add_landing_dust(500, 510)
                r.add_reflect_effect(700, 400)
                r.add_frame_advantage_effect(650, 380, 3)
            r.render(game)
        img = _grab(r)
        assert img.std() > 10, f"{theme}/{stage} rendered a blank frame"
        game.match_over, game.winner = True, 0
        r.render(game)

    m = MenuSystem()
    mr = MenuRenderer(r)
    mr.set_menu(m)
    for state in (MenuState.TITLE, MenuState.CHARACTER_SELECT, MenuState.STAGE_SELECT, MenuState.MATCH_SETTINGS):
        m.state = state
        mr.render(30)
    m.state = MenuState.SETTINGS
    for sub in SettingsSubmenu:
        m.settings_menu_state.submenu = sub
        mr.render(30)

    game = create_game("battlefield", "RUSH", "GRAB")
    r.render(game)
    screens.countdown(r.theme, CountdownState(entrance_complete=True, countdown_frame=90, countdown_value=2))
    screens.pause(r.theme, game, MatchSettings(), 0, 10)
    screens.results(r.theme, MatchResults(0, "RUSH", "GRAB", 1, 0, 50.0, 120.0, 3, 2, 3600, "battlefield"),
                    90, 1, 10)
    assert _grab(r).std() > 10
