"""
Screenshot every menu screen for a theme.

    python3 tools/menu_shot.py --theme neon --out /tmp/menus
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from characters import CharacterType  # noqa: E402
from gfx.capture import read_rgb, write_png  # noqa: E402
from menu_system import GameModeSelection, MenuRenderer, MenuState, MenuSystem, SettingsSubmenu  # noqa: E402
from renderer import X11Renderer, _x11  # noqa: E402

SCREENS = {
    "title": lambda m: setattr(m, "state", MenuState.TITLE),
    "chars": lambda m: (setattr(m, "state", MenuState.CHARACTER_SELECT),
                        setattr(m.char_select, "p1_cursor", 1), setattr(m.char_select, "p2_cursor", 3),
                        setattr(m.char_select, "p2_locked", CharacterType.FLEX)),
    "stages": lambda m: (setattr(m, "state", MenuState.STAGE_SELECT),
                         setattr(m.char_select, "p1_locked", CharacterType.ZONE),
                         setattr(m.char_select, "p2_locked", CharacterType.FLEX)),
    "rules": lambda m: (setattr(m, "state", MenuState.MATCH_SETTINGS),
                        setattr(m.char_select, "p1_locked", CharacterType.RUSH),
                        setattr(m.char_select, "p2_locked", CharacterType.GRAB),
                        setattr(m.match_settings_state, "cursor", 2)),
    "settings": lambda m: (setattr(m, "state", MenuState.SETTINGS),
                           setattr(m.settings_menu_state, "main_cursor", 2)),
    "audio": lambda m: (setattr(m, "state", MenuState.SETTINGS),
                        setattr(m.settings_menu_state, "submenu", SettingsSubmenu.AUDIO)),
    "controls": lambda m: (setattr(m, "state", MenuState.SETTINGS),
                           setattr(m.settings_menu_state, "submenu", SettingsSubmenu.CONTROLS)),
    "gameplay": lambda m: (setattr(m, "state", MenuState.SETTINGS),
                           setattr(m.settings_menu_state, "submenu", SettingsSubmenu.GAMEPLAY),
                           setattr(m.settings_menu_state, "gameplay_cursor", 3)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="neon")
    ap.add_argument("--out", default=".")
    ap.add_argument("--screens", default=",".join(SCREENS))
    ap.add_argument("--frame", type=int, default=120)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    r = X11Renderer(display_name=os.environ.get("DISPLAY", ":99"), theme=args.theme)
    for name in args.screens.split(","):
        m = MenuSystem()
        import themes
        m.available_themes = themes.available()
        SCREENS[name](m)
        mr = MenuRenderer(r)
        mr.set_menu(m)
        mr.render(args.frame)
        _x11.XSync(r.display, 0)
        path = os.path.join(args.out, f"menu_{args.theme}_{name}.png")
        write_png(path, read_rgb(r.display, r.buffer, r.width, r.height))
        print("wrote", path)
    r.close()


if __name__ == "__main__":
    main()
