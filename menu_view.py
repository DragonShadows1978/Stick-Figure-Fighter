"""
Themed menu screens.

One layout for every theme: screens are composed from the active theme's UI
kit (backdrop, panels, buttons, bars, toggles, figures, text), so Neon,
Sketchbook and Silhouette each restyle every menu without their own layout
code. The Classic theme keeps the original MenuRenderer drawing.
"""
import math
import time
from typing import Optional

from characters import CharacterType, get_character_stats
from menu_system import (GameModeSelection, MenuState, SettingsSubmenu, TIME_LABELS, TIME_OPTIONS)

STAGE_INFO = {
    "battlefield": ("Battlefield", "Three platforms · the classic layout"),
    "final_destination": ("Final Destination", "One flat stage · nowhere to hide"),
}


class ThemedMenuView:
    def __init__(self, renderer):
        self.r = renderer
        self.menu = None
        self._gamepads = (0, -1e9)   # (count, last scan time)

    @property
    def t(self):
        return self.r.theme

    # ------------------------------------------------------------------ helpers

    def _footer(self, hint: str):
        t = self.t
        t.text(t.w / 2, t.h - 24, hint, 13, t.palette["text_dim"], align="center")
        import themes
        t.text(t.w - 20, t.h - 24, f"{themes.title(t.name)} · F9", 11, t.palette["text_dim"], align="right",
               alpha=0.7)

    def _header(self, title: str, frame: int, sub: Optional[str] = None, y: float = 66):
        t = self.t
        t.ui_title(t.w / 2, y, title, 44, frame=frame)
        if sub:
            t.text(t.w / 2, y + 42, sub, 15, t.palette["text_dim"], align="center", valign="middle")

    def _char_color(self, char: Optional[CharacterType], idx: int):
        return self.t.color_for_name(char.name if char else None, idx)

    def _row(self, x, y, w, label, value, selected, frame, kind="value", frac=0.0, on=False, h=54):
        """A settings row: label on the left, value control on the right."""
        t = self.t
        if selected:
            t.ui_panel(x, y, w, h, 0.85 + 0.15 * math.sin(frame * 0.15), t.palette["accent"])
        lab_col = t.palette["text"] if selected else t.palette["text_dim"]
        t.text(x + 24, y + h / 2, label, 18, lab_col, valign="middle")
        right = x + w - 24
        if kind == "toggle":
            t.ui_toggle(right - 46, y + h / 2 - 11, on, frame)
        elif kind == "slider":
            t.text(right, y + h / 2, f"{int(round(frac * 100))}%", 16, lab_col, align="right", valign="middle")
            t.ui_bar(right - 230, y + h / 2 - 5, 170, 10, frac, t.palette["accent2"])
        elif kind == "value":
            vw = max(130, t.measure(value, 18) + 40)
            cx = right - vw / 2
            t.text(cx, y + h / 2, value, 18, t.palette["text"], align="center", valign="middle")
            if selected:
                bob = math.sin(frame * 0.2) * 2
                t.ui_arrow(cx - vw / 2 - bob, y + h / 2, -1, t.palette["accent"], 7)
                t.ui_arrow(cx + vw / 2 + bob, y + h / 2, 1, t.palette["accent"], 7)
        elif kind == "button":
            pass

    def _gamepad_count(self) -> int:
        count, when = self._gamepads
        now = time.monotonic()
        if now - when > 2.0:
            try:
                from gamepad_input import GamepadManager
                gp = GamepadManager()
                gp.scan_devices()
                count = len(gp.gamepads)
                gp.cleanup()
            except Exception:
                count = 0
            self._gamepads = (count, now)
        return count

    # ------------------------------------------------------------------ dispatch

    def render(self, frame: int):
        m = self.menu
        if m is None:
            return
        t = self.t
        t.ensure_built()
        t.fx.frame = frame
        state = m.state
        if state == MenuState.TITLE:
            t.ui_backdrop(frame, "menu")
            self._title(frame)
        else:
            t.ui_backdrop(frame, "dim")
            if state == MenuState.CHARACTER_SELECT:
                self._character_select(frame)
            elif state == MenuState.STAGE_SELECT:
                self._stage_select(frame)
            elif state == MenuState.MATCH_SETTINGS:
                self._match_settings(frame)
            elif state == MenuState.SETTINGS:
                self._settings(frame)

    # ------------------------------------------------------------------ title

    def _title(self, frame: int):
        t, m = self.t, self.menu
        w, h = t.w, t.h
        t.ui_title(w / 2, 128, "STICK FIGURE", 80, frame=frame)
        t.ui_title(w / 2, 214, "FIGHTER", 80, color=t.palette["accent2"], frame=frame + 31)
        t.ui_figure(200, 430, t.color_for_name("RUSH", 0), frame, scale=2.0, pose="attack", facing=1)
        t.ui_figure(w - 200, 430, t.color_for_name("ZONE", 1), frame + 40, scale=2.0, pose="guard", facing=-1)
        labels = ["VERSUS", "TRAINING", "REPLAYS", "SETTINGS", "QUIT"]
        subs = ["Two players, one keyboard", "Practice against the CPU", "Watch saved matches",
                "Audio, controls, gameplay & visual theme", "Leave the game"]
        accents = [t.palette["accent"], t.palette["accent2"], t.palette["warn"], t.palette["good"],
                   t.palette["bad"]]
        bw, bh = 340, 54
        y = 300
        for i, lab in enumerate(labels):
            sel = m.title_state.cursor == i
            t.ui_button(w / 2 - bw / 2, y, bw, bh, lab, sel, frame, accents[i], sublabel=subs[i] if sel else None,
                        size=22)
            y += 64
        self._footer("↑↓ select   ·   Attack / Jump confirm")

    # ------------------------------------------------------------------ character select

    def _character_select(self, frame: int):
        t, m = self.t, self.menu
        w = t.w
        cs = m.char_select
        training = m.game_mode == GameModeSelection.TRAINING
        if training:
            self._header("TRAINING", frame, "P1 picks a fighter · ↑↓ picks the CPU's fighter")
        else:
            self._header("CHOOSE YOUR FIGHTER", frame)
        cw, ch, gap = 250, 320, 24
        x0 = (w - (4 * cw + 3 * gap)) / 2
        y0 = 132
        p1c, p2c = t.palette["p1"], t.palette["p2"]
        for i, char in enumerate(m.CHARACTERS):
            x = x0 + i * (cw + gap)
            info = m.get_character_info(char)
            col = self._char_color(char, i)
            hovered = cs.p1_cursor == i or cs.p2_cursor == i
            locked = cs.p1_locked == char or cs.p2_locked == char
            t.ui_panel(x, y0, cw, ch, 1.0 if hovered else 0.0, col if hovered else None)
            pose = "victory" if locked else ("attack" if hovered else "idle")
            t.ui_figure(x + cw / 2, y0 + 168, col, frame + i * 17, scale=1.45, pose=pose,
                        facing=1 if i < 2 else -1)
            t.text(x + cw / 2, y0 + 262, info["name"], 30, col, align="center", valign="middle",
                   role="title")
            t.text(x + cw / 2, y0 + 296, info["description"], 12, t.palette["text_dim"], align="center",
                   valign="middle")
        # cursors
        for pid, cursor, locked, col, label in (
                (0, cs.p1_cursor, cs.p1_locked, p1c, "P1"),
                (1, cs.p2_cursor, cs.p2_locked, (150, 150, 170) if training else p2c, "CPU" if training else "P2")):
            x = x0 + cursor * (cw + gap) + cw / 2
            bob = math.sin(frame * 0.15 + pid * math.pi) * 4
            y = y0 - 18 + bob if pid == 0 else y0 + ch + 20 + bob
            txt = f"{label} READY" if locked else label
            tw = t.measure(txt, 15) + 26
            t.ui_panel(x - tw / 2, y - 14, tw, 28, 1.0, col)
            t.text(x, y, txt, 15, col, align="center", valign="middle")
        # stat panels for hovered characters
        for pid, cursor in ((0, cs.p1_cursor), (1, cs.p2_cursor)):
            char = m.CHARACTERS[cursor]
            info = m.get_character_info(char)
            col = self._char_color(char, cursor)
            pw, ph = 420, 132
            px = 104 if pid == 0 else w - 104 - pw
            py = 506
            t.ui_panel(px, py, pw, ph, 0.0)
            tag = ("CPU" if training else "P2") if pid == 1 else "P1"
            t.text(px + 20, py + 28, f"{tag} · {info['name']}", 17, col)
            for k, key in enumerate(("speed", "weight", "power", "recovery")):
                ry = py + 52 + k * 19
                t.text(px + 20, ry + 5, key.upper(), 11, t.palette["text_dim"], valign="middle")
                t.ui_bar(px + 120, ry, pw - 150, 9, info[key] / 10.0, col)
        hint = ("←→ P1 fighter   ·   ↑↓ CPU fighter   ·   Attack confirm   ·   Special back" if training else
                "←→ choose   ·   Attack / Jump lock in   ·   Special back")
        self._footer(hint)

    # ------------------------------------------------------------------ stage select

    def _stage_select(self, frame: int):
        t, m = self.t, self.menu
        w = t.w
        self._header("SELECT STAGE", frame)
        tw, th, gap = 500, 281, 60
        x0 = (w - (2 * tw + gap)) / 2
        y0 = 138
        for i, sid in enumerate(m.STAGES):
            x = x0 + i * (tw + gap)
            sel = m.stage_select.cursor == i
            t.ui_panel(x - 12, y0 - 12, tw + 24, th + 92, 1.0 if sel else 0.0,
                       t.palette["accent"] if sel else None)
            t.ui_stage_thumb(sid, x, y0, tw, th, frame, sel)
            name, info = STAGE_INFO.get(sid, (sid, ""))
            t.text(x + tw / 2, y0 + th + 34, name.upper(), 26, t.palette["text"] if sel else t.palette["text_dim"],
                   align="center", valign="middle", role="title")
            t.text(x + tw / 2, y0 + th + 62, info, 13, t.palette["text_dim"], align="center", valign="middle")
        self._matchup(w / 2, 600, frame, 0.75)
        self._footer("←→ choose   ·   Attack / Jump confirm   ·   Special back")

    def _matchup(self, cx, cy, frame, scale):
        t, m = self.t, self.menu
        cs = m.char_select
        training = m.game_mode == GameModeSelection.TRAINING
        for pid, char, side in ((0, cs.p1_locked, -1), (1, cs.p2_locked, 1)):
            if char is None:
                continue
            col = self._char_color(char, pid)
            fx = cx + side * 150
            t.ui_figure(fx, cy - 18 * scale, col, frame + pid * 30, scale=scale, pose="guard", facing=-side)
            name = get_character_stats(char).name
            tag = ("CPU" if training else "P2") if pid else "P1"
            t.text(fx + side * 60 * scale, cy, f"{tag} {name}", 17, col,
                   align="left" if side > 0 else "right", valign="middle")
        t.text(cx, cy, "VS", 30, t.palette["accent"], align="center", valign="middle", role="title")

    # ------------------------------------------------------------------ match settings

    def _match_settings(self, frame: int):
        t, m = self.t, self.menu
        w = t.w
        mss = m.match_settings_state
        self._header("MATCH RULES", frame)
        self._matchup(w / 2, 232, frame, 0.95)
        sid = m.STAGES[m.stage_select.cursor] if m.stage_select.locked is None else m.stage_select.locked
        t.text(w / 2, 270, STAGE_INFO.get(sid, (sid, ""))[0].upper(), 13, t.palette["text_dim"], align="center",
               valign="middle")
        pw = 560
        px = (w - pw) / 2
        py = 320
        t.ui_panel(px, py, pw, 270, 0.0)
        self._row(px + 20, py + 24, pw - 40, "STOCKS", str(mss.stock_count), mss.cursor == 0, frame)
        self._row(px + 20, py + 96, pw - 40, "TIME LIMIT", TIME_LABELS[mss.time_limit_idx], mss.cursor == 1,
                  frame)
        t.ui_button(px + pw / 2 - 150, py + 180, 300, 62, "FIGHT!", mss.cursor == 2, frame, t.palette["accent"],
                    size=28)
        self._footer("↑↓ navigate   ·   ←→ change   ·   Attack start   ·   Special back")

    # ------------------------------------------------------------------ settings

    def _settings(self, frame: int):
        sub = self.menu.settings_menu_state.submenu
        if sub == SettingsSubmenu.MAIN:
            self._settings_main(frame)
        elif sub == SettingsSubmenu.AUDIO:
            self._settings_audio(frame)
        elif sub == SettingsSubmenu.CONTROLS:
            self._settings_controls(frame)
        elif sub == SettingsSubmenu.GAMEPLAY:
            self._settings_gameplay(frame)

    def _settings_main(self, frame: int):
        t, m = self.t, self.menu
        w = t.w
        sms = m.settings_menu_state
        self._header("SETTINGS", frame)
        items = [("AUDIO", "Volume and mute"), ("CONTROLS", "Keyboard and gamepad layouts"),
                 ("GAMEPLAY", "Stocks, time, damage numbers, visual theme"), ("BACK", "Save and return")]
        bw, bh = 420, 64
        y = 170
        for i, (lab, sub) in enumerate(items):
            t.ui_button(w / 2 - bw / 2, y, bw, bh, lab, sms.main_cursor == i, frame, sublabel=sub, size=24)
            y += 86
        self._footer("↑↓ select   ·   Attack / Jump open   ·   Special back")

    def _settings_audio(self, frame: int):
        t, m = self.t, self.menu
        w = t.w
        sms = m.settings_menu_state
        audio = m.settings_manager.settings.audio
        self._header("AUDIO", frame)
        pw = 640
        px = (w - pw) / 2
        y = 170
        t.ui_panel(px, y - 20, pw, 330, 0.0)
        self._row(px + 20, y, pw - 40, "MASTER VOLUME", "", sms.audio_cursor == 0, frame, "slider",
                  frac=audio.master_volume)
        self._row(px + 20, y + 70, pw - 40, "EFFECTS VOLUME", "", sms.audio_cursor == 1, frame, "slider",
                  frac=audio.sfx_volume)
        self._row(px + 20, y + 140, pw - 40, "MUTE", "", sms.audio_cursor == 2, frame, "toggle", on=audio.muted)
        t.ui_button(w / 2 - 110, y + 220, 220, 54, "BACK", sms.audio_cursor == 3, frame)
        self._footer("↑↓ navigate   ·   ←→ adjust   ·   Special back")

    def _settings_controls(self, frame: int):
        t, m = self.t, self.menu
        w = t.w
        sms = m.settings_menu_state
        st = m.settings_manager.settings
        self._header("CONTROLS", frame)
        pw, ph, gap = 400, 290, 40
        x0 = (w - (2 * pw + gap)) / 2
        y0 = 140
        keys = {
            0: [("MOVE", "W A S D"), ("JUMP", "SPACE"), ("ATTACK", "J"), ("SPECIAL", "K"), ("SHIELD", "L")],
            1: [("MOVE", "ARROWS"), ("JUMP", "ENTER"), ("ATTACK", "NUM 1"), ("SPECIAL", "NUM 2"),
                ("SHIELD", "NUM 3")],
        }
        for pid, controls in ((0, st.player1_controls), (1, st.player2_controls)):
            x = x0 + pid * (pw + gap)
            sel = sms.controls_cursor == pid
            col = t.palette["p1"] if pid == 0 else t.palette["p2"]
            t.ui_panel(x, y0, pw, ph, 1.0 if sel else 0.0, col if sel else None)
            t.text(x + pw / 2, y0 + 34, f"PLAYER {pid + 1}", 24, col, align="center", valign="middle", role="title")
            for k, (act, key) in enumerate(keys[pid]):
                yy = y0 + 78 + k * 30
                t.text(x + 30, yy, act, 14, t.palette["text_dim"], valign="middle")
                t.text(x + pw - 30, yy, key, 15, t.palette["text"], align="right", valign="middle")
            mapping = controls.gamepad_mapping.upper()
            yy = y0 + ph - 36
            t.text(x + 30, yy, "GAMEPAD", 14, t.palette["text_dim"], valign="middle")
            t.text(x + pw - 70, yy, mapping, 16, t.palette["text"], align="center", valign="middle")
            if sel:
                bob = math.sin(frame * 0.2) * 2
                t.ui_arrow(x + pw - 70 - 52 - bob, yy, -1, t.palette["accent"], 6)
                t.ui_arrow(x + pw - 70 + 52 + bob, yy, 1, t.palette["accent"], 6)
        n = self._gamepad_count()
        status = f"{n} GAMEPAD{'S' if n != 1 else ''} CONNECTED" if n else "NO GAMEPADS DETECTED"
        t.text(w / 2, y0 + ph + 40, status, 14, t.palette["good"] if n else t.palette["text_dim"], align="center",
               valign="middle")
        t.ui_button(w / 2 - 110, y0 + ph + 76, 220, 54, "BACK", sms.controls_cursor == 2, frame)
        self._footer("↑↓ navigate   ·   ←→ gamepad layout   ·   Special back")

    def _settings_gameplay(self, frame: int):
        import themes
        t, m = self.t, self.menu
        w = t.w
        sms = m.settings_menu_state
        gp = m.settings_manager.settings.gameplay
        self._header("GAMEPLAY", frame)
        pw = 640
        px = (w - pw) / 2
        y = 150
        t.ui_panel(px, y - 20, pw, 420, 0.0)
        time_labels = {0: "None", 3: "3 min", 5: "5 min", 8: "8 min"}
        self._row(px + 20, y, pw - 40, "DEFAULT STOCKS", str(gp.default_stock_count), sms.gameplay_cursor == 0,
                  frame)
        self._row(px + 20, y + 66, pw - 40, "DEFAULT TIME", time_labels.get(gp.default_time_limit, "None"),
                  sms.gameplay_cursor == 1, frame)
        self._row(px + 20, y + 132, pw - 40, "DAMAGE NUMBERS", "", sms.gameplay_cursor == 2, frame, "toggle",
                  on=gp.show_damage_numbers)
        self._row(px + 20, y + 198, pw - 40, "VISUAL THEME", themes.title(gp.theme), sms.gameplay_cursor == 3,
                  frame)
        t.ui_button(w / 2 - 110, y + 290, 220, 54, "BACK", sms.gameplay_cursor == 4, frame)
        self._footer("↑↓ navigate   ·   ←→ adjust   ·   F9 cycles themes anywhere   ·   Special back")
