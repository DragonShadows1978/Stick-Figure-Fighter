"""
Themed in-game screens: countdown, pause, results. Drawn with the active
theme's UI kit; main.py falls back to its original drawing for Classic.
"""
import math


def countdown(theme, cs):
    """Overlay for the pre-match entrance + 3-2-1-GO countdown (stage already drawn)."""
    t = theme
    w, h = t.w, t.h
    if not cs.entrance_complete:
        t.g.fill_rect(0, 0, w, h, (0, 0, 0), 0.25)
        t.ui_burst(w / 2, h / 2 - 40, "READY", min(0.6, cs.countdown_frame / 60.0), t.palette["accent2"], 96)
        return
    f = cs.countdown_frame - cs.entrance_frames
    t.g.fill_rect(0, 0, w, h, (0, 0, 0), 0.18)
    if cs.countdown_value > 0:
        t.ui_burst(w / 2, h / 2 - 40, str(cs.countdown_value), (f % 60) / 60.0, t.palette["accent"], 150)
    else:
        t.ui_burst(w / 2, h / 2 - 40, "GO!", min(0.95, (f % 60) / 60.0), t.palette["good"], 150)


def pause(theme, game_state, saved_settings, selection: int, frame: int):
    t = theme
    w, h = t.w, t.h
    t.g.fill_rect(0, 0, w, h, (0, 0, 0), 0.45)
    pw, ph = 400, 334
    px, py = (w - pw) / 2, (h - ph) / 2
    t.ui_panel(px, py, pw, ph, 0.6, t.palette["accent"])
    t.ui_title(w / 2, py + 52, "PAUSED", 44, frame=frame)
    secs = int(game_state.frame / 60)
    info = f"{secs // 60}:{secs % 60:02d}"
    if saved_settings:
        tl = "no limit" if saved_settings.time_limit == 0 else f"{saved_settings.time_limit} min"
        info += f"  ·  {saved_settings.stock_count} stocks  ·  {tl}"
    t.text(w / 2, py + 98, info, 14, t.palette["text_dim"], align="center", valign="middle")
    labels = ["RESUME", "CHARACTER SELECT", "RETURN TO TITLE"]
    accents = [t.palette["good"], t.palette["accent2"], t.palette["bad"]]
    for i, lab in enumerate(labels):
        t.ui_button(px + 40, py + 130 + i * 66, pw - 80, 52, lab, selection == i, frame, accents[i], size=20)


def results(theme, r, timer: int, selection: int, frame: int):
    """Post-match results: winner spotlight, both fighters, stats, rematch / character select."""
    t = theme
    w, h = t.w, t.h
    t.ui_backdrop(frame, "dim")
    if r.winner is not None:
        name = r.p1_character if r.winner == 0 else r.p2_character
        wcol = t.color_for_name(name, r.winner)
        t.ui_title(w / 2, 82, f"{name} WINS!", 64, color=wcol, frame=frame)
    else:
        wcol = t.palette["accent"]
        t.ui_title(w / 2, 82, "DRAW!", 64, frame=frame)
    t.text(w / 2, 132, f"{r.stage_name.replace('_', ' ').upper()}  ·  {r.match_time_str}", 14,
           t.palette["text_dim"], align="center", valign="middle")
    for pid, side in ((0, -1), (1, 1)):
        name = r.p1_character if pid == 0 else r.p2_character
        col = t.color_for_name(name, pid)
        winner = r.winner == pid
        cx = w / 2 + side * 290
        t.ui_figure(cx, 300, col, frame + pid * 20, scale=1.7, pose="victory" if winner else "idle",
                    facing=-side)
        pw, ph = 300, 150
        px, py = cx - pw / 2, 420
        t.ui_panel(px, py, pw, ph, 1.0 if winner else 0.0, col if winner else None)
        t.text(cx, py + 30, f"P{pid + 1}  {name}", 22, col, align="center", valign="middle", role="title")
        stocks = r.p1_stocks if pid == 0 else r.p2_stocks
        dmg = r.p1_damage if pid == 0 else r.p2_damage
        combo = r.p1_max_combo if pid == 0 else r.p2_max_combo
        for k, (lab, val) in enumerate((("STOCKS LEFT", str(stocks)), ("DAMAGE", f"{dmg:.0f}%"),
                                        ("BEST COMBO", str(combo)))):
            yy = py + 66 + k * 26
            t.text(px + 28, yy, lab, 13, t.palette["text_dim"], valign="middle")
            t.text(px + pw - 28, yy, val, 17, t.palette["text"], align="right", valign="middle")
    if timer >= 60:
        bw, bh = 290, 56
        for i, lab in enumerate(("REMATCH", "CHARACTER SELECT")):
            x = w / 2 - bw - 20 + i * (bw + 40)
            t.ui_button(x, 612, bw, bh, lab, selection == i, frame, t.palette["accent"] if i == 0 else
                        t.palette["accent2"], size=19)
    else:
        dots = "." * (1 + (timer // 15) % 3)
        t.text(w / 2, 640, dots, 28, t.palette["text_dim"], align="center", valign="middle")
