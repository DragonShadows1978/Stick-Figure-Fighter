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


def training_overlay(theme, game_state, tm, frame: int):
    """Training-mode HUD: CPU status, toggles, frame data, input history, move list, combos, help."""
    t = theme
    g = t.g
    w, h = t.w, t.h
    pal = t.palette
    sx, sy = t.r.shake_offset

    # --- status pills (top center, under any timer)
    pills = []
    if tm.cpu_controller.state.enabled:
        st = tm.cpu_controller.state
        pills.append((f"CPU · {st.behavior.name.replace('_', ' ')} · LV {st.difficulty.value}", pal["warn"]))
    for flag, label in ((tm.infinite_jumps, "INF JUMPS"), (tm.infinite_shields, "INF SHIELD"),
                        (tm.show_frame_advantage, "FRAME ADV")):
        if flag:
            pills.append((label, pal["accent2"]))
    rec = tm.recording_state
    if rec.is_recording:
        pills.append((f"● REC {len(rec.recorded_inputs)}F", pal["bad"]))
    elif rec.is_playing:
        pills.append((f"PLAY {rec.playback_index}/{len(rec.recorded_inputs)}", pal["good"]))
    if pills:
        widths = [t.measure(p, 13) + 28 for p, _ in pills]
        x = w / 2 - (sum(widths) + 10 * (len(pills) - 1)) / 2
        for (label, col), pw in zip(pills, widths):
            t.ui_panel(x, 16, pw, 28, 0.5, col)
            t.text(x + pw / 2, 30, label, 13, col, align="center", valign="middle")
            x += pw + 10

    # --- frame data cards above the HUD cards
    if tm.frame_data_overlay.enabled:
        for i, fighter in enumerate(game_state.fighters[:2]):
            lines = tm.get_frame_data_text(fighter)
            if not lines:
                continue
            bw, bh = 190, 22 + 18 * len(lines)
            bx = 30 if i == 0 else w - 30 - bw
            by = h - 150 - bh
            t.ui_panel(bx, by, bw, bh, 0.0)
            for k, line in enumerate(lines):
                t.text(bx + 14, by + 20 + k * 18, line, 12, pal["text"] if k == 0 else pal["text_dim"],
                       valign="middle")

    # --- input history columns
    if tm.show_input_history:
        for i, hist in enumerate((tm.input_history.get_p1_history(), tm.input_history.get_p2_history())):
            pw, ph = 112, 268
            px = 16 if i == 0 else w - 16 - pw
            py = 70
            t.ui_panel(px, py, pw, ph, 0.0)
            t.text(px + pw / 2, py + 18, f"P{i + 1} INPUTS", 12, pal["p1"] if i == 0 else pal["p2"],
                   align="center", valign="middle")
            for k, entry in enumerate(list(reversed(hist[-13:]))):
                acts = entry.get_active_inputs()
                if not acts:
                    continue
                s = " ".join(acts)
                has_dir = any(c in s for c in "←→↑↓")
                has_btn = any(c in s for c in "JASGZ")
                col = pal["warn"] if has_dir and has_btn else pal["good"] if has_btn else pal["accent2"]
                t.text(px + 12, py + 42 + k * 17, s, 13, col, valign="middle", alpha=1 - k * 0.05)

    # --- move list
    ml = tm.move_list
    if ml.enabled and ml._cached_moves:
        moves = ml.get_visible_moves()
        pw = 400
        px = w - pw - 20 - (128 if tm.show_input_history else 0)
        py = 64
        ph = 64 + len(moves) * 20
        t.ui_panel(px, py, pw, ph, 0.4, pal["accent"])
        name = ml.current_character_type.name if ml.current_character_type else "?"
        t.text(px + 16, py + 22, f"{name} · MOVE LIST", 15, pal["accent"], valign="middle")
        cols = (("MOVE", 16, "left"), ("INPUT", 160, "left"), ("START", 270, "right"), ("ACTIVE", 330, "right"),
                ("DMG", 380, "right"))
        for lab, cx, al in cols:
            t.text(px + cx, py + 44, lab, 10, pal["text_dim"], valign="middle", align=al)
        for k, mv in enumerate(moves):
            yy = py + 64 + k * 20
            t.text(px + 16, yy, mv.name[:15], 12, pal["text"], valign="middle")
            t.text(px + 160, yy, mv.input_notation, 12, pal["warn"], valign="middle")
            t.text(px + 270, yy, f"{mv.startup_frames}f", 12, pal["good"], valign="middle", align="right")
            t.text(px + 330, yy, str(mv.active_frames), 12, pal["accent2"], valign="middle", align="right")
            t.text(px + 380, yy, f"{mv.damage:.0f}", 12, pal["bad"], valign="middle", align="right")
        if ml.can_scroll_up():
            t.text(px + pw - 18, py + 22, "↑", 13, pal["text"], align="center", valign="middle")
        if ml.can_scroll_down():
            t.text(px + pw - 18, py + ph - 14, "↓", 13, pal["text"], align="center", valign="middle")

    # --- hitbox visualization modes
    hv = tm.hitbox_viz
    if hv.mode.name != "NONE":
        for fighter in game_state.fighters:
            if hv.show_hurtboxes:
                hb = fighter.get_hurtbox_aabb()
                g.fill_rect(hb.x + sx, hb.y + sy, hb.width, hb.height, hv.hurtbox_color, 0.12)
                g.round_rect_outline(hb.x + sx, hb.y + sy, hb.width, hb.height, 3, 1.6, hv.hurtbox_color, 0.9)
            if hv.show_hitboxes:
                for b in fighter.get_active_hitbox_aabbs():
                    g.fill_rect(b.x + sx, b.y + sy, b.width, b.height, hv.hitbox_color, 0.25)
                    g.round_rect_outline(b.x + sx, b.y + sy, b.width, b.height, 3, 2, hv.hitbox_color, 1.0)
        if hv.show_collision_points:
            for cx, cy in hv.get_active_collision_points(game_state.frame):
                g.ring(cx + sx, cy + sy, 9, 2, hv.collision_color)
                g.circle(cx + sx, cy + sy, 3.5, hv.collision_color)

    # --- combo counter over the defender
    if tm.show_combo_counter:
        cc = tm.combo_counter
        for pid in range(2):
            data = cc.get_combo_display(pid)
            if not data:
                continue
            defender = game_state.fighters[1 - pid]
            x, y = defender.position.x + sx, defender.position.y - 110 + sy
            a = data["alpha"]
            n = data["count"]
            if data["is_active"]:
                col = pal["bad"] if n >= 10 else pal["warn"] if n >= 5 else pal["accent2"]
                t.text(x, y, f"{n} HITS", 26, col, align="center", valign="middle", role="title", alpha=a)
                t.text(x, y + 24, f"{data['damage']:.0f}%", 15, pal["text"], align="center", valign="middle",
                       alpha=a)
            else:
                t.text(x, y, f"{n}× DROPPED", 18, pal["text_dim"], align="center", valign="middle", alpha=a)
        b1, d1 = cc.get_best_combo(0)
        b2, d2 = cc.get_best_combo(1)
        if b1 > 1 or b2 > 1:
            t.ui_panel(w / 2 - 110, 54, 220, 30, 0.0)
            t.text(w / 2, 69, f"BEST  P1 {b1} ({d1:.0f}%)   P2 {b2} ({d2:.0f}%)", 11, pal["text_dim"],
                   align="center", valign="middle")

    # --- help bar
    help1 = "F1 CPU · F2 BEHAVIOR · F3 FRAME DATA · F4 LEVEL · F5 INF JUMPS · F6 FRAME ADV"
    help2 = "F7 INPUTS · F8 MOVES · TAB HITBOXES · R DAMAGE · T POSITIONS · F9 THEME"
    t.text(w / 2, h - 30, help1, 11, pal["text_dim"], align="center", valign="middle", alpha=0.85)
    t.text(w / 2, h - 14, help2, 11, pal["text_dim"], align="center", valign="middle", alpha=0.85)


def replay_browser(theme, browser, frame: int):
    t = theme
    w, h = t.w, t.h
    pal = t.palette
    t.ui_backdrop(frame, "dim")
    t.ui_title(w / 2, 66, "REPLAYS", 44, frame=frame)
    if browser.is_empty:
        t.ui_panel(w / 2 - 260, 230, 520, 140, 0.0)
        t.text(w / 2, 285, "NO REPLAYS SAVED YET", 22, pal["text"], align="center", valign="middle")
        t.text(w / 2, 325, "Finished matches are recorded automatically", 14, pal["text_dim"], align="center",
               valign="middle")
    else:
        y = 120
        for idx, entry in browser.get_visible_entries():
            sel = idx == browser.cursor
            t.ui_panel(160, y, w - 320, 58, 1.0 if sel else 0.0, pal["accent"] if sel else None)
            if sel:
                t.ui_arrow(140 + math.sin(frame * 0.2) * 3, y + 29, 1, pal["accent"], 8)
            t.text(184, y + 24, entry.display_name, 18, pal["text"] if sel else pal["text_dim"], valign="middle")
            t.text(184, y + 44, entry.display_info, 12, pal["text_dim"], valign="middle")
            y += 68
    t.text(w / 2, h - 24, "↑↓ select   ·   Attack play   ·   X delete   ·   Special back", 13, pal["text_dim"],
           align="center", valign="middle")
