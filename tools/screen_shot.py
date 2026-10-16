"""
Screenshot the in-game screens (countdown, pause, results) for a theme.

    python3 tools/screen_shot.py --theme neon --out /tmp/screens
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import screens  # noqa: E402
from gfx.capture import read_rgb, write_png  # noqa: E402
from main import CountdownState, MatchResults  # noqa: E402
from match_setup import create_game  # noqa: E402
from menu_system import MatchSettings  # noqa: E402
from renderer import X11Renderer, _x11  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="neon")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    r = X11Renderer(display_name=os.environ.get("DISPLAY", ":99"), theme=args.theme)
    t = r.theme

    def save(name):
        _x11.XSync(r.display, 0)
        path = os.path.join(args.out, f"screen_{args.theme}_{name}.png")
        write_png(path, read_rgb(r.display, r.buffer, r.width, r.height))
        print("wrote", path)

    game = create_game("battlefield", "RUSH", "ZONE")
    cs = CountdownState(entrance_complete=True, countdown_frame=60 + 70, countdown_value=2)
    r.render(game)
    screens.countdown(t, cs)
    save("countdown")

    game.paused = True
    game.frame = 3725
    r.render(game)
    screens.pause(t, game, MatchSettings(stock_count=3, time_limit=5), 1, 40)
    save("pause")

    # training overlay with every panel on
    import random
    from characters import CharacterType, get_character_attacks
    from training_mode import CPUBehavior, HitboxVisualizationMode, TrainingModeManager
    from engine import GameLoop
    tgame = create_game("battlefield", "FLEX", "GRAB")
    tm = TrainingModeManager()
    tm.cpu_controller.enable(CPUBehavior.RANDOM)
    tm.frame_data_overlay.enabled = True
    tm.show_input_history = True
    tm.move_list.set_character(CharacterType.FLEX, get_character_attacks(CharacterType.FLEX))
    tm.move_list.enabled = True
    tm.hitbox_viz.mode = HitboxVisualizationMode.BOTH
    tm.infinite_shields = True
    loop = GameLoop(tgame, headless=True)
    rng = random.Random(3)
    keys = ("left", "right", "up", "down", "jump", "attack", "special", "shield", "grab")
    for fr in range(150):
        p1 = tgame.fighters[0]
        p1.input.right = fr % 40 < 20
        p1.input.attack = fr % 25 == 0
        cpu = tm.get_cpu_inputs(tgame.fighters[1], p1, fr)
        for k in keys:
            setattr(tgame.fighters[1].input, k, bool(cpu.get(k)))
        for i, f in enumerate(tgame.fighters):
            if rng.random() < 0.5:
                tm.record_input_for_history(i, {k: getattr(f.input, k) for k in keys}, fr)
        loop.step()
        tm.update(tgame)
    r.render(tgame)
    screens.training_overlay(t, tgame, tm, 150)
    save("training")

    res = MatchResults(winner=0, p1_character="RUSH", p2_character="ZONE", p1_stocks=2, p2_stocks=0,
                       p1_damage=87.5, p2_damage=142.0, p1_max_combo=6, p2_max_combo=3, total_frames=9420,
                       stage_name="final_destination")
    screens.results(t, res, 90, 0, 200)
    save("results")
    r.close()


if __name__ == "__main__":
    main()
