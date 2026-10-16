"""
Render a scripted fight through the real renderer and save frames as PNG.

    python3 tools/theme_shot.py --theme neon --stage battlefield --p1 ZONE --p2 FLEX \
        --frames 40,95,160 --out /tmp/shots

Needs an X display (Xvfb works). Inputs are scripted so the shot contains
movement, attacks, a projectile and hits.
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import FighterState, GameLoop, InputState  # noqa: E402
from gfx.capture import read_rgb, write_png  # noqa: E402
from match_setup import create_game  # noqa: E402
from renderer import X11Renderer  # noqa: E402


def scripted_inputs(frame, me, opp, rng, pid):
    """A busy little fight: approach, poke, throw specials, jump around."""
    inp = InputState()
    toward = opp.position.x > me.position.x
    dist = abs(opp.position.x - me.position.x)
    phase = (frame // 25 + pid * 2) % 8
    if phase in (0, 4):
        if dist > 150:
            inp.right, inp.left = toward, not toward
        else:
            inp.right, inp.left = not toward, toward
    elif phase == 1:
        inp.special = frame % 25 == 3
    elif phase == 2:
        inp.right, inp.left = toward, not toward
        inp.jump = frame % 25 in (2, 12)
        inp.attack = frame % 25 == 16
    elif phase == 3:
        inp.attack = frame % 8 == 0 and dist < 120
        inp.right, inp.left = (toward, not toward) if dist > 70 else (False, False)
    elif phase == 5:
        inp.shield = True
    elif phase == 6:
        inp.attack = frame % 10 == 0
        inp.up = frame % 20 < 10
    else:
        inp.down = frame % 25 < 6
        inp.special = frame % 25 == 4
    return inp


def wire_effects(loop, r):
    def on_hit(att, dfd, hb, x, y, kb):
        r.add_hit_effect(x=x, y=y, knockback_magnitude=kb, angle=hb.angle, attacker_color=att.color,
                         damage=hb.damage)
        r.add_damage_flash(dfd)

    def on_proj(p, dfd, x, y, kb):
        r.add_hit_effect(x=x, y=y, knockback_magnitude=kb, angle=p.hitbox.angle, attacker_color=p.color,
                         damage=p.hitbox.damage)
        r.add_damage_flash(dfd)

    loop.on_hit = on_hit
    loop.on_projectile_hit = on_proj
    loop.on_reflect = lambda p, f, x, y: r.add_reflect_effect(x, y)
    loop.on_land = lambda f, x, y: r.add_landing_dust(x, y)


def inject_effects(r, game):
    """One of every effect: sparks at 4 damage tiers, dust, reflect, frame advantage, damage flash."""
    w = r.width
    for i, dmg in enumerate((6, 15, 25, 35)):
        r.add_hit_effect(x=200 + i * 140, y=300, knockback_magnitude=6 + i * 5, angle=30 + i * 20,
                         attacker_color=(255, 255, 255), damage=dmg)
    r.add_landing_dust(w - 380, 510)
    r.add_reflect_effect(w - 220, 300)
    r.add_frame_advantage_effect(w - 120, 330, 4)
    r.add_frame_advantage_effect(w - 60, 330, -6)
    for f in game.fighters:
        r.add_damage_flash(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="neon")
    ap.add_argument("--stage", default="battlefield")
    ap.add_argument("--p1", default="ZONE")
    ap.add_argument("--p2", default="FLEX")
    ap.add_argument("--frames", default="60,120,180")
    ap.add_argument("--out", default=".")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--damage", type=float, default=0.0, help="starting damage for both")
    ap.add_argument("--sync", action="store_true", help="XSync every frame (measures server time too)")
    ap.add_argument("--fx-at", type=int, default=-1,
                    help="inject one of every effect type at this frame (capture a few frames later)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    r = X11Renderer(display_name=os.environ.get("DISPLAY", ":99"), theme=args.theme)
    t0 = time.perf_counter()
    game = create_game(args.stage, args.p1, args.p2, stock_count=3)
    for f in game.fighters:
        f.damage_percent = args.damage
    loop = GameLoop(game, headless=True)
    wire_effects(loop, r)
    rng = random.Random(1)
    wanted = sorted(int(x) for x in args.frames.split(","))
    times = []
    for frame in range(wanted[-1] + 1):
        a, b = game.fighters
        a.input = scripted_inputs(frame, a, b, rng, 0)
        b.input = scripted_inputs(frame, b, a, rng, 1)
        loop.step()
        if frame == args.fx_at:
            inject_effects(r, game)
        ts = time.perf_counter()
        r.render(game)
        if args.sync:
            from renderer import _x11
            _x11.XSync(r.display, 0)
        elif r.gfx:
            r.gfx.flush()
        times.append(time.perf_counter() - ts)
        if frame in wanted:
            from renderer import _x11
            _x11.XSync(r.display, 0)
            path = os.path.join(args.out, f"{args.prefix}{args.theme}_{args.stage}_{frame}.png")
            write_png(path, read_rgb(r.display, r.buffer, r.width, r.height))
            print("wrote", path)
    steady = sorted(times[10:]) or times
    print(f"theme={args.theme} first-frame={times[0]*1000:.0f}ms median={steady[len(steady)//2]*1000:.1f}ms "
          f"p95={steady[int(len(steady)*0.95)]*1000:.1f}ms total={time.perf_counter()-t0:.1f}s")
    r.close()


if __name__ == "__main__":
    main()
