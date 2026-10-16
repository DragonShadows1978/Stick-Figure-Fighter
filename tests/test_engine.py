"""
Engine tests that use no Stick Figure Fighter content.

These build fighters, moves and stages from scratch, which is exactly what a
different game built on `engine` would do.
"""
import os
import subprocess
import sys
import time

from engine import (AttackData, AttackType, FighterState, FighterStats, GameConfig,
                    GameLoop, GameState, Hitbox, InputState, Platform, ProjectileSpec,
                    Stage, Vector2, run_headless_match)
from engine.config import FOOT_OFFSET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def flat_stage():
    ground = Platform(x=0, y=500, width=1000, height=50, passthrough=False)
    return Stage(name="Test", platforms=[], main_platform=ground,
                 blast_zones={'left': -500, 'right': 1500, 'top': -500, 'bottom': 1200},
                 spawn_points=[Vector2(300, 500 - FOOT_OFFSET), Vector2(700, 500 - FOOT_OFFSET)])


def hit(dmg=5.0, active=(1, 5), offset=Vector2(20, -50), w=40):
    return Hitbox(offset=offset, width=w, height=30, damage=dmg, knockback_base=3.0,
                  knockback_scaling=0.05, angle=45, active_frames=active)


def move(slot, **kw):
    return AttackData(attack_type=slot, total_frames=kw.pop("total", 20),
                      hitboxes=kw.pop("hitboxes", [hit()]), startup_frames=1,
                      recovery_frames=10, **kw)


def make_game(moveset=None, config=None):
    stats = [FighterStats(name="A", moveset=moveset), FighterStats(name="B")]
    return GameState(flat_stage(), stats, config=config)


def press(loop, fighter, **buttons):
    fighter.input = InputState(**buttons)
    loop.step()


def test_engine_imports_no_game_modules_or_numpy():
    code = ("import sys, engine; "
            "bad=[m for m in sys.modules if m.split('.')[0] in "
            "('numpy','characters','stages','game_engine','match_setup','renderer')]; "
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=REPO_ROOT, check=True).stdout.strip()
    assert out == "[]"


def test_projectile_spec_spawns_mirrored_projectile():
    spec = ProjectileSpec(offset=Vector2(30, -40), velocity=Vector2(5, -1),
                          hitbox=hit(active=(0, 999)), lifetime=60)
    game = make_game({AttackType.NEUTRAL_SPECIAL: move(AttackType.NEUTRAL_SPECIAL, projectile=spec)})
    loop = GameLoop(game, headless=True)
    fighter = game.fighters[0]
    fighter.facing_right = False
    x, y = fighter.position.x, fighter.position.y

    press(loop, fighter, special=True)

    assert fighter.state == FighterState.SPECIAL
    assert len(game.projectiles) == 1
    p = game.projectiles[0]
    assert p.velocity.x == -5 and p.velocity.y == -1
    assert p.position.x == x - 30 - 5  # spawned mirrored, then moved one frame
    assert p.position.y == y - 40 - 1
    # Each spawn gets its own hitbox, so reflect() can't corrupt the move data
    assert p.hitbox is not spec.hitbox


def test_on_start_hook_runs():
    calls = []
    game = make_game({AttackType.DOWN_SPECIAL: move(
        AttackType.DOWN_SPECIAL, on_start=lambda loop, f: calls.append(f.player_id))})
    loop = GameLoop(game, headless=True)
    press(loop, game.fighters[0], down=True, special=True)
    assert calls == [0]


def test_counter_and_reflector_flags_enter_their_states():
    game = make_game({
        AttackType.DOWN_SPECIAL: move(AttackType.DOWN_SPECIAL, is_counter=True),
        AttackType.NEUTRAL_SPECIAL: move(AttackType.NEUTRAL_SPECIAL, is_reflector=True),
    })
    loop = GameLoop(game, headless=True)
    f = game.fighters[0]
    press(loop, f, down=True, special=True)
    assert f.state == FighterState.COUNTER and f.counter_active

    game = make_game({AttackType.NEUTRAL_SPECIAL: move(AttackType.NEUTRAL_SPECIAL, is_reflector=True)})
    loop = GameLoop(game, headless=True)
    f = game.fighters[0]
    press(loop, f, special=True)
    assert f.state == FighterState.REFLECTING


def test_missing_slot_uses_fallback_attack():
    sentinel = move(AttackType.JAB, total=99)
    game = make_game(moveset={})
    loop = GameLoop(game, headless=True, fallback_attack=lambda slot: sentinel)
    f = game.fighters[0]
    press(loop, f, attack=True)
    assert f.current_attack is sentinel


def test_per_game_config_is_respected():
    floaty = GameConfig(gravity=0.1)
    heavy = GameConfig(gravity=1.2)
    heights = []
    for cfg in (floaty, heavy):
        game = make_game(config=cfg)
        loop = GameLoop(game, headless=True)
        f = game.fighters[0]
        press(loop, f, jump=True)
        for _ in range(10):
            press(loop, f)
        heights.append(f.position.y)
    assert heights[0] < heights[1]  # lower gravity = higher after 10 frames (y grows downward)
    assert game.fighters[0].stocks == heavy.stock_count


def test_headless_run_is_not_tied_to_wall_clock():
    game = make_game()
    start = time.perf_counter()
    run_headless_match(game, lambda state, pid: InputState(), max_frames=6000)
    assert game.frame == 6000
    assert time.perf_counter() - start < 5  # real time would be 100 seconds


def test_landing_does_not_sink_feet_into_platform():
    game = make_game()
    loop = GameLoop(game, headless=True)
    f = game.fighters[0]
    f.position.y = 100
    f.grounded = False
    f.state = FighterState.FALLING
    deepest = 0.0
    for _ in range(120):
        press(loop, f)
        deepest = max(deepest, f.feet_y)
    assert f.grounded
    assert deepest == game.stage.main_platform.y
    assert f.state == FighterState.IDLE  # LANDING resolves to IDLE


def test_walking_off_a_ledge_falls():
    game = make_game()
    loop = GameLoop(game, headless=True)
    f = game.fighters[0]
    for _ in range(150):  # ground spans x 0..1000, fighter starts at 300 walking left
        press(loop, f, left=True)
    assert not f.grounded
    assert f.feet_y > game.stage.main_platform.y


def test_holding_down_drops_through_passthrough_platform():
    game = make_game()
    game.stage.platforms.append(Platform(x=200, y=300, width=200, passthrough=True))
    loop = GameLoop(game, headless=True)
    f = game.fighters[0]
    f.position.y = 100
    f.grounded = False
    for _ in range(60):
        press(loop, f)
    assert f.grounded and f.feet_y == 300
    for _ in range(60):
        press(loop, f, down=True)
    assert f.grounded and f.feet_y == game.stage.main_platform.y


def test_airborne_fighter_with_feet_on_surface_lands_instead_of_falling_through():
    game = make_game()
    loop = GameLoop(game, headless=True)
    f = game.fighters[0]
    f.respawn(game.get_spawn_point(0))  # spawn point has feet exactly on the floor
    assert not f.grounded
    for _ in range(10):
        press(loop, f)
    assert f.grounded and f.feet_y == game.stage.main_platform.y


def _counter_exchange(counter_move):
    """A counters B's jab; returns B's damage afterwards."""
    jab = move(AttackType.JAB, hitboxes=[hit(dmg=4.0, active=(1, 6))])
    stats = [FighterStats(name="A", moveset={AttackType.DOWN_SPECIAL: counter_move}),
             FighterStats(name="B", moveset={AttackType.JAB: jab})]
    game = GameState(flat_stage(), stats)
    a, b = game.fighters
    b.position.x = a.position.x + 40
    b.facing_right = False
    loop = GameLoop(game, headless=True)
    a.input = InputState(down=True, special=True)
    b.input = InputState(attack=True)
    loop.step()
    for _ in range(40):
        a.input = InputState()
        b.input = InputState()
        loop.step()
    return a, b


def test_counter_boost_applies_per_hit_without_mutating_move_data():
    counter = move(AttackType.DOWN_SPECIAL, total=40, is_counter=True,
                   hitboxes=[hit(dmg=10.0, active=(10, 14))])
    for _ in range(3):  # same move object reused across matches, like the character cache
        a, b = _counter_exchange(counter)
        assert a.damage_percent == 0
        assert b.damage_percent == 15.0
    assert counter.hitboxes[0].damage == 10.0


def test_multi_frame_hitbox_hits_shield_once():
    jab = move(AttackType.JAB, hitboxes=[hit(dmg=10.0, active=(1, 8))])
    stats = [FighterStats(name="A", moveset={AttackType.JAB: jab}), FighterStats(name="B")]
    game = GameState(flat_stage(), stats)
    a, b = game.fighters
    b.position.x = a.position.x + 40
    loop = GameLoop(game, headless=True)
    b.input = InputState(shield=True)
    loop.step()
    shield_before = b.shield_health
    a.input = InputState(attack=True)
    for _ in range(12):
        loop.step()
        a.input = InputState()
    # one hit = 15 shield damage, plus ~0.1/frame passive drain while held
    assert shield_before - b.shield_health < 17
