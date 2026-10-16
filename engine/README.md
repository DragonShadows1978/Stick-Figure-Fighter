# engine

A deterministic, fixed-timestep 2D platform-fighter simulation. Pure Python,
standard library only, no rendering and no game content.

Stick Figure Fighter is one game built on it. Characters, stages, rendering,
audio, input devices and ML encoding all live outside this package.

## What it does

- Movement: walking, air control, jumps and double jumps, fast fall, friction and gravity
- Platforms: solid floors and passthrough platforms (land from above, hold down to drop
  through), walking off ledges, an offset contact zone that stops landings from sinking in
- Combat: frame-windowed hitboxes against hurtboxes, damage-scaled knockback with weight,
  hitstun, DI, hit freeze, combos, one hit per attack per defender
- Defense: shields (drain, break, stun, regen), spot dodge, rolls, air dodge with invincibility
- Special mechanics, opted into per move: projectiles, counters, reflectors, super armor,
  plus an arbitrary `on_start` hook
- Match rules: stocks, blast zones, respawn invincibility, optional time limit
- `GameLoop.step()` advances exactly one frame. `run()` either paces to the wall clock
  or, when headless, runs as fast as Python can (about 50k frames/s)

## Using it

```python
from engine import (AttackData, AttackType, FighterStats, GameLoop, GameState,
                    Hitbox, InputState, Platform, ProjectileSpec, Stage, Vector2)

jab = AttackData(
    attack_type=AttackType.JAB, total_frames=18, startup_frames=4, recovery_frames=10,
    hitboxes=[Hitbox(offset=Vector2(20, -50), width=40, height=30, damage=4.0,
                     knockback_base=2.0, knockback_scaling=0.05, angle=45,
                     active_frames=(4, 7))])

fireball = AttackData(
    attack_type=AttackType.NEUTRAL_SPECIAL, total_frames=35, startup_frames=10,
    recovery_frames=15, hitboxes=[],
    projectile=ProjectileSpec(
        offset=Vector2(40, -50), velocity=Vector2(7, 0), lifetime=120,
        hitbox=Hitbox(offset=Vector2(0, 0), width=20, height=20, damage=6.0,
                      knockback_base=3.0, knockback_scaling=0.05, angle=30,
                      active_frames=(0, 999))))

hero = FighterStats(name="Hero", walk_speed=3.5, weight=95,
                    moveset={AttackType.JAB: jab, AttackType.NEUTRAL_SPECIAL: fireball})

ground = Platform(x=100, y=550, width=1000, height=50, passthrough=False)
stage = Stage(name="Arena", platforms=[], main_platform=ground,
              blast_zones={'left': -200, 'right': 1400, 'top': -200, 'bottom': 900},
              spawn_points=[Vector2(400, 510), Vector2(800, 510)])

state = GameState(stage, [hero, FighterStats(name="Dummy")])
loop = GameLoop(state, headless=True)
loop.on_hit = lambda attacker, defender, hitbox, x, y, kb: print("hit!", hitbox.damage)

hero_f, dummy_f = state.fighters
while state.frame < 600 and loop.step():
    close = dummy_f.position.x - hero_f.position.x < 60
    hero_f.input = InputState(right=not close, attack=close and state.frame % 20 == 0)
    dummy_f.input = InputState()
```

### Extension points

| Want to... | Use |
|---|---|
| Give a fighter moves | `FighterStats.moveset: Dict[AttackType, AttackData]` |
| Change what an empty move slot does | `GameLoop(state, fallback_attack=fn)` (default `default_attack`) |
| Spawn a projectile from a move | `AttackData.projectile = ProjectileSpec(...)` |
| Counter / reflector / super armor | `AttackData.is_counter` / `is_reflector` / `super_armor` |
| Run custom logic when a move starts | `AttackData.on_start = fn(loop, fighter)` |
| Change physics or rules | `GameState(..., config=GameConfig(gravity=0.4, stock_count=5))` |
| Hook effects, sound, UI | `GameLoop.on_hit`, `on_projectile_hit`, `on_reflect`, `on_land`, `on_dodge`, `on_frame_update`, `on_render`, `on_match_end` |
| Tag fighters for your own code | `FighterStats.character_type` (the engine never reads it) |

### Coordinates

Screen space: x grows to the right, y grows downward. `fighter.position` is the
fighter's center line. Hurtboxes and hitboxes are offsets from it. The feet sit
`GameConfig.foot_offset` below `position` (`fighter.feet_y`), and platform
collision uses the feet.

## Modules

| Module | Contents |
|---|---|
| `config.py` | `GameConfig`, default `CONFIG` |
| `geometry.py` | `Vector2`, `AABB` |
| `input.py` | `InputState` (held buttons, plus press detection against the previous frame) |
| `combat.py` | `AttackType`, `AttackData`, `Hitbox`, `Hurtbox`, `Projectile`, `ProjectileSpec`, `default_attack` |
| `fighter.py` | `Fighter`, `FighterStats`, `FighterState`, `DodgeType` |
| `stage.py` | `Stage`, `Platform`, `ParallaxLayer` |
| `state.py` | `GameState`: fighters, projectiles, timer, combos, win conditions |
| `loop.py` | `GameLoop`, `run_headless_match` |

## Known limitations

- Exactly 2 fighters are assumed in a few places (combo arrays, time-up winner).
- `GRAB`/`THROW_*` slots exist but there's no grab mechanic yet. Up and side
  specials apply a fixed vertical or horizontal boost for every fighter.
- Hurtboxes are a single AABB per fighter that doesn't change with state.

## Tests

`tests/test_engine.py` exercises the engine with no game content at all.
`tests/test_golden.py` replays 18 deterministic matches and checks a per-frame
hash of the full simulation state. If you change sim behaviour on purpose,
re-record with `python3 tests/record_golden.py`.
