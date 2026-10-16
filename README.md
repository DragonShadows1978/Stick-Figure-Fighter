# Stick Figure Fighter

A 2D Smash-style platform fighter in pure Python, rendered with raw X11
calls through ctypes. Originally built by the AtlasForge autonomous R&D system
(January 2026), with self-play reinforcement-learning experiments on top.

## Layout

```
engine/            Reusable platform-fighter simulation (stdlib only). See engine/README.md
characters.py      The four characters: stats, movesets, specials
stages.py          Battlefield and Final Destination
match_setup.py     create_game(): builds a match from character and stage names
ml_interface.py    Observation and action encoding the RL policies use
game_engine.py     Compatibility shim that re-exports the above under the old names

main.py            Game entry point: menus, match flow, results
renderer.py        X11 renderer: stick-figure poses, effects, HUD
input_system.py    Keyboard input       gamepad_input.py   evdev gamepads
audio_system.py    Synthesized audio    menu_system.py     Menus
training_mode.py   Training mode (CPU dummy, frame data, move list)
replay_system.py   Input recording and playback
settings.py        Persistent settings

policy.py, trainer.py, experience.py, self_play.py, train_self_play.py, evaluate.py
                   PPO self-play training (PyTorch)
tensor_gpu.py      Experimental CuPy autograd (only evaluate.py touches it, for set_device)

tests/             pytest: engine unit tests, golden-trace regression, character tests
archive/           Historical one-off scripts and research notes, not used
```

## Characters

| | Style | Signature |
|---|---|---|
| **RUSH** | Fast rushdown, light | Down-B counter (1.5x counterattack) |
| **ZONE** | Ranged zoner | Neutral-B energy ball, side-B beam |
| **GRAB** | Slow heavy grappler | Neutral-B armored charge (super armor) |
| **FLEX** | All-rounder | Neutral-B arcing fireball, down-B reflector |

## Running

Requirements: Python 3.10+, an X11 display, `numpy`. Optional: `sounddevice`
for audio. PyTorch is needed only for RL training.

```bash
python3 main.py                                 # menus
python3 main.py --quick --p1 RUSH --p2 ZONE     # straight into a match
python3 main.py --quick --stage final_destination
python3 main.py --training --p1 FLEX            # training mode
python3 main.py --test                          # subsystem smoke test
```

Controls: P1 uses WASD + Space/J/K/L, P2 uses arrows + Enter/Numpad 1-3.
P pauses, Tab shows hitboxes, Esc quits.

On a machine without a display, run it against a virtual X server:
`Xvfb :99 & python3 main.py --display :99 --quick --max-frames 600`.

## Tests

```bash
pip install pytest
python3 -m pytest tests
```

`tests/test_golden.py` replays 18 deterministic matches (every character pairing)
and checks a per-frame hash of the simulation state, so refactors can't
silently change gameplay. After an intentional gameplay change, re-record with
`python3 tests/record_golden.py` and commit the updated `tests/golden/traces.json`.

## Notes on the trained AI

The RL policies were trained against the simulation before the October 2026
fixes. In that version fighters couldn't walk off ledges, every successful
counter permanently stacked another 1.5x on RUSH's counter for the rest of the
process, and multi-frame hitboxes hit shields every active frame. Retrain
before trusting any old checkpoint. The game itself (`main.py`) doesn't load a
policy.

## Credits

- Built by the AtlasForge autonomous R&D system
- Platform contact offset idea from the Offset Geometric Contact paper
  (SIGGRAPH 2025, University of Utah Graphics Lab)
- Rabbit & Claude
