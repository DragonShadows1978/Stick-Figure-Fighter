# Research Findings - Cycle 13: Polish and Advanced Features

## Research Date: 2025-12-27

## Mission: Frame Advantage Display, Hit Spark Colors, Gamepad Rumble

---

## 1. Frame Advantage Display in Training Mode

### Industry Best Practices (2025)

Sources:
- [Guilty Gear Strive Frame Data Display Update](https://noisypixel.net/guilty-gear-strive-frame-data-ranked-match-update/)
- [2XKO Frame Data Guide](https://2xkocombos.com/mechanics/frame-data/)
- [Street Fighter 6 Frame Data Guide](https://hyperxarenalasvegas.com/mastering-frame-data-in-street-fighter-6-2025-guide/)
- [Sirlin.Net: A Fighting Game First - Showing Frame Advantage](https://www.sirlin.net/posts/a-fighting-game-first-showing-frame-advantage)

### Key Findings:

1. **Guilty Gear Strive (2025 Update)**: Added frame data display to Training Mode showing:
   - Total Frames
   - Active Frames
   - Frame Advantage
   - Follow-up Window

2. **2XKO**: Real-time frame data visualization in training mode showing startup, active, recovery in real-time with frame advantage after each hit/block.

3. **Frame Advantage Definition**:
   - Calculated as: `attacker_recovery - defender_recovery`
   - Positive = attacker can move first (advantage)
   - Negative = defender can move first (disadvantage)
   - Displayed as `+X` or `-X` with color coding:
     - **Green/Blue**: Positive (safe, advantageous)
     - **Red/Orange**: Negative (unsafe, punishable)
     - **Yellow/White**: Neutral (0)

4. **Display Location Options**:
   - Near hit location (momentary popup that fades)
   - In fixed training mode overlay/HUD
   - Both (Guilty Gear style)

### Recommended Implementation:

```python
# Frame advantage calculation
# On hit: attacker_recovery_remaining - defender_hitstun_remaining
# On block: attacker_recovery_remaining - defender_blockstun_remaining

# Display:
# - Show at hit location as floating number
# - Color: green if > 0, red if < 0, white if 0
# - Duration: ~60 frames (1 second) then fade
# - Format: "+5" or "-3"
```

---

## 2. Hit Spark Color Variations Based on Damage

### Industry Examples

Sources:
- [TV Tropes: Hit Spark](https://tvtropes.org/pmwiki/pmwiki.php/Main/HitSpark)
- [Super Smash Bros. Ultimate Hit Effects VFX](https://www.artstation.com/artwork/68JPzr)
- [SmashWiki: Effect](https://www.ssbwiki.com/Effect)
- [Fighting Game Effects Pack](https://www.gamedevmarket.net/asset/fighting-game-effects-pack)

### Key Findings:

1. **Super Smash Bros. Ultimate**:
   - Normal hits: White/yellow sparks
   - Strong hits: More intense, larger effects
   - Kill/KO moves: Red spark effect
   - Element-based coloring (fire = orange/red, electric = blue/white)

2. **General Fighting Game Conventions**:
   - Light hits: Small, white/soft sparks
   - Medium hits: Yellow/brighter sparks
   - Heavy hits: Orange/more chaotic
   - Critical/super moves: Red/intense, larger effects
   - Blocked hits: Blue/ripple effect (different from connecting)

3. **Color Psychology in Combat VFX**:
   - White: Clean, light damage
   - Yellow: Medium impact
   - Orange: Heavy damage
   - Red: Severe/critical damage

### Recommended Implementation:

```python
# Damage thresholds for color variation
# 0-10 damage: White sparks (255, 255, 255)
# 11-20 damage: Yellow sparks (255, 255, 100)
# 21-30 damage: Orange sparks (255, 180, 80)
# 31+ damage: Red sparks (255, 100, 100)

# Pass damage to add_hit_effect() and use it to determine color
```

---

## 3. Gamepad Rumble/Force Feedback with evdev

### Technical Implementation

Sources:
- [Python-evdev Tutorial](https://python-evdev.readthedocs.io/en/latest/tutorial.html)
- [xpadneo Rumble Example](https://github.com/atar-axis/xpadneo/blob/master/misc/examples/python_asyncio_evdev/gamepad.py)
- [python-evdev Rumble PR #23](https://github.com/gvalkov/python-evdev/pull/23)
- [ArchWiki Gamepad](https://wiki.archlinux.org/title/Gamepad)

### Key Findings:

1. **evdev Force Feedback API**:
   ```python
   from evdev import ecodes, InputDevice, ff, list_devices

   # Find device with EV_FF capability
   for name in list_devices():
       dev = InputDevice(name)
       if ecodes.EV_FF in dev.capabilities():
           break

   # Create rumble effect
   rumble = ff.Rumble(strong_magnitude=0xffff, weak_magnitude=0x0000)
   effect_type = ff.EffectType(ff_rumble_effect=rumble)
   duration_ms = 100
   effect = ff.Effect(
       ecodes.FF_RUMBLE, -1, 0,
       ff.Trigger(0, 0),
       ff.Replay(duration_ms, 0),
       effect_type
   )

   # Upload and play
   effect_id = dev.upload_effect(effect)
   dev.write(ecodes.EV_FF, effect_id, 1)

   # Later: erase effect
   dev.erase_effect(effect_id)
   ```

2. **Rumble Types**:
   - `strong_magnitude`: Heavy motor (0x0000 - 0xFFFF)
   - `weak_magnitude`: Light motor (0x0000 - 0xFFFF)
   - Duration in milliseconds

3. **Use Cases in Fighting Games**:
   - Light hit received: Short, weak rumble (50ms, weak motor)
   - Heavy hit received: Longer, strong rumble (150ms, strong motor)
   - Attack connecting: Very short feedback (30ms, weak)
   - KO/big moments: Extended rumble (300ms, both motors)

4. **Requirements**:
   - Device must have write permissions (`/dev/input/eventX`)
   - Check for `EV_FF` capability before attempting rumble
   - Not all controllers support force feedback
   - Testing with `fftest` (linuxconsole package)

### Recommended Implementation:

```python
# Add to GamepadManager:
def supports_rumble(self, gamepad: Gamepad) -> bool:
    """Check if gamepad supports force feedback."""
    try:
        caps = InputDevice(gamepad.path).capabilities()
        return ecodes.EV_FF in caps
    except:
        return False

def rumble(self, gamepad: Gamepad, strong: float, weak: float, duration_ms: int):
    """
    Trigger rumble effect.

    Args:
        strong: Strong motor intensity (0.0 - 1.0)
        weak: Weak motor intensity (0.0 - 1.0)
        duration_ms: Duration in milliseconds
    """
    # Implementation using evdev ff module
```

---

## 4. Additional Polish Features

### Screen Shake Enhancements
- Already implemented in renderer.py
- Can be enhanced to scale with damage

### Crowd/Audience Sounds
- AudioManager already has infrastructure
- Add crowd "ooh" sounds for big hits (knockback > 15)
- Add crowd cheers for KOs

### Stage Parallax (Lower Priority)
- Multiple background layers moving at different speeds
- Requires renderer enhancement

### Character Entrance Animations (Lower Priority)
- Drop-in from top at match start
- Would need new state: ENTRANCE
- 30-60 frame animation before fight starts

---

## Implementation Priority

1. **Frame Advantage Display** - HIGHEST
   - Essential for training mode
   - Helps players learn combo timing
   - Requires: TrainingModeManager modification, renderer overlay

2. **Hit Spark Color Variations** - HIGH
   - Enhances game feel
   - Simple to implement (modify add_hit_effect)
   - Pass damage to determine color

3. **Gamepad Rumble** - MEDIUM
   - Nice-to-have feature
   - Dependent on controller support
   - May not work on all devices

4. **Additional Polish** - LOW
   - Crowd sounds, parallax, entrances
   - Future cycle work

---

## Existing Code Integration Points

### training_mode.py
- `TrainingModeManager`: Add frame advantage tracking
- `FrameDataOverlay`: Already has `show_advantage: bool = True`
- Need: `last_frame_advantage: int`, display method

### renderer.py
- `add_hit_effect()`: Currently takes `knockback_magnitude`, add `damage` parameter
- `draw_hit_spark()`: Modify to use damage-based colors instead of time-based
- Need: Pass damage through effect chain

### gamepad_input.py
- `Gamepad`: Currently opens O_RDONLY, need O_RDWR for rumble
- `GamepadManager`: Add rumble support methods
- Need: evdev ff module imports, effect management

### game_engine.py
- `on_hit` callback: Already passes `hitbox` which contains `damage`
- Can calculate frame advantage from `attacker.attack_frame` and `defender.hitstun_frames`

### main.py
- `_on_hit()`: Modify to pass damage to renderer
- Add gamepad rumble triggers on hit events
