# Research Findings: Cycle 10 - Controller Support & Polish

## Research Date: 2025-12-26

---

## 1. Linux Gamepad Input via evdev

### Background
The Linux evdev interface provides a unified way to read input events from any input device through `/dev/input/eventX` files. For gamepads, this is the modern approach (the legacy `/dev/input/jsX` joystick interface is deprecated).

### Library Choice: python-evdev

**Recommended library:** `python-evdev` (v1.9.0 released Feb 2025)

Key advantages:
- Direct access to raw input events
- Non-blocking async I/O support
- No external dependencies
- Works with all USB gamepads (Xbox, PlayStation, generic)

**Sources:**
- [Python-evdev Tutorial](https://python-evdev.readthedocs.io/en/latest/tutorial.html)
- [Arch Wiki Gamepad](https://wiki.archlinux.org/title/Gamepad)

### Device Detection Pattern

```python
from evdev import InputDevice, list_devices, ecodes

def detect_gamepads():
    """Find gamepad devices by checking for typical capabilities."""
    gamepads = []
    for path in list_devices():
        dev = InputDevice(path)
        caps = dev.capabilities()

        # Gamepads typically have:
        # - EV_KEY (buttons): BTN_GAMEPAD, BTN_A, etc.
        # - EV_ABS (axes): ABS_X, ABS_Y, etc.
        has_buttons = ecodes.EV_KEY in caps
        has_axes = ecodes.EV_ABS in caps

        if has_buttons and has_axes:
            # Check for specific gamepad buttons
            keys = caps.get(ecodes.EV_KEY, [])
            if any(k in keys for k in [ecodes.BTN_GAMEPAD, ecodes.BTN_A,
                                        ecodes.BTN_SOUTH, ecodes.BTN_START]):
                gamepads.append(dev)
    return gamepads
```

### Xbox-Style Controller Mapping (Standard)

| Button/Axis | evdev Code | Game Action |
|-------------|-----------|-------------|
| D-Pad Left  | ABS_HAT0X=-1 | LEFT |
| D-Pad Right | ABS_HAT0X=1 | RIGHT |
| D-Pad Up    | ABS_HAT0Y=-1 | UP |
| D-Pad Down  | ABS_HAT0Y=1 | DOWN |
| A Button    | BTN_SOUTH (304) | JUMP |
| B Button    | BTN_EAST (305) | ATTACK |
| X Button    | BTN_WEST (307) | SPECIAL |
| Y Button    | BTN_NORTH (308) | - |
| LB          | BTN_TL (310) | SHIELD |
| RB          | BTN_TR (311) | GRAB |
| Left Stick X | ABS_X | LEFT/RIGHT |
| Left Stick Y | ABS_Y | UP/DOWN |
| Start       | BTN_START (315) | - |
| Select      | BTN_SELECT (314) | - |

### Non-Blocking Read Pattern (Critical for Game Loop)

```python
import selectors
from evdev import InputDevice, ecodes

selector = selectors.DefaultSelector()
gamepad = InputDevice('/dev/input/eventX')
selector.register(gamepad, selectors.EVENT_READ)

# In game loop (non-blocking)
def poll_gamepad():
    events = []
    ready = selector.select(timeout=0)  # Non-blocking
    for key, mask in ready:
        for event in key.fileobj.read():
            events.append(event)
    return events
```

### Permissions Note
User must be in `input` group to access `/dev/input/event*` devices:
```bash
sudo usermod -a -G input $USER
```

---

## 2. Settings Persistence

### JSON-Based Settings File

Location: `settings.json` in game root directory

**Settings Schema:**
```json
{
    "audio": {
        "master_volume": 0.7,
        "music_volume": 0.5,
        "sfx_volume": 0.8
    },
    "controls": {
        "player1": {
            "type": "keyboard",
            "bindings": {
                "left": "a",
                "right": "d",
                "up": "w",
                "down": "s",
                "jump": "space",
                "attack": "j",
                "special": "k",
                "shield": "l",
                "grab": "u"
            }
        },
        "player2": {
            "type": "keyboard",
            "bindings": {
                "left": "left",
                "right": "right",
                "up": "up",
                "down": "down",
                "jump": "return",
                "attack": "kp_1",
                "special": "kp_2",
                "shield": "kp_3",
                "grab": "kp_0"
            }
        }
    },
    "game": {
        "default_stocks": 3,
        "default_time_limit": 180,
        "default_p1_character": null,
        "default_p2_character": null,
        "default_stage": "battlefield"
    }
}
```

### Implementation Approach

1. Load settings at game startup (create defaults if file missing)
2. Pass settings to relevant subsystems
3. Save settings when changed via settings menu
4. Use Python's built-in `json` module (no external dependencies)

---

## 3. Dodge Afterimage Enhancement

### Current Implementation (renderer.py:1087)

The current `draw_dodge_afterimages()` method:
- Creates afterimages every 3 frames during DODGING state
- Uses a faded version of fighter color
- Draws up to 3 afterimages with increasing opacity

### Required Enhancements

1. **Color by Dodge Type:**
   - SPOT_DODGE: Blue-ish (150, 150, 255)
   - ROLL_FORWARD/BACKWARD: Green-ish (150, 255, 150)
   - AIR_DODGE: Orange-ish (255, 200, 150)

2. **Invincibility Glow:**
   - Pulsing white/blue circles during invincible frames
   - Draw behind fighter for "ethereal" effect

### Code Location
- `renderer.py:1087` - `draw_dodge_afterimages()` method
- `game_engine.py:133-138` - `DodgeType` enum

---

## 4. Integration Points Identified

### input_system.py

**InputManager class** (line 455):
- Currently handles only keyboard input
- `poll()` method needs to also poll gamepad
- `update_fighter()` needs gamepad alternative
- `get_input_state()` should merge keyboard + gamepad

**KeyBindings class** (line 159):
- Works with keysym integers
- Need parallel GamepadBindings class for button codes

### main.py

**StickFighterGame class**:
- Instantiates InputManager at line 159
- Calls `input_manager.poll()` in various modes
- Key checks use raw keysym values (hardcoded)
- Need to abstract into input actions for gamepad support

---

## 5. Summary of Required Work

### High Priority (Main Deliverable)
1. Create `gamepad_input.py` - evdev-based gamepad reader
2. Extend `InputManager` to support gamepads
3. Add gamepad detection at startup
4. Create default Xbox-style button mapping

### Medium Priority
5. Create `settings_manager.py` for persistence
6. Add settings to title menu
7. Implement volume controls in audio system

### Low Priority (Polish)
8. Enhance dodge afterimages with colors
9. Add invincibility glow effect

---

## Sources Consulted

1. [Python-evdev Tutorial](https://python-evdev.readthedocs.io/en/latest/tutorial.html)
2. [Python Read Gamepad Input - CodeRivers](https://coderivers.org/blog/python-read-gamepad-input/)
3. [Arch Wiki Gamepad](https://wiki.archlinux.org/title/Gamepad)
4. [Gamepad Implementation on Linux](https://ruby0x1.github.io/machinery_blog_archive/post/gamepad-implementation-on-linux/index.html)
5. [Talking to gamepads without PyGame](https://blog.thea.codes/talking-to-gamepads-without-pygame/)
