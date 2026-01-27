# Implementation Plan: Cycle 10 - Controller Support & Polish

## Mission Overview

Cycles 1-9 delivered: core game loop with physics, 4 unique characters (RUSH, ZONE, GRAB, FLEX), projectile system, all defensive mechanics (shield, dodge, counter, reflector, super armor), comprehensive visual effects, complete audio system with dodge sound connected, character balance tuning, enhanced training mode with CPU difficulty levels and record/playback, character/stage select menus with animated previews and breathing animation, full game flow with results screen featuring victory/defeat poses and confetti, pause menu, match settings menu with stock/time configuration, title screen with mode selection, pre-match countdown, and verified dodge mechanics (spot dodge, roll, air dodge all functional).

**CURRENT PRIORITY:** Controller Support via evdev (main deliverable) + Visual Polish

---

## Success Criteria

1. **Gamepad Support (Main Deliverable):**
   - Detect USB gamepads at startup via evdev
   - Support Xbox-style controller button layout
   - Map D-pad/Left stick to movement, A=Jump, B=Attack, X=Special, LB=Shield, RB=Grab
   - Graceful fallback if no gamepad detected or evdev unavailable
   - Non-blocking polling integrated into game loop

2. **Settings Persistence:**
   - Save/load settings to `settings.json`
   - Persist audio volumes and default character selections
   - Settings menu accessible from title screen

3. **Dodge Afterimage Enhancement:**
   - Different afterimage colors by dodge type (blue/green/orange)
   - Invincibility glow effect (pulsing circles)

---

## Phase 1: Gamepad Input System

### Step 1.1: Create gamepad_input.py Module

**New File:** `gamepad_input.py`

This module handles all evdev-based gamepad reading with graceful fallbacks.

**Key Classes:**
```python
@dataclass
class GamepadBindings:
    """Maps gamepad buttons/axes to game actions."""
    left_axis: int = ABS_X       # Left stick X (negative = left)
    up_axis: int = ABS_Y         # Left stick Y (negative = up)
    dpad_x: int = ABS_HAT0X      # D-pad X axis
    dpad_y: int = ABS_HAT0Y      # D-pad Y axis
    jump: int = BTN_SOUTH        # A button (304)
    attack: int = BTN_EAST       # B button (305)
    special: int = BTN_WEST      # X button (307)
    shield: int = BTN_TL         # LB (310)
    grab: int = BTN_TR           # RB (311)
    pause: int = BTN_START       # Start (315)

class GamepadInput:
    """Handles gamepad input via evdev."""

    def __init__(self, device_path: str = None):
        """Initialize gamepad. Auto-detect if no path given."""

    def poll(self) -> None:
        """Non-blocking poll for new events."""

    def get_input_state(self) -> InputState:
        """Return current button/axis state as InputState."""

    def close(self) -> None:
        """Release device resources."""
```

**Detection Algorithm:**
1. List all devices in `/dev/input/`
2. Filter for devices with EV_KEY + EV_ABS capabilities
3. Check for gamepad-specific button codes (BTN_GAMEPAD, BTN_A, BTN_SOUTH)
4. Return first matching device or None

**Axis Handling:**
- Left stick: Map to directional input with deadzone (default 8000/32767)
- D-pad: Hat axes where -1/1 represent directions
- Axis values normalized to boolean left/right/up/down

### Step 1.2: Extend InputManager for Gamepad Support

**File:** `input_system.py`

Modify `InputManager` class to optionally use gamepad:

```python
class InputManager:
    def __init__(self, display_name: str = None, use_buffer: bool = True,
                 enable_gamepad: bool = True):
        self.keyboard = KeyboardInput(display_name)
        self.use_buffer = use_buffer

        # Gamepad support (optional)
        self.gamepads: List[GamepadInput] = []
        if enable_gamepad:
            self._detect_gamepads()

        # Player input sources: 'keyboard' or gamepad index
        self.player_sources = [0, 1]  # Default: both keyboard

    def _detect_gamepads(self):
        """Detect available gamepads."""
        try:
            from gamepad_input import detect_gamepads, GamepadInput
            devices = detect_gamepads()
            for dev in devices[:2]:  # Max 2 gamepads
                self.gamepads.append(GamepadInput(dev.path))
                print(f"[INPUT] Gamepad detected: {dev.name}")
        except ImportError:
            print("[INPUT] evdev not available, using keyboard only")
        except Exception as e:
            print(f"[INPUT] Gamepad detection failed: {e}")

    def poll(self):
        """Poll all input devices."""
        self.keyboard.poll()
        for gp in self.gamepads:
            gp.poll()

    def get_input_state(self, player_id: int) -> InputState:
        """Get combined input state for a player."""
        # Get keyboard state
        kb_state = self.keyboard.get_input_state(player_id)

        # Merge with gamepad if assigned
        if player_id < len(self.gamepads):
            gp_state = self.gamepads[player_id].get_input_state()
            return self._merge_inputs(kb_state, gp_state)

        return kb_state

    def _merge_inputs(self, kb: InputState, gp: InputState) -> InputState:
        """Merge keyboard and gamepad inputs (OR logic)."""
        merged = InputState()
        merged.left = kb.left or gp.left
        merged.right = kb.right or gp.right
        # ... etc for all inputs
        return merged
```

### Step 1.3: Install evdev Dependency

**Build Step:**
```bash
pip install evdev
```

Note: User must be in `input` group for device access.

---

## Phase 2: Settings Persistence

### Step 2.1: Create settings_manager.py Module

**New File:** `settings_manager.py`

```python
@dataclass
class GameSettings:
    """All persistent game settings."""
    master_volume: float = 0.7
    sfx_volume: float = 0.8
    music_volume: float = 0.5
    default_p1_character: Optional[str] = None
    default_p2_character: Optional[str] = None
    default_stage: str = "battlefield"
    default_stocks: int = 3
    default_time_limit: int = 180  # seconds, 0 = infinite

class SettingsManager:
    """Handles loading/saving game settings."""

    SETTINGS_FILE = "settings.json"

    def __init__(self, settings_path: str = None):
        self.path = settings_path or self.SETTINGS_FILE
        self.settings = self._load_or_create()

    def _load_or_create(self) -> GameSettings:
        """Load settings from file or create defaults."""
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
                return GameSettings(**data)
        except (FileNotFoundError, json.JSONDecodeError):
            return GameSettings()

    def save(self) -> None:
        """Save current settings to file."""
        with open(self.path, 'w') as f:
            json.dump(dataclasses.asdict(self.settings), f, indent=2)

    def get(self, key: str, default=None):
        """Get a setting value."""
        return getattr(self.settings, key, default)

    def set(self, key: str, value) -> None:
        """Set a setting value and auto-save."""
        setattr(self.settings, key, value)
        self.save()
```

### Step 2.2: Integrate Settings into Main Game

**File:** `main.py`

```python
class StickFighterGame:
    def __init__(self, ...):
        # Load settings at startup
        self.settings = SettingsManager()

        # Apply audio settings
        if self.audio:
            self.audio.set_volume(self.settings.get('master_volume', 0.7))

        # Use default characters if set
        default_p1 = self.settings.get('default_p1_character')
        default_p2 = self.settings.get('default_p2_character')
```

### Step 2.3: Add Settings Menu Option to Title Screen

**File:** `menu_system.py`

Add "Settings" option to TitleMenu with submenu for:
- Audio Volume (slider: 0-100%)
- SFX Volume (slider)
- Music Volume (slider)
- Back

---

## Phase 3: Dodge Afterimage Enhancement

### Step 3.1: Color by Dodge Type

**File:** `renderer.py`, method `draw_dodge_afterimages()`

**Current:** Uses fighter.color for all afterimages.

**Enhancement:** Select afterimage color based on dodge type:

```python
def draw_dodge_afterimages(self, fighter: Fighter):
    """Draw ghost afterimages during dodge with type-specific colors."""
    if fighter.state != FighterState.DODGING:
        self.fighter_afterimages[fighter.player_id] = []
        return

    # Select color based on dodge type
    if fighter.dodge_type == DodgeType.SPOT_DODGE:
        base_color = (150, 150, 255)  # Blue-ish
    elif fighter.dodge_type in (DodgeType.ROLL_FORWARD, DodgeType.ROLL_BACKWARD):
        base_color = (150, 255, 150)  # Green-ish
    elif fighter.dodge_type == DodgeType.AIR_DODGE:
        base_color = (255, 200, 150)  # Orange-ish
    else:
        base_color = fighter.color

    # ... existing afterimage creation code ...

    # Use base_color when drawing afterimages
    self.set_color_rgb(*base_color)
```

### Step 3.2: Invincibility Glow Effect

**File:** `renderer.py`, method `draw_dodge_afterimages()`

Add pulsing glow circles during invincibility:

```python
def draw_dodge_afterimages(self, fighter: Fighter):
    # ... existing code ...

    # Add invincibility glow if fighter has i-frames
    if fighter.invincible_frames > 0:
        glow_x = int(fighter.position.x) + self.shake_offset[0]
        glow_y = int(fighter.position.y - 40) + self.shake_offset[1]

        # Pulsing effect using sine wave
        pulse = abs(math.sin(self.frame_count * 0.3)) * 0.5 + 0.5
        glow_r = int(180 + 75 * pulse)
        glow_g = int(220 + 35 * pulse)
        glow_b = 255

        self.set_color_rgb(glow_r, glow_g, glow_b)

        # Draw concentric circles (outer to inner)
        for radius in [35, 25, 15]:
            self.draw_circle(glow_x, glow_y, radius, filled=False)
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `gamepad_input.py` | **NEW** - evdev gamepad handling |
| `settings_manager.py` | **NEW** - settings persistence |
| `input_system.py` | Add gamepad support to InputManager |
| `main.py` | Integrate settings, gamepad detection at startup |
| `menu_system.py` | Add Settings menu option |
| `renderer.py` | Enhance dodge afterimages |

---

## Testing Plan

### Functional Tests

1. **Gamepad Detection:**
   - Connect USB gamepad
   - Run game, verify "Gamepad detected" message
   - Verify controls work in menus and gameplay

2. **Graceful Fallback:**
   - Run without gamepad connected
   - Verify game works with keyboard only
   - No errors or crashes

3. **Settings Persistence:**
   - Change volume in settings menu
   - Exit and restart game
   - Verify volume setting persisted

4. **Dodge Afterimages:**
   - Perform each dodge type
   - Verify blue (spot), green (roll), orange (air) colors
   - Verify invincibility glow visible

### Edge Cases

- evdev module not installed → graceful fallback
- Gamepad disconnected mid-game → handle gracefully
- Corrupted settings.json → create fresh defaults
- Multiple gamepads → assign to P1 and P2

---

## Implementation Order

1. Create `gamepad_input.py` with detection and polling
2. Extend `InputManager` to use gamepads
3. Test gamepad functionality
4. Create `settings_manager.py`
5. Integrate settings into main.py
6. Add settings menu to title screen
7. Enhance dodge afterimages
8. Final integration testing

---

## Success Metrics

- [ ] Gamepad controls work for full match
- [ ] Settings persist across game restarts
- [ ] Dodge afterimages show correct colors
- [ ] No crashes with gamepad connected/disconnected
- [ ] Fallback to keyboard-only works correctly
