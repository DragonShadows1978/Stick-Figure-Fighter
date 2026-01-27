# Research Findings: Settings Menu UI Implementation

## Date: Cycle 11 Planning
## Mission: Settings Menu UI and Game Polish

---

## 1. Existing Codebase Analysis

### 1.1 Menu System Architecture (`menu_system.py`)

The game uses a state-machine approach for menus:

```
MenuState Enum:
- TITLE -> CHARACTER_SELECT -> STAGE_SELECT -> MATCH_SETTINGS -> READY
```

Key patterns discovered:
- **MenuSystem class**: Contains all menu logic, manages state transitions
- **MenuRenderer class**: Separate rendering class that wraps X11Renderer
- **Input handling**: Uses `_is_pressed()` for debounced edge detection
- **Sound callbacks**: `on_cursor_move`, `on_select`, `on_back` for audio feedback
- **State dataclasses**: `TitleScreenState`, `CharacterSelectState`, etc.

Current title screen options:
1. VERSUS MODE
2. TRAINING MODE
3. QUIT

**Gap**: No "SETTINGS" option exists in the title menu.

### 1.2 Settings System (`settings.py`)

Comprehensive settings infrastructure already exists:

```python
@dataclass AudioSettings:
  - master_volume: float (0-1)
  - sfx_volume: float (0-1)
  - music_volume: float (0-1)
  - muted: bool

@dataclass ControlSettings:
  - Keyboard keybindings
  - use_gamepad: bool
  - gamepad_mapping: str ('default' | 'smash')

@dataclass GameplaySettings:
  - default_stock_count: int
  - default_time_limit: int
  - default_stage: str
  - show_damage_numbers: bool
  - show_frame_data: bool

SettingsManager:
  - load() / save() to JSON file
  - apply_to_audio() / apply_to_input_manager()
  - get_settings_manager() / get_settings() globals
```

**Gap**: Settings are functional but have NO UI to modify them.

### 1.3 Input System (`input_system.py` + `gamepad_input.py`)

Gamepad support is fully implemented:
- `InputManager` handles both keyboard and gamepad input
- `GamepadManager` scans `/dev/input/` for controllers
- Two mapping presets: `DEFAULT_MAPPING`, `SMASH_MAPPING`
- `InputManager.get_gamepad_names()` returns connected gamepads
- `InputManager.set_player_gamepad()` toggles gamepad per player

**Gap**: No UI to display gamepad status or switch mappings.

### 1.4 Main Game Flow (`main.py`)

The `StickFighterGame` class orchestrates everything:
- `GameMode` enum: MENU, COUNTDOWN, MATCH, RESULTS, PAUSED
- `_handle_menu_input()` processes menu navigation
- Menu transitions happen via `self.mode = GameMode.XXX`
- Audio callbacks connected in `__init__`

**Integration point**: Need to add new GameMode for Settings or extend MenuState.

---

## 2. Best Practices Research

### 2.1 Game Settings Menu UI Patterns

Based on analysis of platform fighter games (Smash Bros, Rivals of Aether, Brawlhalla):

1. **Hierarchical Submenus**: Settings split into categories (Audio, Controls, Gameplay)
2. **Visual Sliders**: Volume controls use horizontal bars with percentage display
3. **Toggle Indicators**: On/Off states shown with filled/unfilled markers
4. **Instant Feedback**: Audio sliders play test sound when adjusted
5. **Save on Exit**: Settings persist when leaving menu, not on every change
6. **Back Navigation**: Consistent B/Special button to go back

### 2.2 Fighting Game Control Display Conventions

- Show current binding as key name or icon
- Display "NOT BOUND" for unset keys
- Gamepad detection indicator (connected/disconnected)
- Mapping preset selector with preview

### 2.3 Accessibility Considerations

- Clear visual contrast for selected items
- Cursor indicators that animate
- Sound feedback on navigation
- No time-sensitive menu interactions

---

## 3. Implementation Decisions

### 3.1 Architecture Choice: Extend MenuState

**Decision**: Add new MenuState values rather than separate GameMode.

Rationale:
- Keeps settings as part of menu system (consistent with existing patterns)
- Reuses MenuRenderer infrastructure
- Allows smooth transitions (Settings -> back to Title)

New states:
- `SETTINGS_MAIN` - Audio/Controls/Gameplay/Back
- `SETTINGS_AUDIO` - Volume sliders
- `SETTINGS_CONTROLS` - Keybinding display, gamepad toggle
- `SETTINGS_GAMEPLAY` - Stock count, time, damage numbers

### 3.2 Settings Menu Flow

```
TITLE (add "SETTINGS" option)
  └── SETTINGS_MAIN
        ├── Audio Settings
        │     ├── Master Volume (slider 0-100%)
        │     ├── SFX Volume (slider 0-100%)
        │     ├── Music Volume (slider 0-100%)
        │     └── Mute Toggle (On/Off)
        ├── Controls
        │     ├── Show P1 Bindings (display only)
        │     ├── Show P2 Bindings (display only)
        │     ├── Gamepad Status (auto-detect display)
        │     └── Mapping Preset (default/smash cycle)
        ├── Gameplay
        │     ├── Default Stocks (1-5)
        │     ├── Default Time (None/3/5/8 min)
        │     └── Show Damage Numbers (On/Off)
        └── Back (save & return to TITLE)
```

### 3.3 UI Element Patterns

Following existing patterns from `_render_setting_row()`:
- Row highlight with pulsing animation
- Left/Right arrows for adjustable values
- Consistent spacing and alignment

---

## 4. File Modification Plan

### Files to Modify:
1. **menu_system.py**: Add SETTINGS states, SettingsMenu logic, SettingsRenderer
2. **main.py**: Connect settings menu audio callbacks

### Files to Use (no changes):
- **settings.py**: SettingsManager already has all needed methods
- **input_system.py**: InputManager provides gamepad queries
- **gamepad_input.py**: GamepadManager provides names/status

---

## 5. Success Criteria

1. "SETTINGS" option appears on title screen between TRAINING and QUIT
2. Three submenus (Audio, Controls, Gameplay) are navigable
3. Volume sliders adjust in 10% increments with visual feedback
4. Mute toggle works with immediate audio feedback
5. Gamepad detection shows current status
6. Mapping preset can be cycled (default/smash)
7. Gameplay defaults persist between sessions
8. Settings save to JSON on menu exit
9. Back navigation returns to title screen

---

## 6. Stretch Goals (If Time Permits)

1. **Replay System**: Separate implementation, not blocking settings
2. **Additional Stage Variant**: Already have battlefield and final_destination
3. **Frame Advantage Display**: Training mode enhancement

---

## Sources Consulted

- Existing codebase analysis (primary source)
- Fighting game UI conventions (general knowledge)
- Linux evdev gamepad handling (in gamepad_input.py)
