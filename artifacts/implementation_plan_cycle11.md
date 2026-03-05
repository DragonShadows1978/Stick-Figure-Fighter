# Implementation Plan: Settings Menu UI (Cycle 11)

## Overview

Add a user-accessible Settings menu to Stick Figure Fighter that allows players to adjust audio, controls, and gameplay options. This makes the existing settings.py infrastructure user-visible.

---

## Key Requirements

1. Add "SETTINGS" option to title screen (between TRAINING MODE and QUIT)
2. Create three submenus: Audio, Controls, Gameplay
3. Connect to existing SettingsManager for persistence
4. Maintain consistent UI patterns with existing menus

---

## Implementation Steps

### Step 1: Extend MenuState Enum

**File:** `menu_system.py` (lines ~19-25)

Add new states:
```python
class MenuState(Enum):
    TITLE = auto()
    CHARACTER_SELECT = auto()
    STAGE_SELECT = auto()
    MATCH_SETTINGS = auto()
    READY = auto()
    # NEW:
    SETTINGS_MAIN = auto()      # Audio/Controls/Gameplay submenu selection
    SETTINGS_AUDIO = auto()     # Volume sliders, mute toggle
    SETTINGS_CONTROLS = auto()  # Keybinding display, gamepad options
    SETTINGS_GAMEPLAY = auto()  # Default stocks/time, damage numbers toggle
```

### Step 2: Add Settings State Dataclasses

**File:** `menu_system.py` (after line ~80)

```python
@dataclass
class SettingsMainState:
    """State for main settings menu."""
    cursor: int = 0  # 0=Audio, 1=Controls, 2=Gameplay, 3=Back
    cursor_anim: int = 0

@dataclass
class SettingsAudioState:
    """State for audio settings submenu."""
    cursor: int = 0  # 0=Master, 1=SFX, 2=Music, 3=Mute, 4=Back
    cursor_anim: int = 0

@dataclass
class SettingsControlsState:
    """State for controls settings submenu."""
    cursor: int = 0  # 0=P1 bindings (display), 1=P2 bindings, 2=Gamepad status, 3=Mapping preset, 4=Back
    cursor_anim: int = 0

@dataclass
class SettingsGameplayState:
    """State for gameplay settings submenu."""
    cursor: int = 0  # 0=Default stocks, 1=Default time, 2=Damage numbers, 3=Back
    cursor_anim: int = 0
```

### Step 3: Update TitleScreenState and Title Menu

**File:** `menu_system.py`

Update title screen to have 4 options:
- Line ~103: `MODE_OPTIONS = ["VERSUS MODE", "TRAINING MODE", "SETTINGS", "QUIT"]`

Update `_update_title()` method to:
1. Change cursor wrap from 3 to 4 options
2. Add case for cursor == 2 (SETTINGS) -> transition to SETTINGS_MAIN

### Step 4: Add Settings State Initialization

**File:** `menu_system.py` (in `MenuSystem.__init__`)

Add:
```python
self.settings_main_state = SettingsMainState()
self.settings_audio_state = SettingsAudioState()
self.settings_controls_state = SettingsControlsState()
self.settings_gameplay_state = SettingsGameplayState()

# Import settings manager
from settings import get_settings_manager
self.settings_manager = get_settings_manager()
```

### Step 5: Implement Settings Update Methods

**File:** `menu_system.py`

Add methods (similar to existing `_update_match_settings`):

#### `_update_settings_main(self, p1_inputs, p2_inputs) -> bool`
- Navigate up/down (4 options: Audio, Controls, Gameplay, Back)
- Attack/Jump: Enter selected submenu or go back
- Special: Go back to TITLE, save settings

#### `_update_settings_audio(self, p1_inputs, p2_inputs) -> bool`
- Navigate up/down (5 options: Master, SFX, Music, Mute, Back)
- Left/Right on volume rows: Adjust by 0.1 (10%)
- Attack on Mute: Toggle
- Special/Back option: Return to SETTINGS_MAIN

#### `_update_settings_controls(self, p1_inputs, p2_inputs) -> bool`
- Navigate up/down (5 options: P1 info, P2 info, Gamepad status, Mapping, Back)
- Left/Right on Mapping: Cycle between 'default' and 'smash'
- Special/Back: Return to SETTINGS_MAIN
- Display-only for keybindings (no rebinding in this cycle)

#### `_update_settings_gameplay(self, p1_inputs, p2_inputs) -> bool`
- Navigate up/down (4 options: Stocks, Time, Damage Numbers, Back)
- Left/Right: Adjust stocks (1-5), time (None/3/5/8)
- Attack on Damage Numbers: Toggle
- Special/Back: Return to SETTINGS_MAIN

### Step 6: Add Settings Rendering Methods

**File:** `menu_system.py` (in `MenuRenderer` class)

#### `_render_settings_main(self, frame: int)`
- Background with title "SETTINGS"
- Four options: AUDIO, CONTROLS, GAMEPLAY, BACK
- Highlight current selection with pulsing animation
- Cursor arrows on selected row

#### `_render_settings_audio(self, frame: int)`
- Title: "AUDIO SETTINGS"
- Volume sliders (horizontal bars showing 0-100%)
- Current values displayed as percentage
- Mute toggle showing ON/OFF
- Back option

#### `_render_settings_controls(self, frame: int)`
- Title: "CONTROL SETTINGS"
- P1 bindings table (read-only display)
- P2 bindings table (read-only display)
- Gamepad status: "Connected: [name]" or "No gamepad detected"
- Mapping preset: "default" / "smash" with description
- Back option

#### `_render_settings_gameplay(self, frame: int)`
- Title: "GAMEPLAY SETTINGS"
- Default stocks: 1-5 with arrows
- Default time: None/3 min/5 min/8 min
- Show damage numbers: ON/OFF toggle
- Back option

### Step 7: Wire Settings Rendering in MenuRenderer.render()

**File:** `menu_system.py` (in `MenuRenderer.render()`)

Add cases:
```python
elif self.menu.state == MenuState.SETTINGS_MAIN:
    self._render_settings_main(frame)
elif self.menu.state == MenuState.SETTINGS_AUDIO:
    self._render_settings_audio(frame)
elif self.menu.state == MenuState.SETTINGS_CONTROLS:
    self._render_settings_controls(frame)
elif self.menu.state == MenuState.SETTINGS_GAMEPLAY:
    self._render_settings_gameplay(frame)
```

### Step 8: Wire Settings Updates in MenuSystem.update()

**File:** `menu_system.py` (in `MenuSystem.update()`)

Add cases after MATCH_SETTINGS:
```python
elif self.state == MenuState.SETTINGS_MAIN:
    return self._update_settings_main(p1_inputs, p2_inputs)
elif self.state == MenuState.SETTINGS_AUDIO:
    return self._update_settings_audio(p1_inputs, p2_inputs)
elif self.state == MenuState.SETTINGS_CONTROLS:
    return self._update_settings_controls(p1_inputs, p2_inputs)
elif self.state == MenuState.SETTINGS_GAMEPLAY:
    return self._update_settings_gameplay(p1_inputs, p2_inputs)
```

### Step 9: Update MenuSystem.reset()

**File:** `menu_system.py`

Add reset for settings states:
```python
self.settings_main_state = SettingsMainState()
self.settings_audio_state = SettingsAudioState()
self.settings_controls_state = SettingsControlsState()
self.settings_gameplay_state = SettingsGameplayState()
```

### Step 10: Apply Settings on Exit

**File:** `menu_system.py`

When leaving SETTINGS_MAIN back to TITLE:
```python
self.settings_manager.save()  # Persist to JSON
```

---

## Files Modified

| File | Changes |
|------|---------|
| `menu_system.py` | Add 4 new MenuState values, 4 state dataclasses, 4 update methods, 4 render methods, update title menu |

---

## Success Criteria Checklist

- [ ] "SETTINGS" appears in title menu
- [ ] Can navigate to Audio submenu
- [ ] Can navigate to Controls submenu
- [ ] Can navigate to Gameplay submenu
- [ ] Master volume slider adjusts (0-100%)
- [ ] SFX volume slider adjusts (0-100%)
- [ ] Mute toggle works
- [ ] Gamepad status displays correctly
- [ ] Mapping preset can be cycled
- [ ] Default stocks can be changed (1-5)
- [ ] Default time can be changed
- [ ] Damage numbers toggle works
- [ ] Back navigation works at all levels
- [ ] Settings persist after restart (JSON save)

---

## Testing Plan

1. Launch game with `python main.py`
2. Navigate to SETTINGS from title
3. Test each submenu navigation
4. Adjust audio settings, verify sound changes
5. Check gamepad detection
6. Modify gameplay defaults
7. Exit settings, start match, verify defaults applied
8. Restart game, verify settings persisted

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Large file modification | Single file (menu_system.py) keeps changes contained |
| Breaking existing menus | New states are additive, no changes to existing flow |
| Settings not saving | Use existing SettingsManager.save() which is tested |
| Audio not updating | Call audio.set_master_volume() etc. on value change |

---

## Stretch Goals (If Time Permits)

1. **Replay System** - Record InputState history, save to JSON, playback
2. **Additional stage variant** - Already have two stages
3. **Frame advantage display** - Training mode enhancement
