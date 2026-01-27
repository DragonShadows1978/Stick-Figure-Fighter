# Implementation Plan: Advanced Training Mode & Final Polish

**Cycle 16 - Mission Continuity Healing**
**Date:** December 2025

---

## Mission Objectives

Continue building Stick Figure Fighter with focus on:
1. **Input History Display** - Visual timeline of last 20 inputs in training mode
2. **Hitbox/Hurtbox Visualization Improvements** - Collision point markers, visibility modes
3. **Move List Display** - Show character moves with frame data in training mode

---

## Approach Rationale

Based on research from industry-standard training modes (SF6, Guilty Gear Strive, Smash Bros Ultimate):

1. **Input History** is essential for debugging execution problems - players need to see what inputs actually registered vs what they intended
2. **Hitbox visualization** with collision markers helps understand hit/hurt interactions
3. **Move list with frame data** provides reference without leaving the game

The implementation leverages existing infrastructure:
- `training_mode.py` already has `RecordedInput` dataclass for input tracking
- `renderer.py` has hitbox/hurtbox drawing code that can be extended
- `characters.py` has complete attack data for all 4 characters

**Sources:**
- [Dustloop Wiki - Using Training Mode](https://www.dustloop.com/w/Using_Training_Mode)
- [SF6Frames.com](https://sf6frames.com/) - Frame-by-frame hitbox viewer
- [Sirlin.net - Frame Advantage Display](https://www.sirlin.net/posts/a-fighting-game-first-showing-frame-advantage)

---

## Step-by-Step Implementation

### Step 1: Input History Ring Buffer

**File:** `training_mode.py`

Add an `InputHistoryDisplay` class that maintains a ring buffer of recent inputs:

```python
@dataclass
class InputHistoryEntry:
    """Single entry in input history."""
    directions: Tuple[bool, bool, bool, bool]  # left, right, up, down
    buttons: Tuple[bool, bool, bool, bool, bool]  # jump, attack, special, shield, grab
    held_frames: int = 1  # How many frames this input was held

class InputHistoryDisplay:
    """Tracks and displays recent inputs."""

    def __init__(self, max_entries: int = 20):
        self.entries: List[InputHistoryEntry] = []
        self.max_entries = max_entries
        self.enabled = False

    def update(self, inputs: dict):
        """Add or extend input entry based on current frame."""
        # If same as last entry, increment held_frames
        # Otherwise add new entry
        # Trim to max_entries

    def get_display_data(self) -> List[Tuple[str, str, int]]:
        """Return (direction_str, button_str, held_frames) for rendering."""
```

**Integration Points:**
- Add `input_history` attribute to `TrainingModeManager`
- Call `input_history.update()` each frame from main game loop
- Toggle with F7 key

**Files Modified:**
- `training_mode.py` - Add InputHistoryDisplay class

---

### Step 2: Input History Renderer

**File:** `renderer.py`

Add method to draw input history display:

```python
def draw_input_history(self, history_data: List[Tuple[str, str, int]], x: int, y: int):
    """Draw input history timeline.

    Args:
        history_data: List of (direction_str, button_str, held_frames) from oldest to newest
        x, y: Top-left position for display
    """
    # Draw background box
    # For each entry (most recent at top):
    #   Draw direction arrows (L D U R for left/down/up/right)
    #   Draw button letters (J A S H G)
    #   Draw held frame count if > 1
    # Color code: white=tap, yellow=short hold, orange=long hold
```

**Visual Layout:**
```
+================+
| INPUT HISTORY  |
+================+
| L D     A   3f |  <- most recent
| D       A S    |
| D R         G  |
| R       J      |
| L              |
| L D     A   2f |
+================+
```

**Direction Symbols:**
- Left: L
- Right: R
- Up: U
- Down: D

**Button Letters:**
- Jump: J (cyan)
- Attack: A (red)
- Special: S (yellow)
- Shield: H (blue)
- Grab: G (purple)

**Files Modified:**
- `renderer.py` - Add `draw_input_history()` method
- `renderer.py` - Modify `draw_training_mode_info()` to call it

---

### Step 3: Hitbox Visibility Modes

**File:** `renderer.py`

Replace single `show_hitboxes`/`show_hurtboxes` flags with visibility mode enum:

```python
class HitboxDisplayMode(Enum):
    NONE = 0      # No hitbox visualization
    HITBOX = 1    # Only attack hitboxes (red)
    HURTBOX = 2   # Only hurtboxes (green)
    BOTH = 3      # Both with collision markers
```

**Color Updates:**
- Hitbox: Red outline (already exists)
- Hurtbox: Green outline (already exists)
- Collision point: Yellow filled circle at intersection center

**Files Modified:**
- `renderer.py` - Add `HitboxDisplayMode` enum
- `renderer.py` - Replace `show_hitboxes`/`show_hurtboxes` with `hitbox_mode`
- `renderer.py` - Modify `draw_fighter()` for new modes
- `main.py` - Update Tab key handler to cycle modes

---

### Step 4: Collision Point Detection

**File:** `renderer.py`

Add collision detection for visualization:

```python
def _get_collision_point(self, hitbox_aabb: AABB, hurtbox_aabb: AABB) -> Optional[Tuple[int, int]]:
    """Calculate center of hitbox/hurtbox intersection."""
    # Check if AABBs overlap
    if not self._aabbs_intersect(hitbox_aabb, hurtbox_aabb):
        return None

    # Calculate intersection rectangle
    inter_x = max(hitbox_aabb.x, hurtbox_aabb.x)
    inter_y = max(hitbox_aabb.y, hurtbox_aabb.y)
    inter_x2 = min(hitbox_aabb.x + hitbox_aabb.width, hurtbox_aabb.x + hurtbox_aabb.width)
    inter_y2 = min(hitbox_aabb.y + hitbox_aabb.height, hurtbox_aabb.y + hurtbox_aabb.height)

    # Return center point
    return ((inter_x + inter_x2) // 2, (inter_y + inter_y2) // 2)
```

**Visualization:**
- Draw yellow filled circle (radius 5-8) at collision point
- Draw connecting line from hitbox center to collision point

**Files Modified:**
- `renderer.py` - Add collision point calculation
- `renderer.py` - Modify `draw_fighter()` to show collision markers

---

### Step 5: Move List Data Aggregation

**File:** `training_mode.py`

Add move list data structure:

```python
@dataclass
class MoveListEntry:
    """Data for a single move in the move list."""
    name: str
    input_notation: str  # e.g., "->+A", "v+B"
    startup_frames: int
    total_frames: int
    category: str  # "Normal", "Smash", "Aerial", "Special"

    @property
    def display_string(self) -> str:
        return f"{self.name} ({self.input_notation}) {self.startup_frames}f/{self.total_frames}f"

def get_move_list(character_type: str) -> List[MoveListEntry]:
    """Get formatted move list for a character."""
    from characters import get_character_attacks, CharacterType

    char = CharacterType[character_type]
    attacks = get_character_attacks(char)

    entries = []
    for attack_type, attack_data in attacks.items():
        entries.append(MoveListEntry(
            name=_format_attack_name(attack_type),
            input_notation=_get_input_notation(attack_type),
            startup_frames=attack_data.startup_frames,
            total_frames=attack_data.total_frames,
            category=_get_attack_category(attack_type)
        ))

    return sorted(entries, key=lambda e: (e.category, e.name))
```

**Input Notation Mapping:**
```python
_INPUT_NOTATION = {
    AttackType.JAB: "A",
    AttackType.FORWARD_TILT: "->+A",
    AttackType.UP_TILT: "^+A",
    AttackType.DOWN_TILT: "v+A",
    AttackType.FORWARD_SMASH: "->->+A",
    AttackType.UP_SMASH: "^^+A",
    AttackType.DOWN_SMASH: "vv+A",
    AttackType.NEUTRAL_AIR: "Air A",
    AttackType.FORWARD_AIR: "Air ->+A",
    AttackType.BACK_AIR: "Air <-+A",
    AttackType.UP_AIR: "Air ^+A",
    AttackType.DOWN_AIR: "Air v+A",
    AttackType.NEUTRAL_SPECIAL: "B",
    AttackType.SIDE_SPECIAL: "->+B",
    AttackType.UP_SPECIAL: "^+B",
    AttackType.DOWN_SPECIAL: "v+B",
}
```

**Files Modified:**
- `training_mode.py` - Add `MoveListEntry`, `get_move_list()`

---

### Step 6: Move List Renderer

**File:** `renderer.py`

Add method to draw move list panel:

```python
def draw_move_list(self, character_type: str, x: int, y: int, width: int, height: int):
    """Draw move list panel for a character.

    Uses scrolling if moves exceed visible area.
    """
    move_list = get_move_list(character_type)

    # Draw background panel
    self.set_color_rgb(30, 30, 50)
    self.draw_rect(x, y, width, height, filled=True)
    self.set_color('white')
    self.draw_rect(x, y, width, height, filled=False)

    # Draw title
    self.draw_text(x + 5, y + 15, f"{character_type} - Move List")

    # Draw category headers and moves
    current_y = y + 35
    current_category = None

    for entry in move_list:
        if entry.category != current_category:
            current_category = entry.category
            self.set_color('yellow')
            self.draw_text(x + 5, current_y, current_category.upper())
            current_y += 15

        self.set_color('white')
        self.draw_text(x + 10, current_y, entry.display_string)
        current_y += 13
```

**Position:** Right side of screen (width - 200, 80) with width 190, height 400

**Files Modified:**
- `renderer.py` - Add `draw_move_list()` method
- `renderer.py` - Modify `draw_training_mode_info()` to conditionally draw move list

---

### Step 7: Training Mode Key Bindings

**File:** `main.py`

Add new key handlers:

```python
# F7: Toggle input history display
f7_key = self.input_manager.keyboard.is_key_pressed(0xffc4)  # XK_F7
if f7_key and not self._prev_f7_key:
    enabled = self.training_manager.toggle_input_history()
self._prev_f7_key = f7_key

# F8: Cycle hitbox display mode
f8_key = self.input_manager.keyboard.is_key_pressed(0xffc5)  # XK_F8
if f8_key and not self._prev_f8_key:
    mode = self.training_manager.cycle_hitbox_mode()
    self.renderer.hitbox_mode = mode
self._prev_f8_key = f8_key

# F9: Toggle move list display
f9_key = self.input_manager.keyboard.is_key_pressed(0xffc6)  # XK_F9
if f9_key and not self._prev_f9_key:
    enabled = self.training_manager.toggle_move_list()
self._prev_f9_key = f9_key
```

**Files Modified:**
- `main.py` - Add F7, F8, F9 key handling
- `main.py` - Add `_prev_f7_key`, `_prev_f8_key`, `_prev_f9_key` flags
- `training_mode.py` - Add toggle methods

---

### Step 8: HUD Controls Update

**File:** `renderer.py`

Update training mode controls help text:

```python
# Line 2211-2212 update:
self.draw_text(10, self.height - 45, "F1:CPU F2:Behavior F3:FrameData F4:Difficulty F5:InfJumps F6:FrameAdv")
self.draw_text(10, self.height - 30, "F7:InputHist F8:Hitboxes F9:MoveList R:ResetDmg T:ResetPos")
```

**Files Modified:**
- `renderer.py` - Update HUD text in `draw_training_mode_info()`

---

## Files Summary

| File | Changes |
|------|---------|
| `training_mode.py` | Add InputHistoryDisplay, MoveListEntry, get_move_list(), hitbox mode cycling |
| `renderer.py` | Add draw_input_history(), draw_move_list(), HitboxDisplayMode enum, collision detection |
| `main.py` | Add F7/F8/F9 key handling, pass input data to training manager |

---

## Success Criteria

1. **Input History Display:**
   - Shows last 20 inputs in a clear visual format
   - Direction letters render correctly (L D U R)
   - Button presses shown with distinct colors
   - Held inputs show frame count
   - Toggle with F7 key

2. **Hitbox Visualization:**
   - Tab or F8 cycles through: None -> Hitbox -> Hurtbox -> Both -> None
   - Collision points marked with yellow indicators when both visible
   - Works with all character attacks

3. **Move List Display:**
   - Shows all 17 moves per character
   - Displays input notation and frame data
   - Organized by category (Normal, Smash, Aerial, Special)
   - Character-specific (shows data for P1's character)
   - Toggle with F9 key

4. **Integration:**
   - All features work together without visual overlap
   - Performance remains smooth (60 FPS target)
   - Controls documented in training mode HUD

---

## Testing Plan

### Functional Tests

1. **Input History:**
   - Perform various inputs and verify they appear correctly
   - Hold directions and verify frame count increments
   - Verify buffer wraps correctly after 20 entries

2. **Hitbox Visualization:**
   - Test each mode transition with F8
   - Verify collision markers appear when attacks connect
   - Check visibility with different character attacks

3. **Move List:**
   - Verify all 4 characters show correct move lists
   - Check frame data matches characters.py
   - Verify layout fits on screen without overlap

### Visual Tests

1. Run game in training mode
2. Capture screenshots verifying each feature
3. Confirm no visual artifacts or overlap issues

---

## Estimated Complexity

- **InputHistoryDisplay class:** ~80 lines
- **draw_input_history():** ~50 lines
- **HitboxDisplayMode + collision detection:** ~60 lines
- **MoveListEntry + get_move_list():** ~70 lines
- **draw_move_list():** ~40 lines
- **Main.py key handlers:** ~30 lines

**Total:** ~330 lines of new code
