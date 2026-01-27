# Cycle 19 Implementation Plan - Final Polish Pass

## Mission Overview
Implement final UI polish features for Stick Figure Fighter:
1. Enhanced Pause Menu with navigation
2. Damage Percentage Animation (scale, shake, color flash)
3. Stock Icon enhancement (X marks for lost stocks)

## Step-by-Step Implementation

### Step 1: Damage Animation State Infrastructure
**Files**: `game_engine.py`

Add damage animation tracking to Fighter class:
```python
# In Fighter.__init__:
self.damage_anim_frame: int = 0  # Frames since last hit (for animation)
self.damage_anim_delta: float = 0.0  # Damage amount that triggered animation
self.damage_anim_knockback: float = 0.0  # Knockback magnitude for shake intensity
```

Modify `apply_knockback()` to trigger animation:
```python
def apply_knockback(self, knockback: Vector2, damage: float):
    # ... existing code ...
    # Trigger damage animation
    self.damage_anim_frame = 0
    self.damage_anim_delta = damage
    self.damage_anim_knockback = knockback.magnitude()
```

### Step 2: Damage Animation Rendering
**Files**: `renderer.py`

Create new method `_draw_animated_damage()`:
- Calculate animation progress (0-30 frames)
- Apply scale factor: starts at 1.3x, eases back to 1.0x
- Apply shake offset based on knockback magnitude
- At 100%+ damage: base size increase
- Flash color on initial hit frames

Modify `draw_hud()`:
- Replace static damage text with `_draw_animated_damage()`
- Update Fighter's damage_anim_frame each render

Implementation details:
```python
def _draw_animated_damage(self, x: int, y: int, fighter: 'Fighter'):
    """Draw animated damage percentage with scale and shake."""
    damage = fighter.damage_percent
    anim_frame = getattr(fighter, 'damage_anim_frame', 30)
    anim_knockback = getattr(fighter, 'damage_anim_knockback', 0)

    # Base color from damage level
    color = self._get_damage_color(damage)

    # Animation parameters (0-30 frame animation)
    if anim_frame < 30:
        progress = anim_frame / 30.0

        # Scale: 1.3 -> 1.0 with ease-out
        scale = 1.0 + 0.3 * (1.0 - progress) ** 2

        # Shake: proportional to knockback, decays over time
        shake_intensity = min(8, anim_knockback * 0.5) * (1.0 - progress)
        shake_x = int(math.sin(anim_frame * 1.5) * shake_intensity)
        shake_y = int(math.cos(anim_frame * 2.0) * shake_intensity * 0.6)

        # Flash white on first 4 frames
        if anim_frame < 4:
            color = (255, 255, 255)
    else:
        scale = 1.0
        shake_x = 0
        shake_y = 0

    # High damage size increase
    if damage >= 100:
        scale *= 1.1 + min(0.3, (damage - 100) / 200)

    # Apply transforms and draw
    draw_x = x + shake_x
    draw_y = y + shake_y

    self.set_color_rgb(*color)
    self.draw_text(draw_x, draw_y, f"{damage:.0f}%")

    # Increment animation frame
    if anim_frame < 30:
        fighter.damage_anim_frame = anim_frame + 1
```

### Step 3: Enhanced Pause Menu State
**Files**: `main.py`

Add pause menu option enum and state:
```python
class PauseOption(Enum):
    RESUME = 0
    CHAR_SELECT = 1
    TITLE = 2

# In StickFighterGame.__init__:
self.pause_selection: int = 0
self._prev_up_pause: bool = False
self._prev_down_pause: bool = False
self._prev_confirm_pause: bool = False
```

### Step 4: Pause Menu Navigation
**Files**: `main.py`

Modify `_on_frame_update()` pause handling:
```python
# When paused, handle menu navigation
if state.paused:
    # Up/Down navigation
    up = self.input_manager.keyboard.is_key_pressed(0x0077)  # w
    down = self.input_manager.keyboard.is_key_pressed(0x0073)  # s

    if up and not self._prev_up_pause:
        self.pause_selection = (self.pause_selection - 1) % 3
        if self.audio:
            self.audio.play(SoundType.MENU_MOVE, 0.5)
    if down and not self._prev_down_pause:
        self.pause_selection = (self.pause_selection + 1) % 3
        if self.audio:
            self.audio.play(SoundType.MENU_MOVE, 0.5)

    self._prev_up_pause = up
    self._prev_down_pause = down

    # Confirm selection
    confirm = (self.input_manager.keyboard.is_key_pressed(0x006a) or  # j
               self.input_manager.keyboard.is_key_pressed(0x0020))    # space

    if confirm and not self._prev_confirm_pause:
        if self.pause_selection == 0:  # Resume
            state.paused = False
        elif self.pause_selection == 1:  # Char Select
            # ... navigate to char select
        elif self.pause_selection == 2:  # Title
            # ... navigate to title
    self._prev_confirm_pause = confirm
```

### Step 5: Enhanced Pause Menu Rendering
**Files**: `main.py` or `renderer.py`

Update `_draw_pause_overlay()`:
```python
def _draw_pause_overlay(self):
    """Draw enhanced pause menu overlay."""
    center_x = self.renderer.width // 2
    center_y = self.renderer.height // 2

    # Dark overlay
    self.renderer.set_color_rgb(0, 0, 0)
    for y in range(0, self.renderer.height, 3):
        self.renderer.draw_line(0, y, self.renderer.width, y)

    # Menu box
    box_w, box_h = 300, 250
    self.renderer.set_color_rgb(30, 30, 45)
    self.renderer.draw_rect(center_x - box_w//2, center_y - box_h//2, box_w, box_h, filled=True)

    # Border
    self.renderer.set_color_rgb(100, 100, 130)
    self.renderer.draw_rect(center_x - box_w//2, center_y - box_h//2, box_w, box_h, filled=False)

    # Title
    self.renderer.set_color('yellow')
    self.renderer.draw_text(center_x - 35, center_y - 90, "PAUSED")

    # Options
    options = ["Resume", "Character Select", "Title Screen"]
    option_y = center_y - 40

    for i, option in enumerate(options):
        if i == self.pause_selection:
            # Highlight
            self.renderer.set_color_rgb(80, 80, 120)
            self.renderer.draw_rect(center_x - 100, option_y - 5, 200, 30, filled=True)
            self.renderer.set_color_rgb(100, 255, 100)
        else:
            self.renderer.set_color('white')

        self.renderer.draw_text(center_x - 60, option_y + 5, option)
        option_y += 40

    # Match info
    self.renderer.set_color_rgb(120, 120, 130)
    match_time = self.game_state.frame / 60
    self.renderer.draw_text(center_x - 50, center_y + 80, f"Time: {match_time:.1f}s")

    # Controls hint
    self.renderer.set_color_rgb(80, 80, 90)
    self.renderer.draw_text(center_x - 80, center_y + 100, "W/S: Navigate | J: Select")
```

### Step 6: Stock Icon X Marks
**Files**: `renderer.py`

Modify `_draw_stock_icon()` lost stock rendering:
```python
if not filled:
    # Lost stock - draw with reduced opacity effect (darker)
    r, g, b = 50, 50, 50
    self.set_color_rgb(r, g, b)
    # Draw the icon shape...

    # Add X mark over lost stock
    self.set_color_rgb(150, 50, 50)  # Dark red X
    self.draw_line(x - 6, y - 6, x + 6, y + 6)
    self.draw_line(x - 6, y + 6, x + 6, y - 6)
```

## Success Criteria

1. **Damage Animation**:
   - Damage % scales up and back down when hit
   - Shake effect proportional to knockback
   - Color flash on hit
   - High damage (100%+) shows larger text

2. **Enhanced Pause Menu**:
   - W/S navigates between Resume/Char Select/Title
   - J or Space confirms selection
   - Visual highlight on selected option
   - P key still works for quick resume
   - Shows match time/info

3. **Stock Icons**:
   - Lost stocks have red X overlay
   - Existing character-specific icons preserved

## File Changes Summary

| File | Changes |
|------|---------|
| game_engine.py | Add damage_anim_frame, damage_anim_delta, damage_anim_knockback to Fighter |
| renderer.py | Add _draw_animated_damage(), update draw_hud(), enhance _draw_stock_icon() |
| main.py | Add PauseOption enum, pause_selection state, navigation handling, enhanced _draw_pause_overlay() |

## Testing Plan

1. Start a match, take damage, verify damage % animates (scale + shake)
2. Take heavy hit, verify increased shake intensity
3. Get to 100%+ damage, verify larger base size
4. Pause game, verify menu shows with 3 options
5. Navigate menu with W/S, verify highlight moves
6. Select each option, verify correct action
7. Lose a stock, verify X mark appears on lost stock icon
8. Test quick resume with P key still works

## Dependencies
- Existing damage tracking in Fighter class
- Existing draw_hud() and _draw_stock_icon() methods
- Existing pause handling in main.py
