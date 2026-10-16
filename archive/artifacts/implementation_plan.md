# Implementation Plan: Stick Figure Fighter

---

# Cycle 20: Final Validation & Celebration

## Mission Overview

**This is the FINAL CYCLE (20 of 20).**

Cycles 1-19 delivered a complete platform fighter built entirely from first principles:
- Core game loop with physics, knockback, and hitstun
- 4 unique characters (RUSH, ZONE, GRAB, FLEX) with distinct playstyles
- Complete combat system with normal attacks, specials, and grabs
- All defensive mechanics (shield, dodge, counter, reflector, super armor)
- Comprehensive visual effects (hit sparks, screen shake, damage flash, trails)
- Complete audio system with procedural sound generation
- Full game flow (title, character select, stage select, match, results)
- Training mode with CPU AI (5 difficulty levels), frame data, input history
- Complete replay system (record, playback, browser)
- Gamepad support with rumble feedback
- Parallax backgrounds and stage thumbnails
- Character select animations and sounds
- Performance profiler with FPS counter
- Enhanced pause menu (Resume/Char Select/Title)
- Animated damage percentage (scale/shake/color)
- Stock icons with X marks on lost stocks

**Total: ~19,000+ lines of pure Python + X11, no external game engines.**

---

## Cycle 20 Goals

### 1. COMPREHENSIVE TESTING

Run the game and verify all features work together:

**Import Tests:**
- All modules import without errors
- No circular dependency issues
- All dependencies available

**Functional Tests:**
- Game launches with display :99
- Menu navigation works (keyboard)
- Character selection for both players
- Stage selection with thumbnails
- Match starts with countdown
- Combat works (attacks, knockback, KO)
- Pause menu navigation works
- Match ends with results screen
- Training mode features work
- Replay save and playback works

**Visual Tests (via screenshots):**
- HUD displays correctly
- Damage percentage animates on hit
- Stock icons show X marks when lost
- Pause menu renders with options
- Visual effects appear on hits

### 2. FIX ANY ISSUES FOUND

If testing reveals bugs:
- Fix the issue in the BUILDING stage
- Re-test to confirm fix
- Document what was fixed

### 3. DOCUMENTATION

Create a summary of all features:
- Complete feature list by cycle
- Controls reference
- Character descriptions
- Architecture overview

---

## Success Criteria

1. **Game runs without crashes** - Complete game loop from title to results
2. **All 4 characters work** - Unique moves, specials, colors
3. **All game modes work** - VS, Training, Replays
4. **Cycle 19 features verified** - Pause menu, damage animation, stock icons
5. **Screenshot evidence** - Visual confirmation of working UI

---

## Step-by-Step Plan

### Step 1: Import Validation
**Action:** Run Python import test for all modules
**Expected:** All imports succeed without errors

### Step 2: Launch Test
**Action:** Start game on display :99, capture screenshot
**Expected:** Game window appears, title screen visible

### Step 3: Menu Navigation Test
**Action:** Navigate through menus programmatically or via input simulation
**Expected:** Can reach character select, stage select, settings

### Step 4: Match Test
**Action:** Start a VS match with default settings
**Expected:** Match runs, fighters respond to inputs

### Step 5: Combat Verification
**Action:** Execute attacks, verify knockback and damage
**Expected:** Hits register, damage percentage increases

### Step 6: Cycle 19 Feature Verification
**Action:** Pause game, check pause menu; take damage, check animation
**Expected:** Enhanced pause menu visible, damage animates

### Step 7: Training Mode Test
**Action:** Start training mode, enable CPU, check frame data
**Expected:** Training features work correctly

### Step 8: Replay Test
**Action:** Play a short match, save replay, load and playback
**Expected:** Replay saves and plays back correctly

### Step 9: Documentation
**Action:** Write final feature summary
**Expected:** Complete documentation in artifacts/

---

## Files Involved

Testing only - no code modifications expected unless bugs found:

| File | Purpose |
|------|---------|
| `main.py` | Test launch and game flow |
| `renderer.py` | Verify rendering works |
| `game_engine.py` | Verify combat mechanics |
| `characters.py` | Verify all characters |
| `audio_system.py` | Verify sounds work |
| `training_mode.py` | Verify training features |
| `menu_system.py` | Verify menu navigation |
| `replay_system.py` | Verify replay save/load |

---

## Celebration

When all tests pass, Stick Figure Fighter is **COMPLETE**:
- 20 development cycles
- ~19,000+ lines of code
- Built entirely from first principles
- Pure Python + X11, no game engines
- 4 unique characters with full movesets
- Complete game flow and polish
- Ready for future ML training integration

**The game is ready to play!**

---

# Cycle 13: Polish and Advanced Features

## Mission Overview

Cycles 1-12 delivered: core game loop with physics, 4 unique characters (RUSH, ZONE, GRAB, FLEX), projectile system, all defensive mechanics (shield, dodge, counter, reflector, super armor), comprehensive visual effects, complete audio system, character balance tuning, enhanced training mode with CPU difficulty and record/playback, full game flow (character/stage select, pre-match countdown, pause menu, results screen with victory/defeat poses), gamepad support (evdev), settings persistence (JSON save/load), complete Settings Menu UI with Audio/Controls/Gameplay submenus, COMPLETE REPLAY SYSTEM (recording, playback, browser), and audio settings integration.

**CURRENT PRIORITY:** Frame Advantage Display (primary) + Hit Spark Colors (secondary) + Gamepad Rumble (tertiary)

---

## Success Criteria

1. **Frame Advantage Display:**
   - Shows +/- number after hits in training mode
   - Green for positive, red for negative advantage
   - Displays near hit location
   - Fades out after ~1 second
   - Only active when training mode is enabled

2. **Hit Spark Colors:**
   - 0-10 damage: White sparks
   - 11-20 damage: Yellow sparks
   - 21-30 damage: Orange sparks
   - 31+ damage: Red sparks
   - Visual difference is noticeable

3. **Gamepad Rumble:**
   - Rumble triggers on receiving hit
   - Intensity scales with knockback
   - Duration scales with hit severity
   - Graceful handling when controller doesn't support rumble
   - No crashes on devices without EV_FF

---

## Research Summary

Key findings from research (see `research/research_findings_cycle13.md`):

1. **Frame Advantage Display:** Industry standard shows +/- with color coding near hit location
2. **Hit Spark Colors:** Damage-based color progression (white->yellow->orange->red)
3. **Gamepad Rumble:** evdev ff module provides force feedback API

**Sources Consulted:**
- [Guilty Gear Strive Frame Data Display](https://noisypixel.net/guilty-gear-strive-frame-data-ranked-match-update/)
- [2XKO Frame Data Guide](https://2xkocombos.com/mechanics/frame-data/)
- [Python-evdev Tutorial](https://python-evdev.readthedocs.io/en/latest/tutorial.html)
- [TV Tropes: Hit Spark](https://tvtropes.org/pmwiki/pmwiki.php/Main/HitSpark)
- [Super Smash Bros. Ultimate Hit Effects VFX](https://www.artstation.com/artwork/68JPzr)

---

## Step 1: Frame Advantage Display System

### 1.1 Add Frame Advantage Tracking to TrainingModeManager

**File:** `training_mode.py`

**Changes:**
- Add `FrameAdvantageEvent` dataclass to store hit events with timing info
- Add tracking variables to `TrainingModeManager`:
  - `last_frame_advantage: int = 0`
  - `frame_advantage_display_timer: int = 0`
  - `frame_advantage_position: Tuple[float, float] = (0, 0)`
- Add method `record_hit(attacker, defender, hitbox, contact_x, contact_y)`:
  - Calculate: `frame_advantage = defender.hitstun_frames - attacker_remaining_recovery`
  - Store result and position for display

**Frame Advantage Calculation:**
```python
# On hit:
# attacker_remaining_recovery = attack.total_frames - attacker.attack_frame
# defender_hitstun = applied hitstun from damage
# frame_advantage = defender_hitstun - attacker_remaining_recovery
# Positive = attacker can move first (advantage)
```

### 1.2 Add Frame Advantage Visual Overlay

**File:** `renderer.py`

**Changes:**
- Add new `EffectType.FRAME_ADVANTAGE` enum value
- Add `add_frame_advantage_effect(x, y, advantage, duration=60)` method
- Add `draw_frame_advantage(effect)` method:
  - Display `+X` or `-X` text
  - Color: Green (#00FF00) if positive, Red (#FF4444) if negative, White if 0
  - Position above hit location
  - Fade out over duration

### 1.3 Integrate with Game Hit Callbacks

**File:** `main.py`

**Changes:**
- Modify `_on_hit()` to also notify training manager when in training mode
- Pass attacker, defender, hitbox for advantage calculation
- Trigger renderer's frame advantage effect

**Files Affected:** `training_mode.py`, `renderer.py`, `main.py`

---

## Step 2: Hit Spark Color Variations

### 2.1 Modify add_hit_effect() to Accept Damage Parameter

**File:** `renderer.py`

**Current signature:**
```python
def add_hit_effect(self, x, y, knockback_magnitude, angle=0.0, attacker_color=(255, 255, 255)):
```

**New signature:**
```python
def add_hit_effect(self, x, y, knockback_magnitude, angle=0.0, attacker_color=(255, 255, 255), damage=0):
```

**Changes:**
- Add `damage` parameter
- Calculate spark color based on damage thresholds:
  - 0-10 damage: White (255, 255, 255)
  - 11-20 damage: Yellow (255, 255, 100)
  - 21-30 damage: Orange (255, 180, 80)
  - 31+ damage: Red (255, 100, 100)
- Store color in VisualEffect for draw_hit_spark()

### 2.2 Modify draw_hit_spark() to Use Damage-Based Color

**File:** `renderer.py`

**Current behavior:** Time-based color progression (white -> yellow -> orange)

**New behavior:**
- Use stored `color` from VisualEffect (damage-based)
- Apply intensity/fade on top of base color
- Keep central flash white for all hits

### 2.3 Update Callers to Pass Damage

**File:** `main.py`

**Changes to `_on_hit()` and `_on_projectile_hit()`:**
- Pass `damage=hitbox.damage` to `add_hit_effect()`

**Files Affected:** `renderer.py`, `main.py`

---

## Step 3: Gamepad Rumble Support

### 3.1 Add Rumble Capability Check

**File:** `gamepad_input.py`

**Changes:**
- Add conditional import: `from evdev import InputDevice, ff, ecodes`
- Add `supports_rumble` property to `Gamepad` class
- Open device with O_RDWR for rumble support
- Cache rumble capability check

### 3.2 Add Rumble Effect Method

**File:** `gamepad_input.py`

**Add to `Gamepad` class:**
```python
def rumble(self, strong: float, weak: float, duration_ms: int) -> bool:
    """
    Trigger rumble effect.

    Args:
        strong: Strong motor intensity (0.0 - 1.0)
        weak: Weak motor intensity (0.0 - 1.0)
        duration_ms: Duration in milliseconds

    Returns:
        True if rumble triggered, False otherwise
    """
```

### 3.3 Add GamepadManager Rumble Interface

**File:** `gamepad_input.py`

**Add to `GamepadManager` class:**
```python
def rumble_player(self, player_id: int, strong: float, weak: float, duration_ms: int) -> bool:
    """Trigger rumble for a player's gamepad."""
```

### 3.4 Integrate Rumble with Hit Events

**File:** `main.py`

**Modify `_on_hit()` to trigger rumble:**
- Scale rumble intensity based on knockback magnitude
- Apply to defender's player_id gamepad

**Files Affected:** `gamepad_input.py`, `main.py`

---

## File Summary

| File | Changes |
|------|---------|
| `training_mode.py` | Add frame advantage tracking, record_hit method |
| `renderer.py` | Add FRAME_ADVANTAGE effect type, damage-based spark colors, draw_frame_advantage() |
| `gamepad_input.py` | Add rumble support with evdev ff module |
| `main.py` | Wire up frame advantage display, damage to sparks, rumble triggers |

---

## Testing Strategy

1. **Frame Advantage:**
   - Enable training mode
   - Land attacks and verify +/- display
   - Check color coding (green/red/white)
   - Verify fading behavior

2. **Hit Spark Colors:**
   - Test with jabs (low damage) - white
   - Test with smash attacks (high damage) - orange/red
   - Capture screenshots for visual verification

3. **Gamepad Rumble:**
   - Connect gamepad with rumble support
   - Land hits on player using gamepad
   - Verify rumble intensity scales with damage
   - Test with controllers that don't support rumble (no crash)

---

## Dependencies

- evdev library (already installed for gamepad input)
- evdev.ff module for force feedback
- No new external dependencies required

---

# Cycle 14: Visual Polish & Game Feel Enhancements

## Mission Overview

Cycles 1-13 delivered: core game loop with physics, 4 unique characters (RUSH, ZONE, GRAB, FLEX), projectile system, all defensive mechanics, comprehensive visual effects (hit sparks with damage-based colors), complete audio system with crowd reactions, character balance tuning, enhanced training mode with CPU difficulty, record/playback, and FRAME ADVANTAGE DISPLAY, full game flow (character/stage select, pre-match countdown, pause menu, results screen), gamepad support (evdev) with RUMBLE FEEDBACK, settings persistence, complete Settings Menu UI, and COMPLETE REPLAY SYSTEM.

**CURRENT PRIORITY:** Hit Stop/Freeze (primary) + Parallax Backgrounds (secondary) + Character Entrances (tertiary) + Screen Shake Enhancement

---

## Research Summary

**Sources Consulted:**
- [SmashWiki - Hitlag](https://www.ssbwiki.com/Hitlag)
- [CritPoints - Hitstop Analysis](https://critpoints.net/2017/05/17/hitstophitfreezehitlaghitpausehitshit/)
- [Sakurai's Famitsu Column on Hitstop](https://sourcegaming.info/2015/11/11/thoughts-on-hitstop-sakurais-famitsu-column-vol-490-1/)
- [Gamasutra - Improving Combat Impact](https://www.gamedeveloper.com/audio/improving-the-combat-impact-of-action-games)
- [Envato Tuts+ - Parallax Scrolling](https://code.tutsplus.com/parallax-scrolling-a-simple-effective-way-to-add-depth-to-a-2d-game--cms-21510t)
- [Kevurugames - How to Animate a Fighting Game](https://kevurugames.com/blog/how-to-animate-a-fighting-game/)

**Key Findings:**
1. **Hit Stop:** Freeze both attacker and defender for 2-12 frames based on damage. Characters vibrate slightly. Particles/effects continue.
2. **Parallax:** 3-6 layers with decreasing scroll speeds (0.1, 0.3, 0.5, 0.7, 1.0). Calculate camera midpoint between fighters.
3. **Entrances:** Squash/stretch principles. Different timings per character archetype.

---

## Success Criteria

1. **Hit Stop/Freeze:**
   - Game freezes 2-12 frames on hits based on damage
   - Both attacker and defender freeze
   - Characters vibrate slightly during freeze
   - Effects and particles continue during freeze

2. **Screen Shake Enhancement:**
   - Shakes on hits with knockback > 15
   - Directional bias based on hit angle
   - Already partially implemented - verify integration

3. **Parallax Backgrounds:**
   - 3 distinct layers visible (sky, mountains, near details)
   - Layers move at different speeds during camera pan
   - Works with both stages (Battlefield, Final Destination)

4. **Character Entrances:**
   - Characters drop in before countdown
   - Different timings per archetype (RUSH fast, ZONE slow float, GRAB heavy, FLEX medium)
   - Countdown starts after both fighters land

5. **Knockback Particle Trails:**
   - Speed lines visible on high-knockback hitstun (>18 magnitude)
   - Uses fighter color for tint
   - Fades out as knockback slows

---

## Step 1: Hit Stop/Freeze System

### 1.1 Add Hit Freeze State to GameState

**File:** `game_engine.py`

**Add to GameState class:**
```python
# Hit freeze state
self.hit_freeze_frames: int = 0           # Frames remaining in freeze
self.hit_freeze_fighters: List[int] = []  # Player IDs frozen
self.hit_freeze_vibrate: int = 0          # Vibration frame counter
```

**Add method:**
```python
def apply_hit_freeze(self, attacker_id: int, defender_id: int, damage: float):
    """Apply hit freeze based on damage."""
    base_frames = 2
    damage_frames = int(damage * 0.3)
    self.hit_freeze_frames = min(base_frames + damage_frames, 12)
    self.hit_freeze_fighters = [attacker_id, defender_id]
    self.hit_freeze_vibrate = 0
```

### 1.2 Skip Physics During Freeze

**File:** `game_engine.py`

**Modify GameLoop.step():**
```python
def step(self) -> bool:
    if self.state.match_over:
        return False
    if self.state.paused:
        return True

    # Handle hit freeze
    if self.state.hit_freeze_frames > 0:
        self.state.hit_freeze_frames -= 1
        self.state.hit_freeze_vibrate += 1
        # Skip physics, combos, etc. but continue
        return True

    # Normal update logic...
```

### 1.3 Trigger Freeze on Hit

**File:** `game_engine.py`

**Modify _process_combat() after knockback applied:**
```python
# After defender.apply_knockback():
self.state.apply_hit_freeze(attacker.player_id, defender.player_id, hitbox.damage)
```

### 1.4 Add Vibration During Freeze

**File:** `renderer.py`

**Modify draw_stick_figure():**
```python
def draw_stick_figure(self, fighter: Fighter, game_state: GameState = None):
    # Add vibration offset if in hit freeze
    vibrate_offset_x = 0
    vibrate_offset_y = 0
    if game_state and game_state.hit_freeze_frames > 0:
        if fighter.player_id in game_state.hit_freeze_fighters:
            vibrate_offset_x = (game_state.hit_freeze_vibrate % 2) * 3 - 1
            vibrate_offset_y = ((game_state.hit_freeze_vibrate + 1) % 2) * 2 - 1

    # Apply offset to cx, cy calculations
```

---

## Step 2: Screen Shake Enhancement

### 2.1 Verify Knockback Threshold

**File:** `renderer.py`

**Verify in add_hit_effect():**
- Screen shake already triggered via `self.screen_shake.apply(knockback_magnitude, direction_x, direction_y)`
- Ensure threshold in ScreenShake.apply() starts at knockback > 5 (already exists)

### 2.2 Add Directional Bias

**File:** `renderer.py` - Already implemented in ScreenShake class

**Verify:**
- `direction_x` and `direction_y` calculated from hit angle
- Applied with 0.3 multiplier in ScreenShake.apply()

---

## Step 3: Parallax Background Layers

### 3.1 Add ParallaxLayer Dataclass

**File:** `game_engine.py`

**Add after Stage class:**
```python
@dataclass
class ParallaxLayer:
    """A parallax background layer."""
    scroll_speed: float  # 0.0 = static, 1.0 = full camera speed
    layer_type: str      # 'sky', 'mountains', 'clouds', 'ground_detail'
    color: Tuple[int, int, int] = (50, 50, 80)
```

**Add to Stage.create_basic():**
```python
parallax_layers=[
    ParallaxLayer(0.1, 'sky', (30, 40, 60)),
    ParallaxLayer(0.3, 'mountains', (60, 70, 90)),
    ParallaxLayer(0.5, 'clouds', (80, 90, 110)),
]
```

### 3.2 Add Parallax Drawing Methods

**File:** `renderer.py`

**Add camera calculation:**
```python
def _get_camera_x(self, game_state: GameState) -> float:
    """Get camera X position based on fighter midpoint."""
    p1_x = game_state.fighters[0].position.x
    p2_x = game_state.fighters[1].position.x
    midpoint = (p1_x + p2_x) / 2
    return midpoint - self.width / 2
```

**Add parallax drawing:**
```python
def _draw_parallax_background(self, game_state: GameState):
    """Draw parallax background layers."""
    camera_x = self._get_camera_x(game_state)

    for layer in game_state.stage.parallax_layers:
        offset_x = int(camera_x * layer.scroll_speed)
        self._draw_parallax_layer(layer, offset_x)

def _draw_parallax_layer(self, layer: ParallaxLayer, offset_x: int):
    """Draw a single parallax layer."""
    if layer.layer_type == 'sky':
        self._draw_sky_layer(layer.color)
    elif layer.layer_type == 'mountains':
        self._draw_mountains_layer(layer.color, offset_x)
    elif layer.layer_type == 'clouds':
        self._draw_clouds_layer(layer.color, offset_x)
```

### 3.3 Implement Layer Drawing Functions

**File:** `renderer.py`

```python
def _draw_sky_layer(self, color: Tuple[int, int, int]):
    """Draw gradient sky."""
    for y in range(0, self.height // 2, 2):
        progress = y / (self.height // 2)
        r = int(color[0] + (100 - color[0]) * progress * 0.3)
        g = int(color[1] + (120 - color[1]) * progress * 0.3)
        b = int(color[2] + (150 - color[2]) * progress * 0.3)
        self.set_color_rgb(r, g, b)
        self.draw_line(0, y, self.width, y)

def _draw_mountains_layer(self, color: Tuple[int, int, int], offset_x: int):
    """Draw distant mountain silhouettes."""
    # Draw triangle mountains at different positions
    mountain_data = [(200, 80), (500, 100), (800, 70), (1100, 90)]
    for base_x, height in mountain_data:
        x = base_x - offset_x
        peak_y = self.height // 2 - height
        base_y = self.height // 2 + 50
        # Draw mountain triangle
        self.set_color_rgb(*color)
        for dy in range(peak_y, base_y):
            width = int((dy - peak_y) * 0.8)
            self.draw_line(x - width, dy, x + width, dy)

def _draw_clouds_layer(self, color: Tuple[int, int, int], offset_x: int):
    """Draw floating cloud shapes."""
    cloud_positions = [(150, 80), (400, 120), (700, 60), (950, 100)]
    for base_x, y in cloud_positions:
        x = (base_x - offset_x) % (self.width + 200) - 100
        self.set_color_rgb(*color)
        self.draw_circle(x, y, 25, filled=True)
        self.draw_circle(x + 30, y - 5, 20, filled=True)
        self.draw_circle(x - 20, y + 5, 18, filled=True)
```

### 3.4 Integrate Parallax into Render

**File:** `renderer.py`

**Modify render() at start:**
```python
def render(self, game_state: GameState):
    # Update visual effects first (only when not paused)
    if not game_state.paused:
        self.update_effects()

    self.clear()

    # Draw parallax background first
    if hasattr(game_state.stage, 'parallax_layers') and game_state.stage.parallax_layers:
        self._draw_parallax_background(game_state)

    # Rest of render...
```

---

## Step 4: Character Entrance Animations

### 4.1 Add ENTERING State

**File:** `game_engine.py`

**Add to FighterState enum:**
```python
ENTERING = auto()  # Pre-match entrance animation
```

**Add to Fighter class:**
```python
self.entrance_frame: int = 0  # Frame counter for entrance animation
```

### 4.2 Add Entrance Poses

**File:** `renderer.py`

**Add poses:**
```python
POSE_ENTERING_RUSH = StickFigurePose(
    head_y=-70,
    shoulder_y=-55,
    hip_y=-15,
    left_elbow=(-25, -15),   # Arms back - diving
    left_hand=(-35, -5),
    right_elbow=(25, -15),
    right_hand=(35, -5),
    left_knee=(-5, 35),      # Legs together
    left_foot=(-5, 55),
    right_knee=(5, 35),
    right_foot=(5, 55)
)

POSE_ENTERING_ZONE = StickFigurePose(
    head_y=-65,
    shoulder_y=-50,
    hip_y=-10,
    left_elbow=(-30, 0),     # Arms out - floating
    left_hand=(-45, 5),
    right_elbow=(30, 0),
    right_hand=(45, 5),
    left_knee=(-10, 30),
    left_foot=(-12, 50),
    right_knee=(10, 30),
    right_foot=(12, 50)
)

POSE_ENTERING_GRAB = StickFigurePose(
    head_y=-60,
    shoulder_y=-45,
    hip_y=-8,
    left_elbow=(-20, 15),    # Arms down - heavy drop
    left_hand=(-25, 30),
    right_elbow=(20, 15),
    right_hand=(25, 30),
    left_knee=(-15, 25),
    left_foot=(-20, 45),
    right_knee=(15, 25),
    right_foot=(20, 45)
)

POSE_ENTERING_FLEX = StickFigurePose(
    head_y=-65,
    shoulder_y=-50,
    hip_y=-10,
    left_elbow=(-15, 5),     # Arms ready
    left_hand=(-20, 18),
    right_elbow=(15, 5),
    right_hand=(20, 18),
    left_knee=(-10, 28),
    left_foot=(-12, 50),
    right_knee=(10, 28),
    right_foot=(12, 50)
)
```

### 4.3 Add Entrance Handling to Countdown

**File:** `main.py`

**Modify _handle_countdown():**
```python
def _handle_countdown(self):
    if not self.countdown_state:
        self.mode = GameMode.MATCH
        return

    self.input_manager.poll()
    if self.input_manager.quit_requested:
        self.quit_requested = True
        return

    cs = self.countdown_state

    # Initialize entrance on first frame
    if cs.countdown_frame == 0 and self.game_state:
        for fighter in self.game_state.fighters:
            fighter.state = FighterState.ENTERING
            fighter.entrance_frame = 0
            # Start above screen
            fighter.position.y = -100

    cs.countdown_frame += 1

    # Update entrance animations
    if self.game_state:
        all_landed = True
        for fighter in self.game_state.fighters:
            if fighter.state == FighterState.ENTERING:
                all_landed = False
                self._update_fighter_entrance(fighter)

        # Start countdown sounds only after all fighters land
        if not all_landed:
            # Don't advance countdown values yet
            self._render_entrance()
            return

    # Rest of countdown logic...
```

### 4.4 Implement Entrance Update

**File:** `main.py`

```python
def _update_fighter_entrance(self, fighter: Fighter):
    """Update fighter during entrance animation."""
    fighter.entrance_frame += 1
    char_type = fighter.stats.character_type if fighter.stats else "FLEX"

    # Get entrance timing per character
    timings = {
        "RUSH": (15, 8.0),   # (frames, drop_speed)
        "ZONE": (40, 3.0),
        "GRAB": (25, 6.0),
        "FLEX": (30, 4.5),
    }
    max_frames, drop_speed = timings.get(char_type, (30, 4.5))

    # Get spawn Y position
    spawn_y = self.game_state.stage.spawn_points[fighter.player_id].y

    # Move down
    fighter.position.y += drop_speed

    # Land when reached spawn position
    if fighter.position.y >= spawn_y:
        fighter.position.y = spawn_y
        fighter.state = FighterState.IDLE
        fighter.grounded = True

        # Screen shake for heavy characters
        if char_type == "GRAB" and self.renderer:
            self.renderer.screen_shake.apply(8.0, 0, 1.0)

        # Landing dust
        if self.renderer:
            self.renderer.add_landing_dust(fighter.position.x, fighter.position.y)
```

---

## Step 5: Enhanced Knockback Particle Trails

### 5.1 Add Particle Spawning on High Knockback

**File:** `renderer.py`

**Enhance draw_knockback_trails():**
```python
def draw_knockback_trails(self, fighter: Fighter):
    if fighter.state != FighterState.HITSTUN:
        self.knockback_trails[fighter.player_id] = []
        return

    trails = self.knockback_trails.get(fighter.player_id, [])

    # Existing trail drawing...

    # Add speed line particles for high knockback
    kb_mag = fighter.knockback_vector.magnitude()
    if kb_mag > 18:
        # Draw speed lines behind fighter
        num_lines = min(6, int(kb_mag / 5))
        for i in range(num_lines):
            # Calculate line position behind movement direction
            angle = math.atan2(-fighter.velocity.y, -fighter.velocity.x)
            offset_angle = angle + (i - num_lines/2) * 0.2

            line_start_dist = 30 + i * 10
            line_end_dist = line_start_dist + 20 + kb_mag * 0.5

            x1 = fighter.position.x + math.cos(offset_angle) * line_start_dist
            y1 = fighter.position.y + math.sin(offset_angle) * line_start_dist
            x2 = fighter.position.x + math.cos(offset_angle) * line_end_dist
            y2 = fighter.position.y + math.sin(offset_angle) * line_end_dist

            # Fade based on distance from center
            fade = 1.0 - abs(i - num_lines/2) / (num_lines/2)
            r, g, b = fighter.color
            self.set_color_rgb(int(r * fade), int(g * fade), int(b * fade))
            self.draw_line(int(x1), int(y1), int(x2), int(y2))
```

---

## File Summary

| File | Changes |
|------|---------|
| `game_engine.py` | Hit freeze state, ParallaxLayer, Stage.parallax_layers, ENTERING state, entrance_frame |
| `renderer.py` | Parallax drawing, entrance poses, vibration during freeze, enhanced knockback trails |
| `main.py` | Hit freeze in game loop, entrance animation handling |

---

## Testing Strategy

1. **Hit Stop/Freeze:**
   - Land weak hit (jab): 2-3 frame freeze
   - Land strong hit (smash): 8-12 frame freeze
   - Verify slight vibration during freeze
   - Confirm effects continue during freeze

2. **Parallax Backgrounds:**
   - Move fighters across stage
   - Verify sky static, mountains slow, clouds faster
   - Check at stage edges

3. **Character Entrances:**
   - Start match, observe pre-countdown entrance
   - RUSH drops fast, ZONE floats slow, GRAB shakes on land
   - Countdown starts after all land

4. **Knockback Trails:**
   - Land high-damage smash attack
   - Verify speed lines appear during hitstun
   - Check lines fade and follow knockback direction
