# Research Findings: Stick Figure Fighter

---

# Cycle 20 Research: Final Validation & Celebration

## Overview

This is the **FINAL CYCLE (20 of 20)** of Stick Figure Fighter development. The game is complete and ready for comprehensive validation and documentation.

## Codebase Summary

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 1500+ | Game loop, input handling, menus, game flow |
| `renderer.py` | 2200+ | X11 rendering, visual effects, HUD |
| `game_engine.py` | 1000+ | Physics, combat, game state |
| `characters.py` | 800+ | RUSH, ZONE, GRAB, FLEX definitions |
| `audio_system.py` | 800+ | Procedural sound effects |
| `training_mode.py` | 1400+ | CPU AI, record/playback |
| `menu_system.py` | 600+ | Character/stage select, settings |
| `replay_system.py` | 500+ | Replay recording and playback |
| `gamepad_input.py` | 200+ | Controller support |
| `settings.py` | 300+ | Persistent settings |

**Total**: ~19,000+ lines of pure Python + X11

## Features By Cycle

1. **Cycle 1-2**: Core loop, physics, 4 characters
2. **Cycle 3-4**: Visual effects, audio system
3. **Cycle 5-6**: Training mode, menus
4. **Cycle 7-8**: Defensive mechanics, balance
5. **Cycle 9-10**: Projectiles, full game flow
6. **Cycle 11-12**: Combo counter, replay system
7. **Cycle 13-14**: Hit stop, parallax backgrounds
8. **Cycle 15-16**: Preview animations, profiler
9. **Cycle 17-18**: Stage thumbnails, char select SFX
10. **Cycle 19**: Pause menu, damage animation, stock icons

## Testing Strategy

All modules import successfully. Final testing will verify:
- Full game flow from title to results
- All 4 characters function correctly
- All game modes work (VS, Training, Replays)
- All Cycle 19 features (pause menu, damage animation, stock icons)

---

# Cycle 12 Research: Replay System Implementation

## Executive Summary - Cycle 12

This cycle focuses on:
1. Deterministic replay system using input recording
2. Replay file storage and browser UI
3. Audio settings integration with settings menu

## Sources Consulted

1. [Developing Your Own Replay System - GameDeveloper](https://www.gamedeveloper.com/programming/developing-your-own-replay-system)
2. [Things that can muddle your replay feature - GameDeveloper](https://www.gamedeveloper.com/design/things-that-can-muddle-your-replay-feature)
3. [Instant Replay: Building a Game Engine with Reproducible Behavior - GameDeveloper](https://www.gamedeveloper.com/design/instant-replay-building-a-game-engine-with-reproducible-behavior)
4. [Building Real-Time Fighting Games - Diarkis](https://www.diarkis.io/insights/building-real-time-fighting-games-with-diarkis-rollback-lockstep-and-scalable-infrastructure)

## Two Replay Approaches

### 1. Deterministic Input Recording (CHOSEN)
- Store initial game state + player inputs per frame
- Replay by feeding recorded inputs into game engine
- **Pros**: Compact storage, fast to record
- **Cons**: No random seeking, must replay from start

### 2. State Snapshot Recording
- Store full game state every frame
- Replay by restoring state frame by frame
- **Pros**: Random seeking, frame inspection
- **Cons**: Large file sizes, expensive to record

## Why Deterministic is Ideal for Stick Figure Fighter

1. **Already Deterministic**: The game engine uses a fixed timestep (60fps) with no external random state
2. **Fighting Game Standard**: Industry practice for fighting games
3. **Compact Storage**: Only need ~15 bytes per frame (2 players * ~8 input bits)
4. **ML Training Ready**: Replays can feed directly into training pipelines

## Critical Requirements for Determinism

### Input State Serialization
The `InputState` dataclass has 9 boolean values per player:
- left, right, up, down, jump, attack, special, shield, grab
- Total: 9 boolean values = easily serializable

### Frame Timing
- Fixed 60fps simulation (CONFIG.target_fps = 60)
- Already deterministic - same inputs produce same outputs

## Data Structure Design

### Replay Metadata
```json
{
    "version": "1.0",
    "timestamp": "ISO-8601 datetime",
    "duration_frames": 10800,
    "stage": "battlefield",
    "characters": ["RUSH", "GRAB"],
    "winner": 0,
    "final_stocks": [2, 0],
    "final_damage": [45.5, 152.0]
}
```

### Per-Frame Input Data (JSON for v1)
```json
{
    "frame": 0,
    "p1": {"left": false, "right": true, "jump": true, ...},
    "p2": {"left": false, "right": false, "attack": true, ...}
}
```

## Storage Considerations

- 3 minute match = 10,800 frames
- JSON format: ~50-100 bytes/frame = 500KB-1MB per replay
- With gzip compression: ~50-100KB per replay

## Implementation Integration Points

### game_engine.py
- `InputState` dataclass: Already serializable
- `GameLoop.step()`: Hook for recording inputs
- `GameState`: Initialize from replay metadata

### main.py
- Add `REPLAY` game mode
- Hook `_on_match_end` for save prompt
- Add replay browser state

### menu_system.py
- Add `REPLAYS` option to title screen (after Settings, before Quit)
- Create `ReplayBrowser` submenu

## Audio Settings Integration

### Current State
- `AudioManager.set_volume()` exists for master volume
- Settings menu has sliders but no callback wiring
- `on_settings_changed` callback defined but not used

### Required Changes
1. Wire menu callbacks to audio system
2. Call `audio_system.set_volume()` when slider changes
3. Apply mute toggle
4. Play test sound on volume change (MENU_MOVE)

---

# Cycle 7 Research: Menu Integration and Game Flow

## Executive Summary - Cycle 7

This cycle focuses on integrating the menu system into the main game loop to create a complete game flow:
1. Game state machine for menu -> match -> results -> menu transitions
2. Results screen with winner display and match statistics
3. Match settings integration (stock count, time limit)
4. Menu sound effects and polish

---

## 1. Game State Machine Architecture

### Sources Consulted
- [Game Programming Patterns - State](https://gameprogrammingpatterns.com/state.html)
- [GameDev.net - Making a Fighting Game State Machine](https://gamedev.net/forums/topic/703323-making-a-fighting-game-state-machine/)
- [GameDev.net - Gamestate and Intro, Main Menu, Playing state](https://www.gamedev.net/forums/topic/677463-gamestate-and-intro-main-menu-playing-state/)
- [Apple GameplayKit - State Machines](https://developer.apple.com/library/archive/documentation/General/Conceptual/GameplayKit_Guide/StateMachine.html)

### State Machine Pattern for Fighting Games

From Game Programming Patterns:
> "When an input comes in, if it matches a transition for the current state, the machine changes to the state that transition points to."

This applies to **both** character states (idle, attacking, hitstun) AND game flow states (menu, match, results).

### Recommended Game Flow States

```
MENU_CHARACTER_SELECT -> MENU_STAGE_SELECT -> IN_MATCH -> RESULTS -> MENU_CHARACTER_SELECT
                                  ^                          |
                                  |__________________________|  (rematch option)
```

| State | Description | Allowed Actions |
|-------|-------------|-----------------|
| MENU_CHARACTER_SELECT | Players choose characters | Navigate, select, cancel |
| MENU_STAGE_SELECT | Stage selection | Navigate, select, back |
| IN_MATCH | Active gameplay | Full game controls |
| RESULTS | Winner display | Rematch, return to menu |

### State Transition Implementation

From GameDev.net:
> "States should take care of their own rendering, update themselves, and deal with input themselves. They should also deal with setting the next state to transition to."

Each state needs:
- `update(inputs)` - Process logic and inputs
- `render()` - Draw the state
- `enter()` - Called on state entry (setup)
- `exit()` - Called on state exit (cleanup)

### Integration with Existing Architecture

Current architecture:
- `StickFighterGame` manages game loop
- `MenuSystem` handles menu state internally
- `GameLoop` manages match simulation

Recommended integration:
- Add `GameMode` enum: `MENU`, `MATCH`, `RESULTS`
- `StickFighterGame.mode` tracks current mode
- Route input/update/render based on mode

---

## 2. Results Screen Design

### Fighting Game Results Screen Best Practices

**Essential Elements:**
1. **Winner Announcement** - Clear "Player X Wins!" text
2. **Character Display** - Show winning character
3. **Match Statistics** - Damage dealt, stocks taken, combos
4. **Options** - Rematch or return to menu

**Statistics to Track:**
- Total damage dealt per player
- Stocks taken per player
- Max combo achieved
- Time elapsed

These are already tracked in GameState (Cycle 4 additions):
- `combo_counts`, `max_combos` - per player
- `fighters[i].damage_percent` - current damage
- `frame` - elapsed frames

**Additional tracking needed:**
- `total_damage_dealt[player_id]` - Sum of all damage dealt
- `stocks_taken[player_id]` - Number of KOs

### Results Screen UI Layout

```
+--------------------------------------------+
|                                            |
|           PLAYER 1 WINS!                   |
|                                            |
|     [P1 Character]     VS    [P2 Char]     |
|                                            |
|   =============== STATS ===============    |
|                                            |
|   Damage Dealt:  156%        89%           |
|   Stocks Taken:  3           1             |
|   Max Combo:     5 hits      2 hits        |
|   Time:          1:24.5                    |
|                                            |
|   [REMATCH]           [CHARACTER SELECT]   |
|                                            |
+--------------------------------------------+
```

### Input Handling on Results Screen

- Left/Right: Navigate between options
- Attack/Jump: Confirm selection
- Special: Return to character select (shortcut)

---

## 3. Match Settings Integration

### Settings to Configure

| Setting | Options | Default |
|---------|---------|---------|
| Stock Count | 1, 2, 3, 4, 5 | 3 |
| Time Limit | Off, 3, 5, 8 minutes | Off |

### Applying Settings to GameState

The `GameState` class already supports:
- `stock_count` via `CONFIG.stock_count` (per fighter)
- `time_limit` via constructor parameter (frames)

**Time conversion:** minutes to frames = minutes * 60 * 60

| Minutes | Frames |
|---------|--------|
| 0 (off) | 0 |
| 3 | 10800 |
| 5 | 18000 |
| 8 | 28800 |

### Settings UI Location Options

1. **Add to Stage Select** - Show settings on stage select screen (compact)
2. **Separate Settings Screen** - After stage select, before match (cleaner)
3. **In Character Select** - Settings panel on side (visible early)

**Recommendation:** Add settings panel to character select screen for visibility while keeping flow simple.

---

## 4. Menu Sound Effects

### Audio Integration

From `audio_system.py`, SoundType enum exists with:
- `MATCH_START`, `MATCH_END` - Already available
- No menu-specific sounds

### New Sounds Needed

| Sound | Description | Implementation |
|-------|-------------|----------------|
| CURSOR_MOVE | Short blip on navigation | Short sine sweep |
| MENU_CONFIRM | Selection confirmation | Rising tone |
| MENU_CANCEL | Cancel/back action | Descending tone |
| READY | Player locked in | Short fanfare |

### Sound Generation Patterns

From existing `SoundGenerator` class:
```python
# Cursor move - quick blip
cursor_move = gen.generate_sweep(0.05, 500, 600, 0.4)

# Confirm - rising tone
menu_confirm = gen.mix_sounds([
    gen.generate_sweep(0.1, 400, 800, 0.5),
    gen.generate_sine(0.08, 600, 0.3),
])

# Cancel - descending
menu_cancel = gen.generate_sweep(0.08, 600, 300, 0.4)
```

---

## 5. Implementation Architecture

### Main Game Modes

```python
class GameMode(Enum):
    MENU = auto()        # Character/stage select
    MATCH = auto()       # In-game match
    RESULTS = auto()     # Post-match results
    TRAINING = auto()    # Training mode (optional)
```

### StickFighterGame Extensions

```python
class StickFighterGame:
    def __init__(self, ...):
        self.mode = GameMode.MENU
        self.menu_system = MenuSystem()
        self.results_state = None  # ResultsState when in RESULTS mode

    def run(self):
        while not self.quit_requested:
            if self.mode == GameMode.MENU:
                self._update_menu()
                self._render_menu()
            elif self.mode == GameMode.MATCH:
                self._update_match()
                self._render_match()
            elif self.mode == GameMode.RESULTS:
                self._update_results()
                self._render_results()
```

### Transition Points

**Menu -> Match:**
- Triggered when `MenuSystem.is_ready_to_start()` returns True
- Extract settings from `MenuSystem.settings`
- Create new `GameState` with settings
- Create new `GameLoop`
- Switch mode to MATCH

**Match -> Results:**
- Triggered when `GameState.match_over` is True
- Capture final match statistics
- Create `ResultsState` with stats
- Switch mode to RESULTS

**Results -> Menu:**
- User selects "Character Select" option
- Reset `MenuSystem`
- Switch mode to MENU

**Results -> Match (Rematch):**
- User selects "Rematch" option
- Create new `GameState` with same settings
- Reset fighters to starting state
- Switch mode to MATCH

---

## 6. Pause Menu (If Time Permits)

### Pause Menu Options

1. Resume
2. Character Select (quit match)
3. Quit Game

### Implementation

- P key toggles pause (already implemented)
- When paused, show overlay menu
- Arrow keys navigate, Attack/Jump confirms

### Pause Menu in MATCH Mode

```python
if self.game_state.paused:
    self._update_pause_menu()
    self._render_pause_menu()
else:
    self._update_match_normal()
```

---

## 7. Sources Consulted (Cycle 7)

1. [Game Programming Patterns - State](https://gameprogrammingpatterns.com/state.html)
2. [GameDev.net - Making a Fighting Game State Machine](https://gamedev.net/forums/topic/703323-making-a-fighting-game-state-machine/)
3. [GameDev.net - Gamestate and Intro, Main Menu, Playing state](https://www.gamedev.net/forums/topic/677463-gamestate-and-intro-main-menu-playing-state/)
4. [Apple GameplayKit - State Machines](https://developer.apple.com/library/archive/documentation/General/Conceptual/GameplayKit_Guide/StateMachine.html)
5. [Meshy.ai - Game Loop Fundamentals 2025](https://www.meshy.ai/blog/game-loop)
6. [Ikemen-GO Fighting Game Engine (DeepWiki)](https://deepwiki.com/ikemen-engine/Ikemen-GO)
7. [DualShockers - Most Iconic Character Selection Menus](https://www.dualshockers.com/most-iconic-character-selection-menus/)

---

# Cycle 6 Research: Advanced Training Features & Polish

## Executive Summary - Cycle 6

This cycle focuses on:
1. Enhanced training mode (damage reset, position reset, infinite jumps, record/playback)
2. Improved CPU AI with difficulty levels and advanced behaviors (DI, teching)
3. Character select screen with preview and stats
4. Stage select screen with layout preview

---

## 1. Training Mode Best Practices (Industry Standard 2025)

### Sources Consulted
- [Game Rant: 8 Best Training Modes In Fighting Games](https://gamerant.com/best-training-modes-fighting-games/)
- [Dustloop Wiki: Using Training Mode](https://www.dustloop.com/w/Using_Training_Mode)
- [Street Fighter Wiki: Training Mode](https://streetfighter.fandom.com/wiki/Training_Mode)
- [Capcom SF6 Manual: Recording Tips](https://game.capcom.com/manual/SF6/en/ps5/page/8/6)

### Essential Training Mode Features

#### 1.1 Reset Hotkeys (CRITICAL)
Every modern training mode has instant reset:
- **Damage reset** to 0% for both fighters (R key)
- **Position reset** to starting positions (T key)
- Quick state reset without menu navigation
- SF6, Tekken 8, GGStrive all have single-button reset

**Implementation:** Trivial - just reset `fighter.damage_percent = 0.0` and `fighter.position` to spawn point.

#### 1.2 Record/Playback System (HIGH VALUE)
Core practice tool for fighting games:
- **Multiple recording slots** (typically 4-8) for different scenarios
- **Random slot playback** for unpredictable practice
- "Always Repeat Playback" option for projectile practice
- SF6 has 8 recording slots with selectable playback modes

**Capcom Recording Best Practices:**
- Record attack strings, movement patterns, and specific situations
- Use random playback for mixup practice
- Multiple recording slots enable practicing against different setups without re-recording

#### 1.3 Dummy Behavior Settings
Critical for focused practice:
- Stand/Crouch/Jump modes (already implemented as CPU behaviors)
- Guard settings: Never block, always block, auto-block, random guard
- Recovery settings: Fastest recovery, no recovery, tech/no-tech
- Counter-attack after guard/hit at configurable frames

#### 1.4 Infinite Jumps Toggle
For recovery practice:
- Disable air dodge limit per airborne state
- Reset double jump on each jump
- Allow unlimited jumps until grounded

---

## 2. CPU AI Improvements

### 2.1 Difficulty Levels

**Reaction Time Scaling:**
| Level | Reaction Delay | Description |
|-------|---------------|-------------|
| 1 | 30-45 frames | Very slow, doesn't punish |
| 2 | 20-30 frames | Casual, misses some punishes |
| 3 | 12-20 frames | Intermediate, consistent |
| 4 | 6-12 frames | Advanced, tight punishes |
| 5 | 0-6 frames | Frame-perfect reactions |

**Behavior Scaling:**
| Level | DI Strength | Tech Rate | Combo Follow-up |
|-------|-------------|-----------|-----------------|
| 1 | None | 0% | Never |
| 2 | Random | 20% | Rarely (10%) |
| 3 | Optimal 50% | 50% | Sometimes (30%) |
| 4 | Optimal 80% | 80% | Often (60%) |
| 5 | Optimal 95% | 95% | Always (90%) |

### 2.2 Directional Influence (DI)

**Platform Fighter Essential:**
- DI modifies knockback trajectory based on held direction
- Survival DI: Hold perpendicular to knockback to reduce distance
- Combo DI: Hold away from attacker to escape follow-ups

**Implementation:**
```python
def apply_knockback_with_di(self, knockback: Vector2, damage: float, di_input: Vector2):
    # DI modifies knockback angle by up to 18 degrees
    if di_input.magnitude() > 0.3:  # Deadzone
        di_angle = math.atan2(di_input.y, di_input.x)
        kb_angle = math.atan2(knockback.y, knockback.x)
        angle_diff = di_angle - kb_angle

        # Cap DI influence
        di_influence = min(abs(angle_diff), 0.314)  # ~18 degrees max
        if angle_diff < 0:
            di_influence = -di_influence

        new_angle = kb_angle + di_influence
        magnitude = knockback.magnitude()
        knockback = Vector2(
            math.cos(new_angle) * magnitude,
            math.sin(new_angle) * magnitude
        )
```

### 2.3 Teching (Ground Recovery)

**Platform Fighter Essential:**
- Press shield/tech button before hitting ground to recover instantly
- Tech timing window: 20 frames before landing
- Tech options: Tech in place, tech roll left, tech roll right
- Missed tech = vulnerable state on ground

**CPU Implementation:**
- At difficulty 3+: CPU attempts tech based on tech_rate
- Tech direction based on player position (roll away)

---

## 3. Character Select Screen Design

### 3.1 Screen Layout
```
+------------------------------------------+
|           CHARACTER SELECT               |
+------------------------------------------+
|                                          |
|   [RUSH]  [ZONE]  [GRAB]  [FLEX]        |
|     >P1    P2                            |
|                                          |
|  +------+  +------+                      |
|  | P1   |  | P2   |                      |
|  | RUSH |  | FLEX |                      |
|  | ===  |  | ===  |                      |
|  +------+  +------+                      |
|                                          |
|  Speed: ====                             |
|  Weight: ==                              |
|  Power: ===                              |
|                                          |
|         [PRESS START TO BEGIN]          |
+------------------------------------------+
```

### 3.2 Character Data Display
| Character | Speed | Weight | Power | Playstyle |
|-----------|-------|--------|-------|-----------|
| RUSH | 5/5 | 2/5 | 3/5 | Fast combo rushdown |
| ZONE | 2/5 | 3/5 | 4/5 | Ranged zoner |
| GRAB | 1/5 | 5/5 | 5/5 | Slow grappler |
| FLEX | 3/5 | 3/5 | 3/5 | Balanced all-rounder |

### 3.3 Cursor Navigation
- P1: A/D or Left/Right to move cursor
- P2: Arrow keys to move cursor
- Confirm: Jump button
- Back: Shield button

---

## 4. Stage Select Screen Design

### 4.1 Screen Layout
```
+------------------------------------------+
|            STAGE SELECT                  |
+------------------------------------------+
|                                          |
|   [BATTLEFIELD]    [FINAL DESTINATION]   |
|        *                                 |
|                                          |
|   +---------------------------+          |
|   |  Stage Preview Image     |          |
|   |                          |          |
|   |    ___________           |          |
|   |   ____   ____            |          |
|   |  _____________           |          |
|   +---------------------------+          |
|                                          |
|         [PRESS START]                    |
+------------------------------------------+
```

### 4.2 Stage Options
Current stages:
1. **Battlefield** - Main platform + 3 floating platforms
2. **Final Destination** - Single flat platform

Future options (if time permits):
3. **Small Battlefield** - Main + 1 top platform
4. **Omega Battlefield** - Flat version of Battlefield

---

## 5. Record/Playback Architecture

### 5.1 Recording Data Structure
```python
@dataclass
class RecordingSlot:
    """A single recording of inputs over time."""
    inputs: List[InputState]  # Frame-by-frame inputs
    duration: int  # Total frames recorded
    active: bool = False
    name: str = ""  # Optional label

@dataclass
class RecordingSystem:
    slots: List[RecordingSlot]  # 4-8 slots
    recording_slot: Optional[int] = None  # Currently recording to
    playback_slot: Optional[int] = None  # Currently playing back
    playback_frame: int = 0  # Current frame in playback
    playback_mode: str = "loop"  # "single", "loop", "random"

    def start_recording(self, slot_idx: int):
        self.recording_slot = slot_idx
        self.slots[slot_idx] = RecordingSlot(inputs=[], duration=0, active=True)

    def record_frame(self, input_state: InputState):
        if self.recording_slot is not None:
            slot = self.slots[self.recording_slot]
            slot.inputs.append(input_state)
            slot.duration += 1

    def stop_recording(self):
        self.recording_slot = None

    def get_playback_input(self) -> Optional[InputState]:
        if self.playback_slot is None:
            return None
        slot = self.slots[self.playback_slot]
        if not slot.active or self.playback_frame >= slot.duration:
            if self.playback_mode == "loop":
                self.playback_frame = 0
            else:
                return None
        return slot.inputs[self.playback_frame]
```

### 5.2 Controls
- F4: Start/stop recording to slot 1
- F5: Start/stop playback from slot 1
- Shift+F4: Cycle recording slot (1-4)
- Shift+F5: Cycle playback mode (single/loop/random)

---

## 6. Implementation Priority Order

### Phase 1: Training Mode Hotkeys (Highest Impact, Lowest Effort)
1. **R key: Damage reset** - Reset both fighters to 0%
2. **T key: Position reset** - Reset to starting positions
3. **Infinite jumps toggle** - Checkbox in training mode

### Phase 2: CPU AI Improvements
4. **Difficulty levels** - Scale reaction time and behavior
5. **DI implementation** - Modify knockback based on CPU input
6. **Teching** - Ground recovery at higher difficulties

### Phase 3: Selection Screens
7. **Character select screen** - Visual selection before match
8. **Stage select screen** - Choose stage before match

### Phase 4: Record/Playback (If Time Permits)
9. **Recording system** - Store inputs to slots
10. **Playback system** - Replay recorded inputs

---

## 7. Key Learnings from Research

1. **Reset functionality is universally expected** - Every modern training mode has it
2. **Record/playback is the #1 feature for skill development** - Without it, players must rely on training partners
3. **Frame data display is already implemented** - We're ahead of many indie fighters
4. **CPU difficulty affects reaction time, not just decision making** - Important distinction
5. **DI is essential for platform fighters** - Smash-style games require proper DI simulation
6. **Multiple recording slots** enable practicing against mixups without re-recording

---

## 8. Technical Notes

### Files to Modify
| Feature | Primary Files |
|---------|--------------|
| Damage/Position Reset | training_mode.py, main.py |
| Infinite Jumps | training_mode.py, game_engine.py |
| CPU Difficulty | training_mode.py (CPUController) |
| DI/Teching | game_engine.py (apply_knockback) |
| Character Select | New: character_select.py, renderer.py |
| Stage Select | New: stage_select.py, renderer.py |
| Record/Playback | training_mode.py, input_system.py |

### Input Mappings Available
Current unused keys that could be assigned:
- R, T (training mode resets)
- F4, F5 (recording system)
- 1-4 number keys (recording slot selection)
- G (toggle guard mode)

---

# Cycle 3 Research: Visual Effects & Game Feel

## Executive Summary - Cycle 3

Visual polish and "game juice" are critical for making the platform fighter feel satisfying. Key areas:
1. Hitstop/hitlag for impact feedback
2. Screen shake scaled to knockback
3. Hit sparks and damage flash
4. Projectile trails and glow effects
5. Shield visualization with health-based sizing
6. Dodge afterimages during invincibility

---

## 1. Hit Effects & Impact Feedback

### 1.1 Hitstop/Hitlag (Critical for Platform Fighters)

**Source:** [SmashWiki - Hitlag](https://www.ssbwiki.com/Hitlag), [CritPoints - Hitstop Analysis](https://critpoints.net/2017/05/17/hitstophitfreezehitlaghitpausehitshit/)

Hitstop is a cornerstone of fighting game visual design:

- **Definition**: Both the attacker and victim freeze for a certain number of frames when an attack connects
- **Purpose**:
  1. Visual indicator that attack connected
  2. Gives players time to plan next move
  3. Emphasizes power of strong attacks

**Implementation Formula (Smash Ultimate style):**
```
hitlag_frames = floor(damage_percent / 3) + 3
hitlag_frames = min(hitlag_frames, 20)  # Cap at 20 frames
```

**Visual During Hitstop:**
- Characters vibrate/shake during freeze (horizontal on ground, vertical in air)
- Only affected characters freeze; background/particles continue
- Electric attacks get extended hitstop

### 1.2 Hit Sparks

**Source:** [GameDeveloper - Improving Combat Impact](https://www.gamedeveloper.com/audio/improving-the-combat-impact-of-action-games), [pimen - Battle VFX](https://pimen.itch.io/battle-vfx-hit-spark)

- Typical size: 48x48 or 64x64 pixels
- Typical duration: 6-8 frames
- Color should reflect attack type (fire=orange, electric=yellow/blue, neutral=white)
- Direction of particles should indicate knockback direction
- Spawn at point of contact between hitbox and hurtbox

### 1.3 Victim Flash/Damage Indication

**Source:** [GameDeveloper - Improving Combat Impact](https://www.gamedeveloper.com/audio/improving-the-combat-impact-of-action-games)

- Flash the hit character white/red briefly
- 2-4 frame flash duration is standard
- Can overlay a white sprite or modify color values
- Helps communicate "this character took damage"

---

## 2. Screen Shake Best Practices

**Source:** [davetech.co.uk - Screenshake Analysis](http://www.davetech.co.uk/gamedevscreenshake), [Feel Documentation](https://feel-docs.moremountains.com/screen-shakes.html)

### 2.1 Types of Screen Shake

1. **Random Offset Shake**: Random camera displacement each frame
   - Easy to implement
   - Good for general impact

2. **Directional Shake**: Shake in direction of knockback
   - More intentional feel
   - Reinforces attack direction

3. **Diminishing Shake**: Strong initial shake that fades out
   - Better for sustained feedback
   - Not exhausting on eyes

### 2.2 Implementation Best Practices

- **Scale intensity with knockback magnitude** - weak hits = small shake, smash attacks = big shake
- **Cap maximum shake** - prevent overwhelming the player when multiple hits occur
- **Decay over time** - use exponential decay for natural feel
- **Don't shake on every hit** - only shake above a knockback threshold

**Formula:**
```python
shake_magnitude = min(knockback_magnitude * 0.5, MAX_SHAKE)
shake_duration = min(knockback_magnitude * 0.1, MAX_DURATION)
# Each frame: shake *= 0.9 (decay)
```

---

## 3. Game Feel / "Juice" Principles

**Source:** [GameAnalytics - Game Juice](https://www.gameanalytics.com/blog/squeezing-more-juice-out-of-your-game-design), [GameDev Academy - Game Feel](https://gamedevacademy.org/game-feel-tutorial/)

### 3.1 Core Principles

1. **Every action should have feedback** - no silent actions
2. **Intensity matches significance** - big moves = big effects
3. **Mix feedback types** - visual + audio + camera
4. **Performance awareness** - effects shouldn't tank framerate

### 3.2 Specific Techniques for 2D Games

- **Position animation**: Offset sprites during impacts
- **Scale animation**: Squash/stretch on jumps and landings
- **Rotation animation**: Slight tilt during movement
- **Color manipulation**: Flashes, tints, overlays
- **Particle trails**: Behind fast-moving objects

### 3.3 Avoiding Over-Juicing

**Source:** [Wayline - Perils of Over-Juicing](https://www.wayline.io/blog/the-perils-of-over-juicing)

- Don't mask weak mechanics with flashy effects
- Keep effects readable for competitive play
- Consider accessibility (option to reduce effects)
- Effects should enhance, not obscure gameplay

---

## 4. Platform Fighter-Specific Effects

### 4.1 Shield Effects

- Shield should be semi-transparent bubble/dome
- Size correlates with shield health
- Color shifts: Green (full) -> Yellow -> Red (low)
- Shield break should have distinct "shatter" effect

### 4.2 Dodge/Roll Effects

- Afterimages/ghost trail during invincibility
- Blur effect for air dodge
- Roll shows character rotating/tumbling

### 4.3 Super Armor Indicator

- Glow/aura around character during armor frames
- Sparks fly when hit but not flinched
- Color distinct from other effects (often gold/yellow)

### 4.4 Projectile Effects

- Trail effect showing recent positions
- Glow around projectile
- Distinct colors per character/type
- Flash/explosion on despawn

---

## 5. Implementation Recommendations for Stick Figure Fighter

Based on research, prioritize these effects:

### High Priority (Must Have)
1. **Projectile Rendering** - Colored circles with glow - ALREADY EXISTS in renderer.py
2. **Hit Sparks** - Simple white/orange burst at hit location
3. **Screen Shake** - Directional, scaled to knockback, with decay
4. **Damage Flash** - Brief white flash on hit character

### Medium Priority (Should Have)
1. **Hitstop** - Freeze both fighters on hit, 3-8 frames based on damage
2. **Shield Visualization** - Bubble, health-based size/color - ALREADY EXISTS in renderer.py
3. **Dodge Afterimages** - Ghost trail during invincibility
4. **Shield Break Effect** - Distinct visual for break

### Lower Priority (Nice to Have)
1. **Super Armor Glow** - Aura during armor frames
2. **Counter/Reflector Flash** - Burst effect on successful counter
3. **Knockback Trails** - Trail behind launched characters
4. **Landing Dust** - Small particle puff on landing

---

## 6. Technical Implementation Details

### 6.1 Effect Data Structure

```python
from enum import Enum, auto
from dataclasses import dataclass

class EffectType(Enum):
    HIT_SPARK = auto()
    DAMAGE_FLASH = auto()
    AFTERIMAGE = auto()
    DUST_PARTICLE = auto()
    SHIELD_BREAK = auto()
    SUPER_ARMOR_SPARK = auto()
    COUNTER_FLASH = auto()

@dataclass
class VisualEffect:
    effect_type: EffectType
    position: Vector2
    frame: int = 0
    duration: int = 10
    color: Tuple[int, int, int] = (255, 255, 255)
    scale: float = 1.0
    direction: float = 0.0  # For directional effects
    active: bool = True
```

### 6.2 Screen Shake State

```python
@dataclass
class ScreenShake:
    magnitude: float = 0.0
    duration: int = 0
    direction: Vector2 = Vector2(0, 0)  # Optional directional bias
    decay: float = 0.9

    def update(self):
        if self.duration > 0:
            self.magnitude *= self.decay
            self.duration -= 1
            if self.magnitude < 0.5:
                self.magnitude = 0
                self.duration = 0

    def get_offset(self) -> Vector2:
        if self.magnitude <= 0:
            return Vector2(0, 0)
        # Random offset within magnitude
        angle = random.uniform(0, 2 * math.pi)
        if self.direction.magnitude() > 0:
            # Bias toward knockback direction
            angle = math.atan2(self.direction.y, self.direction.x) + random.uniform(-0.5, 0.5)
        return Vector2(
            math.cos(angle) * self.magnitude,
            math.sin(angle) * self.magnitude
        )
```

### 6.3 Hitstop Implementation

```python
class GameState:
    def __init__(self, ...):
        # ... existing code ...
        self.hitstop_frames: int = 0
        self.hitstop_fighters: List[int] = []  # Player IDs in hitstop

    def apply_hitstop(self, attacker_id: int, defender_id: int, damage: float):
        frames = min(int(damage / 3) + 3, 20)
        self.hitstop_frames = frames
        self.hitstop_fighters = [attacker_id, defender_id]
```

### 6.4 Rendering Order

1. Background/Stage
2. Projectiles (behind fighters)
3. Fighters (apply screen shake offset)
4. Shield bubbles
5. Hit effects/sparks
6. Afterimages
7. HUD (no shake applied)

---

## 7. Sources Consulted (Cycle 3)

1. [Gamedeveloper - Improving Combat Impact](https://www.gamedeveloper.com/audio/improving-the-combat-impact-of-action-games)
2. [SmashWiki - Hitlag](https://www.ssbwiki.com/Hitlag)
3. [CritPoints - Hitstop Analysis](https://critpoints.net/2017/05/17/hitstophitfreezehitlaghitpausehitshit/)
4. [davetech.co.uk - Screenshake Analysis](http://www.davetech.co.uk/gamedevscreenshake)
5. [Feel Documentation - Screen Shakes](https://feel-docs.moremountains.com/screen-shakes.html)
6. [GameAnalytics - Game Juice](https://www.gameanalytics.com/blog/squeezing-more-juice-out-of-your-game-design)
7. [GameDev Academy - Game Feel](https://gamedevacademy.org/game-feel-tutorial/)
8. [Wayline - Perils of Over-Juicing](https://www.wayline.io/blog/the-perils-of-over-juicing)
9. [SourceGaming - Sakurai on Hitstop](https://sourcegaming.info/2015/11/11/thoughts-on-hitstop-sakurais-famitsu-column-490-1/)
10. [pimen - Battle VFX: Hit Spark](https://pimen.itch.io/battle-vfx-hit-spark)

---
---

# Previous Cycles Research (Cycles 0-2)

---

## Executive Summary

Building a platform fighter from first principles requires careful attention to:
1. Fixed timestep physics simulation for deterministic gameplay
2. Smash Brothers-style knockback formulas
3. X11 direct rendering with double buffering
4. Swept AABB collision detection for fast-moving objects
5. State machine-based character control

---

## 1. Fixed Timestep Game Loop

### Source: [Gaffer On Games - Fix Your Timestep](https://gafferongames.com/post/fix_your_timestep/)

**The Problem:** Variable timesteps cause physics instability and non-deterministic behavior.

**The Solution:** Decouple rendering from physics using the accumulator pattern:

```python
dt = 1.0 / 60.0  # 60Hz physics
accumulator = 0.0

while running:
    frame_time = get_frame_time()
    accumulator += frame_time

    while accumulator >= dt:
        previous_state = current_state.copy()
        integrate(current_state, t, dt)
        accumulator -= dt
        t += dt

    # Interpolate for smooth rendering
    alpha = accumulator / dt
    render_state = lerp(previous_state, current_state, alpha)
    render(render_state)
```

**Key Insight:** For fighting games, determinism is critical. The same inputs must produce the same outputs for fair gameplay and potential netcode.

### Spiral of Death Prevention

If physics takes longer than the timestep to compute, the game falls behind. Solution: clamp maximum steps per frame.

```python
MAX_STEPS = 5
steps = 0
while accumulator >= dt and steps < MAX_STEPS:
    # ... physics update
    steps += 1
```

---

## 2. Knockback Physics (Smash Brothers Style)

### Source: [SmashWiki - Knockback](https://www.ssbwiki.com/Knockback)

**Core Formula (Melee onward):**

```
knockback = (((((p/10 + p*d/20) * 200/(w+100) * 1.4) + 18) * s) + b) * r
```

Where:
- `p` = Target's damage percentage (after attack damage is added)
- `d` = Damage dealt by the attack
- `w` = Target's weight (100 = baseline)
- `s` = Attack's knockback scaling / 100
- `b` = Attack's base knockback
- `r` = Multiplier chain (rage, crouch cancel, etc.)

**Launch Speed:** `knockback * 0.03`

**Decay:** Launch speed decays by `0.051` per frame

**Hitstun Formula (Melee):** `hitstun_frames = knockback * 0.4`

### Simplified Implementation for Stick Fighter:

```python
def calculate_knockback(attacker_damage, target_damage, target_weight,
                        base_kb, kb_scaling, attack_damage):
    # Post-hit damage
    p = target_damage + attack_damage

    # Weight factor (heavier = less knockback)
    weight_factor = 200 / (target_weight + 100)

    # Base calculation
    kb = (p / 10 + p * attack_damage / 20) * weight_factor * 1.4
    kb = (kb + 18) * kb_scaling + base_kb

    return kb

def get_launch_velocity(knockback, angle):
    speed = knockback * 0.03
    return Vector2(
        speed * math.cos(angle),
        speed * math.sin(angle)
    )
```

---

## 3. Hitstun and Combo System

### Source: [SmashWiki - Hitstun](https://www.ssbwiki.com/Hitstun)

**Hitstun = inability to act after being hit.** Directly proportional to knockback.

| Game | Hitstun Multiplier |
|------|-------------------|
| SSB64 | 0.533x |
| Melee | 0.40x |
| Brawl+ | 0.40x (with canceling) |
| Ultimate | Variable (balloon KB) |

**For Stick Fighter:** Use Melee's `0.4x` multiplier for responsive combos.

**Tumble Threshold:** If knockback > 80, character enters tumble state.

---

## 4. X11 Direct Rendering

### Sources:
- [X.org Double Buffer Extension](https://www.x.org/releases/X11R7.7/doc/libXext/dbelib.html)
- [tsoding/x11-double-buffering](https://github.com/tsoding/x11-double-buffering)

**Approach Options:**

1. **XImage + Shared Memory (SHM):** Fast pixel manipulation
   - Already have x11_bindings.py with SHM capture
   - Can reverse for rendering

2. **X11 Drawing Primitives:** XDrawLine, XDrawRectangle, etc.
   - Lower overhead for simple shapes
   - Perfect for stick figures

3. **Double Buffer Extension (DBE):** Hardware double-buffering
   - Prevents tearing
   - Requires XdbeAllocateBackBufferName, XdbeSwapBuffers

**Recommended:** Use X11 drawing primitives + SHM back buffer for stick figures. Simple, fast, no external dependencies.

---

## 5. Swept AABB Collision Detection

### Source: [Swept AABB Using Minkowski Difference](https://blog.hamaluik.ca/posts/swept-aabb-collision-using-minkowski-difference/)

**Why Swept?** Fast objects can "tunnel" through obstacles between discrete frames.

**Algorithm:**

1. Compute Minkowski difference: `B_minkowski = B - A`
2. Ray cast from origin along A's velocity
3. Find intersection point with Minkowski AABB
4. `t` value (0-1) indicates collision time

```python
def swept_aabb(box_a, velocity_a, box_b):
    # Minkowski sum: expand B by A's dimensions
    minkowski = AABB(
        box_b.x - box_a.w,
        box_b.y - box_a.h,
        box_b.w + box_a.w,
        box_b.h + box_a.h
    )

    # Ray from A's center along velocity
    origin = Point(box_a.x + box_a.w/2, box_a.y + box_a.h/2)

    # Find entry/exit times for each axis
    t_entry, t_exit = ray_vs_aabb(origin, velocity_a, minkowski)

    if t_entry <= t_exit and t_entry >= 0 and t_entry <= 1:
        return t_entry  # Collision time
    return 1.0  # No collision
```

**Broad-Phase Optimization:** Use spatial hash or grid for O(n) instead of O(n^2) checks.

---

## 6. State Machine Architecture

**Character States (from Game_Idea.md):**

```
IDLE
WALKING
RUNNING
JUMPING
FALLING
ATTACKING
HITSTUN
SHIELDING
GRABBING
GRABBED
LEDGE
```

**State Transition Rules:**

```python
class CharacterState(Enum):
    IDLE = auto()
    WALK = auto()
    RUN = auto()
    JUMP_SQUAT = auto()
    AIRBORNE = auto()
    ATTACK = auto()
    HITSTUN = auto()
    SHIELD = auto()
    # ...

# Each state defines:
# - Entry conditions
# - Exit conditions
# - Available actions
# - Physics modifiers
```

---

## 7. Headless Mode for ML Training

### Key Insight:
Rendering should be completely optional. The game simulation runs identically whether or not frames are drawn.

**Architecture:**

```python
class Game:
    def __init__(self, headless=True):
        self.headless = headless
        self.renderer = None if headless else X11Renderer()

    def step(self, inputs: dict):
        self.physics.update()
        self.update_states()

        if not self.headless:
            self.renderer.draw(self.get_render_state())

        return self.get_observation()
```

**Observation Space (for ML):**
- Character positions (x, y) - normalized
- Character velocities (vx, vy) - normalized
- Damage percentages (0-999 scaled to 0-1)
- States (one-hot encoded)
- Stock counts
- Facing directions

---

## 8. Architecture Decisions

### Chosen Approach:

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Physics | 60Hz fixed timestep | Determinism, matches Smash |
| Collision | Swept AABB | Prevents tunneling at high speeds |
| Rendering | X11 primitives + SHM | No dependencies, fast enough |
| Input | X11 keyboard polling | Already have bindings |
| State | Finite State Machine | Clean transitions, easy to debug |
| ML | Decoupled observation | Can train headless |

### File Structure:

```
stick_figure_fighter/
├── core/
│   ├── game.py           # Main game loop
│   ├── physics.py        # Physics engine
│   ├── collision.py      # AABB, swept collision
│   └── constants.py      # Game constants
├── entities/
│   ├── character.py      # Character class
│   ├── states.py         # State machine
│   └── hitbox.py         # Hitbox/Hurtbox
├── input/
│   ├── keyboard.py       # X11 keyboard input
│   └── action_map.py     # Input abstraction
├── rendering/
│   ├── renderer.py       # X11 renderer
│   └── stick_figure.py   # Procedural stick figure
├── stages/
│   └── stage.py          # Stage/platform data
├── training/
│   ├── environment.py    # RL environment wrapper
│   └── observation.py    # State vectorization
└── main.py               # Entry point
```

---

## Sources Consulted

1. [Gaffer On Games - Fix Your Timestep](https://gafferongames.com/post/fix_your_timestep/)
2. [SmashWiki - Knockback](https://www.ssbwiki.com/Knockback)
3. [SmashWiki - Hitstun](https://www.ssbwiki.com/Hitstun)
4. [X.org Double Buffer Extension](https://www.x.org/releases/X11R7.7/doc/libXext/dbelib.html)
5. [Swept AABB Collision Using Minkowski Difference](https://blog.hamaluik.ca/posts/swept-aabb-collision-using-minkowski-difference/)
6. [GameDev.net - Swept AABB Tutorial](https://www.gamedev.net/tutorials/programming/general-and-gameplay-programming/swept-aabb-collision-detection-and-response-r3084/)
7. [Intersection Tests in 2D](https://noonat.github.io/intersect/)
8. [SSBU Calculator](https://rubendal.github.io/SSBU-Calculator/)
9. [André Leite - Fixed Timestep Game Loop](https://andreleite.com/posts/2025/game-loop/fixed-timestep-game-loop/)

---

# Cycle 1 Research: Character Archetypes & Differentiation

## Executive Summary - Cycle 1

Implementing 4 distinct characters (RUSH, ZONE, GRAB, FLEX) requires:
1. Carefully tuned stat distributions based on Smash Bros Ultimate data
2. Character-specific attack movesets with different frame data, hitboxes, and knockback
3. Visual differentiation through color coding
4. Archetype-faithful gameplay patterns

---

## 1. Fighting Game Character Archetypes

### Sources:
- [FrayTools Wiki - Character Archetypes](https://fraytools.fandom.com/wiki/Character_Archetypes)
- [Smashpedia - Character Archetypes](https://supersmashbros.fandom.com/wiki/Character_Archetypes)
- [PlayStation Compete - Fighting Game Archetypes](https://compete.playstation.com/en-us/all/articles/what-is-a-rushdown-ungabBunga-character-fighting-game-archetypes-fgc-explained)

### Core Archetypes:

| Archetype | Playstyle | Key Traits | Weaknesses |
|-----------|-----------|------------|------------|
| **Rushdown** | Aggressive, pressure-focused | Fast speed, quick attacks, combo potential | Low weight, weak range |
| **Zoner** | Keep-away, projectile control | Projectiles, long-range normals, anti-airs | Slow, weak up close |
| **Grappler** | Command grab focused | High damage, tanky, devastating throws | Slowest, poor approach |
| **All-Rounder** | Balanced fundamentals | Medium everything, adaptable | No overwhelming strengths |

### Rock-Paper-Scissors Dynamics:
- Rushdown beats Zoner (can close distance before projectiles accumulate damage)
- Zoner beats Grappler (can keep grappler at bay for entire match)
- Grappler beats Rushdown (grapplers want close range anyway, punish aggression)
- All-Rounder goes 50-50 with all archetypes

---

## 2. Smash Bros Ultimate Attribute Reference Values

### Sources:
- [SmashWiki - Weight](https://www.ssbwiki.com/Weight)
- [KuroGaneHammer - Run Speed](https://kuroganehammer.com/Ultimate/runspeed)
- [KuroGaneHammer - Air Speed](https://kuroganehammer.com/Ultimate/airSpeed)

### Weight Values (SSBU):
| Category | Range | Examples |
|----------|-------|----------|
| **Lightweight** | 62-85 | Pichu (62), Jigglypuff (68), Sheik (78) |
| **Middleweight** | 86-105 | Mario (98), Cloud (100), Ryu (103) |
| **Heavyweight** | 106-135 | Donkey Kong (127), King K. Rool (133), Bowser (135) |

**Average weight:** 95.92

### Run Speed Values (SSBU):
| Category | Range | Examples |
|----------|-------|----------|
| **Fast** | 2.0-3.85 | Sonic (3.85), Captain Falcon (2.552), Little Mac (2.464) |
| **Medium** | 1.5-2.0 | Mario (1.76), Lucario (1.705) |
| **Slow** | 1.18-1.5 | Incineroar (1.18), Robin (1.265), Jigglypuff (1.271) |

### Air Speed Values (SSBU):
| Category | Range | Examples |
|----------|-------|----------|
| **Fast** | 1.2-1.344 | Yoshi (1.344), Jigglypuff (1.332), Mewtwo (1.313) |
| **Medium** | 1.0-1.2 | Shulk (1.113), Fox (1.11), Samus (1.103) |
| **Slow** | 0.735-1.0 | King Dedede (0.735), Luigi (0.77), Ganondorf (0.83) |

---

## 3. Proposed Character Stat Distributions

Based on existing `FighterStats` defaults and archetype requirements:

### Baseline (Current Default):
```python
walk_speed = 3.0
run_speed = 6.0
air_speed = 4.0
jump_force = 14.0
double_jump_force = 12.0
weight = 100.0
fall_speed = 0.6
fast_fall_speed = 1.2
width = 50
height = 80
```

### RUSH - The Rushdown (Fast, Light, Combo-focused)
```python
# 20% faster ground speed, 15% faster air speed
# 20% lighter weight (more knockback received, but harder to hit)
# Slightly higher jump for aerial approaches
walk_speed = 3.6       # +20%
run_speed = 7.2        # +20%
air_speed = 4.6        # +15%
jump_force = 15.0      # +7%
double_jump_force = 13.0
weight = 80.0          # -20% (takes more knockback)
fall_speed = 0.7       # Slightly faster fall for fast falling mixups
fast_fall_speed = 1.4
width = 45             # Slightly thinner hurtbox
height = 75
```

### ZONE - The Zoner (Projectile-based, Medium stats)
```python
# Average ground speed, good air speed for aerial keepaway
# Slightly below average weight (easier to punish if approached)
walk_speed = 2.8       # -7%
run_speed = 5.4        # -10%
air_speed = 4.4        # +10% (needs aerial mobility for zoning)
jump_force = 14.0      # Standard
double_jump_force = 12.0
weight = 90.0          # -10% (punishable when cornered)
fall_speed = 0.55      # Floatier for projectile control
fast_fall_speed = 1.1
width = 50
height = 80
```

### GRAB - The Grappler (Slow, Heavy, Powerful)
```python
# Slowest character, heaviest weight, highest damage potential
walk_speed = 2.0       # -33%
run_speed = 4.2        # -30%
air_speed = 3.0        # -25%
jump_force = 12.0      # -14% (poor recovery)
double_jump_force = 10.0
weight = 130.0         # +30% (survives much longer)
fall_speed = 0.7       # Fast fall due to weight
fast_fall_speed = 1.3
width = 60             # Larger hurtbox (easier to hit)
height = 90
```

### FLEX - The All-Rounder (Balanced, Fundamental)
```python
# True middle-of-the-road stats
# Excels at nothing but has no weaknesses
walk_speed = 3.0       # Standard
run_speed = 6.0        # Standard
air_speed = 4.0        # Standard
jump_force = 14.0      # Standard
double_jump_force = 12.0
weight = 100.0         # Standard
fall_speed = 0.6
fast_fall_speed = 1.2
width = 50
height = 80
```

---

## 4. Attack Frame Data Design Principles

### Sources:
- [Smash Ultimate Calculator](https://rubendal.github.io/SSBU-Calculator/)
- [McLeodGaming Wiki - Knockback](https://mcleodgaming.fandom.com/wiki/Knockback)

### Attack Property Guidelines by Archetype:

| Property | RUSH | ZONE | GRAB | FLEX |
|----------|------|------|------|------|
| Startup | 3-6 frames | 8-15 frames | 10-20 frames | 5-10 frames |
| Active | 2-4 frames | 3-8 frames | 5-10 frames | 3-6 frames |
| Recovery | 8-15 frames | 15-25 frames | 20-35 frames | 12-20 frames |
| Total | ~20 frames | ~40 frames | ~55 frames | ~30 frames |
| Base KB | Low (1-3) | Medium (3-5) | High (6-10) | Medium (3-5) |
| KB Scaling | High (0.08-0.12) | Medium (0.06-0.08) | Very High (0.10-0.15) | Medium (0.06-0.10) |
| Damage | 2-6% | 5-10% | 10-18% | 5-10% |

### Special Move Design (from Game_Idea.md):

**RUSH Specials:**
- Neutral-B: Rapid punches (multi-hit, low damage per hit, fast startup)
- Side-B: Dash attack with follow-up (covers distance, combo starter)
- Up-B: Rising uppercut (recovery + kill move)
- Down-B: Counter stance (defensive option)

**ZONE Specials:**
- Neutral-B: Chargeable projectile (key zoning tool)
- Side-B: Horizontal beam (covers horizontal space)
- Up-B: Teleport (recovery, no hitbox)
- Down-B: Trap placement (stage control)

**GRAB Specials:**
- Neutral-B: Armored charge (approach tool with super armor)
- Side-B: Command grab (high damage throw)
- Up-B: Spinning rise (poor recovery, some horizontal hitbox)
- Down-B: Ground pound (OTG potential, stage spike)

**FLEX Specials:**
- Neutral-B: Fireball (standard projectile)
- Side-B: Spinning kick (approach + combo tool)
- Up-B: Rising slash (versatile recovery)
- Down-B: Reflector (deals with projectiles)

---

## 5. Visual Differentiation

### Color Scheme:
| Character | Primary Color | RGB Value |
|-----------|--------------|-----------|
| RUSH | Orange | (255, 150, 50) |
| ZONE | Purple | (150, 50, 200) |
| GRAB | Green | (50, 200, 100) |
| FLEX | White/Gray | (200, 200, 200) |

### Pose Differentiation:
- RUSH: Aggressive forward lean, fists raised
- ZONE: Upright stance, arms extended
- GRAB: Wide stance, arms out for grappling
- FLEX: Balanced martial arts stance

---

## 6. Implementation Architecture

### Existing Extension Points (from Cycle 0):

1. **FighterStats** (game_engine.py:402-417)
   - Add character preset factory methods
   - Each character gets a named preset

2. **GameLoop._get_attack_data()** (game_engine.py:771-893)
   - Currently returns generic attack data
   - Need to extend with character_type parameter
   - Create attack dictionaries per character

3. **Fighter.__init__** (game_engine.py:419-468)
   - Add character_type attribute
   - Set color based on character type
   - Initialize character-specific attack data

4. **StickFigurePose** (renderer.py:307-328)
   - Already handles poses well
   - May add character-specific idle poses

### New Components Needed:

1. **characters.py** - Character definitions
   - CharacterType enum (RUSH, ZONE, GRAB, FLEX)
   - get_character_stats(CharacterType) -> FighterStats
   - get_character_color(CharacterType) -> RGB tuple
   - get_character_attacks(CharacterType) -> Dict[AttackType, AttackData]

2. **specials.py** - Special move implementations
   - Projectile class for ZONE
   - Counter state for RUSH
   - Armor mechanic for GRAB
   - Reflector for FLEX

---

## 7. Sources Consulted (Cycle 1)

1. [FrayTools Wiki - Character Archetypes](https://fraytools.fandom.com/wiki/Character_Archetypes)
2. [Smashpedia - Character Archetypes](https://supersmashbros.fandom.com/wiki/Character_Archetypes)
3. [PlayStation Compete - Fighting Game Archetypes](https://compete.playstation.com/en-us/all/articles/what-is-a-rushdown-ungabBunga-character-fighting-game-archetypes-fgc-explained)
4. [SmashWiki - Weight](https://www.ssbwiki.com/Weight)
5. [KuroGaneHammer - Run Speed](https://kuroganehammer.com/Ultimate/runspeed)
6. [KuroGaneHammer - Air Speed](https://kuroganehammer.com/Ultimate/airSpeed)
7. [Game8 - Air Speed List](https://game8.co/games/Super-Smash-Bros-Ultimate/archives/316636)
8. [Game8 - Run Speed List](https://game8.co/games/Super-Smash-Bros-Ultimate/archives/316637)

---

# Cycle 2 Research: Projectile System & Defensive Mechanics

## Executive Summary - Cycle 2

Implementing projectiles and defensive mechanics (shields, dodges, counters, reflectors) requires:
1. First-class projectile entities with ownership tracking
2. Shield health system with depletion and regeneration
3. Frame-based dodge invincibility windows
4. Counter window detection with triggered responses
5. Reflector mechanics that change projectile ownership

---

## 1. Projectile System Design

### Key Principles (from Platform Fighter Engine research)

**Projectile Properties:**
- Position, velocity, owner (for damage/collision ownership)
- Hitbox (for collision detection with fighters)
- Lifespan/active frames (projectiles should despawn after time or distance)
- Destroy conditions: hit opponent, hit shield, go off-stage/blastzone

**From Platform Fighter Engine (2025 update):**
- Changed `destroy_outside_room` to `destroy_outside_blastzone` - projectiles should check blast zones, not screen bounds
- Projectiles can use `hitbox_group_whitelist_id` for complex projectile arrays

**Projectile Behavior Patterns:**

| Pattern | Description | Example |
|---------|-------------|---------|
| Linear | Constant velocity, no gravity | ZONE neutral-B |
| Gravity-affected | Falls over time | FLEX fireball |
| Beam | Long hitbox, travels fast | ZONE side-B |
| Chargeable | Damage/speed scales with charge time | ZONE neutral-B (charged) |
| Multi-hit | Hits multiple times | Potentially for traps |

**Key Implementation Decisions:**
1. Projectiles are separate entities from fighters (not just extended hitboxes)
2. Need a projectile list in GameState updated each frame
3. Projectile ownership transfers on reflection (critical for reflector mechanic)
4. Projectiles should have frame-based lifespan, not distance-based

**Sources:**
- [Platform Fighter Engine - Springroll Games](https://springrollgames.itch.io/platform-fighter-engine)
- [Castagne Fighting Game Engine](https://castagneengine.com/)

---

## 2. Shield Mechanics

### Smash Bros Shield Model

**Core Mechanics:**
- Shield has health (100 HP base)
- Shield depletes when hit (proportional to attack damage)
- Shield regenerates when not active
- Shield break = long stun state (vulnerability window)

**Balancing Considerations:**
- "Having weak shields like in Smash Bros. Melee or Ultimate is different from removing the mechanic entirely"
- Shields create rock-paper-scissors: blocks > attacks > throws > blocks
- Without shields, throw becomes less meaningful

**From [OGIUE MANIAX on Platform Fighter Design](https://ogiuemaniax.com/2020/04/12/shields-or-no-shields-platform-fighters-and-the-question-of-defense/):**
- Many indie fighters (Rivals of Aether, Brawlout) removed shields entirely
- Alternative: one-way shields vulnerable to cross-ups (Vortex Rising)
- Alternative: Focus attack style (Slayers for Hire)

**Implementation Approach for Stick Figure Fighter:**
- Shield health: 100 (depletes by attack damage)
- Shield regen: +0.5 per frame when shield is down
- Shield break threshold: 0 HP
- Shield stun duration: 180 frames (3 seconds)
- Shield damage multiplier: 1.0x attack damage to shield

---

## 3. Dodge Mechanics

### Frame Data Reference (from SmashWiki)

**Spot Dodge:**
- Total frames: ~26-30 frames
- Intangibility window: frames 3-18 typically
- Can be canceled into attacks 5 frames early (Ultimate)

**Roll:**
- Total frames: ~30-35 frames
- Intangibility: frames 4-20 typically
- Distance: ~100-150 units

**Air Dodge:**
- Total frames: ~32-50 frames (varied by game)
- Intangibility: frames 4-20 typically
- One per airborne state (reset on landing/hit)
- Ultimate added directional air dodge

**Dodge Staling (Important Balance Mechanic):**
- Consecutive dodges have:
  - More total frames
  - Fewer intangibility frames
  - Later intangibility start
- Resets after ~60 frames of not dodging

**From [SmashWiki Intangibility](https://www.ssbwiki.com/Intangibility):**
- Intangibility = hurtbox not hit at all
- Different from invincibility (which blocks damage but may cause hitlag)
- Use intangibility for dodges (cleaner implementation)

**Implementation Approach:**
1. SPOT_DODGE: 26 frames total, intangible frames 3-18
2. ROLL: 32 frames total, intangible frames 4-20, moves 120 units
3. AIR_DODGE: 35 frames total, intangible frames 4-25, one per airborne

---

## 4. Counter Mechanic (RUSH down-B)

### Counter Design Patterns

**From [DEV Community - Core Fighting Game Mechanics](https://dev.to/goldenxp/core-mechanics-of-fighting-games-24ej):**
- Counter window on startup (5-15 frames typically)
- If hit during window: auto-retaliate with boosted damage
- If not hit: long recovery (punishable)

**Damage Multiplier Patterns:**
- 1.2x counter damage (weak)
- 1.5x counter damage (standard)
- 2.0x counter damage (strong but risky)

**Risk/Reward Balance:**
- Short counter window = harder to use, more reward
- Long counter window = easier but less damage
- Long recovery on whiff = punishable, prevents spam

**Implementation for RUSH down-B:**
- Total frames: 50
- Counter window: frames 5-20 (15 frame window)
- Counter attack (if triggered): damage = opponent_attack.damage * 1.3
- Recovery on whiff: frames 21-50 (30 frames of vulnerability)
- Can counter projectiles (absorbs projectile, deals melee counterattack)

---

## 5. Reflector Mechanic (FLEX down-B)

### Reflector Design (from SmashWiki)

**Core Mechanics:**
- [SmashWiki Reflection](https://www.ssbwiki.com/Reflection): Reflected projectiles change ownership
- Projectiles reverse direction and go back to original owner
- Damage multiplier: typically 1.5x
- Speed multiplier: typically 1.2x-1.8x

**Reflect Limits:**
- Damage threshold: reflectors can only reflect projectiles below certain damage
- If projectile too strong: reflector breaks or projectile ignores it
- Maximum reflects: 7 times before disappearing (Ultimate)

**Fox Reflector Specifics (from [SmashWiki](https://www.ssbwiki.com/Reflector_(Fox))):**
- Frame 1 startup (instant)
- 1.5x damage multiplier
- Also has a close-range hitbox (4% damage)
- Can jump-cancel in Melee (shine combos)

**Implementation for FLEX down-B:**
- Total frames: 35
- Reflect window: frames 5-12 (8 frame window)
- Reflect hitbox: 70x60 around fighter
- Close-range damage hitbox: 4% damage, low knockback
- Reflected projectile: 1.4x damage, 1.5x speed
- Damage threshold: can reflect projectiles up to 20% damage
- Recovery: frames 13-35 (punishable on whiff)

---

## 6. Super Armor (GRAB neutral-B)

### Armor Mechanics

**Super Armor:**
- Fighter takes damage but doesn't flinch
- No knockback during armored frames
- Damage still accumulates

**Heavy Armor vs Super Armor:**
- Heavy armor: ignores hits below threshold (e.g., <10%)
- Super armor: ignores all knockback regardless of damage

**Implementation for GRAB neutral-B (Armored Charge):**
- Already exists in characters.py with long active frames (15-35)
- Add super armor flag during active frames
- Fighter takes damage but velocity/state unchanged
- Armor breaks at high damage (>50% in one hit)

---

## 7. Trap Mechanic (ZONE down-B)

### Trap Design

**Properties:**
- Stationary hitbox placed on stage
- Triggers when opponent enters
- Can have timed lifespan
- Usually only one active at a time

**Implementation for ZONE down-B:**
- Place trap at fighter's position
- Trap hitbox: 50x30
- Damage: 6%
- Lifespan: 300 frames (5 seconds)
- Triggers on contact: launches opponent
- Max traps: 1 (placing new trap removes old)
- Can be destroyed by attacks

---

## 8. State Machine Extensions

### New States Needed

```
FighterState.DODGE     - Already exists, use for spot dodge/roll/air dodge
FighterState.COUNTER   - New: counter stance (if needed, or use SPECIAL)
FighterState.SHIELDING - Already exists
FighterState.STUNNED   - New: shield break stun (or use HITSTUN with flag)
```

### Invincibility Tracking

```python
# Per-fighter tracking needed:
invincible_frames: int      # Already exists
has_air_dodge: bool         # Reset on landing
dodge_stale_counter: int    # For dodge staling (optional)
last_dodge_frame: int       # For stale reset (optional)
```

---

## 9. Priority and Order of Implementation

Based on mission requirements and dependencies:

1. **Projectile System** (foundation for ZONE and FLEX)
   - Projectile class
   - Projectile list in GameState
   - Update loop in GameLoop.step()
   - Rendering for projectiles

2. **Shield Mechanics** (input already mapped)
   - Shield health, depletion, regeneration
   - Shield break stun
   - Shielding state behavior

3. **Dodge Mechanics** (DODGE state exists)
   - Spot dodge (grounded, shield + down)
   - Roll (grounded, shield + left/right)
   - Air dodge (airborne, shield)
   - Invincibility frame tracking

4. **Counter Mechanic** (RUSH down-B)
   - Counter window detection
   - Triggered counterattack
   - Long recovery on whiff

5. **Reflector Mechanic** (FLEX down-B)
   - Detect projectile in reflector zone
   - Reverse projectile ownership/direction
   - Apply damage/speed multipliers

6. **Secondary Goals** (if time permits)
   - Super armor for GRAB neutral-B
   - Trap placement for ZONE down-B

---

## 10. Sources Consulted (Cycle 2)

1. [Platform Fighter Engine - Springroll Games](https://springrollgames.itch.io/platform-fighter-engine)
2. [Castagne Fighting Game Engine](https://castagneengine.com/)
3. [SmashWiki - Intangibility](https://www.ssbwiki.com/Intangibility)
4. [SmashWiki - Spot Dodge](https://www.ssbwiki.com/Spot_dodge)
5. [SmashWiki - Air Dodge](https://www.ssbwiki.com/Air_dodge)
6. [SmashWiki - Roll](https://www.ssbwiki.com/Roll)
7. [SmashWiki - Reflection](https://www.ssbwiki.com/Reflection)
8. [SmashWiki - Reflector (Fox)](https://www.ssbwiki.com/Reflector_(Fox))
9. [OGIUE MANIAX - Shields or No Shields](https://ogiuemaniax.com/2020/04/12/shields-or-no-shields-platform-fighters-and-the-question-of-defense/)
10. [DEV Community - Core Mechanics of Fighting Games](https://dev.to/goldenxp/core-mechanics-of-fighting-games-24ej)
11. [Smashboards - Roll/Air Dodge Frame Data](https://smashboards.com/threads/roll-air-dodge-frame-data.401323/)

---
---

# Cycle 5 Research: Character Balance Tuning

## Executive Summary - Cycle 5

Character balance tuning focuses on:
1. Knockback formula verification for ~120% KO thresholds
2. Frame data audit across all 4 characters
3. Combo prevention (especially RUSH infinite prevention)
4. Projectile cooldown and balance
5. Counter/reflector timing window verification
6. Stock display and training mode improvements

---

## 1. Knockback Formula Analysis

### Smash Bros Ultimate Reference Formula
```
KB = ((((p/10 + p*d/20) * 200/(w+100) * 1.4) + 18) * s + b) * r
```

Where:
- **p** = target damage percentage (after hit)
- **d** = damage dealt by the attack
- **w** = target weight (100 = baseline)
- **s** = knockback scaling / 100
- **b** = base knockback
- **r** = other multipliers (rage, DI, etc.)

### Our Current Implementation
In `game_engine.py:592-605`:
```python
kb_multiplier = 1.0 + (self.damage_percent * CONFIG.knockback_scaling)
weight_mult = 100.0 / self.stats.weight
scaled_kb = knockback * kb_multiplier * weight_mult
```

CONFIG.knockback_scaling = 0.12 (from game_engine.py:44)

**Issue Analysis:**
At 100% damage: kb_multiplier = 1.0 + (100 * 0.12) = 13.0
At 150% damage: kb_multiplier = 1.0 + (150 * 0.12) = 19.0

This seems **too aggressive**. In Smash, the scaling formula is more nuanced.

### Recommendation
Reduce CONFIG.knockback_scaling to 0.06-0.08 for more gradual growth, or implement a more authentic formula.

---

## 2. KO Threshold Targets

### Stage Layout
From game_engine.py, blast zones:
- Left: -200
- Right: 1400
- Top: -200
- Bottom: 920

Stage width (main platform): ~1200 pixels
Distance to horizontal blast zone from center: ~600-700 pixels

### Target KO Percentages
For a smash attack at center stage:
| Character | Weight | Target KO % |
|-----------|--------|-------------|
| RUSH | 85 | 100-110% |
| ZONE | 95 | 110-120% |
| FLEX | 100 | 115-125% |
| GRAB | 120 | 130-145% |

### Current Character Weights
From `characters.py`:
- RUSH: 85 (light)
- ZONE: 95 (medium-light)
- FLEX: 100 (baseline)
- GRAB: 120 (heavy)

---

## 3. Frame Data Audit

### Attack Frame Data Guidelines

| Attack Type | Fast (RUSH) | Balanced (ZONE/FLEX) | Slow (GRAB) |
|-------------|-------------|---------------------|-------------|
| Jab | 2-4f | 4-6f | 6-10f |
| F-Tilt | 5-7f | 7-10f | 10-14f |
| D-Tilt | 4-6f | 6-8f | 8-12f |
| U-Tilt | 5-8f | 8-10f | 10-14f |
| F-Smash | 12-15f | 15-18f | 18-25f |
| D-Smash | 8-12f | 12-15f | 15-20f |
| U-Smash | 10-13f | 13-16f | 16-22f |
| N-Air | 4-6f | 6-8f | 8-12f |

### Current Character Frame Data (from characters.py)

**RUSH:**
- Jab: 3f startup (GOOD)
- F-Tilt: 5f startup (GOOD)
- F-Smash: 12f startup (GOOD)
- D-Smash: 10f startup (GOOD)
- U-Smash: 11f startup (GOOD)

**ZONE:**
- Jab: 5f startup (GOOD)
- F-Tilt: 8f startup (GOOD)
- F-Smash: 16f startup (GOOD)
- D-Smash: 13f startup (GOOD)
- U-Smash: 14f startup (GOOD)

**GRAB:**
- Jab: 8f startup (GOOD)
- F-Tilt: 10f startup (GOOD)
- F-Smash: 18f startup (GOOD)
- D-Smash: 15f startup (GOOD)
- U-Smash: 16f startup (GOOD)

**FLEX:**
- Jab: 4f startup (GOOD)
- F-Tilt: 7f startup (GOOD)
- F-Smash: 14f startup (GOOD)
- D-Smash: 12f startup (GOOD)
- U-Smash: 13f startup (GOOD)

**Assessment: Frame data distribution is appropriate for archetypes.**

---

## 4. Knockback Value Audit

### Current Smash Attack Values

| Character | F-Smash | Damage | Base KB | KB Scaling |
|-----------|---------|--------|---------|------------|
| RUSH | F-Smash | 15% | 6.0 | 0.12 |
| ZONE | F-Smash | 15% | 7.0 | 0.13 |
| GRAB | F-Smash | 21% | 9.0 | 0.16 |
| FLEX | F-Smash | 14% | 6.5 | 0.12 |

### Knockback Calculation Test
With CONFIG.knockback_scaling = 0.12:

At 120% damage, hitting FLEX (weight 100) with RUSH's F-Smash:
```
kb_multiplier = 1.0 + (120 * 0.12) = 15.4
weight_mult = 100 / 100 = 1.0
base_knockback = 6.0
kb_scaling = 0.12
kb_magnitude = 6.0 + (120 * 0.12) = 20.4
scaled_kb = 20.4 * 15.4 * 1.0 = 314.16
```

This seems very high. Need to verify actual KO distances in testing.

### Recommended Adjustments
1. Reduce CONFIG.knockback_scaling from 0.12 to 0.08
2. Increase base knockback slightly to compensate for lower scaling
3. Test KO thresholds at target percentages

---

## 5. Combo Prevention Analysis

### RUSH Jab Chain Risk
Current RUSH jab values:
- Damage: 2.5%
- Base KB: 1.5
- KB Scaling: 0.03

At 0%: kb = 1.5 + (0 * 0.03) = 1.5 (very low)
At 50%: kb = 1.5 + (50 * 0.03) = 3.0 (still low)
At 100%: kb = 1.5 + (100 * 0.03) = 4.5 (should push out)

**Concern:** At low %, jabs may chain indefinitely.

### Recommendations
1. Increase RUSH jab base knockback from 1.5 to 2.0
2. Increase jab knockback scaling from 0.03 to 0.04
3. Target: jabs should stop comboing around 30-40%

### Hitstun Formula Verification
From CONFIG.hitstun_multiplier = 0.4:
- At KB = 5: hitstun = 2 frames
- At KB = 10: hitstun = 4 frames
- At KB = 20: hitstun = 8 frames

This seems reasonable for preventing infinites.

---

## 6. Projectile Balance

### Current Projectile Speeds (need to verify in game_engine.py)

Projectile properties should follow:
- Fast projectile: Low damage, low knockback
- Slow projectile: Higher damage, medium knockback

### Cooldown Recommendations
| Projectile Type | Recommended Cooldown |
|----------------|---------------------|
| Fast spam projectile | 30-45 frames (0.5-0.75s) |
| Power projectile | 60-90 frames (1-1.5s) |
| Trap/utility | 120-180 frames (2-3s) |

### ZONE Neutral-B Analysis
Needs verification of current damage, speed, and cooldown values.

---

## 7. Counter/Reflector Timing

### Current Values (from mission briefing)
- Counter: 15 frames active window
- Reflector: 7 frames active window

### Smash Bros Reference
- Marth Counter: 6-27f active (22 frames)
- Fox Shine: 1f startup, 2f active reflect

### Assessment
- 15f counter is reasonable (shorter than Marth)
- 7f reflector is tight but rewards prediction

### Recommendations
Counter timing seems fine. Could optionally:
- Increase counter to 18f for more accessibility
- Keep reflector at 7f to maintain skill requirement

---

## 8. Stock Display Improvement

### Current Implementation
Simple colored circles for stocks (renderer.py:1594-1627)

### Recommendations
1. Add character initial letter (R, Z, G, F) inside stock circles
2. Use character colors for stock icons
3. Add animation for lost stocks (fade or fall)

### Technical Approach
Modify `draw_hud()` in renderer.py to:
- Draw filled circle with character color
- Draw text initial letter on top
- Track last stock count for animation

---

## 9. Training Mode Features

### Priority Features
1. **Frame Data Overlay** - Show startup/active/recovery during attacks
2. **CPU Opponent** - Basic AI for solo practice
3. **Damage Display** - Already exists in HUD

### Frame Data Display Format
During attack state, show: "S:Xf A:Xf R:Xf"
- S = startup frames
- A = active frames
- R = recovery frames

### Basic CPU AI Design
Minimal viable AI:
1. Random movement bias toward player
2. Attack when in range
3. Shield occasionally
4. Attempt recovery when off-stage

---

## 10. Sources Consulted (Cycle 5)

1. [Ultimate Frame Data](https://ultimateframedata.com/) - Frame data repository
2. [SmashWiki - Knockback](https://www.ssbwiki.com/Knockback) - Knockback formula
3. [Smashboards Frame Data](https://smashboards.com/threads/smash-ultimate-frame-data-patch-13-0-complete.465028/)
4. [SSBU Calculator](https://rubendal.github.io/SSBU-Calculator/) - Knockback calculations
5. [Dragdown Wiki - Damage & Knockback](https://dragdown.wiki/wiki/SSBU/Damage_&_Knockback)

---

## 11. Implementation Priority Summary

### High Priority (Balance Critical)
1. Reduce CONFIG.knockback_scaling for gradual KO growth
2. Tune smash attack knockback for ~120% KO threshold
3. Increase RUSH jab knockback to prevent infinites

### Medium Priority (Quality of Life)
4. Stock display improvement with character initials
5. Frame data overlay for training mode
6. Verify projectile cooldowns

### Lower Priority (Polish)
7. Lost stock animation
8. Basic CPU AI opponent
9. Counter/reflector timing tweaks
