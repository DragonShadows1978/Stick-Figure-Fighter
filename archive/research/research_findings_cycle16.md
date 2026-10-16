# Research Findings: Advanced Training Mode & Final Polish

**Cycle 16 - Mission Continuity Healing**
**Focus Areas:** Input History Display, Hitbox Visualization, Move List Display

---

## Sources Consulted

### Input History Display
- [Dustloop Wiki - Using Training Mode](https://www.dustloop.com/w/Using_Training_Mode)
- [The Gamer - Street Fighter 6 Training Mode Guide](https://www.thegamer.com/street-fighter-6-sfvi-training-mode-tips-tricks-guide/)
- [GameRant - Best Training Modes in Fighting Games](https://gamerant.com/best-training-modes-fighting-games/)

### Hitbox Visualization
- [SF6Frames.com](https://sf6frames.com/) - Frame-by-frame hitbox viewer
- [Frame Trapped](https://frametrapped.com/) - Street Fighter 5 hitbox tools
- [SmashWiki - Hitbox](https://www.ssbwiki.com/Hitbox)
- [Dustloop Wiki - Hitboxes](https://dustloop.com/w/Hitboxes)

### Move List & Frame Data UI
- [Sirlin.net - Frame Advantage Display](https://www.sirlin.net/posts/a-fighting-game-first-showing-frame-advantage)
- [Dustloop Wiki - Using Frame Data](https://www.dustloop.com/w/Using_Frame_Data)
- [EventHubs - How to Read Frame Data](https://www.eventhubs.com/guides/2009/feb/17/how-read-frame-data-street-fighter-4/)

---

## 1. Input History Display Best Practices

### Key Findings

**Essential Features:**
- Show last 10-20 inputs in a scrolling timeline
- Display directional arrows (←↓↑→) and button icons
- Include timing duration for each input (helps identify sloppy execution)
- Use fighting game notation style (⬇️↘️➡️+A for quarter circle forward + attack)
- Position near player character or in a dedicated HUD area

**Implementation Approach:**
```
Standard visual layout:
[Frame] [Direction] [Buttons]
  -1    ← ↓         A B
  -2    ↓           -
  -3    ↓ ←         A
  ...
```

**Timing Information:**
- Show frame count for held inputs (e.g., "← [3f]" means left held for 3 frames)
- Helps players identify if inputs are too slow or sloppy
- Can use color coding: white for tap, yellow for short hold, orange for long hold

### Existing Code Integration Points

The training mode already tracks `RecordedInput` in `training_mode.py`:
```python
@dataclass
class RecordedInput:
    frame: int
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    jump: bool = False
    attack: bool = False
    special: bool = False
    shield: bool = False
    grab: bool = False
```

Need to:
1. Create a separate input history ring buffer (independent from recording)
2. Add visual renderer in `renderer.py`
3. Toggle display with new training mode key (F7)

---

## 2. Hitbox/Hurtbox Visualization Improvements

### Industry Standard Color Coding

**From SF6Frames and Frame Trapped:**
- **Red border, shaded red:** Normal hitbox (attack area)
- **Green border, shaded green:** Normal hurtbox (vulnerable area)
- **Blue/Cyan:** Grab hitbox
- **Yellow/Orange:** Collision point (where hitbox meets hurtbox)
- **Dark green border:** Air invincible hurtbox
- **Blue shaded:** Projectile invincible hurtbox

**Current Implementation (renderer.py:1643-1661):**
```python
# Draw hitboxes if debug mode
if self.show_hitboxes:
    self.set_color('hitbox')  # Red
    for hitbox_aabb in fighter.get_active_hitbox_aabbs():
        self.draw_rect(...)

# Draw hurtbox if debug mode
if self.show_hurtboxes:
    self.set_color('hurtbox')  # Green
    hurtbox = fighter.get_hurtbox_aabb()
    self.draw_rect(...)
```

### Improvements Needed

1. **Collision Point Markers:**
   - Calculate intersection between hitbox and hurtbox AABBs
   - Draw yellow circle/diamond at contact point
   - Show which part of hitbox connected

2. **Semi-Transparent Fill:**
   - Current implementation only draws outlines
   - Add semi-transparent fill for better visibility
   - Can use stipple pattern for pseudo-transparency in X11

3. **Visibility Modes:**
   - Mode 0: None (default gameplay)
   - Mode 1: Hitbox only (red)
   - Mode 2: Hurtbox only (green)
   - Mode 3: Both with collision highlights

4. **Frame-Specific Coloring:**
   - Different shades for active vs startup vs recovery frames
   - Early active frames: bright red
   - Late active frames: darker red
   - Sweetspot hitboxes: highlighted brighter

---

## 3. Move List Display

### Standard Frame Data Format

**From EventHubs and Dustloop:**
```
Move Name        | Startup | Active | Recovery | Total | On Block | On Hit
-----------------+---------+--------+----------+-------+----------+--------
Jab              |    3    |  2-5   |    9     |  14   |    -1    |   +3
Forward Tilt     |    5    |  5-8   |   14     |  22   |    -5    |   +2
Forward Smash    |   12    | 12-15  |   25     |  40   |   -15    |   KD
```

### UI Layout Considerations

**Fighting Game UI Principles (from research):**
- Red draws attention - use for important info
- Minimize clutter during intense gameplay
- Use color coding for frame advantage (green=safe, red=unsafe)
- Show move input notation alongside frame data

**Proposed Layout:**
```
╔═══════════════════════════════════════╗
║ RUSH - Move List                      ║
╠═══════════════════════════════════════╣
║ NORMALS                               ║
║ ├─ Jab (A)          3f/14f    +3      ║
║ ├─ F.Tilt (→+A)     5f/22f    +2      ║
║ ├─ U.Tilt (↑+A)     4f/20f    +4      ║
║ └─ D.Tilt (↓+A)     4f/18f    +1      ║
║                                       ║
║ SMASH ATTACKS                         ║
║ ├─ F.Smash (→→+A)   12f/40f   -15     ║
║ ├─ U.Smash (↑↑+A)   10f/38f   -12     ║
║ └─ D.Smash (↓↓+A)   8f/35f    -10     ║
║                                       ║
║ SPECIALS                              ║
║ ├─ N.Special (B)    8f/45f    Multi   ║
║ ├─ S.Special (→+B)  6f/35f    -8      ║
║ ├─ U.Special (↑+B)  5f/48f    -25     ║
║ └─ D.Special (↓+B)  5f/55f    Counter ║
╚═══════════════════════════════════════╝
```

### Data Source

All attack data is available from `characters.py`:
```python
def get_character_attacks(character: CharacterType) -> Dict[AttackType, AttackData]
```

Each `AttackData` contains:
- `total_frames`
- `startup_frames`
- `recovery_frames`
- `hitboxes[].active_frames` (tuple of start, end frame)
- `hitboxes[].damage`
- `hitboxes[].knockback_base`

---

## 4. Additional Considerations

### Performance

- Input history updates every frame - use ring buffer for O(1) operations
- Hitbox visualization adds draw calls - use visibility toggle
- Move list is static - render once and cache until character changes

### Integration with Existing Training Mode

Current training mode keys (from main.py):
- F1: Toggle CPU
- F2: Cycle CPU behavior
- F3: Toggle frame data overlay
- F4: Cycle CPU difficulty
- F5: Toggle infinite jumps
- F6: Toggle frame advantage display
- R: Reset damage
- T: Reset position
- Tab: Toggle hitboxes

**Proposed new keys:**
- F7: Toggle input history display
- F8: Cycle hitbox visibility mode (none/hit/hurt/both)
- F9: Toggle move list display

---

## 5. Character-Specific Considerations

### RUSH (Fast Rushdown)
- Fastest jab: 3f startup
- Multi-hit neutral special
- Counter down special (15-frame window)

### ZONE (Ranged Zoner)
- Projectile neutral special
- Beam side special (long range)
- Teleport up special (no hitbox)

### GRAB (Slow Grappler)
- Slowest startup but highest damage
- Command grab side special (20 damage)
- Super armor neutral special

### FLEX (All-Rounder)
- Balanced frame data
- Reflector down special (7-frame window)
- Standard fireball

Each character's move list should highlight their archetype-specific strengths.

---

## 6. Implementation Priority

1. **Input History Display** - Most impactful for learning execution
2. **Hitbox Visualization Improvements** - Collision markers help understanding
3. **Move List Display** - Reference for learning character moves

All three features enhance the training mode experience and help players understand the game's mechanics better.
