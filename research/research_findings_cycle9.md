# Cycle 9 Research Findings: Visual Polish and Dodge Mechanics Verification

## Research Date: 2025-12-26
## Mission: Dodge Mechanics Verification + Visual Polish

---

## 1. Dodge Mechanics Analysis (Mission Critical - Drift Correction)

### 1.1 Current Implementation Status

Based on code review of `game_engine.py`:

**Dodge Types Implemented:**
- `DodgeType.SPOT_DODGE` - 20 frames total, 15 frames invincibility
- `DodgeType.ROLL_FORWARD` / `ROLL_BACKWARD` - 30 frames total, 20 frames invincibility, 0.8x run_speed movement
- `DodgeType.AIR_DODGE` - 25 frames total, 18 frames invincibility, one per airborne state

**Key Mechanics Working:**
- `fighter.air_dodge_available` flag exists (line 529)
- Flag resets when `fighter.grounded` (lines 922-923)
- Dodge state tracked via `fighter.dodge_frames_remaining` (line 531)
- `_start_dodge()` method properly sets invincibility (lines 1190-1206)
- `_update_dodge()` handles roll movement and air dodge DI (lines 1115-1134)

**Input Triggers:**
- Ground roll: Shield pressed + left/right direction
- Spot dodge: Shield pressed + down while shielding
- Air dodge: Shield pressed while airborne (if air_dodge_available)

### 1.2 Visual Feedback Analysis

From `renderer.py`:

**Existing Dodge Effects:**
- `POSE_DODGING` defined (lines 535-547) - unique stance for dodge
- `draw_dodge_afterimages()` method exists (lines 1087-1126)
- Afterimage system stores up to 3 ghost positions
- Afterimages use alpha-based color darkening (X11 limitation)

**Gap Identified:**
- Afterimages only drawn every 3 frames (line 1100)
- No specific visualization distinguishing dodge types
- No sound effect triggered during dodge (audio callback exists but not connected)

### 1.3 Sound Effects Status

From `audio_system.py`:

- `SoundType.DODGE` exists (line 61)
- Sound generated as "woosh" (lines 343-348)
- `on_dodge()` callback exists (lines 752-754)
- **Gap:** `on_dodge()` is not called from game loop when dodge starts

---

## 2. Fighting Game Visual Feedback Best Practices (2024-2025)

### 2.1 Invincibility Frame Communication

Sources: [CritPoints](https://critpoints.net/2017/07/25/how-iframes-augment-dodge-rolls/), [GameDeveloper](https://www.gamedeveloper.com/design/the-esoteric-design-of-fighting-games)

**Key Insights:**
- "No traditional fighting game has ever consistently conveyed invincibility" - major UX problem
- Players must guess "Is that invincible?"
- **Recommendation:** Use visual symbols, color changes, or effects to clearly show invincibility state

**Implementation for Stick Fighter:**
- Flash/glow during invincible frames
- Distinct color tint (e.g., blue/white ghost effect)
- Afterimage trail length indicates remaining invincibility

### 2.2 Sound Effect Timing

Source: [Dan Fornace's 10 Tips](https://fornace.medium.com/dan-fornaces-10-tips-for-making-a-fighting-game-e2c982da2396)

**Key Insight:**
- "Sound effects should play as early as possible and be designed to give the player the tell as soon as they can"
- Don't build up slowly - start big, then bigger

**Implementation:**
- Dodge sound plays immediately on dodge start
- Roll sounds could have directional panning

### 2.3 Hit Effects

Source: [Combat Design Skills](https://gamedesignskills.com/game-design/combat-design/)

**Key Insight:**
- "Hit effects shouldn't scale up slowly - start big and then explode even bigger"
- "As soon as the hit happens, the game state should be noticeably different"

**Current Implementation:** Already has hit sparks, screen shake, damage flash. Good.

---

## 3. Victory Screen Design Patterns

Sources: [SmashWiki Victory Pose](https://www.ssbwiki.com/Victory_pose), [Smashpedia](https://supersmashbros.fandom.com/wiki/Victory_Pose)

### 3.1 Standard Components

1. **Winner Spotlight** - Focus on winning character
2. **Victory Pose** - Character-specific animation (arms up, jump, special pose)
3. **Loser in Background** - Clapping/applauding animation (or defeat pose)
4. **Confetti/Particles** - Celebration effect
5. **Stats Display** - Damage dealt, combos, time

### 3.2 Animation Timing

- Victory screen appears ~1-2 seconds after final KO
- Slow-motion on final hit before transition
- Character holds pose for 3-5 seconds
- Stats fade in after 1 second

---

## 4. Character Select Polish Patterns

### 4.1 Industry Standards

- Large character preview (50-70% of screen space)
- Idle animation with breathing/movement
- "Lock in" triggers attack pose or special animation
- Cursor has animation (bob, glow, trail)
- Character stats displayed as bars or icons

### 4.2 Current State in Stick Fighter

From `menu_system.py`:
- Character slots are 200x280px (relatively small)
- Static silhouette rendering
- Stats bars implemented (Speed, Weight, Power, Recovery)
- Basic cursor with bob animation

**Improvements Needed:**
- Larger character preview
- Breathing/idle animation on preview
- Attack pose on lock-in

---

## 5. Stage Select Preview Patterns

### 5.1 Common Features

- Stage preview shows platform layout
- Animated background (scrolling clouds, moving elements)
- Stage name banner with stylized text
- Mini-map or schematic view

### 5.2 Current State

From `menu_system.py`:
- Basic platform layout preview
- Static background
- Simple stage name text

**Improvements Needed:**
- Animated background scrolling
- Better platform scaling in preview
- Stage name banner animation

---

## 6. Implementation Recommendations

### Priority 1: Dodge Mechanics Verification (Mission Critical)

1. **Connect Dodge Sound:**
   - Add `on_dodge` callback to GameLoop
   - Call it from `_start_dodge()` method
   - Play `SoundType.DODGE` on dodge start

2. **Verify Air Dodge Reset:**
   - Test: Jump, air dodge, land, jump, air dodge should work
   - Current code at line 922-923 handles this

3. **Test All Dodge Types:**
   - Spot dodge: Shield + down while shielding
   - Roll: Shield + direction
   - Air dodge: Shield in air

### Priority 2: Dodge Visual Enhancement

1. **Enhanced Afterimage Effect:**
   - Add slight glow/flash during invincibility frames
   - Different afterimage color for different dodge types
   - More frequent afterimage updates (every 2 frames)

2. **Type-Specific Visualization:**
   - Spot dodge: In-place blur/ghost
   - Roll: Trail with directional blur
   - Air dodge: Radial blur effect

### Priority 3: Character Select Polish

1. **Larger Preview Silhouettes:**
   - Increase character preview size by 30-50%
   - Center more prominently in slot

2. **Idle Animation:**
   - Add breathing animation to preview (using existing breath cycle)
   - Head bob and slight body sway

3. **Lock-in Animation:**
   - Switch to attack pose on selection confirm
   - Brief flash effect

### Priority 4: Results Screen Victory Animation

1. **Victory Pose:**
   - Create POSE_VICTORY (arms raised, slight jump)
   - Winner character uses this pose

2. **Confetti Particles:**
   - Simple falling rectangles (multiple colors)
   - Random spawn across top of screen
   - Gravity-affected fall with slight x-drift

3. **Spotlight Effect:**
   - Radial gradient/glow behind winner
   - Dim background around spotlight

4. **Defeat Pose:**
   - Create POSE_DEFEAT (slumped, arms down)
   - Loser character uses this pose

---

## 7. Technical Constraints

### X11 Limitations

- No true alpha transparency
- Color mixing via pixel blending not supported
- Workaround: Use color darkening/lightening for effects

### Performance Considerations

- Keep particle count under 50 for confetti
- Afterimage storage already uses efficient list pruning
- Effect updates happen in render loop (not blocking)

---

## 8. Success Criteria for Cycle 9

1. **Dodge Mechanics:**
   - [ ] All 4 dodge types trigger correctly (spot, roll fwd, roll back, air)
   - [ ] Air dodge resets on landing
   - [ ] Dodge sound plays on dodge start
   - [ ] Afterimage effect visible during dodge

2. **Character Select:**
   - [ ] Larger character silhouettes
   - [ ] Breathing animation on preview
   - [ ] Attack pose on lock-in

3. **Stage Select:**
   - [ ] Animated background scrolling (optional)
   - [ ] Stage name banner

4. **Results Screen:**
   - [ ] Victory pose for winner
   - [ ] Confetti particle effect
   - [ ] Defeat pose for loser

---

## Sources Consulted

1. [CritPoints - How iFrames Augment Dodge Rolls](https://critpoints.net/2017/07/25/how-iframes-augment-dodge-rolls/)
2. [GameDeveloper - The Esoteric Design of Fighting Games](https://www.gamedeveloper.com/design/the-esoteric-design-of-fighting-games)
3. [Dan Fornace's 10 Tips for Making a Fighting Game](https://fornace.medium.com/dan-fornaces-10-tips-for-making-a-fighting-game-e2c982da2396)
4. [SmashWiki - Victory Pose](https://www.ssbwiki.com/Victory_pose)
5. [Combat Design Skills](https://gamedesignskills.com/game-design/combat-design/)
6. [Smashpedia - Victory Pose](https://supersmashbros.fandom.com/wiki/Victory_Pose)
7. Internal code review: game_engine.py, renderer.py, menu_system.py, audio_system.py
