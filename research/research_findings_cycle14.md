# Cycle 14 Research: Visual Polish & Game Feel

## Research Conducted

### 1. Hit Stop / Hit Freeze Implementation

**Sources Consulted:**
- [SmashWiki - Hitlag](https://www.ssbwiki.com/Hitlag)
- [CritPoints - Hitstop Analysis](https://critpoints.net/2017/05/17/hitstophitfreezehitlaghitpausehitshit/)
- [Sakurai's Famitsu Column on Hitstop](https://sourcegaming.info/2015/11/11/thoughts-on-hitstop-sakurais-famitsu-column-vol-490-1/)
- [Gamasutra - Improving Combat Impact](https://www.gamedeveloper.com/audio/improving-the-combat-impact-of-action-games)

**Key Findings:**
1. **Purpose of Hit Stop:**
   - Gives weight to moves (more powerful = longer hitstop)
   - Allows hit confirmation for offense
   - Gives defender time to react
   - Helps players register that collision happened

2. **Implementation Best Practices:**
   - Freeze character animations for 2-4 frames on significant hits
   - Damage-based scaling: `hitstop_frames = base + (damage * 0.3)`
   - Both attacker AND defender freeze (unlike other fighters, Smash freezes both)
   - Characters should vibrate slightly during freeze for impact feel
   - Keep particles and effects running during freeze
   - Cap maximum hitstop at ~12-15 frames
   - Projectiles get less hitstop than melee attacks
   - Electric attacks can get extended hitstop

3. **Visual During Freeze:**
   - Slight character vibration (2-3 pixel offset oscillation)
   - Keep hurtboxes static even if visual vibrates
   - Screen shake continues during freeze

### 2. Parallax Scrolling Implementation

**Sources Consulted:**
- [Envato Tuts+ - Parallax Scrolling](https://code.tutsplus.com/parallax-scrolling-a-simple-effective-way-to-add-depth-to-a-2d-game--cms-21510t)
- [Wikipedia - Parallax Scrolling](https://en.wikipedia.org/wiki/Parallax_scrolling)
- [LinkedIn - Creating Effective Parallax](https://www.linkedin.com/advice/1/how-can-you-create-effective-parallax-scrolling-ftyne)

**Key Findings:**
1. **Layer Organization:**
   - Typically 3-6 layers
   - Layer 1: Far background (sky, horizon) - slowest
   - Layer 2: Mid-far (mountains, distant structures)
   - Layer 3: Mid (trees, buildings)
   - Layer 4: Near background (just behind action)
   - Layer 5: Game layer (fighters, platforms) - 1:1 movement
   - Layer 6: Foreground overlay (optional) - faster than camera

2. **Speed Calculation:**
   - Foreground moves 1:1 with camera
   - Each layer back multiplies speed by 0.6-0.8
   - Example: Layer speeds [0.1, 0.3, 0.5, 0.7, 1.0]
   - Can use z-depth formula: `speed_multiplier = 1.0 / (1.0 + distance)`

3. **Implementation Pattern:**
   ```
   for each layer:
       layer_offset_x = camera_x * layer.scroll_speed
       draw_layer(layer, -layer_offset_x)
   ```

### 3. Character Entrance Animations

**Sources Consulted:**
- [Kevurugames - How to Animate a Fighting Game](https://kevurugames.com/blog/how-to-animate-a-fighting-game/)
- [Melior Games - Animate Fighting Game](https://meliorgames.com/game-art-design/animate-a-fighting-game-in-6-stages/)
- [Retro Style Games - Fighting Game Animations](https://retrostylegames.com/blog/how-to-make-animations-in-fighting-game/)

**Key Findings:**
1. **Animation Principles:**
   - Squash and stretch for character movement
   - Anticipation before action
   - Follow-through after action
   - Timing based on physics (gravity, momentum)

2. **Entrance Animation Types (per archetype):**
   - Speed characters (RUSH): Fast dash-in or lightning drop
   - Zoners (ZONE): Teleport/fade-in effect
   - Grapplers (GRAB): Heavy landing with screen shake
   - Balanced (FLEX): Standard jump-down with pose

3. **Implementation:**
   - Add ENTERING state to Fighter state machine
   - Characters start above screen or off-screen
   - 30-60 frame entrance animation
   - Transitions to countdown when both fighters land

### 4. Screen Shake Enhancement

**Current Implementation:**
- ScreenShake class exists in renderer.py
- Already applies shake on hits via `add_hit_effect()`
- Random offset with decay

**Enhancements Needed:**
- Verify knockback threshold (should trigger on knockback > 15)
- Add directional bias from hit angle
- Ensure shake integrates with camera offset properly

## Design Decisions

### Priority Order (Impact vs. Complexity)
1. **Hit Stop/Freeze** - HIGH IMPACT, LOW COMPLEXITY
   - Most impactful for game feel
   - Requires changes in game_engine.py (GameState.hit_freeze_frames)
   - Requires changes in main.py (skip physics during freeze)

2. **Screen Shake Enhancement** - MEDIUM IMPACT, LOW COMPLEXITY
   - Already partially implemented
   - Just needs threshold verification and directional bias

3. **Parallax Backgrounds** - MEDIUM IMPACT, MEDIUM COMPLEXITY
   - Visual polish
   - Requires Stage class changes and renderer additions
   - 3 layers: far sky, mid clouds/mountains, near ground detail

4. **Character Entrances** - MEDIUM IMPACT, MEDIUM COMPLEXITY
   - Adds pre-match excitement
   - Requires new Fighter state and animation poses
   - Integration with countdown flow

5. **Knockback Particle Trails** - LOW IMPACT, LOW COMPLEXITY
   - Already have trail system
   - Just needs threshold trigger and particle spawn
