# Research Findings: Visual Effects and Game Feel for Stick Figure Fighter

## Executive Summary

This research focused on best practices for 2D fighting game visual effects, specifically:
1. Hit sparks and impact effects
2. Screen shake implementation
3. Hitstop/freeze frames
4. Shield/dodge/counter visualizations
5. Projectile trails
6. UI enhancements

## Key Techniques from Industry Research

### 1. Hitstop (Freeze Frames)
**Source:** [Hitstop Analysis - CritPoints](https://critpoints.net/2017/05/17/hitstophitfreezehitlaghitpausehitshit/)

**Implementation Guidelines:**
- Light attacks: ~9 frames freeze
- Medium attacks: ~11 frames freeze
- Heavy attacks: ~13 frames freeze
- Both attacker AND defender freeze simultaneously
- Characters should vibrate slightly during hitstop:
  - Grounded: horizontal vibration
  - Airborne: vertical vibration
- Projectiles should have less hitstop than melee attacks

**Why it matters:** Hitstop gives players time to:
- Visually confirm the hit occurred
- Input cancel commands for combos
- Perceive the weight/power of the attack

### 2. Screen Shake
**Source:** [Combat Impact in Action Games - GameDeveloper](https://www.gamedeveloper.com/audio/improving-the-combat-impact-of-action-games)

**Implementation Guidelines:**
- Duration: 0.1-0.3 seconds (6-18 frames at 60fps)
- Randomize direction slightly
- Use easing/decay functions for smooth tapering
- Scale magnitude to knockback force
- Add directional bias matching knockback angle
- Cap maximum shake to prevent disorientation

**Current Status:** Already implemented in `renderer.py:ScreenShake` class

### 3. Hit Sparks / Particle Effects
**Source:** [Battle VFX - pimen](https://pimen.itch.io/battle-vfx-hit-spark)

**Implementation Guidelines:**
- Radiating lines from impact point
- Color progression: white -> yellow -> orange (fade)
- 6-10 frames duration
- Scale size to attack power
- Central bright flash on first 2-3 frames
- Direction aligned with attack angle

**Current Status:** Partially implemented in `renderer.py:draw_hit_spark()`

### 4. Damage Flash
**Source:** [Improving Combat Impact](https://www.gamedeveloper.com/audio/improving-the-combat-impact-of-action-games)

**Implementation Guidelines:**
- Flash fighter white for 3-5 frames when hit
- Alternative: tint toward the attacker's color briefly
- Can also flash the screen/background

**Current Status:** Implemented via `fighter._damage_flash_frames` attribute

### 5. Super Armor Visualization
**Source:** [Smash Wiki - Hitlag](https://www.ssbwiki.com/Hitlag)

**Implementation Guidelines:**
- Golden/yellow glow around character during super armor
- Sparks fly OFF the character when hit during super armor
- Character doesn't flinch visually
- Audio feedback helps sell the effect

**Current Status:** Golden outline implemented but no spark effect on hit

### 6. Dodge/Roll Visualization

**Implementation Guidelines:**
- Afterimage/ghost trail showing previous positions
- Slight transparency reduction for each ghost
- Keep 2-4 afterimages, update every 2-3 frames
- Roll animation should show tumbling pose
- Air dodge could add blur/trail effect

**Current Status:** Afterimage system implemented in `renderer.py:draw_dodge_afterimages()`

### 7. Shield Visualization

**Implementation Guidelines:**
- Semi-transparent bubble around fighter
- Size scales with shield health (shrink as depleted)
- Color gradient: green (healthy) -> yellow -> red (low)
- Crack effects when shield is low
- Flash/pulse on impact
- Shield break should have distinctive animation

**Current Status:** Implemented with color gradient and crack effects

### 8. Counter/Reflector Effects

**Implementation Guidelines:**
- Counter: defensive aura/stance, pulse on trigger, burst on success
- Reflector: hexagonal/geometric barrier, spinning element, flash on reflect

**Current Status:** Both implemented in renderer.py

## Gap Analysis: What's Missing

Based on mission requirements vs current implementation:

### Already Implemented (Cycle 3 additions visible in code):
1. Projectile rendering with trails
2. Hit spark effects (radiating lines)
3. Screen shake with directional bias
4. Damage flash (white flash on hit)
5. Shield visualization with size/color/cracks
6. Dodge afterimages
7. Counter aura
8. Reflector hexagonal effect
9. Super armor golden outline
10. Shield health bar in HUD
11. Character names in HUD
12. Damage % color gradient

### NOT Yet Implemented (from mission):
1. **Hitstop/Freeze Frames** - Critical missing feature
2. **Stock icons** - Currently just circles, should be character-specific
3. **Landing dust particles**
4. **Jump squash/stretch animation**
5. **Knockback trails** (player leaves trail when launched)
6. **Victory/KO screen**
7. **Super armor hit sparks** (sparks when hit during armor)
8. **Shield break explosion effect**

### Priority Assessment:
1. **Hitstop** - HIGH - This is THE most impactful game feel feature for fighting games
2. **Victory/KO screen** - MEDIUM - Important for game completeness
3. **Landing dust** - LOW - Nice polish
4. **Stock icons** - LOW - Visual nicety
5. **Knockback trails** - LOW - Nice polish

## Hitstop Implementation Recommendation

Add to `GameLoop`:
```python
self.hitstop_frames = 0  # Frames remaining in hitstop

def step(self):
    if self.hitstop_frames > 0:
        self.hitstop_frames -= 1
        # Only update screen shake and effects during hitstop
        return True  # Still running, but frozen

    # Normal update logic...
```

Calculate hitstop based on attack damage:
- 3 + (damage / 3) frames, capped at 12
- Attacker and defender both freeze
- Visual vibration during freeze

## Sources Consulted

1. [Improving Combat Impact in Action Games](https://www.gamedeveloper.com/audio/improving-the-combat-impact-of-action-games)
2. [Hitstop Analysis - CritPoints](https://critpoints.net/2017/05/17/hitstophitfreezehitlaghitpausehitshit/)
3. [Smash Wiki - Hitlag](https://www.ssbwiki.com/Hitlag)
4. [Super Game Feel Effects](https://kaiclavier.itch.io/super-game-feel-effects)
5. [Making Games Feel Juicy](https://medium.com/@yemidigitalcash/when-you-play-a-great-game-it-feels-good-d23761b6eccf)
6. [Exaggerated Animation in Fighting Games](https://pixune.com/blog/role-of-exaggerated-animation-in-fighting-games/)
