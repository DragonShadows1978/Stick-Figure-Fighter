# Research Findings: Stick Figure Fighter Audio System and Polish

## Cycle 4 Mission Overview
Continue building Stick Figure Fighter with focus on:
1. Audio system using procedural sound generation
2. Game feel improvements (animation polish)
3. Character balance tuning
4. UI/UX polish

---

## Audio System Research

### Best Libraries for Python Game Audio

#### 1. sounddevice (Recommended for synthesis)
- **Source**: [python-sounddevice GitHub](https://github.com/spatialaudio/python-sounddevice)
- Provides bindings for PortAudio library via CFFI
- Works with NumPy arrays directly for audio data
- Cross-platform (Linux, macOS, Windows)
- Low-latency playback with `sd.play(data, sample_rate)`
- Non-blocking playback that continues in background
- Standard sample rate: 44100 Hz

#### 2. simpleaudio (Best for low-latency playback)
- **Source**: [simpleaudio documentation](https://simpleaudio.readthedocs.io/)
- Optimized specifically for low-latency, real-time playback
- Great for games and interactive applications
- Supports 8-, 16-, 24-bit integer and 32-bit float
- Sample rates: 8, 11.025, 16, 22.05, 32, 44.1, 48, 88.2, 96, 192 kHz
- Accepts numpy arrays

#### 3. pyfxr (Sound effect generation)
- **Source**: [pyfxr GitHub](https://github.com/lordmauve/pyfxr)
- Specifically designed for game sound effects
- Can generate: explosions, hit sounds, jump sounds, powerups
- Fast Cython implementation
- Compatible with Pygame, Pyglet, and sounddevice
- Generates random sounds in categories

### Procedural Sound Synthesis Techniques

#### Synthesis Methods for Game Sounds:

1. **White Noise + Envelope** - Best for:
   - Hit/impact sounds
   - Explosion sounds
   - Percussive effects

2. **Additive Synthesis** - Best for:
   - Tones and beeps
   - Shield ping sounds
   - Musical elements

3. **FM Synthesis** - Best for:
   - Bell-like sounds
   - Complex textures
   - Sci-fi effects

4. **Envelope Shaping (ADSR)** - Essential for:
   - Attack: Speed of sound onset
   - Decay: Fall to sustain level
   - Sustain: Held level
   - Release: Fade out time

### Sound Effect Design Patterns

#### Hit Sounds (varying intensity)
- Base: Short burst of filtered noise
- Intensity scaling: Higher knockback = lower frequency content + longer decay
- Add sine wave "punch" at low frequency (50-100 Hz)

#### Jump Sound
- Quick upward frequency sweep (sine wave)
- Short duration (50-100ms)
- Start ~200 Hz, end ~400 Hz

#### Landing Sound
- Short thump: Low frequency sine burst + noise
- Duration ~50ms
- Frequency centered around 80-150 Hz

#### Shield Hit (Ping/Deflect)
- High frequency sine wave (800-1200 Hz)
- Very short attack, medium decay
- Add harmonics for "metallic" quality

#### KO Sound
- Combination: Impact (low noise burst) + "explosion" (descending noise sweep)
- Longer duration (200-400ms)
- Multiple layered frequencies

### Audio Manager Design

Key requirements:
- **Concurrent sound limiting**: Don't play too many simultaneous sounds
- **Priority system**: Important sounds (KO) should not be cut
- **Volume mixing**: Balance multiple sounds
- **Caching**: Pre-generate common sounds for performance

---

## Game Feel Improvements Research

### Animation Polish Techniques

1. **Squash and Stretch**
   - Compress character on crouch/landing
   - Stretch character during jump rise
   - Creates sense of weight and elasticity

2. **Anticipation Frames**
   - Brief wind-up pose before attacks
   - Telegraphs action to player
   - 2-3 frames typical for fast attacks

3. **Landing Recovery**
   - Character briefly compresses
   - Small pause before full mobility returns
   - Dust particles reinforce impact

4. **Idle Animation (Breathing)**
   - Subtle up/down movement of body
   - Head bob variation
   - Prevents "dead" static pose

### Implementation Approach
- Modify `StickFigurePose` in renderer.py
- Add interpolation between poses
- Use sinusoidal functions for breathing
- Track velocity for squash/stretch factor

---

## Character Balance Considerations

### Frame Data Analysis
Current character stats from characters.py:
- RUSH: 3f jab startup, fast but light
- ZONE: Projectile-focused, slower normals
- GRAB: Slow (8f jab) but high damage
- FLEX: Balanced 5f jab

### Balance Tuning Targets
1. Knockback thresholds for KO at ~150% damage
2. Projectile speeds relative to run speeds
3. Counter/reflector timing windows (currently 15f for counter)
4. Recovery move distances vs stage size

---

## UI Polish Features

### Priority Items
1. **Combo Counter**: Track consecutive hits, display count
2. **Stock Display**: Character-colored circles exist, could add icons
3. **Stage Timer**: Optional time limit mode
4. **Pause Menu**: Currently pause key exists but no menu

---

## Implementation Decisions

### Audio System Architecture
**Choice: sounddevice + numpy for synthesis**
- Rationale: Most flexible for procedural generation
- Can generate sounds in real-time with numpy
- Non-blocking playback suitable for games
- No external audio files needed

### Sound Generation Approach
- Pre-generate common sounds at startup (cache)
- Parameterize by intensity (knockback value)
- Use thread-safe playback queue
- Limit concurrent sounds (max 8-10)

### Integration Points
- GameLoop callbacks already exist: on_hit, on_land, on_reflect
- Add: on_jump, on_shield_hit, on_ko
- Audio manager as singleton, initialized in main.py

---

## Sources

- [python-sounddevice GitHub](https://github.com/spatialaudio/python-sounddevice)
- [simpleaudio documentation](https://simpleaudio.readthedocs.io/)
- [pyfxr documentation](https://pyfxr.readthedocs.io/en/stable/generating.html)
- [Sound Synthesis in Python - Medium](https://medium.com/@noahhradek/sound-synthesis-in-python-4e60614010da)
- [Procedural Audio Generation Techniques](https://peerdh.com/blogs/programming-insights/procedural-audio-generation-techniques-for-adaptive-soundscapes-in-games)
- [Pyo DSP Library](https://dev.to/hexshift/how-to-generate-sound-effects-programmatically-using-python-and-pyo-2eid)
