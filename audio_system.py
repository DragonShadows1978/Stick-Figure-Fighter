"""
Stick Figure Fighter - Audio System

Procedural sound effects generation and audio management.
All sounds are generated programmatically using numpy - no external audio files needed.

Cycle 4: Audio System Implementation
- Hit sounds with intensity scaling
- Jump/landing sounds
- Shield/dodge sounds
- KO/respawn sounds
- Audio manager with concurrent sound limiting
"""
import numpy as np
import threading
import queue
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum, auto

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[AudioSystem] sounddevice not available - audio disabled")


# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================

SAMPLE_RATE = 44100  # Standard CD quality
MAX_CONCURRENT_SOUNDS = 8  # Limit concurrent sounds to prevent clipping
MASTER_VOLUME = 0.5  # Global volume (0.0 - 1.0)


# =============================================================================
# SOUND TYPES
# =============================================================================

class SoundType(Enum):
    """Types of sounds in the game."""
    # Combat
    HIT_LIGHT = auto()      # Light hit (jab, weak attacks)
    HIT_MEDIUM = auto()     # Medium hit (tilts, aerials)
    HIT_STRONG = auto()     # Strong hit (smashes, high knockback)
    HIT_CRITICAL = auto()   # Critical hit (very high knockback)

    # Movement
    JUMP = auto()           # Jump start
    DOUBLE_JUMP = auto()    # Double jump (slightly different)
    LAND = auto()           # Landing on ground
    LAND_HARD = auto()      # Hard landing (after high fall)

    # Defense
    SHIELD_UP = auto()      # Shield activated
    SHIELD_HIT = auto()     # Attack blocked by shield
    SHIELD_BREAK = auto()   # Shield broken
    DODGE = auto()          # Dodge/roll

    # Special
    COUNTER = auto()        # Counter stance
    COUNTER_SUCCESS = auto() # Counter triggered
    REFLECT = auto()        # Projectile reflected
    PROJECTILE_FIRE = auto() # Projectile launched
    PROJECTILE_HIT = auto()  # Projectile hit

    # Game events
    KO = auto()             # Knock out
    RESPAWN = auto()        # Respawn
    MATCH_START = auto()    # Match begins
    MATCH_END = auto()      # Match over

    # Menu sounds
    MENU_MOVE = auto()      # Cursor movement
    MENU_SELECT = auto()    # Selection confirmed
    MENU_BACK = auto()      # Back/cancel

    # Countdown sounds
    COUNTDOWN_3 = auto()    # Countdown "3"
    COUNTDOWN_2 = auto()    # Countdown "2"
    COUNTDOWN_1 = auto()    # Countdown "1"
    COUNTDOWN_GO = auto()   # "GO!"

    # Cycle 13: Crowd reactions
    CROWD_GASP = auto()     # Crowd gasp on big hit
    CROWD_CHEER = auto()    # Crowd cheer on KO or big play

    # Cycle 15: Stage ambient sounds
    AMBIENT_BATTLEFIELD = auto()       # Wind/nature ambience for Battlefield
    AMBIENT_FINAL_DESTINATION = auto() # Cosmic drone for Final Destination

    # Cycle 18: Character select sounds
    CHAR_PREVIEW_RUSH = auto()    # Speed whoosh for RUSH
    CHAR_PREVIEW_ZONE = auto()    # Energy zap for ZONE
    CHAR_PREVIEW_GRAB = auto()    # Heavy thud for GRAB
    CHAR_PREVIEW_FLEX = auto()    # Shield chime for FLEX
    CHAR_LOCK = auto()            # Confirmation sound when locked in


# =============================================================================
# SOUND GENERATOR
# =============================================================================

class SoundGenerator:
    """Generates procedural sound effects using waveforms."""

    @staticmethod
    def generate_sine(duration: float, frequency: float,
                      amplitude: float = 1.0, fade_out: bool = True) -> np.ndarray:
        """Generate a sine wave."""
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        wave = amplitude * np.sin(2 * np.pi * frequency * t)

        if fade_out:
            # Apply exponential fade out
            envelope = np.exp(-t * 5 / duration)
            wave *= envelope

        return wave.astype(np.float32)

    @staticmethod
    def generate_square(duration: float, frequency: float,
                        amplitude: float = 1.0, fade_out: bool = True) -> np.ndarray:
        """Generate a square wave (more harsh/digital sounding)."""
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))

        if fade_out:
            envelope = np.exp(-t * 5 / duration)
            wave *= envelope

        return wave.astype(np.float32)

    @staticmethod
    def generate_noise(duration: float, amplitude: float = 1.0,
                       fade_out: bool = True) -> np.ndarray:
        """Generate white noise."""
        samples = int(SAMPLE_RATE * duration)
        wave = amplitude * (np.random.random(samples) * 2 - 1)

        if fade_out:
            t = np.linspace(0, duration, samples, False)
            envelope = np.exp(-t * 5 / duration)
            wave *= envelope

        return wave.astype(np.float32)

    @staticmethod
    def generate_sweep(duration: float, freq_start: float, freq_end: float,
                       amplitude: float = 1.0, fade_out: bool = True) -> np.ndarray:
        """Generate a frequency sweep (good for jump/swoosh sounds)."""
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        # Linear frequency interpolation
        freq = freq_start + (freq_end - freq_start) * (t / duration)
        # Phase integration for smooth frequency change
        phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
        wave = amplitude * np.sin(phase)

        if fade_out:
            envelope = np.exp(-t * 3 / duration)
            wave *= envelope

        return wave.astype(np.float32)

    @staticmethod
    def mix_sounds(sounds: List[np.ndarray], volumes: Optional[List[float]] = None) -> np.ndarray:
        """Mix multiple sounds together."""
        if not sounds:
            return np.array([], dtype=np.float32)

        # Find max length
        max_len = max(len(s) for s in sounds)

        # Pad shorter sounds
        padded = []
        for s in sounds:
            if len(s) < max_len:
                padded.append(np.pad(s, (0, max_len - len(s))))
            else:
                padded.append(s)

        # Apply volumes and sum
        if volumes is None:
            volumes = [1.0] * len(sounds)

        mixed = np.zeros(max_len, dtype=np.float32)
        for s, v in zip(padded, volumes):
            mixed += s * v

        # Normalize to prevent clipping
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed /= peak

        return mixed

    @staticmethod
    def apply_envelope(sound: np.ndarray, attack: float = 0.01,
                       decay: float = 0.1, sustain: float = 0.7,
                       release: float = 0.2) -> np.ndarray:
        """Apply ADSR envelope to a sound."""
        length = len(sound)
        attack_samples = int(attack * SAMPLE_RATE)
        decay_samples = int(decay * SAMPLE_RATE)
        release_samples = int(release * SAMPLE_RATE)
        sustain_samples = length - attack_samples - decay_samples - release_samples

        if sustain_samples < 0:
            sustain_samples = 0

        envelope = np.zeros(length)

        # Attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay
        start = attack_samples
        end = start + decay_samples
        if end > start:
            envelope[start:end] = np.linspace(1, sustain, min(decay_samples, length - start))

        # Sustain
        start = end
        end = start + sustain_samples
        if end > start and start < length:
            envelope[start:min(end, length)] = sustain

        # Release
        start = end
        if start < length:
            envelope[start:] = np.linspace(sustain, 0, length - start)

        return (sound * envelope).astype(np.float32)


# =============================================================================
# SOUND LIBRARY
# =============================================================================

class SoundLibrary:
    """Pre-generates and caches all game sounds."""

    def __init__(self):
        self.sounds: Dict[SoundType, np.ndarray] = {}
        self._generate_all_sounds()

    def _generate_all_sounds(self):
        """Generate all sound effects."""
        gen = SoundGenerator

        # =====================
        # COMBAT SOUNDS
        # =====================

        # Light hit - short punch sound
        hit_light = gen.mix_sounds([
            gen.generate_noise(0.08, 0.6),
            gen.generate_sine(0.1, 200, 0.4),
            gen.generate_sine(0.08, 800, 0.2),
        ])
        self.sounds[SoundType.HIT_LIGHT] = hit_light

        # Medium hit - more impact
        hit_medium = gen.mix_sounds([
            gen.generate_noise(0.1, 0.7),
            gen.generate_sine(0.12, 150, 0.5),
            gen.generate_sine(0.08, 600, 0.3),
            gen.generate_sweep(0.08, 400, 100, 0.4),
        ])
        self.sounds[SoundType.HIT_MEDIUM] = hit_medium

        # Strong hit - heavy impact with bass
        hit_strong = gen.mix_sounds([
            gen.generate_noise(0.15, 0.8),
            gen.generate_sine(0.2, 80, 0.7),
            gen.generate_sine(0.15, 200, 0.5),
            gen.generate_sweep(0.1, 500, 50, 0.5),
        ])
        self.sounds[SoundType.HIT_STRONG] = hit_strong

        # Critical hit - explosive
        hit_critical = gen.mix_sounds([
            gen.generate_noise(0.2, 0.9),
            gen.generate_sine(0.25, 60, 0.8),
            gen.generate_sine(0.2, 120, 0.6),
            gen.generate_sweep(0.15, 800, 40, 0.6),
            gen.generate_square(0.05, 100, 0.3),
        ])
        self.sounds[SoundType.HIT_CRITICAL] = hit_critical

        # =====================
        # MOVEMENT SOUNDS
        # =====================

        # Jump - upward sweep
        jump = gen.mix_sounds([
            gen.generate_sweep(0.12, 200, 600, 0.4),
            gen.generate_noise(0.05, 0.2),
        ])
        self.sounds[SoundType.JUMP] = jump

        # Double jump - higher pitched sweep
        double_jump = gen.mix_sounds([
            gen.generate_sweep(0.1, 300, 800, 0.5),
            gen.generate_sine(0.08, 500, 0.3),
            gen.generate_noise(0.04, 0.2),
        ])
        self.sounds[SoundType.DOUBLE_JUMP] = double_jump

        # Land - thump
        land = gen.mix_sounds([
            gen.generate_noise(0.06, 0.4),
            gen.generate_sine(0.1, 80, 0.5),
            gen.generate_sweep(0.05, 200, 50, 0.3),
        ])
        self.sounds[SoundType.LAND] = land

        # Hard land - heavier thump
        land_hard = gen.mix_sounds([
            gen.generate_noise(0.1, 0.6),
            gen.generate_sine(0.15, 50, 0.7),
            gen.generate_sweep(0.08, 300, 30, 0.5),
        ])
        self.sounds[SoundType.LAND_HARD] = land_hard

        # =====================
        # DEFENSE SOUNDS
        # =====================

        # Shield up - bubble sound
        shield_up = gen.mix_sounds([
            gen.generate_sweep(0.1, 400, 800, 0.4),
            gen.generate_sine(0.15, 600, 0.3),
        ])
        self.sounds[SoundType.SHIELD_UP] = shield_up

        # Shield hit - ping/deflect
        shield_hit = gen.mix_sounds([
            gen.generate_sine(0.08, 1200, 0.5),
            gen.generate_sine(0.06, 800, 0.4),
            gen.generate_noise(0.05, 0.3),
        ])
        self.sounds[SoundType.SHIELD_HIT] = shield_hit

        # Shield break - shattering glass
        shield_break = gen.mix_sounds([
            gen.generate_noise(0.3, 0.8),
            gen.generate_sweep(0.2, 1000, 200, 0.6),
            gen.generate_sine(0.25, 400, 0.4),
            gen.generate_square(0.1, 300, 0.3),
        ])
        self.sounds[SoundType.SHIELD_BREAK] = shield_break

        # Dodge - woosh
        dodge = gen.mix_sounds([
            gen.generate_sweep(0.15, 300, 600, 0.4),
            gen.generate_noise(0.1, 0.3),
        ])
        self.sounds[SoundType.DODGE] = dodge

        # =====================
        # SPECIAL SOUNDS
        # =====================

        # Counter stance - charging/ready sound
        counter = gen.mix_sounds([
            gen.generate_sine(0.2, 300, 0.4),
            gen.generate_sweep(0.15, 200, 400, 0.3),
        ])
        self.sounds[SoundType.COUNTER] = counter

        # Counter success - impact + whoosh
        counter_success = gen.mix_sounds([
            gen.generate_noise(0.15, 0.7),
            gen.generate_sweep(0.1, 600, 200, 0.6),
            gen.generate_sine(0.12, 400, 0.5),
            gen.generate_square(0.05, 250, 0.3),
        ])
        self.sounds[SoundType.COUNTER_SUCCESS] = counter_success

        # Reflect - ping + reverse
        reflect = gen.mix_sounds([
            gen.generate_sine(0.1, 1000, 0.6),
            gen.generate_sweep(0.08, 800, 1500, 0.4),
            gen.generate_sine(0.05, 600, 0.3),
        ])
        self.sounds[SoundType.REFLECT] = reflect

        # Projectile fire - pew
        projectile_fire = gen.mix_sounds([
            gen.generate_sweep(0.1, 800, 400, 0.5),
            gen.generate_sine(0.08, 600, 0.4),
            gen.generate_noise(0.05, 0.2),
        ])
        self.sounds[SoundType.PROJECTILE_FIRE] = projectile_fire

        # Projectile hit - small explosion
        projectile_hit = gen.mix_sounds([
            gen.generate_noise(0.12, 0.6),
            gen.generate_sine(0.1, 200, 0.5),
            gen.generate_sweep(0.08, 400, 100, 0.4),
        ])
        self.sounds[SoundType.PROJECTILE_HIT] = projectile_hit

        # =====================
        # GAME EVENT SOUNDS
        # =====================

        # KO - big explosion + falling
        ko = gen.mix_sounds([
            gen.generate_noise(0.4, 0.9),
            gen.generate_sine(0.5, 50, 0.8),
            gen.generate_sweep(0.3, 600, 30, 0.7),
            gen.generate_square(0.15, 80, 0.4),
        ])
        self.sounds[SoundType.KO] = ko

        # Respawn - rising tone
        respawn = gen.mix_sounds([
            gen.generate_sweep(0.3, 200, 800, 0.5),
            gen.generate_sine(0.25, 400, 0.4),
            gen.generate_sweep(0.2, 400, 1000, 0.3),
        ])
        self.sounds[SoundType.RESPAWN] = respawn

        # Match start - fanfare-ish
        match_start = gen.mix_sounds([
            gen.generate_sine(0.2, 400, 0.5),
            gen.generate_sine(0.2, 500, 0.4),
            gen.generate_sine(0.3, 600, 0.5),
        ])
        # Add slight delays for chord effect
        start_chord = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
        start_chord[:len(match_start)] = match_start * 0.5
        offset1 = int(SAMPLE_RATE * 0.1)
        start_chord[offset1:offset1+len(match_start)] += match_start * 0.5
        self.sounds[SoundType.MATCH_START] = start_chord

        # Match end - victory jingle
        match_end = gen.mix_sounds([
            gen.generate_sine(0.15, 500, 0.5),
            gen.generate_sine(0.15, 600, 0.4),
            gen.generate_sine(0.25, 800, 0.5),
        ])
        end_jingle = np.zeros(int(SAMPLE_RATE * 0.6), dtype=np.float32)
        end_jingle[:len(match_end)] = match_end
        offset2 = int(SAMPLE_RATE * 0.2)
        end_part2 = gen.generate_sine(0.3, 1000, 0.5)
        end_jingle[offset2:offset2+len(end_part2)] += end_part2
        self.sounds[SoundType.MATCH_END] = end_jingle

        # =====================
        # MENU SOUNDS
        # =====================

        # Menu move - quick blip
        menu_move = gen.mix_sounds([
            gen.generate_sine(0.05, 600, 0.4),
            gen.generate_sine(0.04, 800, 0.3),
        ])
        self.sounds[SoundType.MENU_MOVE] = menu_move

        # Menu select - confirmation beep
        menu_select = gen.mix_sounds([
            gen.generate_sine(0.08, 800, 0.5),
            gen.generate_sine(0.1, 1000, 0.4),
            gen.generate_sweep(0.08, 600, 1200, 0.3),
        ])
        self.sounds[SoundType.MENU_SELECT] = menu_select

        # Menu back - lower descending tone
        menu_back = gen.mix_sounds([
            gen.generate_sweep(0.1, 600, 300, 0.4),
            gen.generate_sine(0.08, 400, 0.3),
        ])
        self.sounds[SoundType.MENU_BACK] = menu_back

        # =====================
        # COUNTDOWN SOUNDS
        # =====================

        # Countdown 3 - low beep
        countdown_3 = gen.mix_sounds([
            gen.generate_sine(0.15, 400, 0.6),
            gen.generate_sine(0.12, 200, 0.3),
        ])
        self.sounds[SoundType.COUNTDOWN_3] = countdown_3

        # Countdown 2 - medium beep
        countdown_2 = gen.mix_sounds([
            gen.generate_sine(0.15, 500, 0.6),
            gen.generate_sine(0.12, 250, 0.3),
        ])
        self.sounds[SoundType.COUNTDOWN_2] = countdown_2

        # Countdown 1 - high beep
        countdown_1 = gen.mix_sounds([
            gen.generate_sine(0.15, 600, 0.6),
            gen.generate_sine(0.12, 300, 0.3),
        ])
        self.sounds[SoundType.COUNTDOWN_1] = countdown_1

        # Countdown GO - big rising chord
        go_sound = gen.mix_sounds([
            gen.generate_sweep(0.3, 400, 1200, 0.7),
            gen.generate_sine(0.25, 800, 0.5),
            gen.generate_sine(0.2, 600, 0.4),
            gen.generate_square(0.1, 400, 0.2),
        ])
        self.sounds[SoundType.COUNTDOWN_GO] = go_sound

        # Cycle 13: Crowd reactions
        self._generate_crowd_sounds(gen)

        # Cycle 15: Stage ambient sounds
        self._generate_stage_ambients(gen)

        # Cycle 18: Character preview sounds
        self._generate_character_preview_sounds(gen)

    def _generate_crowd_sounds(self, gen: SoundGenerator):
        """Cycle 13: Generate crowd reaction sounds."""
        # Crowd gasp - sharp rising multi-voice "ooh" sound
        # Simulated with layered sine waves with slight detune
        gasp_components = []
        for i, freq_mult in enumerate([1.0, 1.01, 0.99, 1.02, 0.98]):
            base_freq = 300 + i * 20
            # Rising pitch simulates surprise
            rising = gen.generate_sweep(0.3, base_freq, base_freq * 1.5, 0.15)
            gasp_components.append(rising)

        # Add some noise for texture
        noise_component = gen.generate_noise(0.15, 0.08)
        gasp_components.append(noise_component)

        crowd_gasp = gen.mix_sounds(gasp_components)
        self.sounds[SoundType.CROWD_GASP] = crowd_gasp

        # Crowd cheer - sustained happy noise with harmonic content
        cheer_components = []
        # Multiple "voice" frequencies
        for freq in [200, 300, 400, 500, 600]:
            # Varied volume and slight timing offsets via different durations
            voice = gen.generate_sine(0.4 + (freq / 3000), freq, 0.12)
            cheer_components.append(voice)

        # Add noise for the roar texture
        noise = gen.generate_noise(0.5, 0.2)
        cheer_components.append(noise)

        # Rising sweep for excitement
        sweep = gen.generate_sweep(0.3, 200, 600, 0.1)
        cheer_components.append(sweep)

        crowd_cheer = gen.mix_sounds(cheer_components)
        self.sounds[SoundType.CROWD_CHEER] = crowd_cheer

    def _generate_stage_ambients(self, gen: SoundGenerator):
        """Cycle 15: Generate stage-specific ambient sounds."""
        # ====================
        # BATTLEFIELD: Wind/nature ambience
        # ====================
        # Use brown noise (filtered white noise) for wind-like sound
        # 3 seconds, loopable
        duration = 3.0
        samples = int(SAMPLE_RATE * duration)

        # Generate white noise and apply low-pass filtering for wind
        white_noise = np.random.random(samples) * 2 - 1

        # Simple low-pass filter: running average
        # This creates a "brown noise" effect for wind
        window_size = 100  # ~500Hz cutoff approximation
        wind_filtered = np.convolve(white_noise, np.ones(window_size) / window_size, mode='same')

        # Volume modulation for natural wind gusts
        t = np.linspace(0, duration, samples)
        modulation = 0.12 + 0.04 * np.sin(t * 0.8) + 0.03 * np.sin(t * 1.3)
        battlefield_ambient = (wind_filtered * modulation).astype(np.float32)

        # Ensure seamless loop: fade first/last 0.1 seconds
        fade_samples = int(SAMPLE_RATE * 0.1)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        battlefield_ambient[:fade_samples] *= fade_in
        battlefield_ambient[-fade_samples:] *= fade_out

        self.sounds[SoundType.AMBIENT_BATTLEFIELD] = battlefield_ambient

        # ====================
        # FINAL DESTINATION: Cosmic drone/hum
        # ====================
        # Low frequency drone with harmonics for space ambience
        duration = 4.0
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples)

        # Base: 60Hz hum
        base_drone = 0.08 * np.sin(2 * np.pi * 60 * t)

        # Harmonics for depth
        harmonic1 = 0.04 * np.sin(2 * np.pi * 90 * t)  # 90Hz
        harmonic2 = 0.03 * np.sin(2 * np.pi * 120 * t)  # 120Hz
        harmonic3 = 0.02 * np.sin(2 * np.pi * 180 * t)  # 180Hz

        # Subtle shimmer: very quiet high-frequency noise bursts
        shimmer = np.zeros(samples)
        num_shimmers = 8
        for i in range(num_shimmers):
            start_idx = int((i / num_shimmers) * samples * 0.9)
            shimmer_len = int(SAMPLE_RATE * 0.1)  # 100ms shimmer
            if start_idx + shimmer_len < samples:
                # High frequency content for sparkle
                shimmer_t = np.linspace(0, 0.1, shimmer_len)
                shimmer_wave = 0.015 * np.sin(2 * np.pi * 2000 * shimmer_t) * np.exp(-shimmer_t * 20)
                shimmer[start_idx:start_idx + shimmer_len] += shimmer_wave

        # Combine all components
        cosmic_ambient = (base_drone + harmonic1 + harmonic2 + harmonic3 + shimmer)

        # Volume envelope for subtle pulsing
        pulse = 0.9 + 0.1 * np.sin(t * 0.5)
        cosmic_ambient = (cosmic_ambient * pulse).astype(np.float32)

        # Ensure seamless loop
        fade_samples = int(SAMPLE_RATE * 0.1)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        cosmic_ambient[:fade_samples] *= fade_in
        cosmic_ambient[-fade_samples:] *= fade_out

        self.sounds[SoundType.AMBIENT_FINAL_DESTINATION] = cosmic_ambient

    def _generate_character_preview_sounds(self, gen: SoundGenerator):
        """
        Cycle 18: Generate character-specific preview sounds for character select.

        Each character has a unique sound that reflects their playstyle:
        - RUSH: Fast whoosh (speed)
        - ZONE: Energy zap/charge (ranged)
        - GRAB: Heavy thud/impact (grappler)
        - FLEX: Shield chime (balanced/defensive)
        """
        # ====================
        # RUSH: Speed whoosh
        # ====================
        # Fast ascending sweep with multiple harmonics
        rush_preview = gen.mix_sounds([
            gen.generate_sweep(0.15, 200, 800, 0.4),
            gen.generate_sweep(0.12, 400, 1200, 0.25),
            gen.generate_noise(0.1, 0.15),  # Wind texture
        ])
        self.sounds[SoundType.CHAR_PREVIEW_RUSH] = rush_preview

        # ====================
        # ZONE: Energy zap
        # ====================
        # Electrical charge/discharge sound
        zone_preview = gen.mix_sounds([
            gen.generate_square(0.08, 600, 0.2),
            gen.generate_sine(0.15, 800, 0.3),
            gen.generate_sweep(0.1, 1000, 400, 0.2),
            gen.generate_noise(0.05, 0.1),  # Crackle
        ])
        self.sounds[SoundType.CHAR_PREVIEW_ZONE] = zone_preview

        # ====================
        # GRAB: Heavy thud
        # ====================
        # Low impact with bass rumble
        grab_preview = gen.mix_sounds([
            gen.generate_sine(0.2, 80, 0.5),   # Deep bass
            gen.generate_sine(0.15, 120, 0.3),
            gen.generate_sweep(0.1, 200, 60, 0.3),  # Descending thud
            gen.generate_noise(0.08, 0.15),  # Impact texture
        ])
        self.sounds[SoundType.CHAR_PREVIEW_GRAB] = grab_preview

        # ====================
        # FLEX: Shield chime
        # ====================
        # Metallic chime with harmonics
        flex_preview = gen.mix_sounds([
            gen.generate_sine(0.2, 523, 0.3),   # C5
            gen.generate_sine(0.18, 659, 0.25), # E5
            gen.generate_sine(0.16, 784, 0.2),  # G5
            gen.generate_sine(0.25, 1047, 0.15), # C6
        ])
        self.sounds[SoundType.CHAR_PREVIEW_FLEX] = flex_preview

        # ====================
        # Character Lock Confirmation
        # ====================
        # Rising confirmation sound
        char_lock = gen.mix_sounds([
            gen.generate_sweep(0.08, 400, 800, 0.4),
            gen.generate_sine(0.12, 800, 0.3),
            gen.generate_sine(0.1, 1200, 0.2),
        ])
        self.sounds[SoundType.CHAR_LOCK] = char_lock

    def get_sound(self, sound_type: SoundType) -> np.ndarray:
        """Get a pre-generated sound."""
        return self.sounds.get(sound_type, np.array([], dtype=np.float32))

    def get_hit_sound_for_knockback(self, knockback_magnitude: float) -> Tuple[SoundType, np.ndarray]:
        """Get appropriate hit sound based on knockback intensity."""
        if knockback_magnitude < 5:
            sound_type = SoundType.HIT_LIGHT
        elif knockback_magnitude < 12:
            sound_type = SoundType.HIT_MEDIUM
        elif knockback_magnitude < 20:
            sound_type = SoundType.HIT_STRONG
        else:
            sound_type = SoundType.HIT_CRITICAL

        return sound_type, self.get_sound(sound_type)

    def get_character_preview_sound(self, character_name: str) -> Optional[SoundType]:
        """
        Cycle 18: Get character-specific preview sound type.

        Args:
            character_name: Character type name (e.g., "RUSH", "ZONE")

        Returns:
            SoundType for the character preview sound, or None if not found
        """
        char_sound_map = {
            "RUSH": SoundType.CHAR_PREVIEW_RUSH,
            "ZONE": SoundType.CHAR_PREVIEW_ZONE,
            "GRAB": SoundType.CHAR_PREVIEW_GRAB,
            "FLEX": SoundType.CHAR_PREVIEW_FLEX,
        }
        return char_sound_map.get(character_name.upper())


# =============================================================================
# AUDIO MANAGER
# =============================================================================

class AudioManager:
    """
    Manages audio playback with concurrent sound limiting.

    Uses a separate thread for non-blocking audio playback.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the audio manager.

        Args:
            enabled: Whether audio is enabled (can be False for headless mode)
        """
        self.enabled = enabled and AUDIO_AVAILABLE
        self.volume = MASTER_VOLUME
        self.sound_library: Optional[SoundLibrary] = None

        # Sound queue for async playback
        self._sound_queue: queue.Queue = queue.Queue()
        self._active_sounds: List[int] = []  # Track active stream IDs
        self._active_count = 0
        self._lock = threading.Lock()

        # Audio playback thread
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Cycle 15: Ambient sound state
        self._ambient_running = False
        self._ambient_thread: Optional[threading.Thread] = None
        self._ambient_sound_type: Optional[SoundType] = None

        if self.enabled:
            self._initialize()

    def _initialize(self):
        """Initialize sound library and playback thread."""
        try:
            # Generate all sounds
            self.sound_library = SoundLibrary()

            # Start playback thread
            self._running = True
            self._thread = threading.Thread(target=self._playback_loop, daemon=True)
            self._thread.start()

            print(f"[AudioSystem] Initialized with {len(self.sound_library.sounds)} sounds")
        except Exception as e:
            print(f"[AudioSystem] Failed to initialize: {e}")
            self.enabled = False

    def _playback_loop(self):
        """Background thread that processes sound queue."""
        while self._running:
            try:
                # Get next sound from queue with timeout
                sound_data = self._sound_queue.get(timeout=0.1)

                if sound_data is None:
                    continue

                # Check concurrent sound limit
                with self._lock:
                    if self._active_count >= MAX_CONCURRENT_SOUNDS:
                        continue  # Drop sound if too many playing
                    self._active_count += 1

                # Play sound
                try:
                    sound = sound_data * self.volume
                    sd.play(sound, SAMPLE_RATE, blocking=False)
                except Exception as e:
                    pass  # Silently handle playback errors
                finally:
                    # Estimate when sound ends and decrement counter
                    duration = len(sound_data) / SAMPLE_RATE
                    threading.Timer(duration, self._sound_finished).start()

            except queue.Empty:
                continue
            except Exception as e:
                pass  # Silently handle errors

    def _sound_finished(self):
        """Called when a sound finishes playing."""
        with self._lock:
            self._active_count = max(0, self._active_count - 1)

    def play(self, sound_type: SoundType, volume_scale: float = 1.0):
        """
        Play a sound effect.

        Args:
            sound_type: Type of sound to play
            volume_scale: Volume multiplier (0.0 - 1.0)
        """
        if not self.enabled or self.sound_library is None:
            return

        sound = self.sound_library.get_sound(sound_type)
        if len(sound) == 0:
            return

        # Scale volume
        scaled_sound = sound * np.clip(volume_scale, 0.0, 1.0)

        # Queue for playback
        try:
            self._sound_queue.put_nowait(scaled_sound)
        except queue.Full:
            pass  # Drop sound if queue is full

    def play_hit(self, knockback_magnitude: float):
        """
        Play appropriate hit sound based on knockback intensity.

        Args:
            knockback_magnitude: The knockback value from the hit
        """
        if not self.enabled or self.sound_library is None:
            return

        sound_type, sound = self.sound_library.get_hit_sound_for_knockback(knockback_magnitude)

        # Scale volume based on intensity
        volume_scale = min(1.0, 0.5 + knockback_magnitude / 30.0)
        scaled_sound = sound * volume_scale

        try:
            self._sound_queue.put_nowait(scaled_sound)
        except queue.Full:
            pass

    def play_projectile_fire(self, character_type: Optional[str] = None):
        """Play projectile fire sound."""
        self.play(SoundType.PROJECTILE_FIRE, 0.7)

    def play_shield_hit(self, damage: float):
        """Play shield hit sound, scaled by damage."""
        volume = min(1.0, 0.5 + damage / 20.0)
        self.play(SoundType.SHIELD_HIT, volume)

    def play_ko(self):
        """Play KO sound."""
        self.play(SoundType.KO, 1.0)

    def play_respawn(self):
        """Play respawn sound."""
        self.play(SoundType.RESPAWN, 0.8)

    def set_volume(self, volume: float):
        """Set master volume (0.0 - 1.0)."""
        self.volume = np.clip(volume, 0.0, 1.0)

    def set_enabled(self, enabled: bool):
        """Enable or disable audio."""
        self.enabled = enabled and AUDIO_AVAILABLE

    def stop_all(self):
        """Stop all currently playing sounds."""
        if AUDIO_AVAILABLE:
            try:
                sd.stop()
            except:
                pass

        with self._lock:
            self._active_count = 0

        # Clear queue
        while not self._sound_queue.empty():
            try:
                self._sound_queue.get_nowait()
            except queue.Empty:
                break

    # =========================================================================
    # Cycle 15: Ambient Sound Methods
    # =========================================================================

    def start_ambient(self, stage_name: str):
        """
        Start stage-specific ambient sound loop.

        Args:
            stage_name: Name of the stage ("battlefield" or "final_destination")
        """
        if not self.enabled or self.sound_library is None:
            return

        # Stop any existing ambient
        self.stop_ambient()

        # Determine which ambient to play based on stage name
        stage_lower = stage_name.lower()
        if "final" in stage_lower or "destination" in stage_lower:
            self._ambient_sound_type = SoundType.AMBIENT_FINAL_DESTINATION
        else:
            # Default to Battlefield for all other stages
            self._ambient_sound_type = SoundType.AMBIENT_BATTLEFIELD

        # Start ambient playback thread
        self._ambient_running = True
        self._ambient_thread = threading.Thread(target=self._ambient_loop, daemon=True)
        self._ambient_thread.start()

    def _ambient_loop(self):
        """Background thread for looping ambient sound."""
        if not self.enabled or self.sound_library is None:
            return

        sound = self.sound_library.get_sound(self._ambient_sound_type)
        if len(sound) == 0:
            return

        # Ambient volume is lower than sfx
        ambient_volume = self.volume * 0.4

        while self._ambient_running:
            try:
                # Play the ambient sound
                scaled_sound = sound * ambient_volume
                sd.play(scaled_sound, SAMPLE_RATE, blocking=True)  # Blocking to loop
            except Exception:
                # If playback fails, wait a bit and try again
                time.sleep(0.5)

    def stop_ambient(self):
        """Stop the ambient sound loop."""
        self._ambient_running = False

        if self._ambient_thread and self._ambient_thread.is_alive():
            # Give it a moment to stop
            self._ambient_thread.join(timeout=0.5)

        self._ambient_thread = None
        self._ambient_sound_type = None

    def cleanup(self):
        """Clean up audio resources."""
        self._running = False
        self.stop_ambient()  # Cycle 15: Stop ambient on cleanup
        self.stop_all()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


# =============================================================================
# AUDIO CALLBACKS (for game integration)
# =============================================================================

class GameAudioCallbacks:
    """
    Wrapper class that provides callback functions for game events.

    Connect these to GameLoop callbacks for automatic audio.
    """

    def __init__(self, audio_manager: AudioManager):
        self.audio = audio_manager
        self._last_jump_frame: Dict[int, int] = {}  # Track last jump frame per player

    def on_hit(self, attacker, defender, hitbox, contact_x, contact_y, knockback_magnitude):
        """Called when an attack hits."""
        self.audio.play_hit(knockback_magnitude)

        # Cycle 13: Crowd gasp on big hits (knockback > 20)
        if knockback_magnitude > 20:
            self.audio.play(SoundType.CROWD_GASP, 0.4)

    def on_projectile_hit(self, projectile, defender, contact_x, contact_y, knockback_magnitude):
        """Called when a projectile hits."""
        self.audio.play(SoundType.PROJECTILE_HIT, 0.8)

    def on_reflect(self, projectile, reflector, x, y):
        """Called when a projectile is reflected."""
        self.audio.play(SoundType.REFLECT, 0.9)

    def on_land(self, fighter, x, y):
        """Called when a fighter lands."""
        # Determine landing intensity based on vertical velocity
        # (We don't have access to velocity here, so use default)
        self.audio.play(SoundType.LAND, 0.6)

    def on_jump(self, fighter, is_double_jump: bool = False):
        """Called when a fighter jumps."""
        if is_double_jump:
            self.audio.play(SoundType.DOUBLE_JUMP, 0.6)
        else:
            self.audio.play(SoundType.JUMP, 0.5)

    def on_shield_hit(self, defender, damage: float):
        """Called when shield is hit."""
        self.audio.play_shield_hit(damage)

    def on_shield_break(self, fighter):
        """Called when shield breaks."""
        self.audio.play(SoundType.SHIELD_BREAK, 1.0)

    def on_dodge(self, fighter):
        """Called when fighter dodges."""
        self.audio.play(SoundType.DODGE, 0.5)

    def on_counter(self, fighter, success: bool):
        """Called when counter is used."""
        if success:
            self.audio.play(SoundType.COUNTER_SUCCESS, 0.9)
        else:
            self.audio.play(SoundType.COUNTER, 0.4)

    def on_projectile_fire(self, fighter, projectile):
        """Called when projectile is fired."""
        self.audio.play(SoundType.PROJECTILE_FIRE, 0.7)

    def on_ko(self, fighter):
        """Called when fighter is KO'd."""
        self.audio.play_ko()
        # Cycle 13: Crowd cheer on KO
        self.audio.play(SoundType.CROWD_CHEER, 0.5)

    def on_respawn(self, fighter):
        """Called when fighter respawns."""
        self.audio.play_respawn()

    def on_match_start(self):
        """Called when match starts."""
        self.audio.play(SoundType.MATCH_START, 0.7)

    def on_match_end(self, winner):
        """Called when match ends."""
        self.audio.play(SoundType.MATCH_END, 0.8)


# =============================================================================
# SINGLETON GLOBAL INSTANCE
# =============================================================================

_audio_manager: Optional[AudioManager] = None

def get_audio_manager() -> AudioManager:
    """Get the global audio manager instance."""
    global _audio_manager
    if _audio_manager is None:
        _audio_manager = AudioManager()
    return _audio_manager

def cleanup_audio():
    """Clean up global audio manager."""
    global _audio_manager
    if _audio_manager is not None:
        _audio_manager.cleanup()
        _audio_manager = None


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Audio System...")

    if not AUDIO_AVAILABLE:
        print("sounddevice not available - cannot test audio")
        exit(1)

    audio = get_audio_manager()

    # Test individual sounds
    print("\nPlaying test sounds:")

    sounds_to_test = [
        (SoundType.HIT_LIGHT, "Light hit"),
        (SoundType.HIT_MEDIUM, "Medium hit"),
        (SoundType.HIT_STRONG, "Strong hit"),
        (SoundType.JUMP, "Jump"),
        (SoundType.LAND, "Land"),
        (SoundType.SHIELD_HIT, "Shield hit"),
        (SoundType.DODGE, "Dodge"),
        (SoundType.PROJECTILE_FIRE, "Projectile fire"),
        (SoundType.KO, "KO"),
    ]

    for sound_type, name in sounds_to_test:
        print(f"  Playing: {name}")
        audio.play(sound_type)
        time.sleep(0.5)

    print("\nTesting hit sound scaling:")
    for kb in [3, 8, 15, 25]:
        print(f"  Knockback {kb}")
        audio.play_hit(kb)
        time.sleep(0.4)

    print("\nAudio test complete!")
    cleanup_audio()
