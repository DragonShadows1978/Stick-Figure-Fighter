"""
Stick Figure Fighter - Character Definitions

Defines the 4 unique playable characters with distinct playstyles:
- RUSH: Fast rushdown with combo focus
- ZONE: Ranged zoner with projectiles
- GRAB: Slow grappler with command grabs
- FLEX: Balanced all-rounder

Each character has:
- Unique FighterStats (speed, weight, jump, etc.)
- Character-specific attacks with different hitbox data
- Neutral/Side/Up/Down specials as defined in Game_Idea.md
- Distinct color for rendering
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto

from game_engine import (
    FighterStats, AttackType, AttackData, Hitbox, Vector2
)


# =============================================================================
# CHARACTER ENUM
# =============================================================================

class CharacterType(Enum):
    """Available character types."""
    RUSH = auto()  # Fast rushdown
    ZONE = auto()  # Ranged zoner
    GRAB = auto()  # Slow grappler
    FLEX = auto()  # Balanced all-rounder


# =============================================================================
# CHARACTER STATS
# =============================================================================

# RUSH - The Rushdown
# Fast, aggressive, combo-focused
# High speed, low weight, low knockback per hit
RUSH_STATS = FighterStats(
    name="RUSH",
    walk_speed=4.0,          # Fast walk
    run_speed=8.0,           # Very fast run
    air_speed=5.5,           # High air mobility
    jump_force=14.5,         # Slightly higher jump
    double_jump_force=13.0,  # Good double jump
    weight=85.0,             # Light (easier to knock out)
    fall_speed=0.55,         # Slightly floaty
    fast_fall_speed=1.1,
    width=45,                # Slightly smaller
    height=75,
)

# ZONE - The Zoner
# Ranged, keepaway, stage control
# Medium speed, medium weight, projectile-based
ZONE_STATS = FighterStats(
    name="ZONE",
    walk_speed=2.8,          # Slower walk
    run_speed=5.5,           # Average run
    air_speed=3.8,           # Lower air mobility
    jump_force=13.0,         # Average jump
    double_jump_force=11.0,  # Average double jump
    weight=95.0,             # Slightly light
    fall_speed=0.6,          # Normal fall
    fast_fall_speed=1.2,
    width=50,
    height=80,
)

# GRAB - The Grappler
# Slow, powerful, command grab focused
# Slow speed, heavy weight, high knockback
GRAB_STATS = FighterStats(
    name="GRAB",
    walk_speed=2.2,          # Very slow walk
    run_speed=4.5,           # Slow run
    air_speed=3.0,           # Poor air mobility
    jump_force=11.5,         # Low jump
    double_jump_force=10.0,  # Weak double jump
    weight=125.0,            # Very heavy (hard to knock out)
    fall_speed=0.75,         # Falls fast
    fast_fall_speed=1.4,
    width=55,                # Larger hitbox
    height=85,
)

# FLEX - The All-Rounder
# Balanced, adaptable, fundamental
# Medium everything
FLEX_STATS = FighterStats(
    name="FLEX",
    walk_speed=3.0,          # Average walk
    run_speed=6.0,           # Average run
    air_speed=4.0,           # Average air mobility
    jump_force=14.0,         # Average jump
    double_jump_force=12.0,  # Average double jump
    weight=100.0,            # Average weight
    fall_speed=0.6,          # Normal fall
    fast_fall_speed=1.2,
    width=50,
    height=80,
)


# =============================================================================
# CHARACTER COLORS (RGB)
# =============================================================================

CHARACTER_COLORS: Dict[CharacterType, Tuple[int, int, int]] = {
    CharacterType.RUSH: (255, 80, 80),     # Red - aggressive
    CharacterType.ZONE: (80, 180, 255),    # Cyan - cool, calculated
    CharacterType.GRAB: (180, 120, 60),    # Brown - earthy, heavy
    CharacterType.FLEX: (120, 220, 120),   # Green - balanced
}


# =============================================================================
# CHARACTER-SPECIFIC ATTACK DATA
# =============================================================================

def create_rush_attacks() -> Dict[AttackType, AttackData]:
    """Create attack data for RUSH character.

    RUSH focuses on:
    - Quick startup frames
    - Low knockback (for combos)
    - Multi-hit moves
    - Low recovery
    """
    attacks = {}

    # JAB - Very fast, low knockback (combo starter, but scales to prevent infinites)
    attacks[AttackType.JAB] = AttackData(
        attack_type=AttackType.JAB,
        total_frames=14,  # Fast recovery
        hitboxes=[Hitbox(
            offset=Vector2(15, -45),
            width=35,
            height=25,
            damage=2.5,       # Slightly increased
            knockback_base=2.0,   # Slightly higher to prevent true infinite at high %
            knockback_scaling=0.04,  # More scaling breaks combos at higher %
            angle=35,         # Slightly more upward to push opponent out
            active_frames=(3, 5)  # Quick startup
        )],
        startup_frames=3,
        recovery_frames=9
    )

    # FORWARD TILT - Combo tool
    attacks[AttackType.FORWARD_TILT] = AttackData(
        attack_type=AttackType.FORWARD_TILT,
        total_frames=22,
        hitboxes=[Hitbox(
            offset=Vector2(20, -50),
            width=50,
            height=30,
            damage=6.0,
            knockback_base=2.5,
            knockback_scaling=0.05,
            angle=35,
            active_frames=(5, 8)
        )],
        startup_frames=5,
        recovery_frames=14
    )

    # UP TILT - Juggle tool
    attacks[AttackType.UP_TILT] = AttackData(
        attack_type=AttackType.UP_TILT,
        total_frames=20,
        hitboxes=[Hitbox(
            offset=Vector2(-15, -85),
            width=50,
            height=35,
            damage=5.0,
            knockback_base=4.0,
            knockback_scaling=0.08,
            angle=85,  # Sends up
            active_frames=(4, 7)
        )],
        startup_frames=4,
        recovery_frames=13
    )

    # DOWN TILT - Poke tool
    attacks[AttackType.DOWN_TILT] = AttackData(
        attack_type=AttackType.DOWN_TILT,
        total_frames=18,
        hitboxes=[Hitbox(
            offset=Vector2(15, -15),
            width=45,
            height=20,
            damage=4.0,
            knockback_base=2.0,
            knockback_scaling=0.04,
            angle=40,
            active_frames=(4, 6)
        )],
        startup_frames=4,
        recovery_frames=12
    )

    # FORWARD SMASH - Kill move (tuned for ~120% KO at center stage)
    attacks[AttackType.FORWARD_SMASH] = AttackData(
        attack_type=AttackType.FORWARD_SMASH,
        total_frames=40,
        hitboxes=[Hitbox(
            offset=Vector2(25, -55),
            width=60,
            height=40,
            damage=15.0,          # Slightly increased for RUSH's primary kill move
            knockback_base=7.0,   # Higher base for reliable KOs
            knockback_scaling=0.11,
            angle=35,
            active_frames=(12, 15)
        )],
        startup_frames=12,
        recovery_frames=25
    )

    # UP SMASH (tuned for vertical KOs at ~120%)
    attacks[AttackType.UP_SMASH] = AttackData(
        attack_type=AttackType.UP_SMASH,
        total_frames=38,
        hitboxes=[Hitbox(
            offset=Vector2(-20, -95),
            width=55,
            height=45,
            damage=14.0,
            knockback_base=7.5,   # Strong vertical launcher
            knockback_scaling=0.12,
            angle=88,
            active_frames=(10, 14)
        )],
        startup_frames=10,
        recovery_frames=24
    )

    # DOWN SMASH
    attacks[AttackType.DOWN_SMASH] = AttackData(
        attack_type=AttackType.DOWN_SMASH,
        total_frames=35,
        hitboxes=[
            Hitbox(  # Front hit
                offset=Vector2(15, -20),
                width=50,
                height=25,
                damage=11.0,
                knockback_base=5.0,
                knockback_scaling=0.10,
                angle=30,
                active_frames=(8, 10)
            ),
            Hitbox(  # Back hit
                offset=Vector2(-45, -20),
                width=50,
                height=25,
                damage=10.0,
                knockback_base=4.5,
                knockback_scaling=0.09,
                angle=150,  # Sends backward
                active_frames=(11, 13)
            )
        ],
        startup_frames=8,
        recovery_frames=22
    )

    # NEUTRAL AIR - Sex kick (long lasting, low knockback)
    attacks[AttackType.NEUTRAL_AIR] = AttackData(
        attack_type=AttackType.NEUTRAL_AIR,
        total_frames=28,
        hitboxes=[Hitbox(
            offset=Vector2(-22, -55),
            width=65,
            height=55,
            damage=5.0,
            knockback_base=2.5,
            knockback_scaling=0.05,
            angle=45,
            active_frames=(4, 18)  # Long active
        )],
        startup_frames=4,
        recovery_frames=10,
        landing_lag=6
    )

    # FORWARD AIR - Knee (kill move in air)
    attacks[AttackType.FORWARD_AIR] = AttackData(
        attack_type=AttackType.FORWARD_AIR,
        total_frames=30,
        hitboxes=[Hitbox(
            offset=Vector2(20, -50),
            width=45,
            height=35,
            damage=10.0,
            knockback_base=5.0,
            knockback_scaling=0.10,
            angle=35,
            active_frames=(6, 9)
        )],
        startup_frames=6,
        recovery_frames=21,
        landing_lag=10
    )

    # BACK AIR - Fast backward kick
    attacks[AttackType.BACK_AIR] = AttackData(
        attack_type=AttackType.BACK_AIR,
        total_frames=26,
        hitboxes=[Hitbox(
            offset=Vector2(-25, -50),
            width=45,
            height=35,
            damage=9.0,
            knockback_base=4.5,
            knockback_scaling=0.09,
            angle=145,  # Backward angle
            active_frames=(7, 10)
        )],
        startup_frames=7,
        recovery_frames=16,
        landing_lag=8
    )

    # UP AIR - Juggle tool
    attacks[AttackType.UP_AIR] = AttackData(
        attack_type=AttackType.UP_AIR,
        total_frames=24,
        hitboxes=[Hitbox(
            offset=Vector2(-18, -85),
            width=55,
            height=40,
            damage=7.0,
            knockback_base=4.0,
            knockback_scaling=0.08,
            angle=82,
            active_frames=(5, 9)
        )],
        startup_frames=5,
        recovery_frames=15,
        landing_lag=7
    )

    # DOWN AIR - Stomp (meteor smash)
    attacks[AttackType.DOWN_AIR] = AttackData(
        attack_type=AttackType.DOWN_AIR,
        total_frames=32,
        hitboxes=[Hitbox(
            offset=Vector2(-15, -20),
            width=40,
            height=35,
            damage=11.0,
            knockback_base=5.0,
            knockback_scaling=0.10,
            angle=270,  # Straight down (meteor)
            active_frames=(10, 13)
        )],
        startup_frames=10,
        recovery_frames=19,
        landing_lag=14
    )

    # NEUTRAL SPECIAL - Rapid Punches (multi-hit)
    attacks[AttackType.NEUTRAL_SPECIAL] = AttackData(
        attack_type=AttackType.NEUTRAL_SPECIAL,
        total_frames=45,
        hitboxes=[
            Hitbox(
                offset=Vector2(18, -50),
                width=35,
                height=30,
                damage=1.5,
                knockback_base=0.5,
                knockback_scaling=0.01,
                angle=45,
                active_frames=(8, 10)
            ),
            Hitbox(
                offset=Vector2(18, -50),
                width=35,
                height=30,
                damage=1.5,
                knockback_base=0.5,
                knockback_scaling=0.01,
                angle=45,
                active_frames=(14, 16)
            ),
            Hitbox(
                offset=Vector2(18, -50),
                width=35,
                height=30,
                damage=1.5,
                knockback_base=0.5,
                knockback_scaling=0.01,
                angle=45,
                active_frames=(20, 22)
            ),
            Hitbox(  # Finisher
                offset=Vector2(22, -50),
                width=45,
                height=35,
                damage=4.0,
                knockback_base=4.0,
                knockback_scaling=0.08,
                angle=40,
                active_frames=(28, 31)
            )
        ],
        startup_frames=8,
        recovery_frames=14
    )

    # SIDE SPECIAL - Dash Attack (quick forward movement with hit)
    attacks[AttackType.SIDE_SPECIAL] = AttackData(
        attack_type=AttackType.SIDE_SPECIAL,
        total_frames=35,
        hitboxes=[Hitbox(
            offset=Vector2(30, -50),
            width=55,
            height=40,
            damage=8.0,
            knockback_base=3.0,
            knockback_scaling=0.07,
            angle=50,
            active_frames=(6, 14)  # Long active during dash
        )],
        startup_frames=6,
        recovery_frames=21
    )

    # UP SPECIAL - Rising Uppercut (recovery)
    attacks[AttackType.UP_SPECIAL] = AttackData(
        attack_type=AttackType.UP_SPECIAL,
        total_frames=48,
        hitboxes=[Hitbox(
            offset=Vector2(10, -70),
            width=45,
            height=55,
            damage=12.0,
            knockback_base=6.0,
            knockback_scaling=0.11,
            angle=80,
            active_frames=(5, 12)
        )],
        startup_frames=5,
        recovery_frames=36  # Long recovery (punishable)
    )

    # DOWN SPECIAL - Counter Stance (RUSH's signature defensive option)
    # Counter window: frames 5-20 (15 frame window as specified)
    # If hit during counter window, counterattack triggers with 1.5x damage boost
    # Punishable on whiff with long recovery
    attacks[AttackType.DOWN_SPECIAL] = AttackData(
        attack_type=AttackType.DOWN_SPECIAL,
        total_frames=55,  # Slightly longer for punish window
        hitboxes=[Hitbox(
            # Counterattack hitbox (only active when counter succeeds)
            offset=Vector2(25, -55),
            width=65,
            height=50,
            damage=11.0,  # Base damage (boosted by 1.5x when counter triggers = 16.5)
            knockback_base=6.5,
            knockback_scaling=0.11,
            angle=45,
            active_frames=(20, 28)  # Counter window ends at 20, attack follows
        )],
        startup_frames=5,
        recovery_frames=47,  # Very punishable on whiff
        is_counter=True  # This is a counter move - 15 frame window (frames 5-20)
    )

    return attacks


def create_zone_attacks() -> Dict[AttackType, AttackData]:
    """Create attack data for ZONE character.

    ZONE focuses on:
    - Longer range
    - Projectile specials
    - Zoning tools
    - Medium frame data
    """
    attacks = {}

    # JAB - Standard
    attacks[AttackType.JAB] = AttackData(
        attack_type=AttackType.JAB,
        total_frames=18,
        hitboxes=[Hitbox(
            offset=Vector2(18, -48),
            width=38,
            height=28,
            damage=3.0,
            knockback_base=2.0,
            knockback_scaling=0.04,
            angle=40,
            active_frames=(5, 8)
        )],
        startup_frames=5,
        recovery_frames=10
    )

    # FORWARD TILT - Long range poke
    attacks[AttackType.FORWARD_TILT] = AttackData(
        attack_type=AttackType.FORWARD_TILT,
        total_frames=28,
        hitboxes=[Hitbox(
            offset=Vector2(25, -52),
            width=70,  # Long range
            height=30,
            damage=7.0,
            knockback_base=3.5,
            knockback_scaling=0.06,
            angle=38,
            active_frames=(8, 12)
        )],
        startup_frames=8,
        recovery_frames=16
    )

    # UP TILT - Anti-air
    attacks[AttackType.UP_TILT] = AttackData(
        attack_type=AttackType.UP_TILT,
        total_frames=26,
        hitboxes=[Hitbox(
            offset=Vector2(-18, -90),
            width=55,
            height=45,
            damage=6.0,
            knockback_base=5.0,
            knockback_scaling=0.09,
            angle=88,
            active_frames=(6, 10)
        )],
        startup_frames=6,
        recovery_frames=16
    )

    # DOWN TILT - Low poke
    attacks[AttackType.DOWN_TILT] = AttackData(
        attack_type=AttackType.DOWN_TILT,
        total_frames=22,
        hitboxes=[Hitbox(
            offset=Vector2(18, -18),
            width=55,
            height=22,
            damage=5.0,
            knockback_base=2.5,
            knockback_scaling=0.05,
            angle=45,
            active_frames=(6, 9)
        )],
        startup_frames=6,
        recovery_frames=13
    )

    # FORWARD SMASH - Energy blast (ZONE's primary KO move, tuned for ~120%)
    attacks[AttackType.FORWARD_SMASH] = AttackData(
        attack_type=AttackType.FORWARD_SMASH,
        total_frames=48,
        hitboxes=[Hitbox(
            offset=Vector2(30, -55),
            width=75,  # Long range
            height=45,
            damage=16.0,          # Higher damage for zoner's commitment
            knockback_base=7.5,   # Strong base knockback
            knockback_scaling=0.12,
            angle=32,
            active_frames=(14, 17)
        )],
        startup_frames=14,
        recovery_frames=31        # Slightly more punishable
    )

    # UP SMASH
    attacks[AttackType.UP_SMASH] = AttackData(
        attack_type=AttackType.UP_SMASH,
        total_frames=42,
        hitboxes=[Hitbox(
            offset=Vector2(-22, -100),
            width=60,
            height=50,
            damage=14.0,
            knockback_base=7.5,
            knockback_scaling=0.14,
            angle=90,
            active_frames=(12, 16)
        )],
        startup_frames=12,
        recovery_frames=26
    )

    # DOWN SMASH
    attacks[AttackType.DOWN_SMASH] = AttackData(
        attack_type=AttackType.DOWN_SMASH,
        total_frames=38,
        hitboxes=[
            Hitbox(
                offset=Vector2(20, -22),
                width=60,
                height=28,
                damage=12.0,
                knockback_base=5.5,
                knockback_scaling=0.11,
                angle=25,
                active_frames=(10, 13)
            ),
            Hitbox(
                offset=Vector2(-50, -22),
                width=60,
                height=28,
                damage=11.0,
                knockback_base=5.0,
                knockback_scaling=0.10,
                angle=155,
                active_frames=(14, 17)
            )
        ],
        startup_frames=10,
        recovery_frames=21
    )

    # NEUTRAL AIR
    attacks[AttackType.NEUTRAL_AIR] = AttackData(
        attack_type=AttackType.NEUTRAL_AIR,
        total_frames=30,
        hitboxes=[Hitbox(
            offset=Vector2(-25, -58),
            width=70,
            height=60,
            damage=6.0,
            knockback_base=3.0,
            knockback_scaling=0.06,
            angle=48,
            active_frames=(6, 14)
        )],
        startup_frames=6,
        recovery_frames=16,
        landing_lag=8
    )

    # FORWARD AIR - Long range aerial
    attacks[AttackType.FORWARD_AIR] = AttackData(
        attack_type=AttackType.FORWARD_AIR,
        total_frames=32,
        hitboxes=[Hitbox(
            offset=Vector2(22, -52),
            width=60,  # Long range
            height=38,
            damage=9.0,
            knockback_base=4.5,
            knockback_scaling=0.09,
            angle=38,
            active_frames=(8, 12)
        )],
        startup_frames=8,
        recovery_frames=20,
        landing_lag=11
    )

    # BACK AIR
    attacks[AttackType.BACK_AIR] = AttackData(
        attack_type=AttackType.BACK_AIR,
        total_frames=30,
        hitboxes=[Hitbox(
            offset=Vector2(-28, -52),
            width=50,
            height=38,
            damage=11.0,
            knockback_base=5.5,
            knockback_scaling=0.11,
            angle=140,
            active_frames=(9, 13)
        )],
        startup_frames=9,
        recovery_frames=17,
        landing_lag=10
    )

    # UP AIR
    attacks[AttackType.UP_AIR] = AttackData(
        attack_type=AttackType.UP_AIR,
        total_frames=28,
        hitboxes=[Hitbox(
            offset=Vector2(-20, -88),
            width=58,
            height=42,
            damage=8.0,
            knockback_base=4.5,
            knockback_scaling=0.09,
            angle=85,
            active_frames=(7, 11)
        )],
        startup_frames=7,
        recovery_frames=17,
        landing_lag=9
    )

    # DOWN AIR - Spike
    attacks[AttackType.DOWN_AIR] = AttackData(
        attack_type=AttackType.DOWN_AIR,
        total_frames=35,
        hitboxes=[Hitbox(
            offset=Vector2(-18, -18),
            width=45,
            height=40,
            damage=10.0,
            knockback_base=4.5,
            knockback_scaling=0.09,
            angle=268,  # Spike
            active_frames=(12, 15)
        )],
        startup_frames=12,
        recovery_frames=20,
        landing_lag=16
    )

    # NEUTRAL SPECIAL - Energy Projectile (ZONE's primary zoning tool)
    # Spawns a projectile that travels horizontally
    attacks[AttackType.NEUTRAL_SPECIAL] = AttackData(
        attack_type=AttackType.NEUTRAL_SPECIAL,
        total_frames=38,          # Slightly faster
        hitboxes=[Hitbox(
            offset=Vector2(50, -55),  # Projectile spawn point
            width=30,
            height=30,
            damage=7.0,           # Moderate damage for zoning
            knockback_base=3.5,   # Low knockback (chip damage, not kill move)
            knockback_scaling=0.06,
            angle=30,
            active_frames=(12, 22)
        )],
        startup_frames=12,        # Faster startup for zoning
        recovery_frames=14,
        spawns_projectile=True    # Indicates this spawns a projectile
    )

    # SIDE SPECIAL - Horizontal Beam (slower startup, higher reward)
    attacks[AttackType.SIDE_SPECIAL] = AttackData(
        attack_type=AttackType.SIDE_SPECIAL,
        total_frames=50,
        hitboxes=[Hitbox(
            offset=Vector2(40, -50),
            width=120,  # Very long range beam
            height=25,
            damage=11.0,          # Higher reward for commitment
            knockback_base=5.0,   # Can set up at mid %
            knockback_scaling=0.08,
            angle=20,
            active_frames=(15, 25)  # Long active
        )],
        startup_frames=15,        # More telegraphed
        recovery_frames=25,
        spawns_projectile=True    # Beam is a projectile
    )

    # UP SPECIAL - Teleport (recovery, no hitbox)
    attacks[AttackType.UP_SPECIAL] = AttackData(
        attack_type=AttackType.UP_SPECIAL,
        total_frames=35,
        hitboxes=[],  # No hitbox - pure recovery
        startup_frames=10,
        recovery_frames=25
    )

    # DOWN SPECIAL - Trap Placement
    attacks[AttackType.DOWN_SPECIAL] = AttackData(
        attack_type=AttackType.DOWN_SPECIAL,
        total_frames=50,
        hitboxes=[Hitbox(
            offset=Vector2(30, -10),  # Trap at feet level
            width=50,
            height=30,
            damage=6.0,
            knockback_base=5.0,
            knockback_scaling=0.08,
            angle=70,
            active_frames=(25, 50)  # Stays active (trap)
        )],
        startup_frames=25,
        recovery_frames=25
    )

    return attacks


def create_grab_attacks() -> Dict[AttackType, AttackData]:
    """Create attack data for GRAB character.

    GRAB focuses on:
    - Slow startup but high damage
    - Command grabs
    - Super armor on some moves
    - Very high knockback
    """
    attacks = {}

    # JAB - Slow but strong
    attacks[AttackType.JAB] = AttackData(
        attack_type=AttackType.JAB,
        total_frames=24,
        hitboxes=[Hitbox(
            offset=Vector2(22, -50),
            width=45,
            height=35,
            damage=5.0,
            knockback_base=3.0,
            knockback_scaling=0.06,
            angle=45,
            active_frames=(8, 12)  # Slow startup
        )],
        startup_frames=8,
        recovery_frames=12
    )

    # FORWARD TILT - Heavy swing
    attacks[AttackType.FORWARD_TILT] = AttackData(
        attack_type=AttackType.FORWARD_TILT,
        total_frames=35,
        hitboxes=[Hitbox(
            offset=Vector2(25, -55),
            width=65,
            height=45,
            damage=12.0,  # High damage
            knockback_base=5.0,
            knockback_scaling=0.10,
            angle=35,
            active_frames=(12, 16)
        )],
        startup_frames=12,
        recovery_frames=19
    )

    # UP TILT - Headbutt
    attacks[AttackType.UP_TILT] = AttackData(
        attack_type=AttackType.UP_TILT,
        total_frames=32,
        hitboxes=[Hitbox(
            offset=Vector2(-15, -95),
            width=60,
            height=50,
            damage=10.0,
            knockback_base=6.0,
            knockback_scaling=0.11,
            angle=88,
            active_frames=(10, 14)
        )],
        startup_frames=10,
        recovery_frames=18
    )

    # DOWN TILT - Sweep
    attacks[AttackType.DOWN_TILT] = AttackData(
        attack_type=AttackType.DOWN_TILT,
        total_frames=30,
        hitboxes=[Hitbox(
            offset=Vector2(18, -20),
            width=60,
            height=28,
            damage=9.0,
            knockback_base=3.5,
            knockback_scaling=0.07,
            angle=30,
            active_frames=(10, 14)
        )],
        startup_frames=10,
        recovery_frames=16
    )

    # FORWARD SMASH - Crushing blow (GRAB's signature kill move)
    # Slowest smash in game but most powerful - KOs at ~100% from center
    attacks[AttackType.FORWARD_SMASH] = AttackData(
        attack_type=AttackType.FORWARD_SMASH,
        total_frames=58,
        hitboxes=[Hitbox(
            offset=Vector2(28, -58),
            width=70,
            height=55,
            damage=24.0,  # MASSIVE damage - reward for landing slow attack
            knockback_base=10.0,  # Very high base for early KOs
            knockback_scaling=0.14,
            angle=30,
            active_frames=(20, 24)  # 20 frame startup is very slow
        )],
        startup_frames=20,        # Very slow but devastating
        recovery_frames=34
    )

    # UP SMASH - Headbutt smash (GRAB's anti-air KO option)
    attacks[AttackType.UP_SMASH] = AttackData(
        attack_type=AttackType.UP_SMASH,
        total_frames=52,
        hitboxes=[Hitbox(
            offset=Vector2(-25, -105),
            width=70,
            height=55,
            damage=22.0,          # High damage
            knockback_base=9.0,   # Strong vertical knockback
            knockback_scaling=0.14,
            angle=90,
            active_frames=(18, 22)
        )],
        startup_frames=18,
        recovery_frames=30
    )

    # DOWN SMASH - Ground slam
    attacks[AttackType.DOWN_SMASH] = AttackData(
        attack_type=AttackType.DOWN_SMASH,
        total_frames=48,
        hitboxes=[
            Hitbox(
                offset=Vector2(25, -25),
                width=70,
                height=35,
                damage=18.0,
                knockback_base=7.0,
                knockback_scaling=0.13,
                angle=25,
                active_frames=(14, 18)
            ),
            Hitbox(
                offset=Vector2(-55, -25),
                width=70,
                height=35,
                damage=16.0,
                knockback_base=6.5,
                knockback_scaling=0.12,
                angle=155,
                active_frames=(18, 22)
            )
        ],
        startup_frames=14,
        recovery_frames=26
    )

    # NEUTRAL AIR - Body slam
    attacks[AttackType.NEUTRAL_AIR] = AttackData(
        attack_type=AttackType.NEUTRAL_AIR,
        total_frames=36,
        hitboxes=[Hitbox(
            offset=Vector2(-30, -60),
            width=80,
            height=70,
            damage=10.0,
            knockback_base=4.5,
            knockback_scaling=0.08,
            angle=50,
            active_frames=(8, 18)
        )],
        startup_frames=8,
        recovery_frames=18,
        landing_lag=14
    )

    # FORWARD AIR - Hammer fist
    attacks[AttackType.FORWARD_AIR] = AttackData(
        attack_type=AttackType.FORWARD_AIR,
        total_frames=38,
        hitboxes=[Hitbox(
            offset=Vector2(25, -55),
            width=55,
            height=50,
            damage=14.0,
            knockback_base=6.0,
            knockback_scaling=0.12,
            angle=32,
            active_frames=(12, 16)
        )],
        startup_frames=12,
        recovery_frames=22,
        landing_lag=16
    )

    # BACK AIR - Elbow
    attacks[AttackType.BACK_AIR] = AttackData(
        attack_type=AttackType.BACK_AIR,
        total_frames=35,
        hitboxes=[Hitbox(
            offset=Vector2(-30, -55),
            width=50,
            height=45,
            damage=15.0,
            knockback_base=7.0,
            knockback_scaling=0.13,
            angle=145,
            active_frames=(11, 15)
        )],
        startup_frames=11,
        recovery_frames=20,
        landing_lag=14
    )

    # UP AIR - Overhead swing
    attacks[AttackType.UP_AIR] = AttackData(
        attack_type=AttackType.UP_AIR,
        total_frames=32,
        hitboxes=[Hitbox(
            offset=Vector2(-22, -92),
            width=65,
            height=48,
            damage=12.0,
            knockback_base=5.5,
            knockback_scaling=0.10,
            angle=85,
            active_frames=(10, 14)
        )],
        startup_frames=10,
        recovery_frames=18,
        landing_lag=12
    )

    # DOWN AIR - Ground pound (huge spike)
    attacks[AttackType.DOWN_AIR] = AttackData(
        attack_type=AttackType.DOWN_AIR,
        total_frames=42,
        hitboxes=[Hitbox(
            offset=Vector2(-20, -15),
            width=55,
            height=45,
            damage=16.0,  # Big damage
            knockback_base=6.5,
            knockback_scaling=0.12,
            angle=270,  # Pure spike
            active_frames=(14, 18)
        )],
        startup_frames=14,
        recovery_frames=24,
        landing_lag=20
    )

    # NEUTRAL SPECIAL - Armored Charge (super armor during active)
    attacks[AttackType.NEUTRAL_SPECIAL] = AttackData(
        attack_type=AttackType.NEUTRAL_SPECIAL,
        total_frames=55,
        hitboxes=[Hitbox(
            offset=Vector2(35, -55),
            width=70,
            height=60,
            damage=15.0,
            knockback_base=6.0,
            knockback_scaling=0.11,
            angle=40,
            active_frames=(15, 35)  # Long active with armor
        )],
        startup_frames=15,
        recovery_frames=20,
        super_armor=True  # Has super armor during active frames
    )

    # SIDE SPECIAL - Command Grab (GRAB's defining move)
    # Slow startup but massive payoff - best raw damage of any single hit
    attacks[AttackType.SIDE_SPECIAL] = AttackData(
        attack_type=AttackType.SIDE_SPECIAL,
        total_frames=55,
        hitboxes=[Hitbox(
            offset=Vector2(30, -50),
            width=55,             # Slightly smaller hitbox for balance
            height=50,
            damage=20.0,          # HUGE damage for grabs
            knockback_base=8.5,
            knockback_scaling=0.12,
            angle=50,             # Sends at good combo/follow-up angle at low %
            active_frames=(14, 20)
        )],
        startup_frames=14,        # Slow, reactable startup
        recovery_frames=35        # Very punishable on whiff
    )

    # UP SPECIAL - Spinning Rise (recovery)
    attacks[AttackType.UP_SPECIAL] = AttackData(
        attack_type=AttackType.UP_SPECIAL,
        total_frames=52,
        hitboxes=[
            Hitbox(
                offset=Vector2(-30, -70),
                width=80,
                height=60,
                damage=4.0,
                knockback_base=2.0,
                knockback_scaling=0.04,
                angle=70,
                active_frames=(8, 15)
            ),
            Hitbox(
                offset=Vector2(-30, -85),
                width=80,
                height=60,
                damage=4.0,
                knockback_base=2.0,
                knockback_scaling=0.04,
                angle=75,
                active_frames=(16, 23)
            ),
            Hitbox(  # Final hit
                offset=Vector2(-25, -90),
                width=70,
                height=55,
                damage=8.0,
                knockback_base=5.0,
                knockback_scaling=0.09,
                angle=80,
                active_frames=(24, 28)
            )
        ],
        startup_frames=8,
        recovery_frames=24
    )

    # DOWN SPECIAL - Ground Pound (comes down hard)
    attacks[AttackType.DOWN_SPECIAL] = AttackData(
        attack_type=AttackType.DOWN_SPECIAL,
        total_frames=60,
        hitboxes=[Hitbox(
            offset=Vector2(-35, -20),
            width=90,
            height=40,
            damage=20.0,
            knockback_base=8.0,
            knockback_scaling=0.14,
            angle=60,
            active_frames=(25, 32)
        )],
        startup_frames=25,
        recovery_frames=28
    )

    return attacks


def create_flex_attacks() -> Dict[AttackType, AttackData]:
    """Create attack data for FLEX character.

    FLEX focuses on:
    - Balanced frame data
    - Solid fundamentals
    - No major weaknesses
    - Standard projectile
    """
    attacks = {}

    # JAB - Standard
    attacks[AttackType.JAB] = AttackData(
        attack_type=AttackType.JAB,
        total_frames=20,
        hitboxes=[Hitbox(
            offset=Vector2(20, -50),
            width=40,
            height=30,
            damage=3.0,
            knockback_base=2.0,
            knockback_scaling=0.05,
            angle=45,
            active_frames=(5, 8)
        )],
        startup_frames=5,
        recovery_frames=12
    )

    # FORWARD TILT - Balanced
    attacks[AttackType.FORWARD_TILT] = AttackData(
        attack_type=AttackType.FORWARD_TILT,
        total_frames=28,
        hitboxes=[Hitbox(
            offset=Vector2(22, -52),
            width=55,
            height=35,
            damage=8.0,
            knockback_base=4.0,
            knockback_scaling=0.08,
            angle=38,
            active_frames=(7, 11)
        )],
        startup_frames=7,
        recovery_frames=17
    )

    # UP TILT - Good juggle
    attacks[AttackType.UP_TILT] = AttackData(
        attack_type=AttackType.UP_TILT,
        total_frames=26,
        hitboxes=[Hitbox(
            offset=Vector2(-18, -88),
            width=55,
            height=42,
            damage=7.0,
            knockback_base=5.0,
            knockback_scaling=0.09,
            angle=86,
            active_frames=(6, 10)
        )],
        startup_frames=6,
        recovery_frames=16
    )

    # DOWN TILT - Trip attack
    attacks[AttackType.DOWN_TILT] = AttackData(
        attack_type=AttackType.DOWN_TILT,
        total_frames=24,
        hitboxes=[Hitbox(
            offset=Vector2(16, -18),
            width=50,
            height=24,
            damage=6.0,
            knockback_base=3.0,
            knockback_scaling=0.06,
            angle=75,  # Pops up
            active_frames=(6, 9)
        )],
        startup_frames=6,
        recovery_frames=15
    )

    # FORWARD SMASH - Standard (FLEX's reliable KO option, ~120% KO threshold)
    attacks[AttackType.FORWARD_SMASH] = AttackData(
        attack_type=AttackType.FORWARD_SMASH,
        total_frames=44,
        hitboxes=[Hitbox(
            offset=Vector2(26, -56),
            width=62,
            height=45,
            damage=17.0,          # Balanced damage
            knockback_base=7.5,   # Reliable KO power
            knockback_scaling=0.12,
            angle=35,
            active_frames=(13, 16)
        )],
        startup_frames=13,
        recovery_frames=28
    )

    # UP SMASH
    attacks[AttackType.UP_SMASH] = AttackData(
        attack_type=AttackType.UP_SMASH,
        total_frames=40,
        hitboxes=[Hitbox(
            offset=Vector2(-22, -98),
            width=58,
            height=48,
            damage=15.0,
            knockback_base=7.5,
            knockback_scaling=0.14,
            angle=88,
            active_frames=(11, 15)
        )],
        startup_frames=11,
        recovery_frames=25
    )

    # DOWN SMASH
    attacks[AttackType.DOWN_SMASH] = AttackData(
        attack_type=AttackType.DOWN_SMASH,
        total_frames=38,
        hitboxes=[
            Hitbox(
                offset=Vector2(18, -22),
                width=55,
                height=28,
                damage=13.0,
                knockback_base=5.5,
                knockback_scaling=0.11,
                angle=28,
                active_frames=(9, 12)
            ),
            Hitbox(
                offset=Vector2(-48, -22),
                width=55,
                height=28,
                damage=12.0,
                knockback_base=5.0,
                knockback_scaling=0.10,
                angle=152,
                active_frames=(13, 16)
            )
        ],
        startup_frames=9,
        recovery_frames=22
    )

    # NEUTRAL AIR - Sex kick
    attacks[AttackType.NEUTRAL_AIR] = AttackData(
        attack_type=AttackType.NEUTRAL_AIR,
        total_frames=30,
        hitboxes=[Hitbox(
            offset=Vector2(-24, -58),
            width=68,
            height=58,
            damage=7.0,
            knockback_base=3.5,
            knockback_scaling=0.07,
            angle=46,
            active_frames=(5, 16)
        )],
        startup_frames=5,
        recovery_frames=14,
        landing_lag=9
    )

    # FORWARD AIR - Kick
    attacks[AttackType.FORWARD_AIR] = AttackData(
        attack_type=AttackType.FORWARD_AIR,
        total_frames=32,
        hitboxes=[Hitbox(
            offset=Vector2(22, -52),
            width=52,
            height=38,
            damage=10.0,
            knockback_base=4.5,
            knockback_scaling=0.09,
            angle=36,
            active_frames=(8, 12)
        )],
        startup_frames=8,
        recovery_frames=20,
        landing_lag=11
    )

    # BACK AIR - Strong back kick
    attacks[AttackType.BACK_AIR] = AttackData(
        attack_type=AttackType.BACK_AIR,
        total_frames=34,
        hitboxes=[Hitbox(
            offset=Vector2(-28, -54),
            width=50,
            height=38,
            damage=13.0,
            knockback_base=6.0,
            knockback_scaling=0.12,
            angle=142,
            active_frames=(10, 14)
        )],
        startup_frames=10,
        recovery_frames=20,
        landing_lag=12
    )

    # UP AIR
    attacks[AttackType.UP_AIR] = AttackData(
        attack_type=AttackType.UP_AIR,
        total_frames=28,
        hitboxes=[Hitbox(
            offset=Vector2(-20, -86),
            width=56,
            height=42,
            damage=9.0,
            knockback_base=4.5,
            knockback_scaling=0.09,
            angle=84,
            active_frames=(6, 10)
        )],
        startup_frames=6,
        recovery_frames=18,
        landing_lag=10
    )

    # DOWN AIR - Stomp
    attacks[AttackType.DOWN_AIR] = AttackData(
        attack_type=AttackType.DOWN_AIR,
        total_frames=34,
        hitboxes=[Hitbox(
            offset=Vector2(-16, -18),
            width=42,
            height=38,
            damage=12.0,
            knockback_base=5.0,
            knockback_scaling=0.10,
            angle=270,  # Spike
            active_frames=(11, 14)
        )],
        startup_frames=11,
        recovery_frames=20,
        landing_lag=15
    )

    # NEUTRAL SPECIAL - Fireball (standard projectile)
    attacks[AttackType.NEUTRAL_SPECIAL] = AttackData(
        attack_type=AttackType.NEUTRAL_SPECIAL,
        total_frames=38,
        hitboxes=[Hitbox(
            offset=Vector2(45, -50),
            width=32,
            height=32,
            damage=6.0,
            knockback_base=3.0,
            knockback_scaling=0.05,
            angle=30,
            active_frames=(12, 25)
        )],
        startup_frames=12,
        recovery_frames=13
    )

    # SIDE SPECIAL - Spinning Kick
    attacks[AttackType.SIDE_SPECIAL] = AttackData(
        attack_type=AttackType.SIDE_SPECIAL,
        total_frames=40,
        hitboxes=[
            Hitbox(
                offset=Vector2(25, -55),
                width=50,
                height=45,
                damage=5.0,
                knockback_base=2.0,
                knockback_scaling=0.04,
                angle=50,
                active_frames=(8, 12)
            ),
            Hitbox(
                offset=Vector2(30, -55),
                width=55,
                height=48,
                damage=8.0,
                knockback_base=5.0,
                knockback_scaling=0.09,
                angle=35,
                active_frames=(18, 22)
            )
        ],
        startup_frames=8,
        recovery_frames=18
    )

    # UP SPECIAL - Rising Slash (recovery)
    attacks[AttackType.UP_SPECIAL] = AttackData(
        attack_type=AttackType.UP_SPECIAL,
        total_frames=45,
        hitboxes=[Hitbox(
            offset=Vector2(8, -72),
            width=48,
            height=58,
            damage=11.0,
            knockback_base=5.5,
            knockback_scaling=0.10,
            angle=78,
            active_frames=(6, 14)
        )],
        startup_frames=6,
        recovery_frames=31
    )

    # DOWN SPECIAL - Reflector (FLEX's signature utility move)
    # 7-frame reflect window (frames 3-10), fast startup but punishable recovery
    attacks[AttackType.DOWN_SPECIAL] = AttackData(
        attack_type=AttackType.DOWN_SPECIAL,
        total_frames=32,
        hitboxes=[Hitbox(
            offset=Vector2(-25, -60),
            width=70,
            height=60,
            damage=5.0,           # Weak melee hitbox
            knockback_base=4.5,   # Mostly for spacing
            knockback_scaling=0.06,
            angle=50,
            active_frames=(3, 10)  # 7-frame reflector window as specified
        )],
        startup_frames=3,         # Fast startup for reactions
        recovery_frames=22,       # Punishable if whiffed
        is_reflector=True         # This is a reflector move
    )

    return attacks


# =============================================================================
# CHARACTER DATA REGISTRY
# =============================================================================

CHARACTER_STATS: Dict[CharacterType, FighterStats] = {
    CharacterType.RUSH: RUSH_STATS,
    CharacterType.ZONE: ZONE_STATS,
    CharacterType.GRAB: GRAB_STATS,
    CharacterType.FLEX: FLEX_STATS,
}

# Lazy-loaded attack data
_character_attacks: Dict[CharacterType, Dict[AttackType, AttackData]] = {}


def get_character_attacks(character: CharacterType) -> Dict[AttackType, AttackData]:
    """Get attack data for a character (lazy loaded)."""
    if character not in _character_attacks:
        if character == CharacterType.RUSH:
            _character_attacks[character] = create_rush_attacks()
        elif character == CharacterType.ZONE:
            _character_attacks[character] = create_zone_attacks()
        elif character == CharacterType.GRAB:
            _character_attacks[character] = create_grab_attacks()
        elif character == CharacterType.FLEX:
            _character_attacks[character] = create_flex_attacks()
    return _character_attacks[character]


def get_character_stats(character: CharacterType) -> FighterStats:
    """Get stats for a character type."""
    return CHARACTER_STATS[character]


def get_character_color(character: CharacterType) -> Tuple[int, int, int]:
    """Get the RGB color for a character."""
    return CHARACTER_COLORS[character]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def list_characters() -> List[CharacterType]:
    """List all available characters."""
    return list(CharacterType)


def get_character_by_name(name: str) -> Optional[CharacterType]:
    """Get character type by name (case insensitive)."""
    name_upper = name.upper()
    for char in CharacterType:
        if char.name == name_upper:
            return char
    return None


if __name__ == "__main__":
    # Test character data
    print("Available characters:")
    for char in list_characters():
        stats = get_character_stats(char)
        color = get_character_color(char)
        attacks = get_character_attacks(char)
        print(f"\n{char.name}:")
        print(f"  Stats: speed={stats.walk_speed}/{stats.run_speed}, "
              f"weight={stats.weight}, jump={stats.jump_force}")
        print(f"  Color: RGB{color}")
        print(f"  Attacks: {len(attacks)} moves defined")

        # Show attack startup frames
        print("  Key moves:")
        for attack_type in [AttackType.JAB, AttackType.FORWARD_SMASH,
                           AttackType.NEUTRAL_SPECIAL, AttackType.UP_SPECIAL]:
            if attack_type in attacks:
                attack = attacks[attack_type]
                print(f"    {attack_type.name}: startup={attack.startup_frames}f, "
                      f"total={attack.total_frames}f")
