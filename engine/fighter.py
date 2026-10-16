"""Fighter entity."""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from .combat import AttackData, AttackType, Hitbox, Hurtbox
from .config import CONFIG, GameConfig
from .geometry import AABB, Vector2
from .input import InputState


class FighterState(Enum):
    """States a fighter can be in."""
    IDLE = auto()
    WALKING = auto()
    RUNNING = auto()
    JUMPING = auto()
    FALLING = auto()
    ATTACKING = auto()
    HITSTUN = auto()
    SHIELDING = auto()
    GRABBING = auto()
    GRABBED = auto()
    LEDGE = auto()
    SPECIAL = auto()
    DODGE = auto()
    LANDING = auto()
    CROUCH = auto()
    DODGING = auto()         # Active dodge (spot dodge, roll, air dodge)
    SHIELD_STUN = auto()     # Shield broken stun
    COUNTER = auto()         # Counter stance (move with is_counter)
    REFLECTING = auto()      # Reflector active (move with is_reflector)
    ENTERING = auto()        # Entrance animation (driven by the game, not the sim)


class DodgeType(Enum):
    """Types of dodge moves."""
    SPOT_DODGE = auto()      # Brief invincibility in place
    ROLL_FORWARD = auto()    # Invincible roll forward
    ROLL_BACKWARD = auto()   # Invincible roll backward
    AIR_DODGE = auto()       # One per airborne state


@dataclass
class FighterStats:
    """Base stats and moveset for a fighter type."""
    name: str
    walk_speed: float = 3.0
    run_speed: float = 6.0
    air_speed: float = 4.0
    jump_force: float = 14.0
    double_jump_force: float = 12.0
    weight: float = 100.0  # Affects knockback received
    fall_speed: float = 0.6
    fast_fall_speed: float = 1.2

    # Collision dimensions
    width: float = 50
    height: float = 80

    # Opaque tag for game code (e.g. a character name). The engine never reads it.
    character_type: Optional[str] = None

    # Moves for this fighter. Slots missing here fall back to the GameLoop's
    # fallback_attack (engine.combat.default_attack unless overridden).
    moveset: Optional[Dict[AttackType, AttackData]] = field(
        default=None, repr=False, compare=False)


class Fighter:
    """A fighter in the game."""

    def __init__(self, player_id: int, stats: FighterStats, spawn_pos: Vector2,
                 color: Optional[Tuple[int, int, int]] = None,
                 config: Optional[GameConfig] = None):
        self.player_id = player_id
        self.stats = stats
        self.config = config or CONFIG

        # Transform
        self.position = spawn_pos.copy()
        self.velocity = Vector2(0, 0)
        self.facing_right = player_id == 0  # P1 faces right, P2 faces left

        # State
        self.state = FighterState.IDLE
        self.state_frame = 0
        self.grounded = True
        self.can_double_jump = True

        # Combat
        self.damage_percent = 0.0
        self.stocks = self.config.stock_count
        self.current_attack: Optional[AttackData] = None
        self.attack_frame = 0

        # Hitstun
        self.hitstun_frames = 0
        self.knockback_vector = Vector2(0, 0)

        # Shield
        self.shield_health = 100.0
        self.shield_stun = 0
        self.shield_regen_delay = 0  # Frames until shield starts regenerating
        self.shield_broken = False

        # Invincibility
        self.invincible_frames = 0

        # Dodge mechanics
        self.air_dodge_available = True  # Reset when grounded
        self.dodge_type: Optional[DodgeType] = None
        self.dodge_frames_remaining = 0
        self.dodge_velocity = 0.0

        # Counter mechanics
        self.counter_active = False
        self.counter_window_frames = 0  # Frames of counter window remaining
        self.counter_success = False    # True if counter was triggered

        # Reflector mechanics
        self.reflector_active = False

        # Super armor
        self.super_armor_active = False

        # Input
        self.input = InputState()

        # Hurtbox (vulnerable area). hurtbox.bottom == position.y in sim coordinates.
        self.hurtbox = Hurtbox(
            offset=Vector2(-stats.width / 2, -stats.height),
            width=stats.width,
            height=stats.height
        )

        # Active hitboxes (set during attacks)
        self.active_hitboxes: List[Hitbox] = []

        # Color for rendering (RGB)
        if color is not None:
            self.color = color
        else:
            self.color = (255, 100, 100) if player_id == 0 else (100, 100, 255)

    def get_hurtbox_aabb(self) -> AABB:
        """Get the current hurtbox as an AABB."""
        return self.hurtbox.get_aabb(self.position)

    def get_attack_data(self, attack_type: AttackType) -> Optional[AttackData]:
        """Get this fighter's move for a slot, or None if its moveset lacks it."""
        moveset = self.stats.moveset
        if moveset and attack_type in moveset:
            return moveset[attack_type]
        return None

    def get_active_hitbox_aabbs(self) -> List[AABB]:
        """Get AABBs for all currently active hitboxes."""
        return [hb.get_aabb(self.position, self.facing_right)
                for hb in self.active_hitboxes]

    def apply_knockback(self, knockback: Vector2, damage: float):
        """Apply knockback to this fighter."""
        if self.invincible_frames > 0:
            return

        self.damage_percent += damage

        # Knockback grows with accumulated damage; lighter fighters fly further
        kb_multiplier = 1.0 + (self.damage_percent * self.config.knockback_scaling)
        weight_mult = 200.0 / (self.stats.weight + 100)

        scaled_kb = knockback * kb_multiplier * weight_mult
        self.knockback_vector = scaled_kb
        self.velocity = scaled_kb.copy()

        self.hitstun_frames = int(scaled_kb.magnitude() * self.config.hitstun_multiplier)
        self.state = FighterState.HITSTUN
        self.state_frame = 0
        self.grounded = False

        # Cancel any current attack
        self.current_attack = None
        self.active_hitboxes = []

    def respawn(self, spawn_pos: Vector2):
        """Respawn the fighter after losing a stock."""
        self.stocks -= 1
        self.position = spawn_pos.copy()
        self.velocity = Vector2(0, 0)
        self.damage_percent = 0.0
        self.state = FighterState.FALLING
        self.state_frame = 0
        self.grounded = False
        self.can_double_jump = True
        self.hitstun_frames = 0
        self.invincible_frames = 120  # 2 seconds of invincibility
        self.current_attack = None
        self.active_hitboxes = []

    def is_dead(self) -> bool:
        """Check if fighter has been knocked out."""
        return self.stocks <= 0
