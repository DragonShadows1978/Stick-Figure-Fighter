"""
Stick Figure Fighter - Core Game Engine

A platform fighter built entirely from first principles.
No external game engines - pure Python with X11 rendering.

This module contains:
- Game loop architecture (fixed timestep simulation)
- Entity component system for fighters
- Physics constants and configuration
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum, auto
import numpy as np

# Forward declaration for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from characters import CharacterType


# =============================================================================
# GAME CONFIGURATION
# =============================================================================

@dataclass
class GameConfig:
    """Global game configuration."""
    # Display
    screen_width: int = 1280
    screen_height: int = 720
    target_fps: int = 60

    # Physics
    gravity: float = 0.6
    terminal_velocity: float = 18.0
    friction: float = 0.85
    air_friction: float = 0.95

    # Combat
    base_knockback: float = 4.0
    knockback_scaling: float = 0.12
    hitstun_multiplier: float = 0.4  # frames of hitstun per knockback unit

    # Game rules
    stock_count: int = 3

    # Stage bounds (blast zones)
    blast_zone_left: float = -200
    blast_zone_right: float = 1480  # screen_width + 200
    blast_zone_top: float = -200
    blast_zone_bottom: float = 920  # screen_height + 200

    @property
    def frame_time(self) -> float:
        """Time per frame in seconds."""
        return 1.0 / self.target_fps


# Global config instance
CONFIG = GameConfig()

# Rendering coordinate offset
# fighter.position.y represents the hip/center of the fighter
# Feet are rendered at position.y + FOOT_OFFSET
# Calculated from pose data: hip_y=-10, foot offset from hip=+50, total=40
FOOT_OFFSET = 40.0  # Distance from position.y to foot rendering position


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

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
    COUNTER = auto()         # Counter stance (RUSH down-B)
    REFLECTING = auto()      # Reflector active (FLEX down-B)
    ENTERING = auto()        # Cycle 14: Character entrance animation


class AttackType(Enum):
    """Types of attacks."""
    NONE = auto()
    JAB = auto()
    FORWARD_TILT = auto()
    UP_TILT = auto()
    DOWN_TILT = auto()
    FORWARD_SMASH = auto()
    UP_SMASH = auto()
    DOWN_SMASH = auto()
    NEUTRAL_AIR = auto()
    FORWARD_AIR = auto()
    BACK_AIR = auto()
    UP_AIR = auto()
    DOWN_AIR = auto()
    NEUTRAL_SPECIAL = auto()
    SIDE_SPECIAL = auto()
    UP_SPECIAL = auto()
    DOWN_SPECIAL = auto()
    GRAB = auto()
    THROW_FORWARD = auto()
    THROW_BACK = auto()
    THROW_UP = auto()
    THROW_DOWN = auto()


class InputAction(Enum):
    """Abstract input actions."""
    NONE = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    JUMP = auto()
    ATTACK = auto()
    SPECIAL = auto()
    SHIELD = auto()
    GRAB = auto()


class DodgeType(Enum):
    """Types of dodge moves."""
    SPOT_DODGE = auto()      # Brief invincibility in place
    ROLL_FORWARD = auto()    # Invincible roll forward
    ROLL_BACKWARD = auto()   # Invincible roll backward
    AIR_DODGE = auto()       # One per airborne state


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Vector2:
    """2D vector for positions and velocities."""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector2':
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> 'Vector2':
        return self * scalar

    def magnitude(self) -> float:
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self) -> 'Vector2':
        mag = self.magnitude()
        if mag == 0:
            return Vector2(0, 0)
        return Vector2(self.x / mag, self.y / mag)

    def copy(self) -> 'Vector2':
        return Vector2(self.x, self.y)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class AABB:
    """Axis-Aligned Bounding Box for collision detection."""
    x: float  # left edge
    y: float  # top edge
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Vector2:
        return Vector2(self.x + self.width / 2, self.y + self.height / 2)

    def intersects(self, other: 'AABB') -> bool:
        """Check if this AABB intersects with another."""
        return (self.left < other.right and
                self.right > other.left and
                self.top < other.bottom and
                self.bottom > other.top)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this AABB."""
        return (self.left <= x <= self.right and
                self.top <= y <= self.bottom)


@dataclass
class Hitbox:
    """Hitbox for attack collision detection."""
    offset: Vector2  # Offset from fighter position
    width: float
    height: float
    damage: float
    knockback_base: float
    knockback_scaling: float
    angle: float  # Knockback angle in degrees (0 = right, 90 = up)
    active_frames: Tuple[int, int]  # (start_frame, end_frame)

    def get_aabb(self, fighter_pos: Vector2, facing_right: bool) -> AABB:
        """Get the AABB for this hitbox given fighter position and facing."""
        x_offset = self.offset.x if facing_right else -self.offset.x - self.width
        return AABB(
            x=fighter_pos.x + x_offset,
            y=fighter_pos.y + self.offset.y,
            width=self.width,
            height=self.height
        )


@dataclass
class Hurtbox:
    """Hurtbox (vulnerable area) of a fighter."""
    offset: Vector2
    width: float
    height: float

    def get_aabb(self, fighter_pos: Vector2) -> AABB:
        """Get the AABB for this hurtbox given fighter position."""
        return AABB(
            x=fighter_pos.x + self.offset.x,
            y=fighter_pos.y + self.offset.y,
            width=self.width,
            height=self.height
        )


@dataclass
class AttackData:
    """Data for a single attack move."""
    attack_type: AttackType
    total_frames: int
    hitboxes: List[Hitbox]
    startup_frames: int  # Frames before hitbox becomes active
    recovery_frames: int  # Frames after hitbox ends before can act
    landing_lag: int = 0  # Extra landing lag during this move (aerials)
    spawns_projectile: bool = False  # True if this attack spawns a projectile
    super_armor: bool = False  # True if attack has super armor during active frames
    is_counter: bool = False  # True if this is a counter move
    is_reflector: bool = False  # True if this is a reflector move


@dataclass
class Projectile:
    """A projectile in the game world."""
    owner_id: int           # Player who spawned it
    position: Vector2       # Current position
    velocity: Vector2       # Movement per frame
    hitbox: Hitbox          # Damage/knockback data
    lifetime: int           # Frames remaining before despawn
    active: bool = True     # Whether it can still hit
    has_gravity: bool = False  # Whether gravity affects it (e.g., fireball)
    reflected: bool = False    # Whether it was reflected

    # Visual properties
    width: float = 20.0
    height: float = 20.0
    color: Tuple[int, int, int] = (255, 200, 50)  # Default yellow

    def get_aabb(self) -> AABB:
        """Get collision box at current position."""
        return AABB(
            x=self.position.x - self.width / 2,
            y=self.position.y - self.height / 2,
            width=self.width,
            height=self.height
        )

    def update(self, gravity: float = 0.0):
        """Update projectile for one frame."""
        if self.has_gravity:
            self.velocity.y += gravity * 0.5  # Slower gravity for projectiles

        self.position = self.position + self.velocity
        self.lifetime -= 1

        if self.lifetime <= 0:
            self.active = False

    def reflect(self):
        """Reflect this projectile (reverses direction and ownership)."""
        self.velocity.x = -self.velocity.x
        self.velocity.y = -self.velocity.y * 0.5  # Reduce vertical
        self.reflected = True
        # Swap owner to the reflector (handled by caller)
        # Boost damage slightly for reflected projectiles
        self.hitbox.damage *= 1.25


@dataclass
class Platform:
    """A platform in the stage."""
    x: float
    y: float
    width: float
    height: float = 10  # Default thin platform
    passthrough: bool = True  # Can drop through
    contact_offset: float = 5.0  # OGC: Offset contact zone for early detection (pixels above surface)

    def get_aabb(self) -> AABB:
        return AABB(self.x, self.y, self.width, self.height)

    @property
    def surface_y(self) -> float:
        """Y position of the top surface."""
        return self.y

    @property
    def contact_zone_y(self) -> float:
        """Y position of the contact detection zone (OGC offset volume)."""
        return self.y - self.contact_offset


@dataclass
class ParallaxLayer:
    """A parallax background layer."""
    depth: float  # 0.0 = foreground (moves with camera), 1.0 = background (doesn't move)
    elements: List[Dict]  # List of {'type': 'line'|'rect'|'circle', 'x': int, 'y': int, ...}
    color: Tuple[int, int, int] = (50, 50, 60)  # RGB color for this layer


@dataclass
class Stage:
    """A stage definition."""
    name: str
    platforms: List[Platform]
    main_platform: Platform  # The main ground platform
    blast_zones: Dict[str, float]  # left, right, top, bottom
    spawn_points: List[Vector2]  # Where fighters spawn
    parallax_layers: List[ParallaxLayer] = field(default_factory=list)  # Background layers

    @staticmethod
    def create_basic() -> 'Stage':
        """Create a basic Battlefield-style stage."""
        main = Platform(
            x=240, y=550, width=800, height=50, passthrough=False, contact_offset=5.0
        )
        left_plat = Platform(x=300, y=420, width=180, passthrough=True, contact_offset=5.0)
        right_plat = Platform(x=780, y=420, width=180, passthrough=True, contact_offset=5.0)
        top_plat = Platform(x=500, y=290, width=280, passthrough=True, contact_offset=5.0)

        # Create parallax background layers for Battlefield
        # Far background (mountains/skyline) - moves very slowly
        far_layer = ParallaxLayer(
            depth=0.9,  # Almost stationary
            color=(30, 35, 50),  # Dark blue-gray
            elements=[
                # Mountain silhouettes
                {'type': 'triangle', 'x': 100, 'y': 600, 'width': 400, 'height': 250},
                {'type': 'triangle', 'x': 400, 'y': 600, 'width': 350, 'height': 200},
                {'type': 'triangle', 'x': 650, 'y': 600, 'width': 450, 'height': 280},
                {'type': 'triangle', 'x': 950, 'y': 600, 'width': 380, 'height': 220},
            ]
        )

        # Mid background (trees/buildings) - moves at medium speed
        mid_layer = ParallaxLayer(
            depth=0.6,
            color=(40, 50, 45),  # Dark forest green
            elements=[
                # Tree-like shapes
                {'type': 'rect', 'x': 50, 'y': 480, 'width': 30, 'height': 120},
                {'type': 'rect', 'x': 150, 'y': 450, 'width': 40, 'height': 150},
                {'type': 'rect', 'x': 280, 'y': 470, 'width': 35, 'height': 130},
                {'type': 'rect', 'x': 900, 'y': 460, 'width': 35, 'height': 140},
                {'type': 'rect', 'x': 1050, 'y': 480, 'width': 40, 'height': 120},
                {'type': 'rect', 'x': 1180, 'y': 440, 'width': 45, 'height': 160},
            ]
        )

        # Near background (foliage/debris) - moves faster
        near_layer = ParallaxLayer(
            depth=0.3,
            color=(50, 60, 55),  # Lighter green
            elements=[
                # Foreground foliage
                {'type': 'circle', 'x': -20, 'y': 580, 'radius': 25},
                {'type': 'circle', 'x': 1300, 'y': 570, 'radius': 30},
                {'type': 'rect', 'x': -50, 'y': 550, 'width': 60, 'height': 100},
                {'type': 'rect', 'x': 1260, 'y': 540, 'width': 70, 'height': 110},
            ]
        )

        return Stage(
            name="Battlefield",
            platforms=[left_plat, right_plat, top_plat],
            main_platform=main,
            blast_zones={
                'left': CONFIG.blast_zone_left,
                'right': CONFIG.blast_zone_right,
                'top': CONFIG.blast_zone_top,
                'bottom': CONFIG.blast_zone_bottom
            },
            spawn_points=[
                Vector2(480, 500),
                Vector2(800, 500)
            ],
            parallax_layers=[far_layer, mid_layer, near_layer]
        )

    @staticmethod
    def create_final_destination() -> 'Stage':
        """Create a Final Destination-style stage (flat, no platforms)."""
        main = Platform(
            x=200, y=550, width=880, height=50, passthrough=False, contact_offset=5.0
        )

        # Create parallax background layers for Final Destination (space theme)
        # Far background (stars) - almost stationary
        star_layer = ParallaxLayer(
            depth=0.95,
            color=(255, 255, 255),  # White stars
            elements=[
                # Scattered stars
                {'type': 'circle', 'x': 50, 'y': 100, 'radius': 2},
                {'type': 'circle', 'x': 200, 'y': 80, 'radius': 1},
                {'type': 'circle', 'x': 350, 'y': 150, 'radius': 2},
                {'type': 'circle', 'x': 500, 'y': 50, 'radius': 1},
                {'type': 'circle', 'x': 650, 'y': 120, 'radius': 2},
                {'type': 'circle', 'x': 800, 'y': 60, 'radius': 1},
                {'type': 'circle', 'x': 950, 'y': 140, 'radius': 2},
                {'type': 'circle', 'x': 1100, 'y': 90, 'radius': 1},
                {'type': 'circle', 'x': 1200, 'y': 180, 'radius': 2},
                {'type': 'circle', 'x': 100, 'y': 250, 'radius': 1},
                {'type': 'circle', 'x': 300, 'y': 300, 'radius': 2},
                {'type': 'circle', 'x': 600, 'y': 220, 'radius': 1},
                {'type': 'circle', 'x': 900, 'y': 280, 'radius': 2},
                {'type': 'circle', 'x': 1150, 'y': 350, 'radius': 1},
            ]
        )

        # Mid background (nebula/cosmic clouds) - slow movement
        nebula_layer = ParallaxLayer(
            depth=0.7,
            color=(60, 40, 80),  # Purple nebula
            elements=[
                {'type': 'circle', 'x': 200, 'y': 300, 'radius': 80},
                {'type': 'circle', 'x': 800, 'y': 200, 'radius': 100},
                {'type': 'circle', 'x': 1100, 'y': 350, 'radius': 70},
            ]
        )

        # Near elements (floating debris) - medium movement
        debris_layer = ParallaxLayer(
            depth=0.4,
            color=(70, 70, 80),  # Gray debris
            elements=[
                {'type': 'rect', 'x': -30, 'y': 400, 'width': 40, 'height': 20},
                {'type': 'rect', 'x': 1280, 'y': 380, 'width': 35, 'height': 25},
                {'type': 'circle', 'x': -50, 'y': 300, 'radius': 15},
                {'type': 'circle', 'x': 1320, 'y': 280, 'radius': 12},
            ]
        )

        return Stage(
            name="Final Destination",
            platforms=[],
            main_platform=main,
            blast_zones={
                'left': CONFIG.blast_zone_left,
                'right': CONFIG.blast_zone_right,
                'top': CONFIG.blast_zone_top,
                'bottom': CONFIG.blast_zone_bottom
            },
            spawn_points=[
                Vector2(440, 500),
                Vector2(840, 500)
            ],
            parallax_layers=[star_layer, nebula_layer, debris_layer]
        )


# =============================================================================
# INPUT BUFFER
# =============================================================================

@dataclass
class InputState:
    """Current state of inputs for a player."""
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    jump: bool = False
    attack: bool = False
    special: bool = False
    shield: bool = False
    grab: bool = False

    # For detecting new presses (not held)
    _prev_jump: bool = False
    _prev_attack: bool = False
    _prev_special: bool = False
    _prev_shield: bool = False
    _prev_grab: bool = False

    @property
    def jump_pressed(self) -> bool:
        """True only on the frame jump was first pressed."""
        return self.jump and not self._prev_jump

    @property
    def attack_pressed(self) -> bool:
        return self.attack and not self._prev_attack

    @property
    def special_pressed(self) -> bool:
        return self.special and not self._prev_special

    @property
    def shield_pressed(self) -> bool:
        return self.shield and not self._prev_shield

    @property
    def grab_pressed(self) -> bool:
        return self.grab and not self._prev_grab

    def update_prev(self):
        """Call at end of frame to update previous state."""
        self._prev_jump = self.jump
        self._prev_attack = self.attack
        self._prev_special = self.special
        self._prev_shield = self.shield
        self._prev_grab = self.grab

    def clear(self):
        """Clear all inputs."""
        self.left = False
        self.right = False
        self.up = False
        self.down = False
        self.jump = False
        self.attack = False
        self.special = False
        self.shield = False
        self.grab = False


# =============================================================================
# FIGHTER CLASS
# =============================================================================

@dataclass
class FighterStats:
    """Base stats for a fighter type."""
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

    # Character type (for attack data lookup)
    character_type: Optional[str] = None  # CharacterType name


class Fighter:
    """A fighter in the game."""

    def __init__(self, player_id: int, stats: FighterStats, spawn_pos: Vector2,
                 color: Optional[Tuple[int, int, int]] = None):
        self.player_id = player_id
        self.stats = stats

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
        self.stocks = CONFIG.stock_count
        self.current_attack: Optional[AttackData] = None
        self.attack_frame = 0

        # Hitstun
        self.hitstun_frames = 0
        self.knockback_vector = Vector2(0, 0)

        # Shield
        self.shield_health = 100.0
        self.shield_stun = 0
        self.shield_regen_delay = 0  # Frames until shield starts regenerating
        self.shield_broken = False    # True if shield was broken

        # Invincibility
        self.invincible_frames = 0

        # Dodge mechanics
        self.air_dodge_available = True  # Reset when grounded
        self.dodge_type: Optional['DodgeType'] = None
        self.dodge_frames_remaining = 0
        self.dodge_velocity = 0.0  # For roll movement

        # Counter mechanics (RUSH down-B)
        self.counter_active = False
        self.counter_window_frames = 0  # Frames of counter window remaining
        self.counter_success = False    # True if counter was triggered

        # Reflector mechanics (FLEX down-B)
        self.reflector_active = False

        # Super armor (GRAB neutral-B)
        self.super_armor_active = False

        # Input
        self.input = InputState()

        # Hurtbox (vulnerable area)
        # Note: hurtbox.bottom == position.y (collision system coordinate)
        # Rendering draws feet at position.y + FOOT_OFFSET
        self.hurtbox = Hurtbox(
            offset=Vector2(-stats.width / 2, -stats.height),
            width=stats.width,
            height=stats.height
        )

        # Active hitboxes (set during attacks)
        self.active_hitboxes: List[Hitbox] = []

        # Color for rendering (RGB) - can be overridden by character
        if color is not None:
            self.color = color
        else:
            self.color = (255, 100, 100) if player_id == 0 else (100, 100, 255)

        # Character-specific attack data (lazy loaded)
        self._attack_data: Optional[Dict[AttackType, AttackData]] = None

    def get_hurtbox_aabb(self) -> AABB:
        """Get the current hurtbox as an AABB."""
        return self.hurtbox.get_aabb(self.position)

    def get_attack_data(self, attack_type: AttackType) -> Optional[AttackData]:
        """Get attack data for this fighter's character."""
        # Load character-specific attack data
        if self._attack_data is None and self.stats.character_type:
            try:
                from characters import get_character_attacks, CharacterType
                char_type = CharacterType[self.stats.character_type]
                self._attack_data = get_character_attacks(char_type)
            except (ImportError, KeyError):
                self._attack_data = {}

        # Return character-specific attack or None
        if self._attack_data and attack_type in self._attack_data:
            return self._attack_data[attack_type]
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

        # Calculate knockback scaling based on damage
        kb_multiplier = 1.0 + (self.damage_percent * CONFIG.knockback_scaling)
        # Weight affects knockback (lighter = more knockback)
        weight_mult = 200.0 / (self.stats.weight + 100)

        scaled_kb = knockback * kb_multiplier * weight_mult
        self.knockback_vector = scaled_kb
        self.velocity = scaled_kb.copy()

        # Calculate hitstun
        self.hitstun_frames = int(scaled_kb.magnitude() * CONFIG.hitstun_multiplier)
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


# =============================================================================
# GAME STATE
# =============================================================================

class GameState:
    """Complete game state for a match."""

    def __init__(self, stage: Stage, fighter_stats: List[FighterStats],
                 fighter_colors: Optional[List[Optional[Tuple[int, int, int]]]] = None,
                 time_limit: int = 0):  # Cycle 4: Optional time limit in frames (0 = no limit)
        self.stage = stage
        self.fighters: List[Fighter] = []
        self.projectiles: List[Projectile] = []  # Active projectiles
        self.frame = 0
        self.match_over = False
        self.winner: Optional[int] = None

        # Cycle 4: Timer support
        self.time_limit = time_limit  # 0 = no limit, otherwise frames until time-up
        self.time_remaining = time_limit  # Counts down from time_limit

        # Cycle 4: Combo tracking (per player)
        self.combo_counts = [0, 0]  # Current combo count per player
        self.combo_timers = [0, 0]  # Frames since last hit (combo drops after ~30 frames)
        self.max_combos = [0, 0]  # Maximum combo achieved this match

        # Cycle 4: Pause state
        self.paused = False

        # Cycle 14: Hit freeze (brief pause on significant hits)
        self.hit_freeze_frames = 0  # Frames remaining for hit freeze effect

        # Cycle 19: Damage animation state (per player)
        self.damage_anim_frames = [0, 0]  # Frames remaining for damage animation
        self.damage_anim_amount = [0.0, 0.0]  # Amount of damage for animation intensity
        self.last_damage = [0.0, 0.0]  # Track previous damage to detect hits

        # Create fighters
        for i, stats in enumerate(fighter_stats):
            spawn_pos = stage.spawn_points[i % len(stage.spawn_points)]
            color = None
            if fighter_colors and i < len(fighter_colors):
                color = fighter_colors[i]
            fighter = Fighter(player_id=i, stats=stats, spawn_pos=spawn_pos, color=color)
            self.fighters.append(fighter)

    def add_projectile(self, projectile: Projectile):
        """Add a projectile to the game."""
        self.projectiles.append(projectile)

    def remove_inactive_projectiles(self):
        """Remove all inactive projectiles."""
        self.projectiles = [p for p in self.projectiles if p.active]

    # =========================================================================
    # Cycle 4: Combo and Timer Methods
    # =========================================================================

    def register_hit(self, attacker_id: int, defender_id: int):
        """
        Register a successful hit for combo tracking.

        Args:
            attacker_id: Player ID who landed the hit
            defender_id: Player ID who got hit
        """
        # Increment attacker's combo
        self.combo_counts[attacker_id] += 1
        self.combo_timers[attacker_id] = 0  # Reset combo timer

        # Track max combo
        if self.combo_counts[attacker_id] > self.max_combos[attacker_id]:
            self.max_combos[attacker_id] = self.combo_counts[attacker_id]

        # Reset defender's combo (they got hit)
        self.combo_counts[defender_id] = 0

    def update_combos(self):
        """Update combo timers and drop combos that have expired."""
        COMBO_WINDOW = 45  # ~0.75 seconds at 60 FPS to continue combo

        for i in range(len(self.combo_timers)):
            if self.combo_counts[i] > 0:
                self.combo_timers[i] += 1
                if self.combo_timers[i] > COMBO_WINDOW:
                    # Combo dropped
                    self.combo_counts[i] = 0
                    self.combo_timers[i] = 0

    def update_timer(self):
        """Update match timer. Returns True if time is up."""
        if self.time_limit > 0 and self.time_remaining > 0:
            self.time_remaining -= 1
            if self.time_remaining <= 0:
                return True
        return False

    def get_time_display(self) -> str:
        """Get formatted time remaining for display."""
        if self.time_limit == 0:
            return ""  # No time limit
        seconds = max(0, self.time_remaining) // 60
        return f"{seconds // 60}:{seconds % 60:02d}"

    def update_damage_animations(self):
        """Update damage percentage animations (Cycle 19)."""
        ANIM_DURATION = 20  # Frames for animation

        for i, fighter in enumerate(self.fighters):
            current_damage = fighter.damage_percent

            # Detect new damage
            if current_damage > self.last_damage[i]:
                damage_taken = current_damage - self.last_damage[i]
                self.damage_anim_frames[i] = ANIM_DURATION
                self.damage_anim_amount[i] = damage_taken

            # Decay animation
            if self.damage_anim_frames[i] > 0:
                self.damage_anim_frames[i] -= 1

            # Update last damage tracker
            self.last_damage[i] = current_damage

    def determine_time_up_winner(self):
        """Determine winner when time runs out based on stocks and damage."""
        p1 = self.fighters[0]
        p2 = self.fighters[1]

        # More stocks = win
        if p1.stocks > p2.stocks:
            self.winner = 0
        elif p2.stocks > p1.stocks:
            self.winner = 1
        # Same stocks: lower damage = win
        elif p1.damage_percent < p2.damage_percent:
            self.winner = 0
        elif p2.damage_percent < p1.damage_percent:
            self.winner = 1
        else:
            # True tie
            self.winner = None

        self.match_over = True

    def is_in_blast_zone(self, pos: Vector2) -> bool:
        """Check if a position is outside the blast zones."""
        bz = self.stage.blast_zones
        return (pos.x < bz['left'] or
                pos.x > bz['right'] or
                pos.y < bz['top'] or
                pos.y > bz['bottom'])

    def apply_hit_freeze(self, knockback_magnitude: float, damage: float):
        """
        Cycle 14: Apply hit freeze effect on significant hits.

        The game briefly pauses physics on impactful hits for "hit stop" effect.
        Duration scales with damage/knockback for dramatic effect.

        Args:
            knockback_magnitude: The knockback force of the hit
            damage: The damage dealt by the hit
        """
        # Only freeze on significant hits (knockback > 8 or damage > 10)
        if knockback_magnitude < 8 and damage < 10:
            return

        # Calculate freeze frames (2-5 frames based on impact)
        # Higher knockback/damage = more freeze
        impact = max(knockback_magnitude / 5, damage / 8)
        freeze_frames = min(5, max(2, int(impact)))

        # Only apply if this is stronger than current freeze
        if freeze_frames > self.hit_freeze_frames:
            self.hit_freeze_frames = freeze_frames

    def get_spawn_point(self, player_id: int) -> Vector2:
        """Get spawn point for a player."""
        return self.stage.spawn_points[player_id % len(self.stage.spawn_points)]

    def check_match_over(self):
        """Check if the match is over."""
        alive_players = [f for f in self.fighters if not f.is_dead()]
        if len(alive_players) <= 1:
            self.match_over = True
            if alive_players:
                self.winner = alive_players[0].player_id


# =============================================================================
# GAME LOOP
# =============================================================================

class GameLoop:
    """
    Main game loop with fixed timestep simulation.

    The game runs at a fixed 60 FPS internally regardless of render rate.
    This ensures deterministic physics for replay and netcode.
    """

    def __init__(self, game_state: GameState, headless: bool = False):
        self.state = game_state
        self.headless = headless
        self.running = False

        # Callbacks for external systems
        self.on_frame_update: Optional[Callable[[GameState], None]] = None
        self.on_render: Optional[Callable[[GameState], None]] = None
        self.on_match_end: Optional[Callable[[GameState], None]] = None

        # Cycle 3: Hit callback for visual effects
        # Signature: on_hit(attacker, defender, hitbox, contact_x, contact_y, knockback_magnitude)
        self.on_hit: Optional[Callable] = None

        # Cycle 3: Projectile hit callback
        # Signature: on_projectile_hit(projectile, defender, contact_x, contact_y, knockback_magnitude)
        self.on_projectile_hit: Optional[Callable] = None

        # Cycle 3: Reflect callback
        # Signature: on_reflect(projectile, reflector, x, y)
        self.on_reflect: Optional[Callable] = None

        # Cycle 3: Landing callback for dust effects
        # Signature: on_land(fighter, x, y)
        self.on_land: Optional[Callable] = None

        # Cycle 9: Dodge callback for sound/visual effects
        # Signature: on_dodge(fighter, dodge_type)
        self.on_dodge: Optional[Callable] = None

        # Timing
        self.frame_time = CONFIG.frame_time
        self.accumulator = 0.0
        self.last_time = 0.0

        # Frame skip for rendering (render every N simulation frames)
        self.render_skip = 1
        self.frames_since_render = 0

    def step(self) -> bool:
        """
        Run a single simulation frame.

        Returns True if match is still ongoing, False if over.
        """
        if self.state.match_over:
            return False

        # Cycle 4: Check pause state
        if self.state.paused:
            return True

        # Cycle 14: Check hit freeze state
        # During hit freeze, skip physics but still render (dramatic pause)
        if self.state.hit_freeze_frames > 0:
            self.state.hit_freeze_frames -= 1
            # Still update input states during freeze
            for fighter in self.state.fighters:
                fighter.input.update_prev()
            return True

        # Update each fighter
        for fighter in self.state.fighters:
            self._update_fighter(fighter)

        # Update projectiles
        self._update_projectiles()

        # Check for knockouts
        self._check_knockouts()

        # Process combat (hitbox vs hurtbox, projectiles)
        self._process_combat()
        self._process_projectile_combat()

        # Clean up inactive projectiles
        self.state.remove_inactive_projectiles()

        # Cycle 4: Update combo timers
        self.state.update_combos()

        # Cycle 19: Update damage animations
        self.state.update_damage_animations()

        # Cycle 4: Update match timer
        if self.state.update_timer():
            # Time's up!
            self.state.determine_time_up_winner()
            return False

        # Check match end
        self.state.check_match_over()

        # Increment frame counter
        self.state.frame += 1

        # Update input previous states
        for fighter in self.state.fighters:
            fighter.input.update_prev()

        return not self.state.match_over

    def _update_projectiles(self):
        """Update all active projectiles."""
        for projectile in self.state.projectiles:
            if projectile.active:
                projectile.update(CONFIG.gravity)

                # Check if projectile is off-stage
                if self.state.is_in_blast_zone(projectile.position):
                    projectile.active = False

    def _update_fighter(self, fighter: Fighter):
        """Update a single fighter for one frame."""
        # Decrement timers
        if fighter.invincible_frames > 0:
            fighter.invincible_frames -= 1

        if fighter.hitstun_frames > 0:
            fighter.hitstun_frames -= 1
            if fighter.hitstun_frames == 0:
                fighter.state = FighterState.FALLING if not fighter.grounded else FighterState.IDLE

        # Shield regeneration
        if fighter.shield_regen_delay > 0:
            fighter.shield_regen_delay -= 1
        elif fighter.state != FighterState.SHIELDING and fighter.shield_health < 100.0:
            fighter.shield_health = min(100.0, fighter.shield_health + 0.15)

        # Shield stun countdown
        if fighter.shield_stun > 0:
            fighter.shield_stun -= 1
            if fighter.shield_stun == 0 and fighter.state == FighterState.SHIELD_STUN:
                fighter.shield_broken = False
                fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

        # Counter window countdown
        if fighter.counter_window_frames > 0:
            fighter.counter_window_frames -= 1
            if fighter.counter_window_frames == 0:
                fighter.counter_active = False

        # Dodge frames countdown
        if fighter.dodge_frames_remaining > 0:
            fighter.dodge_frames_remaining -= 1
            if fighter.dodge_frames_remaining == 0:
                fighter.invincible_frames = 0
                fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

        # Reset air dodge when landing
        if fighter.grounded and not fighter.air_dodge_available:
            fighter.air_dodge_available = True

        # State-based update
        if fighter.state == FighterState.HITSTUN:
            self._update_hitstun(fighter)
        elif fighter.state == FighterState.ATTACKING:
            self._update_attack(fighter)
        elif fighter.state == FighterState.SPECIAL:
            self._update_special(fighter)
        elif fighter.state == FighterState.SHIELDING:
            self._update_shielding(fighter)
        elif fighter.state == FighterState.DODGING:
            self._update_dodge(fighter)
        elif fighter.state == FighterState.COUNTER:
            self._update_counter(fighter)
        elif fighter.state == FighterState.REFLECTING:
            self._update_reflector(fighter)
        elif fighter.state == FighterState.SHIELD_STUN:
            pass  # Just wait out the stun
        else:
            self._update_normal(fighter)

        # Apply physics
        self._apply_physics(fighter)

        # Platform collision
        self._handle_platform_collision(fighter)

        fighter.state_frame += 1

    def _update_normal(self, fighter: Fighter):
        """Update fighter in normal (non-hitstun) state."""
        inp = fighter.input

        # Horizontal movement
        if inp.left and not inp.right:
            if fighter.grounded:
                fighter.velocity.x = -fighter.stats.walk_speed
                fighter.state = FighterState.WALKING
            else:
                fighter.velocity.x = max(
                    fighter.velocity.x - 0.3,
                    -fighter.stats.air_speed
                )
            fighter.facing_right = False
        elif inp.right and not inp.left:
            if fighter.grounded:
                fighter.velocity.x = fighter.stats.walk_speed
                fighter.state = FighterState.WALKING
            else:
                fighter.velocity.x = min(
                    fighter.velocity.x + 0.3,
                    fighter.stats.air_speed
                )
            fighter.facing_right = True
        else:
            if fighter.grounded:
                fighter.velocity.x *= CONFIG.friction
                if abs(fighter.velocity.x) < 0.1:
                    fighter.velocity.x = 0
                    if fighter.state == FighterState.WALKING:
                        fighter.state = FighterState.IDLE

        # Jumping
        if inp.jump_pressed:
            if fighter.grounded:
                fighter.velocity.y = -fighter.stats.jump_force
                fighter.grounded = False
                fighter.state = FighterState.JUMPING
            elif fighter.can_double_jump:
                fighter.velocity.y = -fighter.stats.double_jump_force
                fighter.can_double_jump = False
                fighter.state = FighterState.JUMPING

        # Fast fall
        if inp.down and not fighter.grounded and fighter.velocity.y > 0:
            fighter.velocity.y = max(fighter.velocity.y,
                                     fighter.stats.fast_fall_speed * 10)

        # Attacks
        if inp.attack_pressed and fighter.state not in (FighterState.ATTACKING,):
            self._start_attack(fighter)

        # Special moves
        if inp.special_pressed and fighter.state not in (FighterState.ATTACKING, FighterState.SPECIAL):
            self._start_special(fighter)

        # Shield
        if inp.shield and fighter.grounded and not fighter.shield_broken:
            if fighter.state not in (FighterState.SHIELDING, FighterState.DODGING,
                                     FighterState.ATTACKING, FighterState.SPECIAL):
                fighter.state = FighterState.SHIELDING
                fighter.state_frame = 0

        # Dodge (shield + direction while grounded, or just shield in air for air dodge)
        if inp.shield_pressed:
            if fighter.grounded:
                if inp.left or inp.right:
                    # Roll
                    self._start_dodge(fighter, DodgeType.ROLL_BACKWARD if inp.left == fighter.facing_right else DodgeType.ROLL_FORWARD)
                elif inp.down and fighter.state == FighterState.SHIELDING:
                    # Spot dodge
                    self._start_dodge(fighter, DodgeType.SPOT_DODGE)
            else:
                # Air dodge
                if fighter.air_dodge_available:
                    self._start_dodge(fighter, DodgeType.AIR_DODGE)

    def _update_hitstun(self, fighter: Fighter):
        """Update fighter in hitstun state."""
        # Allow DI (Directional Influence)
        if fighter.input.left:
            fighter.velocity.x -= 0.1
        elif fighter.input.right:
            fighter.velocity.x += 0.1

    def _update_attack(self, fighter: Fighter):
        """Update fighter during attack animation."""
        fighter.attack_frame += 1
        attack = fighter.current_attack

        if attack is None:
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING
            return

        # Update active hitboxes based on frame
        fighter.active_hitboxes = []
        for hitbox in attack.hitboxes:
            start, end = hitbox.active_frames
            if start <= fighter.attack_frame <= end:
                fighter.active_hitboxes.append(hitbox)

        # End attack when animation is done
        if fighter.attack_frame >= attack.total_frames:
            fighter.current_attack = None
            fighter.active_hitboxes = []
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

    def _update_special(self, fighter: Fighter):
        """Update fighter during special move animation."""
        fighter.attack_frame += 1
        attack = fighter.current_attack

        if attack is None:
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING
            return

        # Update active hitboxes based on frame
        fighter.active_hitboxes = []
        for hitbox in attack.hitboxes:
            start, end = hitbox.active_frames
            if start <= fighter.attack_frame <= end:
                fighter.active_hitboxes.append(hitbox)

        # End special when animation is done
        if fighter.attack_frame >= attack.total_frames:
            fighter.current_attack = None
            fighter.active_hitboxes = []
            fighter.super_armor_active = False
            fighter.counter_active = False
            fighter.reflector_active = False
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

    def _update_shielding(self, fighter: Fighter):
        """Update fighter while shielding."""
        inp = fighter.input

        # Deplete shield over time (slow drain while holding)
        fighter.shield_health -= 0.1

        # Check for shield break
        if fighter.shield_health <= 0:
            fighter.shield_health = 0
            fighter.shield_broken = True
            fighter.shield_stun = 180  # 3 seconds stun
            fighter.state = FighterState.SHIELD_STUN
            fighter.state_frame = 0
            return

        # Release shield if button released
        if not inp.shield:
            fighter.state = FighterState.IDLE
            fighter.shield_regen_delay = 60  # 1 second before regen starts

        # Can still dodge out of shield
        if inp.down:
            self._start_dodge(fighter, DodgeType.SPOT_DODGE)
        elif inp.left:
            self._start_dodge(fighter, DodgeType.ROLL_BACKWARD if fighter.facing_right else DodgeType.ROLL_FORWARD)
        elif inp.right:
            self._start_dodge(fighter, DodgeType.ROLL_FORWARD if fighter.facing_right else DodgeType.ROLL_BACKWARD)

    def _update_dodge(self, fighter: Fighter):
        """Update fighter during dodge animation."""
        # Apply roll movement
        if fighter.dodge_type in (DodgeType.ROLL_FORWARD, DodgeType.ROLL_BACKWARD):
            direction = 1 if fighter.dodge_type == DodgeType.ROLL_FORWARD else -1
            if not fighter.facing_right:
                direction = -direction
            fighter.velocity.x = direction * fighter.stats.run_speed * 0.8

        # Air dodge can have directional influence
        if fighter.dodge_type == DodgeType.AIR_DODGE:
            inp = fighter.input
            if inp.left:
                fighter.velocity.x = -fighter.stats.air_speed * 0.5
            elif inp.right:
                fighter.velocity.x = fighter.stats.air_speed * 0.5
            if inp.up:
                fighter.velocity.y = -fighter.stats.air_speed * 0.5
            elif inp.down:
                fighter.velocity.y = fighter.stats.air_speed * 0.5

    def _update_counter(self, fighter: Fighter):
        """Update fighter during counter stance."""
        fighter.attack_frame += 1
        attack = fighter.current_attack

        if attack is None:
            fighter.counter_active = False
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING
            return

        # Counter success triggers counterattack
        if fighter.counter_success:
            # Counter was triggered - hitbox is now active
            fighter.active_hitboxes = []
            for hitbox in attack.hitboxes:
                start, end = hitbox.active_frames
                if start <= fighter.attack_frame <= end:
                    fighter.active_hitboxes.append(hitbox)

        # End counter when animation is done
        if fighter.attack_frame >= attack.total_frames:
            fighter.current_attack = None
            fighter.active_hitboxes = []
            fighter.counter_active = False
            fighter.counter_success = False
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

    def _update_reflector(self, fighter: Fighter):
        """Update fighter during reflector active frames."""
        fighter.attack_frame += 1
        attack = fighter.current_attack

        if attack is None:
            fighter.reflector_active = False
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING
            return

        # Reflector has a hitbox for close-range use
        fighter.active_hitboxes = []
        for hitbox in attack.hitboxes:
            start, end = hitbox.active_frames
            if start <= fighter.attack_frame <= end:
                fighter.active_hitboxes.append(hitbox)
                fighter.reflector_active = True
            else:
                fighter.reflector_active = False

        # End reflector when animation is done
        if fighter.attack_frame >= attack.total_frames:
            fighter.current_attack = None
            fighter.active_hitboxes = []
            fighter.reflector_active = False
            fighter.state = FighterState.IDLE if fighter.grounded else FighterState.FALLING

    def _start_dodge(self, fighter: Fighter, dodge_type: 'DodgeType'):
        """Start a dodge move."""
        fighter.dodge_type = dodge_type
        fighter.state = FighterState.DODGING
        fighter.state_frame = 0

        # Set dodge duration and invincibility based on type
        if dodge_type == DodgeType.SPOT_DODGE:
            fighter.dodge_frames_remaining = 20
            fighter.invincible_frames = 15
        elif dodge_type in (DodgeType.ROLL_FORWARD, DodgeType.ROLL_BACKWARD):
            fighter.dodge_frames_remaining = 30
            fighter.invincible_frames = 20
        elif dodge_type == DodgeType.AIR_DODGE:
            fighter.dodge_frames_remaining = 25
            fighter.invincible_frames = 18
            fighter.air_dodge_available = False

        # Cycle 9: Trigger dodge callback for sound/visual effects
        if self.on_dodge:
            self.on_dodge(fighter, dodge_type)

    def _start_attack(self, fighter: Fighter):
        """Start an attack based on current input and state."""
        inp = fighter.input

        # Determine attack type based on input
        if fighter.grounded:
            if inp.up:
                attack_type = AttackType.UP_TILT
            elif inp.down:
                attack_type = AttackType.DOWN_TILT
            elif inp.left or inp.right:
                attack_type = AttackType.FORWARD_TILT
            else:
                attack_type = AttackType.JAB
        else:
            if inp.up:
                attack_type = AttackType.UP_AIR
            elif inp.down:
                attack_type = AttackType.DOWN_AIR
            elif (inp.left and fighter.facing_right) or (inp.right and not fighter.facing_right):
                attack_type = AttackType.BACK_AIR
            elif (inp.right and fighter.facing_right) or (inp.left and not fighter.facing_right):
                attack_type = AttackType.FORWARD_AIR
            else:
                attack_type = AttackType.NEUTRAL_AIR

        # Try character-specific attack first, fall back to default
        attack = fighter.get_attack_data(attack_type)
        if attack is None:
            attack = self._get_attack_data(attack_type)

        fighter.current_attack = attack
        fighter.attack_frame = 0
        fighter.state = FighterState.ATTACKING

    def _start_special(self, fighter: Fighter):
        """Start a special move based on current input."""
        inp = fighter.input

        # Determine special type based on direction
        if inp.up:
            attack_type = AttackType.UP_SPECIAL
            # Up special typically gives vertical boost (recovery)
            if not fighter.grounded:
                fighter.velocity.y = -fighter.stats.jump_force * 0.8
                fighter.can_double_jump = False
        elif inp.down:
            attack_type = AttackType.DOWN_SPECIAL
        elif inp.left or inp.right:
            attack_type = AttackType.SIDE_SPECIAL
            # Side special may give horizontal movement
            direction = -1 if inp.left else 1
            fighter.facing_right = direction > 0
            fighter.velocity.x = direction * fighter.stats.run_speed * 0.5
        else:
            attack_type = AttackType.NEUTRAL_SPECIAL

        # Try character-specific attack first, fall back to default
        attack = fighter.get_attack_data(attack_type)
        if attack is None:
            attack = self._get_attack_data(attack_type)

        fighter.current_attack = attack
        fighter.attack_frame = 0

        # Handle character-specific special mechanics
        char_type = fighter.stats.character_type

        # RUSH down-B: Counter
        if char_type == "RUSH" and attack_type == AttackType.DOWN_SPECIAL:
            fighter.state = FighterState.COUNTER
            fighter.counter_active = True
            fighter.counter_window_frames = 15  # 15 frame counter window
            fighter.counter_success = False
            return

        # FLEX down-B: Reflector
        if char_type == "FLEX" and attack_type == AttackType.DOWN_SPECIAL:
            fighter.state = FighterState.REFLECTING
            fighter.reflector_active = True
            return

        # GRAB neutral-B: Armored Charge (super armor)
        if char_type == "GRAB" and attack_type == AttackType.NEUTRAL_SPECIAL:
            fighter.super_armor_active = True
            fighter.state = FighterState.SPECIAL
            return

        # ZONE neutral-B: Spawn projectile on startup
        if char_type == "ZONE" and attack_type == AttackType.NEUTRAL_SPECIAL:
            self._spawn_zone_projectile(fighter, "neutral")
            fighter.state = FighterState.SPECIAL
            return

        # ZONE side-B: Horizontal beam projectile
        if char_type == "ZONE" and attack_type == AttackType.SIDE_SPECIAL:
            self._spawn_zone_projectile(fighter, "side")
            fighter.state = FighterState.SPECIAL
            return

        # FLEX neutral-B: Fireball with gravity
        if char_type == "FLEX" and attack_type == AttackType.NEUTRAL_SPECIAL:
            self._spawn_flex_fireball(fighter)
            fighter.state = FighterState.SPECIAL
            return

        fighter.state = FighterState.SPECIAL

    def _spawn_zone_projectile(self, fighter: Fighter, proj_type: str):
        """Spawn a projectile for ZONE character."""
        direction = 1 if fighter.facing_right else -1

        if proj_type == "neutral":
            # Chargeable energy projectile
            projectile = Projectile(
                owner_id=fighter.player_id,
                position=Vector2(
                    fighter.position.x + direction * 50,
                    fighter.position.y - 50
                ),
                velocity=Vector2(direction * 8, 0),  # Fast horizontal
                hitbox=Hitbox(
                    offset=Vector2(0, 0),
                    width=25,
                    height=25,
                    damage=8.0,
                    knockback_base=4.0,
                    knockback_scaling=0.08,
                    angle=25,
                    active_frames=(0, 999)  # Always active
                ),
                lifetime=120,  # 2 seconds
                has_gravity=False,
                width=25,
                height=25,
                color=(80, 180, 255)  # Cyan
            )
        else:  # side
            # Horizontal beam
            projectile = Projectile(
                owner_id=fighter.player_id,
                position=Vector2(
                    fighter.position.x + direction * 60,
                    fighter.position.y - 50
                ),
                velocity=Vector2(direction * 12, 0),  # Very fast
                hitbox=Hitbox(
                    offset=Vector2(0, 0),
                    width=40,
                    height=15,
                    damage=10.0,
                    knockback_base=4.5,
                    knockback_scaling=0.08,
                    angle=20,
                    active_frames=(0, 999)
                ),
                lifetime=90,  # 1.5 seconds
                has_gravity=False,
                width=40,
                height=15,
                color=(100, 200, 255)  # Light blue beam
            )

        self.state.add_projectile(projectile)

    def _spawn_flex_fireball(self, fighter: Fighter):
        """Spawn a fireball for FLEX character."""
        direction = 1 if fighter.facing_right else -1

        projectile = Projectile(
            owner_id=fighter.player_id,
            position=Vector2(
                fighter.position.x + direction * 45,
                fighter.position.y - 50
            ),
            velocity=Vector2(direction * 6, -2),  # Forward and slightly up
            hitbox=Hitbox(
                offset=Vector2(0, 0),
                width=22,
                height=22,
                damage=6.0,
                knockback_base=3.0,
                knockback_scaling=0.05,
                angle=30,
                active_frames=(0, 999)
            ),
            lifetime=150,  # 2.5 seconds
            has_gravity=True,  # Falls with gravity
            width=22,
            height=22,
            color=(255, 120, 50)  # Orange fireball
        )

        self.state.add_projectile(projectile)

    def _get_attack_data(self, attack_type: AttackType) -> AttackData:
        """Get attack data for an attack type (simplified)."""
        # Basic attack data - would be character-specific in full implementation
        if attack_type == AttackType.JAB:
            return AttackData(
                attack_type=attack_type,
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
        elif attack_type in (AttackType.FORWARD_TILT, AttackType.FORWARD_AIR):
            return AttackData(
                attack_type=attack_type,
                total_frames=30,
                hitboxes=[Hitbox(
                    offset=Vector2(25, -55),
                    width=55,
                    height=35,
                    damage=8.0,
                    knockback_base=4.0,
                    knockback_scaling=0.08,
                    angle=40,
                    active_frames=(8, 12)
                )],
                startup_frames=8,
                recovery_frames=18
            )
        elif attack_type in (AttackType.UP_TILT, AttackType.UP_AIR):
            return AttackData(
                attack_type=attack_type,
                total_frames=28,
                hitboxes=[Hitbox(
                    offset=Vector2(-20, -90),
                    width=60,
                    height=40,
                    damage=7.0,
                    knockback_base=5.0,
                    knockback_scaling=0.1,
                    angle=85,
                    active_frames=(6, 10)
                )],
                startup_frames=6,
                recovery_frames=18
            )
        elif attack_type in (AttackType.DOWN_TILT, AttackType.DOWN_AIR):
            return AttackData(
                attack_type=attack_type,
                total_frames=25,
                hitboxes=[Hitbox(
                    offset=Vector2(15, -20),
                    width=50,
                    height=25,
                    damage=6.0,
                    knockback_base=3.0,
                    knockback_scaling=0.06,
                    angle=80 if attack_type == AttackType.DOWN_TILT else 270,
                    active_frames=(7, 10)
                )],
                startup_frames=7,
                recovery_frames=15
            )
        elif attack_type == AttackType.BACK_AIR:
            return AttackData(
                attack_type=attack_type,
                total_frames=32,
                hitboxes=[Hitbox(
                    offset=Vector2(-30, -55),
                    width=50,
                    height=35,
                    damage=12.0,
                    knockback_base=6.0,
                    knockback_scaling=0.12,
                    angle=35,
                    active_frames=(10, 14)
                )],
                startup_frames=10,
                recovery_frames=18
            )
        elif attack_type == AttackType.NEUTRAL_AIR:
            return AttackData(
                attack_type=attack_type,
                total_frames=26,
                hitboxes=[Hitbox(
                    offset=Vector2(-25, -60),
                    width=70,
                    height=60,
                    damage=6.0,
                    knockback_base=3.5,
                    knockback_scaling=0.07,
                    angle=45,
                    active_frames=(5, 18)
                )],
                startup_frames=5,
                recovery_frames=8
            )
        else:
            # Default simple attack
            return AttackData(
                attack_type=attack_type,
                total_frames=20,
                hitboxes=[Hitbox(
                    offset=Vector2(20, -50),
                    width=40,
                    height=30,
                    damage=5.0,
                    knockback_base=3.0,
                    knockback_scaling=0.06,
                    angle=45,
                    active_frames=(5, 10)
                )],
                startup_frames=5,
                recovery_frames=10
            )

    def _compute_safe_displacement(self, fighter: Fighter) -> float:
        """
        OGC-inspired displacement bound calculation.

        Computes the maximum safe vertical movement distance to prevent
        penetration through platforms. Returns the bounded y-displacement.
        """
        if fighter.velocity.y <= 0:  # Not falling, no bound needed
            return fighter.velocity.y

        hurtbox = fighter.get_hurtbox_aabb()
        foot_y = hurtbox.bottom
        max_displacement = fighter.velocity.y

        # Check all platforms for potential contact
        platforms_to_check = [self.state.stage.main_platform] + self.state.stage.platforms

        for platform in platforms_to_check:
            # Skip if no horizontal overlap
            plat_aabb = platform.get_aabb()
            if hurtbox.right < plat_aabb.left or hurtbox.left > plat_aabb.right:
                continue

            # Skip passthrough platforms if dropping through
            if platform.passthrough and fighter.input.down:
                continue

            # Skip passthrough platforms if we're already below them
            if platform.passthrough and foot_y > plat_aabb.top:
                continue

            # Distance to contact zone
            distance_to_contact = platform.contact_zone_y - foot_y

            # If we're above the contact zone and would cross it
            if distance_to_contact >= 0 and distance_to_contact < max_displacement:
                max_displacement = distance_to_contact

        return max_displacement

    def _apply_physics(self, fighter: Fighter):
        """Apply physics (gravity, velocity) to fighter."""
        # Apply gravity
        if not fighter.grounded:
            fighter.velocity.y += CONFIG.gravity
            fighter.velocity.y = min(fighter.velocity.y, CONFIG.terminal_velocity)

            # Air friction
            fighter.velocity.x *= CONFIG.air_friction

        # OGC: Compute safe displacement bound for vertical movement
        safe_y_displacement = self._compute_safe_displacement(fighter)

        # Apply velocity with bounded vertical displacement
        fighter.position.x += fighter.velocity.x
        fighter.position.y += safe_y_displacement

    def _handle_platform_collision(self, fighter: Fighter):
        """Handle collision between fighter and platforms."""
        hurtbox = fighter.get_hurtbox_aabb()

        # Check main platform
        main_plat = self.state.stage.main_platform
        self._check_platform_landing(fighter, main_plat, hurtbox)

        # Check other platforms
        for platform in self.state.stage.platforms:
            self._check_platform_landing(fighter, platform, hurtbox)

    def _check_platform_landing(self, fighter: Fighter, platform: Platform,
                                 hurtbox: AABB):
        """
        Check if fighter lands on a platform.

        OGC: Uses contact zone for detection but snaps to exact surface for positioning.
        """
        plat_aabb = platform.get_aabb()

        # Only check if moving downward
        if fighter.velocity.y <= 0:
            return

        # Check if feet are near the platform surface
        foot_y = hurtbox.bottom
        prev_foot_y = foot_y - fighter.velocity.y

        # For passthrough platforms, only land if coming from above
        if platform.passthrough:
            if prev_foot_y >= plat_aabb.top:
                return
            # Allow dropping through
            if fighter.input.down:
                return

        # Check horizontal overlap
        if hurtbox.right < plat_aabb.left or hurtbox.left > plat_aabb.right:
            return

        # OGC: Check if crossing contact zone (offset volume)
        # This triggers detection early to prevent penetration
        contact_zone_y = platform.contact_zone_y
        if prev_foot_y <= contact_zone_y <= foot_y:
            # Land detected! Snap to EXACT surface, not contact zone
            # This ensures visual correctness while using offset for detection
            # OGC: Snap to exact surface, accounting for rendering foot offset
            # position.y should be FOOT_OFFSET pixels above platform surface
            # so that rendered feet (at position.y + FOOT_OFFSET) align exactly with platform.top
            fighter.position.y = plat_aabb.top - FOOT_OFFSET
            fighter.velocity.y = 0
            fighter.grounded = True
            fighter.can_double_jump = True

            if fighter.state in (FighterState.JUMPING, FighterState.FALLING):
                fighter.state = FighterState.LANDING
                fighter.state_frame = 0
                # Cycle 3: Trigger landing dust effect
                if self.on_land:
                    self.on_land(fighter, fighter.position.x, fighter.position.y)
            elif fighter.state == FighterState.LANDING and fighter.state_frame > 4:
                fighter.state = FighterState.IDLE

    def _check_knockouts(self):
        """Check if any fighters are in blast zones."""
        for fighter in self.state.fighters:
            if self.state.is_in_blast_zone(fighter.position):
                spawn = self.state.get_spawn_point(fighter.player_id)
                fighter.respawn(spawn)

    def _process_combat(self):
        """Process hitbox vs hurtbox collisions."""
        fighters = self.state.fighters

        # For each fighter with active hitboxes
        for attacker in fighters:
            if not attacker.active_hitboxes:
                continue

            for hitbox in attacker.active_hitboxes:
                hitbox_aabb = hitbox.get_aabb(attacker.position, attacker.facing_right)

                # Check against all other fighters
                for defender in fighters:
                    if defender == attacker:
                        continue
                    if defender.invincible_frames > 0:
                        continue
                    if defender.state == FighterState.HITSTUN:
                        continue  # Prevent infinite combos (simplified)

                    hurtbox_aabb = defender.get_hurtbox_aabb()

                    if hitbox_aabb.intersects(hurtbox_aabb):
                        # Check if defender is shielding
                        if defender.state == FighterState.SHIELDING:
                            # Hit shield instead
                            shield_damage = hitbox.damage * 1.5  # Shield takes more damage
                            defender.shield_health -= shield_damage
                            defender.shield_regen_delay = 90  # Delay regen after being hit

                            # Shield stun
                            defender.shield_stun = int(hitbox.damage * 0.5)

                            # Check for shield break
                            if defender.shield_health <= 0:
                                defender.shield_health = 0
                                defender.shield_broken = True
                                defender.shield_stun = 180  # Long stun
                                defender.state = FighterState.SHIELD_STUN
                                defender.state_frame = 0

                            # Remove hitbox after hit
                            attacker.active_hitboxes.remove(hitbox)
                            break

                        # Check if defender is in counter stance
                        if defender.counter_active and defender.counter_window_frames > 0:
                            # Counter triggered!
                            defender.counter_success = True
                            defender.counter_window_frames = 0

                            # Turn to face attacker
                            defender.facing_right = attacker.position.x > defender.position.x

                            # Counter attack hits harder (1.5x damage boost)
                            if defender.current_attack:
                                for hb in defender.current_attack.hitboxes:
                                    hb.damage *= 1.5
                                    hb.knockback_base *= 1.5

                            # Remove attacker's hitbox
                            attacker.active_hitboxes.remove(hitbox)
                            break

                        # Check if attacker has super armor
                        if attacker.super_armor_active:
                            # Super armor: take damage but don't flinch
                            # (The attacker still hits the defender normally)
                            pass

                        # Check if defender has super armor
                        if defender.super_armor_active:
                            # Take damage but don't get knocked back
                            defender.damage_percent += hitbox.damage
                            attacker.active_hitboxes.remove(hitbox)
                            break

                        # Calculate knockback
                        angle_rad = np.radians(hitbox.angle)
                        if not attacker.facing_right:
                            angle_rad = np.pi - angle_rad

                        kb_magnitude = hitbox.knockback_base + (
                            defender.damage_percent * hitbox.knockback_scaling
                        )
                        knockback = Vector2(
                            np.cos(angle_rad) * kb_magnitude,
                            -np.sin(angle_rad) * kb_magnitude
                        )

                        # Calculate contact point (center of hitbox/hurtbox overlap)
                        contact_x = (hitbox_aabb.center.x + hurtbox_aabb.center.x) / 2
                        contact_y = (hitbox_aabb.center.y + hurtbox_aabb.center.y) / 2

                        defender.apply_knockback(knockback, hitbox.damage)

                        # Cycle 14: Apply hit freeze for impactful hits
                        self.state.apply_hit_freeze(knockback.magnitude(), hitbox.damage)

                        # Cycle 4: Register hit for combo tracking
                        self.state.register_hit(attacker.player_id, defender.player_id)

                        # Cycle 3: Call hit callback for visual effects
                        if self.on_hit:
                            self.on_hit(attacker, defender, hitbox, contact_x, contact_y,
                                       knockback.magnitude())

                        # Remove hitbox after hit (no multi-hit for now)
                        attacker.active_hitboxes.remove(hitbox)
                        break

    def _process_projectile_combat(self):
        """Process projectile collisions with fighters."""
        fighters = self.state.fighters

        for projectile in self.state.projectiles:
            if not projectile.active:
                continue

            proj_aabb = projectile.get_aabb()

            for fighter in fighters:
                # Skip owner (unless reflected)
                if fighter.player_id == projectile.owner_id and not projectile.reflected:
                    continue
                if fighter.invincible_frames > 0:
                    continue

                hurtbox_aabb = fighter.get_hurtbox_aabb()

                if proj_aabb.intersects(hurtbox_aabb):
                    # Check if fighter is shielding
                    if fighter.state == FighterState.SHIELDING:
                        shield_damage = projectile.hitbox.damage * 1.5
                        fighter.shield_health -= shield_damage
                        fighter.shield_regen_delay = 90
                        projectile.active = False
                        break

                    # Check if fighter has reflector active
                    if fighter.reflector_active:
                        reflect_x = projectile.position.x
                        reflect_y = projectile.position.y
                        projectile.reflect()
                        projectile.owner_id = fighter.player_id  # Now belongs to reflector

                        # Cycle 3: Call reflect callback for visual effects
                        if self.on_reflect:
                            self.on_reflect(projectile, fighter, reflect_x, reflect_y)
                        break

                    # Normal hit
                    angle_rad = np.radians(projectile.hitbox.angle)
                    if projectile.velocity.x < 0:
                        angle_rad = np.pi - angle_rad

                    kb_magnitude = projectile.hitbox.knockback_base + (
                        fighter.damage_percent * projectile.hitbox.knockback_scaling
                    )
                    knockback = Vector2(
                        np.cos(angle_rad) * kb_magnitude,
                        -np.sin(angle_rad) * kb_magnitude
                    )

                    # Contact point
                    contact_x = projectile.position.x
                    contact_y = projectile.position.y

                    fighter.apply_knockback(knockback, projectile.hitbox.damage)

                    # Cycle 14: Apply hit freeze for impactful projectile hits
                    self.state.apply_hit_freeze(knockback.magnitude(), projectile.hitbox.damage)

                    # Cycle 4: Register projectile hit for combo tracking
                    self.state.register_hit(projectile.owner_id, fighter.player_id)

                    # Cycle 3: Call projectile hit callback for visual effects
                    if self.on_projectile_hit:
                        self.on_projectile_hit(projectile, fighter, contact_x, contact_y,
                                              knockback.magnitude())

                    projectile.active = False
                    break

    def run(self, max_frames: int = 0):
        """
        Run the game loop.

        Args:
            max_frames: Maximum frames to run (0 = until match ends)
        """
        self.running = True
        self.last_time = time.time()
        frames_run = 0

        while self.running and (max_frames == 0 or frames_run < max_frames):
            current_time = time.time()
            delta = current_time - self.last_time
            self.last_time = current_time
            self.accumulator += delta

            # Fixed timestep simulation
            while self.accumulator >= self.frame_time:
                if not self.step():
                    self.running = False
                    break

                self.accumulator -= self.frame_time
                frames_run += 1

                # Callback for external systems
                if self.on_frame_update:
                    self.on_frame_update(self.state)

            # Rendering (if not headless)
            if not self.headless and self.on_render:
                self.frames_since_render += 1
                if self.frames_since_render >= self.render_skip:
                    self.on_render(self.state)
                    self.frames_since_render = 0

        if self.on_match_end:
            self.on_match_end(self.state)

        return self.state

    def stop(self):
        """Stop the game loop."""
        self.running = False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_stats() -> List[FighterStats]:
    """Create default fighter stats for 2 players."""
    return [
        FighterStats(name="Player 1"),
        FighterStats(name="Player 2")
    ]


def create_game(stage_name: str = "battlefield",
                p1_character: Optional[str] = None,
                p2_character: Optional[str] = None,
                stock_count: int = 3,
                time_limit: int = 0) -> GameState:
    """
    Create a new game with optional character selection.

    Args:
        stage_name: Stage to use ("battlefield" or "final_destination")
        p1_character: Character name for P1 (RUSH, ZONE, GRAB, FLEX) or None for default
        p2_character: Character name for P2 (RUSH, ZONE, GRAB, FLEX) or None for default
        stock_count: Number of stocks per player (1-5)
        time_limit: Time limit in frames (0 = no limit)

    Returns:
        GameState ready for play
    """
    if stage_name.lower() == "final_destination":
        stage = Stage.create_final_destination()
    else:
        stage = Stage.create_basic()

    # Get character stats and colors
    stats = []
    colors = []

    for i, char_name in enumerate([p1_character, p2_character]):
        if char_name:
            try:
                from characters import (
                    get_character_stats, get_character_color,
                    get_character_by_name
                )
                char_type = get_character_by_name(char_name)
                if char_type:
                    char_stats = get_character_stats(char_type)
                    # Create a copy with character_type set
                    fighter_stats = FighterStats(
                        name=char_stats.name,
                        walk_speed=char_stats.walk_speed,
                        run_speed=char_stats.run_speed,
                        air_speed=char_stats.air_speed,
                        jump_force=char_stats.jump_force,
                        double_jump_force=char_stats.double_jump_force,
                        weight=char_stats.weight,
                        fall_speed=char_stats.fall_speed,
                        fast_fall_speed=char_stats.fast_fall_speed,
                        width=char_stats.width,
                        height=char_stats.height,
                        character_type=char_type.name
                    )
                    stats.append(fighter_stats)
                    colors.append(get_character_color(char_type))
                else:
                    stats.append(FighterStats(name=f"Player {i + 1}"))
                    colors.append(None)
            except ImportError:
                stats.append(FighterStats(name=f"Player {i + 1}"))
                colors.append(None)
        else:
            stats.append(FighterStats(name=f"Player {i + 1}"))
            colors.append(None)

    # Create game state with stock count and time limit
    game = GameState(stage, stats, colors, time_limit)

    # Set stock count for each fighter
    for fighter in game.fighters:
        fighter.stocks = stock_count

    return game


def run_headless_match(game_state: GameState,
                       input_callback: Callable[[GameState, int], InputState],
                       max_frames: int = 10800) -> GameState:
    """
    Run a match in headless mode.

    Args:
        game_state: The game state to run
        input_callback: Function that returns InputState for each player each frame
                       Takes (game_state, player_id) and returns InputState
        max_frames: Maximum frames (default 3 minutes at 60 FPS)

    Returns:
        Final game state
    """
    loop = GameLoop(game_state, headless=True)

    def frame_update(state: GameState):
        for fighter in state.fighters:
            fighter.input = input_callback(state, fighter.player_id)

    loop.on_frame_update = frame_update
    return loop.run(max_frames)


# =============================================================================
# OBSERVATION FOR ML
# =============================================================================

def get_observation(game_state: GameState, player_id: int) -> np.ndarray:
    """
    Get observation array for a player (for ML training).

    Returns a flat numpy array with normalized game state.
    """
    player = game_state.fighters[player_id]
    opponent = game_state.fighters[1 - player_id]

    # Normalize positions to [0, 1]
    px = player.position.x / CONFIG.screen_width
    py = player.position.y / CONFIG.screen_height
    ox = opponent.position.x / CONFIG.screen_width
    oy = opponent.position.y / CONFIG.screen_height

    # Normalize velocities (assuming max ~20 units)
    pvx = player.velocity.x / 20.0
    pvy = player.velocity.y / 20.0
    ovx = opponent.velocity.x / 20.0
    ovy = opponent.velocity.y / 20.0

    # Damage (0-999 -> 0-1)
    p_dmg = player.damage_percent / 999.0
    o_dmg = opponent.damage_percent / 999.0

    # Stocks (0-3 -> 0-1)
    p_stocks = player.stocks / CONFIG.stock_count
    o_stocks = opponent.stocks / CONFIG.stock_count

    # State one-hot (simplified - just a few key states)
    state_vec = np.zeros(5)
    if player.state == FighterState.IDLE:
        state_vec[0] = 1
    elif player.state in (FighterState.WALKING, FighterState.RUNNING):
        state_vec[1] = 1
    elif player.state in (FighterState.JUMPING, FighterState.FALLING):
        state_vec[2] = 1
    elif player.state == FighterState.ATTACKING:
        state_vec[3] = 1
    elif player.state == FighterState.HITSTUN:
        state_vec[4] = 1

    # Facing direction
    facing = 1.0 if player.facing_right else -1.0
    o_facing = 1.0 if opponent.facing_right else -1.0

    # Grounded
    grounded = 1.0 if player.grounded else 0.0
    o_grounded = 1.0 if opponent.grounded else 0.0

    # Combine all observations
    obs = np.array([
        px, py, pvx, pvy,
        ox, oy, ovx, ovy,
        p_dmg, o_dmg,
        p_stocks, o_stocks,
        facing, o_facing,
        grounded, o_grounded,
        *state_vec
    ], dtype=np.float32)

    return obs


def get_action_from_index(action_idx: int) -> InputState:
    """
    Convert a discrete action index to InputState.

    Action space:
    0: Nothing
    1: Left
    2: Right
    3: Up/Jump
    4: Down
    5: Attack
    6: Left + Attack
    7: Right + Attack
    8: Up + Attack
    9: Down + Attack
    """
    inp = InputState()

    if action_idx == 0:
        pass  # Nothing
    elif action_idx == 1:
        inp.left = True
    elif action_idx == 2:
        inp.right = True
    elif action_idx == 3:
        inp.up = True
        inp.jump = True
    elif action_idx == 4:
        inp.down = True
    elif action_idx == 5:
        inp.attack = True
    elif action_idx == 6:
        inp.left = True
        inp.attack = True
    elif action_idx == 7:
        inp.right = True
        inp.attack = True
    elif action_idx == 8:
        inp.up = True
        inp.attack = True
    elif action_idx == 9:
        inp.down = True
        inp.attack = True

    return inp


ACTION_SPACE_SIZE = 10
OBSERVATION_SIZE = 21  # Size of get_observation output


if __name__ == "__main__":
    # Quick test
    print("Creating game...")
    game = create_game()
    print(f"Stage: {game.stage.name}")
    print(f"Fighters: {len(game.fighters)}")

    # Run a few frames with random inputs
    def random_input(state: GameState, player_id: int) -> InputState:
        inp = InputState()
        inp.right = np.random.random() > 0.5
        inp.attack = np.random.random() > 0.9
        inp.jump = np.random.random() > 0.95
        return inp

    final_state = run_headless_match(game, random_input, max_frames=300)
    print(f"Match ended at frame {final_state.frame}")
    print(f"P1 stocks: {final_state.fighters[0].stocks}, damage: {final_state.fighters[0].damage_percent:.1f}%")
    print(f"P2 stocks: {final_state.fighters[1].stocks}, damage: {final_state.fighters[1].damage_percent:.1f}%")
