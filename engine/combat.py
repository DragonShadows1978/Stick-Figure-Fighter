"""Attacks, hitboxes and projectiles."""
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from .geometry import AABB, Vector2

if TYPE_CHECKING:
    from .fighter import Fighter
    from .loop import GameLoop


class AttackType(Enum):
    """Move slots. Which slot a button press maps to is decided by the GameLoop."""
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


@dataclass
class Hitbox:
    """Hitbox for attack collision detection."""
    offset: Vector2  # Offset from fighter position (for facing right)
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
class Projectile:
    """A projectile in the game world."""
    owner_id: int           # Player who spawned it
    position: Vector2       # Current position
    velocity: Vector2       # Movement per frame
    hitbox: Hitbox          # Damage/knockback data
    lifetime: int           # Frames remaining before despawn
    active: bool = True     # Whether it can still hit
    has_gravity: bool = False
    reflected: bool = False

    # Visual properties
    width: float = 20.0
    height: float = 20.0
    color: Tuple[int, int, int] = (255, 200, 50)

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
        """Reflect this projectile. The caller reassigns owner_id."""
        self.velocity.x = -self.velocity.x
        self.velocity.y = -self.velocity.y * 0.5  # Reduce vertical
        self.reflected = True
        self.hitbox.damage *= 1.25


@dataclass
class ProjectileSpec:
    """
    Data describing a projectile an attack spawns when it starts.

    `offset` and `velocity` are given for a fighter facing right and are
    mirrored horizontally when facing left.
    """
    offset: Vector2
    velocity: Vector2
    hitbox: Hitbox
    lifetime: int
    has_gravity: bool = False
    width: float = 20.0
    height: float = 20.0
    color: Tuple[int, int, int] = (255, 200, 50)

    def spawn(self, owner_id: int, position: Vector2, facing_right: bool) -> Projectile:
        direction = 1 if facing_right else -1
        return Projectile(
            owner_id=owner_id,
            position=Vector2(position.x + direction * self.offset.x,
                             position.y + self.offset.y),
            velocity=Vector2(direction * self.velocity.x, self.velocity.y),
            hitbox=replace(self.hitbox),  # projectiles mutate their hitbox (reflect)
            lifetime=self.lifetime,
            has_gravity=self.has_gravity,
            width=self.width,
            height=self.height,
            color=self.color,
        )


@dataclass
class AttackData:
    """
    Data for a single move.

    Special mechanics are opt-in per move rather than per character:
      - is_counter:   enter counter stance; getting hit during the window
                      triggers this move's hitboxes as a counterattack,
                      boosted by counter_multiplier
      - is_reflector: reflect projectiles while a hitbox is active
      - super_armor:  take damage without knockback for the whole move
      - projectile:   spawn this projectile when the move starts
      - on_start:     arbitrary hook, called as on_start(loop, fighter)
                      after the built-in mechanics are applied
    """
    attack_type: AttackType
    total_frames: int
    hitboxes: List[Hitbox]
    startup_frames: int  # Frames before hitbox becomes active
    recovery_frames: int  # Frames after hitbox ends before can act
    landing_lag: int = 0  # Extra landing lag during this move (aerials)
    spawns_projectile: bool = False
    super_armor: bool = False
    is_counter: bool = False
    is_reflector: bool = False
    counter_multiplier: float = 1.5  # damage/knockback boost when a counter triggers
    projectile: Optional[ProjectileSpec] = None
    on_start: Optional[Callable[['GameLoop', 'Fighter'], None]] = field(
        default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.projectile is not None:
            self.spawns_projectile = True


def _single_hit(attack_type, total, offset, w, h, dmg, kb, kbs, angle, active,
                startup, recovery) -> AttackData:
    return AttackData(
        attack_type=attack_type,
        total_frames=total,
        hitboxes=[Hitbox(offset=offset, width=w, height=h, damage=dmg,
                         knockback_base=kb, knockback_scaling=kbs, angle=angle,
                         active_frames=active)],
        startup_frames=startup,
        recovery_frames=recovery,
    )


def default_attack(attack_type: AttackType) -> AttackData:
    """
    Generic fallback move for any slot a fighter's moveset doesn't define.

    Returns a fresh object every call.
    """
    A = AttackType
    if attack_type == A.JAB:
        return _single_hit(attack_type, 20, Vector2(20, -50), 40, 30, 3.0, 2.0, 0.05, 45, (5, 8), 5, 12)
    if attack_type in (A.FORWARD_TILT, A.FORWARD_AIR):
        return _single_hit(attack_type, 30, Vector2(25, -55), 55, 35, 8.0, 4.0, 0.08, 40, (8, 12), 8, 18)
    if attack_type in (A.UP_TILT, A.UP_AIR):
        return _single_hit(attack_type, 28, Vector2(-20, -90), 60, 40, 7.0, 5.0, 0.1, 85, (6, 10), 6, 18)
    if attack_type in (A.DOWN_TILT, A.DOWN_AIR):
        angle = 80 if attack_type == A.DOWN_TILT else 270
        return _single_hit(attack_type, 25, Vector2(15, -20), 50, 25, 6.0, 3.0, 0.06, angle, (7, 10), 7, 15)
    if attack_type == A.BACK_AIR:
        return _single_hit(attack_type, 32, Vector2(-30, -55), 50, 35, 12.0, 6.0, 0.12, 35, (10, 14), 10, 18)
    if attack_type == A.NEUTRAL_AIR:
        return _single_hit(attack_type, 26, Vector2(-25, -60), 70, 60, 6.0, 3.5, 0.07, 45, (5, 18), 5, 8)
    return _single_hit(attack_type, 20, Vector2(20, -50), 40, 30, 5.0, 3.0, 0.06, 45, (5, 10), 5, 10)
