"""
Stick-figure poses and skeletons.

Pose tables (offsets relative to a fighter's position), the state -> pose
mapping, and Skeleton: the resolved screen-space joints every theme draws
from. PoseAnimator layers smoothing, a continuous walk cycle and
attack-specific poses on top of the base table.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from engine import AttackType, FighterState


@dataclass
class StickFigurePose:
    """
    Pose data for stick figure animation.

    Coordinate System:
    - All pose coordinates are relative to fighter.position
    - hip_y=-10 means hip is 10 pixels ABOVE position.y
    - foot offsets are relative to hip (typically +50 from hip)
    - Actual feet render at: position.y + hip_y + foot_offset ≈ position.y + 40
    - FOOT_OFFSET constant in game_engine.py defines this rendering offset
    """
    # Head position (relative to center)
    head_y: float = -65
    head_radius: float = 12

    # Body (spine)
    shoulder_y: float = -50
    hip_y: float = -10

    # Arms (relative to shoulder)
    left_elbow: Tuple[float, float] = (-15, 5)
    left_hand: Tuple[float, float] = (-25, 20)
    right_elbow: Tuple[float, float] = (15, 5)
    right_hand: Tuple[float, float] = (25, 20)

    # Legs (relative to hip)
    left_knee: Tuple[float, float] = (-10, 25)
    left_foot: Tuple[float, float] = (-15, 50)
    right_knee: Tuple[float, float] = (10, 25)
    right_foot: Tuple[float, float] = (15, 50)

    # Horizontal offset of shoulders/head relative to hips (positive = forward).
    # Ignored by the classic renderer.
    lean: float = 0.0


# Pre-defined poses for different states
POSE_IDLE = StickFigurePose()

POSE_WALKING = StickFigurePose(
    left_elbow=(-12, 8),
    left_hand=(-18, 22),
    right_elbow=(18, 2),
    right_hand=(28, 15),
    left_knee=(-15, 22),
    left_foot=(-20, 48),
    right_knee=(8, 28),
    right_foot=(5, 52)
)

POSE_JUMPING = StickFigurePose(
    head_y=-70,
    shoulder_y=-55,
    hip_y=-15,
    left_elbow=(-20, -5),
    left_hand=(-25, -15),
    right_elbow=(20, -5),
    right_hand=(25, -15),
    left_knee=(-12, 15),
    left_foot=(-15, 35),
    right_knee=(12, 15),
    right_foot=(15, 35)
)

POSE_FALLING = StickFigurePose(
    head_y=-60,
    shoulder_y=-45,
    hip_y=-5,
    left_elbow=(-18, 10),
    left_hand=(-25, 25),
    right_elbow=(18, 10),
    right_hand=(25, 25),
    left_knee=(-8, 20),
    left_foot=(-12, 45),
    right_knee=(8, 20),
    right_foot=(12, 45)
)

POSE_ATTACK = StickFigurePose(
    left_elbow=(-10, 5),
    left_hand=(-15, 15),
    right_elbow=(25, -10),
    right_hand=(50, -5),
    left_knee=(-12, 25),
    left_foot=(-18, 48),
    right_knee=(8, 28),
    right_foot=(5, 52)
)

POSE_HITSTUN = StickFigurePose(
    head_y=-58,
    shoulder_y=-42,
    hip_y=-8,
    left_elbow=(-22, 15),
    left_hand=(-30, 30),
    right_elbow=(22, 15),
    right_hand=(30, 30),
    left_knee=(-15, 20),
    left_foot=(-22, 42),
    right_knee=(10, 22),
    right_foot=(8, 45)
)


POSE_SPECIAL = StickFigurePose(
    head_y=-68,
    shoulder_y=-52,
    hip_y=-12,
    left_elbow=(-8, -8),
    left_hand=(-12, -20),
    right_elbow=(30, -15),
    right_hand=(55, -20),  # Arm extended for special
    left_knee=(-8, 25),
    left_foot=(-10, 50),
    right_knee=(12, 22),
    right_foot=(15, 48)
)

POSE_SHIELDING = StickFigurePose(
    head_y=-60,
    shoulder_y=-45,
    hip_y=-8,
    left_elbow=(-5, 10),
    left_hand=(5, 20),  # Arms crossed in front
    right_elbow=(5, 10),
    right_hand=(-5, 20),
    left_knee=(-8, 22),
    left_foot=(-10, 48),
    right_knee=(8, 22),
    right_foot=(10, 48)
)

POSE_DODGING = StickFigurePose(
    head_y=-55,
    shoulder_y=-40,
    hip_y=-5,
    left_elbow=(-18, 12),
    left_hand=(-25, 25),
    right_elbow=(18, 12),
    right_hand=(25, 25),
    left_knee=(-15, 18),
    left_foot=(-20, 42),
    right_knee=(5, 25),
    right_foot=(0, 50)
)

POSE_COUNTER = StickFigurePose(
    head_y=-65,
    shoulder_y=-50,
    hip_y=-10,
    left_elbow=(-15, 0),
    left_hand=(-10, 15),  # Defensive stance
    right_elbow=(15, 0),
    right_hand=(10, 15),
    left_knee=(-12, 25),
    left_foot=(-18, 50),
    right_knee=(12, 25),
    right_foot=(18, 50)
)

POSE_REFLECTING = StickFigurePose(
    head_y=-65,
    shoulder_y=-50,
    hip_y=-10,
    left_elbow=(-20, -5),
    left_hand=(-30, 5),
    right_elbow=(20, -5),
    right_hand=(30, 5),  # Arms out creating reflector
    left_knee=(-8, 25),
    left_foot=(-12, 50),
    right_knee=(8, 25),
    right_foot=(12, 50)
)

POSE_SHIELD_STUN = StickFigurePose(
    head_y=-55,
    shoulder_y=-40,
    hip_y=-5,
    left_elbow=(-20, 20),
    left_hand=(-25, 35),  # Arms limp/dizzy
    right_elbow=(20, 20),
    right_hand=(25, 35),
    left_knee=(-8, 25),
    left_foot=(-10, 50),
    right_knee=(8, 25),
    right_foot=(10, 50)
)

# =============================================================================
# GAME FEEL POSES (Cycle 4)
# =============================================================================

# Landing squash - compressed pose for landing recovery
POSE_LANDING_SQUASH = StickFigurePose(
    head_y=-50,          # Head lower (squashed)
    head_radius=10,      # Slightly compressed head
    shoulder_y=-38,      # Shoulders compressed down
    hip_y=-2,            # Hips near ground
    left_elbow=(-18, 8),
    left_hand=(-22, 22),
    right_elbow=(18, 8),
    right_hand=(22, 22),
    left_knee=(-15, 18),   # Bent legs
    left_foot=(-20, 38),   # Feet closer to hip
    right_knee=(15, 18),
    right_foot=(20, 38)
)

# Jump anticipation - crouch before jump
POSE_JUMP_CROUCH = StickFigurePose(
    head_y=-52,          # Crouched
    head_radius=11,
    shoulder_y=-40,
    hip_y=-5,
    left_elbow=(-15, 10),
    left_hand=(-20, 25),
    right_elbow=(15, 10),
    right_hand=(20, 25),
    left_knee=(-12, 20),   # Bent knees for crouch
    left_foot=(-15, 42),
    right_knee=(12, 20),
    right_foot=(15, 42)
)

# Jump stretch - elongated at peak of jump
POSE_JUMP_STRETCH = StickFigurePose(
    head_y=-78,          # Head higher (stretched)
    head_radius=13,      # Slightly stretched head
    shoulder_y=-62,      # Shoulders stretched up
    hip_y=-18,           # Hips higher
    left_elbow=(-22, -10),
    left_hand=(-28, -20),
    right_elbow=(22, -10),
    right_hand=(28, -20),
    left_knee=(-10, 18),   # Legs slightly tucked
    left_foot=(-12, 38),
    right_knee=(10, 18),
    right_foot=(12, 38)
)

# Attack wind-up (anticipation frame)
POSE_ATTACK_WINDUP = StickFigurePose(
    head_y=-62,
    shoulder_y=-48,
    hip_y=-12,
    left_elbow=(-15, 5),
    left_hand=(-20, 15),
    right_elbow=(-10, 8),    # Arm pulled back for wind-up
    right_hand=(-15, 20),
    left_knee=(-10, 25),
    left_foot=(-15, 50),
    right_knee=(12, 25),
    right_foot=(18, 50)
)


# =============================================================================
# ENTRANCE POSES (Cycle 14)
# =============================================================================

# Entering pose - falling from sky with dramatic pose
POSE_ENTERING = StickFigurePose(
    head_y=-68,
    head_radius=12,
    shoulder_y=-52,
    hip_y=-12,
    left_elbow=(-22, -8),     # Arms spread wide
    left_hand=(-35, -12),
    right_elbow=(22, -8),
    right_hand=(35, -12),
    left_knee=(-10, 22),      # Legs together
    left_foot=(-12, 48),
    right_knee=(10, 22),
    right_foot=(12, 48)
)


def get_pose_for_state(state: FighterState, frame: int, fighter=None) -> StickFigurePose:
    """Get the appropriate pose for a fighter state with game feel improvements (Cycle 4)."""

    # ==========================================================================
    # IDLE with breathing animation (Cycle 4)
    # ==========================================================================
    if state == FighterState.IDLE:
        # Subtle breathing animation - head bobs slightly every ~60 frames
        breath_cycle = math.sin(frame * 0.1) * 2  # Subtle oscillation
        # Create a breathing variation of POSE_IDLE
        return StickFigurePose(
            head_y=POSE_IDLE.head_y + breath_cycle,
            head_radius=POSE_IDLE.head_radius,
            shoulder_y=POSE_IDLE.shoulder_y + breath_cycle * 0.5,
            hip_y=POSE_IDLE.hip_y,
            left_elbow=POSE_IDLE.left_elbow,
            left_hand=POSE_IDLE.left_hand,
            right_elbow=POSE_IDLE.right_elbow,
            right_hand=POSE_IDLE.right_hand,
            left_knee=POSE_IDLE.left_knee,
            left_foot=POSE_IDLE.left_foot,
            right_knee=POSE_IDLE.right_knee,
            right_foot=POSE_IDLE.right_foot
        )

    # ==========================================================================
    # Walking with smoother animation
    # ==========================================================================
    elif state in (FighterState.WALKING, FighterState.RUNNING):
        # Smoother walk cycle with arm swing
        walk_phase = math.sin(frame * 0.25)  # Smoother walk cycle
        if walk_phase > 0:
            return POSE_WALKING
        else:
            return StickFigurePose(
                left_elbow=(18, 2),
                left_hand=(28, 15),
                right_elbow=(-12, 8),
                right_hand=(-18, 22),
                left_knee=(8, 28),
                left_foot=(5, 52),
                right_knee=(-15, 22),
                right_foot=(-20, 48)
            )

    # ==========================================================================
    # Jump with squash/stretch (Cycle 4)
    # ==========================================================================
    elif state == FighterState.JUMPING:
        # Get vertical velocity if fighter is available
        if fighter is not None:
            vy = fighter.velocity.y
            state_frame = fighter.state_frame

            # First 3 frames: stretch as we leave ground (upward momentum)
            if state_frame < 3 and vy < -5:
                return POSE_JUMP_STRETCH

            # Rising: normal jump pose
            elif vy < 0:
                return POSE_JUMPING

            # Peak of jump: slight float pose
            elif abs(vy) < 3:
                return POSE_JUMP_STRETCH

            # Falling: transition to fall pose
            else:
                return POSE_FALLING
        return POSE_JUMPING

    # ==========================================================================
    # Landing with squash (Cycle 4)
    # ==========================================================================
    elif state == FighterState.LANDING:
        # Check landing frame for squash effect
        if fighter is not None and fighter.state_frame < 4:
            return POSE_LANDING_SQUASH
        return POSE_IDLE

    elif state == FighterState.FALLING:
        return POSE_FALLING

    # ==========================================================================
    # Attack with anticipation frames (Cycle 4)
    # ==========================================================================
    elif state == FighterState.ATTACKING:
        # First 2-3 frames: wind-up/anticipation
        if fighter is not None and fighter.attack_frame < 3:
            attack_data = fighter.current_attack
            # Only show wind-up for attacks with >5 frame startup
            if attack_data is not None and attack_data.startup_frames > 5:
                return POSE_ATTACK_WINDUP
        return POSE_ATTACK

    elif state == FighterState.SPECIAL:
        # Special moves also get anticipation on slow attacks
        if fighter is not None and fighter.attack_frame < 2:
            attack_data = fighter.current_attack
            if attack_data is not None and attack_data.startup_frames > 8:
                return POSE_ATTACK_WINDUP
        return POSE_SPECIAL

    elif state == FighterState.HITSTUN:
        return POSE_HITSTUN
    elif state == FighterState.SHIELDING:
        return POSE_SHIELDING
    elif state == FighterState.DODGING:
        return POSE_DODGING
    elif state == FighterState.COUNTER:
        return POSE_COUNTER
    elif state == FighterState.REFLECTING:
        return POSE_REFLECTING
    elif state == FighterState.SHIELD_STUN:
        return POSE_SHIELD_STUN
    elif state == FighterState.CROUCH:
        return POSE_JUMP_CROUCH
    elif state == FighterState.ENTERING:
        return POSE_ENTERING
    else:
        return POSE_IDLE


# =============================================================================
# ATTACK-SPECIFIC POSES (used by themed renderers)
# "right_*" limbs are the front limbs (x is mirrored by facing).
# =============================================================================

def _p(**kw) -> StickFigurePose:
    return StickFigurePose(**kw)


_STANCE_LEGS = dict(left_knee=(-10, 24), left_foot=(-18, 50), right_knee=(10, 22), right_foot=(16, 50))

ATTACK_POSES: Dict[AttackType, StickFigurePose] = {
    AttackType.JAB: _p(left_elbow=(6, 8), left_hand=(14, -4), right_elbow=(20, -2), right_hand=(44, -4),
                       lean=3, **_STANCE_LEGS),
    AttackType.FORWARD_TILT: _p(lean=-7, left_elbow=(-14, 4), left_hand=(-24, 14), right_elbow=(10, 8),
                                right_hand=(4, 20), left_knee=(-6, 24), left_foot=(-10, 50),
                                right_knee=(22, -4), right_foot=(48, -6)),
    AttackType.UP_TILT: _p(head_y=-68, shoulder_y=-53, left_elbow=(-12, 8), left_hand=(-6, 18),
                           right_elbow=(10, -18), right_hand=(12, -42), **_STANCE_LEGS),
    AttackType.DOWN_TILT: _p(head_y=-40, head_radius=11, shoulder_y=-28, hip_y=2, lean=6,
                             left_elbow=(-14, 10), left_hand=(-18, 24), right_elbow=(14, 10),
                             right_hand=(22, 22), left_knee=(-12, 18), left_foot=(-8, 38),
                             right_knee=(22, 16), right_foot=(48, 36)),
    AttackType.NEUTRAL_AIR: _p(left_elbow=(-22, -6), left_hand=(-38, -12), right_elbow=(22, -6),
                               right_hand=(38, -12), left_knee=(-16, 16), left_foot=(-28, 30),
                               right_knee=(16, 16), right_foot=(28, 30)),
    AttackType.FORWARD_AIR: _p(lean=-4, left_elbow=(-16, 2), left_hand=(-26, -6), right_elbow=(12, 8),
                               right_hand=(4, 18), left_knee=(-4, 20), left_foot=(-14, 32),
                               right_knee=(20, 4), right_foot=(44, 0)),
    AttackType.BACK_AIR: _p(lean=7, left_elbow=(-10, 8), left_hand=(-4, 20), right_elbow=(16, 2),
                            right_hand=(26, 10), left_knee=(-20, 2), left_foot=(-46, 0),
                            right_knee=(10, 18), right_foot=(4, 32)),
    AttackType.UP_AIR: _p(head_y=-60, shoulder_y=-46, left_elbow=(-14, 12), left_hand=(-20, 24),
                          right_elbow=(14, 12), right_hand=(18, 26), left_knee=(-8, 20),
                          left_foot=(-12, 36), right_knee=(10, -14), right_foot=(14, -40)),
    AttackType.DOWN_AIR: _p(head_y=-70, shoulder_y=-55, hip_y=-14, left_elbow=(-16, -10),
                            left_hand=(-22, -24), right_elbow=(16, -10), right_hand=(22, -24),
                            left_knee=(-6, 26), left_foot=(-4, 56), right_knee=(6, 26), right_foot=(4, 56)),
    AttackType.NEUTRAL_SPECIAL: _p(lean=4, left_elbow=(12, 6), left_hand=(36, 2), right_elbow=(18, 0),
                                   right_hand=(42, -4), **_STANCE_LEGS),
    AttackType.SIDE_SPECIAL: _p(lean=9, head_y=-62, shoulder_y=-47, left_elbow=(-14, 6),
                                left_hand=(-26, 12), right_elbow=(22, -6), right_hand=(50, -8),
                                left_knee=(-16, 20), left_foot=(-32, 46), right_knee=(18, 18),
                                right_foot=(32, 44)),
    AttackType.UP_SPECIAL: _p(head_y=-70, shoulder_y=-55, hip_y=-14, left_elbow=(-10, -18),
                              left_hand=(-12, -42), right_elbow=(10, -20), right_hand=(16, -46),
                              left_knee=(-6, 22), left_foot=(-10, 44), right_knee=(8, 18), right_foot=(14, 36)),
    AttackType.DOWN_SPECIAL: _p(head_y=-56, shoulder_y=-42, hip_y=-4, left_elbow=(-18, 8),
                                left_hand=(-30, 18), right_elbow=(18, 8), right_hand=(30, 18),
                                left_knee=(-14, 20), left_foot=(-22, 44), right_knee=(14, 20),
                                right_foot=(22, 44)),
}
for _a, _b in ((AttackType.FORWARD_SMASH, AttackType.FORWARD_TILT), (AttackType.UP_SMASH, AttackType.UP_TILT),
               (AttackType.DOWN_SMASH, AttackType.DOWN_TILT)):
    ATTACK_POSES[_a] = ATTACK_POSES[_b]

POSE_GUARD = _p(left_elbow=(6, 10), left_hand=(14, -2), right_elbow=(14, 8), right_hand=(22, -4),
                **_STANCE_LEGS)


# =============================================================================
# SKELETON
# =============================================================================

_FIELDS = ("head_y", "head_radius", "shoulder_y", "hip_y", "lean")
_PAIRS = ("left_elbow", "left_hand", "right_elbow", "right_hand",
          "left_knee", "left_foot", "right_knee", "right_foot")


def pose_vec(p: StickFigurePose) -> Tuple[float, ...]:
    out = [getattr(p, f) for f in _FIELDS]
    for name in _PAIRS:
        out.extend(getattr(p, name))
    return tuple(out)


def vec_pose(v: Tuple[float, ...]) -> StickFigurePose:
    kw = dict(zip(_FIELDS, v[:5]))
    for i, name in enumerate(_PAIRS):
        kw[name] = (v[5 + i * 2], v[6 + i * 2])
    return StickFigurePose(**kw)


def lerp_vec(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


Pt = Tuple[float, float]


@dataclass
class Skeleton:
    """Screen-space joints. back_* are the far-side limbs, front_* the near/forward ones."""
    head: Pt
    head_r: float
    neck: Pt          # base of the neck (top of spine)
    hip: Pt
    back_elbow: Pt
    back_hand: Pt
    front_elbow: Pt
    front_hand: Pt
    back_knee: Pt
    back_foot: Pt
    front_knee: Pt
    front_foot: Pt
    facing: int = 1

    def chains(self):
        """Polylines: (name, [points]) back-to-front drawing order."""
        return [
            ("back_leg", [self.hip, self.back_knee, self.back_foot]),
            ("back_arm", [self.neck, self.back_elbow, self.back_hand]),
            ("spine", [self.neck, self.hip]),
            ("front_leg", [self.hip, self.front_knee, self.front_foot]),
            ("front_arm", [self.neck, self.front_elbow, self.front_hand]),
        ]

    def joints(self):
        return [self.head, self.neck, self.hip, self.back_elbow, self.back_hand, self.front_elbow,
                self.front_hand, self.back_knee, self.back_foot, self.front_knee, self.front_foot]

    def transformed(self, dx: float = 0.0, dy: float = 0.0, angle: float = 0.0,
                    pivot: Optional[Pt] = None, scale: float = 1.0) -> "Skeleton":
        px, py = pivot or self.hip
        c, s = math.cos(angle), math.sin(angle)

        def f(p):
            x, y = (p[0] - px) * scale, (p[1] - py) * scale
            return (px + x * c - y * s + dx, py + x * s + y * c + dy)

        return Skeleton(f(self.head), self.head_r * scale, f(self.neck), f(self.hip),
                        f(self.back_elbow), f(self.back_hand), f(self.front_elbow), f(self.front_hand),
                        f(self.back_knee), f(self.back_foot), f(self.front_knee), f(self.front_foot),
                        self.facing)

    @property
    def center(self) -> Pt:
        return ((self.neck[0] + self.hip[0]) / 2, (self.neck[1] + self.hip[1]) / 2)


def skeleton_from_pose(pose: StickFigurePose, x: float, y: float, facing: int) -> Skeleton:
    sh_x = x + pose.lean * facing
    sh = (sh_x, y + pose.shoulder_y)
    hip = (x, y + pose.hip_y)

    def arm(off):
        return (sh[0] + off[0] * facing, sh[1] + off[1])

    def leg(off):
        return (hip[0] + off[0] * facing, hip[1] + off[1])

    head = (x + pose.lean * facing * 1.25, y + pose.head_y)
    return Skeleton(head, pose.head_radius, sh, hip,
                    arm(pose.left_elbow), arm(pose.left_hand), arm(pose.right_elbow), arm(pose.right_hand),
                    leg(pose.left_knee), leg(pose.left_foot), leg(pose.right_knee), leg(pose.right_foot),
                    facing)


def _attack_phase_pose(fighter) -> Optional[StickFigurePose]:
    """Blend windup -> strike -> recovery for the fighter's current attack."""
    atk = fighter.current_attack
    if atk is None:
        return None
    strike = ATTACK_POSES.get(atk.attack_type)
    if strike is None:
        return None
    f = fighter.attack_frame
    startup = max(1, atk.startup_frames)
    active_end = max([hb.active_frames[1] for hb in atk.hitboxes], default=startup + 3)
    sv = pose_vec(strike)
    if f < startup:
        wind = pose_vec(POSE_ATTACK_WINDUP if fighter.grounded else POSE_FALLING)
        t = f / startup
        return vec_pose(lerp_vec(wind, sv, t * t))
    if f <= active_end:
        return strike
    rec = max(1, atk.total_frames - active_end)
    t = min(1.0, (f - active_end) / rec)
    rest = pose_vec(POSE_GUARD if fighter.grounded else POSE_FALLING)
    return vec_pose(lerp_vec(sv, rest, t * t * 0.85))


class PoseAnimator:
    """
    Per-fighter animation state for themed renderers: continuous walk cycle,
    attack phases, pose smoothing and knockback tumble.
    """

    def __init__(self):
        self._vec: Dict[int, Tuple[float, ...]] = {}
        self._walk: Dict[int, float] = {}
        self._spin: Dict[int, float] = {}
        self._last_frame: Dict[int, int] = {}

    def reset(self):
        self.__init__()

    def _walk_pose(self, fighter, phase: float) -> StickFigurePose:
        speed = min(1.0, abs(fighter.velocity.x) / max(1.0, fighter.stats.walk_speed))
        amp = 6 + 12 * speed
        s, c = math.sin(phase), math.cos(phase)
        bob = abs(c) * 2.5 * speed
        return StickFigurePose(
            head_y=-65 + bob, shoulder_y=-50 + bob, hip_y=-10 + bob * 0.6, lean=3 * speed,
            left_elbow=(-8 - s * amp * 0.6, 6), left_hand=(-10 - s * amp * 1.0, 20),
            right_elbow=(8 + s * amp * 0.6, 6), right_hand=(10 + s * amp * 1.0, 20),
            left_knee=(s * amp * 0.7 + 3, 24 - max(0.0, -c) * 6 * speed),
            left_foot=(s * amp, 50 - bob * 0.6 - max(0.0, -c) * 9 * speed),
            right_knee=(-s * amp * 0.7 + 3, 24 - max(0.0, c) * 6 * speed),
            right_foot=(-s * amp, 50 - bob * 0.6 - max(0.0, c) * 9 * speed),
        )

    def target_pose(self, fighter, frame: int) -> Tuple[StickFigurePose, float]:
        """(pose, blend speed) for this frame."""
        st = fighter.state
        pid = fighter.player_id
        if st in (FighterState.WALKING, FighterState.RUNNING):
            phase = self._walk.get(pid, 0.0) + abs(fighter.velocity.x) * 0.075
            self._walk[pid] = phase
            return self._walk_pose(fighter, phase), 0.5
        if st in (FighterState.ATTACKING, FighterState.SPECIAL):
            p = _attack_phase_pose(fighter)
            if p is not None:
                return p, 0.7
        if st == FighterState.HITSTUN:
            return POSE_HITSTUN, 0.8
        if st == FighterState.IDLE:
            return get_pose_for_state(st, frame, fighter), 0.25
        return get_pose_for_state(st, fighter.state_frame, fighter), 0.45

    def skeleton(self, fighter, x: float, y: float, frame: int) -> Skeleton:
        pid = fighter.player_id
        facing = 1 if fighter.facing_right else -1
        pose, speed = self.target_pose(fighter, frame)
        target = pose_vec(pose)
        prev = self._vec.get(pid)
        if prev is not None and self._last_frame.get(pid) != frame:
            vec = lerp_vec(prev, target, speed)
        else:
            vec = prev if prev is not None else target
        self._vec[pid] = vec
        self._last_frame[pid] = frame
        skel = skeleton_from_pose(vec_pose(vec), x, y, facing)

        # Tumble on heavy knockback, lean into lighter knockback
        spin = self._spin.get(pid, 0.0)
        if fighter.state == FighterState.HITSTUN:
            kb = math.hypot(fighter.velocity.x, fighter.velocity.y)
            if kb > 11:
                spin += (0.18 + kb * 0.012) * (1 if fighter.velocity.x >= 0 else -1)
            else:
                spin += (max(-0.35, min(0.35, fighter.velocity.x * 0.04)) - spin) * 0.3
        else:
            # unwind to the nearest upright angle
            target_spin = round(spin / (2 * math.pi)) * 2 * math.pi
            spin += (target_spin - spin) * 0.35
            if abs(spin - target_spin) < 0.02:
                spin = 0.0
        self._spin[pid] = spin
        if abs(spin) > 1e-3:
            skel = skel.transformed(angle=spin, pivot=skel.center)
        return skel
