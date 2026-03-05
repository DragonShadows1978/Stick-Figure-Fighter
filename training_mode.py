"""
Stick Figure Fighter - Training Mode (Cycle 6 + Cycle 13 + Cycle 16)

Provides training features:
- Frame data overlay showing startup/active/recovery frames
- CPU opponent with AI difficulty levels (1-5)
- Damage reset (R key), position reset (T key)
- Infinite jumps toggle for recovery practice
- CPU DI and teching at higher difficulties
- Record/playback system for practicing against specific sequences
- Cycle 13: Frame advantage display on hit
- Cycle 16: Input history display, improved hitbox visualization, move list display
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict
import random
import math
import time

from game_engine import (
    GameState, Fighter, FighterState, AttackType, CONFIG, Vector2
)


class CPUBehavior(Enum):
    """CPU behavior modes."""
    STAND = auto()       # Just stands still (for combo practice)
    JUMP = auto()        # Repeatedly jumps (for juggle practice)
    WALK = auto()        # Walks back and forth
    SHIELD = auto()      # Always shielding (for shield pressure practice)
    ATTACK = auto()      # Random attacks (for defensive practice)
    RANDOM = auto()      # Mix of all behaviors


class CPUDifficulty(Enum):
    """CPU difficulty levels (1-5)."""
    LEVEL_1 = 1  # Easiest: No reaction, no DI, no tech
    LEVEL_2 = 2  # Easy: Slow reaction, random DI
    LEVEL_3 = 3  # Medium: Normal reaction, decent DI
    LEVEL_4 = 4  # Hard: Fast reaction, good DI, sometimes techs
    LEVEL_5 = 5  # Very Hard: Frame-perfect reaction, optimal DI, always techs, combos


@dataclass
class CPUState:
    """State for CPU opponent."""
    enabled: bool = False
    behavior: CPUBehavior = CPUBehavior.STAND
    difficulty: CPUDifficulty = CPUDifficulty.LEVEL_3
    action_timer: int = 0
    current_action: str = "idle"
    action_duration: int = 60
    walk_direction: int = 1
    aggression: float = 0.3  # 0-1, higher = more attacks

    # Difficulty-based parameters (computed from difficulty)
    reaction_frames: int = 20  # Frames of delay before CPU reacts
    di_strength: float = 0.5  # How well CPU uses DI (0-1)
    tech_chance: float = 0.0  # Chance to tech when hitting ground
    combo_chance: float = 0.0  # Chance to follow up with combo


class CPUController:
    """CPU opponent with difficulty-based AI for training mode."""

    def __init__(self):
        self.state = CPUState()
        self.last_attack_frame = 0
        self.attack_cooldown = 30  # Minimum frames between attacks
        self.pending_reactions: List[Dict] = []  # Queue of delayed reactions

        # Combo state
        self.combo_target_locked = False
        self.combo_move_queue: List[AttackType] = []

    def enable(self, behavior: CPUBehavior = CPUBehavior.STAND,
               difficulty: CPUDifficulty = CPUDifficulty.LEVEL_3):
        """Enable CPU control."""
        self.state.enabled = True
        self.state.behavior = behavior
        self.set_difficulty(difficulty)
        self.state.action_timer = 0

    def disable(self):
        """Disable CPU control."""
        self.state.enabled = False

    def set_behavior(self, behavior: CPUBehavior):
        """Change CPU behavior mode."""
        self.state.behavior = behavior
        self.state.action_timer = 0

    def set_difficulty(self, difficulty: CPUDifficulty):
        """Set CPU difficulty and compute associated parameters."""
        self.state.difficulty = difficulty
        level = difficulty.value

        # Reaction time decreases with difficulty
        # Level 1: 30 frames, Level 5: 3 frames
        self.state.reaction_frames = max(3, 35 - level * 8)

        # DI strength increases with difficulty
        # Level 1: 0.1, Level 5: 1.0
        self.state.di_strength = 0.1 + (level - 1) * 0.225

        # Tech chance increases with difficulty
        # Level 1-2: 0%, Level 3: 30%, Level 4: 60%, Level 5: 95%
        tech_chances = [0.0, 0.0, 0.3, 0.6, 0.95]
        self.state.tech_chance = tech_chances[level - 1]

        # Combo chance increases with difficulty
        # Level 1-3: 0%, Level 4: 40%, Level 5: 80%
        combo_chances = [0.0, 0.0, 0.0, 0.4, 0.8]
        self.state.combo_chance = combo_chances[level - 1]

        # Attack cooldown decreases with difficulty
        self.attack_cooldown = max(10, 40 - level * 8)

    def cycle_difficulty(self) -> CPUDifficulty:
        """Cycle to next difficulty level."""
        difficulties = list(CPUDifficulty)
        current_idx = difficulties.index(self.state.difficulty)
        next_idx = (current_idx + 1) % len(difficulties)
        self.set_difficulty(difficulties[next_idx])
        return difficulties[next_idx]

    def update(self, cpu_fighter: Fighter, opponent: Fighter, frame: int) -> dict:
        """
        Update CPU and return input commands.

        Returns dict with keys: left, right, up, down, jump, attack, special, shield, grab
        """
        if not self.state.enabled:
            return {}

        inputs = {
            'left': False,
            'right': False,
            'up': False,
            'down': False,
            'jump': False,
            'attack': False,
            'special': False,
            'shield': False,
            'grab': False
        }

        # Get relative position to opponent
        dx = opponent.position.x - cpu_fighter.position.x
        dy = opponent.position.y - cpu_fighter.position.y
        distance = math.sqrt(dx * dx + dy * dy)
        facing_opponent = (dx > 0 and cpu_fighter.facing_right) or (dx < 0 and not cpu_fighter.facing_right)

        # Apply DI when in hitstun (difficulty-based)
        if cpu_fighter.state == FighterState.HITSTUN:
            inputs = self._apply_di(cpu_fighter, opponent, inputs)
            return inputs

        # Attempt tech when about to hit ground (difficulty-based)
        if self._should_tech(cpu_fighter):
            inputs['shield'] = True  # Tech input
            return inputs

        behavior = self.state.behavior

        # Handle RANDOM by picking a behavior periodically
        if behavior == CPUBehavior.RANDOM:
            if self.state.action_timer <= 0:
                behavior = random.choice([
                    CPUBehavior.STAND,
                    CPUBehavior.WALK,
                    CPUBehavior.JUMP,
                    CPUBehavior.SHIELD,
                    CPUBehavior.ATTACK
                ])
                self.state.action_timer = random.randint(60, 180)  # 1-3 seconds
            else:
                self.state.action_timer -= 1

        # STAND: Do nothing (default training dummy)
        if behavior == CPUBehavior.STAND:
            # Face opponent
            if not facing_opponent:
                if dx > 0:
                    inputs['right'] = True
                else:
                    inputs['left'] = True

        # JUMP: Periodically jump
        elif behavior == CPUBehavior.JUMP:
            if cpu_fighter.grounded and self.state.action_timer <= 0:
                inputs['jump'] = True
                self.state.action_timer = 45  # Jump every ~0.75 seconds
            self.state.action_timer -= 1

        # WALK: Walk back and forth
        elif behavior == CPUBehavior.WALK:
            # Change direction at stage edges or periodically
            if cpu_fighter.position.x < 200:
                self.state.walk_direction = 1
            elif cpu_fighter.position.x > CONFIG.screen_width - 200:
                self.state.walk_direction = -1
            elif self.state.action_timer <= 0:
                self.state.walk_direction *= -1
                self.state.action_timer = random.randint(60, 120)

            if self.state.walk_direction > 0:
                inputs['right'] = True
            else:
                inputs['left'] = True
            self.state.action_timer -= 1

        # SHIELD: Hold shield
        elif behavior == CPUBehavior.SHIELD:
            if cpu_fighter.shield_health > 0 and not cpu_fighter.shield_broken:
                inputs['shield'] = True

        # ATTACK: Aggressive CPU that attacks (difficulty-based intelligence)
        elif behavior == CPUBehavior.ATTACK:
            inputs = self._attack_behavior(cpu_fighter, opponent, inputs, frame, distance, dx, facing_opponent)

        return inputs

    def _apply_di(self, cpu_fighter: Fighter, opponent: Fighter, inputs: dict) -> dict:
        """Apply Directional Influence based on difficulty."""
        di_strength = self.state.di_strength

        # Roll for whether to apply DI at all
        if random.random() > di_strength:
            return inputs  # No DI this frame

        # Determine optimal DI direction based on knockback
        kb = cpu_fighter.knockback_vector
        if kb.magnitude() < 1:
            return inputs

        # DI perpendicular to knockback for survival
        # If being knocked left/right, DI up or toward stage center
        # If being knocked up/down, DI left or right toward stage center

        stage_center_x = CONFIG.screen_width / 2

        if abs(kb.x) > abs(kb.y):
            # Horizontal knockback - DI up or toward stage
            if cpu_fighter.position.y < CONFIG.screen_height / 2:
                inputs['down'] = True  # DI down to reduce horizontal trajectory
            else:
                inputs['up'] = True
        else:
            # Vertical knockback - DI toward stage center
            if cpu_fighter.position.x < stage_center_x:
                inputs['right'] = True
            else:
                inputs['left'] = True

        return inputs

    def _should_tech(self, cpu_fighter: Fighter) -> bool:
        """Check if CPU should attempt to tech."""
        if self.state.tech_chance <= 0:
            return False

        # Only tech when in hitstun and about to hit ground
        if cpu_fighter.state != FighterState.HITSTUN:
            return False

        # Check if falling and near ground
        if cpu_fighter.velocity.y <= 0:
            return False

        # Near ground (within 60 pixels)
        ground_y = 550  # Main platform Y
        if cpu_fighter.position.y < ground_y - 60:
            return False

        # Roll for tech
        return random.random() < self.state.tech_chance

    def _attack_behavior(self, cpu_fighter: Fighter, opponent: Fighter,
                         inputs: dict, frame: int, distance: float,
                         dx: float, facing_opponent: bool) -> dict:
        """Handle ATTACK behavior with difficulty-based intelligence."""

        difficulty = self.state.difficulty.value

        # Face opponent
        if not facing_opponent:
            if dx > 0:
                inputs['right'] = True
            else:
                inputs['left'] = True

        # Approach if far (smarter approach at higher difficulty)
        if distance > 150 and cpu_fighter.state == FighterState.IDLE:
            approach_speed = 0.6 + difficulty * 0.1  # Faster approach at higher difficulty

            if random.random() < approach_speed:
                if dx > 0:
                    inputs['right'] = True
                else:
                    inputs['left'] = True

            # Jump approach at higher difficulties
            if difficulty >= 4 and distance > 250 and random.random() < 0.05:
                inputs['jump'] = True

        # Attack if close and off cooldown
        if distance < 100 and (frame - self.last_attack_frame) > self.attack_cooldown:
            if cpu_fighter.state in (FighterState.IDLE, FighterState.WALKING):
                # Pick attack based on difficulty
                attack_roll = random.random()

                if difficulty >= 4:
                    # Smart attack selection at high difficulty
                    inputs = self._smart_attack_selection(cpu_fighter, opponent, inputs, dx, distance)
                else:
                    # Random attack at lower difficulty
                    if attack_roll < 0.4:
                        inputs['attack'] = True  # Jab/tilt
                    elif attack_roll < 0.6:
                        inputs['special'] = True  # Special
                    elif attack_roll < 0.75:
                        inputs['grab'] = True  # Grab
                    else:
                        # Smash attack (attack + direction)
                        inputs['attack'] = True
                        if dx > 0:
                            inputs['right'] = True
                        else:
                            inputs['left'] = True

                self.last_attack_frame = frame

        # Jump sometimes when far
        if distance > 200 and random.random() < 0.02:
            inputs['jump'] = True

        # Punish at high difficulty (react to opponent recovery lag)
        if difficulty >= 5:
            if opponent.state == FighterState.ATTACKING:
                # Opponent is attacking - prepare to punish
                if distance < 80 and opponent.attack_frame > 10:
                    inputs['shield'] = True  # Shield then punish

        return inputs

    def _smart_attack_selection(self, cpu_fighter: Fighter, opponent: Fighter,
                                 inputs: dict, dx: float, distance: float) -> dict:
        """Select attacks intelligently based on opponent state and position."""

        # If opponent is above, use up attacks
        if opponent.position.y < cpu_fighter.position.y - 30:
            inputs['up'] = True
            if distance < 60:
                inputs['attack'] = True  # Up tilt/smash
            else:
                inputs['special'] = True  # Up special (might be anti-air)

        # If opponent is at high percentage, use kill moves
        elif opponent.damage_percent > 100:
            if distance < 80:
                inputs['attack'] = True
                if dx > 0:
                    inputs['right'] = True
                else:
                    inputs['left'] = True

        # Otherwise, use combo starters
        else:
            if distance < 50:
                inputs['attack'] = True  # Jab
            else:
                inputs['attack'] = True
                if dx > 0:
                    inputs['right'] = True
                else:
                    inputs['left'] = True

        return inputs


@dataclass
class RecordedInput:
    """A single frame of recorded input."""
    frame: int
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    jump: bool = False
    attack: bool = False
    special: bool = False
    shield: bool = False
    grab: bool = False


@dataclass
class RecordingState:
    """State for the record/playback system."""
    is_recording: bool = False
    is_playing: bool = False
    recorded_inputs: List[RecordedInput] = field(default_factory=list)
    playback_index: int = 0
    loop_playback: bool = True


@dataclass
class FrameDataOverlay:
    """Displays frame data information for attacks."""
    enabled: bool = False
    show_startup: bool = True
    show_active: bool = True
    show_recovery: bool = True
    show_advantage: bool = True


@dataclass
class FrameAdvantageDisplay:
    """
    Cycle 13: Frame advantage display after hit confirmation.

    Frame advantage = (attacker's recovery) - (defender's hitstun)
    Positive = attacker can act first (advantage)
    Negative = defender can act first (disadvantage)
    """
    active: bool = False
    display_x: float = 0.0
    display_y: float = 0.0
    advantage_frames: int = 0
    display_timer: int = 0  # Frames to show display
    display_duration: int = 60  # 1 second at 60fps

    # Track attack recovery and defender hitstun for calculation
    pending_attacker_recovery: int = 0
    pending_defender_hitstun: int = 0
    last_hit_frame: int = -999

    def trigger(self, x: float, y: float, attacker_recovery: int, defender_hitstun: int, frame: int):
        """Trigger a new frame advantage display."""
        # Frame advantage: negative means attacker recovers faster (advantage)
        # Positive means defender recovers faster (disadvantage for attacker)
        # For the ATTACKER's perspective:
        #   advantage = defender_hitstun - attacker_recovery
        #   positive = you can act before them (good)
        #   negative = they act before you (bad)
        self.advantage_frames = defender_hitstun - attacker_recovery
        self.display_x = x
        self.display_y = y - 40  # Above the hit location
        self.display_timer = self.display_duration
        self.active = True
        self.last_hit_frame = frame

    def update(self):
        """Update display timer."""
        if self.active:
            self.display_timer -= 1
            if self.display_timer <= 0:
                self.active = False

    def get_color(self) -> Tuple[int, int, int]:
        """Get display color based on advantage."""
        if self.advantage_frames > 0:
            # Positive advantage (green) - attacker recovers first
            return (100, 255, 100)
        elif self.advantage_frames < 0:
            # Negative advantage (red) - defender recovers first
            return (255, 100, 100)
        else:
            # Neutral (yellow)
            return (255, 255, 100)

    def get_text(self) -> str:
        """Get display text."""
        if self.advantage_frames > 0:
            return f"+{self.advantage_frames}"
        elif self.advantage_frames < 0:
            return f"{self.advantage_frames}"
        else:
            return "0"


# =============================================================================
# CYCLE 16: INPUT HISTORY DISPLAY
# =============================================================================

@dataclass
class InputHistoryEntry:
    """
    A single entry in the input history timeline.
    Stores which inputs were active on a given frame.
    """
    frame: int
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    jump: bool = False
    attack: bool = False
    special: bool = False
    shield: bool = False
    grab: bool = False

    def is_empty(self) -> bool:
        """Check if no inputs are active."""
        return not any([
            self.left, self.right, self.up, self.down,
            self.jump, self.attack, self.special, self.shield, self.grab
        ])

    def get_active_inputs(self) -> List[str]:
        """Get list of active input names."""
        active = []
        if self.left:
            active.append('←')
        if self.right:
            active.append('→')
        if self.up:
            active.append('↑')
        if self.down:
            active.append('↓')
        if self.jump:
            active.append('J')  # Jump
        if self.attack:
            active.append('A')  # Attack
        if self.special:
            active.append('S')  # Special
        if self.shield:
            active.append('G')  # Guard/Shield
        if self.grab:
            active.append('Z')  # Grab (throw)
        return active


class InputHistoryDisplay:
    """
    Cycle 16: Visual input history display showing last 20 inputs.

    Displays a scrolling timeline of input icons near each player.
    Uses arrow symbols for directions and letter icons for buttons.
    """

    def __init__(self, max_entries: int = 20):
        self.max_entries = max_entries
        self.p1_history: List[InputHistoryEntry] = []
        self.p2_history: List[InputHistoryEntry] = []
        self.enabled = True
        # How many frames between recordings (skip duplicates for cleaner display)
        self.debounce_frames = 2
        self.last_p1_record_frame = -999
        self.last_p2_record_frame = -999

    def record_input(self, player_index: int, inputs: dict, frame: int):
        """
        Record current inputs for a player.
        Only records when inputs change to avoid filling history with duplicates.
        """
        if not self.enabled:
            return

        history = self.p1_history if player_index == 0 else self.p2_history
        last_record = self.last_p1_record_frame if player_index == 0 else self.last_p2_record_frame

        # Create entry from input dict
        entry = InputHistoryEntry(
            frame=frame,
            left=inputs.get('left', False),
            right=inputs.get('right', False),
            up=inputs.get('up', False),
            down=inputs.get('down', False),
            jump=inputs.get('jump', False),
            attack=inputs.get('attack', False),
            special=inputs.get('special', False),
            shield=inputs.get('shield', False),
            grab=inputs.get('grab', False)
        )

        # Skip empty inputs
        if entry.is_empty():
            return

        # Check if this is different from the last entry
        should_record = True
        if history:
            last = history[-1]
            # Compare input states
            same_inputs = (
                last.left == entry.left and
                last.right == entry.right and
                last.up == entry.up and
                last.down == entry.down and
                last.jump == entry.jump and
                last.attack == entry.attack and
                last.special == entry.special and
                last.shield == entry.shield and
                last.grab == entry.grab
            )
            # Only record if inputs changed or debounce time passed
            if same_inputs and (frame - last_record) < self.debounce_frames:
                should_record = False

        if should_record:
            history.append(entry)
            if player_index == 0:
                self.last_p1_record_frame = frame
            else:
                self.last_p2_record_frame = frame

            # Trim to max entries
            if len(history) > self.max_entries:
                if player_index == 0:
                    self.p1_history = history[-self.max_entries:]
                else:
                    self.p2_history = history[-self.max_entries:]

    def get_p1_history(self) -> List[InputHistoryEntry]:
        """Get P1's input history."""
        return self.p1_history

    def get_p2_history(self) -> List[InputHistoryEntry]:
        """Get P2's input history."""
        return self.p2_history

    def clear(self):
        """Clear all history."""
        self.p1_history.clear()
        self.p2_history.clear()

    def toggle(self) -> bool:
        """Toggle input history display."""
        self.enabled = not self.enabled
        return self.enabled


# =============================================================================
# CYCLE 16: HITBOX VISUALIZATION MODE
# =============================================================================

class HitboxVisualizationMode(Enum):
    """Different modes for hitbox/hurtbox visualization."""
    NONE = auto()           # No boxes shown
    HITBOX_ONLY = auto()    # Only attack hitboxes (red)
    HURTBOX_ONLY = auto()   # Only hurtboxes (green)
    BOTH = auto()           # Both hitbox and hurtbox
    COLLISION = auto()      # Both + collision point markers


class HitboxVisualizationSettings:
    """
    Cycle 16: Enhanced hitbox visualization settings.

    Features:
    - Cycle through visibility modes
    - Show collision intersection points
    - Distinct colors for attack/hurt/collision
    """

    def __init__(self):
        self.mode = HitboxVisualizationMode.NONE
        # Colors (RGB)
        self.hitbox_color = (255, 50, 50)       # Red for attack hitboxes
        self.hurtbox_color = (50, 255, 50)      # Green for hurtboxes
        self.collision_color = (255, 255, 50)   # Yellow for collision points
        # Store recent collision points for visualization
        self.collision_points: List[Tuple[float, float, int]] = []  # (x, y, frame)
        self.collision_point_lifetime = 30  # Frames to show collision point

    def cycle_mode(self) -> HitboxVisualizationMode:
        """Cycle through visualization modes."""
        modes = list(HitboxVisualizationMode)
        current_idx = modes.index(self.mode)
        next_idx = (current_idx + 1) % len(modes)
        self.mode = modes[next_idx]
        return self.mode

    def add_collision_point(self, x: float, y: float, frame: int):
        """Add a collision point to visualize."""
        self.collision_points.append((x, y, frame))
        # Limit stored points
        if len(self.collision_points) > 20:
            self.collision_points = self.collision_points[-20:]

    def get_active_collision_points(self, current_frame: int) -> List[Tuple[float, float]]:
        """Get collision points that are still within lifetime."""
        active = []
        for x, y, frame in self.collision_points:
            if current_frame - frame < self.collision_point_lifetime:
                active.append((x, y))
        return active

    def update(self, current_frame: int):
        """Remove expired collision points."""
        self.collision_points = [
            (x, y, frame) for x, y, frame in self.collision_points
            if current_frame - frame < self.collision_point_lifetime
        ]

    @property
    def show_hitboxes(self) -> bool:
        """Should hitboxes be shown?"""
        return self.mode in (
            HitboxVisualizationMode.HITBOX_ONLY,
            HitboxVisualizationMode.BOTH,
            HitboxVisualizationMode.COLLISION
        )

    @property
    def show_hurtboxes(self) -> bool:
        """Should hurtboxes be shown?"""
        return self.mode in (
            HitboxVisualizationMode.HURTBOX_ONLY,
            HitboxVisualizationMode.BOTH,
            HitboxVisualizationMode.COLLISION
        )

    @property
    def show_collision_points(self) -> bool:
        """Should collision points be shown?"""
        return self.mode == HitboxVisualizationMode.COLLISION


# =============================================================================
# CYCLE 16: MOVE LIST DISPLAY
# =============================================================================

@dataclass
class MoveInfo:
    """Information about a single move for the move list."""
    name: str
    input_notation: str      # e.g., "← + A", "↓ + B"
    startup_frames: int
    active_frames: str       # e.g., "5-8"
    recovery_frames: int
    total_frames: int
    damage: float
    notes: str = ""          # e.g., "Meteor", "Counter", "Projectile"


class MoveListDisplay:
    """
    Cycle 16: Display available moves for the selected character.

    Shows input notation, frame data, and special properties.
    Can be scrolled if move list is long.
    """

    def __init__(self):
        self.enabled = False
        self.scroll_offset = 0
        self.visible_moves = 8  # How many moves to show at once
        self.current_character_type = None
        self._cached_moves: List[MoveInfo] = []

    def toggle(self) -> bool:
        """Toggle move list display."""
        self.enabled = not self.enabled
        return self.enabled

    def set_character(self, character_type, attacks_dict: Dict):
        """
        Build move list from character's attack data.

        Args:
            character_type: CharacterType enum
            attacks_dict: Dict[AttackType, AttackData] for the character
        """
        self.current_character_type = character_type
        self._cached_moves.clear()
        self.scroll_offset = 0

        # Define input notations for each attack type
        input_notations = {
            AttackType.JAB: "A",
            AttackType.FORWARD_TILT: "→ + A",
            AttackType.UP_TILT: "↑ + A",
            AttackType.DOWN_TILT: "↓ + A",
            AttackType.FORWARD_SMASH: "→→ + A",
            AttackType.UP_SMASH: "↑↑ + A",
            AttackType.DOWN_SMASH: "↓↓ + A",
            AttackType.NEUTRAL_AIR: "(air) A",
            AttackType.FORWARD_AIR: "(air) → + A",
            AttackType.BACK_AIR: "(air) ← + A",
            AttackType.UP_AIR: "(air) ↑ + A",
            AttackType.DOWN_AIR: "(air) ↓ + A",
            AttackType.NEUTRAL_SPECIAL: "B",
            AttackType.SIDE_SPECIAL: "→ + B",
            AttackType.UP_SPECIAL: "↑ + B",
            AttackType.DOWN_SPECIAL: "↓ + B",
        }

        # Build move list
        for attack_type, attack_data in attacks_dict.items():
            notation = input_notations.get(attack_type, attack_type.name)

            # Calculate damage (sum of all hitboxes)
            total_damage = sum(hb.damage for hb in attack_data.hitboxes) if attack_data.hitboxes else 0

            # Get active frames string
            active_str = ""
            if attack_data.hitboxes:
                start_frame = min(hb.active_frames[0] for hb in attack_data.hitboxes)
                end_frame = max(hb.active_frames[1] for hb in attack_data.hitboxes)
                active_str = f"{start_frame}-{end_frame}"

            # Determine notes
            notes = []
            if hasattr(attack_data, 'is_counter') and attack_data.is_counter:
                notes.append("Counter")
            if hasattr(attack_data, 'is_reflector') and attack_data.is_reflector:
                notes.append("Reflect")
            if hasattr(attack_data, 'super_armor') and attack_data.super_armor:
                notes.append("Armor")
            if hasattr(attack_data, 'spawns_projectile') and attack_data.spawns_projectile:
                notes.append("Proj")
            # Check for meteor/spike
            for hb in attack_data.hitboxes:
                if 250 <= hb.angle <= 290:
                    notes.append("Spike")
                    break

            move_info = MoveInfo(
                name=attack_type.name.replace('_', ' ').title(),
                input_notation=notation,
                startup_frames=attack_data.startup_frames,
                active_frames=active_str,
                recovery_frames=attack_data.recovery_frames,
                total_frames=attack_data.total_frames,
                damage=total_damage,
                notes=", ".join(notes) if notes else ""
            )
            self._cached_moves.append(move_info)

        # Sort moves by type (normals, aerials, specials)
        type_order = [
            AttackType.JAB, AttackType.FORWARD_TILT, AttackType.UP_TILT, AttackType.DOWN_TILT,
            AttackType.FORWARD_SMASH, AttackType.UP_SMASH, AttackType.DOWN_SMASH,
            AttackType.NEUTRAL_AIR, AttackType.FORWARD_AIR, AttackType.BACK_AIR,
            AttackType.UP_AIR, AttackType.DOWN_AIR,
            AttackType.NEUTRAL_SPECIAL, AttackType.SIDE_SPECIAL, AttackType.UP_SPECIAL, AttackType.DOWN_SPECIAL,
        ]
        type_order_map = {t: i for i, t in enumerate(type_order)}

        # Re-sort based on attack type order
        self._cached_moves.sort(
            key=lambda m: type_order_map.get(
                AttackType[m.name.upper().replace(' ', '_')], 999
            )
        )

    def get_visible_moves(self) -> List[MoveInfo]:
        """Get moves currently visible based on scroll offset."""
        start = self.scroll_offset
        end = start + self.visible_moves
        return self._cached_moves[start:end]

    def get_all_moves(self) -> List[MoveInfo]:
        """Get all moves."""
        return self._cached_moves

    def scroll_up(self):
        """Scroll move list up."""
        if self.scroll_offset > 0:
            self.scroll_offset -= 1

    def scroll_down(self):
        """Scroll move list down."""
        max_offset = max(0, len(self._cached_moves) - self.visible_moves)
        if self.scroll_offset < max_offset:
            self.scroll_offset += 1

    def can_scroll_up(self) -> bool:
        """Check if can scroll up."""
        return self.scroll_offset > 0

    def can_scroll_down(self) -> bool:
        """Check if can scroll down."""
        max_offset = max(0, len(self._cached_moves) - self.visible_moves)
        return self.scroll_offset < max_offset


# =============================================================================
# CYCLE 18: TRAINING MODE COMBO COUNTER
# =============================================================================

@dataclass
class ComboCounter:
    """
    Cycle 18: Enhanced combo counter for training mode.

    Tracks:
    - Current combo count (hits in sequence)
    - Total combo damage accumulator
    - Best combo achieved in session
    - Combo drop indicator
    """
    # Per-player combo tracking
    combo_counts: List[int] = field(default_factory=lambda: [0, 0])
    combo_damage: List[float] = field(default_factory=lambda: [0.0, 0.0])
    combo_timers: List[int] = field(default_factory=lambda: [0, 0])

    # Best combos in session
    best_combo_count: List[int] = field(default_factory=lambda: [0, 0])
    best_combo_damage: List[float] = field(default_factory=lambda: [0.0, 0.0])

    # Combo drop visual feedback
    combo_drop_active: List[bool] = field(default_factory=lambda: [False, False])
    combo_drop_timer: List[int] = field(default_factory=lambda: [0, 0])
    combo_drop_count: List[int] = field(default_factory=lambda: [0, 0])  # Last combo when dropped
    combo_drop_damage: List[float] = field(default_factory=lambda: [0.0, 0.0])  # Last damage when dropped

    # Display position (for centering above fighters)
    display_positions: List[Tuple[float, float]] = field(default_factory=lambda: [(0, 0), (0, 0)])

    # Combo window (frames before combo drops)
    COMBO_WINDOW: int = 45  # ~0.75 seconds at 60 FPS
    DROP_DISPLAY_DURATION: int = 60  # 1 second to show "DROPPED"

    def register_hit(self, attacker_id: int, damage: float, position: Tuple[float, float]):
        """
        Register a hit for combo tracking.

        Args:
            attacker_id: 0 for P1, 1 for P2
            damage: Damage dealt by this hit
            position: (x, y) position for display
        """
        # Increment combo
        self.combo_counts[attacker_id] += 1
        self.combo_damage[attacker_id] += damage
        self.combo_timers[attacker_id] = 0  # Reset timer

        # Update display position
        self.display_positions[attacker_id] = position

        # Track best combo
        if self.combo_counts[attacker_id] > self.best_combo_count[attacker_id]:
            self.best_combo_count[attacker_id] = self.combo_counts[attacker_id]
        if self.combo_damage[attacker_id] > self.best_combo_damage[attacker_id]:
            self.best_combo_damage[attacker_id] = self.combo_damage[attacker_id]

        # Clear any combo drop indicator for this player
        self.combo_drop_active[attacker_id] = False

        # Reset the opponent's combo (they got hit)
        opponent_id = 1 - attacker_id
        if self.combo_counts[opponent_id] > 1:
            # They had a combo going - trigger drop indicator
            self.combo_drop_active[opponent_id] = True
            self.combo_drop_timer[opponent_id] = self.DROP_DISPLAY_DURATION
            self.combo_drop_count[opponent_id] = self.combo_counts[opponent_id]
            self.combo_drop_damage[opponent_id] = self.combo_damage[opponent_id]

        self.combo_counts[opponent_id] = 0
        self.combo_damage[opponent_id] = 0.0

    def update(self):
        """Update combo timers and check for drops."""
        for i in range(2):
            # Update active combo timers
            if self.combo_counts[i] > 0:
                self.combo_timers[i] += 1
                # Check for combo drop
                if self.combo_timers[i] >= self.COMBO_WINDOW:
                    if self.combo_counts[i] > 1:
                        # Combo dropped - show indicator
                        self.combo_drop_active[i] = True
                        self.combo_drop_timer[i] = self.DROP_DISPLAY_DURATION
                        self.combo_drop_count[i] = self.combo_counts[i]
                        self.combo_drop_damage[i] = self.combo_damage[i]
                    # Reset combo
                    self.combo_counts[i] = 0
                    self.combo_damage[i] = 0.0

            # Update drop display timer
            if self.combo_drop_active[i]:
                self.combo_drop_timer[i] -= 1
                if self.combo_drop_timer[i] <= 0:
                    self.combo_drop_active[i] = False

    def reset_session(self):
        """Reset all combo data including best combos."""
        self.combo_counts = [0, 0]
        self.combo_damage = [0.0, 0.0]
        self.combo_timers = [0, 0]
        self.best_combo_count = [0, 0]
        self.best_combo_damage = [0.0, 0.0]
        self.combo_drop_active = [False, False]
        self.combo_drop_timer = [0, 0]

    def get_combo_display(self, player_id: int) -> Optional[dict]:
        """
        Get combo display data for a player.

        Returns:
            dict with: count, damage, is_active, position, alpha
            or None if no combo to display
        """
        if self.combo_counts[player_id] <= 1 and not self.combo_drop_active[player_id]:
            return None

        # Active combo
        if self.combo_counts[player_id] > 1:
            # Calculate alpha based on timer (fade as combo is about to drop)
            timer_ratio = self.combo_timers[player_id] / self.COMBO_WINDOW
            alpha = max(0.3, 1.0 - (timer_ratio * 0.7))

            return {
                'count': self.combo_counts[player_id],
                'damage': self.combo_damage[player_id],
                'is_active': True,
                'is_dropping': timer_ratio > 0.6,  # Warn when close to dropping
                'position': self.display_positions[player_id],
                'alpha': alpha
            }

        # Dropped combo (showing indicator)
        if self.combo_drop_active[player_id]:
            alpha = self.combo_drop_timer[player_id] / self.DROP_DISPLAY_DURATION
            return {
                'count': self.combo_drop_count[player_id],
                'damage': self.combo_drop_damage[player_id],
                'is_active': False,
                'is_dropping': False,
                'position': self.display_positions[player_id],
                'alpha': alpha
            }

        return None

    def get_best_combo(self, player_id: int) -> Tuple[int, float]:
        """Get best combo count and damage for a player."""
        return (self.best_combo_count[player_id], self.best_combo_damage[player_id])


class TrainingModeManager:
    """Manages training mode features."""

    def __init__(self):
        self.cpu_controller = CPUController()
        self.frame_data_overlay = FrameDataOverlay()
        self.recording_state = RecordingState()

        # Cycle 13: Frame advantage display
        self.frame_advantage = FrameAdvantageDisplay()
        self.show_frame_advantage = True

        # Cycle 16: Input history display
        self.input_history = InputHistoryDisplay()
        self.show_input_history = True

        # Cycle 16: Enhanced hitbox visualization
        self.hitbox_viz = HitboxVisualizationSettings()

        # Cycle 16: Move list display
        self.move_list = MoveListDisplay()

        # Cycle 18: Enhanced combo counter
        self.combo_counter = ComboCounter()
        self.show_combo_counter = True

        # Training mode toggles
        self.infinite_shields = False
        self.infinite_jumps = False  # Cycle 6: Infinite jumps for recovery practice
        self.auto_reset_damage = False
        self.damage_reset_threshold = 150.0
        self.show_inputs = False

        # Reset positions (stored when training mode starts or T pressed)
        self.p1_start_pos: Optional[Vector2] = None
        self.p2_start_pos: Optional[Vector2] = None

    def toggle_cpu(self, behavior: CPUBehavior = None):
        """Toggle CPU on/off."""
        if self.cpu_controller.state.enabled:
            self.cpu_controller.disable()
            return False
        else:
            if behavior:
                self.cpu_controller.enable(behavior)
            else:
                self.cpu_controller.enable(CPUBehavior.STAND)
            return True

    def toggle_frame_data(self):
        """Toggle frame data overlay."""
        self.frame_data_overlay.enabled = not self.frame_data_overlay.enabled
        return self.frame_data_overlay.enabled

    def toggle_frame_advantage(self):
        """Toggle frame advantage display."""
        self.show_frame_advantage = not self.show_frame_advantage
        return self.show_frame_advantage

    def toggle_input_history(self):
        """Cycle 16: Toggle input history display."""
        self.show_input_history = not self.show_input_history
        self.input_history.enabled = self.show_input_history
        return self.show_input_history

    def cycle_hitbox_mode(self):
        """Cycle 16: Cycle through hitbox visualization modes."""
        return self.hitbox_viz.cycle_mode()

    def toggle_move_list(self):
        """Cycle 16: Toggle move list display."""
        return self.move_list.toggle()

    def toggle_combo_counter(self):
        """Cycle 18: Toggle combo counter display."""
        self.show_combo_counter = not self.show_combo_counter
        return self.show_combo_counter

    def record_combo_hit(self, attacker_id: int, damage: float, position: Tuple[float, float]):
        """
        Cycle 18: Record a hit for combo tracking.

        Args:
            attacker_id: 0 for P1, 1 for P2
            damage: Damage dealt by this hit
            position: (x, y) position for display
        """
        if self.show_combo_counter:
            self.combo_counter.register_hit(attacker_id, damage, position)

    def record_input_for_history(self, player_index: int, inputs: dict, frame: int):
        """Cycle 16: Record input for history display."""
        self.input_history.record_input(player_index, inputs, frame)

    def record_collision(self, x: float, y: float, frame: int):
        """Cycle 16: Record a collision point for visualization."""
        self.hitbox_viz.add_collision_point(x, y, frame)

    def record_hit(self, attacker: Fighter, defender: Fighter, hitbox,
                   contact_x: float, contact_y: float, game_frame: int):
        """
        Record a hit for frame advantage calculation.

        Frame advantage = defender_hitstun - attacker_remaining_recovery
        Positive means attacker can act first (advantage)
        Negative means defender can act first (disadvantage)
        """
        if not self.show_frame_advantage:
            return

        # Get attacker's remaining recovery frames
        attacker_recovery = 0
        if attacker.current_attack:
            total_frames = attacker.current_attack.total_frames
            current_frame = attacker.attack_frame
            attacker_recovery = max(0, total_frames - current_frame)

        # Defender's hitstun was just applied, get it from the fighter
        defender_hitstun = defender.hitstun_frames

        self.frame_advantage.trigger(
            x=contact_x,
            y=contact_y,
            attacker_recovery=attacker_recovery,
            defender_hitstun=defender_hitstun,
            frame=game_frame
        )

    def toggle_infinite_jumps(self):
        """Toggle infinite jumps for recovery practice."""
        self.infinite_jumps = not self.infinite_jumps
        return self.infinite_jumps

    def cycle_cpu_behavior(self):
        """Cycle through CPU behaviors."""
        behaviors = list(CPUBehavior)
        current_idx = behaviors.index(self.cpu_controller.state.behavior)
        next_idx = (current_idx + 1) % len(behaviors)
        self.cpu_controller.set_behavior(behaviors[next_idx])
        return behaviors[next_idx]

    def cycle_cpu_difficulty(self):
        """Cycle through CPU difficulty levels."""
        return self.cpu_controller.cycle_difficulty()

    def get_cpu_inputs(self, cpu_fighter: Fighter, opponent: Fighter, frame: int) -> dict:
        """Get CPU input commands for this frame."""
        return self.cpu_controller.update(cpu_fighter, opponent, frame)

    def reset_damage(self, game_state: GameState):
        """Reset both fighters to 0% damage."""
        for fighter in game_state.fighters:
            fighter.damage_percent = 0.0

    def reset_positions(self, game_state: GameState):
        """Reset fighters to their starting positions."""
        if self.p1_start_pos and self.p2_start_pos:
            game_state.fighters[0].position = self.p1_start_pos.copy()
            game_state.fighters[1].position = self.p2_start_pos.copy()
        else:
            # Use spawn points
            game_state.fighters[0].position = game_state.stage.spawn_points[0].copy()
            game_state.fighters[1].position = game_state.stage.spawn_points[1].copy()

        # Reset velocities and states
        for fighter in game_state.fighters:
            fighter.velocity = Vector2(0, 0)
            fighter.state = FighterState.IDLE
            fighter.hitstun_frames = 0
            fighter.grounded = True
            fighter.can_double_jump = True
            fighter.current_attack = None
            fighter.active_hitboxes = []

    def store_start_positions(self, game_state: GameState):
        """Store current positions as reset targets."""
        self.p1_start_pos = game_state.fighters[0].position.copy()
        self.p2_start_pos = game_state.fighters[1].position.copy()

    # =========================================================================
    # Recording System
    # =========================================================================

    def start_recording(self):
        """Start recording P2 inputs."""
        self.recording_state.is_recording = True
        self.recording_state.is_playing = False
        self.recording_state.recorded_inputs = []
        self.recording_state.playback_index = 0

    def stop_recording(self):
        """Stop recording."""
        self.recording_state.is_recording = False

    def start_playback(self, loop: bool = True):
        """Start playing back recorded inputs."""
        if not self.recording_state.recorded_inputs:
            return False

        self.recording_state.is_recording = False
        self.recording_state.is_playing = True
        self.recording_state.playback_index = 0
        self.recording_state.loop_playback = loop
        return True

    def stop_playback(self):
        """Stop playback."""
        self.recording_state.is_playing = False

    def record_input(self, frame: int, inputs: dict):
        """Record a frame of input during recording mode."""
        if not self.recording_state.is_recording:
            return

        recorded = RecordedInput(
            frame=len(self.recording_state.recorded_inputs),
            left=inputs.get('left', False),
            right=inputs.get('right', False),
            up=inputs.get('up', False),
            down=inputs.get('down', False),
            jump=inputs.get('jump', False),
            attack=inputs.get('attack', False),
            special=inputs.get('special', False),
            shield=inputs.get('shield', False),
            grab=inputs.get('grab', False)
        )
        self.recording_state.recorded_inputs.append(recorded)

    def get_playback_input(self) -> Optional[dict]:
        """Get the next input from playback."""
        if not self.recording_state.is_playing:
            return None

        if not self.recording_state.recorded_inputs:
            self.recording_state.is_playing = False
            return None

        idx = self.recording_state.playback_index
        if idx >= len(self.recording_state.recorded_inputs):
            if self.recording_state.loop_playback:
                idx = 0
                self.recording_state.playback_index = 0
            else:
                self.recording_state.is_playing = False
                return None

        recorded = self.recording_state.recorded_inputs[idx]
        self.recording_state.playback_index += 1

        return {
            'left': recorded.left,
            'right': recorded.right,
            'up': recorded.up,
            'down': recorded.down,
            'jump': recorded.jump,
            'attack': recorded.attack,
            'special': recorded.special,
            'shield': recorded.shield,
            'grab': recorded.grab
        }

    def update(self, game_state: GameState):
        """Update training mode features."""
        # Infinite shields
        if self.infinite_shields:
            for fighter in game_state.fighters:
                fighter.shield_health = 100.0
                fighter.shield_broken = False

        # Infinite jumps (Cycle 6)
        if self.infinite_jumps:
            for fighter in game_state.fighters:
                fighter.can_double_jump = True

        # Auto reset damage
        if self.auto_reset_damage:
            for fighter in game_state.fighters:
                if fighter.damage_percent > self.damage_reset_threshold:
                    fighter.damage_percent = 0.0

        # Cycle 13: Update frame advantage display
        self.frame_advantage.update()

        # Cycle 16: Update hitbox visualization (clear expired collision points)
        self.hitbox_viz.update(game_state.frame)

        # Cycle 18: Update combo counter timers
        if self.show_combo_counter:
            self.combo_counter.update()

    def get_frame_data_text(self, fighter: Fighter) -> List[str]:
        """Get frame data text for current attack."""
        if not self.frame_data_overlay.enabled:
            return []

        lines = []

        if fighter.current_attack and fighter.state in (FighterState.ATTACKING, FighterState.SPECIAL):
            attack = fighter.current_attack
            lines.append(f"Attack: {attack.attack_type.name}")

            if self.frame_data_overlay.show_startup:
                lines.append(f"Startup: {attack.startup_frames}f")

            if self.frame_data_overlay.show_active:
                for i, hb in enumerate(attack.hitboxes):
                    start, end = hb.active_frames
                    lines.append(f"Active: {start}-{end}f")

            if self.frame_data_overlay.show_recovery:
                lines.append(f"Recovery: {attack.recovery_frames}f")
                lines.append(f"Total: {attack.total_frames}f")

            # Current frame in attack
            lines.append(f"Frame: {fighter.attack_frame}/{attack.total_frames}")

        elif fighter.state == FighterState.HITSTUN:
            lines.append(f"Hitstun: {fighter.hitstun_frames}f left")

        elif fighter.state == FighterState.SHIELDING:
            lines.append(f"Shield: {fighter.shield_health:.0f}%")

        elif fighter.state == FighterState.COUNTER:
            lines.append(f"Counter Window")
            lines.append(f"Frame: {fighter.state_frame}")

        elif fighter.state == FighterState.REFLECTING:
            lines.append(f"Reflector Active")
            lines.append(f"Frame: {fighter.state_frame}")

        return lines

    def get_status_text(self) -> List[str]:
        """Get status text for training mode HUD."""
        lines = []

        # CPU status
        if self.cpu_controller.state.enabled:
            behavior = self.cpu_controller.state.behavior.name
            difficulty = self.cpu_controller.state.difficulty.name
            lines.append(f"CPU: {behavior} Lv{self.cpu_controller.state.difficulty.value}")
        else:
            lines.append("CPU: OFF")

        # Toggles
        toggles = []
        if self.infinite_jumps:
            toggles.append("InfJumps")
        if self.infinite_shields:
            toggles.append("InfShield")
        if self.show_frame_advantage:
            toggles.append("FrameAdv")
        if toggles:
            lines.append(" ".join(toggles))

        # Recording status
        if self.recording_state.is_recording:
            frames = len(self.recording_state.recorded_inputs)
            lines.append(f"REC: {frames}f")
        elif self.recording_state.is_playing:
            idx = self.recording_state.playback_index
            total = len(self.recording_state.recorded_inputs)
            lines.append(f"PLAY: {idx}/{total}")

        return lines

    def get_frame_advantage_display(self) -> Optional[Tuple[float, float, str, Tuple[int, int, int]]]:
        """
        Get frame advantage display data if active.

        Returns:
            Tuple of (x, y, text, color) or None if not active
        """
        if not self.show_frame_advantage or not self.frame_advantage.active:
            return None

        return (
            self.frame_advantage.display_x,
            self.frame_advantage.display_y,
            self.frame_advantage.get_text(),
            self.frame_advantage.get_color()
        )


# Singleton instance
training_manager = TrainingModeManager()


def get_training_manager() -> TrainingModeManager:
    """Get the global training mode manager."""
    return training_manager
