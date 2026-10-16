"""Match state."""
from typing import List, Optional, Tuple

from .combat import Projectile
from .config import CONFIG, GameConfig
from .fighter import Fighter, FighterStats
from .geometry import Vector2
from .stage import Stage


class GameState:
    """Complete game state for a match."""

    def __init__(self, stage: Stage, fighter_stats: List[FighterStats],
                 fighter_colors: Optional[List[Optional[Tuple[int, int, int]]]] = None,
                 time_limit: int = 0,  # frames, 0 = no limit
                 config: Optional[GameConfig] = None):
        self.config = config or CONFIG
        self.stage = stage
        self.fighters: List[Fighter] = []
        self.projectiles: List[Projectile] = []  # Active projectiles
        self.frame = 0
        self.match_over = False
        self.winner: Optional[int] = None

        # Timer
        self.time_limit = time_limit  # 0 = no limit, otherwise frames until time-up
        self.time_remaining = time_limit  # Counts down from time_limit

        # Combo tracking (per player)
        self.combo_counts = [0, 0]  # Current combo count per player
        self.combo_timers = [0, 0]  # Frames since last hit (combo drops after ~30 frames)
        self.max_combos = [0, 0]  # Maximum combo achieved this match

        # Pause state
        self.paused = False

        # Hit freeze (brief pause on significant hits)
        self.hit_freeze_frames = 0  # Frames remaining for hit freeze effect

        # Damage counter animation state (per player)
        self.damage_anim_frames = [0, 0]  # Frames remaining for damage animation
        self.damage_anim_amount = [0.0, 0.0]  # Amount of damage for animation intensity
        self.last_damage = [0.0, 0.0]  # Track previous damage to detect hits

        # Create fighters
        for i, stats in enumerate(fighter_stats):
            spawn_pos = stage.spawn_points[i % len(stage.spawn_points)]
            color = None
            if fighter_colors and i < len(fighter_colors):
                color = fighter_colors[i]
            fighter = Fighter(player_id=i, stats=stats, spawn_pos=spawn_pos, color=color,
                              config=self.config)
            self.fighters.append(fighter)

    def add_projectile(self, projectile: Projectile):
        """Add a projectile to the game."""
        self.projectiles.append(projectile)

    def remove_inactive_projectiles(self):
        """Remove all inactive projectiles."""
        self.projectiles = [p for p in self.projectiles if p.active]

    # =========================================================================
    # Combo and Timer Methods
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
        """Update damage percentage animations."""
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
        Apply hit freeze effect on significant hits.

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

