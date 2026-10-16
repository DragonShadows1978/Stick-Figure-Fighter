"""Simulation configuration."""
from dataclasses import dataclass


@dataclass
class GameConfig:
    """Tunable simulation constants. Pass one to GameState to override defaults."""
    # Display (used for normalisation by game code; the sim itself is unitless)
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

    # Default blast zones (stages may define their own)
    blast_zone_left: float = -200
    blast_zone_right: float = 1480
    blast_zone_top: float = -200
    blast_zone_bottom: float = 920

    @property
    def frame_time(self) -> float:
        """Time per frame in seconds."""
        return 1.0 / self.target_fps


# Default config instance, used whenever a GameState isn't given its own.
CONFIG = GameConfig()

# fighter.position.y is the fighter's hip/center line. Fighters standing on a
# platform rest with position.y this far above the surface, which is where the
# stick-figure renderer draws the feet.
FOOT_OFFSET = 40.0
