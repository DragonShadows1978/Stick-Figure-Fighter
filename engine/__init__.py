"""
Platform-fighter simulation engine.

Deterministic, fixed-timestep, headless-capable, and free of game content:
characters, stages, rendering, audio and input devices all live outside this
package and plug in through data (FighterStats, AttackData, Stage) and the
GameLoop callbacks.

Pure Python, standard library only.
"""
from .combat import (AttackData, AttackType, Hitbox, Hurtbox, Projectile,
                     ProjectileSpec, default_attack)
from .config import CONFIG, FOOT_OFFSET, GameConfig
from .fighter import DodgeType, Fighter, FighterState, FighterStats
from .geometry import AABB, Vector2
from .input import InputAction, InputState
from .loop import GameLoop, run_headless_match
from .stage import ParallaxLayer, Platform, Stage
from .state import GameState

__all__ = [
    "AABB", "AttackData", "AttackType", "CONFIG", "DodgeType", "FOOT_OFFSET",
    "Fighter", "FighterState", "FighterStats", "GameConfig", "GameLoop",
    "GameState", "Hitbox", "Hurtbox", "InputAction", "InputState",
    "ParallaxLayer", "Platform", "Projectile", "ProjectileSpec", "Stage",
    "Vector2", "default_attack", "run_headless_match",
]
