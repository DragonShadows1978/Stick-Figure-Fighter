"""
Compatibility shim.

The simulation now lives in the reusable `engine` package; Stick Figure
Fighter's content lives in `characters`, `stages`, `match_setup` and
`ml_interface`. This module re-exports the old names so existing imports keep
working. New code should import from those modules directly.
"""
from engine import *  # noqa: F401,F403
from engine import __all__ as _engine_all
from match_setup import create_default_stats, create_game  # noqa: F401
from ml_interface import (ACTION_SPACE_SIZE, OBSERVATION_SIZE,  # noqa: F401
                          get_action_from_index, get_observation)

__all__ = list(_engine_all) + [
    "create_default_stats", "create_game", "ACTION_SPACE_SIZE",
    "OBSERVATION_SIZE", "get_action_from_index", "get_observation",
]
