"""Build Stick Figure Fighter matches from character and stage names."""
from dataclasses import replace
from typing import List, Optional

from characters import get_character_attacks, get_character_by_name, get_character_color, get_character_stats
from engine import FighterStats, GameState
from stages import get_stage


def create_default_stats() -> List[FighterStats]:
    """Create default (characterless) fighter stats for 2 players."""
    return [
        FighterStats(name="Player 1"),
        FighterStats(name="Player 2")
    ]


def character_fighter_stats(char_name: str) -> Optional[FighterStats]:
    """FighterStats with moveset attached for a character name, or None if unknown."""
    char_type = get_character_by_name(char_name)
    if char_type is None:
        return None
    return replace(get_character_stats(char_type),
                   character_type=char_type.name,
                   moveset=get_character_attacks(char_type))


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
    stage = get_stage(stage_name)

    stats = []
    colors = []
    for i, char_name in enumerate([p1_character, p2_character]):
        fighter_stats = character_fighter_stats(char_name) if char_name else None
        if fighter_stats is not None:
            stats.append(fighter_stats)
            colors.append(get_character_color(get_character_by_name(char_name)))
        else:
            stats.append(FighterStats(name=f"Player {i + 1}"))
            colors.append(None)

    game = GameState(stage, stats, colors, time_limit)
    for fighter in game.fighters:
        fighter.stocks = stock_count
    return game
