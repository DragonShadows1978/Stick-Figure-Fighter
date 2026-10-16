#!/usr/bin/env python3
"""
Generate comprehensive character matchup matrix for Stick Figure Fighter.

Uses existing character-specific policies to create a full win-rate matrix.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

from policy import PolicyNetwork
from game_engine import (
    create_game, GameLoop, get_observation, get_action_from_index,
    OBSERVATION_SIZE, ACTION_SPACE_SIZE
)

CHARACTERS = ['FLEX', 'RUSH', 'ZONE', 'GRAB']


def load_character_policies(checkpoint_dir: str = 'checkpoints/char_variety') -> dict:
    """Load all character-specific policies."""
    policies = {}
    for char in CHARACTERS:
        path = f'{checkpoint_dir}/{char.lower()}_policy.npz'
        if os.path.exists(path):
            policy = PolicyNetwork(OBSERVATION_SIZE, ACTION_SPACE_SIZE, 256)
            policy.load(path)
            policy.to('cuda')
            policies[char] = policy
            print(f"Loaded {char} policy")
        else:
            print(f"Warning: No policy found for {char}")
    return policies


def run_matchup(policy1, char1: str, policy2, char2: str,
                num_games: int = 30) -> dict:
    """Run a matchup between two policies/characters."""
    p1_wins = 0
    total_damage_dealt = [0.0, 0.0]
    total_stocks_taken = [0, 0]

    for _ in range(num_games):
        game = create_game(
            stage_name='battlefield',
            p1_character=char1,
            p2_character=char2,
            stock_count=3
        )
        loop = GameLoop(game, headless=True)

        frame = 0
        prev_stocks = [3, 3]

        while frame < 10800 and not game.match_over:
            obs1 = get_observation(game, 0)
            obs2 = get_observation(game, 1)

            a1, _, _ = policy1.get_action(obs1)
            a2, _, _ = policy2.get_action(obs2)

            game.fighters[0].input = get_action_from_index(a1)
            game.fighters[1].input = get_action_from_index(a2)

            loop.step()
            frame += 1

            # Track stocks taken
            for i in range(2):
                if game.fighters[i].stocks < prev_stocks[i]:
                    total_stocks_taken[1-i] += 1
                prev_stocks[i] = game.fighters[i].stocks

        if game.winner == 0:
            p1_wins += 1

        total_damage_dealt[0] += game.fighters[1].damage_percent
        total_damage_dealt[1] += game.fighters[0].damage_percent

    return {
        'p1_wins': p1_wins,
        'p2_wins': num_games - p1_wins,
        'p1_win_rate': p1_wins / num_games,
        'avg_damage_dealt': [d / num_games for d in total_damage_dealt],
        'avg_stocks_taken': [s / num_games for s in total_stocks_taken]
    }


def generate_matchup_matrix(policies: dict, games_per_matchup: int = 30) -> dict:
    """Generate full matchup matrix."""
    print("\n" + "="*60)
    print("GENERATING MATCHUP MATRIX")
    print(f"Games per matchup: {games_per_matchup}")
    print("="*60)

    characters = list(policies.keys())
    matrix = {c1: {} for c1 in characters}

    for i, char1 in enumerate(characters):
        for j, char2 in enumerate(characters):
            print(f"\n  {char1} vs {char2}...")
            result = run_matchup(
                policies[char1], char1,
                policies[char2], char2,
                num_games=games_per_matchup
            )
            matrix[char1][char2] = result['p1_win_rate']
            print(f"    {char1} wins: {result['p1_win_rate']:.1%}")

    return matrix


def analyze_matchups(matrix: dict) -> dict:
    """Analyze matchup matrix for patterns."""
    characters = list(matrix.keys())
    analysis = {}

    for char in characters:
        wins_vs = {c: matrix[char][c] for c in characters}
        overall_wr = sum(wins_vs.values()) / len(wins_vs)

        # Find best/worst matchups
        best = max(wins_vs.items(), key=lambda x: x[1])
        worst = min(wins_vs.items(), key=lambda x: x[1])

        # Check for counter relationships (>60% advantage)
        counters = [c for c, wr in wins_vs.items() if wr >= 0.60 and c != char]
        countered_by = [c for c, wr in wins_vs.items() if wr <= 0.40 and c != char]

        analysis[char] = {
            'overall_win_rate': overall_wr,
            'best_matchup': {'opponent': best[0], 'win_rate': best[1]},
            'worst_matchup': {'opponent': worst[0], 'win_rate': worst[1]},
            'counters': counters,
            'countered_by': countered_by
        }

    return analysis


def print_matrix(matrix: dict):
    """Print formatted matchup matrix."""
    characters = list(matrix.keys())

    print("\n" + "="*60)
    print("MATCHUP MATRIX (Row vs Column win rate)")
    print("="*60)

    # Header
    header = "        "
    for char in characters:
        header += f" {char:>6}"
    print(header)
    print("-" * len(header))

    # Rows
    for char1 in characters:
        row = f"{char1:>6}  "
        for char2 in characters:
            wr = matrix[char1][char2]
            row += f" {wr:>5.1%}"
        print(row)


def main():
    print("="*60)
    print("STICK FIGURE FIGHTER - CHARACTER MATCHUP ANALYSIS")
    print("="*60)

    # Load policies
    policies = load_character_policies()

    if len(policies) < 2:
        print("Not enough policies found. Need at least 2 character policies.")
        return

    # Generate matrix
    matrix = generate_matchup_matrix(policies, games_per_matchup=30)

    # Print formatted matrix
    print_matrix(matrix)

    # Analyze matchups
    analysis = analyze_matchups(matrix)

    print("\n" + "="*60)
    print("CHARACTER ANALYSIS")
    print("="*60)

    for char, data in analysis.items():
        print(f"\n{char}:")
        print(f"  Overall win rate: {data['overall_win_rate']:.1%}")
        print(f"  Best matchup: vs {data['best_matchup']['opponent']} ({data['best_matchup']['win_rate']:.1%})")
        print(f"  Worst matchup: vs {data['worst_matchup']['opponent']} ({data['worst_matchup']['win_rate']:.1%})")
        if data['counters']:
            print(f"  Counters: {', '.join(data['counters'])}")
        if data['countered_by']:
            print(f"  Countered by: {', '.join(data['countered_by'])}")

    # Save report
    report = {
        'matchup_matrix': matrix,
        'analysis': analysis,
        'characters': list(policies.keys()),
        'games_per_matchup': 30
    }

    report_path = 'checkpoints/char_variety/matchup_analysis.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    return report


if __name__ == '__main__':
    main()
