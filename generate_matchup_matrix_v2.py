#!/usr/bin/env python3
"""
Generate character matchup matrix using the best trained policy.

Uses the same trained policy to play as different characters,
revealing the inherent character balance of the game engine.
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
from evaluate import RandomPolicy, AggressivePolicy
from game_engine import (
    create_game, GameLoop, get_observation, get_action_from_index,
    OBSERVATION_SIZE, ACTION_SPACE_SIZE
)

CHARACTERS = ['FLEX', 'RUSH', 'ZONE', 'GRAB']


def run_matchup(policy, char1: str, char2: str, num_games: int = 30) -> dict:
    """Run a matchup between two characters using the same policy."""
    p1_wins = 0
    total_damage_dealt = [0.0, 0.0]
    total_stocks_taken = [0, 0]
    total_frames = 0

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

            # Same policy plays both sides
            a1, _, _ = policy.get_action(obs1)
            a2, _, _ = policy.get_action(obs2)

            game.fighters[0].input = get_action_from_index(a1)
            game.fighters[1].input = get_action_from_index(a2)

            loop.step()
            frame += 1
            total_frames += 1

            for i in range(2):
                if game.fighters[i].stocks < prev_stocks[i]:
                    total_stocks_taken[1-i] += 1
                prev_stocks[i] = game.fighters[i].stocks

        if game.winner == 0:
            p1_wins += 1

    return {
        'p1_wins': p1_wins,
        'p1_win_rate': p1_wins / num_games,
        'avg_stocks_taken': [s / num_games for s in total_stocks_taken],
        'avg_frames': total_frames / num_games
    }


def run_vs_baselines(policy, character: str, num_games: int = 40) -> dict:
    """Evaluate character vs RandomPolicy and AggressivePolicy."""
    random_opp = RandomPolicy()
    aggressive_opp = AggressivePolicy()

    results = {'vs_random': 0, 'vs_aggressive': 0}

    # vs Random
    wins = 0
    for _ in range(num_games):
        game = create_game('battlefield', character, 'FLEX', stock_count=3)
        loop = GameLoop(game, headless=True)

        frame = 0
        while frame < 10800 and not game.match_over:
            obs = get_observation(game, 0)
            a1, _, _ = policy.get_action(obs)
            a2, _, _ = random_opp.get_action(get_observation(game, 1))

            game.fighters[0].input = get_action_from_index(a1)
            game.fighters[1].input = get_action_from_index(a2)

            loop.step()
            frame += 1

        if game.winner == 0:
            wins += 1

    results['vs_random'] = wins / num_games

    # vs Aggressive
    wins = 0
    for _ in range(num_games):
        game = create_game('battlefield', character, 'FLEX', stock_count=3)
        loop = GameLoop(game, headless=True)

        frame = 0
        while frame < 10800 and not game.match_over:
            obs = get_observation(game, 0)
            a1, _, _ = policy.get_action(obs)
            a2, _, _ = aggressive_opp.get_action(get_observation(game, 1))

            game.fighters[0].input = get_action_from_index(a1)
            game.fighters[1].input = get_action_from_index(a2)

            loop.step()
            frame += 1

        if game.winner == 0:
            wins += 1

    results['vs_aggressive'] = wins / num_games

    return results


def analyze_character_strategy(policy, character: str, num_games: int = 20) -> dict:
    """Analyze action distribution when playing as this character."""
    random_opp = RandomPolicy()
    action_counts = np.zeros(ACTION_SPACE_SIZE)
    total_frames = 0

    for _ in range(num_games):
        game = create_game('battlefield', character, 'FLEX', stock_count=3)
        loop = GameLoop(game, headless=True)

        frame = 0
        while frame < 10800 and not game.match_over:
            obs = get_observation(game, 0)
            action, _, _ = policy.get_action(obs)
            action_counts[action] += 1
            total_frames += 1

            obs2 = get_observation(game, 1)
            a2, _, _ = random_opp.get_action(obs2)

            game.fighters[0].input = get_action_from_index(action)
            game.fighters[1].input = get_action_from_index(a2)

            loop.step()
            frame += 1

    if total_frames > 0:
        dist = action_counts / total_frames
    else:
        dist = action_counts

    action_names = [
        'Nothing', 'Left', 'Right', 'Jump', 'Down',
        'Attack', 'L+Atk', 'R+Atk', 'U+Atk', 'D+Atk'
    ]

    attack_ratio = sum(dist[5:])
    move_ratio = dist[1] + dist[2]
    jump_ratio = dist[3]
    idle_ratio = dist[0]

    return {
        'action_dist': {name: float(dist[i]) for i, name in enumerate(action_names)},
        'attack_ratio': float(attack_ratio),
        'move_ratio': float(move_ratio),
        'jump_ratio': float(jump_ratio),
        'idle_ratio': float(idle_ratio)
    }


def generate_full_report(checkpoint_path: str = 'checkpoints/cycle4/best_vs_aggressive.npz',
                         games_per_matchup: int = 40,
                         output_dir: str = 'checkpoints/matchup_analysis'):
    """Generate comprehensive character matchup report."""
    print("=" * 60)
    print("CHARACTER MATCHUP MATRIX GENERATION")
    print("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load best policy
    policy = PolicyNetwork(OBSERVATION_SIZE, ACTION_SPACE_SIZE, 256)
    policy.load(checkpoint_path)
    policy.to('cuda')
    print(f"Loaded policy: {checkpoint_path}")

    # 1. Character vs Character matchup matrix (same policy both sides)
    print("\n--- Character Matchup Matrix ---")
    print("(Same policy plays both sides to reveal character balance)")

    matrix = {c1: {} for c1 in CHARACTERS}

    for char1 in CHARACTERS:
        for char2 in CHARACTERS:
            result = run_matchup(policy, char1, char2, num_games=games_per_matchup)
            matrix[char1][char2] = result['p1_win_rate']
            print(f"  {char1} vs {char2}: {result['p1_win_rate']:.1%}")

    # Print formatted matrix
    print("\nMatchup Matrix (row vs column):")
    header = "        "
    for char in CHARACTERS:
        header += f" {char:>6}"
    print(header)
    print("-" * len(header))

    for char1 in CHARACTERS:
        row = f"{char1:>6}  "
        for char2 in CHARACTERS:
            wr = matrix[char1][char2]
            row += f" {wr:>5.1%}"
        print(row)

    # 2. Each character vs baselines
    print("\n--- Character vs Baselines ---")
    baseline_results = {}

    for char in CHARACTERS:
        results = run_vs_baselines(policy, char, num_games=40)
        baseline_results[char] = results
        print(f"  {char}: vs Random={results['vs_random']:.1%}, vs Aggressive={results['vs_aggressive']:.1%}")

    # 3. Strategy analysis per character
    print("\n--- Character Strategy Analysis ---")
    strategies = {}

    for char in CHARACTERS:
        strat = analyze_character_strategy(policy, char, num_games=20)
        strategies[char] = strat

        # Classify playstyle
        if strat['attack_ratio'] > 0.5:
            style = 'Hyper-Aggressive'
        elif strat['attack_ratio'] > 0.4:
            style = 'Aggressive'
        elif strat['move_ratio'] > 0.3:
            style = 'Mobile'
        elif strat['idle_ratio'] > 0.15:
            style = 'Defensive'
        else:
            style = 'Balanced'

        strategies[char]['playstyle'] = style
        print(f"  {char}: {style} (Atk={strat['attack_ratio']:.1%}, Move={strat['move_ratio']:.1%})")

    # 4. Identify character tier list
    print("\n--- Character Tier Analysis ---")

    # Calculate overall strength (average win rate)
    character_strength = {}
    for char in CHARACTERS:
        # Average of: matchup wins, baseline wins
        avg_matchup = np.mean([matrix[char][c] for c in CHARACTERS])
        avg_baseline = (baseline_results[char]['vs_random'] + baseline_results[char]['vs_aggressive']) / 2
        character_strength[char] = (avg_matchup * 0.4 + avg_baseline * 0.6)  # Weight baselines more

    sorted_chars = sorted(character_strength.items(), key=lambda x: x[1], reverse=True)

    print("\nOverall Character Rankings:")
    for i, (char, score) in enumerate(sorted_chars, 1):
        vs_agg = baseline_results[char]['vs_aggressive']
        tier = 'S' if vs_agg >= 0.7 else 'A' if vs_agg >= 0.55 else 'B' if vs_agg >= 0.4 else 'C'
        print(f"  {i}. {char}: Score={score:.3f} (vs Aggressive={vs_agg:.1%}) [Tier {tier}]")

    # 5. Identify counter relationships
    print("\n--- Counter Relationships ---")
    counters = []
    for char1 in CHARACTERS:
        for char2 in CHARACTERS:
            if char1 != char2:
                wr = matrix[char1][char2]
                if wr >= 0.60:
                    counters.append((char1, char2, wr))

    if counters:
        for c1, c2, wr in sorted(counters, key=lambda x: x[2], reverse=True):
            print(f"  {c1} counters {c2} ({wr:.1%})")
    else:
        print("  No strong counter relationships detected (all matchups <60%)")

    # Generate report
    report = {
        'checkpoint': checkpoint_path,
        'games_per_matchup': games_per_matchup,
        'matchup_matrix': matrix,
        'baseline_results': baseline_results,
        'character_strategies': strategies,
        'character_rankings': [{'character': c, 'score': s,
                               'vs_aggressive': baseline_results[c]['vs_aggressive']}
                              for c, s in sorted_chars],
        'counter_relationships': [{'counter': c1, 'counters': c2, 'win_rate': wr}
                                 for c1, c2, wr in counters] if counters else []
    }

    report_path = f'{output_dir}/matchup_analysis.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Report saved to: {report_path}")
    print(f"{'='*60}")

    return report


if __name__ == '__main__':
    generate_full_report()
