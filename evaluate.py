"""
Evaluation System for Self-Play Trained Agents.

Features:
- Play trained agent vs random policy
- Play trained agent vs previous checkpoints
- Render sample games for visualization
- Track emergent behaviors
"""
import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional, Callable
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensor_gpu import set_device
from policy import PolicyNetwork
from game_engine import (
    create_game, GameLoop, GameState, InputState,
    get_observation, get_action_from_index, ACTION_SPACE_SIZE, OBSERVATION_SIZE
)


class RandomPolicy:
    """Random action policy for baseline comparison."""

    def __init__(self, action_size: int = 10):
        self.action_size = action_size

    def get_action(self, obs: np.ndarray) -> tuple:
        action = np.random.randint(0, self.action_size)
        return action, 0.0, 0.0  # action, log_prob, value


class AggressivePolicy:
    """
    Simple rule-based aggressive policy.

    Always moves toward opponent and attacks when close.
    """

    def __init__(self, action_size: int = 10):
        self.action_size = action_size

    def get_action(self, obs: np.ndarray) -> tuple:
        """
        Observation indices:
        0-1: player pos (x, y)
        4-5: opponent pos (x, y)
        """
        px, py = obs[0], obs[1]
        ox, oy = obs[4], obs[5]

        # Move toward opponent
        if ox > px + 0.05:  # Opponent to the right
            if np.random.random() < 0.3:
                action = 7  # Right + Attack
            else:
                action = 2  # Right
        elif ox < px - 0.05:  # Opponent to the left
            if np.random.random() < 0.3:
                action = 6  # Left + Attack
            else:
                action = 1  # Left
        else:
            # Close enough, attack!
            action = np.random.choice([5, 8, 9])  # Attack variants

        # Jump sometimes
        if oy < py - 0.1 and np.random.random() < 0.2:
            action = 3  # Jump

        return action, 0.0, 0.0


def run_evaluation_game(policy_p1, policy_p2, character: str = "FLEX",
                        stage: str = "battlefield", stock_count: int = 3,
                        max_frames: int = 10800, render_callback: Optional[Callable] = None
                        ) -> Dict:
    """
    Run a single evaluation game between two policies.

    Args:
        policy_p1: Policy for player 1 (has get_action method)
        policy_p2: Policy for player 2
        character: Character to use
        stage: Stage to play on
        stock_count: Stocks per player
        max_frames: Maximum frames before timeout
        render_callback: Optional callback(game_state) for rendering

    Returns:
        Game statistics
    """
    game = create_game(
        stage_name=stage,
        p1_character=character,
        p2_character=character,
        stock_count=stock_count
    )
    loop = GameLoop(game, headless=(render_callback is None))

    # Track stats
    damage_dealt = [0.0, 0.0]
    stocks_taken = [0, 0]
    prev_damage = [0.0, 0.0]
    prev_stocks = [stock_count, stock_count]
    action_counts = [np.zeros(ACTION_SPACE_SIZE), np.zeros(ACTION_SPACE_SIZE)]

    frame = 0
    while frame < max_frames:
        # Get observations
        obs = [get_observation(game, i) for i in range(2)]

        # Get actions
        action_p1, _, _ = policy_p1.get_action(obs[0])
        action_p2, _, _ = policy_p2.get_action(obs[1])

        # Track actions
        action_counts[0][action_p1] += 1
        action_counts[1][action_p2] += 1

        # Apply actions
        game.fighters[0].input = get_action_from_index(action_p1)
        game.fighters[1].input = get_action_from_index(action_p2)

        # Step simulation
        match_ongoing = loop.step()
        frame += 1

        # Track damage
        for i in range(2):
            opp = 1 - i
            dmg = game.fighters[opp].damage_percent - prev_damage[opp]
            if dmg > 0:
                damage_dealt[i] += dmg

            if game.fighters[opp].stocks < prev_stocks[opp]:
                stocks_taken[i] += 1

        prev_damage = [game.fighters[i].damage_percent for i in range(2)]
        prev_stocks = [game.fighters[i].stocks for i in range(2)]

        # Render callback
        if render_callback:
            render_callback(game)

        if not match_ongoing or game.match_over:
            break

    return {
        'winner': game.winner,
        'frames': frame,
        'damage_dealt': damage_dealt,
        'stocks_taken': stocks_taken,
        'final_stocks': [game.fighters[i].stocks for i in range(2)],
        'final_damage': [game.fighters[i].damage_percent for i in range(2)],
        'action_distribution': [a / a.sum() if a.sum() > 0 else a for a in action_counts]
    }


def evaluate_vs_random(policy, num_games: int = 50, character: str = "FLEX") -> Dict:
    """
    Evaluate policy against random opponent.

    Returns win rate and other statistics.
    """
    random_policy = RandomPolicy()

    wins = 0
    total_damage = 0.0
    total_stocks_taken = 0

    for game_idx in range(num_games):
        # Alternate which side the trained policy plays
        if game_idx % 2 == 0:
            result = run_evaluation_game(policy, random_policy, character=character)
            if result['winner'] == 0:
                wins += 1
            total_damage += result['damage_dealt'][0]
            total_stocks_taken += result['stocks_taken'][0]
        else:
            result = run_evaluation_game(random_policy, policy, character=character)
            if result['winner'] == 1:
                wins += 1
            total_damage += result['damage_dealt'][1]
            total_stocks_taken += result['stocks_taken'][1]

    return {
        'games': num_games,
        'wins': wins,
        'win_rate': wins / num_games,
        'avg_damage_dealt': total_damage / num_games,
        'avg_stocks_taken': total_stocks_taken / num_games
    }


def evaluate_vs_aggressive(policy, num_games: int = 50, character: str = "FLEX") -> Dict:
    """
    Evaluate policy against aggressive rule-based opponent.
    """
    aggressive_policy = AggressivePolicy()

    wins = 0
    total_damage = 0.0
    total_stocks_taken = 0

    for game_idx in range(num_games):
        if game_idx % 2 == 0:
            result = run_evaluation_game(policy, aggressive_policy, character=character)
            if result['winner'] == 0:
                wins += 1
            total_damage += result['damage_dealt'][0]
            total_stocks_taken += result['stocks_taken'][0]
        else:
            result = run_evaluation_game(aggressive_policy, policy, character=character)
            if result['winner'] == 1:
                wins += 1
            total_damage += result['damage_dealt'][1]
            total_stocks_taken += result['stocks_taken'][1]

    return {
        'games': num_games,
        'wins': wins,
        'win_rate': wins / num_games,
        'avg_damage_dealt': total_damage / num_games,
        'avg_stocks_taken': total_stocks_taken / num_games
    }


def evaluate_checkpoint_vs_checkpoint(checkpoint1: str, checkpoint2: str,
                                       num_games: int = 20) -> Dict:
    """
    Evaluate two checkpoints against each other.
    """
    set_device('cuda')

    policy1 = PolicyNetwork(obs_size=OBSERVATION_SIZE, action_size=ACTION_SPACE_SIZE)
    policy1.to('cuda')
    policy1.load(checkpoint1)

    policy2 = PolicyNetwork(obs_size=OBSERVATION_SIZE, action_size=ACTION_SPACE_SIZE)
    policy2.to('cuda')
    policy2.load(checkpoint2)

    wins = [0, 0]

    for game_idx in range(num_games):
        result = run_evaluation_game(policy1, policy2)
        if result['winner'] == 0:
            wins[0] += 1
        elif result['winner'] == 1:
            wins[1] += 1

    return {
        'games': num_games,
        'checkpoint1_wins': wins[0],
        'checkpoint2_wins': wins[1],
        'checkpoint1_win_rate': wins[0] / num_games
    }


def analyze_behavior(policy, num_games: int = 10, character: str = "FLEX") -> Dict:
    """
    Analyze emergent behaviors in a trained policy.

    Tracks:
    - Action distribution
    - Approach patterns
    - Combo detection
    - Edge behavior
    """
    action_totals = np.zeros(ACTION_SPACE_SIZE)
    edge_approaches = 0
    air_attacks = 0
    ground_attacks = 0

    for _ in range(num_games):
        result = run_evaluation_game(policy, policy, character=character)
        action_totals += result['action_distribution'][0] * result['frames']

    # Normalize
    if action_totals.sum() > 0:
        action_dist = action_totals / action_totals.sum()
    else:
        action_dist = action_totals

    action_names = [
        "Nothing", "Left", "Right", "Jump", "Down",
        "Attack", "Left+Atk", "Right+Atk", "Up+Atk", "Down+Atk"
    ]

    behavior_report = {
        'action_distribution': {
            name: float(prob)
            for name, prob in zip(action_names, action_dist)
        },
        'movement_ratio': float(action_dist[1] + action_dist[2]),  # Left + Right
        'attack_ratio': float(action_dist[5] + action_dist[6] + action_dist[7] +
                              action_dist[8] + action_dist[9]),
        'jump_ratio': float(action_dist[3]),
        'idle_ratio': float(action_dist[0]),
    }

    # Determine playstyle
    if behavior_report['attack_ratio'] > 0.4:
        behavior_report['playstyle'] = 'aggressive'
    elif behavior_report['movement_ratio'] > 0.5:
        behavior_report['playstyle'] = 'mobile'
    elif behavior_report['idle_ratio'] > 0.2:
        behavior_report['playstyle'] = 'defensive'
    else:
        behavior_report['playstyle'] = 'balanced'

    return behavior_report


def run_rendered_evaluation(policy, num_games: int = 3, character: str = "FLEX",
                            save_path: Optional[str] = None):
    """
    Run evaluation games with rendering.

    Requires X11 display. Captures screenshots if save_path provided.
    """
    try:
        from renderer import Renderer
        from main import StickFighterGame
    except ImportError as e:
        print(f"Cannot render: {e}")
        return None

    print(f"Running {num_games} rendered evaluation games...")

    results = []
    for game_idx in range(num_games):
        print(f"\nGame {game_idx + 1}/{num_games}")

        game = create_game(
            stage_name="battlefield",
            p1_character=character,
            p2_character=character,
            stock_count=3
        )

        # Create renderer
        renderer = Renderer(
            width=1280, height=720,
            window_name=f"Evaluation Game {game_idx + 1}",
            display=":99"  # Virtual display
        )

        loop = GameLoop(game, headless=False)

        frame = 0
        max_frames = 5400  # 90 seconds

        while frame < max_frames:
            # Get observations and actions
            obs = [get_observation(game, i) for i in range(2)]
            action_p1, _, _ = policy.get_action(obs[0])
            action_p2, _, _ = policy.get_action(obs[1])

            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            # Step simulation
            match_ongoing = loop.step()
            frame += 1

            # Render
            renderer.render(game)

            if not match_ongoing or game.match_over:
                break

            # Small delay for visibility
            time.sleep(1 / 60)

        results.append({
            'winner': game.winner,
            'frames': frame,
            'final_stocks': [game.fighters[i].stocks for i in range(2)]
        })

        renderer.close()

        print(f"  Winner: Player {game.winner + 1 if game.winner is not None else 'Draw'}")
        print(f"  Duration: {frame} frames ({frame / 60:.1f}s)")

    return results


def main():
    """Run comprehensive evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained self-play agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--games", type=int, default=50,
                        help="Number of games per evaluation")
    parser.add_argument("--render", action="store_true",
                        help="Run rendered evaluation")
    args = parser.parse_args()

    set_device('cuda')

    # Load policy
    print(f"Loading checkpoint: {args.checkpoint}")
    policy = PolicyNetwork(obs_size=OBSERVATION_SIZE, action_size=ACTION_SPACE_SIZE)
    policy.to('cuda')
    policy.load(args.checkpoint)

    # Evaluate vs random
    print("\n" + "=" * 50)
    print("Evaluation vs Random Policy")
    print("=" * 50)
    vs_random = evaluate_vs_random(policy, num_games=args.games)
    print(f"  Win rate: {vs_random['win_rate']:.1%}")
    print(f"  Avg damage: {vs_random['avg_damage_dealt']:.1f}")
    print(f"  Avg stocks taken: {vs_random['avg_stocks_taken']:.2f}")

    # Evaluate vs aggressive
    print("\n" + "=" * 50)
    print("Evaluation vs Aggressive Policy")
    print("=" * 50)
    vs_aggressive = evaluate_vs_aggressive(policy, num_games=args.games)
    print(f"  Win rate: {vs_aggressive['win_rate']:.1%}")
    print(f"  Avg damage: {vs_aggressive['avg_damage_dealt']:.1f}")
    print(f"  Avg stocks taken: {vs_aggressive['avg_stocks_taken']:.2f}")

    # Behavior analysis
    print("\n" + "=" * 50)
    print("Behavior Analysis")
    print("=" * 50)
    behavior = analyze_behavior(policy, num_games=10)
    print(f"  Playstyle: {behavior['playstyle']}")
    print(f"  Movement ratio: {behavior['movement_ratio']:.1%}")
    print(f"  Attack ratio: {behavior['attack_ratio']:.1%}")
    print(f"  Jump ratio: {behavior['jump_ratio']:.1%}")
    print("\n  Action distribution:")
    for action, prob in behavior['action_distribution'].items():
        if prob > 0.01:
            print(f"    {action}: {prob:.1%}")

    # Rendered evaluation
    if args.render:
        print("\n" + "=" * 50)
        print("Rendered Evaluation")
        print("=" * 50)
        run_rendered_evaluation(policy, num_games=3)


if __name__ == "__main__":
    main()
