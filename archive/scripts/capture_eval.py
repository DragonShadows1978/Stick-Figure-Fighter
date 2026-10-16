#!/usr/bin/env python3
"""
Capture screenshots during an evaluation game.
Uses the trained policy to play against random opponent.
"""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set display for headless operation
os.environ['DISPLAY'] = ':99'

from policy import PolicyNetwork
from evaluate import RandomPolicy, AggressivePolicy
from game_engine import (
    create_game, GameLoop, get_observation, get_action_from_index,
    OBSERVATION_SIZE, ACTION_SPACE_SIZE
)

# Import vision tool for screenshots
sys.path.insert(0, '/home/vader/mini-mind-v2')
from vision_tool import capture_screen


def run_eval_with_screenshots(
    policy_path: str,
    opponent_type: str = 'random',
    num_games: int = 3,
    capture_interval: int = 60,  # frames between captures
    output_dir: str = '/tmp/eval_captures'
):
    """
    Run evaluation games with screenshot capture.

    Args:
        policy_path: Path to trained policy checkpoint
        opponent_type: 'random' or 'aggressive'
        num_games: Number of games to play
        capture_interval: Frames between screenshots
        output_dir: Directory to save screenshots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load trained policy
    print(f'Loading policy from {policy_path}...')
    policy = PolicyNetwork(obs_size=OBSERVATION_SIZE, action_size=ACTION_SPACE_SIZE)
    policy.to('cuda')
    policy.load(policy_path)

    # Create opponent
    if opponent_type == 'aggressive':
        opponent = AggressivePolicy()
    else:
        opponent = RandomPolicy()

    print(f'Running {num_games} games vs {opponent_type} policy...')
    results = []
    screenshot_count = 0

    for game_idx in range(num_games):
        print(f'\n--- Game {game_idx + 1}/{num_games} ---')

        # Create game
        game = create_game(
            stage_name='battlefield',
            p1_character='FLEX',
            p2_character='FLEX',
            stock_count=3
        )
        loop = GameLoop(game, headless=False)  # Enable rendering

        frame = 0
        max_frames = 10800  # 3 minutes

        while frame < max_frames:
            # Get observations
            obs = [get_observation(game, i) for i in range(2)]

            # Get actions
            action_p1, _, _ = policy.get_action(obs[0])
            action_p2, _, _ = opponent.get_action(obs[1])

            # Apply actions
            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            # Step simulation
            match_ongoing = loop.step()
            frame += 1

            # Capture screenshot at intervals
            if frame % capture_interval == 0:
                screenshot_path = f'{output_dir}/game{game_idx+1}_frame{frame:05d}.png'
                result = capture_screen(':99', screenshot_path)
                if result['success']:
                    screenshot_count += 1
                    if frame % (capture_interval * 5) == 0:
                        print(f'  Frame {frame}: P1 damage={game.fighters[0].damage_percent:.0f}%, '
                              f'P2 damage={game.fighters[1].damage_percent:.0f}%, '
                              f'P1 stocks={game.fighters[0].stocks}, P2 stocks={game.fighters[1].stocks}')

            if not match_ongoing or game.match_over:
                break

            # Small delay for rendering
            time.sleep(1 / 120)  # 120 fps internal, game runs at 60 fps

        # Record result
        winner = game.winner
        results.append({
            'game': game_idx + 1,
            'winner': winner,
            'frames': frame,
            'p1_stocks': game.fighters[0].stocks,
            'p2_stocks': game.fighters[1].stocks,
            'p1_trained': True
        })

        trained_won = winner == 0
        print(f'  Result: {"Trained policy WINS" if trained_won else "Opponent wins"} '
              f'({game.fighters[0].stocks} vs {game.fighters[1].stocks} stocks)')

    # Summary
    print(f'\n=== SUMMARY ===')
    wins = sum(1 for r in results if r['winner'] == 0)
    print(f'Trained policy: {wins}/{num_games} wins ({wins/num_games*100:.0f}%)')
    print(f'Total screenshots captured: {screenshot_count}')
    print(f'Screenshots saved to: {output_dir}')

    return results, output_dir


if __name__ == '__main__':
    results, output_dir = run_eval_with_screenshots(
        policy_path='checkpoints/full_run/checkpoint_001000.npz',
        opponent_type='random',
        num_games=3,
        capture_interval=120,  # Every 2 seconds
        output_dir='/tmp/eval_captures'
    )
