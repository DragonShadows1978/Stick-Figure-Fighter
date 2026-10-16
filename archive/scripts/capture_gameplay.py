#!/usr/bin/env python3
"""
Capture actual gameplay screenshots during evaluation.
Fixes the issue where capture_eval.py shows menus instead of gameplay.

This script properly initializes the X11 renderer and runs matches
with trained policies to capture real combat footage.
"""
import os
import sys
import time
import numpy as np

# Set display for virtual framebuffer
os.environ['DISPLAY'] = ':99'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/vader/mini-mind-v2')

from policy import PolicyNetwork
from evaluate import RandomPolicy, AggressivePolicy
from game_engine import (
    create_game, GameLoop, get_observation, get_action_from_index,
    OBSERVATION_SIZE, ACTION_SPACE_SIZE, GameState
)
from vision_tool import capture_screen


class GameplayCapture:
    """
    Runs evaluation games with proper X11 rendering and screenshot capture.
    """

    def __init__(self, checkpoint_path: str, output_dir: str = '/tmp/gameplay_captures'):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load trained policy
        print(f'Loading policy from {checkpoint_path}...')
        self.policy = PolicyNetwork(obs_size=OBSERVATION_SIZE, action_size=ACTION_SPACE_SIZE)
        self.policy.to('cuda')
        self.policy.load(checkpoint_path)

        # Initialize X11 renderer
        print('Initializing X11 renderer on :99...')
        from renderer import X11Renderer
        self.renderer = X11Renderer(
            display_name=':99',
            title='ML Evaluation Match'
        )

        # Screenshot counter
        self.screenshot_count = 0

    def run_evaluation_game(
        self,
        opponent_type: str = 'random',
        p1_character: str = 'FLEX',
        p2_character: str = 'FLEX',
        capture_interval: int = 60,  # frames between captures
        max_frames: int = 5400,  # 90 seconds at 60fps
        game_index: int = 0
    ) -> dict:
        """
        Run a single evaluation game with screenshot capture.

        Args:
            opponent_type: 'random' or 'aggressive'
            p1_character: Character for trained agent
            p2_character: Character for opponent
            capture_interval: Frames between screenshots
            max_frames: Maximum frames before timeout
            game_index: Game number for filename prefixes

        Returns:
            Dict with game results and stats
        """
        # Create opponent policy
        if opponent_type == 'aggressive':
            opponent = AggressivePolicy()
        else:
            opponent = RandomPolicy()

        # Create game state
        game = create_game(
            stage_name='battlefield',
            p1_character=p1_character,
            p2_character=p2_character,
            stock_count=3
        )

        # Create game loop (headless=True for simulation, we handle rendering manually)
        loop = GameLoop(game, headless=True)

        frame = 0
        screenshots_taken = []
        action_history = []

        print(f'  Starting game {game_index + 1}: {p1_character} vs {opponent_type} {p2_character}')

        while frame < max_frames and not game.match_over:
            # Get observations
            obs_p1 = get_observation(game, 0)
            obs_p2 = get_observation(game, 1)

            # Get actions from policies
            action_p1, _, _ = self.policy.get_action(obs_p1)
            action_p2, _, _ = opponent.get_action(obs_p2)

            # Track actions for behavior analysis
            action_history.append((action_p1, action_p2))

            # Apply actions to fighters
            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            # Step simulation
            loop.step()
            frame += 1

            # Render to X11
            self.renderer.render(game)
            self.renderer.flip()

            # Capture screenshot at intervals
            if frame % capture_interval == 0:
                screenshot_path = f'{self.output_dir}/game{game_index + 1}_frame{frame:05d}.png'
                result = capture_screen(':99', screenshot_path)
                if result['success']:
                    self.screenshot_count += 1
                    screenshots_taken.append(screenshot_path)

                    # Print status every 5 captures
                    if len(screenshots_taken) % 5 == 0:
                        print(f'    Frame {frame}: P1 dmg={game.fighters[0].damage_percent:.0f}% '
                              f'P2 dmg={game.fighters[1].damage_percent:.0f}% '
                              f'stocks={game.fighters[0].stocks}:{game.fighters[1].stocks}')

            # Small delay for rendering visibility (can be adjusted)
            time.sleep(1/120)  # 120fps internal

        # Game ended
        winner = game.winner

        # Analyze action distribution
        action_counts = np.zeros(ACTION_SPACE_SIZE)
        for a1, _ in action_history:
            action_counts[a1] += 1
        action_dist = action_counts / len(action_history) if action_history else action_counts

        results = {
            'game_index': game_index + 1,
            'winner': winner,
            'trained_won': winner == 0,
            'frames': frame,
            'duration_seconds': frame / 60,
            'p1_stocks': game.fighters[0].stocks,
            'p2_stocks': game.fighters[1].stocks,
            'p1_damage': game.fighters[0].damage_percent,
            'p2_damage': game.fighters[1].damage_percent,
            'screenshots': screenshots_taken,
            'action_distribution': action_dist,
        }

        win_text = 'TRAINED WINS' if winner == 0 else 'OPPONENT WINS'
        print(f'  Result: {win_text} ({game.fighters[0].stocks} vs {game.fighters[1].stocks} stocks, {frame} frames)')

        return results

    def run_evaluation_batch(
        self,
        num_games: int = 3,
        opponent_type: str = 'random',
        p1_character: str = 'FLEX',
        p2_character: str = 'FLEX',
        capture_interval: int = 120  # Every 2 seconds
    ) -> list:
        """
        Run multiple evaluation games with screenshot capture.

        Returns:
            List of result dicts from each game
        """
        print(f'\n=== Running {num_games} evaluation games vs {opponent_type} ===')
        print(f'Characters: {p1_character} (trained) vs {p2_character} ({opponent_type})')
        print(f'Capture interval: {capture_interval} frames ({capture_interval/60:.1f}s)')
        print()

        all_results = []

        for i in range(num_games):
            result = self.run_evaluation_game(
                opponent_type=opponent_type,
                p1_character=p1_character,
                p2_character=p2_character,
                capture_interval=capture_interval,
                game_index=i
            )
            all_results.append(result)

            # Brief pause between games
            time.sleep(0.5)

        # Summary
        wins = sum(1 for r in all_results if r['trained_won'])
        print(f'\n=== Summary ===')
        print(f'Win rate: {wins}/{num_games} ({wins/num_games*100:.0f}%)')
        print(f'Total screenshots: {self.screenshot_count}')
        print(f'Output directory: {self.output_dir}')

        # Aggregate action distribution
        total_actions = np.zeros(ACTION_SPACE_SIZE)
        for r in all_results:
            total_actions += r['action_distribution'] * r['frames']
        total_actions /= sum(r['frames'] for r in all_results)

        action_names = [
            'Nothing', 'Left', 'Right', 'Jump', 'Down',
            'Attack', 'L+Atk', 'R+Atk', 'U+Atk', 'D+Atk'
        ]
        print('\nAction distribution:')
        for name, prob in zip(action_names, total_actions):
            if prob > 0.01:
                print(f'  {name}: {prob:.1%}')

        return all_results

    def cleanup(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()


def main():
    """Main entry point for gameplay capture."""
    import argparse

    parser = argparse.ArgumentParser(description='Capture gameplay screenshots during ML evaluation')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/full_run/checkpoint_001000.npz',
                        help='Path to trained policy checkpoint')
    parser.add_argument('--opponent', type=str, default='random',
                        choices=['random', 'aggressive'],
                        help='Opponent type')
    parser.add_argument('--games', type=int, default=3,
                        help='Number of games to run')
    parser.add_argument('--p1', type=str, default='FLEX',
                        choices=['RUSH', 'ZONE', 'GRAB', 'FLEX'],
                        help='P1 (trained) character')
    parser.add_argument('--p2', type=str, default='FLEX',
                        choices=['RUSH', 'ZONE', 'GRAB', 'FLEX'],
                        help='P2 (opponent) character')
    parser.add_argument('--interval', type=int, default=120,
                        help='Frames between screenshots')
    parser.add_argument('--output', type=str, default='/tmp/gameplay_captures',
                        help='Output directory for screenshots')

    args = parser.parse_args()

    capturer = GameplayCapture(
        checkpoint_path=args.checkpoint,
        output_dir=args.output
    )

    try:
        results = capturer.run_evaluation_batch(
            num_games=args.games,
            opponent_type=args.opponent,
            p1_character=args.p1,
            p2_character=args.p2,
            capture_interval=args.interval
        )
    finally:
        capturer.cleanup()

    return results


if __name__ == '__main__':
    main()
