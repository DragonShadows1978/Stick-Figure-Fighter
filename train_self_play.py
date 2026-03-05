#!/usr/bin/env python3
"""
Main Training Script for Stick Figure Fighter Self-Play.

Usage:
    python train_self_play.py [--games N] [--character CHAR] [--resume PATH]

This script:
1. Initializes the PPO agent with CUDA acceleration
2. Runs self-play training with dual agents
3. Tracks Elo ratings across generations
4. Saves checkpoints periodically
5. Evaluates emergent behaviors

Target: 1000 training games to see if agents discover
combos, edge-guarding, spacing, and character-specific strategies.
"""
import os
import sys
import time
import argparse
import json
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set display for headless operation
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

import torch
from self_play import SelfPlayManager, SelfPlayConfig
from evaluate import (
    evaluate_vs_random, evaluate_vs_aggressive,
    analyze_behavior, RandomPolicy
)
from policy import PolicyNetwork
from game_engine import OBSERVATION_SIZE, ACTION_SPACE_SIZE


def print_header():
    """Print training header."""
    print("=" * 60)
    print("  STICK FIGURE FIGHTER - SELF-PLAY TRAINING")
    print("  Using PPO with CUDA Acceleration")
    print("=" * 60)


def print_gpu_info():
    """Print GPU memory info."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
    else:
        print("CUDA not available")


def run_training(args):
    """Run the full training loop."""
    print_header()

    # Configuration
    config = SelfPlayConfig(
        character=args.character,
        stage=args.stage,
        stock_count=args.stocks,
        max_frames_per_game=args.max_frames,

        # Reward shaping
        stock_taken_reward=1.0,
        stock_lost_reward=-1.0,
        damage_dealt_bonus=0.001,  # Per 1% damage
        win_bonus=0.5,
        lose_penalty=-0.5,

        # Training settings
        games_per_iteration=args.games_per_update,
        total_games=args.games,
        save_every=args.save_every,
        eval_every=args.eval_every,

        # Output
        checkpoint_dir=args.checkpoint_dir,
        log_file="training_log.json"
    )

    print(f"\nConfiguration:")
    print(f"  Character: {config.character} vs {config.character}")
    print(f"  Stage: {config.stage}")
    print(f"  Stocks: {config.stock_count}")
    print(f"  Total games: {config.total_games}")
    print(f"  Games per PPO update: {config.games_per_iteration}")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")

    # Initialize training
    print(f"\nInitializing...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print_gpu_info()

    manager = SelfPlayManager(config, device='cuda')

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        manager.load_checkpoint(args.resume)

    # Pre-training evaluation
    if not args.skip_eval:
        print("\nPre-training evaluation vs random...")
        pre_eval = evaluate_vs_random(manager.policy, num_games=20, character=config.character)
        print(f"  Win rate vs random: {pre_eval['win_rate']:.1%}")

    # Training callbacks
    eval_results = []

    def on_update(stats):
        """Called after each PPO update."""
        # Periodic evaluation
        if stats['games_played'] % config.eval_every == 0 and not args.skip_eval:
            print(f"\n[Evaluation at {stats['games_played']} games]")
            vs_random = evaluate_vs_random(manager.policy, num_games=10, character=config.character)
            print(f"  vs Random: {vs_random['win_rate']:.1%} win rate")

            vs_agg = evaluate_vs_aggressive(manager.policy, num_games=10, character=config.character)
            print(f"  vs Aggressive: {vs_agg['win_rate']:.1%} win rate")

            eval_results.append({
                'games': stats['games_played'],
                'vs_random': vs_random,
                'vs_aggressive': vs_agg
            })

    manager.on_update = on_update

    # Run training
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60 + "\n")

    start_time = time.time()
    manager.train()
    elapsed = time.time() - start_time

    # Post-training evaluation
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    print("\nVs Random Policy (50 games):")
    post_random = evaluate_vs_random(manager.policy, num_games=50, character=config.character)
    print(f"  Win rate: {post_random['win_rate']:.1%}")
    print(f"  Avg damage dealt: {post_random['avg_damage_dealt']:.1f}")
    print(f"  Avg stocks taken: {post_random['avg_stocks_taken']:.2f}")

    print("\nVs Aggressive Policy (50 games):")
    post_aggressive = evaluate_vs_aggressive(manager.policy, num_games=50, character=config.character)
    print(f"  Win rate: {post_aggressive['win_rate']:.1%}")
    print(f"  Avg damage dealt: {post_aggressive['avg_damage_dealt']:.1f}")
    print(f"  Avg stocks taken: {post_aggressive['avg_stocks_taken']:.2f}")

    print("\nBehavior Analysis:")
    behavior = analyze_behavior(manager.policy, num_games=20, character=config.character)
    print(f"  Emergent playstyle: {behavior['playstyle']}")
    print(f"  Movement ratio: {behavior['movement_ratio']:.1%}")
    print(f"  Attack ratio: {behavior['attack_ratio']:.1%}")
    print(f"  Jump ratio: {behavior['jump_ratio']:.1%}")
    print(f"  Idle ratio: {behavior['idle_ratio']:.1%}")

    print("\n  Top actions:")
    sorted_actions = sorted(behavior['action_distribution'].items(),
                           key=lambda x: x[1], reverse=True)
    for action, prob in sorted_actions[:5]:
        print(f"    {action}: {prob:.1%}")

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total games: {manager.games_played}")
    print(f"  Total time: {elapsed:.1f}s ({manager.games_played / elapsed:.1f} games/sec)")
    print(f"  Generations: {manager.current_generation}")

    final_rating = manager.elo_tracker.get_rating(f"gen_{manager.current_generation - 1}")
    print(f"  Final Elo: {final_rating.rating:.0f}")
    print(f"  Final win rate: {final_rating.win_rate:.1%}")

    # Elo progression
    print("\nElo Progression (every 100 games):")
    for i in range(0, manager.current_generation, 10):
        rating = manager.elo_tracker.get_rating(f"gen_{i}")
        print(f"  Gen {i:4d}: {rating.rating:.0f}")

    # Save final report
    report = {
        'config': {
            'character': config.character,
            'stage': config.stage,
            'total_games': config.total_games,
            'games_per_iteration': config.games_per_iteration
        },
        'training': {
            'elapsed_seconds': elapsed,
            'games_per_second': manager.games_played / elapsed,
            'final_generation': manager.current_generation,
            'final_elo': final_rating.rating
        },
        'evaluation': {
            'vs_random': post_random,
            'vs_aggressive': post_aggressive,
            'behavior': behavior
        },
        'eval_history': eval_results
    }

    report_path = os.path.join(config.checkpoint_dir, "training_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Self-play training for Stick Figure Fighter"
    )

    # Training parameters
    parser.add_argument("--games", type=int, default=1000,
                        help="Total training games (default: 1000)")
    parser.add_argument("--games-per-update", type=int, default=10,
                        help="Games before each PPO update (default: 10)")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N games (default: 100)")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate every N games (default: 100)")

    # Game settings
    parser.add_argument("--character", type=str, default="FLEX",
                        choices=["FLEX", "RUSH", "ZONE", "GRAB"],
                        help="Character for self-play (default: FLEX)")
    parser.add_argument("--stage", type=str, default="battlefield",
                        choices=["battlefield", "final_destination"],
                        help="Stage to play on (default: battlefield)")
    parser.add_argument("--stocks", type=int, default=3,
                        help="Stocks per player (default: 3)")
    parser.add_argument("--max-frames", type=int, default=10800,
                        help="Max frames per game (default: 10800 = 3min)")

    # Output
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints/self_play",
                        help="Directory for checkpoints")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint file")

    # Evaluation
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip periodic evaluation")

    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
