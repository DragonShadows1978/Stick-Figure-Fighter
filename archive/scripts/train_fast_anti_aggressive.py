#!/usr/bin/env python3
"""
Fast Anti-Aggressive Training - Optimized for speed.

Goal: Achieve >50% win rate vs AggressivePolicy in minimal time.

Strategy:
1. Start from best existing checkpoint
2. Pure aggressive focus (no random warmup - already done)
3. Strong counter-aggression rewards
4. Frequent evaluation to track progress
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

import torch
from policy import PolicyNetwork
from experience import Episode
from trainer import PPOTrainer, TrainingConfig
from evaluate import RandomPolicy, AggressivePolicy, evaluate_vs_random, evaluate_vs_aggressive
from game_engine import (
    create_game, GameLoop, get_observation, get_action_from_index,
    OBSERVATION_SIZE, ACTION_SPACE_SIZE
)


class FastAntiAggressiveTrainer:
    """Optimized trainer focused on beating AggressivePolicy."""

    def __init__(self, device: str = 'cuda', checkpoint_dir: str = 'checkpoints/fast_anti_agg',
                 base_checkpoint: str = None):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Create policy
        self.policy = PolicyNetwork(
            obs_size=OBSERVATION_SIZE,
            action_size=ACTION_SPACE_SIZE,
            hidden_size=256
        )
        self.policy.to(device)

        # Load base checkpoint if provided
        if base_checkpoint and os.path.exists(base_checkpoint):
            print(f"Loading base checkpoint: {base_checkpoint}")
            self.policy.load(base_checkpoint)
        else:
            print("Starting from scratch")

        # Lower LR for fine-tuning
        config = TrainingConfig(
            learning_rate=1e-4,  # Lower for fine-tuning
            batch_size=64,
            epochs_per_update=4,
            gamma=0.99,
            gae_lambda=0.95,
            entropy_coef=0.02,  # Higher entropy for exploration
            value_coef=0.5
        )
        self.trainer = PPOTrainer(self.policy, config, device=device)

        self.aggressive_opp = AggressivePolicy()
        self.random_opp = RandomPolicy()

        self.games_played = 0
        self.best_vs_aggressive = 0.0
        self.training_log = []

    def run_game_fast(self, opponent) -> tuple:
        """Fast game execution with optimized reward."""
        game = create_game('battlefield', 'FLEX', 'FLEX', stock_count=3)
        loop = GameLoop(game, headless=True)

        episode = Episode(player_id=0)
        prev_damage = [0.0, 0.0]
        prev_stocks = [3, 3]

        frame = 0
        max_frames = 7200  # Shorter games for faster training

        while frame < max_frames and not game.match_over:
            obs = [get_observation(game, i) for i in range(2)]

            action_p1, log_prob, value = self.policy.get_action(obs[0])
            action_p2, _, _ = opponent.get_action(obs[1])

            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            match_ongoing = loop.step()
            frame += 1

            # Simple but effective reward
            reward = 0.0

            # Stock changes (most important)
            if game.fighters[1].stocks < prev_stocks[1]:
                reward += 2.0  # Big reward for taking stock
            if game.fighters[0].stocks < prev_stocks[0]:
                reward -= 2.0  # Big penalty for losing stock

            # Damage dealt/taken
            dmg_dealt = game.fighters[1].damage_percent - prev_damage[1]
            dmg_taken = game.fighters[0].damage_percent - prev_damage[0]

            if dmg_dealt > 0:
                reward += dmg_dealt * 0.003

            if dmg_taken > 0:
                # Scale penalty based on our damage state
                if prev_damage[0] >= 80:
                    reward -= dmg_taken * 0.005  # Much worse at high damage
                else:
                    reward -= dmg_taken * 0.002

            prev_damage = [game.fighters[i].damage_percent for i in range(2)]
            prev_stocks = [game.fighters[i].stocks for i in range(2)]

            done = not match_ongoing or game.match_over

            episode.add(obs[0], action_p1, reward, done, log_prob, value)

            if done:
                break

        # Win/loss
        winner = game.winner
        if winner == 0:
            episode.transitions[-1].reward += 1.0
        elif winner == 1:
            episode.transitions[-1].reward -= 1.0

        self.games_played += 1
        return episode, winner

    def train_block(self, num_games: int, opponent, desc: str = "Training"):
        """Train for a block of games."""
        print(f"\n--- {desc} ({num_games} games) ---")
        start = time.time()

        episodes = []
        wins = 0

        for i in range(num_games):
            episode, winner = self.run_game_fast(opponent)
            episodes.append(episode)
            if winner == 0:
                wins += 1

            if len(episodes) >= 10:
                self.trainer.update(episodes)
                episodes = []

        if episodes:
            self.trainer.update(episodes)

        elapsed = time.time() - start
        win_rate = wins / num_games

        print(f"  {wins}/{num_games} wins ({win_rate:.1%}) in {elapsed:.1f}s")
        return win_rate

    def evaluate_quick(self, num_games: int = 30) -> dict:
        """Quick evaluation."""
        vs_agg = evaluate_vs_aggressive(self.policy, num_games=num_games)
        return vs_agg

    def run_training(self, total_games: int = 800):
        """Run fast training focused on aggressive opponent."""
        print("\n" + "="*60)
        print("FAST ANTI-AGGRESSIVE TRAINING")
        print(f"Target: >50% vs AggressivePolicy")
        print("="*60)

        start_time = time.time()

        # Initial eval
        print("\n--- Initial Evaluation ---")
        initial = self.evaluate_quick()
        print(f"  vs Aggressive: {initial['win_rate']:.1%}")
        self.best_vs_aggressive = initial['win_rate']

        # Training blocks with frequent evaluation
        games_per_block = 100
        num_blocks = total_games // games_per_block

        for block in range(num_blocks):
            # Mix of aggressive (80%) and random (20%)
            if np.random.random() < 0.8:
                opponent = self.aggressive_opp
                desc = f"Block {block+1} (vs Aggressive)"
            else:
                opponent = self.random_opp
                desc = f"Block {block+1} (vs Random)"

            self.train_block(games_per_block, opponent, desc)

            # Evaluate every 2 blocks
            if (block + 1) % 2 == 0:
                eval_result = self.evaluate_quick(num_games=40)
                print(f"  Eval: {eval_result['win_rate']:.1%} vs Aggressive")

                if eval_result['win_rate'] > self.best_vs_aggressive:
                    self.best_vs_aggressive = eval_result['win_rate']
                    self.policy.save(f'{self.checkpoint_dir}/checkpoint_best.npz')
                    print(f"  New best! Saved.")

                if eval_result['win_rate'] >= 0.50:
                    print("\n TARGET ACHIEVED!")
                    break

                self.training_log.append({
                    'games': self.games_played,
                    'vs_aggressive': eval_result['win_rate']
                })

        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)

        vs_random = evaluate_vs_random(self.policy, num_games=50)
        vs_aggressive = evaluate_vs_aggressive(self.policy, num_games=60)

        print(f"  vs Random: {vs_random['win_rate']:.1%}")
        print(f"  vs Aggressive: {vs_aggressive['win_rate']:.1%}")

        elapsed = time.time() - start_time

        # Save final
        self.policy.save(f'{self.checkpoint_dir}/checkpoint_final.npz')

        # Report
        report = {
            'training': {
                'total_games': self.games_played,
                'elapsed_seconds': elapsed,
                'games_per_second': self.games_played / elapsed,
                'best_vs_aggressive': self.best_vs_aggressive,
                'log': self.training_log
            },
            'final_evaluation': {
                'vs_random': vs_random,
                'vs_aggressive': vs_aggressive
            }
        }

        report_path = f'{self.checkpoint_dir}/training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n--- Summary ---")
        print(f"Games: {self.games_played}")
        print(f"Time: {elapsed:.1f}s ({self.games_played/elapsed:.1f} g/s)")
        print(f"Best vs Aggressive: {self.best_vs_aggressive:.1%}")
        print(f"Final vs Aggressive: {vs_aggressive['win_rate']:.1%}")

        if vs_aggressive['win_rate'] >= 0.50:
            print("\n SUCCESS! >50% vs Aggressive achieved!")
        else:
            print(f"\n Gap to target: {0.50 - vs_aggressive['win_rate']:.1%}")

        return report


def find_best_checkpoint():
    """Find the best existing checkpoint to start from."""
    checkpoint_dirs = [
        'checkpoints/full_run/checkpoint_001000.npz',
        'checkpoints/curriculum_v3/checkpoint_final.npz',
        'checkpoints/anti_aggressive_v5/checkpoint_final.npz',
        'checkpoints/anti_aggressive_v6/checkpoint_final.npz',
    ]

    for cp in checkpoint_dirs:
        if os.path.exists(cp):
            return cp

    return None


def main():
    base_cp = find_best_checkpoint()

    trainer = FastAntiAggressiveTrainer(
        device='cuda',
        checkpoint_dir='checkpoints/fast_anti_agg',
        base_checkpoint=base_cp
    )

    report = trainer.run_training(total_games=800)
    return report


if __name__ == '__main__':
    main()
