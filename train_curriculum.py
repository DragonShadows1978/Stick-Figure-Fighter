#!/usr/bin/env python3
"""
Curriculum Training for Stick Figure Fighter.

Improves win rate vs AggressivePolicy by training in phases:
1. Self-play training (500 games) - learn fundamentals
2. Training vs Aggressive opponents (500 games) - learn counter-strategies
3. Mixed training (250 games) - maintain both skills

Also includes reward shaping adjustments for better defensive play.
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
from trainer import PPOTrainer, TrainingConfig
from experience import ExperienceBuffer, Episode
from game_engine import (
    create_game, GameLoop, get_observation, get_action_from_index,
    OBSERVATION_SIZE, ACTION_SPACE_SIZE, GameState, FighterState
)
from evaluate import (
    evaluate_vs_random, evaluate_vs_aggressive,
    AggressivePolicy, RandomPolicy, analyze_behavior
)


class CurriculumTrainer:
    """
    Curriculum learning trainer that progressively increases opponent difficulty.
    """

    def __init__(self, checkpoint_path: str = None, device: str = 'cuda'):
        self.device = device

        # Initialize policy
        self.policy = PolicyNetwork(
            obs_size=OBSERVATION_SIZE,
            action_size=ACTION_SPACE_SIZE,
            hidden_size=256
        )
        self.policy.to(device)

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f'Loading checkpoint: {checkpoint_path}')
            self.policy.load(checkpoint_path)

        # Training config with improved reward shaping
        self.config = TrainingConfig(
            clip_ratio=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            entropy_coef=0.015,  # Slightly higher for exploration
            value_coef=0.5,
            max_grad_norm=0.5,
            learning_rate=3e-4,
            batch_size=64,
            epochs_per_update=4,
            games_per_iteration=10
        )

        # Create trainer
        self.trainer = PPOTrainer(self.policy, self.config, device=device)

        # Opponent policies
        self.random_policy = RandomPolicy()
        self.aggressive_policy = AggressivePolicy()

        # Stats
        self.games_played = 0
        self.phase_stats = []

    def collect_episode(
        self,
        opponent_policy,
        character: str = 'FLEX',
        use_defensive_rewards: bool = False
    ) -> tuple:
        """
        Collect a single episode playing against specified opponent.

        Args:
            opponent_policy: Policy object for opponent (has get_action method)
            character: Character to use
            use_defensive_rewards: If True, add rewards for defensive play

        Returns:
            (episode_p1, winner, stats)
        """
        game = create_game(
            stage_name='battlefield',
            p1_character=character,
            p2_character=character,
            stock_count=3
        )

        loop = GameLoop(game, headless=True)

        episode = Episode(player_id=0)
        prev_damage_dealt = 0.0
        prev_stocks_taken = 0
        prev_damage_received = 0.0

        frame = 0
        max_frames = 10800  # 3 minutes

        while frame < max_frames and not game.match_over:
            # Get observations
            obs_p1 = get_observation(game, 0)
            obs_p2 = get_observation(game, 1)

            # Get actions
            action_p1, log_prob, value = self.policy.get_action(obs_p1)
            action_p2, _, _ = opponent_policy.get_action(obs_p2)

            # Apply actions
            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            # Store previous state for reward calculation
            prev_p1_stocks = game.fighters[0].stocks
            prev_p2_stocks = game.fighters[1].stocks
            prev_p1_damage = game.fighters[0].damage_percent
            prev_p2_damage = game.fighters[1].damage_percent

            # Step
            loop.step()
            frame += 1

            # Calculate reward with improved shaping
            reward = 0.0

            # Stock rewards (primary signal)
            if game.fighters[1].stocks < prev_p2_stocks:
                reward += 1.0  # Took opponent stock
                prev_stocks_taken += 1
            if game.fighters[0].stocks < prev_p1_stocks:
                reward -= 1.0  # Lost own stock

            # Damage dealt bonus (secondary signal)
            damage_dealt = game.fighters[1].damage_percent - prev_p2_damage
            if damage_dealt > 0:
                reward += damage_dealt * 0.002  # Increased from 0.001
                prev_damage_dealt += damage_dealt

            # Defensive reward shaping (for curriculum training vs aggressive)
            if use_defensive_rewards:
                damage_received = game.fighters[0].damage_percent - prev_p1_damage
                if damage_received > 0:
                    reward -= damage_received * 0.001  # Small penalty for taking damage
                    prev_damage_received += damage_received

                # Bonus for being in favorable position (opponent at high %)
                if game.fighters[1].damage_percent > 80:
                    reward += 0.002  # Slight bonus for pressuring high-% opponent

                # Bonus for surviving (not being at high %)
                if game.fighters[0].damage_percent < 50 and frame % 60 == 0:
                    reward += 0.001  # Small survival bonus

            # Win/lose bonus at end
            done = game.match_over or frame >= max_frames
            if done:
                if game.winner == 0:
                    reward += 0.5  # Win bonus
                elif game.winner == 1:
                    reward -= 0.5  # Loss penalty

            # Add transition
            episode.add(obs_p1, action_p1, reward, done, log_prob, value)

            if done:
                break

        stats = {
            'frames': frame,
            'winner': game.winner,
            'trained_won': game.winner == 0,
            'damage_dealt': prev_damage_dealt,
            'stocks_taken': prev_stocks_taken,
            'p1_final_stocks': game.fighters[0].stocks,
            'p2_final_stocks': game.fighters[1].stocks,
        }

        return episode, game.winner, stats

    def train_phase(
        self,
        phase_name: str,
        opponent_policy,
        num_games: int,
        use_defensive_rewards: bool = False,
        eval_interval: int = 100
    ) -> dict:
        """
        Run a training phase against a specific opponent.

        Args:
            phase_name: Name for logging
            opponent_policy: Opponent to train against
            num_games: Number of games to play
            use_defensive_rewards: Whether to use defensive reward shaping
            eval_interval: Evaluate every N games

        Returns:
            Phase statistics dict
        """
        print(f'\n{"="*60}')
        print(f'PHASE: {phase_name}')
        print(f'{"="*60}')
        print(f'Games: {num_games}')
        print(f'Defensive rewards: {use_defensive_rewards}')

        phase_wins = 0
        phase_episodes = []
        phase_start = time.time()

        for game_idx in range(num_games):
            # Collect episode
            episode, winner, stats = self.collect_episode(
                opponent_policy,
                use_defensive_rewards=use_defensive_rewards
            )

            if stats['trained_won']:
                phase_wins += 1

            phase_episodes.append(episode)
            self.games_played += 1

            # PPO update every N games
            if len(phase_episodes) >= self.config.games_per_iteration:
                train_stats = self.trainer.update(phase_episodes)
                phase_episodes = []

                # Progress report
                win_rate = phase_wins / (game_idx + 1)
                print(f'  Games {game_idx + 1}/{num_games}: '
                      f'Win rate: {win_rate:.1%}, '
                      f'Policy loss: {train_stats["policy_loss"]:.4f}, '
                      f'Entropy: {train_stats["entropy"]:.4f}')

            # Periodic evaluation
            if (game_idx + 1) % eval_interval == 0:
                print(f'\n  [Evaluation at {game_idx + 1} games]')
                vs_random = evaluate_vs_random(self.policy, num_games=20)
                vs_agg = evaluate_vs_aggressive(self.policy, num_games=20)
                print(f'    vs Random: {vs_random["win_rate"]:.1%}')
                print(f'    vs Aggressive: {vs_agg["win_rate"]:.1%}')

        # Final stats for phase
        elapsed = time.time() - phase_start
        phase_win_rate = phase_wins / num_games

        print(f'\n{phase_name} Complete:')
        print(f'  Win rate: {phase_win_rate:.1%}')
        print(f'  Time: {elapsed:.1f}s ({num_games / elapsed:.1f} games/sec)')

        return {
            'phase': phase_name,
            'games': num_games,
            'wins': phase_wins,
            'win_rate': phase_win_rate,
            'elapsed': elapsed
        }

    def run_curriculum(self, output_dir: str = 'checkpoints/curriculum'):
        """
        Run the full curriculum training.
        """
        os.makedirs(output_dir, exist_ok=True)

        print('\n' + '=' * 60)
        print('CURRICULUM TRAINING')
        print('=' * 60)

        start_time = time.time()
        all_phase_stats = []

        # Initial evaluation
        print('\nInitial Evaluation:')
        init_random = evaluate_vs_random(self.policy, num_games=50)
        init_agg = evaluate_vs_aggressive(self.policy, num_games=50)
        print(f'  vs Random: {init_random["win_rate"]:.1%}')
        print(f'  vs Aggressive: {init_agg["win_rate"]:.1%}')

        # Phase 1: Self-play (vs self/random mix)
        phase1 = self.train_phase(
            'Phase 1: Self-Play Foundations',
            self.random_policy,  # Start with random for exploration
            num_games=300,
            use_defensive_rewards=False
        )
        all_phase_stats.append(phase1)

        # Save Phase 1 checkpoint
        self.policy.save(f'{output_dir}/checkpoint_phase1.npz')
        print(f'Saved Phase 1 checkpoint')

        # Phase 2: Training vs Aggressive (with defensive rewards)
        phase2 = self.train_phase(
            'Phase 2: Anti-Aggro Training',
            self.aggressive_policy,
            num_games=500,
            use_defensive_rewards=True
        )
        all_phase_stats.append(phase2)

        # Save Phase 2 checkpoint
        self.policy.save(f'{output_dir}/checkpoint_phase2.npz')
        print(f'Saved Phase 2 checkpoint')

        # Phase 3: Mixed training (alternate opponents)
        print('\n' + '=' * 60)
        print('Phase 3: Mixed Training')
        print('=' * 60)

        phase3_wins = 0
        for i in range(200):
            # Alternate between random and aggressive
            opponent = self.aggressive_policy if i % 2 == 0 else self.random_policy
            use_def = i % 2 == 0

            episode, winner, stats = self.collect_episode(
                opponent, use_defensive_rewards=use_def
            )
            if stats['trained_won']:
                phase3_wins += 1

            # Update periodically
            if (i + 1) % 10 == 0:
                # Quick update with just this episode
                self.trainer.buffer.add_episode(episode)

            if (i + 1) % 50 == 0:
                print(f'  Games {i + 1}/200: Win rate {phase3_wins/(i+1):.1%}')

        phase3 = {
            'phase': 'Phase 3: Mixed Training',
            'games': 200,
            'wins': phase3_wins,
            'win_rate': phase3_wins / 200
        }
        all_phase_stats.append(phase3)

        # Save final checkpoint
        self.policy.save(f'{output_dir}/checkpoint_final.npz')

        # Final evaluation
        print('\n' + '=' * 60)
        print('FINAL EVALUATION')
        print('=' * 60)

        final_random = evaluate_vs_random(self.policy, num_games=50)
        final_agg = evaluate_vs_aggressive(self.policy, num_games=50)
        behavior = analyze_behavior(self.policy, num_games=20)

        print(f'\nvs Random: {final_random["win_rate"]:.1%}')
        print(f'vs Aggressive: {final_agg["win_rate"]:.1%}')
        print(f'\nBehavior:')
        print(f'  Playstyle: {behavior["playstyle"]}')
        print(f'  Attack ratio: {behavior["attack_ratio"]:.1%}')
        print(f'  Movement ratio: {behavior["movement_ratio"]:.1%}')
        print(f'  Idle ratio: {behavior["idle_ratio"]:.1%}')

        total_time = time.time() - start_time

        # Save report
        report = {
            'total_games': self.games_played,
            'total_time_seconds': total_time,
            'initial_vs_random': init_random['win_rate'],
            'initial_vs_aggressive': init_agg['win_rate'],
            'final_vs_random': final_random['win_rate'],
            'final_vs_aggressive': final_agg['win_rate'],
            'improvement_vs_aggressive': final_agg['win_rate'] - init_agg['win_rate'],
            'phases': all_phase_stats,
            'behavior': behavior
        }

        with open(f'{output_dir}/curriculum_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f'\n{"="*60}')
        print('CURRICULUM TRAINING COMPLETE')
        print(f'{"="*60}')
        print(f'Total games: {self.games_played}')
        print(f'Total time: {total_time / 60:.1f} minutes')
        print(f'Aggressive win rate: {init_agg["win_rate"]:.1%} -> {final_agg["win_rate"]:.1%}')
        print(f'Report saved to: {output_dir}/curriculum_report.json')

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Curriculum training for Stick Fighter')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Initial checkpoint to start from')
    parser.add_argument('--output', type=str, default='checkpoints/curriculum',
                        help='Output directory for checkpoints')

    args = parser.parse_args()

    trainer = CurriculumTrainer(
        checkpoint_path=args.checkpoint,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    trainer.run_curriculum(output_dir=args.output)


if __name__ == '__main__':
    main()
