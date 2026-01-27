#!/usr/bin/env python3
"""
Extended Anti-Aggressive Training - Cycle 4 Target: >50% vs AggressivePolicy

Key improvements from Cycle 3:
1. Extended training (3000 games total)
2. Stronger defensive reward shaping at high damage
3. Counter-aggression timing rewards
4. Population diversity with multiple reward profiles
5. Progressive curriculum with more aggressive focus

Strategy:
- Train longer to allow strategy development
- Heavily reward spacing and timing (counter-aggressive fundamentals)
- Train multiple policies with different emphases for diversity
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from copy import deepcopy

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


@dataclass
class ExtendedTrainingConfig:
    """Configuration for extended anti-aggressive training."""
    # Total training budget
    total_games: int = 3000

    # Phase breakdown (sum should equal total_games)
    warmup_games: int = 200       # vs Random to establish basics
    mixed_light_games: int = 400  # 50% aggressive
    mixed_heavy_games: int = 800  # 75% aggressive
    pure_aggressive_games: int = 1000  # 100% aggressive focus
    alternating_games: int = 600  # Alternating to prevent overfitting

    # Damage-state thresholds
    low_damage: float = 30.0
    mid_damage: float = 60.0
    high_damage: float = 90.0
    danger_damage: float = 120.0

    # Base reward values
    stock_taken: float = 1.5      # Increased from 1.0
    stock_lost: float = -1.5
    win_bonus: float = 0.5
    lose_penalty: float = -0.5

    # Damage-based rewards
    damage_dealt_base: float = 0.002
    damage_taken_low: float = -0.001
    damage_taken_mid: float = -0.002
    damage_taken_high: float = -0.004   # Much higher penalty at high damage
    damage_taken_danger: float = -0.006  # Even higher

    # Counter-aggression rewards
    punish_bonus: float = 0.005     # Bonus for hitting after opponent attacks
    spacing_bonus: float = 0.0005   # Bonus for maintaining safe distance

    # Aggression-state awareness
    opp_high_damage_bonus: float = 0.003  # Extra reward for pressing damaged opp

    # Training parameters
    entropy_coef: float = 0.015     # Slightly higher entropy
    learning_rate: float = 3e-4
    lr_decay: float = 0.95          # Decay LR as training progresses
    games_per_update: int = 10


class CounterAggressiveTrainer:
    """Trainer focused on developing counter-aggressive strategies."""

    def __init__(self, device: str = 'cuda', checkpoint_dir: str = 'checkpoints/extended_anti_agg'):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.config = ExtendedTrainingConfig()

        # Create policy
        self.policy = PolicyNetwork(
            obs_size=OBSERVATION_SIZE,
            action_size=ACTION_SPACE_SIZE,
            hidden_size=256
        )
        self.policy.to(device)

        # Create trainer with config
        train_config = TrainingConfig(
            learning_rate=self.config.learning_rate,
            batch_size=64,
            epochs_per_update=4,
            gamma=0.99,
            gae_lambda=0.95,
            entropy_coef=self.config.entropy_coef,
            value_coef=0.5
        )
        self.trainer = PPOTrainer(self.policy, train_config, device=device)

        # Opponents
        self.random_opp = RandomPolicy()
        self.aggressive_opp = AggressivePolicy()

        # Stats tracking
        self.games_played = 0
        self.phase_stats = []
        self.best_vs_aggressive = 0.0

        # Punish detection state
        self.opponent_last_attacked = False

    def compute_counter_aggressive_reward(self, game, player_idx: int,
                                           prev_damage: List[float],
                                           prev_stocks: List[int],
                                           action: int,
                                           opponent_attacked: bool) -> float:
        """
        Compute reward with counter-aggressive focus.

        Key insight: reward timing and spacing, not just damage dealing.
        """
        cfg = self.config
        opponent_idx = 1 - player_idx

        fighter = game.fighters[player_idx]
        opponent = game.fighters[opponent_idx]

        my_damage = prev_damage[player_idx]
        opp_damage = prev_damage[opponent_idx]

        reward = 0.0

        # Stock changes (highest priority)
        if opponent.stocks < prev_stocks[opponent_idx]:
            reward += cfg.stock_taken
        if fighter.stocks < prev_stocks[player_idx]:
            reward += cfg.stock_lost

        # Damage dealt
        damage_dealt = opponent.damage_percent - prev_damage[opponent_idx]
        if damage_dealt > 0:
            base_reward = damage_dealt * cfg.damage_dealt_base

            # Punish bonus: extra reward if opponent just attacked
            if opponent_attacked:
                base_reward += damage_dealt * cfg.punish_bonus

            # Press advantage: bonus for attacking damaged opponent
            if opp_damage >= cfg.high_damage:
                base_reward += damage_dealt * cfg.opp_high_damage_bonus

            reward += base_reward

        # Damage taken - scale penalty based on our damage state
        damage_taken = fighter.damage_percent - prev_damage[player_idx]
        if damage_taken > 0:
            if my_damage < cfg.low_damage:
                reward += damage_taken * cfg.damage_taken_low
            elif my_damage < cfg.mid_damage:
                reward += damage_taken * cfg.damage_taken_mid
            elif my_damage < cfg.high_damage:
                reward += damage_taken * cfg.damage_taken_high
            else:
                reward += damage_taken * cfg.damage_taken_danger

        # Spacing bonus: small reward for maintaining safe distance when at high damage
        if my_damage >= cfg.mid_damage and action in [1, 2]:  # Movement actions
            distance = abs(fighter.position.x - opponent.position.x) if hasattr(fighter, 'position') else 100
            if 100 < distance < 300:  # Good spacing zone
                reward += cfg.spacing_bonus

        return reward

    def detect_opponent_attack(self, game, opponent_idx: int, prev_state: dict) -> bool:
        """Detect if opponent just executed an attack."""
        opponent = game.fighters[opponent_idx]

        # Check if opponent is in an attack state
        if hasattr(opponent, 'current_state'):
            state = opponent.current_state
            if state in ['ATTACKING', 'SPECIAL']:
                return True

        # Fallback: check if opponent's damage output increased
        # This is a heuristic - opponent attacked if we took damage
        return False

    def run_game(self, opponent, character: str = 'FLEX') -> Tuple[Episode, int, dict]:
        """Run a single game with counter-aggressive rewards."""
        game = create_game(
            stage_name='battlefield',
            p1_character=character,
            p2_character=character,
            stock_count=3
        )
        loop = GameLoop(game, headless=True)

        episode = Episode(player_id=0)
        prev_damage = [0.0, 0.0]
        prev_stocks = [3, 3]

        # Behavior tracking
        action_counts = np.zeros(ACTION_SPACE_SIZE)
        damage_frames = {'low': 0, 'mid': 0, 'high': 0, 'danger': 0}
        punish_attempts = 0
        total_frames = 0

        frame = 0
        max_frames = 10800
        opponent_was_attacking = False

        while frame < max_frames:
            obs = [get_observation(game, i) for i in range(2)]

            # Policy plays P1
            action_p1, log_prob, value = self.policy.get_action(obs[0])
            action_p2, _, _ = opponent.get_action(obs[1])

            action_counts[action_p1] += 1
            total_frames += 1

            # Track damage state distribution
            my_damage = game.fighters[0].damage_percent
            if my_damage < self.config.low_damage:
                damage_frames['low'] += 1
            elif my_damage < self.config.mid_damage:
                damage_frames['mid'] += 1
            elif my_damage < self.config.high_damage:
                damage_frames['high'] += 1
            else:
                damage_frames['danger'] += 1

            # Track punish attempts (attack after opponent attacks)
            if opponent_was_attacking and action_p1 >= 5:  # Attack action
                punish_attempts += 1

            # Detect if opponent is currently attacking (for next frame's punish detection)
            opponent_was_attacking = action_p2 >= 5  # Opponent chose attack

            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            match_ongoing = loop.step()
            frame += 1

            # Compute reward
            reward = self.compute_counter_aggressive_reward(
                game, 0, prev_damage, prev_stocks, action_p1, opponent_was_attacking
            )

            prev_damage = [game.fighters[i].damage_percent for i in range(2)]
            prev_stocks = [game.fighters[i].stocks for i in range(2)]

            done = not match_ongoing or game.match_over

            episode.add(
                obs=obs[0],
                action=action_p1,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value
            )

            if done:
                break

        # Win/loss bonus
        winner = game.winner
        if winner == 0:
            episode.transitions[-1].reward += self.config.win_bonus
        elif winner == 1:
            episode.transitions[-1].reward -= self.config.lose_penalty

        self.games_played += 1

        # Game stats
        stats = {
            'winner': winner == 0,
            'frames': total_frames,
            'action_dist': (action_counts / max(1, total_frames)).tolist(),
            'damage_state_dist': {k: v/max(1, total_frames) for k, v in damage_frames.items()},
            'punish_attempts': punish_attempts
        }

        return episode, winner, stats

    def train_phase(self, name: str, opponent, num_games: int) -> float:
        """Train for one phase."""
        print(f"\n{'='*60}")
        print(f"PHASE: {name}")
        print(f"Games: {num_games}, Opponent: {type(opponent).__name__}")
        print(f"Total so far: {self.games_played}")
        print(f"{'='*60}")

        start_time = time.time()
        wins = 0
        episodes_buffer = []
        games_per_update = self.config.games_per_update

        for game_idx in range(num_games):
            episode, winner, _ = self.run_game(opponent)
            episodes_buffer.append(episode)
            if winner == 0:
                wins += 1

            if len(episodes_buffer) >= games_per_update:
                stats = self.trainer.update(episodes_buffer)
                episodes_buffer = []

                if (game_idx + 1) % 100 == 0:
                    win_rate = wins / (game_idx + 1)
                    print(f"  [{game_idx+1:4d}/{num_games}] "
                          f"WR: {win_rate:.1%} | "
                          f"Loss: {stats['policy_loss']:.4f} | "
                          f"Entropy: {stats['entropy']:.4f}")

        if episodes_buffer:
            self.trainer.update(episodes_buffer)

        elapsed = time.time() - start_time
        win_rate = wins / num_games

        print(f"\nPhase complete: {wins}/{num_games} wins ({win_rate:.1%})")
        print(f"Time: {elapsed:.1f}s ({num_games/elapsed:.1f} games/sec)")

        self.phase_stats.append({
            'phase': name,
            'games': num_games,
            'wins': wins,
            'win_rate': win_rate,
            'elapsed': elapsed
        })

        return win_rate

    def train_mixed_phase(self, name: str, num_games: int, aggressive_ratio: float) -> float:
        """Train with mixed opponents."""
        print(f"\n{'='*60}")
        print(f"PHASE: {name} ({aggressive_ratio:.0%} aggressive)")
        print(f"Games: {num_games}")
        print(f"{'='*60}")

        start_time = time.time()
        wins = 0
        episodes_buffer = []
        games_per_update = self.config.games_per_update

        for game_idx in range(num_games):
            if np.random.random() < aggressive_ratio:
                opponent = self.aggressive_opp
            else:
                opponent = self.random_opp

            episode, winner, _ = self.run_game(opponent)
            episodes_buffer.append(episode)
            if winner == 0:
                wins += 1

            if len(episodes_buffer) >= games_per_update:
                stats = self.trainer.update(episodes_buffer)
                episodes_buffer = []

                if (game_idx + 1) % 100 == 0:
                    win_rate = wins / (game_idx + 1)
                    print(f"  [{game_idx+1:4d}/{num_games}] "
                          f"WR: {win_rate:.1%} | "
                          f"Loss: {stats['policy_loss']:.4f}")

        if episodes_buffer:
            self.trainer.update(episodes_buffer)

        elapsed = time.time() - start_time
        win_rate = wins / num_games

        print(f"\nPhase complete: {wins}/{num_games} wins ({win_rate:.1%})")

        self.phase_stats.append({
            'phase': name,
            'games': num_games,
            'wins': wins,
            'win_rate': win_rate,
            'elapsed': elapsed,
            'aggressive_ratio': aggressive_ratio
        })

        return win_rate

    def evaluate_comprehensive(self, num_games: int = 50) -> Dict:
        """Comprehensive evaluation against all baselines."""
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION")
        print("="*60)

        vs_random = evaluate_vs_random(self.policy, num_games=num_games)
        print(f"  vs Random: {vs_random['win_rate']:.1%} ({vs_random['wins']}/{num_games})")

        vs_aggressive = evaluate_vs_aggressive(self.policy, num_games=num_games)
        print(f"  vs Aggressive: {vs_aggressive['win_rate']:.1%} ({vs_aggressive['wins']}/{num_games})")

        # Track best
        if vs_aggressive['win_rate'] > self.best_vs_aggressive:
            self.best_vs_aggressive = vs_aggressive['win_rate']
            # Save best checkpoint
            self.policy.save(f'{self.checkpoint_dir}/checkpoint_best.npz')
            print(f"  New best vs aggressive! Saved checkpoint.")

        return {
            'vs_random': vs_random,
            'vs_aggressive': vs_aggressive
        }

    def analyze_damage_state_behavior(self, num_games: int = 30) -> Dict:
        """Analyze behavior at different damage states."""
        print("\n--- Damage State Behavior Analysis ---")

        damage_actions = {
            'low': np.zeros(ACTION_SPACE_SIZE),
            'mid': np.zeros(ACTION_SPACE_SIZE),
            'high': np.zeros(ACTION_SPACE_SIZE),
            'danger': np.zeros(ACTION_SPACE_SIZE)
        }
        damage_frames = {'low': 0, 'mid': 0, 'high': 0, 'danger': 0}

        for _ in range(num_games):
            game = create_game('battlefield', 'FLEX', 'FLEX', stock_count=3)
            loop = GameLoop(game, headless=True)

            frame = 0
            while frame < 10800 and not game.match_over:
                obs = get_observation(game, 0)
                action, _, _ = self.policy.get_action(obs)

                damage = game.fighters[0].damage_percent

                if damage < self.config.low_damage:
                    key = 'low'
                elif damage < self.config.mid_damage:
                    key = 'mid'
                elif damage < self.config.high_damage:
                    key = 'high'
                else:
                    key = 'danger'

                damage_actions[key][action] += 1
                damage_frames[key] += 1

                # Random opponent
                obs_opp = get_observation(game, 1)
                action_opp = np.random.randint(0, ACTION_SPACE_SIZE)

                game.fighters[0].input = get_action_from_index(action)
                game.fighters[1].input = get_action_from_index(action_opp)

                loop.step()
                frame += 1

        # Normalize and compute attack ratios
        attack_indices = [5, 6, 7, 8, 9]
        move_indices = [1, 2]

        result = {}
        for key in damage_actions:
            frames = damage_frames[key]
            if frames > 0:
                dist = damage_actions[key] / frames
                attack_ratio = sum(dist[i] for i in attack_indices)
                move_ratio = sum(dist[i] for i in move_indices)

                result[key] = {
                    'frames': frames,
                    'attack_ratio': float(attack_ratio),
                    'move_ratio': float(move_ratio),
                    'action_dist': dist.tolist()
                }

                print(f"  {key:6s}: Attack={attack_ratio:.1%}, Move={move_ratio:.1%} ({frames} frames)")

        # Check for adaptation
        shows_adaptation = False
        if 'low' in result and 'high' in result:
            if result['high']['attack_ratio'] < result['low']['attack_ratio']:
                shows_adaptation = True
                print("  -> Shows defensive adaptation at high damage!")
            if result['high']['move_ratio'] > result['low']['move_ratio']:
                shows_adaptation = True
                print("  -> Increases movement at high damage!")

        result['shows_adaptation'] = shows_adaptation
        return result

    def run_extended_curriculum(self):
        """Run the full 3000-game extended curriculum."""
        cfg = self.config

        print("\n" + "="*60)
        print("EXTENDED ANTI-AGGRESSIVE CURRICULUM TRAINING")
        print(f"Total games planned: {cfg.total_games}")
        print("="*60)

        start_time = time.time()

        # Phase 1: Random warmup (establish basics)
        self.train_phase("Random Warmup", self.random_opp, cfg.warmup_games)
        self.policy.save(f'{self.checkpoint_dir}/checkpoint_warmup.npz')

        # Quick eval
        print("\n--- After Warmup ---")
        self.evaluate_comprehensive(num_games=30)

        # Phase 2: Light mixed (introduce aggressive)
        self.train_mixed_phase("Mixed Light", cfg.mixed_light_games, aggressive_ratio=0.5)
        self.policy.save(f'{self.checkpoint_dir}/checkpoint_mixed_light.npz')

        # Eval
        print("\n--- After Light Mixed ---")
        self.evaluate_comprehensive(num_games=30)

        # Phase 3: Heavy mixed (mostly aggressive)
        self.train_mixed_phase("Mixed Heavy", cfg.mixed_heavy_games, aggressive_ratio=0.75)
        self.policy.save(f'{self.checkpoint_dir}/checkpoint_mixed_heavy.npz')

        # Eval
        print("\n--- After Heavy Mixed ---")
        mid_eval = self.evaluate_comprehensive(num_games=40)

        # Check if we're on track
        vs_agg_rate = mid_eval['vs_aggressive']['win_rate']
        if vs_agg_rate < 0.35:
            print("\n[!] Below 35% vs aggressive - extending heavy aggressive training")
            self.train_phase("Extra Aggressive Focus", self.aggressive_opp, 300)

        # Phase 4: Pure aggressive focus
        self.train_phase("Pure Aggressive", self.aggressive_opp, cfg.pure_aggressive_games)
        self.policy.save(f'{self.checkpoint_dir}/checkpoint_pure_agg.npz')

        # Eval
        print("\n--- After Pure Aggressive ---")
        self.evaluate_comprehensive(num_games=40)

        # Phase 5: Alternating (prevent overfitting)
        print("\n--- Alternating Phase (prevent overfitting) ---")
        for i in range(6):
            self.train_phase(f"Aggressive Block {i+1}", self.aggressive_opp, 60)
            self.train_phase(f"Random Refresh {i+1}", self.random_opp, 40)

            # Check progress
            if (i + 1) % 2 == 0:
                check_eval = self.evaluate_comprehensive(num_games=30)
                if check_eval['vs_aggressive']['win_rate'] >= 0.50:
                    print("\n TARGET ACHIEVED! Stopping early.")
                    break

        # Final checkpoint
        final_path = f'{self.checkpoint_dir}/checkpoint_final.npz'
        self.policy.save(final_path)

        # Final comprehensive evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)

        final_eval = self.evaluate_comprehensive(num_games=60)

        # Damage state analysis
        damage_behavior = self.analyze_damage_state_behavior(num_games=30)

        total_elapsed = time.time() - start_time

        # Generate report
        report = {
            'training': {
                'total_games': self.games_played,
                'total_time_seconds': total_elapsed,
                'games_per_second': self.games_played / total_elapsed,
                'phases': self.phase_stats,
                'best_vs_aggressive': self.best_vs_aggressive
            },
            'final_evaluation': {
                'vs_random': final_eval['vs_random'],
                'vs_aggressive': final_eval['vs_aggressive']
            },
            'damage_state_behavior': damage_behavior,
            'config': {
                'total_games': cfg.total_games,
                'entropy_coef': cfg.entropy_coef,
                'high_damage_penalty': cfg.damage_taken_high,
                'danger_damage_penalty': cfg.damage_taken_danger,
                'punish_bonus': cfg.punish_bonus
            },
            'checkpoints': {
                'final': final_path,
                'best': f'{self.checkpoint_dir}/checkpoint_best.npz'
            }
        }

        report_path = f'{self.checkpoint_dir}/extended_training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_path}")

        # Summary
        vs_rand = final_eval['vs_random']['win_rate']
        vs_agg = final_eval['vs_aggressive']['win_rate']

        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Total training games: {self.games_played}")
        print(f"Total time: {total_elapsed/60:.1f} minutes")
        print(f"Win rate vs Random: {vs_rand:.1%} (target: >80%)")
        print(f"Win rate vs Aggressive: {vs_agg:.1%} (target: >50%)")
        print(f"Best vs Aggressive achieved: {self.best_vs_aggressive:.1%}")
        print(f"Shows damage adaptation: {damage_behavior.get('shows_adaptation', False)}")

        if vs_agg >= 0.50:
            print("\n SUCCESS! >50% vs Aggressive achieved!")
        else:
            print(f"\n Target not met. Gap: {0.50 - vs_agg:.1%}")

        return final_eval


def main():
    print("="*60)
    print("STICK FIGURE FIGHTER - EXTENDED ANTI-AGGRESSIVE TRAINING")
    print("Cycle 4 Target: >50% win rate vs AggressivePolicy")
    print("="*60)

    trainer = CounterAggressiveTrainer(
        device='cuda',
        checkpoint_dir='checkpoints/extended_anti_agg'
    )

    final_eval = trainer.run_extended_curriculum()

    return final_eval


if __name__ == '__main__':
    main()
