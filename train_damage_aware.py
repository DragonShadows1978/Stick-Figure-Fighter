#!/usr/bin/env python3
"""
Damage-State-Aware Curriculum Training for Stick Figure Fighter.

Key improvements:
1. Damage-aware reward shaping:
   - At high damage (>80%): defensive rewards (staying grounded, avoiding risk)
   - At low damage vs damaged opponent: offensive rewards (pressing advantage)
2. Extended training (2000+ games)
3. Population diversity with different strategy focuses
4. Progressive difficulty curriculum
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
class DamageAwareConfig:
    """Configuration for damage-aware training."""
    # Base rewards
    stock_taken: float = 1.0
    stock_lost: float = -1.0
    damage_dealt_base: float = 0.002
    damage_taken_base: float = -0.001
    win_bonus: float = 0.5
    lose_penalty: float = -0.5

    # Damage-state modifiers
    # When at high damage: reduce aggression reward, increase survival
    high_damage_threshold: float = 80.0
    high_damage_taken_penalty: float = -0.003  # Extra penalty when already damaged
    high_damage_defensive_bonus: float = 0.0005  # Bonus for staying grounded

    # When opponent is at high damage: increase aggression
    opp_high_damage_threshold: float = 60.0
    opp_damaged_attack_bonus: float = 0.003  # Extra for attacking damaged opponent

    # Edge-guarding bonus
    edge_zone_bonus: float = 0.001  # Bonus for attacking near edges

    # Training phases
    phase_1_games: int = 400    # Random warmup
    phase_2_games: int = 600    # Mixed (heavier aggressive)
    phase_3_games: int = 600    # Pure aggressive focus
    phase_4_games: int = 400    # Population training

    # PPO settings
    entropy_coef: float = 0.02  # Higher entropy for exploration
    learning_rate: float = 3e-4


class DamageAwareTrainer:
    """Trainer with damage-state-aware reward shaping."""

    def __init__(self, device: str = 'cuda', checkpoint_dir: str = 'checkpoints/damage_aware'):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.config = DamageAwareConfig()

        # Create policy
        self.policy = PolicyNetwork(
            obs_size=OBSERVATION_SIZE,
            action_size=ACTION_SPACE_SIZE,
            hidden_size=256
        )
        self.policy.to(device)

        # Create trainer
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

        # Stats
        self.games_played = 0
        self.phase_stats = []
        self.behavior_log = []

    def compute_damage_aware_reward(self, game, player_idx: int,
                                     prev_damage: List[float],
                                     prev_stocks: List[int],
                                     action: int) -> float:
        """
        Compute reward with damage-state awareness.

        The key insight: reward function should adapt based on game state.
        - At high damage: prioritize survival over damage dealing
        - Against damaged opponent: press the advantage
        """
        cfg = self.config
        opponent_idx = 1 - player_idx

        fighter = game.fighters[player_idx]
        opponent = game.fighters[opponent_idx]

        reward = 0.0

        # Stock changes (highest priority)
        if opponent.stocks < prev_stocks[opponent_idx]:
            reward += cfg.stock_taken
        if fighter.stocks < prev_stocks[player_idx]:
            reward += cfg.stock_lost

        # Base damage rewards
        damage_dealt = opponent.damage_percent - prev_damage[opponent_idx]
        damage_taken = fighter.damage_percent - prev_damage[player_idx]

        if damage_dealt > 0:
            base_dmg_reward = damage_dealt * cfg.damage_dealt_base

            # Bonus for attacking a damaged opponent
            if prev_damage[opponent_idx] >= cfg.opp_high_damage_threshold:
                base_dmg_reward += damage_dealt * cfg.opp_damaged_attack_bonus

            # Edge zone bonus (opponent near edge)
            stage_center = 400  # Approximate stage center
            opp_x = opponent.position.x if hasattr(opponent, 'position') else 400
            if abs(opp_x - stage_center) > 250:  # Near edge
                base_dmg_reward += damage_dealt * cfg.edge_zone_bonus

            reward += base_dmg_reward

        # Damage taken penalties - scale based on our damage state
        if damage_taken > 0:
            if prev_damage[player_idx] >= cfg.high_damage_threshold:
                # Extra penalty when already at high damage
                reward += damage_taken * cfg.high_damage_taken_penalty
            else:
                reward += damage_taken * cfg.damage_taken_base

        # Survival bonus at high damage
        if prev_damage[player_idx] >= cfg.high_damage_threshold:
            # Small bonus for staying grounded (safer)
            if hasattr(fighter, 'grounded') and fighter.grounded:
                reward += cfg.high_damage_defensive_bonus

        return reward

    def run_game(self, opponent, character: str = 'FLEX') -> tuple:
        """Run a single game with damage-aware rewards."""
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

        # Track behavior for analysis
        action_counts = np.zeros(ACTION_SPACE_SIZE)
        high_damage_frames = 0
        total_frames = 0

        frame = 0
        max_frames = 10800

        while frame < max_frames:
            obs = [get_observation(game, i) for i in range(2)]

            # Policy plays P1
            action_p1, log_prob, value = self.policy.get_action(obs[0])
            action_p2, _, _ = opponent.get_action(obs[1])

            action_counts[action_p1] += 1
            total_frames += 1

            if game.fighters[0].damage_percent >= self.config.high_damage_threshold:
                high_damage_frames += 1

            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            match_ongoing = loop.step()
            frame += 1

            # Compute damage-aware reward
            reward = self.compute_damage_aware_reward(
                game, 0, prev_damage, prev_stocks, action_p1
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

        # Log behavior
        behavior = {
            'game': self.games_played,
            'winner': winner == 0,
            'frames': total_frames,
            'high_damage_ratio': high_damage_frames / max(1, total_frames),
            'action_dist': (action_counts / max(1, total_frames)).tolist()
        }
        self.behavior_log.append(behavior)

        return episode, winner

    def train_phase(self, name: str, opponent, num_games: int,
                    games_per_update: int = 10) -> float:
        """Train for one curriculum phase."""
        print(f"\n{'='*60}")
        print(f"PHASE: {name}")
        print(f"Games: {num_games}, Opponent: {type(opponent).__name__}")
        print(f"{'='*60}")

        start_time = time.time()
        wins = 0
        episodes_buffer = []

        for game_idx in range(num_games):
            episode, winner = self.run_game(opponent)
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

    def train_mixed_phase(self, name: str, num_games: int,
                          aggressive_ratio: float = 0.7) -> float:
        """Train with mixed opponents (weighted toward aggressive)."""
        print(f"\n{'='*60}")
        print(f"PHASE: {name} (Mixed: {aggressive_ratio:.0%} aggressive)")
        print(f"Games: {num_games}")
        print(f"{'='*60}")

        start_time = time.time()
        wins = 0
        episodes_buffer = []
        games_per_update = 10

        for game_idx in range(num_games):
            # Select opponent based on ratio
            if np.random.random() < aggressive_ratio:
                opponent = self.aggressive_opp
            else:
                opponent = self.random_opp

            episode, winner = self.run_game(opponent)
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

        self.phase_stats.append({
            'phase': name,
            'games': num_games,
            'wins': wins,
            'win_rate': win_rate,
            'elapsed': elapsed,
            'aggressive_ratio': aggressive_ratio
        })

        return win_rate

    def evaluate_all(self, num_games: int = 50) -> Dict:
        """Comprehensive evaluation."""
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION")
        print("="*60)

        vs_random = evaluate_vs_random(self.policy, num_games=num_games)
        print(f"  vs Random: {vs_random['win_rate']:.1%} ({vs_random['wins']}/{num_games})")

        vs_aggressive = evaluate_vs_aggressive(self.policy, num_games=num_games)
        print(f"  vs Aggressive: {vs_aggressive['win_rate']:.1%} ({vs_aggressive['wins']}/{num_games})")

        return {
            'vs_random': vs_random,
            'vs_aggressive': vs_aggressive
        }

    def analyze_damage_adaptation(self, num_games: int = 30) -> Dict:
        """Analyze how agent behaves at different damage levels."""
        print("\n--- Analyzing Damage State Adaptation ---")

        # Track actions at different damage levels
        low_damage_actions = np.zeros(ACTION_SPACE_SIZE)   # <40%
        mid_damage_actions = np.zeros(ACTION_SPACE_SIZE)   # 40-80%
        high_damage_actions = np.zeros(ACTION_SPACE_SIZE)  # >80%

        low_frames = mid_frames = high_frames = 0

        for _ in range(num_games):
            game = create_game('battlefield', 'FLEX', 'FLEX', stock_count=3)
            loop = GameLoop(game, headless=True)

            frame = 0
            while frame < 10800 and not game.match_over:
                obs = get_observation(game, 0)
                action, _, _ = self.policy.get_action(obs)

                damage = game.fighters[0].damage_percent

                if damage < 40:
                    low_damage_actions[action] += 1
                    low_frames += 1
                elif damage < 80:
                    mid_damage_actions[action] += 1
                    mid_frames += 1
                else:
                    high_damage_actions[action] += 1
                    high_frames += 1

                # Random opponent
                obs_opp = get_observation(game, 1)
                action_opp = np.random.randint(0, ACTION_SPACE_SIZE)

                game.fighters[0].input = get_action_from_index(action)
                game.fighters[1].input = get_action_from_index(action_opp)

                loop.step()
                frame += 1

        # Normalize
        if low_frames > 0:
            low_damage_actions /= low_frames
        if mid_frames > 0:
            mid_damage_actions /= mid_frames
        if high_frames > 0:
            high_damage_actions /= high_frames

        # Compute attack ratios
        attack_indices = [5, 6, 7, 8, 9]  # Attack, L+Atk, R+Atk, U+Atk, D+Atk

        low_attack = sum(low_damage_actions[i] for i in attack_indices)
        mid_attack = sum(mid_damage_actions[i] for i in attack_indices)
        high_attack = sum(high_damage_actions[i] for i in attack_indices)

        print(f"\nAttack Ratio by Damage State:")
        print(f"  Low (<40%):   {low_attack:.1%} ({low_frames} frames)")
        print(f"  Mid (40-80%): {mid_attack:.1%} ({mid_frames} frames)")
        print(f"  High (>80%):  {high_attack:.1%} ({high_frames} frames)")

        # Check for adaptation
        adaptation_score = 0
        if high_attack < mid_attack:  # Should be more defensive at high damage
            adaptation_score += 1
            print("  -> Shows defensive adaptation at high damage")
        if mid_attack <= low_attack:  # Should maintain or increase aggression
            adaptation_score += 1
            print("  -> Maintains aggression at mid damage")

        return {
            'low_damage': {
                'attack_ratio': float(low_attack),
                'frames': low_frames,
                'action_dist': low_damage_actions.tolist()
            },
            'mid_damage': {
                'attack_ratio': float(mid_attack),
                'frames': mid_frames,
                'action_dist': mid_damage_actions.tolist()
            },
            'high_damage': {
                'attack_ratio': float(high_attack),
                'frames': high_frames,
                'action_dist': high_damage_actions.tolist()
            },
            'shows_adaptation': adaptation_score >= 1
        }

    def run_extended_curriculum(self):
        """Run the full 2000+ game curriculum with damage awareness."""
        print("\n" + "="*60)
        print("DAMAGE-AWARE EXTENDED CURRICULUM TRAINING")
        print("Total planned games: 2000")
        print("="*60)

        cfg = self.config

        # Phase 1: Random warmup (establish baseline skills)
        self.train_phase("Random Warmup", self.random_opp, cfg.phase_1_games)
        self.policy.save(f'{self.checkpoint_dir}/checkpoint_phase1.npz')

        # Mid-curriculum eval
        print("\n--- Mid-Curriculum Evaluation (after warmup) ---")
        self.evaluate_all(num_games=30)

        # Phase 2: Mixed training with heavy aggressive bias
        self.train_mixed_phase("Mixed Heavy Aggressive", cfg.phase_2_games, aggressive_ratio=0.75)
        self.policy.save(f'{self.checkpoint_dir}/checkpoint_phase2.npz')

        # Mid eval
        print("\n--- Mid-Curriculum Evaluation (after mixed) ---")
        mid_eval = self.evaluate_all(num_games=30)

        # Phase 3: Pure aggressive focus
        self.train_phase("Aggressive Focus", self.aggressive_opp, cfg.phase_3_games)
        self.policy.save(f'{self.checkpoint_dir}/checkpoint_phase3.npz')

        # Phase 4: Alternating to prevent overfitting
        print("\n--- Phase 4: Alternating Training ---")
        for i in range(4):
            self.train_phase(f"Aggressive Round {i+1}", self.aggressive_opp, 50)
            self.train_phase(f"Random Refresh {i+1}", self.random_opp, 50)

        # Final checkpoint
        final_path = f'{self.checkpoint_dir}/checkpoint_final.npz'
        self.policy.save(final_path)

        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)

        final_eval = self.evaluate_all(num_games=50)

        # Damage adaptation analysis
        adaptation = self.analyze_damage_adaptation(num_games=30)

        # Generate report
        report = {
            'training': {
                'total_games': self.games_played,
                'phases': self.phase_stats,
                'config': {
                    'high_damage_threshold': cfg.high_damage_threshold,
                    'opp_high_damage_threshold': cfg.opp_high_damage_threshold,
                    'damage_dealt_base': cfg.damage_dealt_base,
                    'damage_taken_base': cfg.damage_taken_base,
                    'high_damage_taken_penalty': cfg.high_damage_taken_penalty
                }
            },
            'final_evaluation': final_eval,
            'damage_adaptation': adaptation,
            'checkpoint': final_path
        }

        report_path = f'{self.checkpoint_dir}/damage_aware_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_path}")

        # Success check
        vs_agg = final_eval['vs_aggressive']['win_rate']
        vs_rand = final_eval['vs_random']['win_rate']

        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Win rate vs Random: {vs_rand:.1%} (target: >80%)")
        print(f"Win rate vs Aggressive: {vs_agg:.1%} (target: >50%)")
        print(f"Shows damage adaptation: {adaptation['shows_adaptation']}")

        if vs_agg >= 0.50:
            print("\n SUCCESS! >50% vs Aggressive achieved!")
        else:
            print(f"\n Target not met yet. Gap: {0.50 - vs_agg:.1%}")

        return final_eval


def main():
    print("="*60)
    print("STICK FIGURE FIGHTER - DAMAGE-AWARE TRAINING")
    print("Goal: >50% win rate vs AggressivePolicy")
    print("="*60)

    trainer = DamageAwareTrainer(
        device='cuda',
        checkpoint_dir='checkpoints/damage_aware'
    )

    final_eval = trainer.run_extended_curriculum()

    return final_eval


if __name__ == '__main__':
    main()
