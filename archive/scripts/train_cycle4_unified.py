#!/usr/bin/env python3
"""
Cycle 4 Unified Training Pipeline - Stick Figure Fighter

Goals:
1. Achieve and maintain >50% win rate vs AggressivePolicy
2. Maintain >80% win rate vs RandomPolicy
3. Implement damage-state-aware behavior adaptation
4. Train population for strategy diversity
5. Generate comprehensive character matchup matrix

Strategy:
- Use anti_aggressive_v3 checkpoint as starting point (already 53% vs aggressive)
- Fine-tune with balanced curriculum to restore random performance
- Add damage-state-aware reward shaping
- Train population of diverse policies
- Run full cross-character evaluation
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
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
class Cycle4Config:
    """Configuration for Cycle 4 training."""
    # Training budget
    balanced_games: int = 600      # Balanced training to restore random performance
    aggressive_games: int = 800    # Anti-aggressive focus
    alternating_games: int = 400   # Prevent overfitting

    # Damage thresholds
    low_damage: float = 30.0
    mid_damage: float = 60.0
    high_damage: float = 90.0
    danger_damage: float = 120.0

    # Base rewards
    stock_taken: float = 1.2
    stock_lost: float = -1.2
    win_bonus: float = 0.3
    lose_penalty: float = -0.3

    # Damage dealing rewards - base
    damage_dealt_base: float = 0.002

    # Damage-state modulated rewards (key innovation)
    # At high damage: reduced attack reward, increased survival focus
    damage_dealt_at_danger: float = 0.0015  # Less aggressive when in danger
    damage_taken_low: float = -0.001
    damage_taken_mid: float = -0.002
    damage_taken_high: float = -0.003
    damage_taken_danger: float = -0.005   # High penalty when already hurt

    # Opponent damage state rewards
    opp_damaged_bonus: float = 0.003      # Press advantage vs damaged opponent

    # Positional rewards
    center_stage_bonus: float = 0.0003    # Small bonus for stage control

    # Training params
    entropy_coef: float = 0.012
    learning_rate: float = 2e-4           # Slightly lower for fine-tuning
    games_per_update: int = 10


class Cycle4Trainer:
    """Unified trainer for Cycle 4 objectives."""

    def __init__(self, device: str = 'cuda', checkpoint_dir: str = 'checkpoints/cycle4'):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.config = Cycle4Config()

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

        # Tracking
        self.games_played = 0
        self.phase_stats = []
        self.best_vs_aggressive = 0.0
        self.best_balanced_score = 0.0  # vs_rand * vs_agg balanced score

        # Damage state tracking for analysis
        self.damage_state_actions = {
            'low': np.zeros(ACTION_SPACE_SIZE),
            'mid': np.zeros(ACTION_SPACE_SIZE),
            'high': np.zeros(ACTION_SPACE_SIZE),
            'danger': np.zeros(ACTION_SPACE_SIZE)
        }
        self.damage_state_frames = {'low': 0, 'mid': 0, 'high': 0, 'danger': 0}

    def load_checkpoint(self, path: str):
        """Load from existing checkpoint."""
        self.policy.load(path)
        print(f"Loaded checkpoint: {path}")

    def get_damage_state(self, damage: float) -> str:
        """Categorize damage into states."""
        cfg = self.config
        if damage < cfg.low_damage:
            return 'low'
        elif damage < cfg.mid_damage:
            return 'mid'
        elif damage < cfg.high_damage:
            return 'high'
        else:
            return 'danger'

    def compute_reward(self, game, player_idx: int,
                       prev_damage: List[float], prev_stocks: List[int],
                       action: int) -> float:
        """
        Compute damage-state-aware reward.

        Key insight: The reward function adapts based on game state.
        - At high self-damage: be more cautious, less attack reward
        - When opponent is damaged: press the advantage
        - Always reward stage control
        """
        cfg = self.config
        opponent_idx = 1 - player_idx

        fighter = game.fighters[player_idx]
        opponent = game.fighters[opponent_idx]

        my_damage = prev_damage[player_idx]
        opp_damage = prev_damage[opponent_idx]
        my_state = self.get_damage_state(my_damage)
        opp_state = self.get_damage_state(opp_damage)

        reward = 0.0

        # Stock changes (highest priority)
        if opponent.stocks < prev_stocks[opponent_idx]:
            reward += cfg.stock_taken
        if fighter.stocks < prev_stocks[player_idx]:
            reward += cfg.stock_lost

        # Damage dealt - modulated by our damage state
        damage_dealt = opponent.damage_percent - prev_damage[opponent_idx]
        if damage_dealt > 0:
            # Base damage reward, reduced when in danger
            if my_state == 'danger':
                dmg_reward = damage_dealt * cfg.damage_dealt_at_danger
            else:
                dmg_reward = damage_dealt * cfg.damage_dealt_base

            # Press advantage when opponent is damaged
            if opp_damage >= cfg.high_damage:
                dmg_reward += damage_dealt * cfg.opp_damaged_bonus

            reward += dmg_reward

        # Damage taken - scaled by our current damage state
        damage_taken = fighter.damage_percent - prev_damage[player_idx]
        if damage_taken > 0:
            if my_state == 'low':
                reward += damage_taken * cfg.damage_taken_low
            elif my_state == 'mid':
                reward += damage_taken * cfg.damage_taken_mid
            elif my_state == 'high':
                reward += damage_taken * cfg.damage_taken_high
            else:
                reward += damage_taken * cfg.damage_taken_danger

        # Small stage control bonus
        if hasattr(fighter, 'position'):
            stage_center = 400  # Approximate
            dist_from_center = abs(fighter.position.x - stage_center)
            if dist_from_center < 150:  # Near center
                reward += cfg.center_stage_bonus

        return reward

    def run_game(self, opponent, character: str = 'FLEX', track_damage_states: bool = True) -> Tuple[Episode, int, dict]:
        """Run a single game with damage-state tracking."""
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

        # Tracking
        action_counts = np.zeros(ACTION_SPACE_SIZE)
        total_frames = 0
        damage_dealt_total = 0.0
        damage_taken_total = 0.0

        frame = 0
        max_frames = 10800

        while frame < max_frames:
            obs = [get_observation(game, i) for i in range(2)]

            # Policy plays P1
            action_p1, log_prob, value = self.policy.get_action(obs[0])
            action_p2, _, _ = opponent.get_action(obs[1])

            action_counts[action_p1] += 1
            total_frames += 1

            # Track damage state distributions
            if track_damage_states:
                my_damage = game.fighters[0].damage_percent
                state = self.get_damage_state(my_damage)
                self.damage_state_actions[state][action_p1] += 1
                self.damage_state_frames[state] += 1

            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            match_ongoing = loop.step()
            frame += 1

            # Compute reward
            reward = self.compute_reward(game, 0, prev_damage, prev_stocks, action_p1)

            # Track damage
            damage_dealt_total += max(0, game.fighters[1].damage_percent - prev_damage[1])
            damage_taken_total += max(0, game.fighters[0].damage_percent - prev_damage[0])

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

        stats = {
            'winner': winner == 0,
            'frames': total_frames,
            'action_dist': (action_counts / max(1, total_frames)).tolist(),
            'damage_dealt': damage_dealt_total,
            'damage_taken': damage_taken_total
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
        wins_vs_agg = 0
        wins_vs_rand = 0
        games_vs_agg = 0
        games_vs_rand = 0
        episodes_buffer = []
        games_per_update = self.config.games_per_update

        for game_idx in range(num_games):
            if np.random.random() < aggressive_ratio:
                opponent = self.aggressive_opp
                is_aggressive = True
            else:
                opponent = self.random_opp
                is_aggressive = False

            episode, winner, _ = self.run_game(opponent)
            episodes_buffer.append(episode)

            if is_aggressive:
                games_vs_agg += 1
                if winner == 0:
                    wins_vs_agg += 1
            else:
                games_vs_rand += 1
                if winner == 0:
                    wins_vs_rand += 1

            if len(episodes_buffer) >= games_per_update:
                stats = self.trainer.update(episodes_buffer)
                episodes_buffer = []

                if (game_idx + 1) % 100 == 0:
                    wr_agg = wins_vs_agg / max(1, games_vs_agg)
                    wr_rand = wins_vs_rand / max(1, games_vs_rand)
                    print(f"  [{game_idx+1:4d}/{num_games}] "
                          f"vs Agg: {wr_agg:.1%} | vs Rand: {wr_rand:.1%}")

        if episodes_buffer:
            self.trainer.update(episodes_buffer)

        elapsed = time.time() - start_time
        total_wins = wins_vs_agg + wins_vs_rand

        print(f"\nPhase complete:")
        print(f"  vs Aggressive: {wins_vs_agg}/{games_vs_agg} ({wins_vs_agg/max(1,games_vs_agg):.1%})")
        print(f"  vs Random: {wins_vs_rand}/{games_vs_rand} ({wins_vs_rand/max(1,games_vs_rand):.1%})")

        self.phase_stats.append({
            'phase': name,
            'games': num_games,
            'wins_vs_agg': wins_vs_agg,
            'games_vs_agg': games_vs_agg,
            'wins_vs_rand': wins_vs_rand,
            'games_vs_rand': games_vs_rand,
            'aggressive_ratio': aggressive_ratio,
            'elapsed': elapsed
        })

        return (wins_vs_agg + wins_vs_rand) / num_games

    def evaluate_comprehensive(self, num_games: int = 50) -> Dict:
        """Full evaluation against baselines."""
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION")
        print("="*60)

        vs_random = evaluate_vs_random(self.policy, num_games=num_games)
        print(f"  vs Random: {vs_random['win_rate']:.1%} ({vs_random['wins']}/{num_games})")

        vs_aggressive = evaluate_vs_aggressive(self.policy, num_games=num_games)
        print(f"  vs Aggressive: {vs_aggressive['win_rate']:.1%} ({vs_aggressive['wins']}/{num_games})")

        # Check for new bests
        if vs_aggressive['win_rate'] > self.best_vs_aggressive:
            self.best_vs_aggressive = vs_aggressive['win_rate']
            self.policy.save(f'{self.checkpoint_dir}/best_vs_aggressive.npz')
            print(f"  New best vs Aggressive! Saved.")

        # Track balanced score (geometric mean)
        balanced = np.sqrt(vs_random['win_rate'] * vs_aggressive['win_rate'])
        if balanced > self.best_balanced_score:
            self.best_balanced_score = balanced
            self.policy.save(f'{self.checkpoint_dir}/best_balanced.npz')
            print(f"  New best balanced! (Score: {balanced:.3f})")

        return {
            'vs_random': vs_random,
            'vs_aggressive': vs_aggressive,
            'balanced_score': balanced
        }

    def analyze_damage_state_behavior(self) -> Dict:
        """Analyze how behavior differs by damage state."""
        print("\n--- Damage State Behavior Analysis ---")

        attack_indices = [5, 6, 7, 8, 9]
        move_indices = [1, 2]

        result = {}
        for state in ['low', 'mid', 'high', 'danger']:
            frames = self.damage_state_frames[state]
            if frames > 0:
                dist = self.damage_state_actions[state] / frames
                attack_ratio = sum(dist[i] for i in attack_indices)
                move_ratio = sum(dist[i] for i in move_indices)

                result[state] = {
                    'frames': frames,
                    'attack_ratio': float(attack_ratio),
                    'move_ratio': float(move_ratio)
                }

                print(f"  {state:6s}: Attack={attack_ratio:.1%}, Move={move_ratio:.1%} ({frames} frames)")

        # Check for adaptation
        if 'low' in result and 'danger' in result:
            if result['danger']['attack_ratio'] < result['low']['attack_ratio']:
                print("  -> Shows defensive adaptation at high damage!")
                result['shows_adaptation'] = True
            else:
                print("  -> No clear adaptation detected")
                result['shows_adaptation'] = False
        else:
            result['shows_adaptation'] = None

        return result

    def run_cycle4_training(self, start_checkpoint: Optional[str] = None):
        """Run the full Cycle 4 training curriculum."""
        print("\n" + "="*60)
        print("CYCLE 4 UNIFIED TRAINING")
        print("="*60)

        start_time = time.time()
        cfg = self.config

        # Load best starting point
        if start_checkpoint and os.path.exists(start_checkpoint):
            self.load_checkpoint(start_checkpoint)
        else:
            # Try to find best existing checkpoint
            candidates = [
                'checkpoints/anti_aggressive_v3/checkpoint_0600.npz',
                'checkpoints/anti_aggressive_v4/checkpoint_0300.npz',
            ]
            for cp in candidates:
                if os.path.exists(cp):
                    self.load_checkpoint(cp)
                    break

        # Initial evaluation
        print("\n--- Initial State ---")
        initial_eval = self.evaluate_comprehensive(num_games=40)

        # Phase 1: Balanced mixed training to restore random performance
        # while maintaining aggressive capability
        self.train_mixed_phase("Balanced Mixed", cfg.balanced_games, aggressive_ratio=0.5)
        self.policy.save(f'{self.checkpoint_dir}/phase1_balanced.npz')

        # Eval
        print("\n--- After Balanced Phase ---")
        self.evaluate_comprehensive(num_games=40)

        # Phase 2: Aggressive focus with some random refresh
        self.train_mixed_phase("Aggressive Focus", cfg.aggressive_games, aggressive_ratio=0.75)
        self.policy.save(f'{self.checkpoint_dir}/phase2_aggressive.npz')

        # Eval
        print("\n--- After Aggressive Focus ---")
        mid_eval = self.evaluate_comprehensive(num_games=50)

        # Check if we need more aggressive training
        if mid_eval['vs_aggressive']['win_rate'] < 0.50:
            print("\n[!] Below 50% vs aggressive - extra training")
            self.train_phase("Extra Aggressive", self.aggressive_opp, 400)

        # Phase 3: Alternating to prevent overfitting
        print("\n--- Alternating Phase ---")
        for i in range(4):
            self.train_phase(f"Aggressive {i+1}", self.aggressive_opp, 50)
            self.train_phase(f"Random Refresh {i+1}", self.random_opp, 50)

            if (i + 1) % 2 == 0:
                check = self.evaluate_comprehensive(num_games=40)
                if check['vs_aggressive']['win_rate'] >= 0.55 and check['vs_random']['win_rate'] >= 0.80:
                    print("\n All targets achieved! Stopping early.")
                    break

        # Final checkpoint
        self.policy.save(f'{self.checkpoint_dir}/final.npz')

        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)

        final_eval = self.evaluate_comprehensive(num_games=60)

        # Damage state analysis
        damage_behavior = self.analyze_damage_state_behavior()

        total_elapsed = time.time() - start_time

        # Generate report
        report = {
            'cycle': 4,
            'training': {
                'total_games': self.games_played,
                'total_time_seconds': total_elapsed,
                'phases': self.phase_stats,
                'best_vs_aggressive': self.best_vs_aggressive,
                'best_balanced_score': self.best_balanced_score
            },
            'initial_evaluation': {
                'vs_random': initial_eval['vs_random'],
                'vs_aggressive': initial_eval['vs_aggressive']
            },
            'final_evaluation': {
                'vs_random': final_eval['vs_random'],
                'vs_aggressive': final_eval['vs_aggressive'],
                'balanced_score': final_eval['balanced_score']
            },
            'damage_state_behavior': damage_behavior,
            'checkpoints': {
                'final': f'{self.checkpoint_dir}/final.npz',
                'best_vs_aggressive': f'{self.checkpoint_dir}/best_vs_aggressive.npz',
                'best_balanced': f'{self.checkpoint_dir}/best_balanced.npz'
            }
        }

        report_path = f'{self.checkpoint_dir}/cycle4_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_path}")

        # Summary
        vs_rand = final_eval['vs_random']['win_rate']
        vs_agg = final_eval['vs_aggressive']['win_rate']

        print("\n" + "="*60)
        print("CYCLE 4 RESULTS SUMMARY")
        print("="*60)
        print(f"Total training games: {self.games_played}")
        print(f"Total time: {total_elapsed/60:.1f} minutes")
        print(f"\nWin rate vs Random: {vs_rand:.1%} (target: >80%)")
        print(f"Win rate vs Aggressive: {vs_agg:.1%} (target: >50%)")
        print(f"Balanced score: {final_eval['balanced_score']:.3f}")
        print(f"\nBest achieved vs Aggressive: {self.best_vs_aggressive:.1%}")
        print(f"Shows damage adaptation: {damage_behavior.get('shows_adaptation', 'Unknown')}")

        success = vs_rand >= 0.80 and vs_agg >= 0.50
        if success:
            print("\n SUCCESS! All targets achieved!")
        else:
            missing = []
            if vs_rand < 0.80:
                missing.append(f"vs Random needs +{0.80-vs_rand:.1%}")
            if vs_agg < 0.50:
                missing.append(f"vs Aggressive needs +{0.50-vs_agg:.1%}")
            print(f"\n Targets not fully met: {', '.join(missing)}")

        return final_eval


def main():
    print("="*60)
    print("STICK FIGURE FIGHTER - CYCLE 4 UNIFIED TRAINING")
    print("="*60)

    trainer = Cycle4Trainer(
        device='cuda',
        checkpoint_dir='checkpoints/cycle4'
    )

    # Start from best existing checkpoint
    final_eval = trainer.run_cycle4_training(
        start_checkpoint='checkpoints/anti_aggressive_v3/checkpoint_0600.npz'
    )

    return final_eval


if __name__ == '__main__':
    main()
