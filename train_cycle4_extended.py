#!/usr/bin/env python3
"""
Cycle 4 Extended Training Pipeline - Stick Figure Fighter

This cycle focuses on:
1. Extended training (2000+ games) to achieve >50% vs AggressivePolicy
2. Damage-state-aware reward shaping (defensive at high %, aggressive vs damaged opponents)
3. Population-based training for strategy diversity
4. Comprehensive character matchup matrix generation
5. Documentation of emergent behaviors

Training Strategy:
- Start from best existing checkpoint
- Phase 1: Warmup balanced training (200 games)
- Phase 2: Aggressive focus with damage-aware rewards (1200 games)
- Phase 3: Alternating training to prevent overfitting (600 games)
- Total: 2000 games minimum
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

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
class DamageAwareConfig:
    """Configuration for damage-state-aware training."""
    # Training budget
    warmup_games: int = 200
    aggressive_focus_games: int = 1200
    alternating_games: int = 600

    # Damage state thresholds
    low_damage: float = 30.0
    mid_damage: float = 60.0
    high_damage: float = 90.0
    danger_damage: float = 120.0

    # Base rewards
    stock_taken: float = 1.2
    stock_lost: float = -1.2
    win_bonus: float = 0.4
    lose_penalty: float = -0.4

    # Damage dealing - modulated by both players' damage states
    damage_dealt_base: float = 0.002
    damage_dealt_vs_damaged: float = 0.004  # Press advantage when opponent is hurt
    damage_dealt_when_safe: float = 0.003   # More aggressive when we're safe
    damage_dealt_when_danger: float = 0.001 # Less aggressive when in danger

    # Damage taken - scaled by current damage state
    damage_taken_low: float = -0.001
    damage_taken_mid: float = -0.002
    damage_taken_high: float = -0.004
    damage_taken_danger: float = -0.006

    # Combo potential detection
    consecutive_hit_bonus: float = 0.003

    # Stage control
    center_stage_bonus: float = 0.0003
    near_edge_penalty: float = -0.0002

    # Training params
    entropy_coef: float = 0.015
    learning_rate: float = 2.5e-4
    games_per_update: int = 10
    eval_every: int = 200


class ExtendedTrainer:
    """Extended trainer with damage-state-aware rewards and comprehensive tracking."""

    def __init__(self, device: str = 'cuda', checkpoint_dir: str = 'checkpoints/cycle4_extended'):
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

        # Tracking
        self.games_played = 0
        self.training_history = []
        self.best_vs_aggressive = 0.0
        self.best_vs_random = 0.0
        self.best_balanced = 0.0

        # Damage state behavior analysis
        self.damage_state_actions = {
            'low': np.zeros(ACTION_SPACE_SIZE),
            'mid': np.zeros(ACTION_SPACE_SIZE),
            'high': np.zeros(ACTION_SPACE_SIZE),
            'danger': np.zeros(ACTION_SPACE_SIZE)
        }
        self.damage_state_frames = {'low': 0, 'mid': 0, 'high': 0, 'danger': 0}

        # Opponent damage state tracking
        self.opp_damage_attacks = {
            'low': 0, 'mid': 0, 'high': 0, 'danger': 0
        }
        self.opp_damage_frames = {'low': 0, 'mid': 0, 'high': 0, 'danger': 0}

        # Combo tracking
        self.consecutive_hits = 0
        self.combo_stats = defaultdict(int)

    def load_checkpoint(self, path: str) -> bool:
        """Load from existing checkpoint."""
        if os.path.exists(path):
            self.policy.load(path)
            print(f"Loaded checkpoint: {path}")
            return True
        return False

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

        Key innovations:
        1. Self-damage awareness: More cautious at high %, more aggressive when safe
        2. Opponent-damage awareness: Press advantage when opponent is damaged
        3. Consecutive hit bonus: Reward combo potential
        4. Stage control: Reward center stage, penalize near edges
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
            self.consecutive_hits = 0  # Reset combo
        if fighter.stocks < prev_stocks[player_idx]:
            reward += cfg.stock_lost
            self.consecutive_hits = 0

        # Damage dealt - modulated by BOTH players' states
        damage_dealt = opponent.damage_percent - prev_damage[opponent_idx]
        if damage_dealt > 0:
            # Base reward depends on our damage state
            if my_state == 'danger':
                dmg_reward = damage_dealt * cfg.damage_dealt_when_danger
            elif my_state == 'low':
                dmg_reward = damage_dealt * cfg.damage_dealt_when_safe
            else:
                dmg_reward = damage_dealt * cfg.damage_dealt_base

            # Bonus for pressing advantage vs damaged opponent
            if opp_state in ['high', 'danger']:
                dmg_reward += damage_dealt * cfg.damage_dealt_vs_damaged

            # Combo bonus
            self.consecutive_hits += 1
            if self.consecutive_hits > 1:
                combo_bonus = min(self.consecutive_hits - 1, 4) * cfg.consecutive_hit_bonus
                dmg_reward += combo_bonus
                self.combo_stats[self.consecutive_hits] += 1

            reward += dmg_reward

            # Track opponent damage attacks
            self.opp_damage_attacks[opp_state] += 1
        else:
            # Reset combo on no damage
            if action in [5, 6, 7, 8, 9]:  # Attack action but no damage
                self.consecutive_hits = 0

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

        # Stage control
        if hasattr(fighter, 'position'):
            stage_center = 400
            stage_width = 300  # Approximate safe zone half-width
            dist_from_center = abs(fighter.position.x - stage_center)

            if dist_from_center < 100:  # Near center
                reward += cfg.center_stage_bonus
            elif dist_from_center > stage_width:  # Near edge
                reward += cfg.near_edge_penalty

        return reward

    def run_game(self, opponent, track_damage_states: bool = True) -> Tuple[Episode, int, dict]:
        """Run a single game with comprehensive tracking."""
        game = create_game(
            stage_name='battlefield',
            p1_character='FLEX',
            p2_character='FLEX',
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
        self.consecutive_hits = 0

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
                opp_damage = game.fighters[1].damage_percent
                my_state = self.get_damage_state(my_damage)
                opp_state = self.get_damage_state(opp_damage)

                self.damage_state_actions[my_state][action_p1] += 1
                self.damage_state_frames[my_state] += 1
                self.opp_damage_frames[opp_state] += 1

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
            'damage_taken': damage_taken_total,
            'max_combo': max(self.combo_stats.keys()) if self.combo_stats else 0
        }

        return episode, winner, stats

    def train_phase(self, name: str, opponent, num_games: int,
                    print_every: int = 100) -> dict:
        """Train for one phase with detailed tracking."""
        print(f"\n{'='*60}")
        print(f"PHASE: {name}")
        print(f"Games: {num_games}, Opponent: {type(opponent).__name__}")
        print(f"Total games so far: {self.games_played}")
        print(f"{'='*60}")

        start_time = time.time()
        wins = 0
        episodes_buffer = []
        games_per_update = self.config.games_per_update

        phase_damage_dealt = 0.0
        phase_damage_taken = 0.0

        for game_idx in range(num_games):
            episode, winner, stats = self.run_game(opponent)
            episodes_buffer.append(episode)
            if winner == 0:
                wins += 1
            phase_damage_dealt += stats['damage_dealt']
            phase_damage_taken += stats['damage_taken']

            if len(episodes_buffer) >= games_per_update:
                train_stats = self.trainer.update(episodes_buffer)
                episodes_buffer = []

                if (game_idx + 1) % print_every == 0:
                    win_rate = wins / (game_idx + 1)
                    print(f"  [{game_idx+1:4d}/{num_games}] "
                          f"WR: {win_rate:.1%} | "
                          f"Loss: {train_stats['policy_loss']:.4f} | "
                          f"Ent: {train_stats['entropy']:.4f}")

        if episodes_buffer:
            self.trainer.update(episodes_buffer)

        elapsed = time.time() - start_time
        win_rate = wins / num_games

        result = {
            'phase': name,
            'games': num_games,
            'wins': wins,
            'win_rate': win_rate,
            'elapsed': elapsed,
            'avg_damage_dealt': phase_damage_dealt / num_games,
            'avg_damage_taken': phase_damage_taken / num_games
        }

        print(f"\nPhase complete: {wins}/{num_games} wins ({win_rate:.1%})")
        print(f"Time: {elapsed:.1f}s ({num_games/elapsed:.1f} games/sec)")
        print(f"Avg damage dealt: {result['avg_damage_dealt']:.1f}")

        self.training_history.append(result)
        return result

    def train_mixed_phase(self, name: str, num_games: int,
                          aggressive_ratio: float) -> dict:
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

            if len(episodes_buffer) >= self.config.games_per_update:
                self.trainer.update(episodes_buffer)
                episodes_buffer = []

                if (game_idx + 1) % 200 == 0:
                    wr_agg = wins_vs_agg / max(1, games_vs_agg)
                    wr_rand = wins_vs_rand / max(1, games_vs_rand)
                    print(f"  [{game_idx+1:4d}/{num_games}] "
                          f"vs Agg: {wr_agg:.1%} | vs Rand: {wr_rand:.1%}")

        if episodes_buffer:
            self.trainer.update(episodes_buffer)

        elapsed = time.time() - start_time

        result = {
            'phase': name,
            'games': num_games,
            'wins_vs_agg': wins_vs_agg,
            'games_vs_agg': games_vs_agg,
            'wins_vs_rand': wins_vs_rand,
            'games_vs_rand': games_vs_rand,
            'wr_vs_agg': wins_vs_agg / max(1, games_vs_agg),
            'wr_vs_rand': wins_vs_rand / max(1, games_vs_rand),
            'elapsed': elapsed
        }

        print(f"\nPhase complete:")
        print(f"  vs Aggressive: {wins_vs_agg}/{games_vs_agg} ({result['wr_vs_agg']:.1%})")
        print(f"  vs Random: {wins_vs_rand}/{games_vs_rand} ({result['wr_vs_rand']:.1%})")

        self.training_history.append(result)
        return result

    def evaluate(self, num_games: int = 50) -> dict:
        """Full evaluation against baselines."""
        print("\n--- Evaluation ---")

        vs_random = evaluate_vs_random(self.policy, num_games=num_games)
        vs_aggressive = evaluate_vs_aggressive(self.policy, num_games=num_games)

        print(f"  vs Random: {vs_random['win_rate']:.1%}")
        print(f"  vs Aggressive: {vs_aggressive['win_rate']:.1%}")

        # Update bests and save checkpoints
        if vs_aggressive['win_rate'] > self.best_vs_aggressive:
            self.best_vs_aggressive = vs_aggressive['win_rate']
            self.policy.save(f'{self.checkpoint_dir}/best_vs_aggressive.npz')
            print(f"  -> New best vs Aggressive!")

        if vs_random['win_rate'] > self.best_vs_random:
            self.best_vs_random = vs_random['win_rate']
            self.policy.save(f'{self.checkpoint_dir}/best_vs_random.npz')

        balanced = np.sqrt(vs_random['win_rate'] * vs_aggressive['win_rate'])
        if balanced > self.best_balanced:
            self.best_balanced = balanced
            self.policy.save(f'{self.checkpoint_dir}/best_balanced.npz')
            print(f"  -> New best balanced!")

        return {
            'vs_random': vs_random,
            'vs_aggressive': vs_aggressive,
            'balanced_score': balanced
        }

    def analyze_damage_adaptation(self) -> dict:
        """Analyze how behavior differs by damage state."""
        print("\n--- Damage-State Behavior Analysis ---")

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
                    'move_ratio': float(move_ratio),
                    'idle_ratio': float(dist[0])
                }

                print(f"  {state:6s}: Attack={attack_ratio:.1%}, Move={move_ratio:.1%}, Idle={dist[0]:.1%}")

        # Check for adaptation
        if 'low' in result and 'danger' in result:
            low_attack = result['low']['attack_ratio']
            danger_attack = result['danger']['attack_ratio']
            low_move = result['low']['move_ratio']
            danger_move = result['danger']['move_ratio']

            if danger_attack < low_attack:
                print(f"\n  SHOWS DEFENSIVE ADAPTATION: {low_attack:.1%} -> {danger_attack:.1%} attack ratio")
                result['shows_defensive_adaptation'] = True
                result['attack_reduction'] = low_attack - danger_attack
            else:
                print(f"\n  No defensive adaptation detected")
                result['shows_defensive_adaptation'] = False

            if danger_move > low_move:
                print(f"  SHOWS EVASIVE ADAPTATION: {low_move:.1%} -> {danger_move:.1%} movement")
                result['shows_evasive_adaptation'] = True
            else:
                result['shows_evasive_adaptation'] = False

        # Analyze pressing advantage vs damaged opponents
        print("\n  --- Attack Rate vs Opponent Damage State ---")
        for state in ['low', 'mid', 'high', 'danger']:
            if self.opp_damage_frames[state] > 0:
                attack_rate = self.opp_damage_attacks[state] / self.opp_damage_frames[state]
                print(f"    Opp {state:6s}: {attack_rate:.1%} attack rate")
                result[f'attack_rate_vs_opp_{state}'] = float(attack_rate)

        # Check for pressing advantage
        if 'attack_rate_vs_opp_low' in result and 'attack_rate_vs_opp_danger' in result:
            if result['attack_rate_vs_opp_danger'] > result['attack_rate_vs_opp_low']:
                print(f"\n  SHOWS ADVANTAGE PRESSING behavior")
                result['presses_advantage'] = True
            else:
                result['presses_advantage'] = False

        return result

    def analyze_combos(self) -> dict:
        """Analyze combo statistics."""
        print("\n--- Combo Analysis ---")

        if self.combo_stats:
            total_combos = sum(self.combo_stats.values())
            print(f"  Total multi-hit combos: {total_combos}")

            for combo_len in sorted(self.combo_stats.keys()):
                count = self.combo_stats[combo_len]
                print(f"    {combo_len}-hit: {count} ({count/total_combos:.1%})")
        else:
            print("  No combos detected")

        return dict(self.combo_stats)

    def find_best_checkpoint(self) -> Optional[str]:
        """Find the best existing checkpoint to start from."""
        candidates = [
            'checkpoints/cycle4/best_vs_aggressive.npz',
            'checkpoints/cycle4/best_balanced.npz',
            'checkpoints/anti_aggressive_v3/checkpoint_0600.npz',
            'checkpoints/anti_aggressive_v5/checkpoint_final.npz',
            'checkpoints/full_run/checkpoint_001000.npz',
        ]

        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def run_extended_training(self) -> dict:
        """Run the full extended training pipeline."""
        print("\n" + "="*70)
        print("CYCLE 4 EXTENDED TRAINING - 2000+ GAMES")
        print("Focus: >50% vs Aggressive, Damage-State Adaptation")
        print("="*70)

        start_time = time.time()
        cfg = self.config

        # Find and load best starting point
        best_checkpoint = self.find_best_checkpoint()
        if best_checkpoint:
            self.load_checkpoint(best_checkpoint)
        else:
            print("Starting from scratch (no checkpoint found)")

        # Initial evaluation
        print("\n--- Initial State ---")
        initial_eval = self.evaluate(num_games=50)

        # Phase 1: Warmup with balanced training
        self.train_mixed_phase("Warmup Balanced", cfg.warmup_games, aggressive_ratio=0.5)
        self.policy.save(f'{self.checkpoint_dir}/phase1_warmup.npz')
        self.evaluate(num_games=40)

        # Phase 2: Heavy aggressive focus with damage-aware rewards
        # Split into sub-phases for checkpointing
        games_per_subphase = 400
        num_subphases = cfg.aggressive_focus_games // games_per_subphase

        for i in range(num_subphases):
            self.train_mixed_phase(
                f"Aggressive Focus {i+1}/{num_subphases}",
                games_per_subphase,
                aggressive_ratio=0.80  # 80% aggressive opponents
            )

            # Checkpoint and evaluate
            self.policy.save(f'{self.checkpoint_dir}/phase2_sub{i+1}.npz')
            eval_result = self.evaluate(num_games=40)

            # Early stopping if targets achieved
            if (eval_result['vs_aggressive']['win_rate'] >= 0.55 and
                eval_result['vs_random']['win_rate'] >= 0.85):
                print("\n Early success! Targets exceeded.")
                break

        # Phase 3: Alternating to prevent overfitting
        print("\n" + "="*60)
        print("ALTERNATING PHASE - Prevent Overfitting")
        print("="*60)

        alt_rounds = cfg.alternating_games // 100
        for i in range(alt_rounds):
            # Aggressive round
            self.train_phase(f"Alt Aggressive {i+1}", self.aggressive_opp, 60)
            # Random refresh
            self.train_phase(f"Alt Random {i+1}", self.random_opp, 40)

            if (i + 1) % 3 == 0:
                check = self.evaluate(num_games=40)
                if (check['vs_aggressive']['win_rate'] >= 0.55 and
                    check['vs_random']['win_rate'] >= 0.80):
                    print("\n All targets achieved!")
                    break

        # Final checkpoint and evaluation
        self.policy.save(f'{self.checkpoint_dir}/final.npz')

        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)

        final_eval = self.evaluate(num_games=60)

        # Analyze behaviors
        damage_behavior = self.analyze_damage_adaptation()
        combo_stats = self.analyze_combos()

        total_elapsed = time.time() - start_time

        # Generate comprehensive report
        report = {
            'cycle': 4,
            'variant': 'extended',
            'training': {
                'total_games': self.games_played,
                'total_time_seconds': total_elapsed,
                'phases': self.training_history,
                'start_checkpoint': best_checkpoint,
                'best_vs_aggressive': self.best_vs_aggressive,
                'best_vs_random': self.best_vs_random,
                'best_balanced': self.best_balanced
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
            'combo_stats': combo_stats,
            'success_criteria': {
                'vs_aggressive_target': 0.50,
                'vs_aggressive_achieved': final_eval['vs_aggressive']['win_rate'],
                'vs_aggressive_met': final_eval['vs_aggressive']['win_rate'] >= 0.50,
                'vs_random_target': 0.80,
                'vs_random_achieved': final_eval['vs_random']['win_rate'],
                'vs_random_met': final_eval['vs_random']['win_rate'] >= 0.80,
                'shows_damage_adaptation': damage_behavior.get('shows_defensive_adaptation', False),
                'presses_advantage': damage_behavior.get('presses_advantage', False)
            },
            'checkpoints': {
                'final': f'{self.checkpoint_dir}/final.npz',
                'best_vs_aggressive': f'{self.checkpoint_dir}/best_vs_aggressive.npz',
                'best_balanced': f'{self.checkpoint_dir}/best_balanced.npz'
            }
        }

        report_path = f'{self.checkpoint_dir}/training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Summary
        print("\n" + "="*70)
        print("CYCLE 4 EXTENDED TRAINING COMPLETE")
        print("="*70)
        print(f"Total games: {self.games_played}")
        print(f"Total time: {total_elapsed/60:.1f} minutes")
        print(f"\nFinal Results:")
        vs_rand = final_eval['vs_random']['win_rate']
        vs_agg = final_eval['vs_aggressive']['win_rate']
        print(f"  vs Random: {vs_rand:.1%} (target: >80%) {'PASS' if vs_rand >= 0.80 else 'FAIL'}")
        print(f"  vs Aggressive: {vs_agg:.1%} (target: >50%) {'PASS' if vs_agg >= 0.50 else 'FAIL'}")
        print(f"\nBest achieved:")
        print(f"  Best vs Aggressive: {self.best_vs_aggressive:.1%}")
        print(f"  Best vs Random: {self.best_vs_random:.1%}")
        print(f"\nBehavior Adaptations:")
        print(f"  Defensive at danger: {damage_behavior.get('shows_defensive_adaptation', 'Unknown')}")
        print(f"  Presses advantage: {damage_behavior.get('presses_advantage', 'Unknown')}")

        success = vs_rand >= 0.80 and vs_agg >= 0.50
        if success:
            print("\n SUCCESS - All targets achieved!")
        else:
            print("\n Some targets not met.")

        print(f"\nReport saved to: {report_path}")

        return report


def main():
    print("="*70)
    print("STICK FIGURE FIGHTER - CYCLE 4 EXTENDED TRAINING")
    print("2000+ games with damage-state-aware reward shaping")
    print("="*70)

    trainer = ExtendedTrainer(
        device='cuda',
        checkpoint_dir='checkpoints/cycle4_extended'
    )

    report = trainer.run_extended_training()

    return report


if __name__ == '__main__':
    main()
