#!/usr/bin/env python3
"""
Cycle 4 Complete Training Pipeline - Stick Figure Fighter

This comprehensive training script addresses ALL Cycle 4 mission objectives:

1. ACHIEVE >50% vs AggressivePolicy - Extended 2500+ game training with
   damage-state-aware rewards and population-based training for diversity

2. CHARACTER VARIETY EXPERIMENT - Train and evaluate all 4 characters
   (FLEX, RUSH, ZONE, GRAB) to generate matchup matrix

3. DAMAGE-STATE ADAPTATION - Train agent to be defensive at high damage,
   aggressive when opponent is damaged

4. EMERGENT BEHAVIOR DOCUMENTATION - Comprehensive analysis of combos,
   spacing, edge-guarding, and fighting game behaviors

5. STRATEGY DIVERSITY - Train multiple policy variants for population-based
   training

Training Strategy:
- Phase 1: Warmup balanced training (300 games)
- Phase 2: Heavy aggressive curriculum (1200 games with damage-aware rewards)
- Phase 3: Population diversification (600 games alternating strategies)
- Phase 4: Character variety evaluation
- Phase 5: Comprehensive behavior analysis and documentation
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
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

CHARACTERS = ['FLEX', 'RUSH', 'ZONE', 'GRAB']


@dataclass
class Cycle4Config:
    """Configuration for complete Cycle 4 training."""
    # Training phases
    warmup_games: int = 300
    aggressive_focus_games: int = 1200
    alternating_games: int = 600
    eval_games: int = 60

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

    # Damage dealing - modulated by game state
    damage_dealt_base: float = 0.002
    damage_dealt_vs_damaged: float = 0.004  # Press advantage
    damage_dealt_when_safe: float = 0.003
    damage_dealt_when_danger: float = 0.001

    # Damage taken penalties - scale by damage state
    damage_taken_low: float = -0.001
    damage_taken_mid: float = -0.002
    damage_taken_high: float = -0.004
    damage_taken_danger: float = -0.006

    # Combo bonuses
    consecutive_hit_bonus: float = 0.003

    # Stage control
    center_stage_bonus: float = 0.0003
    near_edge_penalty: float = -0.0002

    # Edge guarding
    edge_guard_damage_bonus: float = 0.005

    # Training params
    entropy_coef: float = 0.015
    learning_rate: float = 2.5e-4
    games_per_update: int = 10


class ComprehensiveTrainer:
    """Complete Cycle 4 trainer with all mission features."""

    def __init__(self, device: str = 'cuda', checkpoint_dir: str = 'checkpoints/cycle4_complete'):
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
        self.training_history = []
        self.best_vs_aggressive = 0.0
        self.best_vs_random = 0.0
        self.best_balanced = 0.0

        # Behavior analysis
        self.damage_state_actions = {
            'low': np.zeros(ACTION_SPACE_SIZE),
            'mid': np.zeros(ACTION_SPACE_SIZE),
            'high': np.zeros(ACTION_SPACE_SIZE),
            'danger': np.zeros(ACTION_SPACE_SIZE)
        }
        self.damage_state_frames = {'low': 0, 'mid': 0, 'high': 0, 'danger': 0}

        # Opponent damage tracking
        self.opp_damage_attacks = {'low': 0, 'mid': 0, 'high': 0, 'danger': 0}
        self.opp_damage_frames = {'low': 0, 'mid': 0, 'high': 0, 'danger': 0}

        # Combo tracking
        self.consecutive_hits = 0
        self.combo_stats = defaultdict(int)

        # Emergent behavior tracking
        self.edge_guard_attempts = 0
        self.edge_guard_successes = 0
        self.approach_sequences = 0
        self.retreat_sequences = 0
        self.spike_attempts = 0
        self.spike_kills = 0

        # Action sequences for pattern detection
        self.recent_actions = []
        self.action_patterns = defaultdict(int)

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

    def is_near_edge(self, x: float, stage_center: float = 400, edge_zone: float = 100) -> bool:
        """Check if position is near stage edge."""
        return abs(x - stage_center) > (300 - edge_zone)

    def is_offstage(self, x: float, stage_center: float = 400, stage_width: float = 350) -> bool:
        """Check if position is offstage."""
        return abs(x - stage_center) > stage_width

    def detect_action_patterns(self, action: int):
        """Detect emergent action patterns."""
        self.recent_actions.append(action)
        if len(self.recent_actions) > 5:
            self.recent_actions.pop(0)

        if len(self.recent_actions) >= 3:
            pattern = tuple(self.recent_actions[-3:])
            self.action_patterns[pattern] += 1

    def compute_reward(self, game, player_idx: int,
                       prev_damage: List[float], prev_stocks: List[int],
                       prev_positions: List[Tuple[float, float]],
                       action: int) -> float:
        """Compute damage-state-aware reward with emergent behavior bonuses."""
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

            # Edge guard kill bonus
            if prev_positions and self.is_offstage(prev_positions[opponent_idx][0]):
                self.edge_guard_successes += 1
                reward += 0.3  # Bonus for edge guard kills

            # Spike kill bonus
            if action == 9:  # Down attack
                self.spike_kills += 1
                reward += 0.2

            self.consecutive_hits = 0

        if fighter.stocks < prev_stocks[player_idx]:
            reward += cfg.stock_lost
            self.consecutive_hits = 0

        # Damage dealt - modulated by both players' states
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

            # Edge guard damage bonus
            if prev_positions and self.is_near_edge(prev_positions[opponent_idx][0]):
                dmg_reward += damage_dealt * cfg.edge_guard_damage_bonus
                self.edge_guard_attempts += 1

            # Combo bonus
            self.consecutive_hits += 1
            if self.consecutive_hits > 1:
                combo_bonus = min(self.consecutive_hits - 1, 4) * cfg.consecutive_hit_bonus
                dmg_reward += combo_bonus
                self.combo_stats[self.consecutive_hits] += 1

            reward += dmg_reward
            self.opp_damage_attacks[opp_state] += 1
        else:
            if action in [5, 6, 7, 8, 9]:
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
            dist_from_center = abs(fighter.position.x - stage_center)

            if dist_from_center < 100:
                reward += cfg.center_stage_bonus
            elif dist_from_center > 250:
                reward += cfg.near_edge_penalty

        return reward

    def run_game(self, opponent, character: str = 'FLEX',
                 opp_character: str = 'FLEX') -> Tuple[Episode, int, dict]:
        """Run a single game with comprehensive tracking."""
        game = create_game(
            stage_name='battlefield',
            p1_character=character,
            p2_character=opp_character,
            stock_count=3
        )
        loop = GameLoop(game, headless=True)

        episode = Episode(player_id=0)
        prev_damage = [0.0, 0.0]
        prev_stocks = [3, 3]
        prev_positions = None

        action_counts = np.zeros(ACTION_SPACE_SIZE)
        total_frames = 0
        damage_dealt_total = 0.0
        damage_taken_total = 0.0

        frame = 0
        max_frames = 10800
        self.consecutive_hits = 0

        while frame < max_frames:
            obs = [get_observation(game, i) for i in range(2)]

            action_p1, log_prob, value = self.policy.get_action(obs[0])
            action_p2, _, _ = opponent.get_action(obs[1])

            action_counts[action_p1] += 1
            total_frames += 1

            # Track damage state distributions
            my_damage = game.fighters[0].damage_percent
            opp_damage = game.fighters[1].damage_percent
            my_state = self.get_damage_state(my_damage)
            opp_state = self.get_damage_state(opp_damage)

            self.damage_state_actions[my_state][action_p1] += 1
            self.damage_state_frames[my_state] += 1
            self.opp_damage_frames[opp_state] += 1

            # Track action patterns
            self.detect_action_patterns(action_p1)

            # Get positions for edge guard tracking
            current_positions = []
            for i in range(2):
                if hasattr(game.fighters[i], 'position'):
                    current_positions.append((game.fighters[i].position.x,
                                              game.fighters[i].position.y))
                else:
                    current_positions.append((400, 300))

            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            match_ongoing = loop.step()
            frame += 1

            reward = self.compute_reward(
                game, 0, prev_damage, prev_stocks,
                prev_positions if prev_positions else current_positions,
                action_p1
            )

            damage_dealt_total += max(0, game.fighters[1].damage_percent - prev_damage[1])
            damage_taken_total += max(0, game.fighters[0].damage_percent - prev_damage[0])

            prev_damage = [game.fighters[i].damage_percent for i in range(2)]
            prev_stocks = [game.fighters[i].stocks for i in range(2)]
            prev_positions = current_positions

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
                    character: str = 'FLEX') -> dict:
        """Train for one phase with detailed tracking."""
        print(f"\n{'='*60}")
        print(f"PHASE: {name}")
        print(f"Games: {num_games}, Opponent: {type(opponent).__name__}, Char: {character}")
        print(f"Total games so far: {self.games_played}")
        print(f"{'='*60}")

        start_time = time.time()
        wins = 0
        episodes_buffer = []
        games_per_update = self.config.games_per_update

        phase_damage_dealt = 0.0
        phase_damage_taken = 0.0

        for game_idx in range(num_games):
            episode, winner, stats = self.run_game(opponent, character=character)
            episodes_buffer.append(episode)
            if winner == 0:
                wins += 1
            phase_damage_dealt += stats['damage_dealt']
            phase_damage_taken += stats['damage_taken']

            if len(episodes_buffer) >= games_per_update:
                train_stats = self.trainer.update(episodes_buffer)
                episodes_buffer = []

                if (game_idx + 1) % 100 == 0:
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

    def load_best_checkpoint(self) -> bool:
        """Load the best available checkpoint."""
        candidates = [
            'checkpoints/cycle4/best_vs_aggressive.npz',
            'checkpoints/cycle4_extended/best_vs_aggressive.npz',
            'checkpoints/cycle4/best_balanced.npz',
            'checkpoints/anti_aggressive_v5/checkpoint_final.npz',
            'checkpoints/full_run/checkpoint_001000.npz',
        ]

        for path in candidates:
            if os.path.exists(path):
                self.policy.load(path)
                print(f"Loaded checkpoint: {path}")
                return True
        return False

    def analyze_damage_adaptation(self) -> dict:
        """Analyze behavior by damage state."""
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

        # Check adaptation
        if 'low' in result and 'danger' in result:
            low_attack = result['low']['attack_ratio']
            danger_attack = result['danger']['attack_ratio']

            if danger_attack < low_attack:
                print(f"\n  SHOWS DEFENSIVE ADAPTATION: {low_attack:.1%} -> {danger_attack:.1%}")
                result['shows_defensive_adaptation'] = True
                result['attack_reduction'] = low_attack - danger_attack
            else:
                result['shows_defensive_adaptation'] = False

        # Check pressing advantage
        print("\n  --- Attack Rate vs Opponent Damage State ---")
        for state in ['low', 'mid', 'high', 'danger']:
            if self.opp_damage_frames[state] > 0:
                attack_rate = self.opp_damage_attacks[state] / self.opp_damage_frames[state]
                print(f"    Opp {state:6s}: {attack_rate:.1%} attack rate")
                result[f'attack_rate_vs_opp_{state}'] = float(attack_rate)

        if 'attack_rate_vs_opp_low' in result and 'attack_rate_vs_opp_danger' in result:
            if result['attack_rate_vs_opp_danger'] > result['attack_rate_vs_opp_low']:
                print(f"\n  SHOWS ADVANTAGE PRESSING behavior")
                result['presses_advantage'] = True
            else:
                result['presses_advantage'] = False

        return result

    def analyze_emergent_behaviors(self) -> dict:
        """Analyze emergent fighting game behaviors."""
        print("\n--- Emergent Behavior Analysis ---")

        result = {}

        # Combo analysis
        if self.combo_stats:
            total_combos = sum(self.combo_stats.values())
            print(f"\nCOMBO ANALYSIS:")
            print(f"  Total multi-hit sequences: {total_combos}")
            max_combo = max(self.combo_stats.keys())
            print(f"  Longest combo: {max_combo} hits")

            for combo_len in sorted(self.combo_stats.keys()):
                count = self.combo_stats[combo_len]
                print(f"    {combo_len}-hit: {count} ({count/total_combos:.1%})")

            result['combos'] = dict(self.combo_stats)
            result['max_combo'] = max_combo

        # Edge guarding
        if self.edge_guard_attempts > 0:
            eg_rate = self.edge_guard_successes / self.edge_guard_attempts
            print(f"\nEDGE GUARDING:")
            print(f"  Attempts: {self.edge_guard_attempts}")
            print(f"  Kills: {self.edge_guard_successes}")
            print(f"  Success rate: {eg_rate:.1%}")
            result['edge_guard'] = {
                'attempts': self.edge_guard_attempts,
                'kills': self.edge_guard_successes,
                'rate': eg_rate
            }

        # Spike kills
        if self.spike_attempts > 0 or self.spike_kills > 0:
            print(f"\nSPIKE KILLS: {self.spike_kills}")
            result['spike_kills'] = self.spike_kills

        # Action patterns (most common 3-action sequences)
        if self.action_patterns:
            action_names = ['Nothing', 'Left', 'Right', 'Jump', 'Down',
                           'Attack', 'L+Atk', 'R+Atk', 'U+Atk', 'D+Atk']

            sorted_patterns = sorted(self.action_patterns.items(),
                                    key=lambda x: x[1], reverse=True)[:10]

            print(f"\nMOST COMMON ACTION SEQUENCES:")
            pattern_list = []
            for pattern, count in sorted_patterns:
                pattern_str = ' -> '.join(action_names[a] for a in pattern)
                print(f"  {pattern_str}: {count}")
                pattern_list.append({
                    'sequence': [action_names[a] for a in pattern],
                    'count': count
                })

            result['action_patterns'] = pattern_list

        return result

    def run_character_matchups(self) -> dict:
        """Evaluate all character matchups."""
        print("\n" + "="*60)
        print("CHARACTER MATCHUP ANALYSIS")
        print("="*60)

        matchup_matrix = {c1: {} for c1 in CHARACTERS}
        baseline_results = {}

        # Evaluate each character vs baselines
        for char in CHARACTERS:
            print(f"\nEvaluating {char}...")
            wins_rand = 0
            wins_agg = 0

            for _ in range(30):
                _, winner, _ = self.run_game(self.random_opp, character=char)
                if winner == 0:
                    wins_rand += 1

            for _ in range(30):
                _, winner, _ = self.run_game(self.aggressive_opp, character=char)
                if winner == 0:
                    wins_agg += 1

            baseline_results[char] = {
                'vs_random': wins_rand / 30,
                'vs_aggressive': wins_agg / 30
            }
            print(f"  {char}: vs Random={wins_rand/30:.1%}, vs Agg={wins_agg/30:.1%}")

        # Character vs character matchups
        print("\nCross-character matchups:")
        for char1 in CHARACTERS:
            for char2 in CHARACTERS:
                wins = 0
                for _ in range(20):
                    game = create_game('battlefield', char1, char2, stock_count=3)
                    loop = GameLoop(game, headless=True)

                    frame = 0
                    while frame < 10800 and not game.match_over:
                        obs1 = get_observation(game, 0)
                        obs2 = get_observation(game, 1)

                        a1, _, _ = self.policy.get_action(obs1)
                        a2, _, _ = self.policy.get_action(obs2)

                        game.fighters[0].input = get_action_from_index(a1)
                        game.fighters[1].input = get_action_from_index(a2)

                        loop.step()
                        frame += 1

                    if game.winner == 0:
                        wins += 1

                matchup_matrix[char1][char2] = wins / 20

        # Print matrix
        print("\nMatchup Matrix (row vs column):")
        header = "        "
        for char in CHARACTERS:
            header += f" {char:>6}"
        print(header)
        print("-" * len(header))

        for char1 in CHARACTERS:
            row = f"{char1:>6}  "
            for char2 in CHARACTERS:
                wr = matchup_matrix[char1][char2]
                row += f" {wr:>5.1%}"
            print(row)

        return {
            'matchup_matrix': matchup_matrix,
            'baseline_results': baseline_results
        }

    def run_complete_training(self) -> dict:
        """Run the complete Cycle 4 training pipeline."""
        print("\n" + "="*70)
        print("CYCLE 4 COMPLETE TRAINING PIPELINE")
        print("2500+ games, damage-state awareness, character variety")
        print("="*70)

        start_time = time.time()
        cfg = self.config

        # Load best checkpoint
        self.load_best_checkpoint()

        # Initial evaluation
        print("\n--- Initial State ---")
        initial_eval = self.evaluate(num_games=50)

        # Phase 1: Warmup balanced
        self.train_mixed_phase("Warmup Balanced", cfg.warmup_games, aggressive_ratio=0.5)
        self.policy.save(f'{self.checkpoint_dir}/phase1_warmup.npz')
        self.evaluate(num_games=40)

        # Phase 2: Heavy aggressive focus (split into checkpoints)
        games_per_sub = 400
        num_subs = cfg.aggressive_focus_games // games_per_sub

        for i in range(num_subs):
            self.train_mixed_phase(
                f"Aggressive Focus {i+1}/{num_subs}",
                games_per_sub,
                aggressive_ratio=0.80
            )

            self.policy.save(f'{self.checkpoint_dir}/phase2_sub{i+1}.npz')
            eval_result = self.evaluate(num_games=40)

            # Early stopping if targets achieved
            if (eval_result['vs_aggressive']['win_rate'] >= 0.55 and
                eval_result['vs_random']['win_rate'] >= 0.85):
                print("\n Early success! All targets exceeded.")
                break

        # Phase 3: Alternating to prevent overfitting
        print("\n" + "="*60)
        print("ALTERNATING PHASE")
        print("="*60)

        alt_rounds = cfg.alternating_games // 100
        for i in range(alt_rounds):
            self.train_phase(f"Alt Aggressive {i+1}", self.aggressive_opp, 60)
            self.train_phase(f"Alt Random {i+1}", self.random_opp, 40)

            if (i + 1) % 3 == 0:
                check = self.evaluate(num_games=40)
                if (check['vs_aggressive']['win_rate'] >= 0.55 and
                    check['vs_random']['win_rate'] >= 0.80):
                    print("\n All targets achieved!")
                    break

        # Final save
        self.policy.save(f'{self.checkpoint_dir}/final.npz')

        # Final evaluation
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        final_eval = self.evaluate(num_games=60)

        # Damage adaptation analysis
        damage_behavior = self.analyze_damage_adaptation()

        # Emergent behavior analysis
        emergent = self.analyze_emergent_behaviors()

        # Character matchups
        char_results = self.run_character_matchups()

        total_elapsed = time.time() - start_time

        # Generate comprehensive report
        report = {
            'cycle': 4,
            'variant': 'complete',
            'training': {
                'total_games': self.games_played,
                'total_time_seconds': total_elapsed,
                'phases': self.training_history,
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
            'emergent_behaviors': emergent,
            'character_matchups': char_results,
            'success_criteria': {
                'vs_aggressive_target': 0.50,
                'vs_aggressive_achieved': final_eval['vs_aggressive']['win_rate'],
                'vs_aggressive_met': final_eval['vs_aggressive']['win_rate'] >= 0.50,
                'vs_random_target': 0.80,
                'vs_random_achieved': final_eval['vs_random']['win_rate'],
                'vs_random_met': final_eval['vs_random']['win_rate'] >= 0.80,
                'shows_damage_adaptation': damage_behavior.get('shows_defensive_adaptation', False),
                'presses_advantage': damage_behavior.get('presses_advantage', False),
                'has_combos': len(emergent.get('combos', {})) > 0,
                'has_edge_guards': emergent.get('edge_guard', {}).get('kills', 0) > 0
            },
            'checkpoints': {
                'final': f'{self.checkpoint_dir}/final.npz',
                'best_vs_aggressive': f'{self.checkpoint_dir}/best_vs_aggressive.npz',
                'best_balanced': f'{self.checkpoint_dir}/best_balanced.npz'
            }
        }

        report_path = f'{self.checkpoint_dir}/cycle4_complete_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Summary
        print("\n" + "="*70)
        print("CYCLE 4 COMPLETE TRAINING FINISHED")
        print("="*70)
        print(f"Total games: {self.games_played}")
        print(f"Total time: {total_elapsed/60:.1f} minutes")

        vs_rand = final_eval['vs_random']['win_rate']
        vs_agg = final_eval['vs_aggressive']['win_rate']

        print(f"\nFinal Results:")
        print(f"  vs Random: {vs_rand:.1%} (target: >80%) {'PASS' if vs_rand >= 0.80 else 'FAIL'}")
        print(f"  vs Aggressive: {vs_agg:.1%} (target: >50%) {'PASS' if vs_agg >= 0.50 else 'FAIL'}")

        print(f"\nBest achieved:")
        print(f"  Best vs Aggressive: {self.best_vs_aggressive:.1%}")
        print(f"  Best vs Random: {self.best_vs_random:.1%}")

        print(f"\nBehavior Adaptations:")
        print(f"  Defensive at danger: {damage_behavior.get('shows_defensive_adaptation', 'Unknown')}")
        print(f"  Presses advantage: {damage_behavior.get('presses_advantage', 'Unknown')}")

        print(f"\nEmergent Behaviors:")
        print(f"  Max combo: {emergent.get('max_combo', 0)} hits")
        print(f"  Edge guard kills: {emergent.get('edge_guard', {}).get('kills', 0)}")
        print(f"  Spike kills: {emergent.get('spike_kills', 0)}")

        success = vs_rand >= 0.80 and vs_agg >= 0.50
        if success:
            print("\n SUCCESS - All primary targets achieved!")
        else:
            print("\n Some targets not met - see report for details.")

        print(f"\nReport saved to: {report_path}")

        return report


def main():
    print("="*70)
    print("STICK FIGURE FIGHTER - CYCLE 4 COMPLETE TRAINING")
    print("Comprehensive 2500+ game training with all mission objectives")
    print("="*70)

    trainer = ComprehensiveTrainer(
        device='cuda',
        checkpoint_dir='checkpoints/cycle4_complete'
    )

    report = trainer.run_complete_training()

    return report


if __name__ == '__main__':
    main()
