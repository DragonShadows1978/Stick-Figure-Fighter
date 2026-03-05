#!/usr/bin/env python3
"""
Cycle 5: Defensive Training Experiment

This experiment tests whether damage-state-aware defensive behavior
provides measurable win rate improvement over hyper-aggressive play.

Hypothesis:
- Current hyper-aggressive policy maintains ~50% attack rate regardless of damage
- Defensive adaptation (reducing aggression when damage >80%) might improve survival
- Alternative: overwhelming offense may still be superior

Methodology:
1. Train a "defensive" policy with heavy penalties for attacking at high damage
2. Train a control "hyper-aggressive" policy with no defensive incentives
3. Compare both against baselines and each other
4. Document whether defensive adaptation helps or hurts
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

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


class DefensiveRewardShaper:
    """Shapes rewards to encourage defensive play at high damage."""

    def __init__(self,
                 danger_threshold: float = 80.0,
                 critical_threshold: float = 120.0,
                 defense_mode: str = 'defensive'):
        """
        Args:
            danger_threshold: Damage % where defensive play starts
            critical_threshold: Damage % where survival is critical
            defense_mode: 'defensive' or 'aggressive' (control)
        """
        self.danger_threshold = danger_threshold
        self.critical_threshold = critical_threshold
        self.defense_mode = defense_mode

        # Tracking
        self.attack_counts_by_state = {'safe': 0, 'danger': 0, 'critical': 0}
        self.total_frames_by_state = {'safe': 0, 'danger': 0, 'critical': 0}

    def get_damage_state(self, damage: float) -> str:
        """Categorize damage into states."""
        if damage < self.danger_threshold:
            return 'safe'
        elif damage < self.critical_threshold:
            return 'danger'
        else:
            return 'critical'

    def compute_reward(self, game, player_idx: int,
                       prev_damage: List[float], prev_stocks: List[int],
                       action: int) -> float:
        """Compute reward with optional defensive shaping."""
        opponent_idx = 1 - player_idx
        fighter = game.fighters[player_idx]
        opponent = game.fighters[opponent_idx]

        my_damage = prev_damage[player_idx]
        my_state = self.get_damage_state(my_damage)

        self.total_frames_by_state[my_state] += 1
        is_attack = action >= 5  # Actions 5-9 are attacks
        if is_attack:
            self.attack_counts_by_state[my_state] += 1

        reward = 0.0

        # Stock changes (primary objectives)
        if opponent.stocks < prev_stocks[opponent_idx]:
            reward += 1.5
        if fighter.stocks < prev_stocks[player_idx]:
            reward -= 1.5

        # Damage dealt
        damage_dealt = opponent.damage_percent - prev_damage[opponent_idx]
        if damage_dealt > 0:
            # Base reward for damage
            if self.defense_mode == 'aggressive':
                # Aggressive: consistent damage reward regardless of our state
                reward += damage_dealt * 0.003
            else:
                # Defensive: modulated by our damage state
                if my_state == 'safe':
                    reward += damage_dealt * 0.004  # Encourage aggression when safe
                elif my_state == 'danger':
                    reward += damage_dealt * 0.002  # Less reward for risky attacks
                else:  # critical
                    reward += damage_dealt * 0.001  # Minimal reward - focus on survival

        # Damage taken
        damage_taken = fighter.damage_percent - prev_damage[player_idx]
        if damage_taken > 0:
            if self.defense_mode == 'aggressive':
                reward -= damage_taken * 0.002  # Uniform penalty
            else:
                # Defensive: increasing penalties as damage rises
                if my_state == 'safe':
                    reward -= damage_taken * 0.001
                elif my_state == 'danger':
                    reward -= damage_taken * 0.003
                else:  # critical
                    reward -= damage_taken * 0.006  # Heavy penalty for getting hit at high %

        # DEFENSIVE MODE SPECIFIC: Penalize attacking at high damage
        if self.defense_mode == 'defensive' and is_attack:
            if my_state == 'danger':
                reward -= 0.001  # Small penalty for aggression at danger
            elif my_state == 'critical':
                reward -= 0.003  # Larger penalty for aggression at critical

        # Encourage movement/evasion at high damage in defensive mode
        if self.defense_mode == 'defensive' and my_state in ['danger', 'critical']:
            if action in [1, 2, 3]:  # Left, Right, Jump (evasive actions)
                reward += 0.0005

        return reward

    def get_attack_rate_by_state(self) -> Dict[str, float]:
        """Get attack rate for each damage state."""
        result = {}
        for state in ['safe', 'danger', 'critical']:
            frames = self.total_frames_by_state[state]
            if frames > 0:
                result[state] = self.attack_counts_by_state[state] / frames
            else:
                result[state] = 0.0
        return result

    def reset_tracking(self):
        """Reset tracking counters."""
        self.attack_counts_by_state = {'safe': 0, 'danger': 0, 'critical': 0}
        self.total_frames_by_state = {'safe': 0, 'danger': 0, 'critical': 0}


class DefensiveExperiment:
    """Runs the defensive vs aggressive comparison experiment."""

    def __init__(self, device: str = 'cuda', output_dir: str = 'checkpoints/cycle5_defensive'):
        self.device = device
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create two policies
        self.defensive_policy = PolicyNetwork(
            obs_size=OBSERVATION_SIZE,
            action_size=ACTION_SPACE_SIZE,
            hidden_size=256
        )
        self.defensive_policy.to(device)

        self.aggressive_policy = PolicyNetwork(
            obs_size=OBSERVATION_SIZE,
            action_size=ACTION_SPACE_SIZE,
            hidden_size=256
        )
        self.aggressive_policy.to(device)

        # Create trainers
        train_config = TrainingConfig(
            learning_rate=2.5e-4,
            batch_size=64,
            epochs_per_update=4,
            gamma=0.99,
            gae_lambda=0.95,
            entropy_coef=0.015,
            value_coef=0.5
        )

        self.defensive_trainer = PPOTrainer(self.defensive_policy, train_config, device=device)
        self.aggressive_trainer = PPOTrainer(self.aggressive_policy, train_config, device=device)

        # Reward shapers
        self.defensive_shaper = DefensiveRewardShaper(defense_mode='defensive')
        self.aggressive_shaper = DefensiveRewardShaper(defense_mode='aggressive')

        # Opponents
        self.random_opp = RandomPolicy()
        self.aggressive_opp = AggressivePolicy()

    def load_base_checkpoint(self, path: str = None):
        """Load a base checkpoint to initialize both policies."""
        if path is None:
            candidates = [
                'checkpoints/cycle4_complete/best_vs_aggressive.npz',
                'checkpoints/cycle4/best_vs_aggressive.npz',
                'checkpoints/cycle4_extended/best_vs_aggressive.npz',
                'checkpoints/full_run/checkpoint_001000.npz',
            ]
            for p in candidates:
                if os.path.exists(p):
                    path = p
                    break

        if path and os.path.exists(path):
            self.defensive_policy.load(path)
            self.aggressive_policy.load(path)
            print(f"Loaded base checkpoint: {path}")
            return True
        return False

    def run_game(self, policy, shaper: DefensiveRewardShaper,
                 opponent, character: str = 'FLEX') -> Tuple[Episode, int, dict]:
        """Run a single game."""
        game = create_game(
            stage_name='battlefield',
            p1_character=character,
            p2_character='FLEX',
            stock_count=3
        )
        loop = GameLoop(game, headless=True)

        episode = Episode(player_id=0)
        prev_damage = [0.0, 0.0]
        prev_stocks = [3, 3]

        action_counts = np.zeros(ACTION_SPACE_SIZE)
        total_frames = 0
        damage_dealt = 0.0
        damage_taken = 0.0

        frame = 0
        max_frames = 10800

        while frame < max_frames:
            obs = [get_observation(game, i) for i in range(2)]

            action_p1, log_prob, value = policy.get_action(obs[0])
            action_p2, _, _ = opponent.get_action(obs[1])

            action_counts[action_p1] += 1
            total_frames += 1

            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            loop.step()
            frame += 1

            reward = shaper.compute_reward(game, 0, prev_damage, prev_stocks, action_p1)

            damage_dealt += max(0, game.fighters[1].damage_percent - prev_damage[1])
            damage_taken += max(0, game.fighters[0].damage_percent - prev_damage[0])

            prev_damage = [game.fighters[i].damage_percent for i in range(2)]
            prev_stocks = [game.fighters[i].stocks for i in range(2)]

            done = game.match_over

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
            episode.transitions[-1].reward += 0.5
        elif winner == 1:
            episode.transitions[-1].reward -= 0.5

        stats = {
            'winner': winner == 0,
            'frames': total_frames,
            'action_dist': (action_counts / max(1, total_frames)).tolist(),
            'damage_dealt': damage_dealt,
            'damage_taken': damage_taken
        }

        return episode, winner, stats

    def train_policy(self, policy, trainer, shaper: DefensiveRewardShaper,
                     name: str, num_games: int, games_per_update: int = 10) -> dict:
        """Train a policy with the specified shaper."""
        print(f"\n{'='*60}")
        print(f"TRAINING: {name}")
        print(f"Games: {num_games}")
        print(f"{'='*60}")

        start_time = time.time()
        wins_vs_agg = 0
        wins_vs_rand = 0
        games_vs_agg = 0
        games_vs_rand = 0

        episodes_buffer = []

        for game_idx in range(num_games):
            # Alternate opponents (70% aggressive, 30% random)
            if np.random.random() < 0.7:
                opponent = self.aggressive_opp
                is_aggressive = True
            else:
                opponent = self.random_opp
                is_aggressive = False

            episode, winner, _ = self.run_game(policy, shaper, opponent)
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
                trainer.update(episodes_buffer)
                episodes_buffer = []

            if (game_idx + 1) % 100 == 0:
                wr_agg = wins_vs_agg / max(1, games_vs_agg)
                wr_rand = wins_vs_rand / max(1, games_vs_rand)
                print(f"  [{game_idx+1:4d}/{num_games}] vs Agg: {wr_agg:.1%}, vs Rand: {wr_rand:.1%}")

        if episodes_buffer:
            trainer.update(episodes_buffer)

        elapsed = time.time() - start_time

        # Get behavior analysis
        attack_rates = shaper.get_attack_rate_by_state()

        result = {
            'name': name,
            'games': num_games,
            'elapsed': elapsed,
            'wins_vs_agg': wins_vs_agg,
            'games_vs_agg': games_vs_agg,
            'wins_vs_rand': wins_vs_rand,
            'games_vs_rand': games_vs_rand,
            'wr_vs_agg': wins_vs_agg / max(1, games_vs_agg),
            'wr_vs_rand': wins_vs_rand / max(1, games_vs_rand),
            'attack_rates_by_damage_state': attack_rates
        }

        print(f"\n{name} Training Complete:")
        print(f"  vs Aggressive: {result['wr_vs_agg']:.1%}")
        print(f"  vs Random: {result['wr_vs_rand']:.1%}")
        print(f"  Attack rates: Safe={attack_rates['safe']:.1%}, "
              f"Danger={attack_rates['danger']:.1%}, Critical={attack_rates['critical']:.1%}")

        return result

    def evaluate_policy(self, policy, name: str) -> dict:
        """Evaluate a policy against baselines."""
        print(f"\n--- Evaluating {name} ---")

        vs_random = evaluate_vs_random(policy, num_games=50)
        vs_aggressive = evaluate_vs_aggressive(policy, num_games=50)

        print(f"  vs Random: {vs_random['win_rate']:.1%}")
        print(f"  vs Aggressive: {vs_aggressive['win_rate']:.1%}")

        return {
            'name': name,
            'vs_random': vs_random,
            'vs_aggressive': vs_aggressive
        }

    def head_to_head(self, num_games: int = 100) -> dict:
        """Run head-to-head matches between defensive and aggressive policies."""
        print(f"\n{'='*60}")
        print(f"HEAD-TO-HEAD: Defensive vs Aggressive")
        print(f"{'='*60}")

        defensive_wins = 0

        for game_idx in range(num_games):
            game = create_game(
                stage_name='battlefield',
                p1_character='FLEX',
                p2_character='FLEX',
                stock_count=3
            )
            loop = GameLoop(game, headless=True)

            frame = 0
            while frame < 10800 and not game.match_over:
                obs1 = get_observation(game, 0)
                obs2 = get_observation(game, 1)

                a1, _, _ = self.defensive_policy.get_action(obs1)
                a2, _, _ = self.aggressive_policy.get_action(obs2)

                game.fighters[0].input = get_action_from_index(a1)
                game.fighters[1].input = get_action_from_index(a2)

                loop.step()
                frame += 1

            if game.winner == 0:
                defensive_wins += 1

            if (game_idx + 1) % 25 == 0:
                print(f"  [{game_idx+1:3d}/{num_games}] Defensive leads: {defensive_wins}/{game_idx+1}")

        result = {
            'total_games': num_games,
            'defensive_wins': defensive_wins,
            'aggressive_wins': num_games - defensive_wins,
            'defensive_win_rate': defensive_wins / num_games
        }

        print(f"\nHead-to-Head Result:")
        print(f"  Defensive: {defensive_wins} wins ({result['defensive_win_rate']:.1%})")
        print(f"  Aggressive: {num_games - defensive_wins} wins ({1 - result['defensive_win_rate']:.1%})")

        if result['defensive_win_rate'] > 0.55:
            print("  CONCLUSION: Defensive adaptation provides measurable advantage!")
        elif result['defensive_win_rate'] < 0.45:
            print("  CONCLUSION: Hyper-aggressive play is superior!")
        else:
            print("  CONCLUSION: No significant difference between strategies.")

        return result

    def run_experiment(self, training_games: int = 600) -> dict:
        """Run the complete defensive vs aggressive experiment."""
        print("\n" + "="*70)
        print("CYCLE 5: DEFENSIVE TRAINING EXPERIMENT")
        print("Testing: Does damage-state-aware defense improve win rates?")
        print("="*70)

        start_time = time.time()

        # Load base checkpoint
        self.load_base_checkpoint()

        # Initial evaluation
        print("\n--- Initial State (both policies identical) ---")
        initial_def = self.evaluate_policy(self.defensive_policy, "Initial")

        # Train defensive policy
        self.defensive_shaper.reset_tracking()
        defensive_training = self.train_policy(
            self.defensive_policy, self.defensive_trainer, self.defensive_shaper,
            "Defensive Policy", training_games
        )

        # Save defensive policy
        self.defensive_policy.save(f'{self.output_dir}/defensive_policy.npz')

        # Train aggressive policy
        self.aggressive_shaper.reset_tracking()
        aggressive_training = self.train_policy(
            self.aggressive_policy, self.aggressive_trainer, self.aggressive_shaper,
            "Aggressive Policy", training_games
        )

        # Save aggressive policy
        self.aggressive_policy.save(f'{self.output_dir}/aggressive_policy.npz')

        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATIONS")
        print("="*60)

        final_def = self.evaluate_policy(self.defensive_policy, "Defensive")
        final_agg = self.evaluate_policy(self.aggressive_policy, "Aggressive")

        # Head-to-head comparison
        h2h = self.head_to_head(100)

        total_elapsed = time.time() - start_time

        # Compile results
        report = {
            'experiment': 'Defensive vs Aggressive Training',
            'hypothesis': 'Damage-state-aware defense improves survival and win rate',
            'training_games': training_games,
            'total_time_seconds': total_elapsed,
            'initial_evaluation': {
                'vs_random': initial_def['vs_random']['win_rate'],
                'vs_aggressive': initial_def['vs_aggressive']['win_rate']
            },
            'defensive_policy': {
                'training': defensive_training,
                'final_evaluation': {
                    'vs_random': final_def['vs_random'],
                    'vs_aggressive': final_def['vs_aggressive']
                }
            },
            'aggressive_policy': {
                'training': aggressive_training,
                'final_evaluation': {
                    'vs_random': final_agg['vs_random'],
                    'vs_aggressive': final_agg['vs_aggressive']
                }
            },
            'head_to_head': h2h,
            'behavioral_analysis': {
                'defensive_attack_rates': defensive_training['attack_rates_by_damage_state'],
                'aggressive_attack_rates': aggressive_training['attack_rates_by_damage_state'],
                'shows_defensive_adaptation': (
                    defensive_training['attack_rates_by_damage_state']['critical'] <
                    defensive_training['attack_rates_by_damage_state']['safe'] * 0.8
                )
            },
            'conclusions': {}
        }

        # Draw conclusions
        def_vs_agg = final_def['vs_aggressive']['win_rate']
        agg_vs_agg = final_agg['vs_aggressive']['win_rate']

        if def_vs_agg > agg_vs_agg + 0.05:
            report['conclusions']['vs_aggressive_baseline'] = 'Defensive policy superior'
            report['conclusions']['defensive_helps_vs_aggressive'] = True
        elif agg_vs_agg > def_vs_agg + 0.05:
            report['conclusions']['vs_aggressive_baseline'] = 'Aggressive policy superior'
            report['conclusions']['defensive_helps_vs_aggressive'] = False
        else:
            report['conclusions']['vs_aggressive_baseline'] = 'No significant difference'
            report['conclusions']['defensive_helps_vs_aggressive'] = None

        if h2h['defensive_win_rate'] > 0.55:
            report['conclusions']['head_to_head'] = 'Defensive adaptation provides advantage'
        elif h2h['defensive_win_rate'] < 0.45:
            report['conclusions']['head_to_head'] = 'Hyper-aggressive play is superior'
        else:
            report['conclusions']['head_to_head'] = 'Strategies are evenly matched'

        # Save report
        report_path = f'{self.output_dir}/defensive_experiment_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
        print(f"\nTime: {total_elapsed/60:.1f} minutes")
        print(f"\nDefensive Policy Results:")
        print(f"  vs Random: {final_def['vs_random']['win_rate']:.1%}")
        print(f"  vs Aggressive: {final_def['vs_aggressive']['win_rate']:.1%}")
        print(f"  Attack at Safe: {defensive_training['attack_rates_by_damage_state']['safe']:.1%}")
        print(f"  Attack at Danger: {defensive_training['attack_rates_by_damage_state']['danger']:.1%}")
        print(f"  Attack at Critical: {defensive_training['attack_rates_by_damage_state']['critical']:.1%}")

        print(f"\nAggressive Policy Results:")
        print(f"  vs Random: {final_agg['vs_random']['win_rate']:.1%}")
        print(f"  vs Aggressive: {final_agg['vs_aggressive']['win_rate']:.1%}")
        print(f"  Attack at Safe: {aggressive_training['attack_rates_by_damage_state']['safe']:.1%}")
        print(f"  Attack at Danger: {aggressive_training['attack_rates_by_damage_state']['danger']:.1%}")
        print(f"  Attack at Critical: {aggressive_training['attack_rates_by_damage_state']['critical']:.1%}")

        print(f"\nHead-to-Head: Defensive wins {h2h['defensive_win_rate']:.1%}")
        print(f"\nCONCLUSION: {report['conclusions']['head_to_head']}")

        print(f"\nReport saved to: {report_path}")

        return report


def main():
    experiment = DefensiveExperiment(
        device='cuda',
        output_dir='checkpoints/cycle5_defensive'
    )

    report = experiment.run_experiment(training_games=600)
    return report


if __name__ == '__main__':
    main()
