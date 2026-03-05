#!/usr/bin/env python3
"""
Cycle 5: Meta-Game Analysis

This experiment tests whether the hyper-aggressive strategy is truly optimal
or just locally optimal by testing against mixed-strategy opponents.

Hypothesis:
- Current policy excels against pure Random and pure Aggressive opponents
- May fail against mixed-strategy opponents that alternate tactics
- May be exploitable by adaptive opponents

Methodology:
1. Create mixed-strategy opponents that alternate between aggression and defense
2. Create "counter-strategy" opponents that try to exploit hyper-aggression
3. Evaluate current policy against these varied opponents
4. Determine if the current strategy is globally or locally optimal
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
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


class MixedStrategyOpponent:
    """An opponent that alternates between strategies based on game state."""

    def __init__(self, name: str, strategy: str):
        """
        Args:
            name: Opponent name
            strategy: One of 'alternating', 'adaptive', 'punisher', 'turtle'
        """
        self.name = name
        self.strategy = strategy
        self.random_opp = RandomPolicy()
        self.aggressive_opp = AggressivePolicy()

        # State tracking
        self.frames_in_mode = 0
        self.current_mode = 'aggressive'
        self.opponent_attack_count = 0
        self.total_frames = 0

        # Strategy-specific params
        if strategy == 'alternating':
            self.switch_interval = 120  # Switch every 2 seconds (60fps)
        elif strategy == 'adaptive':
            self.aggression_threshold = 0.5  # Switch based on opponent aggression
        elif strategy == 'punisher':
            self.whiff_window = 30  # Frames to wait after opponent attacks
            self.frames_since_opp_attack = 100
        elif strategy == 'turtle':
            self.damage_threshold = 80  # Only attack when safe

    def reset(self):
        """Reset state for new game."""
        self.frames_in_mode = 0
        self.current_mode = 'aggressive'
        self.opponent_attack_count = 0
        self.total_frames = 0
        self.frames_since_opp_attack = 100

    def observe_opponent_action(self, opponent_action: int):
        """Track opponent's actions for adaptive strategies."""
        self.total_frames += 1
        if opponent_action >= 5:  # Attack action
            self.opponent_attack_count += 1
            self.frames_since_opp_attack = 0
        else:
            self.frames_since_opp_attack += 1

    def get_action(self, obs: np.ndarray, own_damage: float = 0,
                   opponent_damage: float = 0) -> Tuple[int, float, float]:
        """Get action based on mixed strategy."""
        self.frames_in_mode += 1

        # Determine mode based on strategy
        if self.strategy == 'alternating':
            if self.frames_in_mode >= self.switch_interval:
                self.current_mode = 'defensive' if self.current_mode == 'aggressive' else 'aggressive'
                self.frames_in_mode = 0

        elif self.strategy == 'adaptive':
            if self.total_frames > 60:
                opp_aggression = self.opponent_attack_count / self.total_frames
                if opp_aggression > self.aggression_threshold:
                    self.current_mode = 'defensive'  # Play defensive vs aggression
                else:
                    self.current_mode = 'aggressive'  # Punish passivity

        elif self.strategy == 'punisher':
            # Wait for opponent to whiff, then punish
            if self.frames_since_opp_attack < self.whiff_window:
                self.current_mode = 'defensive'  # Wait
            else:
                self.current_mode = 'aggressive'  # Punish

        elif self.strategy == 'turtle':
            # Only attack when we're safe
            if own_damage > self.damage_threshold:
                self.current_mode = 'defensive'
            else:
                self.current_mode = 'aggressive'

        # Execute based on current mode
        if self.current_mode == 'aggressive':
            return self.aggressive_opp.get_action(obs)
        else:
            # Defensive mode: prefer movement and blocking
            action_probs = np.array([
                0.15,  # Nothing
                0.25,  # Left
                0.25,  # Right
                0.20,  # Jump
                0.05,  # Down
                0.02,  # Attack
                0.02,  # L+Atk
                0.02,  # R+Atk
                0.02,  # U+Atk
                0.02   # D+Atk
            ])
            action = np.random.choice(ACTION_SPACE_SIZE, p=action_probs)
            return action, 0.0, 0.0


class CounterAggressiveOpponent:
    """An opponent specifically designed to counter hyper-aggressive play."""

    def __init__(self):
        self.name = "CounterAggressive"
        self.frames_since_dodge = 0
        self.opponent_attack_streak = 0

    def reset(self):
        self.frames_since_dodge = 0
        self.opponent_attack_streak = 0

    def observe_opponent_action(self, opponent_action: int):
        if opponent_action >= 5:
            self.opponent_attack_streak += 1
        else:
            self.opponent_attack_streak = 0
        self.frames_since_dodge += 1

    def get_action(self, obs: np.ndarray, own_damage: float = 0,
                   opponent_damage: float = 0) -> Tuple[int, float, float]:
        """Counter hyper-aggression with spacing and punishes."""

        # If opponent is on an attack streak, evade
        if self.opponent_attack_streak > 2:
            # Jump or move away
            action = np.random.choice([1, 2, 3], p=[0.3, 0.3, 0.4])  # L, R, Jump
            self.frames_since_dodge = 0
            return action, 0.0, 0.0

        # If we just dodged, punish
        if self.frames_since_dodge < 15:
            # Counter attack
            action = np.random.choice([5, 6, 7, 8], p=[0.4, 0.2, 0.2, 0.2])
            return action, 0.0, 0.0

        # Default: spacing and poking
        action_probs = np.array([
            0.05,  # Nothing
            0.20,  # Left
            0.20,  # Right
            0.15,  # Jump
            0.05,  # Down
            0.15,  # Attack
            0.05,  # L+Atk
            0.05,  # R+Atk
            0.05,  # U+Atk
            0.05   # D+Atk
        ])
        action = np.random.choice(ACTION_SPACE_SIZE, p=action_probs)
        return action, 0.0, 0.0


class MetaGameExperiment:
    """Runs the meta-game analysis experiment."""

    def __init__(self, device: str = 'cuda', output_dir: str = 'checkpoints/cycle5_metagame'):
        self.device = device
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load the trained policy
        self.policy = PolicyNetwork(
            obs_size=OBSERVATION_SIZE,
            action_size=ACTION_SPACE_SIZE,
            hidden_size=256
        )
        self.policy.to(device)

        # Create mixed-strategy opponents
        self.opponents = {
            'random': RandomPolicy(),
            'aggressive': AggressivePolicy(),
            'alternating': MixedStrategyOpponent('Alternating', 'alternating'),
            'adaptive': MixedStrategyOpponent('Adaptive', 'adaptive'),
            'punisher': MixedStrategyOpponent('Punisher', 'punisher'),
            'turtle': MixedStrategyOpponent('Turtle', 'turtle'),
            'counter_aggressive': CounterAggressiveOpponent()
        }

    def load_policy(self, path: str = None):
        """Load trained policy."""
        if path is None:
            candidates = [
                'checkpoints/cycle4_complete/best_vs_aggressive.npz',
                'checkpoints/cycle4/best_vs_aggressive.npz',
                'checkpoints/cycle5_defensive/defensive_policy.npz',
                'checkpoints/full_run/checkpoint_001000.npz',
            ]
            for p in candidates:
                if os.path.exists(p):
                    path = p
                    break

        if path and os.path.exists(path):
            self.policy.load(path)
            print(f"Loaded policy: {path}")
            return path
        return None

    def evaluate_vs_opponent(self, opponent, opponent_name: str,
                             num_games: int = 50) -> dict:
        """Evaluate policy against a specific opponent."""
        wins = 0
        total_damage_dealt = 0.0
        total_damage_taken = 0.0
        total_frames = 0

        for game_idx in range(num_games):
            game = create_game(
                stage_name='battlefield',
                p1_character='FLEX',
                p2_character='FLEX',
                stock_count=3
            )
            loop = GameLoop(game, headless=True)

            # Reset opponent state
            if hasattr(opponent, 'reset'):
                opponent.reset()

            frame = 0
            max_frames = 10800
            prev_damage = [0.0, 0.0]

            while frame < max_frames and not game.match_over:
                obs1 = get_observation(game, 0)
                obs2 = get_observation(game, 1)

                action_p1, _, _ = self.policy.get_action(obs1)

                # Get opponent action (handle mixed strategy opponents)
                if hasattr(opponent, 'observe_opponent_action'):
                    # Let opponent observe our last action
                    if frame > 0:
                        opponent.observe_opponent_action(action_p1)

                if isinstance(opponent, (MixedStrategyOpponent, CounterAggressiveOpponent)):
                    action_p2, _, _ = opponent.get_action(
                        obs2,
                        own_damage=game.fighters[1].damage_percent,
                        opponent_damage=game.fighters[0].damage_percent
                    )
                else:
                    action_p2, _, _ = opponent.get_action(obs2)

                game.fighters[0].input = get_action_from_index(action_p1)
                game.fighters[1].input = get_action_from_index(action_p2)

                loop.step()
                frame += 1

                # Track damage
                total_damage_dealt += max(0, game.fighters[1].damage_percent - prev_damage[1])
                total_damage_taken += max(0, game.fighters[0].damage_percent - prev_damage[0])
                prev_damage = [game.fighters[i].damage_percent for i in range(2)]

            total_frames += frame
            if game.winner == 0:
                wins += 1

        result = {
            'opponent': opponent_name,
            'games': num_games,
            'wins': wins,
            'win_rate': wins / num_games,
            'avg_damage_dealt': total_damage_dealt / num_games,
            'avg_damage_taken': total_damage_taken / num_games,
            'avg_game_length': total_frames / num_games
        }

        return result

    def run_full_evaluation(self, games_per_opponent: int = 60) -> dict:
        """Evaluate against all opponent types."""
        print(f"\n{'='*60}")
        print("META-GAME ANALYSIS: Testing Against Mixed Strategies")
        print(f"{'='*60}")

        results = {}

        for opp_name, opponent in self.opponents.items():
            print(f"\n--- vs {opp_name.title()} ---")
            result = self.evaluate_vs_opponent(opponent, opp_name, games_per_opponent)
            results[opp_name] = result

            print(f"  Win rate: {result['win_rate']:.1%}")
            print(f"  Avg damage dealt: {result['avg_damage_dealt']:.1f}")
            print(f"  Avg damage taken: {result['avg_damage_taken']:.1f}")

        return results

    def analyze_exploitability(self, results: dict) -> dict:
        """Analyze if the policy is exploitable by specific strategies."""
        print(f"\n{'='*60}")
        print("EXPLOITABILITY ANALYSIS")
        print(f"{'='*60}")

        baseline_wr = results['aggressive']['win_rate']
        analysis = {
            'baseline_win_rate': baseline_wr,
            'exploits': [],
            'robust_against': [],
            'exploitability_score': 0.0
        }

        for opp_name, result in results.items():
            if opp_name in ['random', 'aggressive']:
                continue

            wr = result['win_rate']
            diff = baseline_wr - wr

            if diff > 0.15:  # More than 15% drop
                analysis['exploits'].append({
                    'opponent': opp_name,
                    'win_rate': wr,
                    'drop': diff
                })
                print(f"  VULNERABLE to {opp_name}: {wr:.1%} (drop of {diff:.1%})")
            elif wr >= baseline_wr - 0.05:
                analysis['robust_against'].append({
                    'opponent': opp_name,
                    'win_rate': wr
                })
                print(f"  ROBUST vs {opp_name}: {wr:.1%}")
            else:
                print(f"  NEUTRAL vs {opp_name}: {wr:.1%}")

        # Calculate exploitability score
        if analysis['exploits']:
            avg_drop = np.mean([e['drop'] for e in analysis['exploits']])
            analysis['exploitability_score'] = avg_drop
            print(f"\n  Exploitability score: {avg_drop:.1%}")
        else:
            print(f"\n  Policy is ROBUST - no significant exploits found!")

        return analysis

    def test_strategy_rock_paper_scissors(self) -> dict:
        """Test if there's a rock-paper-scissors dynamic between strategies."""
        print(f"\n{'='*60}")
        print("STRATEGY INTERACTIONS")
        print(f"{'='*60}")

        # For this we'd need multiple trained policies
        # For now, just test our policy's matchup spread

        matchups = {}
        for opp_name in self.opponents.keys():
            result = self.evaluate_vs_opponent(
                self.opponents[opp_name], opp_name, num_games=30
            )
            matchups[opp_name] = result['win_rate']

        # Find counters
        sorted_matchups = sorted(matchups.items(), key=lambda x: x[1])

        print("\nMatchup Spread (worst to best):")
        for opp, wr in sorted_matchups:
            indicator = "WEAK" if wr < 0.45 else "STRONG" if wr > 0.55 else "EVEN"
            print(f"  {opp:20s}: {wr:.1%} [{indicator}]")

        return matchups

    def run_experiment(self) -> dict:
        """Run the complete meta-game analysis."""
        print("\n" + "="*70)
        print("CYCLE 5: META-GAME ANALYSIS")
        print("Testing if hyper-aggressive strategy is globally optimal")
        print("="*70)

        start_time = time.time()

        # Load policy
        loaded_path = self.load_policy()

        # Full evaluation against all opponents
        evaluation_results = self.run_full_evaluation(games_per_opponent=60)

        # Analyze exploitability
        exploitability = self.analyze_exploitability(evaluation_results)

        # Test strategy interactions
        matchup_spread = self.test_strategy_rock_paper_scissors()

        total_elapsed = time.time() - start_time

        # Compile report
        report = {
            'experiment': 'Meta-Game Analysis',
            'hypothesis': 'Hyper-aggressive may be locally but not globally optimal',
            'policy_path': loaded_path,
            'total_time_seconds': total_elapsed,
            'evaluation_results': evaluation_results,
            'exploitability_analysis': exploitability,
            'matchup_spread': matchup_spread,
            'conclusions': {}
        }

        # Draw conclusions
        baseline_wr = evaluation_results['aggressive']['win_rate']
        min_wr = min(r['win_rate'] for r in evaluation_results.values())
        max_drop = baseline_wr - min_wr

        if max_drop > 0.20:
            report['conclusions']['optimality'] = 'LOCALLY OPTIMAL - significant exploits exist'
            report['conclusions']['recommendation'] = 'Train against mixed-strategy opponents'
        elif max_drop > 0.10:
            report['conclusions']['optimality'] = 'MODERATELY ROBUST - some weaknesses'
            report['conclusions']['recommendation'] = 'Consider adding counter-strategy training'
        else:
            report['conclusions']['optimality'] = 'GLOBALLY OPTIMAL - robust against all strategies'
            report['conclusions']['recommendation'] = 'Current strategy is optimal'

        report['conclusions']['worst_matchup'] = min(
            evaluation_results.items(),
            key=lambda x: x[1]['win_rate']
        )[0]
        report['conclusions']['worst_win_rate'] = min_wr

        # Save report
        report_path = f'{self.output_dir}/metagame_analysis.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("META-GAME ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nTime: {total_elapsed/60:.1f} minutes")
        print(f"\nBaseline (vs Aggressive): {baseline_wr:.1%}")
        print(f"Worst matchup: {report['conclusions']['worst_matchup']} ({min_wr:.1%})")
        print(f"Maximum drop: {max_drop:.1%}")
        print(f"\nCONCLUSION: {report['conclusions']['optimality']}")
        print(f"RECOMMENDATION: {report['conclusions']['recommendation']}")

        if exploitability['exploits']:
            print(f"\nVulnerable to:")
            for exp in exploitability['exploits']:
                print(f"  - {exp['opponent']}: {exp['win_rate']:.1%}")

        print(f"\nReport saved to: {report_path}")

        return report


def main():
    experiment = MetaGameExperiment(
        device='cuda',
        output_dir='checkpoints/cycle5_metagame'
    )

    report = experiment.run_experiment()
    return report


if __name__ == '__main__':
    main()
