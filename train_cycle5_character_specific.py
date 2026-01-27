#!/usr/bin/env python3
"""
Cycle 5: Character-Specific Policy Training

This experiment trains separate policies for each character (FLEX, RUSH, ZONE, GRAB)
to see if specialized training outperforms a universal policy.

Hypothesis:
- Each character has unique attributes (speed, range, throws)
- Character-specific training should exploit these attributes
- RUSH should learn to use speed advantage
- ZONE should learn to use range advantage
- GRAB should learn throw-based strategies

Methodology:
1. Train 4 separate policies, one per character
2. Compare against universal policy trained on all characters
3. Evaluate both same-character and cross-character matchups
4. Document which characters benefit most from specialization
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

CHARACTERS = ['FLEX', 'RUSH', 'ZONE', 'GRAB']

# Character-specific reward bonuses based on their unique attributes
CHARACTER_BONUSES = {
    'FLEX': {
        'description': 'Balanced fighter - no specialization',
        'bonuses': {}
    },
    'RUSH': {
        'description': 'Speed demon - rewards quick approaches and combos',
        'bonuses': {
            'approach_bonus': 0.002,  # Bonus for closing distance
            'combo_bonus': 0.003,     # Extra combo bonus
            'movement_penalty': -0.001 # Penalize staying still
        }
    },
    'ZONE': {
        'description': 'Range specialist - rewards keeping distance and poking',
        'bonuses': {
            'range_damage_bonus': 0.003,  # Bonus for damage at range
            'close_penalty': -0.001,      # Penalty for getting too close
            'spacing_bonus': 0.001        # Bonus for maintaining optimal distance
        }
    },
    'GRAB': {
        'description': 'Grappler - rewards throws and close combat',
        'bonuses': {
            'close_bonus': 0.002,      # Bonus for staying close
            'throw_bonus': 0.005,      # Extra reward for throws
            'approach_bonus': 0.001    # Bonus for approaching
        }
    }
}


class CharacterRewardShaper:
    """Shapes rewards based on character-specific attributes."""

    def __init__(self, character: str):
        self.character = character
        self.bonuses = CHARACTER_BONUSES[character]['bonuses']

        # Tracking
        self.damage_by_distance = {'close': 0.0, 'mid': 0.0, 'far': 0.0}
        self.approach_sequences = 0
        self.retreat_sequences = 0

    def get_distance_category(self, distance: float) -> str:
        """Categorize distance between fighters."""
        if distance < 100:
            return 'close'
        elif distance < 200:
            return 'mid'
        else:
            return 'far'

    def compute_reward(self, game, player_idx: int,
                       prev_damage: List[float], prev_stocks: List[int],
                       prev_positions: List[Tuple[float, float]],
                       action: int) -> float:
        """Compute character-specific reward."""
        opponent_idx = 1 - player_idx
        fighter = game.fighters[player_idx]
        opponent = game.fighters[opponent_idx]

        reward = 0.0

        # Stock changes (primary objectives - same for all characters)
        if opponent.stocks < prev_stocks[opponent_idx]:
            reward += 1.5
        if fighter.stocks < prev_stocks[player_idx]:
            reward -= 1.5

        # Damage dealt
        damage_dealt = opponent.damage_percent - prev_damage[opponent_idx]
        if damage_dealt > 0:
            reward += damage_dealt * 0.003

            # Track damage by distance for analysis
            if prev_positions:
                distance = abs(prev_positions[0][0] - prev_positions[1][0])
                dist_cat = self.get_distance_category(distance)
                self.damage_by_distance[dist_cat] += damage_dealt

                # Character-specific damage bonuses
                if self.character == 'ZONE' and 'range_damage_bonus' in self.bonuses:
                    if dist_cat in ['mid', 'far']:
                        reward += damage_dealt * self.bonuses['range_damage_bonus']

                if self.character == 'GRAB' and 'throw_bonus' in self.bonuses:
                    if dist_cat == 'close':
                        reward += damage_dealt * self.bonuses['throw_bonus']

        # Damage taken
        damage_taken = fighter.damage_percent - prev_damage[player_idx]
        if damage_taken > 0:
            reward -= damage_taken * 0.002

        # Position-based bonuses
        if hasattr(fighter, 'position') and hasattr(opponent, 'position'):
            distance = abs(fighter.position.x - opponent.position.x)
            dist_cat = self.get_distance_category(distance)

            # Movement tracking for approach detection
            if prev_positions:
                prev_dist = abs(prev_positions[0][0] - prev_positions[1][0])
                curr_dist = distance

                # RUSH bonus: approaching
                if self.character == 'RUSH' and 'approach_bonus' in self.bonuses:
                    if curr_dist < prev_dist - 5:  # Getting closer
                        reward += self.bonuses['approach_bonus']
                        self.approach_sequences += 1

                # GRAB bonus: staying close
                if self.character == 'GRAB' and 'close_bonus' in self.bonuses:
                    if dist_cat == 'close':
                        reward += self.bonuses['close_bonus']

                # ZONE spacing bonus
                if self.character == 'ZONE' and 'spacing_bonus' in self.bonuses:
                    if dist_cat == 'mid':  # Optimal zone distance
                        reward += self.bonuses['spacing_bonus']

                # ZONE close penalty
                if self.character == 'ZONE' and 'close_penalty' in self.bonuses:
                    if dist_cat == 'close':
                        reward += self.bonuses['close_penalty']

        # RUSH movement penalty (penalize standing still)
        if self.character == 'RUSH' and 'movement_penalty' in self.bonuses:
            if action == 0:  # Nothing
                reward += self.bonuses['movement_penalty']

        return reward

    def get_stats(self) -> Dict:
        """Get character-specific behavior stats."""
        total_damage = sum(self.damage_by_distance.values())
        return {
            'character': self.character,
            'damage_by_distance': self.damage_by_distance,
            'approach_sequences': self.approach_sequences,
            'retreat_sequences': self.retreat_sequences,
            'close_damage_ratio': self.damage_by_distance['close'] / max(1, total_damage),
            'mid_damage_ratio': self.damage_by_distance['mid'] / max(1, total_damage),
            'far_damage_ratio': self.damage_by_distance['far'] / max(1, total_damage)
        }


class CharacterSpecificExperiment:
    """Runs character-specific policy training experiment."""

    def __init__(self, device: str = 'cuda', output_dir: str = 'checkpoints/cycle5_char_specific'):
        self.device = device
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create policy for each character
        self.policies = {}
        self.trainers = {}
        self.shapers = {}

        for char in CHARACTERS:
            policy = PolicyNetwork(
                obs_size=OBSERVATION_SIZE,
                action_size=ACTION_SPACE_SIZE,
                hidden_size=256
            )
            policy.to(device)
            self.policies[char] = policy

            train_config = TrainingConfig(
                learning_rate=2.5e-4,
                batch_size=64,
                epochs_per_update=4,
                gamma=0.99,
                gae_lambda=0.95,
                entropy_coef=0.015,
                value_coef=0.5
            )
            self.trainers[char] = PPOTrainer(policy, train_config, device=device)
            self.shapers[char] = CharacterRewardShaper(char)

        # Universal policy for comparison
        self.universal_policy = PolicyNetwork(
            obs_size=OBSERVATION_SIZE,
            action_size=ACTION_SPACE_SIZE,
            hidden_size=256
        )
        self.universal_policy.to(device)
        self.universal_trainer = PPOTrainer(
            self.universal_policy,
            TrainingConfig(learning_rate=2.5e-4, batch_size=64, epochs_per_update=4),
            device=device
        )

        # Opponents
        self.random_opp = RandomPolicy()
        self.aggressive_opp = AggressivePolicy()

    def load_base_checkpoint(self, path: str = None):
        """Load base checkpoint for all policies."""
        if path is None:
            candidates = [
                'checkpoints/cycle4_complete/best_vs_aggressive.npz',
                'checkpoints/cycle4/best_vs_aggressive.npz',
                'checkpoints/full_run/checkpoint_001000.npz',
            ]
            for p in candidates:
                if os.path.exists(p):
                    path = p
                    break

        if path and os.path.exists(path):
            for char in CHARACTERS:
                self.policies[char].load(path)
            self.universal_policy.load(path)
            print(f"Loaded base checkpoint: {path}")
            return True
        return False

    def run_game(self, policy, shaper: CharacterRewardShaper,
                 opponent, character: str) -> Tuple[Episode, int, dict]:
        """Run a single game with character-specific rewards."""
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
        prev_positions = None

        action_counts = np.zeros(ACTION_SPACE_SIZE)
        total_frames = 0

        frame = 0
        max_frames = 10800

        while frame < max_frames:
            obs = [get_observation(game, i) for i in range(2)]

            action_p1, log_prob, value = policy.get_action(obs[0])
            action_p2, _, _ = opponent.get_action(obs[1])

            action_counts[action_p1] += 1
            total_frames += 1

            # Get positions
            current_positions = []
            for i in range(2):
                if hasattr(game.fighters[i], 'position'):
                    current_positions.append((game.fighters[i].position.x,
                                              game.fighters[i].position.y))
                else:
                    current_positions.append((400, 300))

            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            loop.step()
            frame += 1

            reward = shaper.compute_reward(
                game, 0, prev_damage, prev_stocks,
                prev_positions if prev_positions else current_positions,
                action_p1
            )

            prev_damage = [game.fighters[i].damage_percent for i in range(2)]
            prev_stocks = [game.fighters[i].stocks for i in range(2)]
            prev_positions = current_positions

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

        stats = {
            'winner': winner == 0,
            'frames': total_frames,
            'action_dist': (action_counts / max(1, total_frames)).tolist()
        }

        return episode, winner, stats

    def train_character_policy(self, character: str, num_games: int) -> dict:
        """Train a character-specific policy."""
        print(f"\n{'='*60}")
        print(f"TRAINING: {character} Specialist")
        print(f"Strategy: {CHARACTER_BONUSES[character]['description']}")
        print(f"{'='*60}")

        policy = self.policies[character]
        trainer = self.trainers[character]
        shaper = self.shapers[character]

        start_time = time.time()
        wins = 0
        episodes_buffer = []
        games_per_update = 10

        for game_idx in range(num_games):
            # Mix opponents
            opponent = self.aggressive_opp if np.random.random() < 0.6 else self.random_opp

            episode, winner, _ = self.run_game(policy, shaper, opponent, character)
            episodes_buffer.append(episode)

            if winner == 0:
                wins += 1

            if len(episodes_buffer) >= games_per_update:
                trainer.update(episodes_buffer)
                episodes_buffer = []

            if (game_idx + 1) % 100 == 0:
                print(f"  [{game_idx+1:4d}/{num_games}] Win rate: {wins/(game_idx+1):.1%}")

        if episodes_buffer:
            trainer.update(episodes_buffer)

        elapsed = time.time() - start_time

        # Save checkpoint
        policy.save(f'{self.output_dir}/{character.lower()}_specialist.npz')

        # Get behavior stats
        char_stats = shaper.get_stats()

        result = {
            'character': character,
            'games': num_games,
            'wins': wins,
            'win_rate': wins / num_games,
            'elapsed': elapsed,
            'behavior_stats': char_stats
        }

        print(f"\n{character} Training Complete:")
        print(f"  Win rate: {result['win_rate']:.1%}")
        print(f"  Damage at close: {char_stats['close_damage_ratio']:.1%}")
        print(f"  Damage at mid: {char_stats['mid_damage_ratio']:.1%}")
        print(f"  Damage at far: {char_stats['far_damage_ratio']:.1%}")

        return result

    def train_universal_policy(self, num_games: int) -> dict:
        """Train a universal policy on all characters."""
        print(f"\n{'='*60}")
        print(f"TRAINING: Universal Policy (all characters)")
        print(f"{'='*60}")

        start_time = time.time()
        wins_by_char = {c: 0 for c in CHARACTERS}
        games_by_char = {c: 0 for c in CHARACTERS}
        episodes_buffer = []
        games_per_update = 10

        # Generic shaper (no character bonuses)
        shaper = CharacterRewardShaper('FLEX')

        for game_idx in range(num_games):
            # Rotate through characters
            character = CHARACTERS[game_idx % len(CHARACTERS)]
            opponent = self.aggressive_opp if np.random.random() < 0.6 else self.random_opp

            episode, winner, _ = self.run_game(
                self.universal_policy, shaper, opponent, character
            )
            episodes_buffer.append(episode)

            games_by_char[character] += 1
            if winner == 0:
                wins_by_char[character] += 1

            if len(episodes_buffer) >= games_per_update:
                self.universal_trainer.update(episodes_buffer)
                episodes_buffer = []

            if (game_idx + 1) % 200 == 0:
                total_wins = sum(wins_by_char.values())
                print(f"  [{game_idx+1:4d}/{num_games}] Overall: {total_wins/(game_idx+1):.1%}")

        if episodes_buffer:
            self.universal_trainer.update(episodes_buffer)

        elapsed = time.time() - start_time

        # Save
        self.universal_policy.save(f'{self.output_dir}/universal_policy.npz')

        result = {
            'games': num_games,
            'elapsed': elapsed,
            'wins_by_character': wins_by_char,
            'games_by_character': games_by_char,
            'win_rates_by_character': {
                c: wins_by_char[c] / max(1, games_by_char[c])
                for c in CHARACTERS
            }
        }

        print(f"\nUniversal Training Complete:")
        for char in CHARACTERS:
            wr = result['win_rates_by_character'][char]
            print(f"  {char}: {wr:.1%}")

        return result

    def evaluate_all_policies(self, games_per_eval: int = 30) -> dict:
        """Evaluate all policies against baselines."""
        print(f"\n{'='*60}")
        print("EVALUATING ALL POLICIES")
        print(f"{'='*60}")

        results = {}

        # Evaluate character-specific policies
        for char in CHARACTERS:
            print(f"\n--- {char} Specialist ---")

            vs_random = evaluate_vs_random(
                self.policies[char], num_games=games_per_eval, character=char
            )
            vs_aggressive = evaluate_vs_aggressive(
                self.policies[char], num_games=games_per_eval, character=char
            )

            results[f'{char}_specialist'] = {
                'character': char,
                'vs_random': vs_random,
                'vs_aggressive': vs_aggressive
            }

            print(f"  vs Random: {vs_random['win_rate']:.1%}")
            print(f"  vs Aggressive: {vs_aggressive['win_rate']:.1%}")

        # Evaluate universal policy on each character
        print(f"\n--- Universal Policy ---")
        for char in CHARACTERS:
            vs_random = evaluate_vs_random(
                self.universal_policy, num_games=games_per_eval, character=char
            )
            vs_aggressive = evaluate_vs_aggressive(
                self.universal_policy, num_games=games_per_eval, character=char
            )

            results[f'universal_{char}'] = {
                'character': char,
                'vs_random': vs_random,
                'vs_aggressive': vs_aggressive
            }

            print(f"  {char} vs Random: {vs_random['win_rate']:.1%}, vs Aggressive: {vs_aggressive['win_rate']:.1%}")

        return results

    def run_cross_character_tournament(self, games_per_matchup: int = 20) -> dict:
        """Run a tournament between all character-specific policies."""
        print(f"\n{'='*60}")
        print("CROSS-CHARACTER TOURNAMENT")
        print(f"{'='*60}")

        matchup_matrix = {c1: {c2: 0.0 for c2 in CHARACTERS} for c1 in CHARACTERS}

        for char1 in CHARACTERS:
            for char2 in CHARACTERS:
                wins = 0

                for _ in range(games_per_matchup):
                    game = create_game(
                        stage_name='battlefield',
                        p1_character=char1,
                        p2_character=char2,
                        stock_count=3
                    )
                    loop = GameLoop(game, headless=True)

                    frame = 0
                    while frame < 10800 and not game.match_over:
                        obs1 = get_observation(game, 0)
                        obs2 = get_observation(game, 1)

                        a1, _, _ = self.policies[char1].get_action(obs1)
                        a2, _, _ = self.policies[char2].get_action(obs2)

                        game.fighters[0].input = get_action_from_index(a1)
                        game.fighters[1].input = get_action_from_index(a2)

                        loop.step()
                        frame += 1

                    if game.winner == 0:
                        wins += 1

                matchup_matrix[char1][char2] = wins / games_per_matchup

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

        return matchup_matrix

    def run_experiment(self, games_per_character: int = 400) -> dict:
        """Run the complete character-specific training experiment."""
        print("\n" + "="*70)
        print("CYCLE 5: CHARACTER-SPECIFIC POLICY TRAINING")
        print("Testing: Does specialized training outperform universal training?")
        print("="*70)

        start_time = time.time()

        # Load base checkpoint
        self.load_base_checkpoint()

        # Train character-specific policies
        char_training_results = {}
        for char in CHARACTERS:
            char_training_results[char] = self.train_character_policy(
                char, games_per_character
            )

        # Train universal policy with same total games
        total_games = games_per_character * len(CHARACTERS)
        universal_training = self.train_universal_policy(total_games)

        # Evaluate all policies
        evaluation_results = self.evaluate_all_policies(games_per_eval=40)

        # Run tournament
        tournament_matrix = self.run_cross_character_tournament(games_per_matchup=25)

        total_elapsed = time.time() - start_time

        # Compare specialist vs universal
        comparisons = {}
        for char in CHARACTERS:
            spec_key = f'{char}_specialist'
            univ_key = f'universal_{char}'

            spec_vs_agg = evaluation_results[spec_key]['vs_aggressive']['win_rate']
            univ_vs_agg = evaluation_results[univ_key]['vs_aggressive']['win_rate']

            comparisons[char] = {
                'specialist_vs_aggressive': spec_vs_agg,
                'universal_vs_aggressive': univ_vs_agg,
                'specialist_advantage': spec_vs_agg - univ_vs_agg,
                'specialist_better': spec_vs_agg > univ_vs_agg
            }

        # Compile report
        report = {
            'experiment': 'Character-Specific Policy Training',
            'hypothesis': 'Specialized training outperforms universal training',
            'games_per_character': games_per_character,
            'total_games': total_games * 2,  # Specialist + Universal
            'total_time_seconds': total_elapsed,
            'character_training': char_training_results,
            'universal_training': universal_training,
            'evaluation_results': evaluation_results,
            'tournament_matrix': tournament_matrix,
            'specialist_vs_universal_comparison': comparisons,
            'conclusions': {}
        }

        # Draw conclusions
        specialist_wins = sum(1 for c in CHARACTERS if comparisons[c]['specialist_better'])

        if specialist_wins >= 3:
            report['conclusions']['overall'] = 'Character specialization provides clear advantage'
        elif specialist_wins >= 2:
            report['conclusions']['overall'] = 'Mixed results - some characters benefit from specialization'
        else:
            report['conclusions']['overall'] = 'Universal training is generally superior'

        # Find best character
        best_char = max(CHARACTERS, key=lambda c: evaluation_results[f'{c}_specialist']['vs_aggressive']['win_rate'])
        report['conclusions']['best_specialist'] = best_char
        report['conclusions']['best_specialist_wr'] = evaluation_results[f'{best_char}_specialist']['vs_aggressive']['win_rate']

        # Save report
        report_path = f'{self.output_dir}/character_specific_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
        print(f"\nTotal time: {total_elapsed/60:.1f} minutes")

        print("\nSpecialist vs Universal Comparison:")
        for char in CHARACTERS:
            comp = comparisons[char]
            better = "SPECIALIST" if comp['specialist_better'] else "UNIVERSAL"
            print(f"  {char}: Specialist {comp['specialist_vs_aggressive']:.1%} vs "
                  f"Universal {comp['universal_vs_aggressive']:.1%} -> {better}")

        print(f"\nBest specialist: {best_char} ({report['conclusions']['best_specialist_wr']:.1%})")
        print(f"\nCONCLUSION: {report['conclusions']['overall']}")

        print(f"\nReport saved to: {report_path}")

        return report


def main():
    experiment = CharacterSpecificExperiment(
        device='cuda',
        output_dir='checkpoints/cycle5_char_specific'
    )

    report = experiment.run_experiment(games_per_character=400)
    return report


if __name__ == '__main__':
    main()
