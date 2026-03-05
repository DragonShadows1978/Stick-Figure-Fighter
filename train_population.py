#!/usr/bin/env python3
"""
Population-Based Training for Strategy Diversity.

Creates a population of policies with different reward emphases:
1. BALANCED - Standard rewards
2. DEFENSIVE - Higher damage-taken penalties
3. AGGRESSIVE - Higher damage-dealt rewards
4. COUNTER - Focus on punish timing
5. SPACING - Focus on positional control

After training, policies compete in a tournament to measure diversity.
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
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


@dataclass
class PolicyProfile:
    """Configuration for a specific policy profile."""
    name: str
    description: str

    # Reward multipliers (relative to base)
    damage_dealt_mult: float = 1.0
    damage_taken_mult: float = 1.0
    stock_mult: float = 1.0
    spacing_mult: float = 1.0
    punish_mult: float = 1.0
    survival_mult: float = 1.0


# Define the population profiles
POPULATION_PROFILES = [
    PolicyProfile(
        name="BALANCED",
        description="Standard balanced rewards",
        damage_dealt_mult=1.0,
        damage_taken_mult=1.0,
        stock_mult=1.0,
        spacing_mult=1.0,
        punish_mult=1.0,
        survival_mult=1.0
    ),
    PolicyProfile(
        name="DEFENSIVE",
        description="Heavy penalty for taking damage",
        damage_dealt_mult=0.8,
        damage_taken_mult=2.0,
        stock_mult=1.2,
        spacing_mult=1.5,
        punish_mult=0.8,
        survival_mult=2.0
    ),
    PolicyProfile(
        name="AGGRESSIVE",
        description="High reward for dealing damage",
        damage_dealt_mult=2.0,
        damage_taken_mult=0.5,
        stock_mult=1.5,
        spacing_mult=0.3,
        punish_mult=0.5,
        survival_mult=0.5
    ),
    PolicyProfile(
        name="COUNTER",
        description="Focus on punish timing",
        damage_dealt_mult=0.8,
        damage_taken_mult=1.0,
        stock_mult=1.0,
        spacing_mult=1.2,
        punish_mult=3.0,
        survival_mult=1.0
    ),
    PolicyProfile(
        name="SPACING",
        description="Focus on positional control",
        damage_dealt_mult=0.7,
        damage_taken_mult=1.0,
        stock_mult=1.0,
        spacing_mult=3.0,
        punish_mult=1.0,
        survival_mult=1.2
    )
]


class PopulationTrainer:
    """Trains a population of diverse policies."""

    def __init__(self, device: str = 'cuda', checkpoint_dir: str = 'checkpoints/population'):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Create policies for each profile
        self.policies: Dict[str, PolicyNetwork] = {}
        self.trainers: Dict[str, PPOTrainer] = {}
        self.profiles = {p.name: p for p in POPULATION_PROFILES}

        for profile in POPULATION_PROFILES:
            policy = PolicyNetwork(
                obs_size=OBSERVATION_SIZE,
                action_size=ACTION_SPACE_SIZE,
                hidden_size=256
            )
            policy.to(device)
            self.policies[profile.name] = policy

            config = TrainingConfig(
                learning_rate=3e-4,
                batch_size=64,
                epochs_per_update=4,
                gamma=0.99,
                gae_lambda=0.95,
                entropy_coef=0.015,
                value_coef=0.5
            )
            self.trainers[profile.name] = PPOTrainer(policy, config, device=device)

        # Opponents
        self.random_opp = RandomPolicy()
        self.aggressive_opp = AggressivePolicy()

        # Base reward values
        self.base_rewards = {
            'damage_dealt': 0.002,
            'damage_taken': -0.002,
            'stock_taken': 1.0,
            'stock_lost': -1.0,
            'win': 0.5,
            'lose': -0.5,
            'spacing_bonus': 0.001,
            'punish_bonus': 0.005,
            'survival_bonus': 0.0005
        }

    def compute_reward(self, profile: PolicyProfile, game, player_idx: int,
                       prev_damage: List[float], prev_stocks: List[int],
                       action: int, opponent_attacked: bool) -> float:
        """Compute reward based on profile multipliers."""
        opponent_idx = 1 - player_idx
        fighter = game.fighters[player_idx]
        opponent = game.fighters[opponent_idx]

        base = self.base_rewards
        reward = 0.0

        # Stock changes
        if opponent.stocks < prev_stocks[opponent_idx]:
            reward += base['stock_taken'] * profile.stock_mult
        if fighter.stocks < prev_stocks[player_idx]:
            reward += base['stock_lost'] * profile.stock_mult

        # Damage dealt
        damage_dealt = opponent.damage_percent - prev_damage[opponent_idx]
        if damage_dealt > 0:
            dmg_reward = damage_dealt * base['damage_dealt'] * profile.damage_dealt_mult

            # Punish bonus
            if opponent_attacked:
                dmg_reward += damage_dealt * base['punish_bonus'] * profile.punish_mult

            reward += dmg_reward

        # Damage taken
        damage_taken = fighter.damage_percent - prev_damage[player_idx]
        if damage_taken > 0:
            reward += damage_taken * base['damage_taken'] * profile.damage_taken_mult

        # Spacing bonus (for movement near optimal range)
        if action in [1, 2]:  # Movement
            distance = abs(fighter.position.x - opponent.position.x) if hasattr(fighter, 'position') else 100
            if 150 < distance < 300:
                reward += base['spacing_bonus'] * profile.spacing_mult

        # Survival bonus at high damage
        if fighter.damage_percent >= 80:
            if hasattr(fighter, 'grounded') and fighter.grounded:
                reward += base['survival_bonus'] * profile.survival_mult

        return reward

    def run_game(self, policy: PolicyNetwork, profile: PolicyProfile,
                 opponent, character: str = 'FLEX') -> Tuple[Episode, int]:
        """Run a single game for a specific policy."""
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

        frame = 0
        max_frames = 10800
        opponent_was_attacking = False

        while frame < max_frames:
            obs = [get_observation(game, i) for i in range(2)]

            action_p1, log_prob, value = policy.get_action(obs[0])
            action_p2, _, _ = opponent.get_action(obs[1])

            # Detect opponent attack
            opponent_was_attacking = action_p2 >= 5

            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            match_ongoing = loop.step()
            frame += 1

            reward = self.compute_reward(
                profile, game, 0, prev_damage, prev_stocks,
                action_p1, opponent_was_attacking
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

        # Win/loss
        winner = game.winner
        if winner == 0:
            episode.transitions[-1].reward += self.base_rewards['win']
        elif winner == 1:
            episode.transitions[-1].reward += self.base_rewards['lose']

        return episode, winner

    def train_policy(self, policy_name: str, num_games: int = 600) -> float:
        """Train a single policy."""
        print(f"\n--- Training {policy_name} ---")
        print(f"Profile: {self.profiles[policy_name].description}")

        policy = self.policies[policy_name]
        trainer = self.trainers[policy_name]
        profile = self.profiles[policy_name]

        episodes = []
        wins = 0
        games_per_update = 10

        start_time = time.time()

        for i in range(num_games):
            # Mix of opponents (heavier aggressive focus)
            if i < num_games // 4:
                opponent = self.random_opp
            elif np.random.random() < 0.75:
                opponent = self.aggressive_opp
            else:
                opponent = self.random_opp

            episode, winner = self.run_game(policy, profile, opponent)
            if winner == 0:
                wins += 1
            episodes.append(episode)

            if len(episodes) >= games_per_update:
                trainer.update(episodes)
                episodes = []

            if (i + 1) % 100 == 0:
                print(f"  {policy_name}: {i+1}/{num_games} - WR: {wins/(i+1):.1%}")

        if episodes:
            trainer.update(episodes)

        elapsed = time.time() - start_time
        win_rate = wins / num_games

        # Save checkpoint
        policy.save(f'{self.checkpoint_dir}/{policy_name.lower()}_policy.npz')

        print(f"  Completed: {win_rate:.1%} win rate in {elapsed:.1f}s")
        return win_rate

    def evaluate_policy_vs_baselines(self, policy_name: str, num_games: int = 40) -> Dict:
        """Evaluate a policy against baselines."""
        policy = self.policies[policy_name]

        vs_random = evaluate_vs_random(policy, num_games=num_games)
        vs_aggressive = evaluate_vs_aggressive(policy, num_games=num_games)

        return {
            'vs_random': vs_random['win_rate'],
            'vs_aggressive': vs_aggressive['win_rate']
        }

    def run_tournament(self, games_per_matchup: int = 30) -> Dict[str, Dict[str, float]]:
        """Run a round-robin tournament between all policies."""
        print("\n" + "="*60)
        print("POPULATION TOURNAMENT")
        print("="*60)

        policy_names = list(self.policies.keys())
        results = {name: {} for name in policy_names}

        for i, p1_name in enumerate(policy_names):
            for j, p2_name in enumerate(policy_names):
                if i >= j:  # Skip self and already computed (symmetric)
                    continue

                p1_wins = 0
                p1_policy = self.policies[p1_name]
                p2_policy = self.policies[p2_name]

                for _ in range(games_per_matchup):
                    game = create_game('battlefield', 'FLEX', 'FLEX', stock_count=3)
                    loop = GameLoop(game, headless=True)

                    frame = 0
                    while frame < 10800 and not game.match_over:
                        obs1 = get_observation(game, 0)
                        obs2 = get_observation(game, 1)

                        a1, _, _ = p1_policy.get_action(obs1)
                        a2, _, _ = p2_policy.get_action(obs2)

                        game.fighters[0].input = get_action_from_index(a1)
                        game.fighters[1].input = get_action_from_index(a2)

                        loop.step()
                        frame += 1

                    if game.winner == 0:
                        p1_wins += 1

                win_rate = p1_wins / games_per_matchup
                results[p1_name][p2_name] = win_rate
                results[p2_name][p1_name] = 1.0 - win_rate

                print(f"  {p1_name} vs {p2_name}: {win_rate:.1%}")

        # Add self matchups (50%)
        for name in policy_names:
            results[name][name] = 0.5

        return results

    def analyze_behavior_differences(self, num_games: int = 20) -> Dict:
        """Analyze how each policy's behavior differs."""
        print("\n--- Behavior Analysis ---")

        behaviors = {}
        for name, policy in self.policies.items():
            action_counts = np.zeros(ACTION_SPACE_SIZE)
            total_frames = 0

            for _ in range(num_games):
                game = create_game('battlefield', 'FLEX', 'FLEX', stock_count=3)
                loop = GameLoop(game, headless=True)

                frame = 0
                while frame < 10800 and not game.match_over:
                    obs = get_observation(game, 0)
                    action, _, _ = policy.get_action(obs)
                    action_counts[action] += 1
                    total_frames += 1

                    obs_opp = get_observation(game, 1)
                    action_opp = np.random.randint(0, ACTION_SPACE_SIZE)

                    game.fighters[0].input = get_action_from_index(action)
                    game.fighters[1].input = get_action_from_index(action_opp)

                    loop.step()
                    frame += 1

            if total_frames > 0:
                dist = action_counts / total_frames
                attack_ratio = sum(dist[i] for i in [5, 6, 7, 8, 9])
                move_ratio = dist[1] + dist[2]
                idle_ratio = dist[0]

                behaviors[name] = {
                    'attack_ratio': float(attack_ratio),
                    'move_ratio': float(move_ratio),
                    'idle_ratio': float(idle_ratio),
                    'action_dist': dist.tolist()
                }

                print(f"  {name}: Attack={attack_ratio:.1%}, Move={move_ratio:.1%}, Idle={idle_ratio:.1%}")

        return behaviors

    def run_population_training(self, games_per_policy: int = 600) -> Dict:
        """Run complete population training and evaluation."""
        print("\n" + "="*60)
        print("POPULATION-BASED TRAINING")
        print(f"Policies: {len(POPULATION_PROFILES)}")
        print(f"Games per policy: {games_per_policy}")
        print("="*60)

        start_time = time.time()

        # Train each policy
        training_results = {}
        for profile in POPULATION_PROFILES:
            win_rate = self.train_policy(profile.name, num_games=games_per_policy)
            training_results[profile.name] = win_rate

        # Evaluate each policy against baselines
        print("\n" + "="*60)
        print("BASELINE EVALUATION")
        print("="*60)

        baseline_results = {}
        for name in self.policies.keys():
            baseline_results[name] = self.evaluate_policy_vs_baselines(name)
            print(f"  {name}: vs Random={baseline_results[name]['vs_random']:.1%}, "
                  f"vs Aggressive={baseline_results[name]['vs_aggressive']:.1%}")

        # Run tournament
        tournament_results = self.run_tournament()

        # Analyze behaviors
        behavior_results = self.analyze_behavior_differences()

        elapsed = time.time() - start_time

        # Find best vs aggressive
        best_name = max(baseline_results.keys(),
                       key=lambda k: baseline_results[k]['vs_aggressive'])
        best_vs_agg = baseline_results[best_name]['vs_aggressive']

        # Generate report
        report = {
            'training': {
                'games_per_policy': games_per_policy,
                'total_games': games_per_policy * len(POPULATION_PROFILES),
                'elapsed_seconds': elapsed,
                'policy_results': training_results
            },
            'baseline_evaluation': baseline_results,
            'tournament': tournament_results,
            'behaviors': behavior_results,
            'best_vs_aggressive': {
                'policy': best_name,
                'win_rate': best_vs_agg
            },
            'profiles': {p.name: p.description for p in POPULATION_PROFILES}
        }

        report_path = f'{self.checkpoint_dir}/population_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_path}")

        # Summary
        print("\n" + "="*60)
        print("POPULATION TRAINING SUMMARY")
        print("="*60)
        print(f"Total training time: {elapsed/60:.1f} minutes")
        print(f"\nBest vs Aggressive: {best_name} ({best_vs_agg:.1%})")

        print("\nTournament rankings:")
        rankings = []
        for name in self.policies.keys():
            avg_wr = np.mean([v for k, v in tournament_results[name].items() if k != name])
            rankings.append((name, avg_wr))
        rankings.sort(key=lambda x: x[1], reverse=True)
        for i, (name, avg_wr) in enumerate(rankings, 1):
            print(f"  {i}. {name}: {avg_wr:.1%}")

        return report


def main():
    print("="*60)
    print("STICK FIGURE FIGHTER - POPULATION-BASED TRAINING")
    print("Creating diverse policy population for strategy variety")
    print("="*60)

    trainer = PopulationTrainer(
        device='cuda',
        checkpoint_dir='checkpoints/population'
    )

    report = trainer.run_population_training(games_per_policy=600)

    return report


if __name__ == '__main__':
    main()
