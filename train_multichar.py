#!/usr/bin/env python3
"""
Multi-Character Training for Stick Figure Fighter.

Trains separate policies for each character type (FLEX, RUSH, ZONE, GRAB)
and then evaluates them against each other to see if character-specific
strategies emerge.
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

import torch
from policy import PolicyNetwork
from trainer import PPOTrainer, TrainingConfig
from experience import ExperienceBuffer, Episode
from game_engine import (
    create_game, GameLoop, get_observation, get_action_from_index,
    OBSERVATION_SIZE, ACTION_SPACE_SIZE
)
from evaluate import RandomPolicy, AggressivePolicy


class CharacterTrainer:
    """
    Trains character-specific policies.
    """

    def __init__(self, character: str, device: str = 'cuda'):
        self.character = character
        self.device = device

        # Initialize policy
        self.policy = PolicyNetwork(
            obs_size=OBSERVATION_SIZE,
            action_size=ACTION_SPACE_SIZE,
            hidden_size=256
        )
        self.policy.to(device)

        # Training config
        self.config = TrainingConfig(
            clip_ratio=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            learning_rate=3e-4,
            batch_size=64,
            epochs_per_update=4,
            games_per_iteration=10
        )

        self.trainer = PPOTrainer(self.policy, self.config, device=device)
        self.games_played = 0

    def collect_episode(self, opponent_policy, opponent_character: str = None) -> tuple:
        """Collect a training episode."""
        if opponent_character is None:
            opponent_character = self.character

        game = create_game(
            stage_name='battlefield',
            p1_character=self.character,
            p2_character=opponent_character,
            stock_count=3
        )

        loop = GameLoop(game, headless=True)
        episode = Episode(player_id=0)

        frame = 0
        max_frames = 10800
        damage_dealt = 0.0

        while frame < max_frames and not game.match_over:
            obs_p1 = get_observation(game, 0)
            obs_p2 = get_observation(game, 1)

            action_p1, log_prob, value = self.policy.get_action(obs_p1)
            action_p2, _, _ = opponent_policy.get_action(obs_p2)

            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            prev_p2_dmg = game.fighters[1].damage_percent
            prev_p2_stocks = game.fighters[1].stocks
            prev_p1_stocks = game.fighters[0].stocks

            loop.step()
            frame += 1

            # Reward calculation
            reward = 0.0
            if game.fighters[1].stocks < prev_p2_stocks:
                reward += 1.0
            if game.fighters[0].stocks < prev_p1_stocks:
                reward -= 1.0
            dmg = game.fighters[1].damage_percent - prev_p2_dmg
            if dmg > 0:
                reward += dmg * 0.001
                damage_dealt += dmg

            done = game.match_over or frame >= max_frames
            if done:
                if game.winner == 0:
                    reward += 0.5
                elif game.winner == 1:
                    reward -= 0.5

            episode.add(obs_p1, action_p1, reward, done, log_prob, value)
            if done:
                break

        return episode, game.winner, {'damage_dealt': damage_dealt, 'frames': frame}

    def train(self, num_games: int, opponent_policy=None) -> Dict:
        """Train for specified number of games."""
        if opponent_policy is None:
            opponent_policy = RandomPolicy()

        wins = 0
        episodes = []

        for i in range(num_games):
            episode, winner, stats = self.collect_episode(opponent_policy)
            if winner == 0:
                wins += 1
            episodes.append(episode)
            self.games_played += 1

            if len(episodes) >= self.config.games_per_iteration:
                self.trainer.update(episodes)
                episodes = []

        return {
            'games': num_games,
            'wins': wins,
            'win_rate': wins / num_games
        }

    def evaluate(self, opponent_policy, opponent_char: str, num_games: int = 50) -> Dict:
        """Evaluate against a specific opponent."""
        wins = 0
        total_damage = 0.0
        total_stocks = 0

        for _ in range(num_games):
            game = create_game(
                stage_name='battlefield',
                p1_character=self.character,
                p2_character=opponent_char,
                stock_count=3
            )
            loop = GameLoop(game, headless=True)

            frame = 0
            while frame < 10800 and not game.match_over:
                obs_p1 = get_observation(game, 0)
                obs_p2 = get_observation(game, 1)

                action_p1, _, _ = self.policy.get_action(obs_p1)
                action_p2, _, _ = opponent_policy.get_action(obs_p2)

                game.fighters[0].input = get_action_from_index(action_p1)
                game.fighters[1].input = get_action_from_index(action_p2)

                loop.step()
                frame += 1

            if game.winner == 0:
                wins += 1
            total_damage += game.fighters[1].damage_percent
            total_stocks += 3 - game.fighters[1].stocks

        return {
            'wins': wins,
            'win_rate': wins / num_games,
            'avg_damage': total_damage / num_games,
            'avg_stocks_taken': total_stocks / num_games
        }


def analyze_character_actions(policy: PolicyNetwork, character: str, num_games: int = 20) -> Dict:
    """Analyze action distribution for a character policy."""
    action_counts = np.zeros(ACTION_SPACE_SIZE)
    random_opp = RandomPolicy()

    for _ in range(num_games):
        game = create_game(
            stage_name='battlefield',
            p1_character=character,
            p2_character='FLEX',
            stock_count=3
        )
        loop = GameLoop(game, headless=True)

        frame = 0
        while frame < 10800 and not game.match_over:
            obs = get_observation(game, 0)
            action, _, _ = policy.get_action(obs)
            action_counts[action] += 1

            obs_p2 = get_observation(game, 1)
            action_p2, _, _ = random_opp.get_action(obs_p2)

            game.fighters[0].input = get_action_from_index(action)
            game.fighters[1].input = get_action_from_index(action_p2)

            loop.step()
            frame += 1

    total = action_counts.sum()
    action_probs = action_counts / total if total > 0 else action_counts

    action_names = [
        'Nothing', 'Left', 'Right', 'Jump', 'Down',
        'Attack', 'L+Atk', 'R+Atk', 'U+Atk', 'D+Atk'
    ]

    return {
        name: float(prob) for name, prob in zip(action_names, action_probs)
    }


def run_multichar_experiment(output_dir: str = 'checkpoints/multichar'):
    """Run the multi-character training experiment."""
    os.makedirs(output_dir, exist_ok=True)

    characters = ['FLEX', 'RUSH', 'ZONE', 'GRAB']
    trainers: Dict[str, CharacterTrainer] = {}

    print('\n' + '=' * 60)
    print('MULTI-CHARACTER TRAINING EXPERIMENT')
    print('=' * 60)
    print(f'Characters: {", ".join(characters)}')

    # Phase 1: Train each character vs random
    print('\n--- Phase 1: Training vs Random ---')
    for char in characters:
        print(f'\nTraining {char}...')
        trainer = CharacterTrainer(char, device='cuda')
        stats = trainer.train(num_games=500, opponent_policy=RandomPolicy())
        trainers[char] = trainer
        print(f'  {char}: {stats["win_rate"]:.1%} win rate vs random')

        # Save checkpoint
        trainer.policy.save(f'{output_dir}/{char.lower()}_policy.npz')

    # Phase 2: Character matchup evaluation
    print('\n--- Phase 2: Character Matchup Matrix ---')
    matchup_results = {}

    for p1_char in characters:
        matchup_results[p1_char] = {}
        for p2_char in characters:
            if p1_char == p2_char:
                # Mirror match - evaluate vs aggressive
                result = trainers[p1_char].evaluate(
                    AggressivePolicy(), p2_char, num_games=30
                )
                matchup_results[p1_char][p2_char] = result['win_rate']
            else:
                # Cross-character match
                result = trainers[p1_char].evaluate(
                    trainers[p2_char].policy, p2_char, num_games=30
                )
                matchup_results[p1_char][p2_char] = result['win_rate']

    # Print matchup matrix
    print('\n  Matchup Matrix (row vs column):')
    print(f'  {"":8s}' + ''.join(f'{c:8s}' for c in characters))
    for p1 in characters:
        row = f'  {p1:8s}'
        for p2 in characters:
            row += f'{matchup_results[p1][p2]:.1%}   '
        print(row)

    # Phase 3: Analyze character-specific strategies
    print('\n--- Phase 3: Strategy Analysis ---')
    strategy_analysis = {}

    for char in characters:
        print(f'\n{char}:')
        actions = analyze_character_actions(trainers[char].policy, char, num_games=10)
        strategy_analysis[char] = actions

        # Sort actions by frequency
        sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)

        # Determine playstyle
        attack_ratio = sum(v for k, v in actions.items() if 'Atk' in k or k == 'Attack')
        move_ratio = actions['Left'] + actions['Right']
        jump_ratio = actions['Jump']

        if attack_ratio > 0.4:
            style = 'Aggressive'
        elif move_ratio > 0.3:
            style = 'Mobile'
        elif jump_ratio > 0.15:
            style = 'Aerial'
        else:
            style = 'Balanced'

        print(f'  Playstyle: {style}')
        print(f'  Attack ratio: {attack_ratio:.1%}')
        print(f'  Movement ratio: {move_ratio:.1%}')
        print(f'  Top actions:')
        for action, prob in sorted_actions[:4]:
            print(f'    {action}: {prob:.1%}')

    # Save report
    report = {
        'characters': characters,
        'matchup_matrix': matchup_results,
        'strategy_analysis': strategy_analysis,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    report_path = f'{output_dir}/multichar_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'\n{"="*60}')
    print('EXPERIMENT COMPLETE')
    print(f'{"="*60}')
    print(f'Report saved to: {report_path}')

    return report


if __name__ == '__main__':
    run_multichar_experiment()
