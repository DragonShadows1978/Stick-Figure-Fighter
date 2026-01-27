#!/usr/bin/env python3
"""
Character Variety Training Experiment.

Trains separate policies for FLEX, RUSH, ZONE, GRAB characters
and analyzes whether character-specific strategies emerge.
"""
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

import torch
from policy import PolicyNetwork
from trainer import PPOTrainer, TrainingConfig
from experience import Episode
from game_engine import (
    create_game, GameLoop, get_observation, get_action_from_index,
    OBSERVATION_SIZE, ACTION_SPACE_SIZE
)
from evaluate import RandomPolicy, AggressivePolicy


def collect_episode(policy, opponent, character, opp_character):
    """Collect a single training episode."""
    game = create_game(
        stage_name='battlefield',
        p1_character=character,
        p2_character=opp_character,
        stock_count=3
    )

    loop = GameLoop(game, headless=True)
    episode = Episode(player_id=0)

    frame = 0
    max_frames = 10800

    while frame < max_frames and not game.match_over:
        obs = get_observation(game, 0)
        obs_opp = get_observation(game, 1)

        action, log_prob, value = policy.get_action(obs)
        action_opp, _, _ = opponent.get_action(obs_opp)

        prev_p1_stocks = game.fighters[0].stocks
        prev_p2_stocks = game.fighters[1].stocks
        prev_p2_dmg = game.fighters[1].damage_percent

        game.fighters[0].input = get_action_from_index(action)
        game.fighters[1].input = get_action_from_index(action_opp)

        loop.step()
        frame += 1

        # Reward
        reward = 0.0
        if game.fighters[1].stocks < prev_p2_stocks:
            reward += 1.0
        if game.fighters[0].stocks < prev_p1_stocks:
            reward -= 1.0

        dmg_dealt = game.fighters[1].damage_percent - prev_p2_dmg
        if dmg_dealt > 0:
            reward += dmg_dealt * 0.001

        done = game.match_over or frame >= max_frames
        if done:
            if game.winner == 0:
                reward += 0.5
            elif game.winner == 1:
                reward -= 0.5

        episode.add(obs, action, reward, done, log_prob, value)
        if done:
            break

    return episode, game.winner == 0


def train_character(character: str, num_games: int = 400, device: str = 'cuda'):
    """Train a single character policy."""
    print(f'\n--- Training {character} ---')

    policy = PolicyNetwork(obs_size=OBSERVATION_SIZE, action_size=ACTION_SPACE_SIZE)
    policy.to(device)

    config = TrainingConfig(
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

    trainer = PPOTrainer(policy, config, device=device)
    random_opp = RandomPolicy()

    episodes = []
    wins = 0

    for i in range(num_games):
        episode, won = collect_episode(policy, random_opp, character, 'FLEX')
        if won:
            wins += 1
        episodes.append(episode)

        if len(episodes) >= config.games_per_iteration:
            trainer.update(episodes)
            episodes = []

        if (i + 1) % 100 == 0:
            print(f'  {character}: {i + 1}/{num_games} games, WR={wins/(i+1):.1%}')

    return policy, wins / num_games


def analyze_character_behavior(policy, character: str, num_games: int = 20):
    """Analyze action distribution for a character."""
    random_opp = RandomPolicy()
    action_counts = np.zeros(ACTION_SPACE_SIZE)
    total_frames = 0

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
            total_frames += 1

            obs_opp = get_observation(game, 1)
            action_opp, _, _ = random_opp.get_action(obs_opp)

            game.fighters[0].input = get_action_from_index(action)
            game.fighters[1].input = get_action_from_index(action_opp)

            loop.step()
            frame += 1

    action_probs = action_counts / total_frames if total_frames > 0 else action_counts

    action_names = [
        'Nothing', 'Left', 'Right', 'Jump', 'Down',
        'Attack', 'L+Atk', 'R+Atk', 'U+Atk', 'D+Atk'
    ]

    return {name: float(prob) for name, prob in zip(action_names, action_probs)}


def run_character_variety_experiment(output_dir: str = 'checkpoints/char_variety'):
    """Run the full character variety experiment."""
    os.makedirs(output_dir, exist_ok=True)

    characters = ['FLEX', 'RUSH', 'ZONE', 'GRAB']
    policies = {}
    win_rates = {}
    behaviors = {}

    print('=' * 60)
    print('CHARACTER VARIETY EXPERIMENT')
    print('=' * 60)

    start_time = time.time()

    # Phase 1: Train each character
    for char in characters:
        policy, wr = train_character(char, num_games=400)
        policies[char] = policy
        win_rates[char] = wr

        # Save checkpoint
        policy.save(f'{output_dir}/{char.lower()}_policy.npz')
        print(f'  {char} trained: {wr:.1%} win rate vs random')

    # Phase 2: Cross-character evaluation
    print('\n--- Cross-Character Matchups ---')
    matchup_matrix = {}

    for p1 in characters:
        matchup_matrix[p1] = {}
        for p2 in characters:
            if p1 == p2:
                # Mirror: evaluate vs AggressivePolicy
                wins = 0
                for _ in range(30):
                    game = create_game('battlefield', p1, p2, stock_count=3)
                    loop = GameLoop(game, headless=True)
                    agg = AggressivePolicy()

                    frame = 0
                    while frame < 10800 and not game.match_over:
                        obs1 = get_observation(game, 0)
                        obs2 = get_observation(game, 1)

                        a1, _, _ = policies[p1].get_action(obs1)
                        a2, _, _ = agg.get_action(obs2)

                        game.fighters[0].input = get_action_from_index(a1)
                        game.fighters[1].input = get_action_from_index(a2)

                        loop.step()
                        frame += 1

                    if game.winner == 0:
                        wins += 1

                matchup_matrix[p1][p2] = wins / 30
            else:
                # Cross-character match
                wins = 0
                for _ in range(30):
                    game = create_game('battlefield', p1, p2, stock_count=3)
                    loop = GameLoop(game, headless=True)

                    frame = 0
                    while frame < 10800 and not game.match_over:
                        obs1 = get_observation(game, 0)
                        obs2 = get_observation(game, 1)

                        a1, _, _ = policies[p1].get_action(obs1)
                        a2, _, _ = policies[p2].get_action(obs2)

                        game.fighters[0].input = get_action_from_index(a1)
                        game.fighters[1].input = get_action_from_index(a2)

                        loop.step()
                        frame += 1

                    if game.winner == 0:
                        wins += 1

                matchup_matrix[p1][p2] = wins / 30

    # Print matchup matrix
    print('\nMatchup Matrix (row vs column):')
    print(f'{"":8s}' + ''.join(f'{c:8s}' for c in characters))
    for p1 in characters:
        row = f'{p1:8s}'
        for p2 in characters:
            row += f'{matchup_matrix[p1][p2]:.1%}   '
        print(row)

    # Phase 3: Analyze strategies
    print('\n--- Character Strategy Analysis ---')
    for char in characters:
        behaviors[char] = analyze_character_behavior(policies[char], char, num_games=15)

        # Compute behavior metrics
        b = behaviors[char]
        attack_ratio = sum(v for k, v in b.items() if 'Atk' in k or k == 'Attack')
        move_ratio = b['Left'] + b['Right']
        jump_ratio = b['Jump']

        if attack_ratio > 0.4:
            style = 'Aggressive'
        elif move_ratio > 0.3:
            style = 'Mobile'
        elif jump_ratio > 0.15:
            style = 'Aerial'
        else:
            style = 'Balanced'

        print(f'\n{char}: {style}')
        print(f'  Attack: {attack_ratio:.1%}, Move: {move_ratio:.1%}, Jump: {jump_ratio:.1%}')

    elapsed = time.time() - start_time

    # Generate report
    report = {
        'characters': characters,
        'training': {
            'games_per_char': 400,
            'elapsed_seconds': elapsed,
        },
        'win_rates_vs_random': win_rates,
        'matchup_matrix': matchup_matrix,
        'behaviors': behaviors,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    report_path = f'{output_dir}/character_variety_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'\n{"="*60}')
    print('EXPERIMENT COMPLETE')
    print(f'{"="*60}')
    print(f'Time: {elapsed / 60:.1f} minutes')
    print(f'Report: {report_path}')

    return report


if __name__ == '__main__':
    run_character_variety_experiment()
