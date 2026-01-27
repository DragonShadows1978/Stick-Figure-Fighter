#!/usr/bin/env python3
"""
Quick character comparison - trains 4 characters with minimal games
and evaluates cross-character matchups.
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
    max_frames = 5400  # 90 seconds, shorter games

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
            reward += dmg_dealt * 0.002

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


def quick_train_character(character: str, num_games: int = 50, device: str = 'cuda'):
    """Quick train a single character policy."""
    print(f'  Training {character}...', end=' ', flush=True)

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

    print(f'{wins/num_games:.0%} WR')
    return policy, wins / num_games


def evaluate_matchup(policy1, policy2, char1, char2, num_games=20):
    """Evaluate matchup between two policies."""
    wins = 0
    for _ in range(num_games):
        game = create_game('battlefield', char1, char2, stock_count=3)
        loop = GameLoop(game, headless=True)

        frame = 0
        while frame < 5400 and not game.match_over:
            obs1 = get_observation(game, 0)
            obs2 = get_observation(game, 1)

            a1, _, _ = policy1.get_action(obs1)
            a2, _, _ = policy2.get_action(obs2)

            game.fighters[0].input = get_action_from_index(a1)
            game.fighters[1].input = get_action_from_index(a2)

            loop.step()
            frame += 1

        if game.winner == 0:
            wins += 1

    return wins / num_games


def analyze_strategy(policy, character, num_games=10):
    """Analyze action distribution for a character."""
    random_opp = RandomPolicy()
    action_counts = np.zeros(ACTION_SPACE_SIZE)
    total_frames = 0

    for _ in range(num_games):
        game = create_game('battlefield', character, 'FLEX', stock_count=3)
        loop = GameLoop(game, headless=True)

        frame = 0
        while frame < 5400 and not game.match_over:
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

    # Compute metrics
    attack_ratio = sum(action_probs[5:10])  # Attack actions are 5-9
    movement_ratio = action_probs[1] + action_probs[2]  # Left, Right
    jump_ratio = action_probs[3]
    idle_ratio = action_probs[0]

    if attack_ratio > 0.4:
        style = 'Aggressive'
    elif movement_ratio > 0.3:
        style = 'Mobile'
    elif idle_ratio > 0.2:
        style = 'Defensive'
    else:
        style = 'Balanced'

    return {
        'style': style,
        'attack_ratio': float(attack_ratio),
        'movement_ratio': float(movement_ratio),
        'jump_ratio': float(jump_ratio),
        'idle_ratio': float(idle_ratio)
    }


def main():
    characters = ['FLEX', 'RUSH', 'ZONE', 'GRAB']
    policies = {}
    strategies = {}

    print('=' * 60)
    print('QUICK CHARACTER COMPARISON')
    print('=' * 60)

    start_time = time.time()

    # Train each character
    print('\nPhase 1: Training (50 games each)')
    for char in characters:
        policy, wr = quick_train_character(char, num_games=50)
        policies[char] = policy

    # Analyze strategies
    print('\nPhase 2: Strategy Analysis')
    for char in characters:
        strategies[char] = analyze_strategy(policies[char], char, num_games=10)
        s = strategies[char]
        print(f'  {char}: {s["style"]} (Atk:{s["attack_ratio"]:.0%}, Mv:{s["movement_ratio"]:.0%}, Idle:{s["idle_ratio"]:.0%})')

    # Cross-character matchups
    print('\nPhase 3: Cross-Character Matchups (20 games each)')
    matchups = {}
    for p1 in characters:
        matchups[p1] = {}
        for p2 in characters:
            if p1 != p2:
                wr = evaluate_matchup(policies[p1], policies[p2], p1, p2, num_games=20)
                matchups[p1][p2] = wr

    # Print matchup table
    print('\nMatchup Matrix (row vs column win rate):')
    print(f'{"":8}' + ''.join(f'{c:8}' for c in characters))
    for p1 in characters:
        row = f'{p1:8}'
        for p2 in characters:
            if p1 == p2:
                row += '  --    '
            else:
                row += f'{matchups[p1][p2]:.0%}     '
        print(row)

    elapsed = time.time() - start_time

    # Find interesting patterns
    print('\n--- Strategy Insights ---')

    # Who's most aggressive?
    most_agg = max(strategies.items(), key=lambda x: x[1]['attack_ratio'])
    print(f'Most aggressive: {most_agg[0]} ({most_agg[1]["attack_ratio"]:.0%} attacks)')

    # Who's most mobile?
    most_mob = max(strategies.items(), key=lambda x: x[1]['movement_ratio'])
    print(f'Most mobile: {most_mob[0]} ({most_mob[1]["movement_ratio"]:.0%} movement)')

    # Best overall in cross-character?
    avg_wr = {}
    for p1 in characters:
        wins = [matchups[p1][p2] for p2 in characters if p1 != p2]
        avg_wr[p1] = sum(wins) / len(wins)

    best = max(avg_wr.items(), key=lambda x: x[1])
    print(f'Best cross-char performance: {best[0]} ({best[1]:.0%} avg win rate)')

    # Save results
    output_dir = 'checkpoints/quick_char'
    os.makedirs(output_dir, exist_ok=True)

    for char in characters:
        policies[char].save(f'{output_dir}/{char.lower()}_policy.npz')

    report = {
        'characters': characters,
        'training_games': 100,
        'elapsed_seconds': elapsed,
        'strategies': strategies,
        'matchups': matchups,
        'avg_win_rates': avg_wr
    }

    with open(f'{output_dir}/quick_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f'\n{"="*60}')
    print(f'Complete in {elapsed:.0f}s')
    print(f'Report: {output_dir}/quick_comparison_report.json')


if __name__ == '__main__':
    main()
