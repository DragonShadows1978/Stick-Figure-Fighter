#!/usr/bin/env python3
"""
Specialized training to beat AggressivePolicy.

Focuses on:
1. Starting from the best baseline checkpoint
2. Training specifically against aggressive opponents
3. Reward shaping for defensive play and punishing
4. Higher training volume vs aggressive
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
from evaluate import (
    evaluate_vs_random, evaluate_vs_aggressive,
    AggressivePolicy, RandomPolicy, analyze_behavior
)


def collect_episode_anti_aggro(policy, opponent, character='FLEX'):
    """
    Collect episode with anti-aggro reward shaping.

    Rewards:
    - Stock take: +1.0
    - Stock loss: -1.0
    - Damage dealt: +0.002 per %
    - Damage taken: -0.0015 per % (penalize taking damage vs aggro)
    - High opponent damage survival bonus: +0.003 per frame
    - Win bonus: +0.5
    """
    game = create_game(
        stage_name='battlefield',
        p1_character=character,
        p2_character=character,
        stock_count=3
    )

    loop = GameLoop(game, headless=True)
    episode = Episode(player_id=0)

    frame = 0
    max_frames = 10800

    while frame < max_frames and not game.match_over:
        obs_p1 = get_observation(game, 0)
        obs_p2 = get_observation(game, 1)

        action, log_prob, value = policy.get_action(obs_p1)
        action_opp, _, _ = opponent.get_action(obs_p2)

        # Store previous state
        prev_p1_stocks = game.fighters[0].stocks
        prev_p2_stocks = game.fighters[1].stocks
        prev_p1_dmg = game.fighters[0].damage_percent
        prev_p2_dmg = game.fighters[1].damage_percent

        game.fighters[0].input = get_action_from_index(action)
        game.fighters[1].input = get_action_from_index(action_opp)

        loop.step()
        frame += 1

        # Calculate reward
        reward = 0.0

        # Stock rewards
        if game.fighters[1].stocks < prev_p2_stocks:
            reward += 1.0
        if game.fighters[0].stocks < prev_p1_stocks:
            reward -= 1.0

        # Damage dealt bonus
        dmg_dealt = game.fighters[1].damage_percent - prev_p2_dmg
        if dmg_dealt > 0:
            reward += dmg_dealt * 0.002

        # Damage taken penalty (anti-aggro focus)
        dmg_taken = game.fighters[0].damage_percent - prev_p1_dmg
        if dmg_taken > 0:
            reward -= dmg_taken * 0.0015

        # Survival bonus when opponent is at high %
        if game.fighters[1].damage_percent > 80 and game.fighters[0].damage_percent < 60:
            reward += 0.002

        done = game.match_over or frame >= max_frames
        if done:
            if game.winner == 0:
                reward += 0.5
            elif game.winner == 1:
                reward -= 0.5

        episode.add(obs_p1, action, reward, done, log_prob, value)

        if done:
            break

    return episode, game.winner == 0


def train_anti_aggressive(
    base_checkpoint: str,
    output_dir: str,
    num_games: int = 800,
    eval_interval: int = 100
):
    """
    Train policy to beat aggressive opponents.
    """
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('ANTI-AGGRESSIVE TRAINING')
    print('=' * 60)

    # Load base policy
    policy = PolicyNetwork(obs_size=OBSERVATION_SIZE, action_size=ACTION_SPACE_SIZE)
    policy.to('cuda')

    if os.path.exists(base_checkpoint):
        print(f'Loading base checkpoint: {base_checkpoint}')
        policy.load(base_checkpoint)
    else:
        print('No base checkpoint, starting fresh')

    # Config with higher entropy for exploration
    config = TrainingConfig(
        clip_ratio=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.02,  # Higher entropy for learning new strategies
        value_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=2e-4,  # Slightly lower LR
        batch_size=64,
        epochs_per_update=4,
        games_per_iteration=10
    )

    trainer = PPOTrainer(policy, config, device='cuda')
    aggressive = AggressivePolicy()

    # Initial evaluation
    print('\nInitial evaluation:')
    init_random = evaluate_vs_random(policy, num_games=30)
    init_agg = evaluate_vs_aggressive(policy, num_games=30)
    print(f'  vs Random: {init_random["win_rate"]:.1%}')
    print(f'  vs Aggressive: {init_agg["win_rate"]:.1%}')

    start_time = time.time()
    episodes = []
    wins = 0
    eval_history = []

    print(f'\nTraining {num_games} games vs AggressivePolicy...')

    for game_idx in range(num_games):
        episode, won = collect_episode_anti_aggro(policy, aggressive)

        if won:
            wins += 1

        episodes.append(episode)

        # PPO update
        if len(episodes) >= config.games_per_iteration:
            stats = trainer.update(episodes)
            episodes = []

            running_wr = wins / (game_idx + 1)
            print(f'  Game {game_idx + 1}/{num_games}: '
                  f'WR={running_wr:.1%}, '
                  f'Loss={stats["policy_loss"]:.4f}, '
                  f'Ent={stats["entropy"]:.3f}')

        # Periodic evaluation
        if (game_idx + 1) % eval_interval == 0:
            eval_random = evaluate_vs_random(policy, num_games=20)
            eval_agg = evaluate_vs_aggressive(policy, num_games=30)

            eval_history.append({
                'games': game_idx + 1,
                'vs_random': eval_random['win_rate'],
                'vs_aggressive': eval_agg['win_rate']
            })

            print(f'\n  [Eval at {game_idx + 1}] '
                  f'vs Random: {eval_random["win_rate"]:.1%}, '
                  f'vs Aggressive: {eval_agg["win_rate"]:.1%}\n')

            # Save checkpoint
            policy.save(f'{output_dir}/checkpoint_{game_idx + 1:04d}.npz')

            # Stop if we hit target
            if eval_agg['win_rate'] >= 0.55:
                print('  Reached target win rate!')

    elapsed = time.time() - start_time

    # Final evaluation
    print('\n' + '=' * 60)
    print('FINAL EVALUATION')
    print('=' * 60)

    final_random = evaluate_vs_random(policy, num_games=50)
    final_agg = evaluate_vs_aggressive(policy, num_games=50)
    behavior = analyze_behavior(policy, num_games=20)

    print(f'\nvs Random: {final_random["win_rate"]:.1%}')
    print(f'vs Aggressive: {final_agg["win_rate"]:.1%}')
    print(f'\nBehavior:')
    print(f'  Playstyle: {behavior["playstyle"]}')
    print(f'  Attack: {behavior["attack_ratio"]:.1%}')
    print(f'  Movement: {behavior["movement_ratio"]:.1%}')
    print(f'  Idle: {behavior["idle_ratio"]:.1%}')

    # Save final checkpoint
    policy.save(f'{output_dir}/checkpoint_final.npz')

    # Save report
    report = {
        'config': {
            'base_checkpoint': base_checkpoint,
            'num_games': num_games,
            'entropy_coef': config.entropy_coef
        },
        'initial': {
            'vs_random': init_random['win_rate'],
            'vs_aggressive': init_agg['win_rate']
        },
        'final': {
            'vs_random': final_random['win_rate'],
            'vs_aggressive': final_agg['win_rate']
        },
        'improvement': {
            'vs_random': final_random['win_rate'] - init_random['win_rate'],
            'vs_aggressive': final_agg['win_rate'] - init_agg['win_rate']
        },
        'training': {
            'elapsed_seconds': elapsed,
            'games_per_second': num_games / elapsed,
            'running_win_rate': wins / num_games
        },
        'behavior': behavior,
        'eval_history': eval_history
    }

    report_path = f'{output_dir}/anti_aggressive_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'\nReport saved to: {report_path}')
    print(f'Total time: {elapsed / 60:.1f} minutes')

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/full_run/checkpoint_001000.npz')
    parser.add_argument('--games', type=int, default=800)
    parser.add_argument('--output', type=str, default='checkpoints/anti_aggressive')

    args = parser.parse_args()

    train_anti_aggressive(
        base_checkpoint=args.checkpoint,
        output_dir=args.output,
        num_games=args.games
    )


if __name__ == '__main__':
    main()
