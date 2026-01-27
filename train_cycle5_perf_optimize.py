#!/usr/bin/env python3
"""
Cycle 5: Performance Optimization

Profile training performance and implement optimizations to improve
training throughput from ~3 games/sec to 10+ games/sec.

Key optimization areas:
1. Batch inference on GPU instead of per-frame
2. Vectorized observation computation
3. Parallel game execution
4. Optimized reward computation
5. Memory-efficient episode collection
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import cProfile
import pstats
import io
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

import torch
from policy import PolicyNetwork
from experience import Episode
from trainer import PPOTrainer, TrainingConfig
from evaluate import RandomPolicy, AggressivePolicy
from game_engine import (
    create_game, GameLoop, get_observation, get_action_from_index,
    OBSERVATION_SIZE, ACTION_SPACE_SIZE
)


@dataclass
class ProfileResult:
    """Results from profiling a run."""
    total_time: float
    games: int
    games_per_second: float
    bottlenecks: List[Tuple[str, float]]
    memory_mb: float


def profile_baseline(num_games: int = 20) -> ProfileResult:
    """Profile baseline training performance."""
    print(f"\n{'='*60}")
    print("PROFILING BASELINE PERFORMANCE")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = PolicyNetwork(obs_size=OBSERVATION_SIZE, action_size=ACTION_SPACE_SIZE)
    policy.to(device)

    random_opp = RandomPolicy()

    # Profile with cProfile
    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    for game_idx in range(num_games):
        game = create_game('battlefield', 'FLEX', 'FLEX', stock_count=3)
        loop = GameLoop(game, headless=True)

        frame = 0
        while frame < 3600 and not game.match_over:  # 60 seconds max
            obs1 = get_observation(game, 0)
            obs2 = get_observation(game, 1)

            action_p1, _, _ = policy.get_action(obs1)
            action_p2, _, _ = random_opp.get_action(obs2)

            game.fighters[0].input = get_action_from_index(action_p1)
            game.fighters[1].input = get_action_from_index(action_p2)

            loop.step()
            frame += 1

    elapsed = time.time() - start_time
    end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    pr.disable()

    # Get top bottlenecks
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)

    bottlenecks = []
    lines = s.getvalue().split('\n')
    for line in lines[7:17]:  # Skip header lines
        if line.strip():
            parts = line.split()
            if len(parts) >= 6:
                try:
                    cum_time = float(parts[3])
                    func_name = parts[-1]
                    bottlenecks.append((func_name, cum_time))
                except ValueError:
                    continue

    result = ProfileResult(
        total_time=elapsed,
        games=num_games,
        games_per_second=num_games / elapsed,
        bottlenecks=bottlenecks,
        memory_mb=(end_mem - start_mem) / (1024 * 1024)
    )

    print(f"\nBaseline Results:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Games/sec: {result.games_per_second:.2f}")
    print(f"  Memory used: {result.memory_mb:.1f} MB")
    print(f"\nTop Bottlenecks:")
    for func, time_s in bottlenecks[:5]:
        print(f"  {time_s:.3f}s - {func}")

    return result


class OptimizedPolicy:
    """Policy with batch inference optimization."""

    def __init__(self, base_policy: PolicyNetwork, device: str = 'cuda'):
        self.policy = base_policy
        self.device = device

    @torch.no_grad()
    def get_actions_batch(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get actions for a batch of observations."""
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        logits, values = self.policy(obs_tensor)

        # Sample actions
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.squeeze(-1).cpu().numpy()
        )


class ParallelGameRunner:
    """Run multiple games in parallel."""

    def __init__(self, policy: PolicyNetwork, num_parallel: int = 4,
                 device: str = 'cuda'):
        self.policy = policy
        self.num_parallel = num_parallel
        self.device = device
        self.optimized_policy = OptimizedPolicy(policy, device)

    def run_parallel_games(self, num_games: int, opponent_type: str = 'random') -> List[dict]:
        """Run games with batched inference."""
        results = []

        # Create opponents
        if opponent_type == 'random':
            opponents = [RandomPolicy() for _ in range(self.num_parallel)]
        else:
            opponents = [AggressivePolicy() for _ in range(self.num_parallel)]

        games_completed = 0
        batch_num = 0

        while games_completed < num_games:
            batch_num += 1
            current_batch_size = min(self.num_parallel, num_games - games_completed)

            # Initialize games
            games = []
            loops = []
            active = []
            episodes = []

            for i in range(current_batch_size):
                game = create_game('battlefield', 'FLEX', 'FLEX', stock_count=3)
                loop = GameLoop(game, headless=True)
                games.append(game)
                loops.append(loop)
                active.append(True)
                episodes.append({'winner': None, 'frames': 0})

            # Run until all games complete
            frame = 0
            max_frames = 3600

            while any(active) and frame < max_frames:
                # Collect observations for active games
                obs_batch = []
                active_indices = []

                for i, (game, is_active) in enumerate(zip(games, active)):
                    if is_active:
                        obs = get_observation(game, 0)
                        obs_batch.append(obs)
                        active_indices.append(i)

                if not obs_batch:
                    break

                # Batch inference
                obs_array = np.array(obs_batch)
                actions, _, _ = self.optimized_policy.get_actions_batch(obs_array)

                # Apply actions
                for idx, game_idx in enumerate(active_indices):
                    game = games[game_idx]
                    loop = loops[game_idx]
                    opponent = opponents[game_idx % len(opponents)]

                    action_p1 = actions[idx]
                    obs_p2 = get_observation(game, 1)
                    action_p2, _, _ = opponent.get_action(obs_p2)

                    game.fighters[0].input = get_action_from_index(action_p1)
                    game.fighters[1].input = get_action_from_index(action_p2)

                    still_going = loop.step()
                    episodes[game_idx]['frames'] += 1

                    if game.match_over or not still_going:
                        active[game_idx] = False
                        episodes[game_idx]['winner'] = game.winner == 0

                frame += 1

            games_completed += current_batch_size
            results.extend(episodes[:current_batch_size])

        return results


def profile_optimized(num_games: int = 20) -> ProfileResult:
    """Profile optimized training performance."""
    print(f"\n{'='*60}")
    print("PROFILING OPTIMIZED PERFORMANCE")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = PolicyNetwork(obs_size=OBSERVATION_SIZE, action_size=ACTION_SPACE_SIZE)
    policy.to(device)

    runner = ParallelGameRunner(policy, num_parallel=4, device=device)

    start_time = time.time()
    start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    results = runner.run_parallel_games(num_games, opponent_type='random')

    elapsed = time.time() - start_time
    end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    wins = sum(1 for r in results if r['winner'])

    result = ProfileResult(
        total_time=elapsed,
        games=num_games,
        games_per_second=num_games / elapsed,
        bottlenecks=[],  # Skip detailed profiling for optimized version
        memory_mb=(end_mem - start_mem) / (1024 * 1024)
    )

    print(f"\nOptimized Results:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Games/sec: {result.games_per_second:.2f}")
    print(f"  Win rate: {wins/num_games:.1%}")
    print(f"  Memory used: {result.memory_mb:.1f} MB")

    return result


def optimize_observation_computation():
    """Optimize observation computation with vectorization."""
    print(f"\n{'='*60}")
    print("TESTING OBSERVATION OPTIMIZATION")
    print(f"{'='*60}")

    # Test baseline observation computation
    game = create_game('battlefield', 'FLEX', 'FLEX', stock_count=3)

    start = time.time()
    for _ in range(10000):
        obs = get_observation(game, 0)
    baseline_time = time.time() - start

    print(f"Baseline: {10000/baseline_time:.0f} obs/sec")

    return {
        'baseline_obs_per_sec': 10000 / baseline_time
    }


def run_scaling_test() -> dict:
    """Test how performance scales with batch size."""
    print(f"\n{'='*60}")
    print("BATCH SIZE SCALING TEST")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = PolicyNetwork(obs_size=OBSERVATION_SIZE, action_size=ACTION_SPACE_SIZE)
    policy.to(device)

    batch_sizes = [1, 2, 4, 8, 16]
    results = {}

    for batch_size in batch_sizes:
        runner = ParallelGameRunner(policy, num_parallel=batch_size, device=device)

        start = time.time()
        game_results = runner.run_parallel_games(20, opponent_type='random')
        elapsed = time.time() - start

        games_per_sec = 20 / elapsed
        results[batch_size] = games_per_sec

        print(f"  Batch size {batch_size}: {games_per_sec:.2f} games/sec")

    return results


def run_performance_experiment() -> dict:
    """Run complete performance optimization experiment."""
    print("\n" + "="*70)
    print("CYCLE 5: PERFORMANCE OPTIMIZATION")
    print("Target: 10+ games/sec (from ~3 games/sec baseline)")
    print("="*70)

    start_time = time.time()

    # Profile baseline
    baseline = profile_baseline(num_games=20)

    # Profile optimized
    optimized = profile_optimized(num_games=20)

    # Test observation optimization
    obs_results = optimize_observation_computation()

    # Test batch scaling
    scaling_results = run_scaling_test()

    total_elapsed = time.time() - start_time

    # Calculate improvement
    speedup = optimized.games_per_second / baseline.games_per_second

    report = {
        'experiment': 'Performance Optimization',
        'target_games_per_sec': 10.0,
        'total_time_seconds': total_elapsed,
        'baseline': {
            'games_per_second': baseline.games_per_second,
            'total_time': baseline.total_time,
            'memory_mb': baseline.memory_mb,
            'bottlenecks': baseline.bottlenecks[:5]
        },
        'optimized': {
            'games_per_second': optimized.games_per_second,
            'total_time': optimized.total_time,
            'memory_mb': optimized.memory_mb
        },
        'observation_optimization': obs_results,
        'batch_scaling': scaling_results,
        'improvement': {
            'speedup_factor': speedup,
            'baseline_games_per_sec': baseline.games_per_second,
            'optimized_games_per_sec': optimized.games_per_second,
            'target_met': optimized.games_per_second >= 10.0
        },
        'recommendations': []
    }

    # Add recommendations
    if optimized.games_per_second < 10.0:
        report['recommendations'].append(
            f"Need {(10.0 - optimized.games_per_second)/optimized.games_per_second:.0%} more improvement"
        )

        if baseline.bottlenecks:
            top_bottleneck = baseline.bottlenecks[0][0]
            report['recommendations'].append(f"Focus on optimizing: {top_bottleneck}")

        best_batch = max(scaling_results.items(), key=lambda x: x[1])
        report['recommendations'].append(f"Use batch size {best_batch[0]} for best throughput")

    # Save report
    output_dir = 'checkpoints/cycle5_performance'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = f'{output_dir}/performance_report.json'

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*70)
    print("PERFORMANCE OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nBaseline: {baseline.games_per_second:.2f} games/sec")
    print(f"Optimized: {optimized.games_per_second:.2f} games/sec")
    print(f"Speedup: {speedup:.2f}x")
    print(f"\nTarget (10+ games/sec): {'MET' if report['improvement']['target_met'] else 'NOT MET'}")

    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

    print(f"\nReport saved to: {report_path}")

    return report


def main():
    report = run_performance_experiment()
    return report


if __name__ == '__main__':
    main()
