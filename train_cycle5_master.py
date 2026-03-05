#!/usr/bin/env python3
"""
Cycle 5: Master Training Script

This script orchestrates all Cycle 5 experiments:
1. Defensive Training Experiment
2. Character-Specific Training
3. Meta-Game Analysis
4. Performance Optimization
5. Final Report Generation

Run this script to execute the complete Cycle 5 mission.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'


def run_defensive_experiment(quick: bool = False):
    """Run the defensive vs aggressive experiment."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: DEFENSIVE TRAINING")
    print("="*70)

    from train_cycle5_defensive import DefensiveExperiment

    experiment = DefensiveExperiment(
        device='cuda',
        output_dir='checkpoints/cycle5_defensive'
    )

    training_games = 200 if quick else 600
    report = experiment.run_experiment(training_games=training_games)

    return report


def run_character_specific_experiment(quick: bool = False):
    """Run the character-specific training experiment."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: CHARACTER-SPECIFIC TRAINING")
    print("="*70)

    from train_cycle5_character_specific import CharacterSpecificExperiment

    experiment = CharacterSpecificExperiment(
        device='cuda',
        output_dir='checkpoints/cycle5_char_specific'
    )

    games_per_char = 150 if quick else 400
    report = experiment.run_experiment(games_per_character=games_per_char)

    return report


def run_metagame_analysis():
    """Run the meta-game analysis."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: META-GAME ANALYSIS")
    print("="*70)

    from train_cycle5_metagame import MetaGameExperiment

    experiment = MetaGameExperiment(
        device='cuda',
        output_dir='checkpoints/cycle5_metagame'
    )

    report = experiment.run_experiment()

    return report


def run_performance_optimization():
    """Run the performance optimization profiling."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: PERFORMANCE OPTIMIZATION")
    print("="*70)

    from train_cycle5_perf_optimize import run_performance_experiment

    report = run_performance_experiment()

    return report


def generate_final_report():
    """Generate the comprehensive final report."""
    print("\n" + "="*70)
    print("GENERATING FINAL REPORT")
    print("="*70)

    from generate_final_report import generate_final_report as gen_report

    report = gen_report()

    return report


def run_all_experiments(quick: bool = False, skip_experiments: list = None):
    """Run all Cycle 5 experiments."""
    if skip_experiments is None:
        skip_experiments = []

    print("\n" + "="*70)
    print("CYCLE 5: MASTER TRAINING SCRIPT")
    print("Final Cycle - Complete ML Self-Play Training Mission")
    print("="*70)
    print(f"\nMode: {'QUICK (reduced games)' if quick else 'FULL'}")
    print(f"Skip: {skip_experiments if skip_experiments else 'None'}")

    start_time = time.time()
    results = {}

    # Experiment 1: Defensive Training
    if 'defensive' not in skip_experiments:
        try:
            results['defensive'] = run_defensive_experiment(quick=quick)
            print("\nDefensive experiment completed successfully!")
        except Exception as e:
            print(f"\nDefensive experiment failed: {e}")
            results['defensive'] = {'error': str(e)}
    else:
        print("\nSkipping defensive experiment")

    # Experiment 2: Character-Specific Training
    if 'character' not in skip_experiments:
        try:
            results['character_specific'] = run_character_specific_experiment(quick=quick)
            print("\nCharacter-specific experiment completed successfully!")
        except Exception as e:
            print(f"\nCharacter-specific experiment failed: {e}")
            results['character_specific'] = {'error': str(e)}
    else:
        print("\nSkipping character-specific experiment")

    # Experiment 3: Meta-Game Analysis
    if 'metagame' not in skip_experiments:
        try:
            results['metagame'] = run_metagame_analysis()
            print("\nMeta-game analysis completed successfully!")
        except Exception as e:
            print(f"\nMeta-game analysis failed: {e}")
            results['metagame'] = {'error': str(e)}
    else:
        print("\nSkipping meta-game analysis")

    # Experiment 4: Performance Optimization
    if 'performance' not in skip_experiments:
        try:
            results['performance'] = run_performance_optimization()
            print("\nPerformance optimization completed successfully!")
        except Exception as e:
            print(f"\nPerformance optimization failed: {e}")
            results['performance'] = {'error': str(e)}
    else:
        print("\nSkipping performance optimization")

    # Final Report
    if 'report' not in skip_experiments:
        try:
            results['final_report'] = generate_final_report()
            print("\nFinal report generated successfully!")
        except Exception as e:
            print(f"\nFinal report generation failed: {e}")
            results['final_report'] = {'error': str(e)}
    else:
        print("\nSkipping final report generation")

    total_elapsed = time.time() - start_time

    # Save master results
    master_report = {
        'cycle': 5,
        'mission': 'Final Cycle - ML Self-Play Training',
        'mode': 'quick' if quick else 'full',
        'total_time_seconds': total_elapsed,
        'completed_at': datetime.now().isoformat(),
        'experiments': {
            'defensive': 'defensive' not in skip_experiments,
            'character_specific': 'character' not in skip_experiments,
            'metagame': 'metagame' not in skip_experiments,
            'performance': 'performance' not in skip_experiments,
            'final_report': 'report' not in skip_experiments
        },
        'results_summary': {}
    }

    # Extract key results
    if 'defensive' in results and 'error' not in results['defensive']:
        master_report['results_summary']['defensive'] = {
            'head_to_head': results['defensive'].get('head_to_head', {}),
            'conclusion': results['defensive'].get('conclusions', {}).get('head_to_head', 'N/A')
        }

    if 'character_specific' in results and 'error' not in results['character_specific']:
        master_report['results_summary']['character_specific'] = {
            'conclusion': results['character_specific'].get('conclusions', {}).get('overall', 'N/A'),
            'best_specialist': results['character_specific'].get('conclusions', {}).get('best_specialist', 'N/A')
        }

    if 'metagame' in results and 'error' not in results['metagame']:
        master_report['results_summary']['metagame'] = {
            'optimality': results['metagame'].get('conclusions', {}).get('optimality', 'N/A'),
            'worst_matchup': results['metagame'].get('conclusions', {}).get('worst_matchup', 'N/A')
        }

    if 'performance' in results and 'error' not in results['performance']:
        master_report['results_summary']['performance'] = {
            'baseline_gps': results['performance'].get('baseline', {}).get('games_per_second', 0),
            'optimized_gps': results['performance'].get('optimized', {}).get('games_per_second', 0),
            'target_met': results['performance'].get('improvement', {}).get('target_met', False)
        }

    # Save master report
    Path('checkpoints/cycle5_master').mkdir(parents=True, exist_ok=True)
    master_path = 'checkpoints/cycle5_master/cycle5_master_report.json'
    with open(master_path, 'w') as f:
        json.dump(master_report, f, indent=2)

    # Print final summary
    print("\n" + "="*70)
    print("CYCLE 5 COMPLETE")
    print("="*70)
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print(f"\nExperiments completed:")

    for exp, completed in master_report['experiments'].items():
        status = "DONE" if completed else "SKIPPED"
        if exp in results and 'error' in results.get(exp, {}):
            status = "FAILED"
        print(f"  {exp}: {status}")

    print(f"\nResults Summary:")
    for exp, summary in master_report['results_summary'].items():
        print(f"\n  {exp}:")
        for key, value in summary.items():
            print(f"    {key}: {value}")

    print(f"\nMaster report saved to: {master_path}")
    print(f"Final documentation: artifacts/final_mission_report.md")

    return master_report


def main():
    parser = argparse.ArgumentParser(description='Cycle 5 Master Training Script')
    parser.add_argument('--quick', action='store_true',
                        help='Run in quick mode with reduced games')
    parser.add_argument('--skip', nargs='*', default=[],
                        choices=['defensive', 'character', 'metagame', 'performance', 'report'],
                        help='Experiments to skip')
    parser.add_argument('--only', type=str,
                        choices=['defensive', 'character', 'metagame', 'performance', 'report'],
                        help='Run only a specific experiment')

    args = parser.parse_args()

    if args.only:
        # Run only the specified experiment
        skip = ['defensive', 'character', 'metagame', 'performance', 'report']
        skip.remove(args.only)
        return run_all_experiments(quick=args.quick, skip_experiments=skip)
    else:
        return run_all_experiments(quick=args.quick, skip_experiments=args.skip)


if __name__ == '__main__':
    main()
