#!/usr/bin/env python3
"""
Behavior Analysis & Taxonomy for Stick Figure Fighter ML Agents.

Analyzes trained policies to identify:
- Emergent combat strategies (combos, spacing, edge-guarding)
- Action sequence patterns
- Character-specific behaviors
- Fighting game behavior taxonomy
"""
import os
import sys
import json
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':99'

import torch
from policy import PolicyNetwork
from game_engine import (
    create_game, GameLoop, get_observation, get_action_from_index,
    OBSERVATION_SIZE, ACTION_SPACE_SIZE, GameState, FighterState
)
from evaluate import RandomPolicy


# Action indices mapping
ACTION_NAMES = {
    0: 'Nothing',
    1: 'Left',
    2: 'Right',
    3: 'Jump',
    4: 'Down',
    5: 'Attack',
    6: 'Left+Attack',
    7: 'Right+Attack',
    8: 'Up+Attack',
    9: 'Down+Attack'
}

# Fighting game behavior categories
BEHAVIOR_CATEGORIES = {
    'neutral': ['spacing', 'movement', 'poking'],
    'offense': ['pressure', 'combos', 'mixups'],
    'defense': ['blocking', 'retreating', 'punishing'],
    'edgeguard': ['offstage_chase', 'ledge_trap', 'recovery_denial'],
    'recovery': ['returning_to_stage', 'avoiding_edgeguard']
}


class ActionSequenceAnalyzer:
    """Analyzes action sequences for patterns and combos."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.sequences: List[List[int]] = []
        self.sequence_counts = Counter()

    def record_sequence(self, actions: List[int]):
        """Record an action sequence for analysis."""
        self.sequences.append(actions)

        # Count n-grams of different sizes
        for n in range(2, min(6, len(actions) + 1)):
            for i in range(len(actions) - n + 1):
                seq = tuple(actions[i:i+n])
                self.sequence_counts[seq] += 1

    def get_common_sequences(self, min_length: int = 2, top_k: int = 20) -> List[Tuple[tuple, int]]:
        """Get most common action sequences."""
        filtered = [(seq, count) for seq, count in self.sequence_counts.items()
                    if len(seq) >= min_length]
        return sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]

    def identify_combo_patterns(self) -> Dict[str, List[Tuple[tuple, int]]]:
        """Identify potential combo patterns (attack sequences)."""
        combos = {
            'short_combos': [],  # 2-3 hit
            'medium_combos': [],  # 4-5 hit
            'long_combos': []  # 6+ hit
        }

        attack_actions = {5, 6, 7, 8, 9}  # All attack-related actions

        for seq, count in self.sequence_counts.items():
            # Check if sequence is mostly attacks
            attack_ratio = sum(1 for a in seq if a in attack_actions) / len(seq)
            if attack_ratio >= 0.6 and count >= 3:
                seq_len = len(seq)
                if seq_len <= 3:
                    combos['short_combos'].append((seq, count))
                elif seq_len <= 5:
                    combos['medium_combos'].append((seq, count))
                else:
                    combos['long_combos'].append((seq, count))

        # Sort each category
        for key in combos:
            combos[key] = sorted(combos[key], key=lambda x: x[1], reverse=True)[:10]

        return combos


class PositionalAnalyzer:
    """Analyzes positional behaviors (spacing, edge-guarding, etc.)."""

    def __init__(self, stage_width: int = 1100, stage_center: int = 550):
        self.stage_width = stage_width
        self.stage_center = stage_center

        # Position-action correlation
        self.position_actions = defaultdict(lambda: defaultdict(int))

        # Edge scenarios
        self.edge_scenarios = {
            'near_left_edge': [],  # x < 200
            'near_right_edge': [],  # x > 900
            'center_stage': [],  # 400 < x < 700
            'offstage_left': [],  # x < 100
            'offstage_right': [],  # x > 1000
        }

    def record_position_action(self, x: float, y: float, action: int, opponent_x: float, opponent_y: float):
        """Record action taken at a specific position."""
        # Bucket positions
        x_bucket = int(x // 100) * 100
        y_bucket = int(y // 50) * 50

        self.position_actions[(x_bucket, y_bucket)][action] += 1

        # Categorize edge scenarios
        if x < 200:
            self.edge_scenarios['near_left_edge'].append(action)
        elif x > 900:
            self.edge_scenarios['near_right_edge'].append(action)
        elif 400 < x < 700:
            self.edge_scenarios['center_stage'].append(action)

        if x < 100:
            self.edge_scenarios['offstage_left'].append(action)
        elif x > 1000:
            self.edge_scenarios['offstage_right'].append(action)

    def get_edge_behavior_summary(self) -> Dict:
        """Summarize edge-related behaviors."""
        summary = {}

        for scenario, actions in self.edge_scenarios.items():
            if len(actions) > 10:
                action_dist = Counter(actions)
                total = len(actions)
                top_actions = action_dist.most_common(5)

                summary[scenario] = {
                    'sample_size': total,
                    'top_actions': [(ACTION_NAMES[a], c/total) for a, c in top_actions],
                    'attack_ratio': sum(1 for a in actions if a >= 5) / total,
                    'movement_ratio': sum(1 for a in actions if a in [1, 2]) / total
                }

        return summary


class DamageStateAnalyzer:
    """Analyzes behavior changes based on damage state."""

    def __init__(self):
        # Action distributions at different damage states
        self.damage_state_actions = {
            'low_damage': [],  # 0-40%
            'mid_damage': [],  # 40-80%
            'high_damage': [],  # 80-120%
            'danger': []  # 120%+
        }

        self.opponent_damage_state_actions = {
            'opponent_low': [],
            'opponent_mid': [],
            'opponent_high': [],
            'opponent_danger': []
        }

    def record_damage_state(self, own_damage: float, opponent_damage: float, action: int):
        """Record action taken at specific damage states."""
        # Own damage categories
        if own_damage < 40:
            self.damage_state_actions['low_damage'].append(action)
        elif own_damage < 80:
            self.damage_state_actions['mid_damage'].append(action)
        elif own_damage < 120:
            self.damage_state_actions['high_damage'].append(action)
        else:
            self.damage_state_actions['danger'].append(action)

        # Opponent damage categories
        if opponent_damage < 40:
            self.opponent_damage_state_actions['opponent_low'].append(action)
        elif opponent_damage < 80:
            self.opponent_damage_state_actions['opponent_mid'].append(action)
        elif opponent_damage < 120:
            self.opponent_damage_state_actions['opponent_high'].append(action)
        else:
            self.opponent_damage_state_actions['opponent_danger'].append(action)

    def get_damage_behavior_changes(self) -> Dict:
        """Analyze how behavior changes with damage."""
        own_damage_summary = {}
        opp_damage_summary = {}

        for state, actions in self.damage_state_actions.items():
            if len(actions) > 20:
                action_dist = Counter(actions)
                total = len(actions)
                own_damage_summary[state] = {
                    'attack_ratio': sum(1 for a in actions if a >= 5) / total,
                    'defensive_ratio': sum(1 for a in actions if a in [0, 1, 2, 3]) / total,
                    'top_action': ACTION_NAMES[action_dist.most_common(1)[0][0]]
                }

        for state, actions in self.opponent_damage_state_actions.items():
            if len(actions) > 20:
                action_dist = Counter(actions)
                total = len(actions)
                opp_damage_summary[state] = {
                    'attack_ratio': sum(1 for a in actions if a >= 5) / total,
                    'top_action': ACTION_NAMES[action_dist.most_common(1)[0][0]]
                }

        return {
            'own_damage_behavior': own_damage_summary,
            'opponent_damage_behavior': opp_damage_summary
        }


def run_comprehensive_analysis(
    policy_path: str,
    character: str = 'FLEX',
    num_games: int = 30
) -> Dict:
    """
    Run comprehensive behavior analysis on a trained policy.
    """
    print(f'\n{"="*60}')
    print(f'COMPREHENSIVE BEHAVIOR ANALYSIS')
    print(f'{"="*60}')
    print(f'Policy: {policy_path}')
    print(f'Character: {character}')
    print(f'Games: {num_games}')

    # Load policy
    policy = PolicyNetwork(obs_size=OBSERVATION_SIZE, action_size=ACTION_SPACE_SIZE)
    policy.to('cuda')
    policy.load(policy_path)

    # Initialize analyzers
    seq_analyzer = ActionSequenceAnalyzer()
    pos_analyzer = PositionalAnalyzer()
    dmg_analyzer = DamageStateAnalyzer()

    random_opp = RandomPolicy()

    # Collect data across games
    all_actions = []
    wins = 0
    total_damage_dealt = 0
    total_stocks_taken = 0

    for game_idx in range(num_games):
        game = create_game(
            stage_name='battlefield',
            p1_character=character,
            p2_character='FLEX',
            stock_count=3
        )
        loop = GameLoop(game, headless=True)

        game_actions = []
        frame = 0

        while frame < 10800 and not game.match_over:
            obs = get_observation(game, 0)
            obs_p2 = get_observation(game, 1)

            action, _, _ = policy.get_action(obs)
            action_p2, _, _ = random_opp.get_action(obs_p2)

            # Record for analysis
            game_actions.append(action)
            all_actions.append(action)

            # Position analysis
            p1 = game.fighters[0]
            p2 = game.fighters[1]
            pos_analyzer.record_position_action(
                p1.position.x, p1.position.y, action,
                p2.position.x, p2.position.y
            )

            # Damage state analysis
            dmg_analyzer.record_damage_state(
                p1.damage_percent, p2.damage_percent, action
            )

            # Apply actions
            game.fighters[0].input = get_action_from_index(action)
            game.fighters[1].input = get_action_from_index(action_p2)

            prev_p2_stocks = p2.stocks
            prev_p2_dmg = p2.damage_percent

            loop.step()
            frame += 1

            if p2.stocks < prev_p2_stocks:
                total_stocks_taken += 1
            total_damage_dealt += max(0, p2.damage_percent - prev_p2_dmg)

        # Record sequence for this game
        seq_analyzer.record_sequence(game_actions)

        if game.winner == 0:
            wins += 1

        if (game_idx + 1) % 10 == 0:
            print(f'  Processed {game_idx + 1}/{num_games} games...')

    # Generate analysis report
    print('\nGenerating analysis report...')

    # Overall action distribution
    action_counts = Counter(all_actions)
    total_actions = len(all_actions)
    action_dist = {ACTION_NAMES[a]: c/total_actions for a, c in action_counts.items()}

    # Behavior classification
    attack_ratio = sum(1 for a in all_actions if a >= 5) / total_actions
    movement_ratio = sum(1 for a in all_actions if a in [1, 2]) / total_actions
    jump_ratio = sum(1 for a in all_actions if a == 3) / total_actions
    idle_ratio = sum(1 for a in all_actions if a == 0) / total_actions

    # Classify playstyle
    if attack_ratio > 0.45:
        playstyle = 'Hyper-Aggressive'
    elif attack_ratio > 0.35:
        playstyle = 'Aggressive'
    elif attack_ratio > 0.25:
        playstyle = 'Balanced'
    elif movement_ratio > 0.3:
        playstyle = 'Evasive'
    else:
        playstyle = 'Passive'

    # Combo analysis
    combos = seq_analyzer.identify_combo_patterns()

    # Edge behavior
    edge_behavior = pos_analyzer.get_edge_behavior_summary()

    # Damage state behavior
    damage_behavior = dmg_analyzer.get_damage_behavior_changes()

    report = {
        'summary': {
            'character': character,
            'games_analyzed': num_games,
            'win_rate': wins / num_games,
            'avg_damage_dealt': total_damage_dealt / num_games,
            'avg_stocks_taken': total_stocks_taken / num_games,
            'playstyle': playstyle
        },
        'action_distribution': action_dist,
        'behavior_ratios': {
            'attack_ratio': attack_ratio,
            'movement_ratio': movement_ratio,
            'jump_ratio': jump_ratio,
            'idle_ratio': idle_ratio
        },
        'combo_patterns': {
            'short_combos': [(tuple(ACTION_NAMES[a] for a in seq), count)
                            for seq, count in combos['short_combos'][:5]],
            'medium_combos': [(tuple(ACTION_NAMES[a] for a in seq), count)
                             for seq, count in combos['medium_combos'][:5]]
        },
        'common_sequences': [(tuple(ACTION_NAMES[a] for a in seq), count)
                            for seq, count in seq_analyzer.get_common_sequences()[:10]],
        'edge_behavior': edge_behavior,
        'damage_state_behavior': damage_behavior
    }

    # Print summary
    print('\n--- ANALYSIS RESULTS ---')
    print(f'\nPlaystyle: {playstyle}')
    print(f'Win rate: {wins/num_games:.1%}')
    print(f'Avg damage dealt: {total_damage_dealt/num_games:.1f}')
    print(f'Avg stocks taken: {total_stocks_taken/num_games:.2f}')

    print(f'\nBehavior Ratios:')
    print(f'  Attack: {attack_ratio:.1%}')
    print(f'  Movement: {movement_ratio:.1%}')
    print(f'  Jump: {jump_ratio:.1%}')
    print(f'  Idle: {idle_ratio:.1%}')

    print(f'\nTop Actions:')
    sorted_actions = sorted(action_dist.items(), key=lambda x: x[1], reverse=True)
    for action, prob in sorted_actions[:5]:
        print(f'  {action}: {prob:.1%}')

    print(f'\nCommon Sequences:')
    for seq, count in seq_analyzer.get_common_sequences()[:5]:
        seq_names = [ACTION_NAMES[a] for a in seq]
        print(f'  {" -> ".join(seq_names)}: {count}x')

    if combos['short_combos']:
        print(f'\nPotential Combos:')
        for seq, count in combos['short_combos'][:3]:
            seq_names = [ACTION_NAMES[a] for a in seq]
            print(f'  {" -> ".join(seq_names)}: {count}x')

    return report


def generate_behavior_taxonomy(analysis_report: Dict) -> str:
    """
    Generate a formatted behavior taxonomy document.
    """
    doc = []
    doc.append('# Stick Figure Fighter - ML Agent Behavior Taxonomy')
    doc.append('')
    doc.append('## Overview')
    doc.append(f'- Character: {analysis_report["summary"]["character"]}')
    doc.append(f'- Playstyle: {analysis_report["summary"]["playstyle"]}')
    doc.append(f'- Win Rate: {analysis_report["summary"]["win_rate"]:.1%}')
    doc.append('')

    doc.append('## 1. Action Distribution')
    for action, prob in sorted(analysis_report['action_distribution'].items(),
                                key=lambda x: x[1], reverse=True):
        bar = '#' * int(prob * 50)
        doc.append(f'  {action:15s} [{bar:50s}] {prob:.1%}')
    doc.append('')

    doc.append('## 2. Behavioral Ratios')
    ratios = analysis_report['behavior_ratios']
    doc.append(f'  Attack:    {ratios["attack_ratio"]:.1%}')
    doc.append(f'  Movement:  {ratios["movement_ratio"]:.1%}')
    doc.append(f'  Aerial:    {ratios["jump_ratio"]:.1%}')
    doc.append(f'  Passive:   {ratios["idle_ratio"]:.1%}')
    doc.append('')

    doc.append('## 3. Emergent Patterns')
    doc.append('')
    doc.append('### Common Action Sequences')
    for seq, count in analysis_report['common_sequences'][:8]:
        doc.append(f'  {" -> ".join(seq)}: {count}x')
    doc.append('')

    doc.append('### Potential Combo Patterns')
    if analysis_report['combo_patterns']['short_combos']:
        for seq, count in analysis_report['combo_patterns']['short_combos'][:5]:
            doc.append(f'  {" -> ".join(seq)}: {count}x')
    else:
        doc.append('  No significant combo patterns detected.')
    doc.append('')

    doc.append('## 4. Positional Behaviors')
    doc.append('')
    edge = analysis_report.get('edge_behavior', {})
    for scenario, data in edge.items():
        doc.append(f'### {scenario.replace("_", " ").title()}')
        doc.append(f'  Samples: {data.get("sample_size", 0)}')
        doc.append(f'  Attack tendency: {data.get("attack_ratio", 0):.1%}')
        doc.append(f'  Movement tendency: {data.get("movement_ratio", 0):.1%}')
        if 'top_actions' in data:
            top = ', '.join([f'{a}({p:.0%})' for a, p in data['top_actions'][:3]])
            doc.append(f'  Top actions: {top}')
        doc.append('')

    doc.append('## 5. Damage State Adaptation')
    doc.append('')
    dmg = analysis_report.get('damage_state_behavior', {})

    if 'own_damage_behavior' in dmg:
        doc.append('### When Taking Damage')
        for state, data in dmg['own_damage_behavior'].items():
            doc.append(f'  {state.replace("_", " ").title()}:')
            doc.append(f'    Attack ratio: {data.get("attack_ratio", 0):.1%}')
            doc.append(f'    Top action: {data.get("top_action", "N/A")}')
        doc.append('')

    if 'opponent_damage_behavior' in dmg:
        doc.append('### When Opponent is Damaged')
        for state, data in dmg['opponent_damage_behavior'].items():
            doc.append(f'  {state.replace("_", " ").title()}:')
            doc.append(f'    Attack ratio: {data.get("attack_ratio", 0):.1%}')
            doc.append(f'    Top action: {data.get("top_action", "N/A")}')
    doc.append('')

    doc.append('## 6. Fighting Game Behavior Classifications')
    doc.append('')

    # Classify observed behaviors
    playstyle = analysis_report['summary']['playstyle']
    ratios = analysis_report['behavior_ratios']

    doc.append('### Neutral Game')
    if ratios['movement_ratio'] > 0.25:
        doc.append('  - Uses movement for spacing and positioning')
    if ratios['attack_ratio'] < 0.3:
        doc.append('  - Plays patient, waits for openings')
    else:
        doc.append('  - Aggressive approach, constant pressure')
    doc.append('')

    doc.append('### Offense')
    if 'combo_patterns' in analysis_report and analysis_report['combo_patterns']['short_combos']:
        doc.append('  - Shows evidence of combo awareness')
    if ratios['attack_ratio'] > 0.35:
        doc.append('  - High attack frequency suggests pressure-focused playstyle')
    doc.append('')

    doc.append('### Edge Situations')
    if 'near_left_edge' in edge or 'near_right_edge' in edge:
        near_edge = edge.get('near_left_edge', edge.get('near_right_edge', {}))
        if near_edge.get('attack_ratio', 0) > 0.4:
            doc.append('  - Aggressively pursues opponents near edge')
        else:
            doc.append('  - Conservative near edge, focuses on positioning')
    doc.append('')

    return '\n'.join(doc)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze trained policy behavior')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/full_run/checkpoint_001000.npz',
                        help='Path to policy checkpoint')
    parser.add_argument('--character', type=str, default='FLEX',
                        choices=['FLEX', 'RUSH', 'ZONE', 'GRAB'])
    parser.add_argument('--games', type=int, default=30,
                        help='Number of games to analyze')
    parser.add_argument('--output', type=str, default='artifacts',
                        help='Output directory')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    report = run_comprehensive_analysis(
        args.checkpoint,
        args.character,
        args.games
    )

    # Save JSON report
    report_path = f'{args.output}/behavior_analysis.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\nJSON report saved to: {report_path}')

    # Generate taxonomy document
    taxonomy = generate_behavior_taxonomy(report)
    taxonomy_path = f'{args.output}/behavior_taxonomy.md'
    with open(taxonomy_path, 'w') as f:
        f.write(taxonomy)
    print(f'Taxonomy document saved to: {taxonomy_path}')


if __name__ == '__main__':
    main()
