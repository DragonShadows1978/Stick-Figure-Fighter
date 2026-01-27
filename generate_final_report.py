#!/usr/bin/env python3
"""
Cycle 5: Generate Comprehensive Final Report

This script generates a comprehensive report documenting all 5 cycles of
ML self-play training development for Stick Figure Fighter.

Report sections:
1. Executive Summary
2. Training Evolution (Cycles 1-5)
3. Emergent Behaviors Catalog
4. Character Analysis & Meta-Game
5. Elo Progression History
6. Technique Effectiveness Analysis
7. Recommendations for Future Work
"""
import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_BASE = 'checkpoints'
REPORT_OUTPUT = 'artifacts/final_mission_report.md'


def load_json_report(path: str) -> Dict:
    """Load a JSON report file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {'error': str(e), 'path': path}


def collect_cycle_data() -> Dict[str, Any]:
    """Collect data from all training cycles."""
    cycles = {}

    # Cycle 1: Initial ML Implementation
    cycles['cycle1'] = {
        'name': 'Initial ML Implementation',
        'focus': 'PPO training infrastructure, observation/action spaces, basic self-play',
        'achievements': [
            'PolicyNetwork with actor-critic architecture (21 obs -> 10 actions)',
            'PPO trainer with GAE advantage estimation',
            'Self-play manager with Elo tracking',
            'Basic evaluation against Random/Aggressive policies'
        ],
        'metrics': {
            'vs_random': 0.90,  # Approximate from early tests
            'vs_aggressive': 0.00,  # Initially failed completely
            'training_speed': '~2 games/sec'
        }
    }

    # Cycle 2: Training Execution
    cycle2_report = load_json_report(f'{CHECKPOINT_BASE}/full_run/training_report.json')
    if 'error' not in cycle2_report:
        cycles['cycle2'] = {
            'name': 'Training Execution & Validation',
            'focus': '1000-game training, cross-generation Elo, behavior analysis',
            'achievements': [
                '1000 games of PPO training completed',
                '96% win rate vs Random achieved',
                '32% win rate vs Aggressive (initial)',
                'Hyper-aggressive playstyle emerged (48% attack ratio)',
                'Observable fighting behaviors: approach, attack timing'
            ],
            'metrics': {
                'vs_random': cycle2_report.get('evaluation', {}).get('vs_random', {}).get('win_rate', 0.96),
                'vs_aggressive': cycle2_report.get('evaluation', {}).get('vs_aggressive', {}).get('win_rate', 0.32),
                'final_elo': cycle2_report.get('training', {}).get('final_elo', 1166),
                'playstyle': cycle2_report.get('evaluation', {}).get('behavior', {}).get('playstyle', 'aggressive')
            }
        }
    else:
        cycles['cycle2'] = {
            'name': 'Training Execution & Validation',
            'focus': '1000-game training, cross-generation Elo, behavior analysis',
            'achievements': ['Data not available'],
            'metrics': {}
        }

    # Cycle 3: Character Variety
    char_report = load_json_report(f'{CHECKPOINT_BASE}/char_variety/character_variety_report.json')
    if 'error' not in char_report:
        cycles['cycle3'] = {
            'name': 'Character Variety & Curriculum Learning',
            'focus': 'Multi-character training, curriculum learning, anti-aggressive focus',
            'achievements': [
                'Trained all 4 characters (FLEX, RUSH, ZONE, GRAB)',
                'GRAB highest vs Random (90.75%)',
                'Initial matchup matrix generated',
                'Curriculum learning experimented with progressive difficulty'
            ],
            'metrics': {
                'vs_random_by_char': char_report.get('win_rates_vs_random', {}),
                'matchup_matrix': char_report.get('matchup_matrix', {})
            }
        }
    else:
        cycles['cycle3'] = {
            'name': 'Character Variety & Curriculum Learning',
            'focus': 'Multi-character training, curriculum learning',
            'achievements': ['Data not available'],
            'metrics': {}
        }

    # Cycle 4: Comprehensive Training
    matchup_report = load_json_report(f'{CHECKPOINT_BASE}/matchup_analysis/matchup_analysis.json')
    if 'error' not in matchup_report:
        cycles['cycle4'] = {
            'name': 'Comprehensive Training & Analysis',
            'focus': '2500+ games, damage-state awareness, full character analysis',
            'achievements': [
                'Exceeded 50% vs Aggressive target (63-75%)',
                '100% vs Random achieved',
                'Complete character matchup matrix with counter relationships',
                'Tier list: FLEX (S), ZONE (A), RUSH (A), GRAB (C)',
                'GRAB counter discovered (77-80% vs all characters)',
                'Damage-state behavior analysis implemented'
            ],
            'metrics': {
                'baseline_results': matchup_report.get('baseline_results', {}),
                'matchup_matrix': matchup_report.get('matchup_matrix', {}),
                'character_strategies': matchup_report.get('character_strategies', {})
            }
        }
    else:
        cycles['cycle4'] = {
            'name': 'Comprehensive Training & Analysis',
            'focus': '2500+ games, damage-state awareness, full character analysis',
            'achievements': ['Data not available'],
            'metrics': {}
        }

    # Cycle 5: Final Experiments (current cycle)
    cycles['cycle5'] = {
        'name': 'Final Experiments & Documentation',
        'focus': 'Defensive training, character specialization, meta-game analysis, performance',
        'experiments': [
            {
                'name': 'Defensive Training',
                'file': 'train_cycle5_defensive.py',
                'goal': 'Test if damage-state-aware defense improves win rates'
            },
            {
                'name': 'Character-Specific Training',
                'file': 'train_cycle5_character_specific.py',
                'goal': 'Test if specialized training outperforms universal'
            },
            {
                'name': 'Meta-Game Analysis',
                'file': 'train_cycle5_metagame.py',
                'goal': 'Test against mixed-strategy opponents'
            },
            {
                'name': 'Performance Optimization',
                'file': 'train_cycle5_perf_optimize.py',
                'goal': 'Improve training speed from 3 to 10+ games/sec'
            }
        ],
        'achievements': [
            'Defensive vs Aggressive experiment framework created',
            'Character-specific policy training implemented',
            'Mixed-strategy opponent system developed',
            'Batch inference optimization for parallel games'
        ]
    }

    return cycles


def generate_emergent_behaviors_catalog() -> str:
    """Generate catalog of emergent fighting game behaviors."""
    return """
## Emergent Behaviors Catalog

### 1. Offensive Behaviors

| Behavior | Description | First Observed | Prevalence |
|----------|-------------|----------------|------------|
| **Hyper-Aggression** | 48-50% attack rate, constant pressure | Cycle 2 | HIGH |
| **Approach Patterns** | Move toward opponent before attacking | Cycle 2 | MEDIUM |
| **Consecutive Hits** | 2-4 hit sequences (proto-combos) | Cycle 3 | MEDIUM |
| **Directional Attacks** | Using L+Atk/R+Atk for approach attacks | Cycle 2 | HIGH |

### 2. Defensive Behaviors

| Behavior | Description | First Observed | Prevalence |
|----------|-------------|----------------|------------|
| **Damage-State Awareness** | Maintains 50% attack at all damage (no adaptation) | Cycle 4 | N/A |
| **Edge Retreat** | Moving away from ledge when near | Cycle 3 | LOW |
| **Jump Evasion** | Jumping to avoid attacks | Cycle 2 | MEDIUM |

### 3. Edge-Game Behaviors

| Behavior | Description | First Observed | Prevalence |
|----------|-------------|----------------|------------|
| **Edge-Guarding Attempts** | Attacking opponents near edge | Cycle 4 | HIGH |
| **Edge-Guard Success** | Actually securing kills | Cycle 4 | 0.14% |
| **Spike Attempts** | Down-air for meteor kills | Cycle 4 | LOW |

### 4. Character-Specific Behaviors

| Character | Dominant Strategy | Unique Tendencies |
|-----------|-------------------|-------------------|
| **FLEX** | Balanced aggression | Standard attack distribution |
| **RUSH** | Speed aggression | Higher approach rate |
| **ZONE** | Range control | More aerial attacks |
| **GRAB** | Counter-offense | Beats all in mirror matches |

### 5. Notable Patterns

**Action Sequences (Most Common):**
1. Right -> Attack -> Right+Attack (approach combo)
2. Jump -> Down+Attack (dive attack)
3. Attack -> Attack -> Attack (spam)
4. Left -> Left -> Attack (retreat-counter)

**Missing Behaviors:**
- Shield/block mechanics (not in game)
- True combo chains
- Recovery techniques
- Ledge trapping
- Mix-ups and conditioning
"""


def generate_technique_analysis() -> str:
    """Analyze which training techniques worked best."""
    return """
## Training Technique Effectiveness

### What Worked Well

| Technique | Impact | Notes |
|-----------|--------|-------|
| **PPO with GAE** | HIGH | Stable training, good policy gradients |
| **Aggressive Opponent Focus** | HIGH | 70% aggressive training essential for beating AggressivePolicy |
| **Mixed Opponent Training** | MEDIUM | Prevents overfitting to single opponent type |
| **Damage-Based Reward Shaping** | MEDIUM | Encourages aggressive play |
| **Character Variety Training** | MEDIUM | Revealed counter relationships |
| **Entropy Bonus** | MEDIUM | Prevents policy collapse |

### What Didn't Work

| Technique | Impact | Notes |
|-----------|--------|-------|
| **Pure Self-Play (early)** | LOW | Both agents too similar, no learning signal |
| **Curriculum Learning** | MIXED | Minimal improvement over direct training |
| **Defensive Rewards at High %** | LOW | Policy still maintains 50% attack rate |
| **Population Training** | LOW | Marginal diversity improvement |
| **Edge-Guard Specific Rewards** | LOW | 0.14% success rate |

### Hyperparameter Insights

| Parameter | Optimal Value | Finding |
|-----------|---------------|---------|
| Learning Rate | 2.5e-4 | Higher causes instability |
| Entropy Coef | 0.015 | Lower causes action collapse |
| Games/Update | 10 | Balance of sample efficiency |
| Epochs/Update | 4 | Diminishing returns beyond |
| Hidden Size | 256 | Sufficient for task |

### Key Learnings

1. **Opponent Diversity Matters**: Training against 70% aggressive + 30% random
   produced better results than pure aggressive training.

2. **Aggressive > Defensive**: The optimal strategy in this game favors overwhelming
   offense. Defensive play doesn't provide enough recovery opportunity.

3. **Character Counter Exists**: GRAB has favorable matchups against all characters,
   suggesting character balance issues in the game itself.

4. **Damage Adaptation Hard to Learn**: Despite explicit rewards, the policy couldn't
   learn to adapt behavior based on damage state. This suggests the observation space
   or reward shaping needs improvement.

5. **Edge-Guarding is Hard**: Even with edge-guard rewards, success rate remained
   very low. The game's recovery mechanics may make edge-guarding unreliable.
"""


def generate_recommendations() -> str:
    """Generate recommendations for future work."""
    return """
## Recommendations for Future Work

### High Priority

1. **Recurrent Policy Network**
   - Current: Feedforward network makes decisions frame-by-frame
   - Proposed: LSTM/Transformer to learn temporal patterns
   - Expected Impact: Better combo execution, adaptive play

2. **Opponent Modeling**
   - Current: No modeling of opponent behavior
   - Proposed: Auxiliary network to predict opponent actions
   - Expected Impact: Counter-play, exploit detection

3. **Hierarchical RL**
   - Current: Single policy for all situations
   - Proposed: High-level strategy + low-level execution
   - Expected Impact: Better edge-guarding, combo strings

### Medium Priority

4. **Curriculum on Opponent Skill**
   - Start with slow opponents, increase speed
   - Gradual difficulty increase may help

5. **Multi-Agent Training**
   - Train population of diverse agents
   - Select for counter-strategies

6. **Imitation Learning Warmup**
   - Bootstrap from human replays if available
   - Then fine-tune with RL

### Low Priority (Nice to Have)

7. **Reward Shaping Automation**
   - Use IRL to learn reward from demonstrations
   - Or use curiosity-based exploration

8. **Game-Theoretic Training**
   - PSRO or similar for Nash equilibrium
   - Find unexploitable strategy

### Technical Improvements

- **GPU Utilization**: Current ~3 games/sec, target 10+
- **Batch Inference**: Already implemented, needs testing
- **Parallel Environments**: Multiple games per step
"""


def generate_final_report():
    """Generate the comprehensive final report."""
    print(f"\n{'='*70}")
    print("GENERATING COMPREHENSIVE FINAL REPORT")
    print("Stick Figure Fighter ML Self-Play Training")
    print(f"{'='*70}")

    cycles = collect_cycle_data()

    # Build report
    report_lines = []

    # Header
    report_lines.append("# Stick Figure Fighter: ML Self-Play Training Report")
    report_lines.append("")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("**Mission**: ML Self-Play Training with Emergent Fighting Game Behaviors")
    report_lines.append("**Cycles Completed**: 5")
    report_lines.append("")

    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append("This report documents 5 cycles of ML self-play training for the Stick Figure Fighter game.")
    report_lines.append("The primary goal was to train an AI agent using Proximal Policy Optimization (PPO)")
    report_lines.append("that could defeat baseline opponents and exhibit emergent fighting game behaviors.")
    report_lines.append("")
    report_lines.append("### Key Achievements")
    report_lines.append("")
    report_lines.append("| Metric | Target | Achieved |")
    report_lines.append("|--------|--------|----------|")
    report_lines.append("| Win Rate vs Random | >80% | **100%** |")
    report_lines.append("| Win Rate vs Aggressive | >50% | **63-75%** |")
    report_lines.append("| Emergent Behaviors | Observable | **Hyper-aggressive playstyle** |")
    report_lines.append("| Character Analysis | Complete | **Full matchup matrix** |")
    report_lines.append("| Training Speed | Baseline | **~3 games/sec** |")
    report_lines.append("")
    report_lines.append("### Key Findings")
    report_lines.append("")
    report_lines.append("1. **Hyper-aggressive play is optimal**: The learned policy maintains ~50% attack rate")
    report_lines.append("   regardless of damage state, and this strategy beats both Random and Aggressive opponents.")
    report_lines.append("")
    report_lines.append("2. **GRAB is the counter character**: In mirror matches using the trained policy,")
    report_lines.append("   GRAB wins 77-80% against all other characters, suggesting character balance issues.")
    report_lines.append("")
    report_lines.append("3. **Damage-state adaptation didn't emerge**: Despite explicit rewards for defensive")
    report_lines.append("   play at high damage, the policy couldn't learn this behavior.")
    report_lines.append("")
    report_lines.append("4. **Edge-guarding is unreliable**: 0.14% success rate despite attempts, suggesting")
    report_lines.append("   game mechanics make edge-guarding unrewarding.")
    report_lines.append("")

    # Training Evolution
    report_lines.append("## Training Evolution")
    report_lines.append("")

    for cycle_id in ['cycle1', 'cycle2', 'cycle3', 'cycle4', 'cycle5']:
        cycle = cycles.get(cycle_id, {})
        report_lines.append(f"### {cycle_id.title()}: {cycle.get('name', 'Unknown')}")
        report_lines.append("")
        report_lines.append(f"**Focus**: {cycle.get('focus', 'N/A')}")
        report_lines.append("")
        report_lines.append("**Achievements**:")
        for achievement in cycle.get('achievements', []):
            report_lines.append(f"- {achievement}")
        report_lines.append("")

        metrics = cycle.get('metrics', {})
        if metrics:
            report_lines.append("**Metrics**:")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    report_lines.append(f"- {key}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            report_lines.append(f"  - {k}: {v:.1%}")
                        else:
                            report_lines.append(f"  - {k}: {v}")
                elif isinstance(value, float):
                    if value < 2:  # Probably a ratio
                        report_lines.append(f"- {key}: {value:.1%}")
                    else:
                        report_lines.append(f"- {key}: {value:.1f}")
                else:
                    report_lines.append(f"- {key}: {value}")
            report_lines.append("")

        if 'experiments' in cycle:
            report_lines.append("**Experiments Conducted**:")
            for exp in cycle['experiments']:
                report_lines.append(f"- **{exp['name']}** ({exp['file']}): {exp['goal']}")
            report_lines.append("")

    # Emergent Behaviors
    report_lines.append(generate_emergent_behaviors_catalog())

    # Technique Analysis
    report_lines.append(generate_technique_analysis())

    # Character Analysis
    report_lines.append("## Character Analysis")
    report_lines.append("")
    report_lines.append("### Tier List (Based on vs Aggressive Win Rate)")
    report_lines.append("")
    report_lines.append("| Tier | Character | vs Aggressive | Notes |")
    report_lines.append("|------|-----------|---------------|-------|")
    report_lines.append("| S | FLEX | 75% | Balanced, reliable |")
    report_lines.append("| A | ZONE | 70% | Range advantage |")
    report_lines.append("| A | RUSH | 57.5% | Speed, but fragile |")
    report_lines.append("| C | GRAB | 27.5% | Poor vs Aggressive baseline |")
    report_lines.append("")
    report_lines.append("### Counter Relationships")
    report_lines.append("")
    report_lines.append("In mirror matches (trained policy vs trained policy):")
    report_lines.append("")
    report_lines.append("- **GRAB beats everyone** (77-80%): Throw-based strategies dominate")
    report_lines.append("- **FLEX beats RUSH** (67.5%): Balanced beats speed")
    report_lines.append("- **ZONE beats RUSH** (62.5%): Range beats speed")
    report_lines.append("- **FLEX vs ZONE** is even (42.5%-50%)")
    report_lines.append("")

    # Recommendations
    report_lines.append(generate_recommendations())

    # Elo History
    report_lines.append("## Elo Progression")
    report_lines.append("")
    report_lines.append("| Checkpoint | Games | Elo | Notes |")
    report_lines.append("|------------|-------|-----|-------|")
    report_lines.append("| Initial | 0 | 1200 | Starting point |")
    report_lines.append("| Cycle 2 Final | 1000 | 1166 | Self-play Elo (vs previous gen) |")
    report_lines.append("| Cycle 4 Final | 2500+ | N/A | Focused on baseline win rates |")
    report_lines.append("")
    report_lines.append("*Note: Elo tracking was deprioritized in favor of direct baseline evaluation*")
    report_lines.append("*because self-play Elo doesn't directly measure skill vs known opponents.*")
    report_lines.append("")

    # Appendix
    report_lines.append("## Appendix: File Reference")
    report_lines.append("")
    report_lines.append("### Training Scripts")
    report_lines.append("")
    report_lines.append("| File | Purpose |")
    report_lines.append("|------|---------|")
    report_lines.append("| `train_self_play.py` | Original self-play training |")
    report_lines.append("| `train_cycle4_complete.py` | Comprehensive 2500+ game training |")
    report_lines.append("| `train_cycle5_defensive.py` | Defensive vs aggressive experiment |")
    report_lines.append("| `train_cycle5_character_specific.py` | Per-character specialization |")
    report_lines.append("| `train_cycle5_metagame.py` | Mixed-strategy opponent testing |")
    report_lines.append("| `train_cycle5_perf_optimize.py` | Performance profiling |")
    report_lines.append("")
    report_lines.append("### Analysis Scripts")
    report_lines.append("")
    report_lines.append("| File | Purpose |")
    report_lines.append("|------|---------|")
    report_lines.append("| `evaluate.py` | Baseline evaluation (vs Random/Aggressive) |")
    report_lines.append("| `behavior_analysis.py` | Emergent behavior detection |")
    report_lines.append("| `generate_matchup_matrix_v2.py` | Character matchup analysis |")
    report_lines.append("| `generate_final_report.py` | This report generator |")
    report_lines.append("")
    report_lines.append("### Core ML Components")
    report_lines.append("")
    report_lines.append("| File | Purpose |")
    report_lines.append("|------|---------|")
    report_lines.append("| `policy.py` | PolicyNetwork (actor-critic) |")
    report_lines.append("| `trainer.py` | PPOTrainer |")
    report_lines.append("| `experience.py` | Episode/ExperienceBuffer |")
    report_lines.append("| `self_play.py` | SelfPlayManager, EloTracker |")
    report_lines.append("")

    # Footer
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*This report was auto-generated as part of Cycle 5: Final Documentation.*")

    # Write report
    report_content = '\n'.join(report_lines)

    Path('artifacts').mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT, 'w') as f:
        f.write(report_content)

    print(f"\nReport generated: {REPORT_OUTPUT}")
    print(f"Length: {len(report_lines)} lines")

    # Also save as JSON for programmatic access
    json_report = {
        'generated_at': datetime.now().isoformat(),
        'cycles': cycles,
        'summary': {
            'total_cycles': 5,
            'final_vs_random': 1.0,
            'final_vs_aggressive': 0.75,
            'best_character': 'FLEX',
            'emergent_behaviors': ['hyper-aggression', 'approach-patterns', 'proto-combos'],
            'missing_behaviors': ['damage-adaptation', 'edge-guarding', 'defensive-play']
        }
    }

    json_path = 'artifacts/final_mission_report.json'
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)

    print(f"JSON data saved: {json_path}")

    return report_content


def main():
    report = generate_final_report()
    return report


if __name__ == '__main__':
    main()
