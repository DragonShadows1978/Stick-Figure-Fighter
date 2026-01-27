# Stick Figure Fighter - ML Agent Behavior Taxonomy

## Overview
- Character: FLEX
- Playstyle: Hyper-Aggressive
- Win Rate: 90.0%

## 1. Action Distribution
  Down+Attack     [#####                                             ] 11.9%
  Nothing         [#####                                             ] 11.5%
  Down            [#####                                             ] 11.3%
  Right           [#####                                             ] 10.7%
  Up+Attack       [#####                                             ] 10.1%
  Right+Attack    [####                                              ] 9.9%
  Left            [####                                              ] 9.1%
  Jump            [####                                              ] 9.0%
  Left+Attack     [####                                              ] 8.7%
  Attack          [###                                               ] 7.7%

## 2. Behavioral Ratios
  Attack:    48.4%
  Movement:  19.8%
  Aerial:    9.0%
  Passive:   11.5%

## 3. Emergent Patterns

### Common Action Sequences
  Down -> Down: 1074x
  Down+Attack -> Down+Attack: 959x
  Nothing -> Nothing: 876x
  Nothing -> Down+Attack: 824x
  Down+Attack -> Nothing: 777x
  Nothing -> Down: 765x
  Down+Attack -> Down: 758x
  Down+Attack -> Right: 752x

### Potential Combo Patterns
  Down+Attack -> Down+Attack: 959x
  Up+Attack -> Up+Attack: 734x
  Down+Attack -> Up+Attack: 729x
  Up+Attack -> Down+Attack: 719x
  Right+Attack -> Right+Attack: 691x

## 4. Positional Behaviors

### Near Right Edge
  Samples: 47190
  Attack tendency: 48.3%
  Movement tendency: 19.7%
  Top actions: Down(12%), Down+Attack(12%), Nothing(11%)

### Center Stage
  Samples: 7213
  Attack tendency: 49.0%
  Movement tendency: 20.4%
  Top actions: Up+Attack(15%), Down+Attack(13%), Nothing(12%)

### Offstage Right
  Samples: 42898
  Attack tendency: 48.1%
  Movement tendency: 20.0%
  Top actions: Down(12%), Down+Attack(12%), Nothing(11%)

## 5. Damage State Adaptation

### When Taking Damage
  Low Damage:
    Attack ratio: 48.4%
    Top action: Down+Attack
  Mid Damage:
    Attack ratio: 30.0%
    Top action: Left

### When Opponent is Damaged
  Opponent Low:
    Attack ratio: 48.4%
    Top action: Down+Attack

## 6. Fighting Game Behavior Classifications

### Neutral Game
  - Aggressive approach, constant pressure

### Offense
  - Shows evidence of combo awareness
  - High attack frequency suggests pressure-focused playstyle

### Edge Situations
  - Aggressively pursues opponents near edge
