# Stick Figure Fighter

## Core Concept

A platform fighter in the style of Super Smash Brothers, built entirely from first principles. Stick figure art style with physics-based combat. No external game engines or frameworks.

## Design Philosophy

- **First Principles Only**: No Godot, no Pygame, no existing game frameworks. Build physics, rendering, input, and collision from scratch.
- **ML-Ready Architecture**: System must integrate with custom tensor.py for future self-play neural network training.
- **Procedural Art**: Stick figures generated programmatically - no external sprite dependencies.

---

## Core Mechanics

### Physics System (Smash Brothers Style)
- **Knockback Scaling**: Damage % increases knockback received
- **Ring-Out Victory**: No health bars - knock opponents off stage to win
- **Momentum-Based Movement**: Velocity and acceleration, not instant movement
- **Directional Influence (DI)**: Players can influence trajectory when hit
- **Platform Interactions**: Drop-through platforms, edge grabbing

### Combat System
- **Normal Attacks**: Light, medium, heavy (ground and aerial variants)
- **Special Attacks**: Character-specific B-moves (neutral, side, up, down)
- **Shielding**: Defensive option, depletes with use
- **Grabbing**: Throw opponents in 4 directions
- **Dodging**: Spot dodge, air dodge, roll

### Win Condition
- Stock-based (default 3 lives)
- Knocked past blast zones = stock lost
- Last player standing wins

---

## Input System (CRITICAL CHALLENGE)

### Requirements
- **Keyboard Support**: Full gameplay possible on keyboard alone
- **Controller Support**: Gamepad support (Linux evdev)
- **Input Abstraction Layer**: Unified action system regardless of device

### Unified Action Map
```
MOVE_LEFT, MOVE_RIGHT
JUMP, FAST_FALL
ATTACK_LIGHT, ATTACK_HEAVY
SPECIAL (+ directional variants)
SHIELD
GRAB
DODGE
```

### Input Implementation Path
- Read raw input from /dev/input/* (evdev) for controllers
- Read keyboard state via X11 or direct input
- Map both to abstract action events
- Handle simultaneous inputs (attack + direction = directional attack)

---

## Characters (4 Distinct Playstyles)

### 1. RUSH - The Rushdown
- **Playstyle**: Fast, aggressive, combo-focused
- **Attributes**: High speed, low weight, low knockback per hit
- **Specials**:
  - Neutral-B: Rapid punches
  - Side-B: Dash attack with follow-up
  - Up-B: Rising uppercut (recovery)
  - Down-B: Counter stance
- **Gameplan**: Overwhelm with speed and combos, struggle to kill

### 2. ZONE - The Zoner
- **Playstyle**: Ranged, keepaway, stage control
- **Attributes**: Medium speed, medium weight, projectile-based
- **Specials**:
  - Neutral-B: Chargeable projectile
  - Side-B: Horizontal beam
  - Up-B: Teleport (recovery)
  - Down-B: Trap placement
- **Gameplan**: Control space, rack damage from distance, confirm kills with smash attacks

### 3. GRAB - The Grappler
- **Playstyle**: Slow, powerful, command grab focused
- **Attributes**: Slow speed, heavy weight, high knockback
- **Specials**:
  - Neutral-B: Armored charge
  - Side-B: Command grab (throwable)
  - Up-B: Spinning rise (recovery)
  - Down-B: Ground pound
- **Gameplan**: Get in, land grabs, kill early with raw power

### 4. FLEX - The All-Rounder
- **Playstyle**: Balanced, adaptable, fundamental
- **Attributes**: Medium everything
- **Specials**:
  - Neutral-B: Fireball (standard projectile)
  - Side-B: Spinning kick
  - Up-B: Rising slash (recovery)
  - Down-B: Reflector
- **Gameplan**: Solid in all situations, wins through fundamentals

---

## Technical Architecture

### Rendering Pipeline
- Direct framebuffer or X11 surface rendering
- Stick figures as procedural line/circle primitives
- 60 FPS target
- Resolution: 1280x720 (scalable)

### Physics Engine
- Fixed timestep simulation (60Hz)
- AABB collision detection
- Swept collision for fast objects
- Knockback formula: `base_knockback + (damage% * knockback_scaling) * weight_modifier`

### Stage System
- Platform data structure (position, width, passthrough flag)
- Blast zones (boundaries for ring-outs)
- At least 2 stages for variety

### State Machine
- Per-character state: IDLE, WALKING, RUNNING, JUMPING, FALLING, ATTACKING, HITSTUN, SHIELDING, GRABBING, GRABBED, LEDGE
- State determines available actions and physics behavior

---

## Self-Play Training System

The core purpose of this project: an AI that teaches itself to play through pure self-play. No human demonstrations, no expert replays, no hand-crafted heuristics.

### The Test
Can a from-scratch ML stack bootstrap fighting game intelligence through self-play alone?

### Training Architecture

**Dual Agent Setup**:
- Two neural networks control opposing fighters
- Both networks share architecture, may share or diverge weights
- Games run headless at maximum speed during training
- Periodic evaluation matches rendered for observation

**Training Loop**:
```
while training:
    1. Reset game state (random stage, random/fixed characters)
    2. Both agents observe game state
    3. Both agents select actions via policy network
    4. Game simulates one frame
    5. Collect (state, action, reward) tuples
    6. When match ends:
       - Compute returns with discount factor
       - Update networks via tensor_gpu.py backprop
    7. Periodically save checkpoints
```

**Network Architecture** (starting point):
- Input: Flattened game state observation
- Hidden: 2-3 fully connected layers (256-512 units)
- Output heads:
  - Policy: Softmax over action space
  - Value: Single scalar (expected return)

**Algorithm Options**:
- PPO (Proximal Policy Optimization) - stable, sample efficient
- Simple policy gradient with baseline - easier to implement from scratch
- Self-play with Elo tracking to measure improvement

### Observation Space
- Character positions (x, y) - normalized to stage bounds
- Character velocities (vx, vy) - normalized
- Damage percentages (0-999 scaled to 0-1)
- Current states (one-hot encoded)
- Stock counts
- Facing directions
- Current action frames (for timing reads)
- Stage geometry (platform positions, blast zones)

### Action Space
Discrete actions matching the unified input map:
- Movement: LEFT, RIGHT, NONE
- Vertical: JUMP, FAST_FALL, NONE
- Attack: LIGHT, HEAVY, SPECIAL, NONE
- Defense: SHIELD, DODGE, GRAB, NONE
- Directional modifiers applied contextually

**Action Masking**: Invalid actions masked based on current state (can't jump while jumping, can't attack during hitstun)

### Reward Signal

**Sparse Rewards** (main signal):
- +1.0 per stock taken
- -1.0 per stock lost
- Match win bonus: +0.5

**Dense Rewards** (optional, may help early training):
- +0.01 per 10% damage dealt
- -0.005 per 10% damage received
- Small penalty for time (encourages aggression)

**Reward Shaping Risks**: Dense rewards can lead to degenerate strategies (farming damage without killing). Start sparse, add dense only if learning stalls.

### Emergent Behavior Goals
Through self-play, agents should discover:
- Combo sequences (hitstun chains)
- Edge guarding (intercepting recovery)
- Spacing and footsies
- Character-specific strategies
- Punish game (reacting to whiffed moves)
- Recovery mixups

### Training Infrastructure

**Headless Mode**:
- Game runs without rendering during training
- Physics and state machine execute at max CPU/GPU speed
- Single training machine can run thousands of games per hour

**Checkpointing**:
- Save network weights periodically
- Save replay buffer for analysis
- Track Elo ratings across training generations

**Evaluation**:
- Periodic rendered matches for human observation
- Elo tracking against fixed policy snapshots
- Behavior metrics: aggression, combo length, recovery success rate

### Hardware Utilization
- **GPU (RTX 3070)**: Neural network forward/backward passes via tensor_gpu.py
- **CPU**: Game simulation (physics, state machine, collision)
- **Parallelization**: Multiple game instances feeding experience to single learner

### Files
- `tensor_gpu.py` - Custom autograd with CUDA acceleration (40x speedup)
- `vision/` - X11 screen capture (for visualization/debugging, not training input)

### Reference Implementation
**Location**: `/home/vader/mini-mind-v2/workspace/brotato_bot/`

The Brotato Bot project contains working examples of game ML systems:
- `src/core/tensor.py` / `tensor_gpu.py` - Autograd implementation
- `src/vision/` - Screen capture, game state detection, sprite classification
- `src/ai/` - Neural network architectures, training loops
- Integration patterns for connecting ML to game state

**IMPORTANT**: This folder is a reference only. Do NOT modify files in brotato_bot/.
- Fork/copy any needed components into this project
- Adapt patterns to Stick Figure Fighter's architecture
- Keep this project self-contained

---

## Constraints & Non-Goals

### MUST Have
- Playable 1v1 local multiplayer
- Keyboard support
- Controller support
- 4 unique characters
- Smash-style knockback physics
- At least 2 stages

### MUST NOT Have
- External game engines (Godot, Unity, Pygame, etc.)
- External sprite assets
- Online multiplayer (out of scope)
- Story mode

### NICE TO Have
- Training mode with CPU
- Frame data display
- Hitbox visualization toggle
- Replay system

---

## Success Criteria

1. **Playable**: Two humans can play a complete match with keyboard/controller
2. **Distinct**: All 4 characters feel meaningfully different
3. **Responsive**: Input latency < 16ms (one frame at 60fps)
4. **Coherent**: All systems built from first principles, fully understood by the codebase
5. **ML-Ready**: Architecture supports future neural network integration

---

## Notes

- This is a training ground for the custom ML stack
- Visual fidelity is NOT a priority - clarity and functionality are
- The constraint of "no frameworks" forces deep understanding of every system
- Stick figures sidestep the sprite generation problem while remaining functional
