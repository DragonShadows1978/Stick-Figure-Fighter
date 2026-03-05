# Self-Play Training System

## The Goal

Train a neural network to play Stick Figure Fighter through pure self-play. No human demonstrations, no hardcoded strategies, no reward shaping tricks. Two agents fight each other, learn from the outcomes, and improve.

## What's Already Built

The game provides a complete ML interface:

### Observation Space (21 features)
```python
from game_engine import get_observation
obs = get_observation(game_state, player_id)  # Returns numpy array (21,)
```

Features (all normalized):
- Player position (x, y)
- Player velocity (vx, vy)
- Opponent position (x, y)
- Opponent velocity (vx, vy)
- Player damage %
- Opponent damage %
- Player stocks
- Opponent stocks
- Player facing direction
- Opponent facing direction
- Player state one-hot (5 states: idle, moving, airborne, attacking, hitstun)

### Action Space (10 discrete actions)
```python
from game_engine import get_action_from_index
input_state = get_action_from_index(action_idx)  # Returns InputState
```

Actions:
- 0: Nothing
- 1: Left
- 2: Right
- 3: Jump
- 4: Down
- 5: Attack
- 6: Left + Attack
- 7: Right + Attack
- 8: Up + Attack
- 9: Down + Attack

### Headless Game Loop
```python
from game_engine import create_game, GameLoop

game = create_game()
loop = GameLoop(game, headless=True)

# Step the game
game.fighters[0].input = get_action_from_index(action_p1)
game.fighters[1].input = get_action_from_index(action_p2)
loop.step()
```

## Training Architecture

### Network Structure
Simple actor-critic with shared backbone:

```
Input (21) → Linear(256) → ReLU → Linear(256) → ReLU
                                      ↓
                            ┌─────────┴─────────┐
                            ↓                   ↓
                     Linear(10)            Linear(1)
                     Softmax                  Value
                     (Policy)              (Baseline)
```

### Algorithm: PPO (Proximal Policy Optimization)
- Stable, sample efficient
- Works well with self-play
- Clip ratio prevents catastrophic updates

### Training Loop
```
for iteration in range(max_iterations):
    # Collect experience
    for game in range(games_per_iteration):
        reset game
        while not game_over:
            obs_p1 = get_observation(game, 0)
            obs_p2 = get_observation(game, 1)

            action_p1 = policy.sample(obs_p1)
            action_p2 = policy.sample(obs_p2)

            apply actions, step game

            store (obs, action, reward, done) for both players

    # Compute advantages and returns
    # Update policy with PPO objective
    # Update value function
```

### Reward Signal

**Sparse (primary):**
- +1.0: Take a stock
- -1.0: Lose a stock
- +0.5: Win match
- -0.5: Lose match

**Dense (optional, if learning stalls):**
- +0.001 per 1% damage dealt
- -0.0005 per 1% damage received

Start with sparse rewards only. Dense rewards risk degenerate strategies.

## Implementation Tasks

### 1. Policy Network (`policy.py`)
- Actor-critic network using tensor_gpu.py
- Forward pass returns (action_logits, value)
- Sample action from softmax distribution
- Compute log probabilities for PPO

### 2. Experience Buffer (`experience.py`)
- Store trajectories: (obs, action, reward, done, log_prob, value)
- Compute discounted returns (gamma=0.99)
- Compute GAE advantages (lambda=0.95)
- Batch sampling for updates

### 3. PPO Trainer (`trainer.py`)
- Collect rollouts from self-play games
- Compute PPO loss (clipped surrogate objective)
- Value function loss
- Entropy bonus for exploration
- Gradient updates via tensor_gpu.py

### 4. Self-Play Manager (`self_play.py`)
- Run parallel games (CPU-bound game sim)
- Feed experience to single GPU learner
- Track Elo ratings across training
- Save checkpoints periodically
- Log metrics (win rate, avg game length, damage dealt)

### 5. Evaluation (`evaluate.py`)
- Play trained agent vs random policy
- Play trained agent vs previous checkpoints
- Render sample games for visualization
- Track emergent behaviors

## Hyperparameters (Starting Point)

```python
# PPO
clip_ratio = 0.2
gamma = 0.99
gae_lambda = 0.95
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5

# Training
learning_rate = 3e-4
batch_size = 64
epochs_per_iteration = 4
games_per_iteration = 100
max_iterations = 10000

# Network
hidden_size = 256
num_layers = 2
```

## Expected Progression

**Early training (iterations 0-1000):**
- Random flailing
- Discover movement works
- Occasionally land hits by accident

**Mid training (iterations 1000-5000):**
- Intentional approach and attack
- Basic punish game (hit opponent when they whiff)
- Learn to recover when knocked off

**Late training (iterations 5000+):**
- Combo sequences
- Edge guarding
- Spacing and footsies
- Character-specific strategies

## Files

- `tensor_gpu.py` - CUDA autograd (already exists)
- `game_engine.py` - Game with ML interface (already exists)
- `policy.py` - Neural network (TO BUILD)
- `experience.py` - Trajectory buffer (TO BUILD)
- `trainer.py` - PPO implementation (TO BUILD)
- `self_play.py` - Training orchestration (TO BUILD)
- `evaluate.py` - Evaluation tools (TO BUILD)

## Reference

Brotato Bot at `/home/vader/mini-mind-v2/workspace/brotato_bot/` has working examples of:
- Neural network architectures (`src/ai/`)
- Training loops
- Game state integration

Fork patterns as needed. Do NOT modify brotato_bot.

## Success Criteria

1. **Learning signal**: Loss decreases, win rate vs random increases
2. **Emergent behavior**: Agents discover strategies not explicitly programmed
3. **Self-improvement**: Later checkpoints beat earlier checkpoints
4. **Stability**: Training doesn't collapse or diverge

## The Test

Can a from-scratch ML stack bootstrap fighting game intelligence through self-play alone?

No shortcuts. No human knowledge injected. Just two neural networks, a reward signal, and tensor_gpu.py.
