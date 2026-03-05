"""
Experience Buffer for PPO Self-Play Training.

Stores trajectories and computes:
- Discounted returns (Monte Carlo or TD)
- GAE (Generalized Advantage Estimation)
"""
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Transition:
    """Single transition in the environment."""
    obs: np.ndarray           # Observation at this step
    action: int               # Action taken
    reward: float             # Reward received
    done: bool                # Episode terminated?
    log_prob: float           # Log probability of action
    value: float              # Value estimate at this state
    next_obs: Optional[np.ndarray] = None  # Next observation (for TD)


@dataclass
class Episode:
    """Complete episode trajectory."""
    transitions: List[Transition] = field(default_factory=list)
    player_id: int = 0        # Which player this episode is from
    final_value: float = 0.0  # Bootstrap value for non-terminal

    def add(self, obs, action, reward, done, log_prob, value, next_obs=None):
        """Add a transition to this episode."""
        self.transitions.append(Transition(
            obs=obs, action=action, reward=reward, done=done,
            log_prob=log_prob, value=value, next_obs=next_obs
        ))

    def __len__(self):
        return len(self.transitions)


class ExperienceBuffer:
    """
    Buffer for storing and processing experience for PPO.

    Features:
    - Stores transitions from multiple episodes
    - Computes GAE advantages
    - Provides batched sampling for training
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Args:
            gamma: Discount factor for returns
            gae_lambda: Lambda for GAE
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.episodes: List[Episode] = []

    def add_episode(self, episode: Episode):
        """Add a complete episode to the buffer."""
        self.episodes.append(episode)

    def clear(self):
        """Clear all stored episodes."""
        self.episodes = []

    @property
    def size(self) -> int:
        """Total number of transitions in buffer."""
        return sum(len(ep) for ep in self.episodes)

    def compute_returns_and_advantages(self) -> Dict[str, np.ndarray]:
        """
        Compute returns and GAE advantages for all transitions.

        Returns:
            Dictionary with numpy arrays:
            - obs: (N, obs_size)
            - actions: (N,)
            - old_log_probs: (N,)
            - returns: (N,)
            - advantages: (N,)
            - values: (N,)
        """
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_advantages = []
        all_values = []

        for episode in self.episodes:
            if len(episode) == 0:
                continue

            # Extract transition data
            obs_list = [t.obs for t in episode.transitions]
            rewards = np.array([t.reward for t in episode.transitions])
            values = np.array([t.value for t in episode.transitions])
            dones = np.array([t.done for t in episode.transitions])

            T = len(episode)

            # Append bootstrap value for GAE computation
            # If episode ended, bootstrap is 0; otherwise use final_value
            if dones[-1]:
                next_value = 0.0
            else:
                next_value = episode.final_value

            values_extended = np.append(values, next_value)

            # Compute GAE advantages
            advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0

            for t in reversed(range(T)):
                # TD error: r + gamma * V(s') - V(s)
                # For terminal states, V(s') = 0
                if dones[t]:
                    delta = rewards[t] - values[t]
                    gae = delta
                else:
                    delta = rewards[t] + self.gamma * values_extended[t + 1] - values[t]
                    gae = delta + self.gamma * self.gae_lambda * gae

                advantages[t] = gae

            # Returns = advantages + values
            returns = advantages + values

            # Collect data
            all_obs.extend(obs_list)
            all_actions.extend([t.action for t in episode.transitions])
            all_log_probs.extend([t.log_prob for t in episode.transitions])
            all_returns.extend(returns.tolist())
            all_advantages.extend(advantages.tolist())
            all_values.extend(values.tolist())

        # Convert to numpy arrays
        obs_array = np.array(all_obs, dtype=np.float32)
        actions_array = np.array(all_actions, dtype=np.float32)
        log_probs_array = np.array(all_log_probs, dtype=np.float32)
        returns_array = np.array(all_returns, dtype=np.float32)
        advantages_array = np.array(all_advantages, dtype=np.float32)
        values_array = np.array(all_values, dtype=np.float32)

        # Normalize advantages (important for PPO stability)
        if len(advantages_array) > 1:
            adv_mean = advantages_array.mean()
            adv_std = advantages_array.std() + 1e-8
            advantages_array = (advantages_array - adv_mean) / adv_std

        return {
            'obs': obs_array,
            'actions': actions_array,
            'old_log_probs': log_probs_array,
            'returns': returns_array,
            'advantages': advantages_array,
            'values': values_array
        }

    def get_batches(self, batch_size: int, device: str = 'cuda') -> List[Dict[str, torch.Tensor]]:
        """
        Get randomized mini-batches for PPO updates.

        Args:
            batch_size: Size of each mini-batch
            device: Device for tensors ('cuda' or 'cpu')

        Returns:
            List of batch dictionaries with torch.Tensor values
        """
        data = self.compute_returns_and_advantages()
        N = len(data['obs'])

        if N == 0:
            return []

        # Shuffle indices
        indices = np.random.permutation(N)

        batches = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_indices = indices[start:end]

            batch = {
                'obs': torch.from_numpy(data['obs'][batch_indices]).float().to(device),
                'actions': torch.from_numpy(data['actions'][batch_indices]).long().to(device),
                'old_log_probs': torch.from_numpy(data['old_log_probs'][batch_indices]).float().to(device),
                'returns': torch.from_numpy(data['returns'][batch_indices]).float().to(device),
                'advantages': torch.from_numpy(data['advantages'][batch_indices]).float().to(device),
            }
            batches.append(batch)

        return batches

    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics about stored episodes."""
        if len(self.episodes) == 0:
            return {'num_episodes': 0, 'total_transitions': 0}

        total_rewards = []
        episode_lengths = []

        for ep in self.episodes:
            total_reward = sum(t.reward for t in ep.transitions)
            total_rewards.append(total_reward)
            episode_lengths.append(len(ep))

        return {
            'num_episodes': len(self.episodes),
            'total_transitions': self.size,
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_length': np.mean(episode_lengths),
            'min_length': min(episode_lengths),
            'max_length': max(episode_lengths)
        }


class RolloutBuffer:
    """
    Simpler buffer that stores flat arrays of experience.

    More memory efficient than Episode-based storage for PPO.
    """

    def __init__(self, size: int, obs_dim: int, gamma: float = 0.99,
                 gae_lambda: float = 0.95, device: str = 'cuda'):
        self.size = size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Pre-allocate arrays
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)

        # Computed during finalization
        self.returns = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool,
            log_prob: float, value: float):
        """Add a single transition."""
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value

        self.pos += 1
        if self.pos >= self.size:
            self.full = True
            self.pos = 0

    def compute_returns(self, final_value: float = 0.0):
        """Compute returns and advantages using GAE."""
        n = self.size if self.full else self.pos

        # Compute GAE
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = final_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = self.values[t + 1]

            delta = (self.rewards[t] + self.gamma * next_value * next_non_terminal
                     - self.values[t])
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

        # Normalize advantages
        if n > 1:
            adv_mean = self.advantages[:n].mean()
            adv_std = self.advantages[:n].std() + 1e-8
            self.advantages[:n] = (self.advantages[:n] - adv_mean) / adv_std

    def get_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Get mini-batches for training."""
        n = self.size if self.full else self.pos
        indices = np.random.permutation(n)

        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch = {
                'obs': torch.from_numpy(self.obs[idx]).float().to(self.device),
                'actions': torch.from_numpy(self.actions[idx]).long().to(self.device),
                'old_log_probs': torch.from_numpy(self.log_probs[idx]).float().to(self.device),
                'returns': torch.from_numpy(self.returns[idx]).float().to(self.device),
                'advantages': torch.from_numpy(self.advantages[idx]).float().to(self.device),
            }
            batches.append(batch)

        return batches

    def clear(self):
        """Reset the buffer."""
        self.pos = 0
        self.full = False


def test_experience_buffer():
    """Test the experience buffer."""
    print("Testing ExperienceBuffer...")

    buffer = ExperienceBuffer(gamma=0.99, gae_lambda=0.95)

    # Create some fake episodes
    for ep_id in range(5):
        episode = Episode(player_id=0)
        for t in range(50):
            obs = np.random.randn(21).astype(np.float32)
            action = np.random.randint(0, 10)
            reward = np.random.randn() * 0.1
            done = t == 49
            log_prob = np.log(0.1)
            value = np.random.randn()

            episode.add(obs, action, reward, done, log_prob, value)

        buffer.add_episode(episode)

    print(f"Buffer size: {buffer.size}")

    # Compute returns
    data = buffer.compute_returns_and_advantages()
    print(f"Obs shape: {data['obs'].shape}")
    print(f"Advantages mean: {data['advantages'].mean():.4f}, std: {data['advantages'].std():.4f}")

    # Get batches
    batches = buffer.get_batches(batch_size=64, device='cuda')
    print(f"Number of batches: {len(batches)}")

    # Test RolloutBuffer
    print("\nTesting RolloutBuffer...")
    rollout = RolloutBuffer(size=256, obs_dim=21, device='cuda')

    for i in range(256):
        obs = np.random.randn(21).astype(np.float32)
        rollout.add(obs, action=i % 10, reward=0.1, done=i == 255,
                   log_prob=-2.3, value=0.5)

    rollout.compute_returns(final_value=0.0)
    batches = rollout.get_batches(batch_size=64)
    print(f"RolloutBuffer batches: {len(batches)}")

    print("Experience buffer tests passed!")


if __name__ == "__main__":
    test_experience_buffer()
