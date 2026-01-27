"""
PPO Trainer for Stick Figure Fighter Self-Play.

Implements Proximal Policy Optimization with:
- Clipped surrogate objective
- Value function clipping
- Entropy bonus for exploration
- Gradient clipping
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from policy import PolicyNetwork
from experience import ExperienceBuffer, Episode


@dataclass
class TrainingConfig:
    """PPO training hyperparameters."""
    # PPO parameters
    clip_ratio: float = 0.2           # PPO clipping threshold
    gamma: float = 0.99               # Discount factor
    gae_lambda: float = 0.95          # GAE lambda
    entropy_coef: float = 0.01        # Entropy bonus coefficient
    value_coef: float = 0.5           # Value loss coefficient
    max_grad_norm: float = 0.5        # Gradient clipping

    # Optimization
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs_per_update: int = 4        # PPO epochs per data collection

    # Rollout
    steps_per_rollout: int = 2048     # Steps before update
    games_per_iteration: int = 10     # Games to play per training iteration


class PPOTrainer:
    """
    PPO trainer for self-play.

    Manages policy network, optimizer, and training loop.
    """

    def __init__(self, policy: PolicyNetwork, config: TrainingConfig,
                 device: str = 'cuda'):
        self.policy = policy
        self.config = config
        self.device = device

        # Move policy to device
        self.policy.to(device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )

        # Experience buffer
        self.buffer = ExperienceBuffer(
            gamma=config.gamma,
            gae_lambda=config.gae_lambda
        )

        # Training statistics
        self.update_count = 0
        self.total_steps = 0
        self.stats_history: List[Dict] = []

    def compute_ppo_loss(self, obs: torch.Tensor, actions: torch.Tensor,
                         old_log_probs: torch.Tensor, returns: torch.Tensor,
                         advantages: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss for a single batch.

        Args:
            obs: Observations, shape (batch, obs_dim)
            actions: Actions taken, shape (batch,)
            old_log_probs: Log probs from rollout, shape (batch,)
            returns: Computed returns, shape (batch,)
            advantages: Computed advantages, shape (batch,)

        Returns:
            (loss, info): Total loss and info dict with component losses
        """
        # Evaluate actions with current policy
        new_log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)

        # Compute probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        policy_loss_1 = ratio * advantages
        policy_loss_2 = torch.clamp(
            ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
        ) * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(values, returns)

        # Total loss
        total_loss = (
            policy_loss +
            self.config.value_coef * value_loss -
            self.config.entropy_coef * entropy
        )

        # Info dict
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
            clip_fraction = ((ratio - 1).abs() > self.config.clip_ratio).float().mean()

        info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item(),
            'approx_kl': approx_kl.item(),
            'clip_fraction': clip_fraction.item()
        }

        return total_loss, info

    def update(self, episodes: List[Episode]) -> Dict:
        """
        Perform PPO update from collected episodes.

        Args:
            episodes: List of episodes to learn from

        Returns:
            Dictionary of training statistics
        """
        # Add episodes to buffer
        self.buffer.clear()
        for ep in episodes:
            self.buffer.add_episode(ep)

        # Get buffer statistics
        buffer_stats = self.buffer.get_episode_stats()

        if buffer_stats['total_transitions'] == 0:
            return {'update': self.update_count, 'total_steps': self.total_steps,
                    'mean_reward': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        # Compute returns and advantages
        data = self.buffer.compute_returns_and_advantages()

        # Convert to tensors
        obs = torch.from_numpy(data['obs']).float().to(self.device)
        actions = torch.from_numpy(data['actions']).long().to(self.device)
        old_log_probs = torch.from_numpy(data['old_log_probs']).float().to(self.device)
        returns = torch.from_numpy(data['returns']).float().to(self.device)
        advantages = torch.from_numpy(data['advantages']).float().to(self.device)

        # Training loop
        all_infos = []
        n_samples = obs.shape[0]

        for epoch in range(self.config.epochs_per_update):
            # Shuffle indices
            indices = torch.randperm(n_samples)

            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch_idx = indices[start:end]

                # Get batch
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Compute loss
                loss, info = self.compute_ppo_loss(
                    batch_obs, batch_actions, batch_old_log_probs,
                    batch_returns, batch_advantages
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()

                all_infos.append(info)

        # Aggregate statistics
        stats = {
            'update': self.update_count,
            'total_steps': self.total_steps,
            'episodes': buffer_stats['num_episodes'],
            'transitions': buffer_stats['total_transitions'],
            'mean_reward': buffer_stats.get('mean_reward', 0),
            'policy_loss': np.mean([i['policy_loss'] for i in all_infos]),
            'value_loss': np.mean([i['value_loss'] for i in all_infos]),
            'entropy': np.mean([i['entropy'] for i in all_infos]),
            'approx_kl': np.mean([i['approx_kl'] for i in all_infos]),
            'clip_fraction': np.mean([i['clip_fraction'] for i in all_infos]),
        }

        self.update_count += 1
        self.total_steps += buffer_stats['total_transitions']
        self.stats_history.append(stats)

        return stats

    def save(self, filepath: str):
        """Save trainer state."""
        self.policy.save(filepath)

    def load(self, filepath: str):
        """Load trainer state."""
        self.policy.load(filepath)


def test_ppo_trainer():
    """Test the PPO trainer."""
    print("Testing PPO Trainer...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create policy and trainer
    policy = PolicyNetwork(obs_size=21, action_size=10, hidden_size=256)
    config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=64,
        epochs_per_update=2
    )
    trainer = PPOTrainer(policy, config, device=device)

    # Create fake episodes
    episodes = []
    for _ in range(5):
        episode = Episode(player_id=0)
        for t in range(100):
            obs = np.random.randn(21).astype(np.float32)
            action = np.random.randint(0, 10)
            reward = np.random.randn() * 0.1
            done = t == 99
            log_prob = np.log(0.1)
            value = np.random.randn()
            episode.add(obs, action, reward, done, log_prob, value)
        episodes.append(episode)

    # Perform update
    stats = trainer.update(episodes)
    print(f"Update stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Test save/load
    trainer.save('/tmp/test_trainer.pt')
    trainer.load('/tmp/test_trainer.pt')
    print("Save/load: OK")

    print("PPO Trainer tests passed!")


if __name__ == "__main__":
    test_ppo_trainer()
