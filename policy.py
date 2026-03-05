"""
Policy Network for Stick Figure Fighter Self-Play Training.

Actor-Critic architecture using PyTorch for CUDA acceleration.
Takes game observations (21 features) and outputs:
- Policy: Softmax over 10 discrete actions
- Value: Scalar estimate of expected return
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Actor-Critic network for PPO self-play.

    Architecture:
        Input (21) -> Linear(256) -> ReLU -> Linear(256) -> ReLU
                                        |
                              +---------+---------+
                              |                   |
                        Linear(10)           Linear(1)
                        Softmax              (Value)
    """

    def __init__(self, obs_size: int = 21, action_size: int = 10,
                 hidden_size: int = 256, dropout: float = 0.0):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # Shared backbone
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Policy head (actor)
        self.policy_head = nn.Linear(hidden_size, action_size)

        # Value head (critic)
        self.value_head = nn.Linear(hidden_size, 1)

        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize weights with orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

        # Scale policy head for smaller initial actions
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.value_head.bias, 0)

        # Track device
        self._device = 'cpu'

    def to(self, device):
        """Move module to device."""
        self._device = device
        return super().to(device)

    def cuda(self):
        """Move to CUDA."""
        return self.to('cuda')

    def cpu(self):
        """Move to CPU."""
        return self.to('cpu')

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            obs: Observation tensor, shape (batch_size, obs_size) or (obs_size,)

        Returns:
            (action_logits, value): Tuple of policy logits and value estimate
        """
        # Handle single observation
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Shared backbone
        x = F.relu(self.fc1(obs))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout(x)

        # Policy head
        action_logits = self.policy_head(x)

        # Value head
        value = self.value_head(x)

        return action_logits, value

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Sample an action from the policy.

        Args:
            obs: Observation as numpy array
            deterministic: If True, return argmax action instead of sampling

        Returns:
            (action, log_prob, value): Action index, log probability, value estimate
        """
        with torch.no_grad():
            # Convert observation to tensor
            obs_tensor = torch.from_numpy(obs).float().to(self._device)

            # Forward pass
            logits, value = self.forward(obs_tensor)

            # Get probabilities via softmax
            probs = F.softmax(logits, dim=-1)
            probs_np = probs.cpu().numpy().flatten()

            # Sample action
            if deterministic:
                action = int(np.argmax(probs_np))
            else:
                # Add small epsilon for numerical stability
                probs_np = np.clip(probs_np, 1e-8, 1.0)
                probs_np /= probs_np.sum()
                action = np.random.choice(self.action_size, p=probs_np)

            # Compute log probability
            log_prob = float(np.log(probs_np[action] + 1e-8))

            # Get value
            value_np = float(value.cpu().numpy().flatten()[0])

        return action, log_prob, value_np

    def evaluate_actions(self, obs_batch: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a batch of observations and actions.

        Used during PPO update to compute new log probs and values.

        Args:
            obs_batch: Batch of observations, shape (batch_size, obs_size)
            actions: Batch of action indices, shape (batch_size,)

        Returns:
            (log_probs, values, entropy): Log probs of actions, values, policy entropy
        """
        # Forward pass
        logits, values = self.forward(obs_batch)

        # Log probabilities
        log_probs_all = F.log_softmax(logits, dim=-1)

        # Select log probs for taken actions
        batch_size = logits.shape[0]
        action_indices = actions.long()
        log_probs = log_probs_all.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # Entropy for exploration bonus
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(dim=-1).mean()

        return log_probs, values.squeeze(-1), entropy

    def save(self, filepath: str):
        """Save model weights to file."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str):
        """Load model weights from file."""
        self.load_state_dict(torch.load(filepath, map_location=self._device, weights_only=True))

    def clone(self) -> 'PolicyNetwork':
        """Create a copy of this network with the same weights."""
        clone = PolicyNetwork(
            self.obs_size, self.action_size, self.hidden_size
        )
        clone.to(self._device)
        clone.load_state_dict(self.state_dict())
        return clone


def test_policy_network():
    """Test the policy network."""
    print("Testing PolicyNetwork...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create network
    net = PolicyNetwork(obs_size=21, action_size=10, hidden_size=256)
    net.to(device)

    # Test single observation
    obs = np.random.randn(21).astype(np.float32)
    action, log_prob, value = net.get_action(obs)
    print(f"Single obs: action={action}, log_prob={log_prob:.4f}, value={value:.4f}")

    # Test batch observation
    obs_batch = torch.randn(32, 21, device=device)
    logits, values = net.forward(obs_batch)
    print(f"Batch forward: logits={logits.shape}, values={values.shape}")

    # Test action evaluation
    actions = torch.randint(0, 10, (32,), device=device)
    log_probs, values, entropy = net.evaluate_actions(obs_batch, actions)
    print(f"Evaluate: log_probs shape={log_probs.shape}, entropy={entropy.item():.4f}")

    # Test save/load
    net.save('/tmp/test_policy.pt')
    net2 = PolicyNetwork(obs_size=21, action_size=10, hidden_size=256)
    net2.to(device)
    net2.load('/tmp/test_policy.pt')
    print("Save/load: OK")

    # Test clone
    net3 = net.clone()
    print("Clone: OK")

    print("PolicyNetwork tests passed!")


if __name__ == "__main__":
    test_policy_network()
