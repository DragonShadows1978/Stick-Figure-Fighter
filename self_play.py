"""
Self-Play Training Manager for Stick Figure Fighter.

Orchestrates:
- Dual-agent self-play games
- Experience collection
- Elo rating tracking
- Checkpoint management
- Training loop
"""
import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from policy import PolicyNetwork
from experience import Episode, ExperienceBuffer
from trainer import PPOTrainer, TrainingConfig
from game_engine import (
    create_game, GameLoop, GameState, InputState,
    get_observation, get_action_from_index, ACTION_SPACE_SIZE, OBSERVATION_SIZE
)


@dataclass
class EloRating:
    """Elo rating tracker for a policy."""
    rating: float = 1200.0
    games: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins / self.games


class EloTracker:
    """
    Tracks Elo ratings across training.

    Uses standard Elo formula:
    E_a = 1 / (1 + 10^((R_b - R_a) / 400))
    New_R_a = R_a + K * (S_a - E_a)

    where S_a is actual score (1 for win, 0 for loss, 0.5 for draw)
    """

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1200.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, EloRating] = {}
        self.history: List[Dict] = []

    def get_rating(self, agent_id: str) -> EloRating:
        """Get or create rating for an agent."""
        if agent_id not in self.ratings:
            self.ratings[agent_id] = EloRating(rating=self.initial_rating)
        return self.ratings[agent_id]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A vs player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, agent_a: str, agent_b: str, winner: Optional[int]):
        """
        Update ratings after a match.

        Args:
            agent_a: ID of first agent (player 0)
            agent_b: ID of second agent (player 1)
            winner: 0 if agent_a won, 1 if agent_b won, None for draw
        """
        ra = self.get_rating(agent_a)
        rb = self.get_rating(agent_b)

        # Calculate expected scores
        expected_a = self.expected_score(ra.rating, rb.rating)
        expected_b = 1.0 - expected_a

        # Actual scores
        if winner == 0:
            score_a, score_b = 1.0, 0.0
            ra.wins += 1
            rb.losses += 1
        elif winner == 1:
            score_a, score_b = 0.0, 1.0
            ra.losses += 1
            rb.wins += 1
        else:
            score_a, score_b = 0.5, 0.5

        # Update ratings
        ra.rating += self.k_factor * (score_a - expected_a)
        rb.rating += self.k_factor * (score_b - expected_b)
        ra.games += 1
        rb.games += 1

        # Record history
        self.history.append({
            'agent_a': agent_a,
            'agent_b': agent_b,
            'winner': winner,
            'rating_a': ra.rating,
            'rating_b': rb.rating
        })

    def save(self, filepath: str):
        """Save Elo tracker state."""
        data = {
            'ratings': {k: {
                'rating': v.rating,
                'games': v.games,
                'wins': v.wins,
                'losses': v.losses
            } for k, v in self.ratings.items()},
            'history': self.history[-1000:]  # Keep last 1000 games
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Load Elo tracker state."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.ratings = {}
        for k, v in data['ratings'].items():
            self.ratings[k] = EloRating(
                rating=v['rating'],
                games=v['games'],
                wins=v['wins'],
                losses=v['losses']
            )
        self.history = data['history']


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""
    # Game settings
    character: str = "FLEX"           # Character to use (FLEX vs FLEX for balanced)
    stage: str = "battlefield"
    stock_count: int = 3
    max_frames_per_game: int = 10800  # 3 minutes at 60 FPS

    # Reward shaping
    stock_taken_reward: float = 1.0
    stock_lost_reward: float = -1.0
    damage_dealt_bonus: float = 0.001  # Per 1% damage dealt
    damage_received_penalty: float = 0.0  # Per 1% damage received (optional)
    win_bonus: float = 0.5
    lose_penalty: float = -0.5

    # Training settings
    games_per_iteration: int = 100    # Games before PPO update
    total_games: int = 1000           # Total games to train
    save_every: int = 100             # Save checkpoint every N games
    eval_every: int = 50              # Evaluate every N games

    # Output
    checkpoint_dir: str = "checkpoints"
    log_file: str = "training_log.json"


class SelfPlayManager:
    """
    Manages self-play training loop.

    Two agents play against each other, collecting experience.
    Both agents use the same policy (current best).
    Periodically, we update the policy and track Elo improvement.
    """

    def __init__(self, config: SelfPlayConfig, device: str = 'cuda'):
        self.config = config
        self.device = device

        # Create policy network
        self.policy = PolicyNetwork(
            obs_size=OBSERVATION_SIZE,
            action_size=ACTION_SPACE_SIZE,
            hidden_size=256
        )
        self.policy.to(device)

        # Create trainer
        training_config = TrainingConfig(
            learning_rate=3e-4,
            batch_size=64,
            epochs_per_update=4,
            gamma=0.99,
            gae_lambda=0.95,
            entropy_coef=0.01,
            value_coef=0.5
        )
        self.trainer = PPOTrainer(self.policy, training_config, device=device)

        # Elo tracking
        self.elo_tracker = EloTracker()
        self.current_generation = 0

        # Previous generation policy for cross-gen Elo
        self.previous_policy: Optional[PolicyNetwork] = None

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Training statistics
        self.games_played = 0
        self.training_log: List[Dict] = []

        # Callbacks
        self.on_game_end: Optional[Callable] = None
        self.on_update: Optional[Callable] = None

    def run_cross_gen_game(self, policy_new: PolicyNetwork, policy_old: PolicyNetwork) -> Optional[int]:
        """
        Run a single game between current and previous generation.

        Args:
            policy_new: Current generation policy
            policy_old: Previous generation policy

        Returns:
            winner: 0 if new policy won, 1 if old policy won, None for draw
        """
        game = create_game(
            stage_name=self.config.stage,
            p1_character=self.config.character,
            p2_character=self.config.character,
            stock_count=self.config.stock_count
        )
        loop = GameLoop(game, headless=True)

        frame = 0
        while frame < self.config.max_frames_per_game:
            # Get observations
            obs = [get_observation(game, i) for i in range(2)]

            # New policy plays as P1, old policy plays as P2
            action_new, _, _ = policy_new.get_action(obs[0])
            action_old, _, _ = policy_old.get_action(obs[1])

            # Apply actions
            game.fighters[0].input = get_action_from_index(action_new)
            game.fighters[1].input = get_action_from_index(action_old)

            # Step simulation
            match_ongoing = loop.step()
            frame += 1

            if not match_ongoing or game.match_over:
                break

        return game.winner

    def evaluate_cross_gen(self, n_games: int = 5) -> List[Optional[int]]:
        """
        Run cross-generational evaluation games.

        Args:
            n_games: Number of games to play

        Returns:
            List of winners (0 = new gen, 1 = old gen, None = draw)
        """
        if self.previous_policy is None:
            return []

        results = []
        for i in range(n_games):
            # Alternate which side plays new/old
            if i % 2 == 0:
                winner = self.run_cross_gen_game(self.policy, self.previous_policy)
            else:
                winner = self.run_cross_gen_game(self.previous_policy, self.policy)
                # Flip winner since sides are swapped
                if winner is not None:
                    winner = 1 - winner
            results.append(winner)

        return results

    def run_game(self, render: bool = False) -> Tuple[List[Episode], Optional[int]]:
        """
        Run a single self-play game.

        Args:
            render: Whether to render the game (for evaluation)

        Returns:
            (episodes, winner): Episodes for both players and winner ID
        """
        # Create game
        game = create_game(
            stage_name=self.config.stage,
            p1_character=self.config.character,
            p2_character=self.config.character,
            stock_count=self.config.stock_count
        )
        loop = GameLoop(game, headless=not render)

        # Episodes for each player
        episodes = [Episode(player_id=0), Episode(player_id=1)]

        # Track damage for reward shaping
        prev_damage = [0.0, 0.0]
        prev_stocks = [game.fighters[0].stocks, game.fighters[1].stocks]

        frame = 0
        while frame < self.config.max_frames_per_game:
            # Get observations
            obs = [get_observation(game, i) for i in range(2)]

            # Get actions from policy
            actions = []
            log_probs = []
            values = []

            for i in range(2):
                action, log_prob, value = self.policy.get_action(obs[i])
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

            # Apply actions
            for i in range(2):
                game.fighters[i].input = get_action_from_index(actions[i])

            # Step simulation
            match_ongoing = loop.step()
            frame += 1

            # Compute rewards
            rewards = [0.0, 0.0]

            for i in range(2):
                opponent = 1 - i
                fighter = game.fighters[i]
                opp = game.fighters[opponent]

                # Damage dealt bonus
                damage_dealt = opp.damage_percent - prev_damage[opponent]
                if damage_dealt > 0:
                    rewards[i] += damage_dealt * self.config.damage_dealt_bonus

                # Damage received penalty
                damage_received = fighter.damage_percent - prev_damage[i]
                if damage_received > 0:
                    rewards[i] += damage_received * self.config.damage_received_penalty

                # Stock changes
                if opp.stocks < prev_stocks[opponent]:
                    # We took their stock
                    rewards[i] += self.config.stock_taken_reward
                if fighter.stocks < prev_stocks[i]:
                    # We lost a stock
                    rewards[i] += self.config.stock_lost_reward

            # Update tracking
            prev_damage = [game.fighters[i].damage_percent for i in range(2)]
            prev_stocks = [game.fighters[i].stocks for i in range(2)]

            # Check for match end
            done = not match_ongoing or game.match_over

            # Add to episodes
            for i in range(2):
                episodes[i].add(
                    obs=obs[i],
                    action=actions[i],
                    reward=rewards[i],
                    done=done,
                    log_prob=log_probs[i],
                    value=values[i]
                )

            if done:
                break

        # Match end rewards
        winner = game.winner

        if winner is not None:
            # Apply win/lose bonus
            final_rewards = [0.0, 0.0]
            final_rewards[winner] = self.config.win_bonus
            final_rewards[1 - winner] = self.config.lose_penalty

            # Add final transitions with win/lose rewards
            for i in range(2):
                if len(episodes[i].transitions) > 0:
                    episodes[i].transitions[-1].reward += final_rewards[i]

        return episodes, winner

    def train(self, progress_callback: Optional[Callable] = None):
        """
        Run the full training loop.

        Args:
            progress_callback: Optional callback(games_played, stats) for progress updates
        """
        print(f"Starting self-play training: {self.config.total_games} games")
        print(f"Character: {self.config.character} vs {self.config.character}")
        print(f"Device: {self.device}")

        start_time = time.time()
        iteration = 0

        while self.games_played < self.config.total_games:
            # Collect experience from multiple games
            all_episodes = []
            iteration_wins = [0, 0]

            for game_idx in range(self.config.games_per_iteration):
                episodes, winner = self.run_game(render=False)
                all_episodes.extend(episodes)
                self.games_played += 1

                if winner is not None:
                    iteration_wins[winner] += 1

                if self.games_played >= self.config.total_games:
                    break

            # Perform PPO update
            update_stats = self.trainer.update(all_episodes)

            # Run cross-generational Elo evaluation (5 games)
            cross_gen_wins = [0, 0]
            if self.current_generation > 0 and self.previous_policy is not None:
                cross_gen_results = self.evaluate_cross_gen(n_games=5)
                for winner in cross_gen_results:
                    if winner == 0:
                        cross_gen_wins[0] += 1  # New gen won
                    elif winner == 1:
                        cross_gen_wins[1] += 1  # Old gen won

                    # Update Elo between generations
                    self.elo_tracker.update(
                        f"gen_{self.current_generation}",
                        f"gen_{self.current_generation - 1}",
                        winner
                    )

            # Save current policy as previous for next iteration
            self.previous_policy = self.policy.clone()

            # Log statistics
            current_rating = self.elo_tracker.get_rating(f"gen_{self.current_generation}")
            stats = {
                'iteration': iteration,
                'games_played': self.games_played,
                'generation': self.current_generation,
                'elo_rating': current_rating.rating,
                'win_rate': current_rating.win_rate,
                'p1_wins': iteration_wins[0],
                'p2_wins': iteration_wins[1],
                'cross_gen_wins': cross_gen_wins[0],
                'cross_gen_losses': cross_gen_wins[1],
                'policy_loss': update_stats['policy_loss'],
                'value_loss': update_stats['value_loss'],
                'entropy': update_stats['entropy'],
                'mean_reward': update_stats['mean_reward'],
                'elapsed_time': time.time() - start_time
            }
            self.training_log.append(stats)

            # Print progress
            cross_gen_str = f"vs prev: {cross_gen_wins[0]}-{cross_gen_wins[1]}" if self.current_generation > 0 else "gen 0"
            print(f"[{self.games_played:5d}/{self.config.total_games}] "
                  f"Gen {self.current_generation} | "
                  f"Elo: {current_rating.rating:.0f} | "
                  f"{cross_gen_str} | "
                  f"Loss: {update_stats['policy_loss']:.4f} | "
                  f"Entropy: {update_stats['entropy']:.4f}")

            # Callback
            if progress_callback:
                progress_callback(self.games_played, stats)

            if self.on_update:
                self.on_update(stats)

            # Save checkpoint
            if self.games_played % self.config.save_every == 0:
                self.save_checkpoint()

            # Increment generation
            self.current_generation += 1
            iteration += 1

        # Final save
        self.save_checkpoint()

        elapsed = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"Total games: {self.games_played}")
        print(f"Total time: {elapsed:.1f}s ({self.games_played / elapsed:.1f} games/sec)")
        print(f"Final Elo: {self.elo_tracker.get_rating(f'gen_{self.current_generation - 1}').rating:.0f}")

    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{self.games_played:06d}.npz"
        )
        self.policy.save(checkpoint_path)

        # Save Elo tracker
        elo_path = os.path.join(self.config.checkpoint_dir, "elo_tracker.json")
        self.elo_tracker.save(elo_path)

        # Save training log
        log_path = os.path.join(self.config.checkpoint_dir, self.config.log_file)
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f)

        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        self.policy.load(checkpoint_path)

        # Try to load Elo tracker
        elo_path = os.path.join(self.config.checkpoint_dir, "elo_tracker.json")
        if os.path.exists(elo_path):
            self.elo_tracker.load(elo_path)

        print(f"Loaded checkpoint: {checkpoint_path}")

    def evaluate(self, num_games: int = 10, render: bool = False) -> Dict:
        """
        Evaluate the current policy.

        Args:
            num_games: Number of games to play
            render: Whether to render games

        Returns:
            Evaluation statistics
        """
        wins = [0, 0]
        game_lengths = []
        total_damage = [0.0, 0.0]

        for _ in range(num_games):
            episodes, winner = self.run_game(render=render)
            if winner is not None:
                wins[winner] += 1

            game_lengths.append(len(episodes[0]))

            # Sum damage dealt
            for i, ep in enumerate(episodes):
                dmg = sum(t.reward for t in ep.transitions
                         if t.reward > 0 and t.reward < 1.0)
                total_damage[i] += dmg

        return {
            'games': num_games,
            'p1_wins': wins[0],
            'p2_wins': wins[1],
            'win_rate': wins[0] / num_games if num_games > 0 else 0,
            'avg_game_length': np.mean(game_lengths),
            'avg_damage_p1': total_damage[0] / num_games,
            'avg_damage_p2': total_damage[1] / num_games
        }


def main():
    """Run self-play training."""
    config = SelfPlayConfig(
        character="FLEX",
        stage="battlefield",
        stock_count=3,
        games_per_iteration=10,  # Games before PPO update
        total_games=1000,
        save_every=100,
        checkpoint_dir="checkpoints/self_play"
    )

    manager = SelfPlayManager(config, device='cuda')

    # Run training
    manager.train()

    # Final evaluation
    print("\nFinal Evaluation:")
    eval_stats = manager.evaluate(num_games=20)
    print(f"  Win rate: {eval_stats['win_rate']:.1%}")
    print(f"  Avg game length: {eval_stats['avg_game_length']:.0f} frames")


if __name__ == "__main__":
    main()
