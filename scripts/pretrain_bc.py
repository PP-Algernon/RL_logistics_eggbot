"""Stage 1: BC (Behavior Cloning) Pretraining

Trains the actor network using supervised learning on demonstration data.
The trained weights are saved in rsl_rl checkpoint format for PPO fine-tuning.

Key points:
- Network structure must match rsl_rl's ActorCritic exactly
- Only train the actor (policy), critic stays randomly initialized
- Apply observation normalization consistent with PPO training
- Output checkpoint compatible with rsl_rl's --resume flag
"""

from __future__ import annotations

import argparse
import h5py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path

# Import rsl_rl for network structure
from rsl_rl.modules import ActorCritic


class BCTrainer:
    """Behavior Cloning trainer that outputs rsl_rl-compatible checkpoints"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str = "elu",
        learning_rate: float = 3e-4,
        device: str = "cuda",
    ):
        """Initialize BC trainer

        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            actor_hidden_dims: Hidden layer sizes for actor MLP
            critic_hidden_dims: Hidden layer sizes for critic MLP
            activation: Activation function name
            learning_rate: Learning rate for Adam optimizer
            device: Device to train on
        """
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Create ActorCritic matching rsl_rl structure
        self.actor_critic = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        ).to(self.device)

        # Only optimize actor parameters
        self.optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=learning_rate)

        # Loss function: Mean Squared Error
        self.criterion = nn.MSELoss()

        # Observation normalization
        self.obs_mean = None
        self.obs_std = None

        print(f"\n{'='*80}")
        print(f"BC Trainer Initialized")
        print(f"{'='*80}")
        print(f"Observation dim: {obs_dim}")
        print(f"Action dim: {action_dim}")
        print(f"Actor hidden dims: {actor_hidden_dims}")
        print(f"Critic hidden dims: {critic_hidden_dims}")
        print(f"Activation: {activation}")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {device}")
        print(f"{'='*80}\n")

    def compute_obs_normalization(self, observations: torch.Tensor):
        """Compute running mean/std for observation normalization

        Args:
            observations: [N, obs_dim] tensor of observations
        """
        self.obs_mean = observations.mean(dim=0)
        self.obs_std = observations.std(dim=0) + 1e-8  # Add epsilon for numerical stability

        print(f"Observation normalization computed:")
        print(f"  Mean: min={self.obs_mean.min():.4f}, max={self.obs_mean.max():.4f}")
        print(f"  Std: min={self.obs_std.min():.4f}, max={self.obs_std.max():.4f}")

    def normalize_obs(self, observations: torch.Tensor) -> torch.Tensor:
        """Normalize observations using computed mean/std

        Args:
            observations: [N, obs_dim] tensor

        Returns:
            Normalized observations
        """
        if self.obs_mean is None or self.obs_std is None:
            raise ValueError("Must call compute_obs_normalization first")
        return (observations - self.obs_mean) / self.obs_std

    def train_epoch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        batch_size: int = 4096,
    ) -> float:
        """Train for one epoch

        Args:
            observations: [N, obs_dim] normalized observations
            actions: [N, action_dim] actions
            batch_size: Batch size for training

        Returns:
            Average loss for this epoch
        """
        num_samples = len(observations)
        indices = torch.randperm(num_samples, device=self.device)

        total_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_obs = observations[batch_indices]
            batch_actions = actions[batch_indices]

            # Forward pass: get action mean from actor
            action_mean = self.actor_critic.act_inference(batch_obs)

            # Compute loss
            loss = self.criterion(action_mean, batch_actions)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        batch_size: int = 4096,
    ) -> float:
        """Evaluate on validation set

        Args:
            observations: [N, obs_dim] normalized observations
            actions: [N, action_dim] actions
            batch_size: Batch size for evaluation

        Returns:
            Average loss
        """
        self.actor_critic.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(observations), batch_size):
                batch_obs = observations[i : i + batch_size]
                batch_actions = actions[i : i + batch_size]

                action_mean = self.actor_critic.act_inference(batch_obs)
                loss = self.criterion(action_mean, batch_actions)

                total_loss += loss.item()
                num_batches += 1

        self.actor_critic.train()
        return total_loss / num_batches

    def save_checkpoint(self, filepath: str, epoch: int, loss: float):
        """Save checkpoint in rsl_rl format

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            loss: Current loss value
        """
        checkpoint = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter": 0,  # PPO will start from iteration 0
            "infos": {
                "bc_pretrain": True,
                "bc_epoch": epoch,
                "bc_loss": loss,
                "obs_mean": self.obs_mean.cpu().numpy().tolist() if self.obs_mean is not None else None,
                "obs_std": self.obs_std.cpu().numpy().tolist() if self.obs_std is not None else None,
            },
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")


def load_dataset(filepath: str, device: str = "cuda", train_split: float = 0.9):
    """Load demonstration dataset from HDF5

    Args:
        filepath: Path to HDF5 file
        device: Device to load tensors to
        train_split: Fraction of data to use for training (rest is validation)

    Returns:
        (train_obs, train_actions, val_obs, val_actions)
    """
    print(f"\nLoading dataset from {filepath}...")

    with h5py.File(filepath, "r") as f:
        obs_array = f["obs"][:]
        action_array = f["action"][:]

        # Print metadata
        if "env_name" in f.attrs:
            print(f"Environment: {f.attrs['env_name']}")
        if "success_rate" in f.attrs:
            print(f"Demo success rate: {f.attrs['success_rate']*100:.1f}%")

    num_samples = len(obs_array)
    print(f"Total samples: {num_samples:,}")
    print(f"Observation shape: {obs_array.shape}")
    print(f"Action shape: {action_array.shape}")

    # Convert to tensors
    observations = torch.tensor(obs_array, dtype=torch.float32, device=device)
    actions = torch.tensor(action_array, dtype=torch.float32, device=device)

    # Split train/val
    num_train = int(num_samples * train_split)
    indices = torch.randperm(num_samples, device=device)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_obs = observations[train_indices]
    train_actions = actions[train_indices]
    val_obs = observations[val_indices]
    val_actions = actions[val_indices]

    print(f"Train samples: {len(train_obs):,}")
    print(f"Validation samples: {len(val_obs):,}")

    return train_obs, train_actions, val_obs, val_actions


def train_bc(
    data_path: str,
    output_path: str,
    actor_hidden_dims: list[int],
    critic_hidden_dims: list[int],
    activation: str = "elu",
    learning_rate: float = 3e-4,
    num_epochs: int = 100,
    batch_size: int = 4096,
    train_split: float = 0.9,
    device: str = "cuda",
):
    """Train BC model and save checkpoint

    Args:
        data_path: Path to demonstration HDF5 file
        output_path: Path to save checkpoint
        actor_hidden_dims: Actor network hidden dimensions
        critic_hidden_dims: Critic network hidden dimensions
        activation: Activation function
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        train_split: Train/val split ratio
        device: Device to train on
    """
    # Load dataset
    train_obs, train_actions, val_obs, val_actions = load_dataset(
        data_path, device=device, train_split=train_split
    )

    # Get dimensions
    obs_dim = train_obs.shape[1]
    action_dim = train_actions.shape[1]

    # Create trainer
    trainer = BCTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        learning_rate=learning_rate,
        device=device,
    )

    # Compute observation normalization
    trainer.compute_obs_normalization(train_obs)

    # Normalize datasets
    train_obs_norm = trainer.normalize_obs(train_obs)
    val_obs_norm = trainer.normalize_obs(val_obs)

    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting BC Training")
    print(f"{'='*80}\n")

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_obs_norm, train_actions, batch_size=batch_size)

        # Validate
        val_loss = trainer.evaluate(val_obs_norm, val_actions, batch_size=batch_size)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            trainer.save_checkpoint(output_path, epoch, val_loss)

        # Early stopping check
        if epoch - best_epoch > 20:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for 20 epochs)")
            break

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    print(f"Checkpoint saved to: {output_path}")
    print(f"{'='*80}\n")

    # Next steps instructions
    print(f"Next Steps:")
    print(f"1. Test BC policy with play.py:")
    print(f"   ./isaaclab.sh -p scripts/rsl_rl/play.py --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \\")
    print(f"       --checkpoint {output_path}")
    print(f"\n2. Start PPO fine-tuning:")
    print(f"   ./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \\")
    print(f"       --num_envs 2048 --max_iterations 4000 --resume --load_checkpoint {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description="BC Pretraining for mobile grasp task")
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/eggtart_demo.hdf5",
        help="Path to demonstration HDF5 file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/bc_pretrained.pt",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--actor_hidden_dims",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="Actor hidden layer dimensions",
    )
    parser.add_argument(
        "--critic_hidden_dims",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="Critic hidden layer dimensions",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="elu",
        choices=["elu", "relu", "tanh"],
        help="Activation function",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--train_split", type=float, default=0.9, help="Train/val split ratio")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train_bc(
        data_path=args.data,
        output_path=args.output,
        actor_hidden_dims=args.actor_hidden_dims,
        critic_hidden_dims=args.critic_hidden_dims,
        activation=args.activation,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        train_split=args.train_split,
        device=args.device,
    )


if __name__ == "__main__":
    main()
