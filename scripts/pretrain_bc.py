"""Train a raw-observation BC actor compatible with the bundled rsl_rl 3.1.2.

Validation holds out entire successful episodes. BC optimizer state is saved
separately; PPO starts a fresh optimizer when loading this checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

# Use the same fork in BC, PPO training, and policy evaluation.
RSL_RL_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "rsl_rl_lib-3.1.2"
sys.path.insert(0, str(RSL_RL_ROOT))

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic


from demo_dataset import load_dataset


class BCTrainer:
    def __init__(self, obs_dim, action_dim, actor_hidden_dims=(256, 128, 64),
                 critic_hidden_dims=(256, 128, 64), activation="elu", learning_rate=3e-4,
                 device="cuda", init_noise_std=1.0):
        self.device = torch.device(device)
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.policy_cfg = dict(
            actor_hidden_dims=list(actor_hidden_dims), critic_hidden_dims=list(critic_hidden_dims),
            activation=activation, init_noise_std=init_noise_std,
            actor_obs_normalization=False, critic_obs_normalization=False,
        )
        example = TensorDict({"policy": torch.zeros(1, obs_dim, device=self.device)}, batch_size=[1])
        self.actor_critic = ActorCritic(
            obs=example, obs_groups={"policy": ["policy"], "critic": ["policy"]},
            num_actions=action_dim, **self.policy_cfg,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.actor.parameters(), lr=learning_rate)

    def predict(self, obs):
        obs = obs.to(self.device)
        return self.actor_critic.act_inference(TensorDict({"policy": obs}, batch_size=[len(obs)]))

    def train_epoch(self, observations, actions, batch_size=4096):
        self.actor_critic.train()
        order = torch.randperm(len(observations), device=observations.device)
        loss_sum = torch.zeros((), device=self.device)
        for batch in order.split(batch_size):
            prediction = self.predict(observations[batch])
            loss = torch.nn.functional.mse_loss(prediction, actions[batch].to(self.device))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), 1.0)
            self.optimizer.step()
            loss_sum += loss.detach() * len(batch)
        return (loss_sum / len(observations)).item()

    @torch.no_grad()
    def evaluate(self, observations, actions, batch_size=4096):
        self.actor_critic.eval()
        squared_error = torch.zeros(self.action_dim, device=self.device)
        for start in range(0, len(observations), batch_size):
            pred = self.predict(observations[start:start + batch_size])
            target = actions[start:start + batch_size].to(self.device)
            squared_error += (pred - target).square().sum(dim=0)
        per_action = squared_error / len(observations)
        return {"mse": per_action.mean().item(), "base_mse": per_action[:3].mean().item(),
                "arm_mse": per_action[3:8].mean().item(), "gripper_mse": per_action[8].item()}

    def save_checkpoint(self, filepath, epoch, metrics, metadata):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.actor_critic.state_dict(),
            "bc_optimizer_state_dict": self.optimizer.state_dict(),
            "iter": 0,
            "infos": {"bc_pretrain": True, "bc_epoch": epoch, "bc_loss": metrics["mse"],
                      "obs_mean": None, "obs_std": None, "obs_dim": self.obs_dim,
                      "action_dim": self.action_dim, "policy_cfg": self.policy_cfg,
                      "validation": metrics, **metadata},
        }, filepath)


def train_bc(data_path, output_path, actor_hidden_dims=(256, 128, 64), critic_hidden_dims=(256, 128, 64),
             activation="elu", learning_rate=3e-4, num_epochs=100, batch_size=4096,
             train_split=0.9, device="cuda", seed=42):
    if num_epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    train_obs, train_act, val_obs, val_act, metadata = load_dataset(data_path, device, train_split, seed)
    trainer = BCTrainer(train_obs.shape[1], train_act.shape[1], actor_hidden_dims, critic_hidden_dims,
                        activation, learning_rate, device)
    print(f"rsl_rl source: {RSL_RL_ROOT}")
    print("Observation normalization: disabled (matches PPO); actions retain their original scale")
    initial_metrics = trainer.evaluate(val_obs, val_act, batch_size)
    print(f"Untrained validation MSE: {initial_metrics['mse']:.6f}")
    history = []
    best_loss, best_epoch = float("inf"), 0
    for epoch in range(1, num_epochs + 1):
        loss = trainer.train_epoch(train_obs, train_act, batch_size)
        metrics = trainer.evaluate(val_obs, val_act, batch_size)
        if not math.isfinite(loss) or not all(math.isfinite(v) for v in metrics.values()):
            raise RuntimeError("Training produced non-finite loss")
        history.append({"epoch": epoch, "train_mse": loss, **metrics})
        print(f"Epoch {epoch:3d}/{num_epochs} | Train {loss:.6f} | Val {metrics['mse']:.6f} | "
              f"base {metrics['base_mse']:.6f} arm {metrics['arm_mse']:.6f} grip {metrics['gripper_mse']:.6f}",
              flush=True)
        if metrics["mse"] < best_loss:
            best_loss, best_epoch = metrics["mse"], epoch
            trainer.save_checkpoint(output_path, epoch, metrics, metadata)
        if epoch - best_epoch >= 20:
            print("Early stopping: validation has not improved for 20 epochs")
            break
    report = {"initial_validation": initial_metrics, "best_epoch": best_epoch,
              "best_val_mse": best_loss, "history": history}
    Path(str(output_path) + ".metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Best validation MSE: {best_loss:.6f} at epoch {best_epoch}; checkpoint: {output_path}")
    print("PPO initialization: train.py --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 "
          f"--resume --checkpoint {Path(output_path).resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/eggtart_demo.hdf5")
    parser.add_argument("--output", default="checkpoints/bc_pretrained.pt")
    parser.add_argument("--actor_hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--activation", choices=["elu", "relu", "tanh"], default="elu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_bc(args.data, args.output, args.actor_hidden_dims, args.critic_hidden_dims, args.activation,
             args.lr, args.epochs, args.batch_size, args.train_split, args.device, args.seed)


if __name__ == "__main__":
    main()
