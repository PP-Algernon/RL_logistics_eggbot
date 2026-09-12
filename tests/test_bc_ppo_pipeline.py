"""CPU regressions for demonstration splitting and the BC -> PPO/DAPG handoff.

Run with a Python environment providing torch, tensordict, h5py, and numpy:
    python -m unittest discover -s tests -v
"""
from pathlib import Path
import argparse
import copy
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "rsl_rl"))

import h5py
import numpy as np
import torch
from tensordict import TensorDict

from pretrain_bc import BCTrainer, load_dataset
from cli_args import add_rsl_rl_args, resolve_checkpoint
from rsl_rl.runners import OnPolicyRunner


class DummyEnv:
    num_envs, num_actions, device = 4, 9, "cpu"

    def get_observations(self):
        return TensorDict({"policy": torch.randn(self.num_envs, 44)}, batch_size=[self.num_envs])


def make_runner(policy_cfg):
    cfg = {
        "policy": {"class_name": "ActorCritic", **copy.deepcopy(policy_cfg)},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 2, "num_mini_batches": 2},
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 4, "save_interval": 1,
    }
    runner = OnPolicyRunner(DummyEnv(), cfg)
    runner.logger_type = "tensorboard"
    return runner


def update(runner):
    with torch.inference_mode():
        obs = runner.env.get_observations()
        for _ in range(4):
            actions = runner.alg.act(obs)
            obs = runner.env.get_observations()
            runner.alg.process_env_step(obs, -actions.square().mean(-1), torch.zeros(4), {})
        runner.alg.compute_returns(obs)
    return runner.alg.update()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "demo.hdf5"
        lengths = np.array([2, 3, 4, 5], dtype=np.int64)
        episode_ids = np.repeat(np.arange(4), lengths)
        obs = np.tile(episode_ids[:, None], (1, 44)).astype(np.float32)
        actions = np.full((len(obs), 9), -4.0, dtype=np.float32)
        with h5py.File(self.data, "w") as f:
            f["obs"], f["action"], f["episode_lengths"] = obs, actions, lengths
            f.attrs["successful_only"] = True
            f.attrs["env_name"] = "test-task"

    def test_episode_split_and_dataset_identity(self):
        train, act, val, _, meta = load_dataset(self.data)
        self.assertFalse(set(train[:, 0].tolist()) & set(val[:, 0].tolist()))
        self.assertEqual(len(train) + len(val), 14)
        self.assertTrue((act == -4).all())  # Absolute arm targets must not be clipped.
        restored = load_dataset(self.data, seed=7, split_metadata=meta)
        torch.testing.assert_close(restored[0], train)
        torch.testing.assert_close(restored[2], val)
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_dataset(self.data, expected_env="different-task")
        invalid = copy.deepcopy(meta)
        invalid["val_episode_indices"] = invalid["train_episode_indices"]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            load_dataset(self.data, split_metadata=invalid)
        with h5py.File(self.data, "r+") as f:
            f["action"][0, 0] = 2
        with self.assertRaisesRegex(ValueError, "differs"):
            load_dataset(self.data, split_metadata=meta)

    def test_reject_invalid_demonstrations(self):
        with h5py.File(self.data, "r+") as f:
            f.attrs["successful_only"] = False
        with self.assertRaisesRegex(ValueError, "successful_only"):
            load_dataset(self.data)
        with h5py.File(self.data, "r+") as f:
            f.attrs["successful_only"] = True
            f["obs"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            load_dataset(self.data)

    def test_legacy_dataset_task_label(self):
        with h5py.File(self.data, "r+") as f:
            f.attrs["env_name"] = "Isaac-Mobile-Grasp-Eggtart-BCPPO-v0"
        original = load_dataset(self.data)
        migrated = load_dataset(self.data, expected_env="Isaac-Mobile-Grasp-Eggtart-BCPPO-v0",
                                split_metadata=original[-1])
        self.assertEqual(migrated[-1], original[-1])
        torch.testing.assert_close(migrated[0], original[0])

    def test_bc_checkpoint_and_ppo_dapg_resume(self):
        obs, actions, val, val_actions, metadata = load_dataset(self.data)
        trainer = BCTrainer(44, 9, device="cpu")
        trainer.train_epoch(obs, actions, batch_size=4)
        checkpoint = Path(self.temp.name) / "bc.pt"
        trainer.save_checkpoint(checkpoint, 1, trainer.evaluate(val, val_actions), metadata)
        runner = make_runner(trainer.policy_cfg)
        runner.load(checkpoint)  # Must skip the incompatible BC optimizer automatically.
        self.assertFalse(runner.alg.optimizer.state)
        torch.testing.assert_close(runner.alg.policy.act_inference(
            TensorDict({"policy": obs}, batch_size=[len(obs)])), trainer.predict(obs))
        self.assertEqual(runner.demo_dataset_metadata, metadata)
        baseline_losses = update(runner)
        self.assertNotIn("dapg_bc", baseline_losses)
        self.assertEqual(runner.alg.dapg_updates, 0)

        runner.alg.set_demonstrations(TensorDict({"policy": obs}, batch_size=[len(obs)]),
                                      actions, batch_size=8)
        runner.alg.policy.zero_grad(set_to_none=True)
        runner.alg.policy.act(runner.env.get_observations())
        distribution = runner.alg.policy.distribution
        means = runner.alg.policy.action_mean.detach().clone()
        runner.alg.demonstration_loss().backward()
        self.assertIs(runner.alg.policy.distribution, distribution)
        torch.testing.assert_close(runner.alg.policy.action_mean, means)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in runner.alg.policy.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in runner.alg.policy.critic.parameters()))
        self.assertIsNone(runner.alg.policy.std.grad)

        losses = update(runner)
        self.assertTrue(all(np.isfinite(v) for v in losses.values()))
        self.assertAlmostEqual(losses["dapg_coefficient"], 0.1)
        runner.current_learning_iteration = 3
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = 2e-5
        ppo_checkpoint = Path(self.temp.name) / "ppo.pt"
        runner.save(ppo_checkpoint)
        resumed = make_runner(trainer.policy_cfg)
        resumed.load(ppo_checkpoint)
        self.assertTrue(resumed.alg.optimizer.state)
        self.assertEqual(resumed.alg.learning_rate, 2e-5)
        self.assertEqual(resumed.current_learning_iteration, 3)
        self.assertEqual(resumed.demo_dataset_metadata, metadata)
        self.assertEqual(resumed.alg.dapg_state_dict(), runner.alg.dapg_state_dict())
        self.assertIsNone(resumed.alg.demos)  # Data must be explicitly attached.
        resumed.alg.set_demonstrations(TensorDict({"policy": obs}, batch_size=[len(obs)]),
                                       actions, batch_size=8)
        self.assertAlmostEqual(update(resumed)["dapg_coefficient"], 0.095)

        incompatible = torch.load(checkpoint, weights_only=False)
        incompatible["infos"]["obs_mean"] = torch.zeros(44)
        torch.save(incompatible, checkpoint)
        with self.assertRaisesRegex(ValueError, "external normalization"):
            runner.load(checkpoint)

    def test_checkpoint_cli(self):
        parser = argparse.ArgumentParser()
        add_rsl_rl_args(parser)
        args = parser.parse_args(["--resume", "--load_checkpoint", str(self.data)])
        self.assertTrue(args.resume)
        self.assertEqual(resolve_checkpoint("unused", ".*", args.checkpoint), str(self.data))
        self.assertFalse(parser.parse_args(["--no-resume"]).resume)
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint("unused", ".*", str(self.data.parent / "missing.pt"))


if __name__ == "__main__":
    unittest.main()
