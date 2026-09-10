"""Stage 0: Demonstration Data Collection for BC Pretraining

Collects (observation, action) pairs using a scripted teacher policy.
The teacher implements:
- Navigation: P-controller to approach target
- Arm control: Follow scripted grasp posture
- Gripper: Close when target is within reach

Output: HDF5 file (datasets/eggtart_demo.hdf5) with observations and actions
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstration data for BC pretraining")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel environments")
parser.add_argument("--task", type=str, default="Isaac-Mobile-Grasp-Eggtart-Static-v0", help="Environment name")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum collection steps")
parser.add_argument("--noise_scale", type=float, default=0.05, help="Action noise std")
parser.add_argument("--output", type=str, default="datasets/eggtart_demo.hdf5", help="Output HDF5 file path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports after launching Isaac Sim"""

import h5py
import numpy as np
import torch
from datetime import datetime

import gymnasium as gym

# Import to register custom environments
import eggtart_grasp.tasks.mobile_grasp  # noqa: F401

# Isaac Lab imports
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate


class ScriptedTeacher:
    """Scripted policy that generates demonstration data

    State machine:
    1. NAVIGATE: Move base toward target
    2. REACH: Extend arm to grasp pose
    3. GRASP: Close gripper
    4. LIFT: Retract arm to lift object
    """

    NAVIGATE = 0
    REACH = 1
    GRASP = 2
    LIFT = 3

    def __init__(
        self,
        env,
        grasp_offset: tuple[float, float, float],
        approach_threshold: float = 0.15,
        reach_threshold: float = 0.05,
        grasp_joint_pos: list[float] = None,
    ):
        """Initialize scripted teacher

        Args:
            env: Isaac Lab environment (unwrapped)
            grasp_offset: Grasp point offset from EE body origin
            approach_threshold: Distance to start reaching (m)
            reach_threshold: Distance to close gripper (m)
            grasp_joint_pos: Target joint positions for grasp posture
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.grasp_offset = torch.tensor(grasp_offset, device=self.device, dtype=torch.float32)
        self.approach_threshold = approach_threshold
        self.reach_threshold = reach_threshold

        # Default grasp joint positions
        if grasp_joint_pos is None:
            grasp_joint_pos = [0.0, -0.5, 0.8, 0.3, 0.0]
        self.grasp_joint_pos = torch.tensor(grasp_joint_pos, device=self.device, dtype=torch.float32)

        # State tracking
        self.state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_in_state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Resolve entity configs
        self._resolve_entities()

    def _resolve_entities(self):
        """Get indices for robot bodies/joints"""
        self.robot_cfg = SceneEntityCfg("robot")
        self.robot_cfg.resolve(self.env.scene)

        self.ee_cfg = SceneEntityCfg("robot", body_names="link_005")
        self.ee_cfg.resolve(self.env.scene)

        self.target_cfg = SceneEntityCfg("target")
        self.target_cfg.resolve(self.env.scene)

    def _grasp_point_w(self) -> torch.Tensor:
        """Compute grasp point in world frame"""
        robot = self.env.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, self.ee_cfg.body_ids[0]]
        ee_quat = robot.data.body_quat_w[:, self.ee_cfg.body_ids[0]]
        offset = self.grasp_offset.unsqueeze(0).expand(self.num_envs, -1)
        return ee_pos + quat_apply(ee_quat, offset)

    def _target_in_base_frame(self) -> torch.Tensor:
        """Get target position in base frame"""
        robot = self.env.scene["robot"]
        target = self.env.scene["target"]

        base_pos_w = robot.data.root_pos_w
        base_quat_w = robot.data.root_quat_w
        target_pos_w = target.data.root_pos_w

        rel_pos_w = target_pos_w - base_pos_w
        base_quat_inv = quat_conjugate(base_quat_w)
        rel_pos_b = quat_apply(base_quat_inv, rel_pos_w)

        return rel_pos_b

    def compute_actions(self) -> torch.Tensor:
        """Compute scripted actions based on current state

        Returns:
            actions: [num_envs, 9] tensor (3 base vel + 5 arm joints + 1 gripper)
        """
        actions = torch.zeros(self.num_envs, 9, device=self.device)

        robot = self.env.scene["robot"]
        target = self.env.scene["target"]

        # Compute distances
        grasp_point = self._grasp_point_w()
        target_pos = target.data.root_pos_w
        dist_to_target = torch.norm(grasp_point - target_pos, dim=1)

        # Update state machine
        self._update_states(dist_to_target)

        # --- Base velocity (indices 0:3) ---
        target_b = self._target_in_base_frame()
        navigate = (self.state == self.NAVIGATE) | (self.state == self.REACH)

        if navigate.any():
            kp_lin = 1.0
            kp_ang = 2.0

            vx = torch.clamp(kp_lin * target_b[:, 0], -0.6, 0.6)
            vy = torch.clamp(kp_lin * target_b[:, 1], -0.6, 0.6)
            target_angle = torch.atan2(target_b[:, 1], target_b[:, 0])
            wz = torch.clamp(kp_ang * target_angle, -1.5, 1.5)

            actions[navigate, 0] = vx[navigate]
            actions[navigate, 1] = vy[navigate]
            actions[navigate, 2] = wz[navigate]

        grasp_lift = (self.state == self.GRASP) | (self.state == self.LIFT)
        actions[grasp_lift, 0:3] = 0.0

        # --- Arm joints (indices 3:8) ---
        robot_default_joints = robot.data.default_joint_pos[:, :5]
        reaching = (self.state == self.REACH) | (self.state == self.GRASP) | (self.state == self.LIFT)

        if reaching.any():
            target_pos = self.grasp_joint_pos.unsqueeze(0).expand(self.num_envs, -1)
            arm_actions = (target_pos - robot_default_joints) / 0.5
            actions[reaching, 3:8] = arm_actions[reaching]
        else:
            actions[~reaching, 3:8] = 0.0

        # --- Gripper (index 8) ---
        grasping = (self.state == self.GRASP) | (self.state == self.LIFT)
        actions[grasping, 8] = -1.0
        actions[~grasping, 8] = 0.0

        self.step_in_state += 1

        return actions

    def _update_states(self, dist_to_target: torch.Tensor):
        """Update state machine based on distance to target"""
        nav_to_reach = (self.state == self.NAVIGATE) & (dist_to_target < self.approach_threshold)
        self.state[nav_to_reach] = self.REACH
        self.step_in_state[nav_to_reach] = 0

        reach_to_grasp = (self.state == self.REACH) & (dist_to_target < self.reach_threshold)
        self.state[reach_to_grasp] = self.GRASP
        self.step_in_state[reach_to_grasp] = 0

        grasp_duration = 10
        grasp_to_lift = (self.state == self.GRASP) & (self.step_in_state >= grasp_duration)
        self.state[grasp_to_lift] = self.LIFT
        self.step_in_state[grasp_to_lift] = 0

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset state machine for given environments"""
        if env_ids is None:
            self.state.zero_()
            self.step_in_state.zero_()
        else:
            self.state[env_ids] = 0
            self.step_in_state[env_ids] = 0


def main():
    """Main collection function"""

    print(f"\n{'='*80}")
    print(f"Demonstration Data Collection for BC Pretraining")
    print(f"{'='*80}")
    print(f"Environment: {args_cli.task}")
    print(f"Num envs: {args_cli.num_envs}")
    print(f"Max steps: {args_cli.max_steps}")
    print(f"Action noise: {args_cli.noise_scale}")
    print(f"Output: {args_cli.output}")
    print(f"{'='*80}\n")

    # Create environment
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)

    # Unwrap to get base environment
    from isaaclab.envs import ManagerBasedRLEnv
    if hasattr(env, 'unwrapped'):
        base_env = env.unwrapped
    else:
        base_env = env

    # Get grasp offset from assets
    from eggtart_grasp.assets.eggtart import EGGTART_EE_GRASP_OFFSET, EGGTART_GRASP_JOINT_POS

    # Create teacher
    teacher = ScriptedTeacher(
        base_env,
        grasp_offset=EGGTART_EE_GRASP_OFFSET,
        approach_threshold=0.15,
        reach_threshold=0.05,
        grasp_joint_pos=EGGTART_GRASP_JOINT_POS,
    )

    # Storage
    obs_buffer = []
    action_buffer = []
    success_count = 0
    total_episodes = 0

    # Reset
    obs, _ = env.reset()
    teacher.reset()

    print("Starting data collection...")

    for step in range(args_cli.max_steps):
        # Get scripted actions
        actions = teacher.compute_actions()

        # Add noise for exploration
        if args_cli.noise_scale > 0:
            noise = torch.randn_like(actions) * args_cli.noise_scale
            noisy_actions = actions + noise
        else:
            noisy_actions = actions

        # Store data
        obs_buffer.append(obs.cpu().numpy())
        action_buffer.append(noisy_actions.cpu().numpy())

        # Step environment
        obs, rewards, terminated, truncated, infos = env.step(noisy_actions)

        # Track resets
        reset_mask = terminated | truncated
        if reset_mask.any():
            reset_ids = torch.where(reset_mask)[0]
            teacher.reset(reset_ids)
            total_episodes += len(reset_ids)

            # Count successes
            if "Episode_Reward/grasp" in infos:
                grasp_rewards = infos["Episode_Reward/grasp"]
                if isinstance(grasp_rewards, torch.Tensor):
                    success_count += (grasp_rewards[reset_ids] > 0).sum().item()

        # Progress
        if (step + 1) % 1000 == 0:
            collected = (step + 1) * args_cli.num_envs
            success_rate = (success_count / max(total_episodes, 1)) * 100
            print(f"Step {step+1}/{args_cli.max_steps} | Collected: {collected:,} samples | "
                  f"Episodes: {total_episodes} | Success rate: {success_rate:.1f}%")

    # Concatenate buffers
    print("\nProcessing collected data...")
    obs_array = np.concatenate(obs_buffer, axis=0)
    action_array = np.concatenate(action_buffer, axis=0)

    print(f"Total samples: {len(obs_array):,}")
    print(f"Observation shape: {obs_array.shape}")
    print(f"Action shape: {action_array.shape}")

    # Save to HDF5
    print(f"\nSaving to {args_cli.output}...")
    with h5py.File(args_cli.output, "w") as f:
        f.create_dataset("obs", data=obs_array, compression="gzip")
        f.create_dataset("action", data=action_array, compression="gzip")

        # Metadata
        f.attrs["env_name"] = args_cli.task
        f.attrs["num_envs"] = args_cli.num_envs
        f.attrs["num_samples"] = len(obs_array)
        f.attrs["noise_scale"] = args_cli.noise_scale
        f.attrs["timestamp"] = datetime.now().isoformat()
        f.attrs["success_rate"] = success_count / max(total_episodes, 1)

    env.close()

    print(f"\n{'='*80}")
    print(f"Collection complete!")
    print(f"Saved {len(obs_array):,} samples to {args_cli.output}")
    print(f"Success rate: {(success_count/max(total_episodes, 1))*100:.1f}%")
    print(f"{'='*80}\n")

    # Close simulation
    simulation_app.close()


if __name__ == "__main__":
    main()
