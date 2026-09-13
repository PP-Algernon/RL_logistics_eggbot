"""Custom termination terms for the Eggtart mobile-grasp task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_tipped(
    env: ManagerBasedRLEnv,
    min_up_proj: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base has tipped over.

    Projects the world up-axis onto the robot base and terminates when the base z-axis no longer
    points sufficiently upward (``< min_up_proj``).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # Base local z-axis expressed in world frame.
    base_z_axis = torch.zeros((env.num_envs, 3), device=env.device)
    base_z_axis[:, 2] = 1.0
    base_z_world = quat_apply(robot.data.root_quat_w, base_z_axis)
    return base_z_world[:, 2] < min_up_proj


class LiftSuccess(ManagerTermBase):
    """物块质心连续达到举升高度 dwell_time 秒后成功终止，仅判断高度。"""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._lift_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            self._lift_steps.zero_()
        else:
            self._lift_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_height_threshold: float,
        dwell_time: float,
        lift_dwell_time: float,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    ) -> torch.Tensor:
        if not (math.isfinite(dwell_time) and math.isfinite(lift_dwell_time)
                and 0.0 <= lift_dwell_time < dwell_time):
            raise ValueError("LiftSuccess 要求有限保持时间，且 dwell_time > lift_dwell_time >= 0")
        if not math.isfinite(lift_height_threshold):
            raise ValueError("LiftSuccess 的 lift_height_threshold 必须为有限值")
        target: RigidObject = env.scene[target_cfg.name]
        lifted = target.data.root_pos_w[:, 2] >= lift_height_threshold
        self._lift_steps = torch.where(lifted, self._lift_steps + 1, 0)
        return self._lift_steps >= math.ceil(dwell_time / env.step_dt)


lift_success = LiftSuccess


class TargetDropped(ManagerTermBase):
    """目标曾达到举升高度后跌落才终止，忽略回合初始的自然下落。"""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._was_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        """仅清除被重置环境的举升记录。"""
        if env_ids is None:
            self._was_lifted.zero_()
        else:
            self._was_lifted[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_height_threshold: float,
        drop_height_threshold: float = 0.05,
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    ) -> torch.Tensor:
        target: RigidObject = env.scene[target_cfg.name]
        height = target.data.root_pos_w[:, 2]
        self._was_lifted |= height >= lift_height_threshold
        return self._was_lifted & (height < drop_height_threshold)


target_dropped = TargetDropped
