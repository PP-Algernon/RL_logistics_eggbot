"""Custom termination terms for the Eggtart mobile-grasp task."""

from __future__ import annotations

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
