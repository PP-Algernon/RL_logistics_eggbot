"""Moving-target event terms for the Eggtart mobile-grasp task.

The target is spawned with gravity disabled (see the env scene cfg), so once given a horizontal
velocity it coasts in a straight line at constant height -- a simple "moving target". The reset
event randomises its start pose + velocity; the interval event periodically re-randomises its
velocity so it changes direction during an episode.

For positioning/velocity at reset, the stock ``mdp.reset_root_state_uniform`` is used directly in
the env cfg; this module adds the periodic velocity re-randomisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_target_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    stage2_start_step: int = 64000,   # 阶段2开始
    stage3_start_step: int = 120000,  # 阶段3开始
) -> None:
    """Set a new random horizontal velocity on the target for the given envs.

    逆向课程学习集成：
    - 阶段1 (step < 64000):  随机速度（与 target_approach 配合）
    - 阶段2 (64000 <= step < 120000): 目标静止
    - 阶段3 (step >= 120000): 恢复随机速度

    ``velocity_range`` maps axis name ("x", "y") to a (min, max) range in m/s. Unspecified axes
    are set to zero. Designed to be used as an ``interval`` event so the target changes heading
    mid-episode.

    Args:
        env: 环境实例
        env_ids: 要更新的环境索引
        velocity_range: 速度范围字典 {"x": (min, max), "y": (min, max)}
        asset_cfg: 目标实体配置
        stage2_start_step: 阶段2开始步数（此后目标静止）
        stage3_start_step: 阶段3开始步数（此后恢复移动）
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = asset._ALL_INDICES  # type: ignore[attr-defined]

    n = len(env_ids)
    current_step = env.common_step_counter

    # 阶段2：目标静止
    if stage2_start_step <= current_step < stage3_start_step:
        new_vel = torch.zeros((n, 6), device=env.device)
    else:
        # 阶段1和阶段3：随机速度
        new_vel = torch.zeros((n, 6), device=env.device)  # (vx, vy, vz, wx, wy, wz)
        axis_to_col = {"x": 0, "y": 1, "z": 2}
        for axis, (lo, hi) in velocity_range.items():
            col = axis_to_col[axis]
            new_vel[:, col] = torch.empty(n, device=env.device).uniform_(lo, hi)

    root_vel = asset.data.root_vel_w[env_ids].clone()
    root_vel[:] = new_vel
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


def target_approach_ee_direction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    approach_speed: float,
    activation_distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="link_005"),
    grasp_offset: tuple[float, float, float] | None = None,
    direction_offset: tuple[float, float, float] | None = None,
    stage1_end_step: int = 64000,
) -> None:
    """逆向课程学习：目标主动缓慢靠近末端方向点（阶段1辅助学习）

    当抓取点距离目标小于 activation_distance 时，目标会以 approach_speed 的速度
    缓慢移动向 direction_offset 点，帮助策略在初期更容易完成"对准-闭合"动作。

    仅在 env.common_step_counter < stage1_end_step 时生效（阶段1）。

    典型用法（阶段1事件）:
        - activation_distance = 0.15  # 抓取点进入15cm内才触发
        - approach_speed = 0.05       # 目标以5cm/s缓慢靠近
        - grasp_offset = EGGTART_EE_GRASP_OFFSET
        - direction_offset = EGGTART_EE_GRASP_DERECT_OFFSET
        - stage1_end_step = 64000    # 阶段1结束步数

    Args:
        env: 环境实例
        env_ids: 要更新的环境索引
        approach_speed: 目标靠近的速度（m/s）
        activation_distance: 触发距离阈值（m）
        robot_cfg: 机器人实体配置
        target_cfg: 目标实体配置
        ee_cfg: link5 实体配置
        grasp_offset: 抓取点相对 link5 的偏移（局部系）
        direction_offset: 方向参考点相对 link5 的偏移（局部系）
        stage1_end_step: 阶段1结束步数（超过此步数则不执行）
    """
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import quat_apply

    # 阶段1结束后不执行
    if env.common_step_counter >= stage1_end_step:
        return

    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    if env_ids is None:
        env_ids = target._ALL_INDICES  # type: ignore[attr-defined]

    if grasp_offset is None or direction_offset is None:
        # 没有提供偏移量，不执行靠近逻辑
        return

    # 计算抓取点和方向点的世界坐标
    ee_pos_w = robot.data.body_pos_w[env_ids, ee_cfg.body_ids[0]]
    ee_quat_w = robot.data.body_quat_w[env_ids, ee_cfg.body_ids[0]]

    grasp_off = torch.tensor(grasp_offset, device=env.device, dtype=ee_pos_w.dtype)
    grasp_off = grasp_off.unsqueeze(0).expand(len(env_ids), -1)
    grasp_pos_w = ee_pos_w + quat_apply(ee_quat_w, grasp_off)

    dir_off = torch.tensor(direction_offset, device=env.device, dtype=ee_pos_w.dtype)
    dir_off = dir_off.unsqueeze(0).expand(len(env_ids), -1)
    dir_pos_w = ee_pos_w + quat_apply(ee_quat_w, dir_off)

    # 计算抓取点到目标的距离
    target_pos = target.data.root_pos_w[env_ids]
    dist_to_grasp = torch.norm(grasp_pos_w - target_pos, dim=1)

    # 只有距离小于阈值的环境才激活靠近行为
    active_mask = dist_to_grasp < activation_distance

    if not active_mask.any():
        return

    # 计算目标应该移动的方向（从当前位置指向方向点）
    to_direction = dir_pos_w - target_pos
    # 只考虑水平方向（保持高度不变）
    to_direction[:, 2] = 0
    direction_dist = torch.norm(to_direction, dim=1, keepdim=True)

    # 归一化方向向量
    to_direction = to_direction / (direction_dist + 1e-6)

    # 设置速度：active 的环境以 approach_speed 靠近，其他保持原速度
    new_vel = target.data.root_vel_w[env_ids].clone()

    # 只修改激活的环境
    active_indices = torch.where(active_mask)[0]
    if len(active_indices) > 0:
        new_vel[active_indices, 0] = to_direction[active_indices, 0] * approach_speed
        new_vel[active_indices, 1] = to_direction[active_indices, 1] * approach_speed
        new_vel[active_indices, 2] = 0  # 保持高度

    target.write_root_velocity_to_sim(new_vel, env_ids=env_ids)
