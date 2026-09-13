"""Target placement curriculum and optional target motion events."""

from __future__ import annotations

import math
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


def target_tracking_strength(current_step: int, stage1_end_step: int, stage1_start_step: int = 0) -> float:
    """第一阶段由 1 线性衰减到 0，第二阶段起恒为 0。"""
    if stage1_end_step <= stage1_start_step:
        raise ValueError("目标跟踪课程的结束步数必须大于开始步数")
    return min(1.0, max(0.0, (stage1_end_step - current_step) / (stage1_end_step - stage1_start_step)))


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
    stage1_start_step: int = 0,
    approach_gain: float = 6.0,
    max_grasp_height: float = 0.12,
    lift_height_threshold: float = 0.12,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["end_effector_joint"]),
    gripper_closed_threshold: float = 0.35,
) -> None:
    """第一阶段帮助地面物块对准低位夹爪，随全局训练步数渐退。

    夹爪张开、抓取点低于 max_grasp_height、目标未举起且距离在触发范围内时，
    用比例控制将目标水平速度引向跟踪点，速度上限 approach_speed (m/s)。
    越靠近跟踪点速度越小，防止恒速穿过夹爪；direction_offset 省略时跟踪抓取点。
    approach_gain 单位为 1/s。辅助强度在 stage1_start_step 到 stage1_end_step
    之间线性下降，用于混合自然速度和跟踪速度；第二阶段起完全不写物体状态。

    仅写激活环境的水平速度，保留竖直和角速度，绝不移动目标位姿或将其悬空。
    闭爪或举起后立即停止写速度，使夹持、提升和掉落由物理接触决定。
    """
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import quat_apply

    strength = target_tracking_strength(env.common_step_counter, stage1_end_step, stage1_start_step)
    if strength == 0.0 or approach_speed == 0.0:
        return
    if not all(math.isfinite(value) and value > 0 for value in (approach_speed, approach_gain, activation_distance)):
        raise ValueError("跟踪速度、比例增益和触发距离必须为有限正数")

    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    if env_ids is None:
        env_ids = target._ALL_INDICES  # type: ignore[attr-defined]

    # 计算抓取点和方向点的世界坐标
    ee_pos_w = robot.data.body_pos_w[env_ids, ee_cfg.body_ids[0]]
    ee_quat_w = robot.data.body_quat_w[env_ids, ee_cfg.body_ids[0]]

    grasp_off = torch.tensor(grasp_offset or (0.0, 0.0, 0.0), device=env.device, dtype=ee_pos_w.dtype)
    grasp_off = grasp_off.unsqueeze(0).expand(len(env_ids), -1)
    grasp_pos_w = ee_pos_w + quat_apply(ee_quat_w, grasp_off)

    dir_off = torch.tensor(direction_offset or grasp_offset or (0.0, 0.0, 0.0), device=env.device, dtype=ee_pos_w.dtype)
    dir_off = dir_off.unsqueeze(0).expand(len(env_ids), -1)
    dir_pos_w = ee_pos_w + quat_apply(ee_quat_w, dir_off)

    # 计算抓取点到目标的距离
    target_pos = target.data.root_pos_w[env_ids]
    dist_to_grasp = torch.norm(grasp_pos_w - target_pos, dim=1)

    gripper = env.scene[gripper_cfg.name]
    gripper_open = gripper.data.joint_pos[env_ids, gripper_cfg.joint_ids[0]] > gripper_closed_threshold
    active_mask = (
        (dist_to_grasp < activation_distance)
        & (grasp_pos_w[:, 2] <= max_grasp_height)
        & (target_pos[:, 2] < lift_height_threshold)
        & gripper_open
    )
    active_indices = active_mask.nonzero(as_tuple=False).flatten()
    if active_indices.numel() == 0:
        return

    active_ids = env_ids[active_indices]
    delta_xy = (dir_pos_w - target_pos)[active_indices, :2]
    # 单步目标位移不超过剩余距离，末端附近平滑收敛。
    tracking_vel = delta_xy * min(approach_gain, 1.0 / env.step_dt)
    speed = torch.linalg.vector_norm(tracking_vel, dim=1, keepdim=True)
    tracking_vel *= torch.clamp(approach_speed / speed.clamp_min(1e-6), max=1.0)
    new_vel = target.data.root_vel_w[active_ids].clone()
    new_vel[:, :2] = torch.lerp(new_vel[:, :2], tracking_vel, strength)
    target.write_root_velocity_to_sim(new_vel, env_ids=active_ids)


def place_target_in_reference_frame(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    positions: torch.Tensor,
    reference_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Place a stationary target using reference-body XY and absolute world Z.

    ``positions`` contains one (local x, local y, world z) row per env ID.
    Collection and curriculum resets share this transform, including zero velocity.
    """
    from isaaclab.utils.math import quat_apply

    robot = env.scene[reference_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = reference_cfg.body_ids[0]
    local_positions = positions.clone()
    local_positions[:, 2] = 0.0
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] = robot.data.body_pos_w[env_ids, body_id] + quat_apply(
        robot.data.body_quat_w[env_ids, body_id], local_positions
    )
    root_state[:, 2] = positions[:, 2]
    root_state[:, 7:13] = 0.0
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_target_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    stage1_pose_range: dict[str, tuple[float, float]],
    stage2_pose_range: dict[str, tuple[float, float]],
    stage3_pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="link_001"),
    stage2_start_step: int = 36000,
    stage3_start_step: int = 72000,
) -> None:
    """按阶段采样位置：XY 为 link_001 局部坐标，Z 为世界高度，初速度为零。

    Eggtart 的前方为局部 -Y，侧向为局部 X，与采集教师一致。
    """

    asset: RigidObject = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = asset._ALL_INDICES  # type: ignore[attr-defined]

    n = len(env_ids)
    current_step = env.common_step_counter

    # 根据当前步数选择位置范围
    if current_step < stage2_start_step:
        pose_range = stage1_pose_range
    elif current_step < stage3_start_step:
        pose_range = stage2_pose_range
    else:
        pose_range = stage3_pose_range

    # XY 采样在参考连杆局部系，Z 直接指定相对世界地面的高度。
    position_samples = torch.zeros((n, 3), device=env.device)
    axis_to_col = {"x": 0, "y": 1, "z": 2}
    for axis, (lo, hi) in pose_range.items():
        col = axis_to_col[axis]
        position_samples[:, col] = torch.empty(n, device=env.device).uniform_(lo, hi)

    place_target_in_reference_frame(env, env_ids, position_samples, reference_cfg, asset_cfg)
