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
    """脚本化策略教师，生成演示数据

    状态机：
    0. ALIGN: 原地旋转对准目标，不进行平移
    1. NAVIGATE: 直线靠近目标，保持朝向
    2. REACH: 机械臂切换到 grasp 姿态
    3. CLOSING/HOLDING: 底盘停止，夹爪闭合并保持
    4. RETRACT: 机械臂线性插值回 nominal 姿态
    """

    ALIGN = 0
    NAVIGATE = 1
    REACH = 2
    CLOSING = 3
    HOLDING = 4
    RETRACT = 5
    DONE = 6

    def __init__(
        self,
        env,
        grasp_offset: tuple[float, float, float],
        direction_offset: tuple[float, float, float],
        nominal_joint_pos: list[float],
        align_threshold: float = 0.01,            # 对齐完成的角度阈值（弧度，约5.7度）
        approach_threshold: float = 0.45,        # 开始伸手的距离阈值（米）
        reach_threshold: float = 0.03,           # 抓取判定的最大距离（米）
        line_tolerance: float = 0.01,            # 抓取线段的容差（米）
        grasp_joint_pos: list[float] | None = None,
        target_spawn_distance: float = 0.7,      # 目标生成距离（米）
        target_spawn_lateral: float = 0.0,       # 目标侧向偏移（米，正值为右侧）
        target_spawn_height: float = 0.10,       # 目标生成高度（米）
        target_approach_speed: float = 0.05,     # 目标靠近速度（米/秒）
        target_tracking_distance: float = 0.20,  # 目标开始跟踪的距离（米）
        target_front_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),  # 前方方向向量
        close_time: float = 0.30,                # 夹爪闭合时间（秒）
        hold_time: float = 0.40,                 # 夹爪保持时间（秒）
        retract_time: float = 1.00,              # 机械臂收回时间（秒）
    ):
        """初始化脚本化教师

        Args:
            env: Isaac Lab 环境（unwrapped）
            grasp_offset: 抓取点相对 EE body 原点的偏移（米）
            direction_offset: 方向参考点相对 EE body 原点的偏移（米）
            nominal_joint_pos: 五个机械臂关节的 nominal 姿态（弧度）
            align_threshold: 对齐完成的角度阈值（弧度）- 当朝向误差小于此值时，从 ALIGN 切换到 NAVIGATE
            approach_threshold: 开始伸手的距离阈值（米）- 当 link1 距目标小于此值时，从 NAVIGATE 切换到 REACH
            reach_threshold: 抓取判定的最大距离（米）- 目标必须在此距离内才能触发抓取
            line_tolerance: 抓取线段的容差（米）- 目标到抓取线段的最大距离
            grasp_joint_pos: 五个机械臂关节的 grasp 姿态（弧度），默认使用 nominal
            target_spawn_distance: 目标生成距离（米）- 目标在机器人正前方此距离处生成
            target_spawn_lateral: 目标侧向偏移（米）- 正值为工作面右侧，负值为左侧
            target_spawn_height: 目标生成高度（米）- 相对地面的高度
            target_approach_speed: 目标靠近速度（米/秒）- 目标主动靠近夹爪的速度
            target_tracking_distance: 目标开始跟踪的距离（米）- 抓取点进入此距离内时目标开始主动靠近
            target_front_axis: 前方方向向量 - 机械臂伸出的方向，默认 (0, -1, 0) 表示 -Y 方向
            close_time: 夹爪闭合时间（秒）
            hold_time: 夹爪保持时间（秒）
            retract_time: 机械臂收回时间（秒）
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.grasp_offset = torch.tensor(grasp_offset, device=self.device, dtype=torch.float32)
        self.direction_offset = torch.tensor(direction_offset, device=self.device, dtype=torch.float32)
        self.nominal_joint_pos = torch.tensor(nominal_joint_pos, device=self.device, dtype=torch.float32)
        self.align_threshold = align_threshold
        self.approach_threshold = approach_threshold
        self.reach_threshold = reach_threshold
        self.line_tolerance = line_tolerance
        self.target_spawn_distance = target_spawn_distance
        self.target_spawn_lateral = target_spawn_lateral
        self.target_spawn_height = target_spawn_height
        self.target_approach_speed = target_approach_speed
        self.target_tracking_distance = target_tracking_distance
        front_axis = torch.tensor(target_front_axis, device=self.device, dtype=torch.float32)
        self.target_front_axis = front_axis / torch.linalg.vector_norm(front_axis).clamp_min(1e-6)

        # The action term is JointPositionAction(scale=0.5, use_default_offset=True), so
        # zero means nominal and this is the normalized action for the grasp posture.
        if grasp_joint_pos is None:
            grasp_joint_pos = nominal_joint_pos
        self.grasp_joint_pos = torch.tensor(grasp_joint_pos, device=self.device, dtype=torch.float32)
        self.grasp_arm_action = (self.grasp_joint_pos - self.nominal_joint_pos) / 0.5

        dt = float(getattr(env, "step_dt", 1.0 / 30.0))
        self.close_steps = max(1, int(round(close_time / dt)))
        self.hold_steps = max(1, int(round(hold_time / dt)))
        self.retract_steps = max(1, int(round(retract_time / dt)))

        # State tracking
        self.state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_in_state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frozen_arm = torch.zeros(self.num_envs, 5, device=self.device)

        # Resolve entity configs
        self._resolve_entities()

    def _resolve_entities(self):
        """Get indices for robot bodies/joints"""
        self.robot_cfg = SceneEntityCfg("robot")
        self.robot_cfg.resolve(self.env.scene)

        self.ee_cfg = SceneEntityCfg("robot", body_names="link_005")
        self.ee_cfg.resolve(self.env.scene)

        # link_001 是机械臂基座，用于计算朝向
        self.link1_cfg = SceneEntityCfg("robot", body_names="link_001")
        self.link1_cfg.resolve(self.env.scene)

        self.target_cfg = SceneEntityCfg("target")
        self.target_cfg.resolve(self.env.scene)

    def _grasp_point_w(self) -> torch.Tensor:
        """Compute grasp point in world frame"""
        robot = self.env.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, self.ee_cfg.body_ids[0]]
        ee_quat = robot.data.body_quat_w[:, self.ee_cfg.body_ids[0]]
        offset = self.grasp_offset.unsqueeze(0).expand(self.num_envs, -1)
        return ee_pos + quat_apply(ee_quat, offset)

    def _grasp_line_points_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the grasp and direction points in world coordinates."""
        robot = self.env.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, self.ee_cfg.body_ids[0]]
        ee_quat = robot.data.body_quat_w[:, self.ee_cfg.body_ids[0]]
        grasp_offset = self.grasp_offset.unsqueeze(0).expand(self.num_envs, -1)
        direction_offset = self.direction_offset.unsqueeze(0).expand(self.num_envs, -1)
        return (
            ee_pos + quat_apply(ee_quat, grasp_offset),
            ee_pos + quat_apply(ee_quat, direction_offset),
        )

    def _target_on_grasp_line(self) -> torch.Tensor:
        """判断目标是否在抓取线段上且距离足够近

        抓取线段由 grasp_offset 和 direction_offset 两点定义。
        判定条件：
        1. 目标在线段上（投影比例在 [0, 1] 之间）
        2. 目标到线段的距离 <= line_tolerance
        3. 目标到抓取点的距离 <= reach_threshold

        Returns:
            bool tensor [num_envs]: 是否满足抓取条件
        """
        target = self.env.scene["target"]
        grasp_pos, direction_pos = self._grasp_line_points_w()
        target_pos = target.data.root_pos_w

        # 计算线段方向和长度
        line = direction_pos - grasp_pos
        line_sq = torch.sum(line * line, dim=1).clamp_min(1e-8)

        # 计算目标在线段上的投影
        target_from_grasp = target_pos - grasp_pos
        projection = torch.sum(target_from_grasp * line, dim=1) / line_sq

        # 计算目标到线段的最近点
        closest = grasp_pos + projection.unsqueeze(1) * line
        line_distance = torch.linalg.vector_norm(target_pos - closest, dim=1)
        grasp_distance = torch.linalg.vector_norm(target_from_grasp, dim=1)

        # 三个条件都满足才触发抓取
        on_segment = (projection >= 0.0) & (projection <= 1.0)
        return on_segment & (line_distance <= self.line_tolerance) & (grasp_distance <= self.reach_threshold)

    def _place_target_in_front(self, env_ids: torch.Tensor | None = None) -> None:
        """将目标放置在每个机器人正前方的固定位置

        使用机器人的当前朝向，将目标放置在工作面（机械臂伸出方向）前方。
        """
        robot = self.env.scene["robot"]
        target = self.env.scene["target"]
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        base_pos = robot.data.root_pos_w[env_ids]
        base_quat = robot.data.root_quat_w[env_ids]

        # 在机器人局部坐标系中构造目标位置
        # target_front_axis 定义了工作面方向（对 Eggtart 是 -Y 方向）
        local_pos = self.target_front_axis.unsqueeze(0).expand(env_ids.numel(), -1).clone()
        local_pos = local_pos * self.target_spawn_distance  # 前方距离

        # 侧向偏移沿局部 X 轴（垂直于前方方向）
        # 正值 = 面向工作面时的右侧，负值 = 左侧
        local_pos[:, 0] += self.target_spawn_lateral

        # 设置高度（局部 Z 轴）
        local_pos[:, 2] = self.target_spawn_height

        # 转换到世界坐标系
        root_state = target.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = base_pos + quat_apply(base_quat, local_pos)
        root_state[:, 7:13] = 0.0  # 速度清零
        target.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _track_target(self) -> None:
        """让目标主动靠近末端执行器（使用 mdp.target 中的实现）

        在 CLOSING 之前的状态，如果抓取点距离目标小于 target_tracking_distance，
        目标会以 target_approach_speed 的速度缓慢移动向 direction_offset 点。
        这可以帮助策略更容易完成"对准-闭合"动作。

        注意：ALIGN 状态不启用目标跟踪，让机器人先完成对准。
        """
        active = (self.state > self.ALIGN) & (self.state < self.CLOSING)
        if not active.any():
            return
        from eggtart_grasp.tasks.mobile_grasp.mdp.target import target_approach_ee_direction

        env_ids = torch.nonzero(active, as_tuple=False).flatten()
        target_approach_ee_direction(
            self.env,
            env_ids,
            approach_speed=self.target_approach_speed,          # 目标靠近速度
            activation_distance=self.target_tracking_distance,  # 触发距离
            robot_cfg=self.robot_cfg,
            target_cfg=self.target_cfg,
            ee_cfg=self.ee_cfg,
            grasp_offset=tuple(self.grasp_offset.tolist()),
            direction_offset=tuple(self.direction_offset.tolist()),
            stage1_end_step=2**62,  # 永不结束（演示收集时始终启用）
        )

    def _target_in_base_frame(self) -> torch.Tensor:
        """获取目标在底盘坐标系下的位置

        Returns:
            [num_envs, 3]: 目标在底盘局部坐标系的位置
        """
        robot = self.env.scene["robot"]
        target = self.env.scene["target"]

        base_pos_w = robot.data.root_pos_w
        base_quat_w = robot.data.root_quat_w
        target_pos_w = target.data.root_pos_w

        rel_pos_w = target_pos_w - base_pos_w
        base_quat_inv = quat_conjugate(base_quat_w)
        rel_pos_b = quat_apply(base_quat_inv, rel_pos_w)

        return rel_pos_b

    def _compute_desired_heading(self) -> torch.Tensor:
        """计算让机械臂 link1（基座）朝向对准目标所需的朝向角

        使用 link1 坐标系计算误差：
        - link1 是机械臂旋转基座，其朝向直接影响抓取方向
        - 计算目标在 link1 坐标系下的方向角
        - 返回底盘需要旋转的角度

        Returns:
            [num_envs]: 所需朝向角（弧度）
        """
        robot = self.env.scene["robot"]
        target = self.env.scene["target"]

        # 获取 link1 和 target 的世界坐标
        link1_pos_w = robot.data.body_pos_w[:, self.link1_cfg.body_ids[0]]
        link1_quat_w = robot.data.body_quat_w[:, self.link1_cfg.body_ids[0]]
        target_pos_w = target.data.root_pos_w

        # 计算目标相对于 link1 的位置（link1 局部坐标系）
        rel_pos_w = target_pos_w - link1_pos_w
        link1_quat_inv = quat_conjugate(link1_quat_w)
        target_in_link1 = quat_apply(link1_quat_inv, rel_pos_w)

        # 在 link1 坐标系的水平面上计算目标方向角（忽略 Z）
        # 目标应该在 link1 的某个特定方向（比如 -Y 方向，与 base 的工作面一致）
        target_angle_link1 = torch.atan2(target_in_link1[:, 1], target_in_link1[:, 0])

        # link1 的期望朝向（在 link1 坐标系中，工作面方向）
        # 假设 link1 的 -Y 方向是机械臂前方
        front_xy = self.target_front_axis[:2]
        desired_angle_link1 = torch.atan2(front_xy[1], front_xy[0])

        # link1 需要旋转的角度
        heading_error_link1 = target_angle_link1 - desired_angle_link1

        # 归一化到 [-π, π]
        heading_error = torch.atan2(torch.sin(heading_error_link1), torch.cos(heading_error_link1))

        return heading_error

    def compute_actions(self) -> torch.Tensor:
        """根据当前状态计算脚本化动作
        Returns:
            actions: [num_envs, 9] tensor (3 底盘速度 + 5 机械臂关节 + 1 夹爪)
        """
        actions = torch.zeros(self.num_envs, 9, device=self.device)

        robot = self.env.scene["robot"]
        target = self.env.scene["target"]

        # 更新状态机（使用抓取线段判定准则）
        # 使用 link1 到目标的距离来判断状态转换
        link1_pos_w = robot.data.body_pos_w[:, self.link1_cfg.body_ids[0]]
        dist_to_target = torch.linalg.vector_norm(link1_pos_w - target.data.root_pos_w, dim=1)
        heading_error = self._compute_desired_heading()
        self._update_states(dist_to_target, heading_error, self._target_on_grasp_line())
        self._track_target()

        # === 底盘速度控制（indices 0:3：vx, vy, wz）===
        target_b = self._target_in_base_frame()
        heading_error = self._compute_desired_heading()

        # P 控制器增益
        kp_lin = 1.0   # 线速度比例增益
        kp_ang = 2.0   # 角速度比例增益

        # ALIGN 状态：只旋转，不平移
        aligning = self.state == self.ALIGN
        if aligning.any():
            wz = torch.clamp(kp_ang * heading_error, -1.5, 1.5)     # 角速度限制 ±1.5 rad/s
            actions[aligning, 0] = 0.0  # 不前后移动
            actions[aligning, 1] = 0.0  # 不左右移动
            actions[aligning, 2] = wz[aligning]

        # NAVIGATE 状态：直线靠近，轻微修正朝向
        navigating = self.state == self.NAVIGATE
        if navigating.any():
            # 计算目标速度（底盘局部坐标系）
            vx = torch.clamp(kp_lin * target_b[:, 0], -0.6, 0.6)    # X 方向速度限制 ±0.6 m/s
            vy = torch.clamp(kp_lin * target_b[:, 1], -0.6, 0.6)    # Y 方向速度限制 ±0.6 m/s

            # 轻微修正朝向误差（降低增益，避免绕圈）
            wz = torch.clamp(kp_ang * 0.3 * heading_error, -0.8, 0.8)  # 大幅降低角速度

            actions[navigating, 0] = vx[navigating]
            actions[navigating, 1] = vy[navigating]
            actions[navigating, 2] = wz[navigating]

        # REACH 状态：只靠近，不旋转（避免绕圈）
        reaching = self.state == self.REACH
        if reaching.any():
            # 只计算平移速度，不再修正朝向
            vx = torch.clamp(kp_lin * 0.5 * target_b[:, 0], -0.3, 0.3)  # 降速靠近
            vy = torch.clamp(kp_lin * 0.5 * target_b[:, 1], -0.3, 0.3)  # 降速靠近

            actions[reaching, 0] = vx[reaching]
            actions[reaching, 1] = vy[reaching]
            actions[reaching, 2] = 0.0  # 不旋转，避免绕圈

        # CLOSING 之后底盘停止
        stopped = self.state >= self.CLOSING
        actions[stopped, 0:3] = 0.0

        # === 机械臂关节控制（indices 3:8：5 个关节）===
        # ALIGN/NAVIGATE: 保持 nominal 姿态（action=0，因为 use_default_offset=True）
        # REACH: 切换到 grasp 姿态
        reaching = self.state == self.REACH
        actions[reaching, 3:8] = self.grasp_arm_action

        # CLOSING/HOLDING: 冻结在接管瞬间的姿态
        freeze = (self.state == self.CLOSING) | (self.state == self.HOLDING)
        actions[freeze, 3:8] = self.frozen_arm[freeze]

        # RETRACT: 线性插值回 nominal（action=0）
        retract = self.state == self.RETRACT
        if retract.any():
            alpha = (self.step_in_state[retract].float() / self.retract_steps).clamp(0.0, 1.0)
            actions[retract, 3:8] = self.frozen_arm[retract] * (1.0 - alpha).unsqueeze(1)

        # DONE: 保持 nominal（action=0）

        # === 夹爪控制（index 8）===
        # CLOSING 之后保持闭合（action=-1.0）
        grasping = self.state >= self.CLOSING
        actions[grasping, 8] = -1.0

        self.step_in_state += 1
        self._advance_states()

        return actions

    def _update_states(self, dist_to_target: torch.Tensor, heading_error: torch.Tensor, grasp_aligned: torch.Tensor):
        """根据距离、朝向和对齐状态更新状态机

        状态只能单向递进：ALIGN → NAVIGATE → REACH → CLOSING → HOLDING → RETRACT → DONE

        Args:
            dist_to_target: link1 到目标的距离
            heading_error: 朝向误差（弧度）
            grasp_aligned: 目标是否在抓取线段上且满足抓取条件
        """
        # ALIGN -> NAVIGATE: 朝向对齐完成后开始直线靠近
        align_to_nav = (self.state == self.ALIGN) & (torch.abs(heading_error) < self.align_threshold)
        if align_to_nav.any():
            self.state[align_to_nav] = self.NAVIGATE
            self.step_in_state[align_to_nav] = 0

        # NAVIGATE -> REACH: 距离小于阈值时开始伸手
        nav_to_reach = (self.state == self.NAVIGATE) & (dist_to_target < self.approach_threshold)
        if nav_to_reach.any():
            self.state[nav_to_reach] = self.REACH
            self.step_in_state[nav_to_reach] = 0

        # REACH -> CLOSING: 满足抓取条件时开始闭合夹爪
        reach_to_close = (self.state == self.REACH) & grasp_aligned
        if reach_to_close.any():
            self.frozen_arm[reach_to_close] = self.grasp_arm_action  # 冻结当前机械臂姿态
            self.state[reach_to_close] = self.CLOSING
            self.step_in_state[reach_to_close] = 0

    def _advance_states(self) -> None:
        """根据时间推进状态机

        状态转换（单向递进）：
        - CLOSING -> HOLDING: 闭合时间到达后保持
        - HOLDING -> RETRACT: 保持时间到达后开始收回
        - RETRACT -> DONE: 收回时间到达后完成

        状态只能向前递进，不能回退。
        """
        to_hold = (self.state == self.CLOSING) & (self.step_in_state >= self.close_steps)
        to_retract = (self.state == self.HOLDING) & (self.step_in_state >= self.hold_steps)
        to_done = (self.state == self.RETRACT) & (self.step_in_state >= self.retract_steps)
        for mask, next_state in (
            (to_hold, self.HOLDING),
            (to_retract, self.RETRACT),
            (to_done, self.DONE),
        ):
            if mask.any():
                self.state[mask] = next_state
                self.step_in_state[mask] = 0

    def reset(self, env_ids: torch.Tensor | None = None):
        """重置指定环境的状态机

        同时将目标放置在机器人正前方的固定位置。

        Args:
            env_ids: 要重置的环境索引，None 表示重置所有环境
        """
        if env_ids is None:
            self.state.zero_()
            self.step_in_state.zero_()
            self.frozen_arm.zero_()
            self._place_target_in_front()
        else:
            self.state[env_ids] = 0
            self.step_in_state[env_ids] = 0
            self.frozen_arm[env_ids] = 0.0
            self._place_target_in_front(env_ids)


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
    from eggtart_grasp.tasks.mobile_grasp.config.eggtart import grasp_env_cfg

    _ENV_CFG_MAP = {
        "Isaac-Mobile-Grasp-Eggtart-v0": grasp_env_cfg.EggtartMobileGraspEnvCfg,
        "Isaac-Mobile-Grasp-Eggtart-Static-v0": grasp_env_cfg.EggtartMobileGraspEnvStaticCfg,
        "Isaac-Mobile-Grasp-Eggtart-Play-v0": grasp_env_cfg.EggtartMobileGraspEnvCfg_PLAY,
        "Isaac-Mobile-Grasp-Eggtart-BCPPO-v0": grasp_env_cfg.EggtartMobileGraspEnvStaticBCPPOCfg,
    }
    if args_cli.task not in _ENV_CFG_MAP:
        raise ValueError(f"Unknown task: {args_cli.task}. Available: {list(_ENV_CFG_MAP)}")
    env_cfg = _ENV_CFG_MAP[args_cli.task]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Unwrap to get base environment
    from isaaclab.envs import ManagerBasedRLEnv
    if hasattr(env, 'unwrapped'):
        base_env = env.unwrapped
    else:
        base_env = env

    # Get grasp offset from assets
    from eggtart_grasp.assets.eggtart import (
        EGGTART_ARM_JOINT_NAMES,
        EGGTART_BASE_FORWARD_AXIS,
        EGGTART_EE_GRASP_DERECT_OFFSET,
        EGGTART_EE_GRASP_OFFSET,
        EGGTART_GRASP_JOINT_POS,
        EGGTART_NOMINAL_JOINT_POS,
    )

    # EGGTART_GRASP_JOINT_POS 是按关节名索引的 dict，ScriptedTeacher 需要 5 个臂关节
    # 的有序 list（顺序与 EGGTART_ARM_JOINT_NAMES 一致）
    grasp_joint_pos_list = [EGGTART_GRASP_JOINT_POS[name] for name in EGGTART_ARM_JOINT_NAMES]
    nominal_joint_pos_list = [EGGTART_NOMINAL_JOINT_POS[name] for name in EGGTART_ARM_JOINT_NAMES]

    # 创建脚本化教师
    print(f"\n初始化 ScriptedTeacher 参数:")
    print(f"  - 前方方向向量: {EGGTART_BASE_FORWARD_AXIS}")
    print(f"  - 目标生成距离: 0.7 米")
    print(f"  - 目标侧向偏移: 0.0 米（正值=右侧，负值=左侧）")
    print(f"  - 目标生成高度: 0.10 米")
    print(f"  - 抓取点偏移: {EGGTART_EE_GRASP_OFFSET}")
    print(f"  - 方向点偏移: {EGGTART_EE_GRASP_DERECT_OFFSET}\n")

    teacher = ScriptedTeacher(
        base_env,
        grasp_offset=EGGTART_EE_GRASP_OFFSET,
        direction_offset=EGGTART_EE_GRASP_DERECT_OFFSET,
        nominal_joint_pos=nominal_joint_pos_list,
        grasp_joint_pos=grasp_joint_pos_list,
        align_threshold=0.1,           # 对齐完成的角度阈值（弧度，约5.7度）
        approach_threshold=0.15,       # 开始伸手的距离阈值（米）
        reach_threshold=0.03,          # 抓取判定的最大距离（米）
        line_tolerance=0.01,           # 抓取线段的容差（米）
        target_spawn_distance=0.7,     # 目标生成距离（米）
        target_spawn_lateral=0.0,      # 目标侧向偏移（米，正值为右侧）
        target_spawn_height=0.10,      # 目标生成高度（米）
        target_approach_speed=0.05,    # 目标靠近速度（米/秒）
        target_tracking_distance=0.20, # 目标开始跟踪的距离（米）
        target_front_axis=EGGTART_BASE_FORWARD_AXIS,  # 前方方向向量
        close_time=0.30,               # 夹爪闭合时间（秒）
        hold_time=0.40,                # 夹爪保持时间（秒）
        retract_time=1.00,             # 机械臂收回时间（秒）
    )

    # Storage
    obs_buffer = []
    action_buffer = []
    success_count = 0
    total_episodes = 0

    def _flatten_obs(obs):
        """把 gymnasium 返回的 obs（可能是 dict，键为观测组名）拼成 [num_envs, obs_dim] tensor。

        顺序与 RslRlVecEnvWrapper（PPO 训练用）一致：按 dict 键序拼接；
        观测配置里只有一个 'policy' 组时，即该组本身。
        """
        if isinstance(obs, dict):
            print(f"[INFO] obs 为 dict，键: {list(obs.keys())}") if not obs_buffer else None
            return torch.cat([obs[k] for k in obs.keys()], dim=-1)
        return obs

    # Reset
    obs, _ = env.reset()
    obs = _flatten_obs(obs)
    teacher.reset()

    # 打印第一个环境的初始位置信息（用于调试坐标系）
    robot = base_env.scene["robot"]
    target = base_env.scene["target"]
    print(f"\n[调试] 第一个环境的初始位置:")
    print(f"  机器人位置: {robot.data.root_pos_w[0].cpu().numpy()}")
    print(f"  目标位置: {target.data.root_pos_w[0].cpu().numpy()}")
    rel_pos = target.data.root_pos_w[0] - robot.data.root_pos_w[0]
    print(f"  相对位置 (目标-机器人): {rel_pos.cpu().numpy()}")
    print(f"  水平距离: {torch.norm(rel_pos[:2]).item():.3f} 米\n")

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
        obs = _flatten_obs(obs)

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
    import os
    output_dir = os.path.dirname(args_cli.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

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
