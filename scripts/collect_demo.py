"""Stage 0: Demonstration Data Collection for BC Pretraining

Collects (observation, action) pairs from successful scripted grasps only.
The teacher implements:
- Navigation: P-controller to approach target
- Arm control: Follow scripted grasp posture
- Gripper: Close when target is within reach

Output: HDF5 file (datasets/eggtart_demo.hdf5) with observations and actions
"""

from __future__ import annotations

import argparse

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
    3. CLOSING: 底盘停止，夹爪闭合
    4. HOLDING: 保持夹持
    5. RETRACT: 收回后保持，验证物体稳定抬升
    6. DONE: 本次尝试结束，成功时继续夹持直到环境重置
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
        align_threshold: float = 0.1,            # 对齐完成的角度阈值（弧度，约5.7度）
        approach_threshold: float = 0.45,        # 开始伸手的距离阈值（米）
        grasp_joint_pos: list[float] | None = None,
        target_spawn_distance: float = 0.7,      # 目标生成距离（米）
        target_spawn_lateral: float = 0.0,       # 目标侧向偏移（米，正值为右侧）
        target_spawn_height: float = 0.10,       # 目标生成高度（米）
        target_approach_speed: float = 0.0,      # 默认静止；环境内的辅助移动也须关闭
        target_tracking_distance: float = 0.20,  # 目标开始跟踪的距离（米）
        target_front_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),  # 前方方向向量
        close_time: float = 2.00,                # 闭合超时（秒），碰撞阻塞不能靠延长等待解决
        hold_time: float = 1.80,                 # 夹爪保持时间（秒）
        retract_time: float = 1.00,              # 机械臂收回时间（秒）
        gripper_closed_threshold: float = 0.30,  # 闭合角参考阈值，爪尖接触另用持续停转判定
        lift_height: float = 0.15,               # 抬起判定（米）：目标抬升超过此值才算成功
        grasp_horiz_tol: float = 0.008,           # 抓取判定水平容差（米）——配合 REACH 底盘微调
        grasp_z_tol: float = 0.02,              # 抓取判定 Z 容差（米）
        reach_timeout: float = 4.0,              # REACH 超时（秒）——臂伸不到位/判定不过则放弃，防爪全程张开干等
        lift_hold_time: float = 0.5,            # 收回后继续夹持并验证的时间（秒）
    ):
        """初始化脚本化教师

        Args:
            env: Isaac Lab 环境（unwrapped）
            grasp_offset: 抓取点相对 EE body 原点的偏移（米）
            direction_offset: 方向参考点相对 EE body 原点的偏移（米）
            nominal_joint_pos: 五个机械臂关节的 nominal 姿态（弧度）
            align_threshold: 对齐完成的角度阈值（弧度）- 当朝向误差小于此值时，从 ALIGN 切换到 NAVIGATE
            approach_threshold: 开始伸手的距离阈值（米）- 当 link1 距目标小于此值时，从 NAVIGATE 切换到 REACH
            grasp_joint_pos: 五个机械臂关节的 grasp 姿态（弧度），默认使用 nominal
            target_spawn_distance: 目标生成距离（米）- 目标在机器人正前方此距离处生成
            target_spawn_lateral: 目标侧向偏移（米）- 正值为工作面右侧，负值为左侧
            target_spawn_height: 目标生成高度（米）- 相对地面的高度
            target_approach_speed: 目标靠近速度（米/秒）- 目标主动靠近夹爪的速度
            target_tracking_distance: 目标开始跟踪的距离（米）- 抓取点进入此距离内时目标开始主动靠近
            target_front_axis: 前方方向向量 - 机械臂伸出的方向，默认 (0, -1, 0) 表示 -Y 方向
            close_time: 夹爪闭合时间（秒）——超时未夹紧则放弃本 episode
            hold_time: 夹爪保持时间（秒）
            retract_time: 机械臂收回时间（秒）
            gripper_closed_threshold: 夹爪夹紧判定阈值（rad）——接触确认，真夹紧由抬起验证兜底
            lift_height: 抬起判定（米）——目标抬升超过此值才算成功
            grasp_horiz_tol: 抓取判定水平容差（米）
            grasp_z_tol: 抓取判定 Z 容差（米）
            lift_hold_time: 收回后须连续满足抬升高度的保持时间（秒）
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.grasp_offset = torch.tensor(grasp_offset, device=self.device, dtype=torch.float32)
        # direction_offset 仅保留用于兼容（旧线段判定已废弃），不再参与抓取判定
        self.direction_offset = torch.tensor(direction_offset, device=self.device, dtype=torch.float32)
        self.nominal_joint_pos = torch.tensor(nominal_joint_pos, device=self.device, dtype=torch.float32)
        self.align_threshold = align_threshold
        self.approach_threshold = approach_threshold
        self.target_spawn_distance = target_spawn_distance
        self.target_spawn_lateral = target_spawn_lateral
        self.target_spawn_height = target_spawn_height
        self.target_approach_speed = target_approach_speed
        self.target_tracking_distance = target_tracking_distance
        self.gripper_closed_threshold = gripper_closed_threshold
        self.lift_height = lift_height
        self.grasp_horiz_tol = grasp_horiz_tol
        self.grasp_z_tol = grasp_z_tol
        self.reach_timeout = reach_timeout
        self.dt = float(env.step_dt)
        front_axis = torch.tensor(target_front_axis, device=self.device, dtype=torch.float32)
        self.target_front_axis = front_axis / torch.linalg.vector_norm(front_axis).clamp_min(1e-6)

        # The action term is JointPositionAction(scale=0.5, use_default_offset=True), so
        # zero means nominal and this is the normalized action for the grasp posture.
        if grasp_joint_pos is None:
            grasp_joint_pos = nominal_joint_pos
        self.grasp_joint_pos = torch.tensor(grasp_joint_pos, device=self.device, dtype=torch.float32)
        self.grasp_arm_action = (self.grasp_joint_pos - self.nominal_joint_pos) / 0.5

        dt = self.dt
        self.close_steps = max(1, int(round(close_time / dt)))
        self.hold_steps = max(1, int(round(hold_time / dt)))
        self.retract_steps = max(1, int(round(retract_time / dt)))
        self.reach_steps = max(1, int(round(reach_timeout / dt)))
        self.lift_hold_steps = max(1, int(round(lift_hold_time / dt)))

        # State tracking
        self.state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_in_state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frozen_arm = torch.zeros(self.num_envs, 5, device=self.device)
        self.reach_integral = torch.zeros(self.num_envs, 2, device=self.device)
        self.arm_settled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.close_start_q = torch.zeros(self.num_envs, device=self.device)
        self.contact_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 每 env 闭爪前的目标高度（用于抬起验证）与成功标记
        self.init_target_z = torch.zeros(self.num_envs, device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lift_stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._debug_counter = 0

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

        # 夹爪关节（力控，用于"夹住"验证）
        self.gripper_cfg = SceneEntityCfg("robot", joint_names="end_effector_joint")
        self.gripper_cfg.resolve(self.env.scene)
        self.gripper_joint_idx = self.gripper_cfg.joint_ids[0]

        self.arm_cfg = SceneEntityCfg("robot", joint_names=[f"link_00{i}_joint" for i in range(1, 6)], preserve_order=True)
        self.arm_cfg.resolve(self.env.scene)

        self.target_cfg = SceneEntityCfg("target")
        self.target_cfg.resolve(self.env.scene)

    def _grasp_point_w(self) -> torch.Tensor:
        """Compute grasp point in world frame"""
        robot = self.env.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, self.ee_cfg.body_ids[0]]
        ee_quat = robot.data.body_quat_w[:, self.ee_cfg.body_ids[0]]
        offset = self.grasp_offset.unsqueeze(0).expand(self.num_envs, -1)
        return ee_pos + quat_apply(ee_quat, offset)

    def _debug_grasp(self) -> None:
        """抓取阶段调试：打印前 3 个已进入 REACH 的 env 的抓取几何与夹爪状态

        每 50 步打印一次，用于定位"夹爪没闭合"卡在哪：
        - state 停在 REACH = grasp_ready 判定没过（看 horiz/zdiff 与容差）
        - CLOSING 中 q_grip 不动：检查接触阻塞、夹持力矩和动作传递
        - q_grip 稳定停住不等于成功，必须继续查看抬升阶段的 tgt_z
        """
        self._debug_counter += 1
        if self._debug_counter % 50 != 0:
            return
        active = (self.state >= self.REACH) & (self.state < self.DONE)
        if not active.any():
            return
        ids = torch.nonzero(active, as_tuple=False)[:3].flatten()
        grasp_pos = self._grasp_point_w()[ids]
        target = self.env.scene["target"]
        tpos = target.data.root_pos_w[ids]
        robot = self.env.scene["robot"]
        q = robot.data.joint_pos[ids, self.gripper_joint_idx]
        qd = robot.data.joint_vel[ids, self.gripper_joint_idx]
        arm_error = (robot.data.joint_pos[ids][:, self.arm_cfg.joint_ids] - self.grasp_joint_pos).abs().amax(dim=1)
        grasp_b = self._grasp_point_in_base_frame()[ids]
        target_b = self._target_in_base_frame()[ids]
        error_xy = target_b[:, :2] - grasp_b[:, :2]
        print(f"[DEBUG grasp] step={self.env.common_step_counter} active_envs={active.sum().item()}")
        for i, idx in enumerate(ids):
            d = tpos[i] - grasp_pos[i]
            print(
                f"  env{idx.item()}: state={self.state[idx].item()} "
                f"state_step={self.step_in_state[idx].item()} "
                f"q_grip={q[i].item():+.3f} "
                f"qd_grip={qd[i].item():+.3f} arm_err={arm_error[i].item():.3f} "
                f"horiz={torch.norm(d[:2]).item():.3f}(tol {self.grasp_horiz_tol}) "
                f"err_xy=({error_xy[i, 0].item():+.3f},{error_xy[i, 1].item():+.3f}) "
                f"zdiff={abs(d[2].item()):.3f}(tol {self.grasp_z_tol}) "
                f"tgt_z={tpos[i, 2].item():.3f} grasp_z={grasp_pos[i, 2].item():.3f}"
            )

    def _grasp_ready(self) -> torch.Tensor:
        """抓取判定：抓取点与目标在水平面足够近 + Z 容差内

        水平误差用于底盘闭环控制；高度容差必须与实际夹持区域一致。

        Returns:
            bool tensor [num_envs]
        """
        target = self.env.scene["target"]
        grasp_pos = self._grasp_point_w()
        diff = target.data.root_pos_w - grasp_pos
        horiz = torch.linalg.vector_norm(diff[:, :2], dim=1)
        zdiff = torch.abs(diff[:, 2])
        return (horiz <= self.grasp_horiz_tol) & (zdiff <= self.grasp_z_tol)

    def _gripper_closed(self) -> torch.Tensor:
        """关节是否越过闭合角参考阈值；此判据也可能表示空夹。"""
        robot = self.env.scene["robot"]
        q = robot.data.joint_pos[:, self.gripper_joint_idx]
        return q < self.gripper_closed_threshold

    def _arm_ready(self) -> torch.Tensor:
        robot = self.env.scene["robot"]
        arm_error = (robot.data.joint_pos[:, self.arm_cfg.joint_ids] - self.grasp_joint_pos).abs().amax(dim=1)
        arm_speed = robot.data.joint_vel[:, self.arm_cfg.joint_ids].abs().amax(dim=1)
        return (arm_error < 0.10) & (arm_speed < 0.15)

    def _target_lifted(self) -> torch.Tensor:
        """相对闭爪前的目标高度验证抬升，排除生成后自由下落的影响。"""
        target = self.env.scene["target"]
        z = target.data.root_pos_w[:, 2]
        return z > self.init_target_z + self.lift_height

    def _place_target_in_front(self, env_ids: torch.Tensor | None = None) -> None:
        """将目标放置在机械臂基座（link_001）正前方的固定位置

        使用 link_001（机械臂基座）的位置和朝向作为参考坐标系，
        目标沿 link_001 局部系的 target_front_axis 方向（默认 -Y）生成。
        """
        robot = self.env.scene["robot"]
        target = self.env.scene["target"]
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        # 用 link_001（机械臂基座）作参考系，而不是 base_link root
        link1_pos = robot.data.body_pos_w[env_ids, self.link1_cfg.body_ids[0]]
        link1_quat = robot.data.body_quat_w[env_ids, self.link1_cfg.body_ids[0]]

        # 在 link_001 局部坐标系中构造目标位置
        # target_front_axis 定义了工作面方向（对 Eggtart 是 -Y 方向）
        local_pos = self.target_front_axis.unsqueeze(0).expand(env_ids.numel(), -1).clone()
        local_pos = local_pos * self.target_spawn_distance  # 前方距离

        # 侧向偏移沿局部 X 轴（垂直于前方方向）
        # 正值 = 面向工作面时的右侧，负值 = 左侧
        local_pos[:, 0] += self.target_spawn_lateral
        local_pos[:, 2] = 0.0  # 高度在下面用世界系直接指定

        # 转换到世界坐标系（XY 用 link_001 系旋转，Z 用世界系地面高度）
        root_state = target.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = link1_pos + quat_apply(link1_quat, local_pos)
        # 生成高度使用世界系；目标随后自由落到地面。
        root_state[:, 2] = self.target_spawn_height
        root_state[:, 7:13] = 0.0  # 速度清零
        target.write_root_state_to_sim(root_state, env_ids=env_ids)
        # 记录目标初始高度（用于抬起验证）
        self.init_target_z[env_ids] = target.data.root_pos_w[env_ids, 2]

    def _track_target(self) -> None:
        """让目标主动靠近末端执行器（使用 mdp.target 中的实现）

        在 CLOSING 之前的状态，如果抓取点距离目标小于 target_tracking_distance，
        目标会以 target_approach_speed 的速度缓慢移动向 direction_offset 点。
        这可以帮助策略更容易完成"对准-闭合"动作。

        注意：ALIGN 状态不启用目标跟踪，让机器人先完成对准。
        """
        # 采集时目标必须静止：tracking 关闭（speed<=0 即不调用）。
        # 若目标一边被机器人追一边自己动，底盘必然绕圈。
        if self.target_approach_speed <= 0.0:
            return
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

    def _grasp_point_in_base_frame(self) -> torch.Tensor:
        """获取抓取点在底盘坐标系下的位置。

        REACH 阶段需要修正的是“抓取点到目标”的误差，而不是 link_001
        到目标的距离。后者依赖一个固定的机械臂前伸量，实际关节尚未完全
        到位或存在底盘滑行时会产生几厘米的系统误差。
        """
        robot = self.env.scene["robot"]
        base_pos_w = robot.data.root_pos_w
        base_quat_inv = quat_conjugate(robot.data.root_quat_w)
        grasp_pos_w = self._grasp_point_w()
        return quat_apply(base_quat_inv, grasp_pos_w - base_pos_w)

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

        # 当前观测来自上一次物理步，先据此推进阶段，再生成新阶段动作。
        self._advance_states()

        # 更新状态机（使用抓取判定准则）
        # 导航距离用水平面投影，避免臂基座的高度影响伸手时机。
        link1_pos_w = robot.data.body_pos_w[:, self.link1_cfg.body_ids[0]]
        target_diff = link1_pos_w - target.data.root_pos_w
        dist_to_target = torch.linalg.vector_norm(target_diff[:, :2], dim=1)
        heading_error = self._compute_desired_heading()
        grasp_ready = self._grasp_ready()
        self._update_states(dist_to_target, heading_error, grasp_ready)
        self._track_target()
        self._debug_grasp()

        # === 底盘速度控制（indices 0:3：vx, vy, wz）===
        target_b = self._target_in_base_frame()
        heading_error = self._compute_desired_heading()

        # P 控制器增益
        kp_lin = 1.0   # 线速度比例增益
        kp_ang = 1.0   # 角速度比例增益（比原 2.0 更小，防止过冲振荡）
        align_deadband = 0.03  # 朝向死区（弧度）：小于此角度不旋转，防振荡

        # ALIGN 状态：只旋转，不平移（带死区）
        aligning = self.state == self.ALIGN
        if aligning.any():
            err = heading_error[aligning]
            wz = torch.where(
                torch.abs(err) < align_deadband,
                torch.zeros_like(err),
                torch.clamp(kp_ang * err, -1.0, 1.0),  # 角速度限制 ±1.0 rad/s
            )
            actions[aligning, 0] = 0.0  # 不前后移动
            actions[aligning, 1] = 0.0  # 不左右移动
            actions[aligning, 2] = wz

        # NAVIGATE 状态：直线靠近，轻微修正朝向
        navigating = self.state == self.NAVIGATE
        if navigating.any():
            # 目标在底盘局部坐标系下的水平距离和方向角
            # base_link 的原点偏离车身约 0.435 m。平移方向必须从臂基座
            # 指向目标，不能把 target_b 当作机器人中心到目标的向量。
            direction_b = quat_apply(
                quat_conjugate(robot.data.root_quat_w), -target_diff
            )[navigating, :2]
            r = torch.norm(direction_b, dim=1)
            angle = torch.atan2(direction_b[:, 1], direction_b[:, 0])

            # 平动速度正比于距离，0.30 m 内进一步减速。
            brake_dist = 0.30
            v = kp_lin * r
            v = v * torch.clamp(r / brake_dist, 0.0, 1.0)
            v = torch.clamp(v, -0.6, 0.6)

            # 转向优先（防画圈）：用朝向误差而非"目标在底盘系的角度"。
            # 注意：目标在正前方（-Y）时 atan2 给 ±π/2，若按角度砍速会把
            # 速度永久砍到 0.15（这就是"慢吞吞挪动"的根因）。
            # ALIGN 已保证 |heading_error|<0.1 才进 NAVIGATE，这里 factor≈1。
            angle_factor = torch.clamp(1.0 - torch.abs(heading_error[navigating]) / 0.5, 0.6, 1.0)
            v = v * angle_factor

            # 速度方向始终指向目标（全向底盘可任意方向平移）
            vx = v * torch.cos(angle)
            vy = v * torch.sin(angle)
            # 角速度：仅修正 link1 朝向，带死区 + 低增益
            err = heading_error[navigating]
            wz = torch.where(
                torch.abs(err) < align_deadband,
                torch.zeros_like(err),
                torch.clamp(0.6 * err, -0.6, 0.6),
            )

            actions[navigating, 0] = vx
            actions[navigating, 1] = vy
            actions[navigating, 2] = wz

        # REACH 状态：低速微调对准（补底盘惯性滑行留下的最后几厘米），对准后停止
        reaching = self.state == self.REACH
        if reaching.any():
            # 先在目标前方降下机械臂，等臂停稳再平移，避免下降时扫走目标。
            self.arm_settled |= reaching & self._arm_ready()
            still_adjust = reaching & ~grasp_ready & self.arm_settled
            if still_adjust.any():
                # 直接用实际抓取点的平面误差闭环修正。旧实现用
                # ``dist(link1,target) - 0.231`` 估算误差；机械臂在运动中
                # 的前伸量并不固定，因而会稳定地留下 5~7 cm 残差。
                grasp_b = self._grasp_point_in_base_frame()
                error_xy = target_b[still_adjust, :2] - grasp_b[still_adjust, :2]
                # 每轴独立限速；动作再经 HolonomicBaseAction 缩放成 m/s。
                # 轮地摩擦会让纯 P 控制在厘米级误差处停止；积分消除该残差。
                self.reach_integral[still_adjust] = torch.where(
                    self.reach_integral[still_adjust] * error_xy < 0,
                    0.0,
                    self.reach_integral[still_adjust],
                )
                self.reach_integral[still_adjust] = (
                    self.reach_integral[still_adjust] + error_xy * self.dt
                ).clamp(-0.10, 0.10)
                correction = torch.clamp(4.0 * error_xy + 2.0 * self.reach_integral[still_adjust], -0.25, 0.25)
                actions[still_adjust, 0:2] = correction
            # 不对准的 env 不转向（已有 ALIGN 保证朝向）；对准的保持静止（初始为 0）

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
        # 力控的零动作不保持开度，恒正力矩则会顶到 +1.57 的硬限位。
        # 用角度反馈维持适度张开，下降过程中也持续保持开度。
        opening = self.state < self.CLOSING
        q = robot.data.joint_pos[:, self.gripper_joint_idx]
        # 关节已有隐式阻尼；30 Hz 下再用高增益显式 PD 会造成开度振荡。
        actions[opening, 8] = (0.8 - q[opening]).clamp(-0.3, 0.3)
        # 保持 0.3 Nm 夹持力矩，成功后的 DONE 也持续施力，直到环境重置。
        # 力控下 action=0 会卸力，并不表示保持当前开度。
        grasping = (self.state == self.CLOSING) | (self.state == self.HOLDING) | (self.state == self.RETRACT)
        grasping |= (self.state == self.DONE) & self.success
        actions[grasping, 8] = -0.3

        self.step_in_state += 1

        return actions

    def _update_states(self, dist_to_target: torch.Tensor, heading_error: torch.Tensor, grasp_ready: torch.Tensor):
        """根据距离、朝向和对齐状态更新状态机

        状态只能单向递进：ALIGN → NAVIGATE → REACH → CLOSING → HOLDING → RETRACT → DONE

        Args:
            dist_to_target: link1 到目标的水平距离
            heading_error: 朝向误差（弧度）
            grasp_ready: 抓取点是否已对准目标（水平+Z 容差内）
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

        # REACH -> CLOSING: 抓取点对准目标（水平+Z 容差内）时开始闭合夹爪
        robot = self.env.scene["robot"]
        reach_to_close = (self.state == self.REACH) & grasp_ready & self._arm_ready()
        if reach_to_close.any():
            self.frozen_arm[reach_to_close] = self.grasp_arm_action
            self.init_target_z[reach_to_close] = self.env.scene["target"].data.root_pos_w[reach_to_close, 2]
            self.close_start_q[reach_to_close] = robot.data.joint_pos[reach_to_close, self.gripper_joint_idx]
            self.contact_steps[reach_to_close] = 0
            self.state[reach_to_close] = self.CLOSING
            self.step_in_state[reach_to_close] = 0

        # REACH 超时：臂伸不到位/判定一直不过 -> 放弃（防爪全程张开干等到 300 步超时）
        reach_timeout_mask = (self.state == self.REACH) & (self.step_in_state >= self.reach_steps)
        if reach_timeout_mask.any():
            self.state[reach_timeout_mask] = self.DONE
            self.step_in_state[reach_timeout_mask] = 0

    def _advance_states(self) -> None:
        """根据时间推进状态机（验证驱动）

        状态转换（单向递进）：
        - CLOSING -> HOLDING: 闭合到位或出现持续停转后保持；超时 -> DONE
        - HOLDING -> RETRACT: 保持时间到，开始试抬
        - RETRACT -> DONE: 收回后持续夹持并满足高度门槛 -> success，否则失败

        抬起验证放在 RETRACT 阶段：臂从 grasp 姿态收回 nominal 姿态，
        只有真夹住物体才会把它带离地面。
        空闭（爪合拢但没夹住）的 env 收回时物体留在地面 -> 失败。
        """
        gripper_closed = self._gripper_closed()
        lifted = self._target_lifted()

        # 爪尖夹持与夹爪根部夹持的停止角不同，不能要求所有有效抓取都
        # 小于 0.33 rad。闭合已有行程、随后持续停住，可作为试抬的候选；
        # 这不是成功判定，撞地或空夹仍会在抬升验证中失败。
        robot = self.env.scene["robot"]
        q = robot.data.joint_pos[:, self.gripper_joint_idx]
        qd = robot.data.joint_vel[:, self.gripper_joint_idx]
        stalled = ((self.close_start_q - q) > 0.04) & (qd.abs() < 0.05)
        closing = self.state == self.CLOSING
        self.contact_steps = torch.where(closing & stalled, self.contact_steps + 1, 0)
        contact_candidate = self.contact_steps >= max(1, round(0.2 / self.dt))
        # 软接触可能持续缓慢压合，不能一直等待速度恰好低于阈值。
        contact_candidate |= ((self.step_in_state * self.dt) >= 0.8) & ((self.close_start_q - q) > 0.04)
        close_ready = gripper_closed | contact_candidate

        # 收回后再观察一段时间，短暂越过高度门槛又掉落不算成功。
        verifying = (self.state == self.RETRACT) & (self.step_in_state > self.retract_steps)
        self.lift_stable_steps = torch.where(verifying & lifted, self.lift_stable_steps + 1, 0)

        # 成功路径
        to_hold = closing & close_ready
        to_retract = (self.state == self.HOLDING) & (self.step_in_state >= self.hold_steps)
        to_done = (self.state == self.RETRACT) & (
            self.step_in_state >= self.retract_steps + self.lift_hold_steps
        )
        # 超时失败路径（未夹住）
        close_timeout = closing & (self.step_in_state >= self.close_steps) & ~close_ready

        for mask, next_state in (
            (to_hold, self.HOLDING),
            (to_retract, self.RETRACT),
            (close_timeout, self.DONE),
            (to_done, self.DONE),
        ):
            if mask.any():
                self.state[mask] = next_state
                self.step_in_state[mask] = 0

        # 成功标记：收回后连续满足抬升高度，DONE 动作继续夹持。
        if to_done.any():
            self.success[to_done] = self.lift_stable_steps[to_done] >= self.lift_hold_steps

    def reset(self, env_ids: torch.Tensor | None = None):
        """重置指定环境的状态机

        同时将目标放置在机械臂正前方的固定位置。

        Args:
            env_ids: 要重置的环境索引，None 表示重置所有环境
        """
        if env_ids is None:
            self.state.zero_()
            self.step_in_state.zero_()
            self.frozen_arm.zero_()
            self.reach_integral.zero_()
            self.arm_settled.zero_()
            self.close_start_q.zero_()
            self.contact_steps.zero_()
            self.success.zero_()
            self.lift_stable_steps.zero_()
            self._place_target_in_front()
        else:
            self.state[env_ids] = 0
            self.step_in_state[env_ids] = 0
            self.frozen_arm[env_ids] = 0.0
            self.reach_integral[env_ids] = 0.0
            self.arm_settled[env_ids] = False
            self.close_start_q[env_ids] = 0.0
            self.contact_steps[env_ids] = 0
            self.success[env_ids] = False
            self.lift_stable_steps[env_ids] = 0
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
    # Static 配置仍带有阶段 1 的目标跟随事件；仅关闭 teacher 跟随并不够。
    env_cfg.events.target_approach_stage1 = None
    env_cfg.events.randomize_target_velocity = None
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Unwrap to get base environment
    if hasattr(env, 'unwrapped'):
        base_env = env.unwrapped
    else:
        base_env = env

    # Get grasp offset from assets
    from eggtart_grasp.assets.eggtart import (
        EGGTART_ARM_JOINT_NAMES,
        EGGTART_BASE_FORWARD_AXIS,
        EGGTART_EE_GRASP_DERECT_OFFSET,
        EGGTART_GRASP_JOINT_POS,
        EGGTART_NOMINAL_JOINT_POS,
    )

    # EGGTART_GRASP_JOINT_POS 是按关节名索引的 dict，ScriptedTeacher 需要 5 个臂关节
    # 的有序 list（顺序与 EGGTART_ARM_JOINT_NAMES 一致）
    grasp_joint_pos_list = [EGGTART_GRASP_JOINT_POS[name] for name in EGGTART_ARM_JOINT_NAMES]
    nominal_joint_pos_list = [EGGTART_NOMINAL_JOINT_POS[name] for name in EGGTART_ARM_JOINT_NAMES]
    # 地面上的 30 mm 立方体需在爪尖夹持。原姿态会让爪尖压地，且原
    # grasp_offset 位于爪内较高处。这里只校准采集教师，不改训练资产配置。
    grasp_joint_pos_list[1] = -2.74
    collection_grasp_offset = (0.010, -0.032, -0.083)

    # 创建脚本化教师
    print(f"\n初始化 ScriptedTeacher 参数:")
    print(f"  - 前方方向向量: {EGGTART_BASE_FORWARD_AXIS}")
    print(f"  - 目标生成距离: 0.5 米（沿 link_001 机械臂基座正前方）")
    print(f"  - 目标侧向偏移: 0.0 米（正值=右侧，负值=左侧）")
    print(f"  - 目标生成高度: 0.10 米")
    print(f"  - 目标主动靠近: 已关闭（采集时目标静止，防止底盘追目标绕圈）")
    print(f"  - 抓取点偏移: {collection_grasp_offset}（爪尖夹持区域）")
    print(f"  - 方向点偏移: {EGGTART_EE_GRASP_DERECT_OFFSET}\n")

    teacher = ScriptedTeacher(
        base_env,
        grasp_offset=collection_grasp_offset,
        direction_offset=EGGTART_EE_GRASP_DERECT_OFFSET,
        nominal_joint_pos=nominal_joint_pos_list,
        grasp_joint_pos=grasp_joint_pos_list,
        align_threshold=0.1,           # 对齐完成的角度阈值（弧度，约5.7度）
        approach_threshold=0.35,       # 在目标前留出下降空间，臂停稳后再微调靠近
        target_spawn_distance=0.5,     # 目标生成距离（米，沿 link_001 正前方）
        target_spawn_lateral=0.0,      # 目标侧向偏移（米，正值为右侧）
        target_spawn_height=0.10,      # 目标生成高度（米，**世界系地面高度**，与 env init 一致）
        target_approach_speed=0.0,
        target_tracking_distance=0.20, # 目标开始跟踪的距离（米）
        target_front_axis=EGGTART_BASE_FORWARD_AXIS,  # 前方方向向量
        close_time=2.00,
        hold_time=1.80,                # 夹爪保持时间（秒）
        retract_time=1.50,             # 平缓抬起，避免把爪尖夹持的物体甩出
    )
    print(f"  - 成功条件: 相对闭爪前抬升 {teacher.lift_height:.2f} m，收回后保持 "
          f"{teacher.lift_hold_steps * teacher.dt:.2f} s；成功后持续夹持至重置")

    # Storage
    obs_buffer = []
    action_buffer = []
    episode_lengths = []
    success_count = 0
    total_episodes = 0
    reach_timeout_count = 0
    close_timeout_count = 0
    # 只保留成功轨迹：每个 env 维护当前 episode 的临时轨迹，
    # 结束时 teacher.success 为 True 才拼入全局缓冲（失败轨迹丢弃，防止污染 BC）。
    ep_obs: list[list[np.ndarray]] = [[] for _ in range(base_env.num_envs)]
    ep_act: list[list[np.ndarray]] = [[] for _ in range(base_env.num_envs)]

    def _flatten_obs(obs):
        """把 gymnasium 返回的 obs（可能是 dict，键为观测组名）拼成 [num_envs, obs_dim] tensor。

        顺序与 RslRlVecEnvWrapper（PPO 训练用）一致：按 dict 键序拼接；
        观测配置里只有一个 'policy' 组时，即该组本身。
        """
        if isinstance(obs, dict):
            return torch.cat([obs[k] for k in obs.keys()], dim=-1)
        return obs

    # Reset
    env.reset()
    teacher.reset()
    # teacher.reset 重新放置了目标，须刷新观测，避免首个样本仍指向旧目标。
    obs = _flatten_obs(base_env.observation_manager.compute(update_history=False))

    # 打印第一个环境的初始位置信息（用于调试坐标系）
    robot = base_env.scene["robot"]
    target = base_env.scene["target"]
    print(f"\n[调试] 第一个环境的初始位置:")
    print(f"  机器人位置: {robot.data.root_pos_w[0].cpu().numpy()}")
    print(f"  目标位置: {target.data.root_pos_w[0].cpu().numpy()}")
    rel_pos = target.data.root_pos_w[0] - robot.data.root_pos_w[0]
    print(f"  相对位置 (目标-机器人): {rel_pos.cpu().numpy()}")
    print(f"  水平距离: {torch.norm(rel_pos[:2]).item():.3f} 米\n")

    # 记录每条成功轨迹的初始状态（供 replay_demo.py 精确回放）：
    # 机器人 root 位姿/关节角 + 目标 root 位姿，轨迹保存时一并写入 hdf5
    init_robot_pos = torch.zeros(base_env.num_envs, 3, device=base_env.device)
    init_robot_quat = torch.zeros(base_env.num_envs, 4, device=base_env.device)
    init_robot_joint = torch.zeros(base_env.num_envs, robot.num_joints, device=base_env.device)
    init_target_pos = torch.zeros(base_env.num_envs, 3, device=base_env.device)
    init_target_quat = torch.zeros(base_env.num_envs, 4, device=base_env.device)
    init_robot_pos_list, init_robot_quat_list = [], []
    init_robot_joint_list, init_target_pos_list, init_target_quat_list = [], [], []

    def record_init_states(env_ids: torch.Tensor) -> None:
        """记录指定 env 的当前状态作为其新 episode 的初始状态"""
        idx = env_ids
        init_robot_pos[idx] = robot.data.root_pos_w[idx].clone()
        init_robot_quat[idx] = robot.data.root_quat_w[idx].clone()
        init_robot_joint[idx] = robot.data.joint_pos[idx].clone()
        init_target_pos[idx] = target.data.root_pos_w[idx].clone()
        init_target_quat[idx] = target.data.root_quat_w[idx].clone()

    record_init_states(torch.arange(base_env.num_envs, device=base_env.device))

    print("Starting data collection...")

    for step in range(args_cli.max_steps):
        # Get scripted actions
        previous_state = teacher.state.clone()
        actions = teacher.compute_actions()
        finished = (previous_state != teacher.DONE) & (teacher.state == teacher.DONE)
        # 在尝试结束时统计，避免已抬升却要等到环境第 300 步重置才显示成功。
        success_count += teacher.success[finished].sum().item()
        total_episodes += finished.sum().item()
        reach_timeout_count += (finished & (previous_state == teacher.REACH)).sum().item()
        close_timeout_count += (finished & (previous_state == teacher.CLOSING)).sum().item()

        # 当前物理状态确认上一步轨迹的结果；仅保存已执行的动作。
        # DONE 等待段不再采样，失败和未完成的轨迹也不进入数据集。
        for i in torch.where(finished)[0].tolist():
            if teacher.success[i].item() and ep_obs[i]:
                obs_buffer.append(np.stack(ep_obs[i]))
                action_buffer.append(np.stack(ep_act[i]))
                episode_lengths.append(len(ep_obs[i]))
                # 初始状态与轨迹一一对应（顺序与 episode_lengths 一致）
                init_robot_pos_list.append(init_robot_pos[i].cpu().numpy())
                init_robot_quat_list.append(init_robot_quat[i].cpu().numpy())
                init_robot_joint_list.append(init_robot_joint[i].cpu().numpy())
                init_target_pos_list.append(init_target_pos[i].cpu().numpy())
                init_target_quat_list.append(init_target_quat[i].cpu().numpy())
            ep_obs[i] = []
            ep_act[i] = []

        # Add noise for exploration
        if args_cli.noise_scale > 0:
            noise = torch.randn_like(actions) * args_cli.noise_scale
            # 精确对准、闭爪和抬升时保持教师的闭环动作；否则已停止的底盘
            # 也会被噪声推动，冻结的机械臂姿态也会不停改变。
            noise[teacher.state >= teacher.REACH] = 0.0
            noise[:, 3:] = 0.0
            noisy_actions = actions + noise
        else:
            noisy_actions = actions

        # Store data：先入各 env 的当前 episode 暂存
        obs_np = obs.cpu().numpy()
        act_np = noisy_actions.cpu().numpy()
        for i in torch.where(teacher.state != teacher.DONE)[0].tolist():
            # copy 避免一个 env 的切片一直持有整批环境的 NumPy 数组。
            ep_obs[i].append(obs_np[i].copy())
            ep_act[i].append(act_np[i].copy())

        # Step environment
        obs, rewards, terminated, truncated, infos = env.step(noisy_actions)
        obs = _flatten_obs(obs)

        # Track resets
        reset_mask = terminated | truncated
        if reset_mask.any():
            reset_ids = torch.where(reset_mask)[0]
            # 尚未完成便被环境截断的尝试也计为失败，已结束的不要重复统计。
            total_episodes += (teacher.state[reset_ids] != teacher.DONE).sum().item()
            # 未完成被截断：丢弃暂存轨迹
            for i in reset_ids.tolist():
                ep_obs[i] = []
                ep_act[i] = []
            teacher.reset(reset_ids)
            refreshed_obs = _flatten_obs(base_env.observation_manager.compute(update_history=False))
            obs[reset_ids] = refreshed_obs[reset_ids]
            # 新 episode 的初始状态以 reset 后的实际状态为准
            record_init_states(reset_ids)

        # Progress
        if (step + 1) % 1000 == 0:
            collected = sum(episode_lengths)
            success_rate = (success_count / max(total_episodes, 1)) * 100
            print(f"Step {step+1}/{args_cli.max_steps} | Kept: {collected:,} successful samples | "
                  f"Episodes: {total_episodes} | Success rate: {success_rate:.1f}%")

    # Concatenate buffers
    print("\nProcessing collected data...")
    obs_array = np.concatenate(obs_buffer, axis=0) if obs_buffer else np.empty((0, obs.shape[-1]), dtype=np.float32)
    action_array = np.concatenate(action_buffer, axis=0) if action_buffer else np.empty((0, 9), dtype=np.float32)
    if not obs_buffer:
        print("No successful trajectories completed; saving an empty dataset (failed/incomplete attempts excluded).")

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
        f.create_dataset("episode_lengths", data=np.asarray(episode_lengths, dtype=np.int64))
        # 每条成功轨迹的初始状态（供 replay_demo.py 精确回放）
        if init_robot_pos_list:
            f.create_dataset("init_robot_root_pos", data=np.stack(init_robot_pos_list), compression="gzip")
            f.create_dataset("init_robot_root_quat", data=np.stack(init_robot_quat_list), compression="gzip")
            f.create_dataset("init_robot_joint_pos", data=np.stack(init_robot_joint_list), compression="gzip")
            f.create_dataset("init_target_root_pos", data=np.stack(init_target_pos_list), compression="gzip")
            f.create_dataset("init_target_root_quat", data=np.stack(init_target_quat_list), compression="gzip")

        # Metadata
        f.attrs["env_name"] = args_cli.task
        f.attrs["num_envs"] = args_cli.num_envs
        f.attrs["num_samples"] = len(obs_array)
        f.attrs["noise_scale"] = args_cli.noise_scale
        f.attrs["timestamp"] = datetime.now().isoformat()
        f.attrs["success_rate"] = success_count / max(total_episodes, 1)
        f.attrs["completed_attempts"] = total_episodes
        f.attrs["successful_attempts"] = success_count
        f.attrs["reach_timeouts"] = reach_timeout_count
        f.attrs["close_timeouts"] = close_timeout_count
        f.attrs["successful_only"] = True
        f.attrs["saved_episodes"] = len(episode_lengths)
        f.attrs["lift_height"] = teacher.lift_height
        f.attrs["lift_hold_time"] = teacher.lift_hold_steps * teacher.dt

    env.close()

    print(f"\n{'='*80}")
    print(f"Collection complete!")
    print(f"Saved {len(obs_array):,} samples to {args_cli.output}")
    print(f"Success rate: {(success_count/max(total_episodes, 1))*100:.1f}%")
    print(f"Attempts: {total_episodes} | Lifted: {success_count} | "
          f"REACH timeouts: {reach_timeout_count} | CLOSING timeouts: {close_timeout_count}")
    print(f"{'='*80}\n")

    # Close simulation
    simulation_app.close()


if __name__ == "__main__":
    main()
