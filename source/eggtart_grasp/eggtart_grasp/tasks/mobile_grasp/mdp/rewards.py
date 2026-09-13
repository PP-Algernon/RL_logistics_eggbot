"""Eggtart 移动抓取任务的分阶段奖励项

阶段设计（除特殊说明外均为稠密奖励）：
  1. ``base_to_target_xy_tanh``  -- 引导底盘在水平面接近目标
  2. ``base_facing_target``      -- 引导底盘正面朝向目标
  3. ``ee_to_target_tanh``       -- 引导末端执行器到达目标（远/中场）
  3'. ``ee_to_target_precision``  -- 近场精修：只在最后几厘米内有梯度，训练后期打开
  4. ``grasp_bonus``             -- 当末端执行器接近目标且夹爪闭合时给予稀疏奖励
  4'. ``grasp_bonus_dwell``      -- 同上，但要求**连续停留**一段时间后闭爪才算（治夹早）
  5. ``retract_bonus``           -- 抓取后奖励机械臂回到初始姿态

惩罚项：
  - ``base_velocity_l2``       -- 罚底盘"到位后还在动"（治绕圈），带到位/对准门控
  - ``gripper_premature_close`` -- 罚"还没到位就闭爪"（治夹早，和 dwell 配合用）

注意：
    当前实现中，"抓取"状态在每个时间步瞬时评估，目标物体并未刚性固连到夹爪。
    未来可升级为带锁定标志的真实拾取-搬运任务（参考 env-cfg 中的 TODO）。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _base_center_w(robot: Articulation, base_cfg: SceneEntityCfg) -> torch.Tensor:
    """底盘几何中心在世界系下的坐标（默认取四个轮子 body 位置的均值）

    **为什么不能直接用 root_pos_w**：
        ``root_pos_w`` 是 base_link 原点，而 base_link 原点由 CAD 导出决定，
        和底盘几何中心可以差很远。本机器人实测（见 URDF / 仿真里打印的 body 坐标）：

            轮心在 base_link 系 = (-0.333, -0.282, 0.042)

        也就是原点在底盘外面 0.435 m 处。如果拿 root_pos_w 当"底盘位置"：

        1. ``base_to_target_xy_tanh`` 会把这个"幽灵点"拉到目标上，真正的车身
           反而停在离目标 0.435 m、方位约 130°（左后方）的地方；
        2. ``base_facing_target`` 算"指向目标的方向"时基点也偏 0.435 m，
           近距离下方位角误差能有几十度，等距离趋近 0 时方向向量退化成噪声。

        两项合起来的结果就是：训练出来车身的左后方对着目标。所以底盘相关的奖励
        必须用几何中心，不能用 root 原点。

    Args:
        robot: 机器人 articulation
        base_cfg: 指向底盘参考 body 的实体配置，须带 ``body_names``（如 ``"wheel_.*"``）。
            必须在 RewTerm 的 params 里显式传入，manager 只解析 params 里的 cfg，
            函数签名里的默认值不会被解析（``body_ids`` 会是 None）。

    Returns:
        shape (num_envs, 3) 的世界系坐标
    """
    return robot.data.body_pos_w[:, base_cfg.body_ids, :].mean(dim=1)

def base_to_target_xy_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    standoff: float = 0.4,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wheel_.*"),
    arm_link1_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """奖励底盘在水平面停到目标旁边合适的距离上

    用 tanh 核把"参考点到目标的水平距离与 standoff 的偏差"映射到 [0, 1]。
    距离等于 standoff 时奖励最高，越偏离越低。

    **参考点选择**：
        - 如果提供 arm_link1_cfg，使用机械臂 link1 的坐标原点作为参考点
        - 否则使用底盘几何中心（四个轮子的均值）作为参考点

    使用 link1 作为参考点的优势：
        - link1 是机械臂的第一个关节，位置更接近机械臂工作空间的起点
        - 相比底盘中心，link1 的位置能更直接地反映机械臂能否够到目标
        - 当底盘朝向改变时，link1 的位置变化更能体现机械臂接近目标的实际效果

    位置基点不能用 root 原点，原因见 :func:`_base_center_w`。

    **关于 standoff**：
        不能奖励"距离趋近 0"——那是让底盘压到目标上面去，机械臂反而没法伸展。
        实测零位姿下末端执行器在轮心前方约 0.19 m、高 0.28 m，机械臂还能往前伸，
        所以参考点停在目标外 0.3~0.4 m 比较合适。默认 0.35 m。

    Args:
        env: 环境实例
        std: 平滑参数，控制奖励曲线的陡峭程度
        standoff: 期望的参考点到目标的水平距离（米）
        robot_cfg: 机器人实体配置
        target_cfg: 目标物体实体配置
        base_cfg: 底盘参考 body 配置，默认四个轮子（用于 arm_link1_cfg=None 时）
        arm_link1_cfg: 机械臂 link1 实体配置，如果提供则使用 link1 作为参考点

    Returns:
        shape (num_envs,) 的奖励张量，范围 [0, 1]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # 选择参考点：link1 或底盘中心
    if arm_link1_cfg is not None:
        # 使用机械臂 link1 的坐标原点
        ref_pos_xy = robot.data.body_pos_w[:, arm_link1_cfg.body_ids[0], :2]
    else:
        # 使用底盘几何中心（轮子均值）
        ref_pos_xy = _base_center_w(robot, base_cfg)[:, :2]

    dist_xy = torch.norm(ref_pos_xy - target.data.root_pos_w[:, :2], dim=1)
    return 1.0 - torch.tanh((dist_xy - standoff).abs() / std)

def base_facing_target(
    env: ManagerBasedRLEnv,
    std: float,
    forward_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wheel_.*"),
) -> torch.Tensor:
    """奖励底盘正面朝向目标

    以底盘"工作面"（机械臂伸出的方向）与目标方位的夹角为误差，用高斯核给奖励。

    **方位角的基点必须是底盘几何中心，不是 root 原点**（原因见 :func:`_base_center_w`）。
    本机器人 root 原点在车身外 0.435 m，用它算方位角会有几十度的系统性偏差，
    且距离趋近 0 时方向向量退化——这是"训练出来左后方对着目标"的直接原因。

    关于 ``forward_axis``：
        base_link 的坐标轴由 CAD 导出决定，不一定是 +X 朝前。本机器人的四个轮子
        旋转轴都沿 ±X（见 URDF 的 ``wheel_00*_joint`` 的 ``axis``），说明滚动前进
        方向是 Y；机械臂基座和末端执行器相对轮心都朝 -Y 伸出，因此工作面是 **-Y**。
        默认值 (0, -1, 0) 就是这个方向。换机器人或重新导出 URDF 后需要复核。

    为什么用夹角而不是余弦：
        直接对余弦做 tanh 会在小 std 下严重饱和（std=0.3 时，偏差 60° 得 0.966，
        完全对齐得 0.998），策略感受不到梯度。改用夹角误差的高斯核，
        误差越小奖励上升越明显。

    Args:
        env: 环境实例
        std: 夹角误差的高斯核宽度（弧度）。0.6 rad ≈ 34°，误差达到 std 时奖励降到 0.37
        forward_axis: 底盘工作面在 base_link 系中的方向，默认 (0, -1, 0)
        robot_cfg: 机器人实体配置
        target_cfg: 目标物体实体配置
        base_cfg: 底盘参考 body 配置，默认四个轮子

    Returns:
        shape (num_envs,) 的奖励张量，范围 (0, 1]
        - 1.0: 工作面正对目标
        - exp(-1)≈0.37: 偏差等于 std
        - 趋近 0: 背对目标
    """
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # 把体系下的工作面方向旋转到世界系
    robot_quat = robot.data.root_quat_w  # (num_envs, 4)
    axis_b = torch.tensor(forward_axis, device=env.device, dtype=robot_quat.dtype)
    axis_b = axis_b.unsqueeze(0).expand(robot_quat.shape[0], -1)
    forward_w = quat_apply(robot_quat, axis_b)  # (num_envs, 3)

    # 投影到水平面（忽略高度差和俯仰）
    fwd_xy = forward_w[:, :2]
    fwd_xy = fwd_xy / (torch.norm(fwd_xy, dim=1, keepdim=True) + 1e-6)

    # 方位角从底盘几何中心量起（不是 root 原点）
    base_center_w = _base_center_w(robot, base_cfg)
    to_target_xy = (target.data.root_pos_w - base_center_w)[:, :2]
    to_target_xy = to_target_xy / (torch.norm(to_target_xy, dim=1, keepdim=True) + 1e-6)

    # 用 atan2(叉积z, 点积) 求带符号夹角，范围 [-pi, pi]
    dot = (fwd_xy * to_target_xy).sum(dim=1)
    cross_z = fwd_xy[:, 0] * to_target_xy[:, 1] - fwd_xy[:, 1] * to_target_xy[:, 0]
    heading_err = torch.atan2(cross_z, dot).abs()

    return torch.exp(-torch.square(heading_err / std))

def base_velocity_l2(
    env: ManagerBasedRLEnv,
    standoff: float = 0.4,
    arrive_tol: float = 0.15,
    align_tol: float = 0.6,
    ang_vel_scale: float = 0.3,
    forward_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wheel_.*"),
) -> torch.Tensor:
    """惩罚底盘"到位之后还在动"（水平线速度 + 偏航角速度），带平滑门控

    **为什么需要这一项（现有惩罚项为什么不够）**
        1. ``action_rate_l2`` 罚的是动作的**变化量**。底盘匀速绕圈时动作近似恒定，
           变化量几乎为 0，绕圈基本不花钱，罚不到。
        2. ``joint_vel_l2`` 目前只作用在机械臂关节上；而且更根本的是
           :class:`HolonomicBaseAction` 是直接 ``write_root_velocity_to_sim()``
           写根节点速度的，**完全绕过了轮子关节**——轮子关节速度始终是 0，
           任何基于关节的惩罚都约束不到底盘运动。
        3. Isaac Lab 自带的 ``lin_vel_z_l2`` / ``ang_vel_xy_l2`` 管的是竖直方向和
           翻滚俯仰，不管水平 xy 平移和偏航。

    **为什么底盘会绕圈**
        ``base_to_target_xy_tanh`` 在"轮心到目标距离 == standoff"时最大，
        ``base_facing_target`` 在"工作面指向目标"时最大。这两个条件在
        **半径 standoff 的整个圆周上都同时满足**——沿切向漂移不损失任何奖励，
        于是策略就在圆上打转。这一项的作用是打破这个对称性（给"停住"一个理由），
        而不只是让动作平滑一点。

    **为什么要门控（不能无条件罚速度）**
        无条件罚速度会和 ``base_approach`` 直接对抗，策略可能干脆原地不动；
        而且目标每 2~4 s 会重新随机速度，底盘本来就需要重新追。
        所以只在"已经到位 **且** 已经对准"时才全力生效：

            gate = exp(-(dist_err/arrive_tol)^2) * exp(-(heading_err/align_tol)^2)

        远处或没对准时 gate≈0（放开让它去追），停好了 gate≈1（罚它别晃）。
        门控是平滑的高斯核而不是硬阈值，避免在边界上产生奖励断崖。

    Args:
        env: 环境实例
        standoff: 期望的底盘中心到目标水平距离（米），应与 ``base_approach`` 保持一致
        arrive_tol: "到位"判定的高斯核宽度（米），距离偏差超过它 gate 迅速衰减
        align_tol: "对准"判定的高斯核宽度（弧度），建议与 ``base_facing`` 的 std 一致
        ang_vel_scale: 偏航角速度平方在惩罚里的相对权重。绕圈主要是切向线速度+偏航，
            角速度量级通常比线速度大，所以缩小一点，默认 0.3
        forward_axis: 底盘工作面在 base_link 系中的方向，须与 ``base_facing`` 一致
        robot_cfg: 机器人实体配置
        target_cfg: 目标物体实体配置
        base_cfg: 底盘参考 body 配置，默认四个轮子

    Returns:
        shape (num_envs,) 的**非负**惩罚量（越大越该罚）。
        在 env-cfg 里给它配**负权重**，不要在这里取负号。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # ---- 速度项：水平线速度 + 偏航角速度 ----
    # root_lin_vel_w 是刚体整体速度，底盘是刚性的，用 root 的速度没问题
    # （root_pos_w 的位置偏移不影响线速度，但会影响旋转带来的那部分——
    #  绕圈时 root 和轮心的线速度差是 omega × r，这里角速度项已经单独罚了）
    lin_vel_xy = robot.data.root_lin_vel_w[:, :2]
    ang_vel_z = robot.data.root_ang_vel_w[:, 2]
    vel_sq = torch.sum(torch.square(lin_vel_xy), dim=1) + ang_vel_scale * torch.square(ang_vel_z)

    # ---- 门控1：是否已经到位 ----
    base_center_w = _base_center_w(robot, base_cfg)
    dist_xy = torch.norm(base_center_w[:, :2] - target.data.root_pos_w[:, :2], dim=1)
    arrived = torch.exp(-torch.square((dist_xy - standoff) / arrive_tol))

    # ---- 门控2：是否已经对准 ----
    robot_quat = robot.data.root_quat_w
    axis_b = torch.tensor(forward_axis, device=env.device, dtype=robot_quat.dtype)
    axis_b = axis_b.unsqueeze(0).expand(robot_quat.shape[0], -1)
    fwd_xy = quat_apply(robot_quat, axis_b)[:, :2]
    fwd_xy = fwd_xy / (torch.norm(fwd_xy, dim=1, keepdim=True) + 1e-6)

    to_target_xy = (target.data.root_pos_w - base_center_w)[:, :2]
    to_target_xy = to_target_xy / (torch.norm(to_target_xy, dim=1, keepdim=True) + 1e-6)

    dot = (fwd_xy * to_target_xy).sum(dim=1)
    cross_z = fwd_xy[:, 0] * to_target_xy[:, 1] - fwd_xy[:, 1] * to_target_xy[:, 0]
    heading_err = torch.atan2(cross_z, dot).abs()
    aligned = torch.exp(-torch.square(heading_err / align_tol))

    return vel_sq * arrived * aligned

def _grasp_point_w(
    robot: Articulation,
    ee_cfg: SceneEntityCfg,
    grasp_offset: tuple[float, float, float] | None = None,
    use_link5_com: bool = False,
    link5_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """真实抓取点在世界系下的坐标

    支持两种计算方式：
    1. 使用 end_effector body 原点 + grasp_offset (默认)
    2. 使用 link_005 的质心位置 (use_link5_com=True)

    **方式1（默认）**：
    夹爪是"固定爪 + 活动爪"结构：固定爪(part_034)挂在 link_005 上，活动爪(part_035)
    挂在 end_effector 上，**真正的抓取点在两爪之间**，不是 end_effector 的 body 原点。
    实测偏差 (end_effector 局部系) = (-0.0138, -0.0246, -0.0016)，|d| = 0.028 m，
    主要在前方 2.7 cm。见 assets/eggtart.py 的 EGGTART_EE_GRASP_OFFSET。

    2.8 cm 相对 GRASP_REACH_THRESHOLD=0.05 m 是同一量级，不修的话策略会把
    body 原点怼到目标上，抓取点其实还在目标后面 2.7 cm，夹爪合上是空的。

    **方式2（新增）**：
    直接使用 link_005 的质心位置作为参考点，不需要额外偏移。
    link_005 是机械臂最后一个连杆，其质心位置能更好地代表机械臂末端的实际位置。

    Args:
        robot: 机器人 articulation
        ee_cfg: 末端执行器实体配置（须带 body_names）
        grasp_offset: 抓取点相对 body 原点的偏移，None 表示不偏移（退化为 body 原点）
        use_link5_com: 是否使用 link5 质心作为参考点（忽略 grasp_offset）
        link5_cfg: link5 实体配置，use_link5_com=True 时必须提供

    Returns:
        shape (num_envs, 3) 的世界系坐标
    """
    if use_link5_com:
        # 使用 link5 质心位置
        if link5_cfg is None:
            raise ValueError("link5_cfg must be provided when use_link5_com=True")
        return robot.data.body_com_pos_w[:, link5_cfg.body_ids[0]]
    else:
        # 使用 end_effector body 原点 + 可选偏移
        ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids[0]]
        if grasp_offset is None:
            return ee_pos_w
        ee_quat_w = robot.data.body_quat_w[:, ee_cfg.body_ids[0]]
        off = torch.tensor(grasp_offset, device=ee_pos_w.device, dtype=ee_pos_w.dtype)
        off = off.unsqueeze(0).expand(ee_pos_w.shape[0], -1)
        return ee_pos_w + quat_apply(ee_quat_w, off)


def _ee_to_target_distance(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    grasp_offset: tuple[float, float, float] | None = None,
    use_link5_com: bool = False,
    link5_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """计算**抓取点**到目标的欧氏距离（辅助函数）

    Args:
        env: 环境实例
        ee_cfg: 末端执行器实体配置
        target_cfg: 目标物体实体配置
        grasp_offset: 抓取点相对 end_effector body 原点的偏移（局部系）
        use_link5_com: 是否使用 link5 质心作为参考点
        link5_cfg: link5 实体配置，use_link5_com=True 时必须提供

    Returns:
        shape (num_envs,) 的距离张量，单位：米
    """
    robot: Articulation = env.scene[ee_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    grasp_w = _grasp_point_w(robot, ee_cfg, grasp_offset, use_link5_com, link5_cfg)
    return torch.norm(grasp_w - target.data.root_pos_w, dim=1)

def ee_to_target_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    grasp_offset: tuple[float, float, float] | None = None,
    use_link5_com: bool = False,
    link5_cfg: SceneEntityCfg | None = None,
    narrow_std: float | None = None,
    wide_weight: float = 0.6,
) -> torch.Tensor:
    """奖励抓取点接近目标（tanh 核，可选双核）

    三维空间距离越近，奖励越高。

    支持两种计算方式：
    1. end_effector body 原点 + grasp_offset (默认)
    2. link5 质心位置 (use_link5_com=True)

    Args:
        env: 环境实例
        std: 平滑参数
        ee_cfg: 末端执行器实体配置
        target_cfg: 目标物体实体配置
        grasp_offset: 抓取点相对 end_effector body 原点的偏移（局部系）
        use_link5_com: 是否使用 link5 质心作为参考点
        link5_cfg: link5 实体配置，use_link5_com=True 时必须提供
        narrow_std: 窄核宽度。None 表示单核（向后兼容）；给值则启用双核，
            用于在近场提供陡峭梯度，把抓取点推进最后 10cm
        wide_weight: 双核中宽核的占比，窄核占 1-wide_weight

    Returns:
        shape (num_envs,) 的奖励张量，范围 [0, 1]
    """
    dist = _ee_to_target_distance(env, ee_cfg, target_cfg, grasp_offset, use_link5_com, link5_cfg)
    if narrow_std is None:
        return 1.0 - torch.tanh(dist / std)
    # 双核：宽核保远场引导，窄核给近场精修。
    # 单核解决不了这个矛盾——实测 std=0.2 时 0.1m 处 r≈0.54、梯度已很平，
    # 策略推不动最后 10cm；而把 std 降到 0.1 虽然近场变陡，远场
    # (0.6->0.3m) 梯度会从 0.45 崩到 0.025，等于砍掉 95% 的"伸手靠近"引导。
    # 0.6×std(0.3) + 0.4×std(0.08) 下远场 1.36×、精修 1.26×（相对旧 std=0.2），
    # 中场保留 73%。
    wide = 1.0 - torch.tanh(dist / std)
    narrow = 1.0 - torch.tanh(dist / narrow_std)
    return wide_weight * wide + (1.0 - wide_weight) * narrow


def ee_to_target_precision(
    env: ManagerBasedRLEnv,
    std: float = 0.02,
    support: float = 0.05,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    grasp_offset: tuple[float, float, float] | None = None,
    use_link5_com: bool = False,
    link5_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """近场精修奖励：只在抓取点进入 ``support`` 后才有梯度，用于训练后期精准对齐

    **为什么 ee_to_target_tanh 不够（这一项要解决什么）**
        ``1 - tanh(d/std)`` 在 d 远小于 std 时已经饱和到 ~1。即使双核里最窄的
        std=0.08，d 从 0.02 m 收到 0.005 m 也只带来约 0.019 的奖励增量——
        在权重 3.0 下是 0.056，比 action_rate / joint_vel 的抖动噪声还小。
        也就是说"看着差不多对上了"和"真的对准了"在旧项里几乎同分，
        策略没有理由再往里挤最后那 2 cm。

        本项换成高斯核，把奖励的**陡峭区间放在 0 附近**而不是 std 附近：
        d=0.02 → 0.61，d=0.01 → 0.88，d=0.005 → 0.97。同样从 0.02 收到 0.005，
        奖励涨 0.36（是旧项 0.019 的 19 倍），最后一段才真正有推力。

    **为什么要减掉边界值再归一化（不能直接返回高斯核）**
        直接返回 ``exp(-(d/std)^2)`` 在 d=support 处还有一个不为 0 的残值
        （support=0.05, std=0.02 时是 0.0019），乘上大权重后会在 support 边界
        形成一道奖励断崖——策略可以靠"反复跨进跨出边界"薅分，而且断崖处的
        梯度不连续对 PPO 的 value 拟合不友好。
        这里做 ``(k(d) - k(support)) / (1 - k(support))``，让奖励在 d=support
        处**连续地**归零、在 d=0 处正好是 1.0，边界内外平滑接上。

    **为什么带 support 硬截断（而不是让高斯自然衰减）**
        这一项的定位是"精修"，不是"引导"。远场引导由 ``ee_reach`` 负责，
        两项叠加时如果本项在远处也有值，等于把 ee_reach 的形状改了。
        截断保证它在 5 cm 外恒为 0，可以独立调权重而不影响已经训好的远场行为。

    **和 curriculum 的配合**
        建议 ``weight: 0 -> N`` 在阶段 3 前后拉起来（见 env-cfg 的
        ``ee_precision_sched``）。从 0 开始拉是关键：本项只增不减地往总回报里
        加东西，不会像"把 ee_reach 的 std 调小"那样降低策略已有的收益——
        后者在上一次训练里直接导致策略放弃伸手、退回去刷底盘分。

        也可以再叠一层 ``reward_param_schedule`` 逐步收紧 ``std``
        （如 0.03 -> 0.02 -> 0.015），做"先粗对齐再精对齐"。因为本项在
        d→0 处始终是 1.0，收紧 std 不会降低"已经对准"状态的分数，
        只是把要求提高，比在 ee_reach 上收紧安全得多。

    Args:
        env: 环境实例
        std: 高斯核宽度（米）。控制"多近才算对准"。
            0.02 → d=2cm 得 0.61；0.015 → d=2cm 得 0.36（更严格）
        support: 支撑半径（米）。抓取点在此距离外奖励恒为 0。
            应当 >= 2*std，否则截断点落在高斯核还很陡的地方，
            归一化后近场梯度会被压平（support=2*std 时 k(support)=0.018，尚可；
            support < 1.5*std 则不推荐）。
        ee_cfg: 末端执行器实体配置
        target_cfg: 目标物体实体配置
        grasp_offset: 抓取点相对 body 原点的偏移（局部系）
        use_link5_com: 是否使用 link5 质心作为参考点
        link5_cfg: link5 实体配置，use_link5_com=True 时必须提供

    Returns:
        shape (num_envs,) 的奖励张量，范围 [0, 1]
        - 抓取点与目标重合 -> 1.0
        - 抓取点在 support 处 -> 0.0（连续，无断崖）
        - 抓取点在 support 之外 -> 0.0
    """
    if std <= 0.0:
        raise ValueError(f"ee_to_target_precision: std 必须为正，收到 {std}")
    if support <= 0.0:
        raise ValueError(f"ee_to_target_precision: support 必须为正，收到 {support}")

    dist = _ee_to_target_distance(env, ee_cfg, target_cfg, grasp_offset, use_link5_com, link5_cfg)

    # 高斯核 + 边界归零：(k(d) - k(support)) / (1 - k(support))
    # k(support) 是标量，用 math.exp 算，避免每步建临时张量
    kernel = torch.exp(-torch.square(dist / std))
    edge = math.exp(-((support / std) ** 2))
    reward = (kernel - edge) / (1.0 - edge)

    # support 之外恒为 0（clamp 已能保证非负，这里显式截断表达意图）
    return torch.where(dist < support, reward.clamp(min=0.0), torch.zeros_like(reward))


def ee_to_target_distance_l2(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    grasp_offset: tuple[float, float, float] | None = None,
    use_link5_com: bool = False,
    link5_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """返回抓取点到目标的 L2 距离（配合负权重使用）
    直接返回距离值，通常配合负权重作为惩罚项。

    Args:
        env: 环境实例
        ee_cfg: 末端执行器实体配置
        target_cfg: 目标物体实体配置
        grasp_offset: 抓取点相对 end_effector body 原点的偏移（局部系）
        use_link5_com: 是否使用 link5 质心作为参考点
        link5_cfg: link5 实体配置，use_link5_com=True 时必须提供

    Returns:
        shape (num_envs,) 的距离张量，单位：米
    """
    return _ee_to_target_distance(env, ee_cfg, target_cfg, grasp_offset, use_link5_com, link5_cfg)


def ee_grasp_direction_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="link_005"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    grasp_offset: tuple[float, float, float] | None = None,
    grasp_direction_offset: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """奖励夹爪朝向对准目标（夹持方向指向目标）

    在 link5 坐标系下，抓取点（grasp_offset）到方向点（grasp_direction_offset）的向量
    定义了夹爪的"夹持方向"。当这个方向对准目标时，获得最大奖励。

    典型用法：
        grasp_offset = EGGTART_EE_GRASP_OFFSET          # 夹持中心点
        grasp_direction_offset = EGGTART_EE_GRASP_DERECT_OFFSET  # 方向参考点
        两点连线方向 = 夹爪开口朝向（应该对准目标）

    Args:
        env: 环境实例
        std: 角度误差的高斯核宽度（弧度）
        ee_cfg: link5 实体配置
        target_cfg: 目标物体实体配置
        grasp_offset: 抓取点相对 link5 body 原点的偏移（局部系）
        grasp_direction_offset: 方向参考点相对 link5 body 原点的偏移（局部系）

    Returns:
        shape (num_envs,)，范围 [0, 1]
        - 夹持方向对准目标 -> 1.0
        - 夹持方向垂直于目标 -> ~0.0
    """
    robot: Articulation = env.scene[ee_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    if grasp_offset is None or grasp_direction_offset is None:
        raise ValueError("grasp_offset and grasp_direction_offset are required")

    # 计算抓取点和方向点在世界系下的位置
    ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids[0]]
    ee_quat_w = robot.data.body_quat_w[:, ee_cfg.body_ids[0]]

    # 抓取点世界坐标
    grasp_off = torch.tensor(grasp_offset, device=ee_pos_w.device, dtype=ee_pos_w.dtype)
    grasp_off = grasp_off.unsqueeze(0).expand(ee_pos_w.shape[0], -1)
    grasp_pos_w = ee_pos_w + quat_apply(ee_quat_w, grasp_off)

    # 方向参考点世界坐标
    dir_off = torch.tensor(grasp_direction_offset, device=ee_pos_w.device, dtype=ee_pos_w.dtype)
    dir_off = dir_off.unsqueeze(0).expand(ee_pos_w.shape[0], -1)
    dir_pos_w = ee_pos_w + quat_apply(ee_quat_w, dir_off)

    # 夹持方向（单位向量）
    grasp_dir = dir_pos_w - grasp_pos_w
    grasp_dir = grasp_dir / (torch.norm(grasp_dir, dim=1, keepdim=True) + 1e-6)

    # 目标方向（从抓取点指向目标）
    to_target = target.data.root_pos_w - grasp_pos_w
    to_target = to_target / (torch.norm(to_target, dim=1, keepdim=True) + 1e-6)

    # 对齐度（点积）：1 = 完全对准，0 = 垂直，-1 = 反向
    alignment = (grasp_dir * to_target).sum(dim=1)

    # 转换为角度误差（弧度）
    angle_error = torch.acos(torch.clamp(alignment, -1.0, 1.0))

    # 基础对齐奖励（高斯核）
    base_reward = torch.exp(-torch.square(angle_error / std))

    # 额外奖励：倾向于从上至下（夹持方向应有向下分量）
    # grasp_dir 的 Z 分量：负值表示向下，正值表示向上
    downward_component = -grasp_dir[:, 2]  # 取负号，让向下为正
    # 归一化到 [0, 1]：完全向下=1.0，水平=0.5，完全向上=0.0
    downward_bias = torch.clamp(downward_component * 0.5 + 0.5, 0.0, 1.0)

    # 组合：基础对齐 × (1 + 向下偏好)，让从上方对准的姿态获得更高奖励
    # 从上方对准：base_reward=1.0, downward_bias=0.8 → 1.0 × 1.8 = 1.8
    # 从侧面对准：base_reward=1.0, downward_bias=0.5 → 1.0 × 1.5 = 1.5
    # 从下方对准：base_reward=1.0, downward_bias=0.2 → 1.0 × 1.2 = 1.2
    return base_reward * (1.0 + downward_bias)


class grasp_bonus_dwell(ManagerTermBase):
    """抓取奖励（带停留计时）：抓取点在阈值内**连续停留**足够久后，闭合夹爪才算有效

    **为什么需要计时**
        原来的 :func:`grasp_bonus` 是逐步瞬时判定的：只要"这一帧"抓取点进了
        reach_threshold 且夹爪闭合就给奖励。策略于是学会一边冲向目标一边提前闭爪——
        因为闭爪本身零成本，早闭一点还能提高"恰好在某一帧同时满足两个条件"的概率。
        结果就是夹爪夹早了，还没稳定对准就合上，实际抓不到东西。

        加上"连续停留 dwell_time 秒"的门槛后，提前闭爪不再有收益：奖励只在
        抓取点已经稳定停在目标上之后才生效，策略必须先对准、稳住，再闭爪。

    **计数器语义**
        每个 env 维护一个 ``_dwell_steps`` 计数器（单位：env step，不是物理 step）。
        抓取点在阈值内则 +1，一旦离开阈值立刻**归零**（要求"连续"，不是"累计"）。
        这就是为什么必须写成类：需要跨 step 的 per-env 状态，而且状态要能在
        episode 重置时清掉 —— RewardManager 会对类式项自动调用 ``reset(env_ids)``。

    **和 grasp_bonus 的区别**
        除了计时，本项还把奖励做成了软的：满足条件时返回 1.0，
        并且在停留时间还没攒够时返回 0（不给部分奖励，避免又变成"早闭爪也有点分"）。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # per-env 的连续停留步数计数器
        self._dwell_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # 把 dwell_time(秒) 换算成 env step 数；env.step_dt = sim.dt * decimation
        dwell_time = cfg.params.get("dwell_time", 0.3)
        self._dwell_steps_required = max(1, int(round(dwell_time / env.step_dt)))
        print(
            f"[grasp_bonus_dwell] dwell_time={dwell_time}s, step_dt={env.step_dt:.4f}s"
            f" -> 需要连续 {self._dwell_steps_required} 个 env step 在阈值内"
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """episode 重置时清掉计数器（由 RewardManager 自动调用）"""
        if env_ids is None:
            self._dwell_steps.zero_()
        else:
            self._dwell_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reach_threshold: float,
        gripper_closed_threshold: float,
        dwell_time: float = 0.6,
        ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
        gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["end_effector_joint"]),
        target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
        grasp_offset: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            env: 环境实例
            reach_threshold: 到达阈值（米），抓取点到目标的距离小于它算"在位"
            gripper_closed_threshold: 夹爪闭合角度阈值（弧度），必须 > -0.2（硬限位）
            dwell_time: 需要连续停留的时间（秒）。只在 __init__ 里读，改它要重建环境
            ee_cfg: 末端执行器实体配置
            gripper_cfg: 夹爪关节实体配置
            target_cfg: 目标物体实体配置
            grasp_offset: 抓取点相对 end_effector body 原点的偏移（局部系）

        Returns:
            shape (num_envs,) 的奖励张量，0 或 1
        """
        robot: Articulation = env.scene[ee_cfg.name]
        dist = _ee_to_target_distance(env, ee_cfg, target_cfg, grasp_offset)
        is_near = dist < reach_threshold

        # 连续停留：在阈值内 +1，离开就归零
        self._dwell_steps = torch.where(
            is_near, self._dwell_steps + 1, torch.zeros_like(self._dwell_steps)
        )
        dwelled = self._dwell_steps >= self._dwell_steps_required

        gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids[0]]
        is_closed = gripper_pos < gripper_closed_threshold

        return (dwelled & is_closed).float()


def gripper_premature_close(
    env: ManagerBasedRLEnv,
    reach_threshold: float,
    gripper_closed_threshold: float,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["end_effector_joint"]),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    grasp_offset: tuple[float, float, float] | None = None,
    lift_threshold: float = 0.03,  # 目标抬起3cm以上就不惩罚
) -> torch.Tensor:
    """惩罚"还没到位就闭爪"（治夹爪夹早了）

    只加停留计时是"不给奖励"，属于消极约束——提前闭爪虽然拿不到分，但也不亏，
    策略仍可能保持闭爪的习惯（尤其闭爪几乎不花动作代价）。本项主动给它记上一笔：
    抓取点还在 reach_threshold **之外**、夹爪却是闭合的，就返回 1。

    **重要修改**：增加"目标被抓起"门控。如果目标已经被抬起（说明抓取成功了），
    则不再惩罚闭爪行为。这样在提起物体后，机械臂可以自由调整姿态而不受惩罚。

    配负权重使用。这样"张着爪接近、到位稳住后再闭合"才是最优策略。

    Args:
        env: 环境实例
        reach_threshold: 到达阈值（米）
        gripper_closed_threshold: 夹爪闭合角度阈值（弧度）
        ee_cfg: 末端执行器实体配置
        gripper_cfg: 夹爪关节实体配置
        target_cfg: 目标物体实体配置
        grasp_offset: 抓取点相对 end_effector body 原点的偏移（局部系）
        lift_threshold: 目标抬起多高时不再惩罚（米），默认3cm

    Returns:
        shape (num_envs,) 的**非负**惩罚量（0 或 1），在 env-cfg 里配负权重
    """
    robot: Articulation = env.scene[ee_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    dist = _ee_to_target_distance(env, ee_cfg, target_cfg, grasp_offset)
    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids[0]]

    # 判断目标是否被抬起（相对地面高度）
    ground_height = 0.015  # 目标在地面时的高度
    target_height = target.data.root_pos_w[:, 2]
    is_lifted = (target_height - ground_height) > lift_threshold

    # 原逻辑：远离目标 + 夹爪闭合 → 惩罚
    is_far = dist >= reach_threshold
    is_closed = gripper_pos < gripper_closed_threshold

    # 新增门控：如果目标已被抬起，不惩罚
    penalty = (is_far & is_closed & ~is_lifted).float()

    return penalty


def gripper_close_when_near(
    env: ManagerBasedRLEnv,
    reach_threshold: float,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["end_effector_joint"]),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    grasp_offset: tuple[float, float, float] | None = None,
    gripper_open_pos: float = 1.0,
    gripper_full_close_pos: float = 0.05,
) -> torch.Tensor:
    """惩罚"接近目标时不闭合夹爪"（引导性惩罚，强制policy学会基本抓取动作）

    改为惩罚版本：当接近目标时，夹爪越打开，惩罚越大（配合负权重使用）。
    这比正向奖励"闭合有加分"更强——不闭合会真的扣分。

    **力控下的归一化要点**：
        夹爪改成力控后，夹住物体时关节**到不了硬限位**（被物体挡住）。
        实测 30 mm 立方体停在 q≈0.29。如果还按"闭到 -0.2 才算满分"归一化，
        策略把物体夹稳了也只能拿到约 59% 的分，白白留下一截拿不到的奖励，
        梯度会一直推着它加大夹持力去挤物体。
        所以满分点用 ``gripper_full_close_pos``（默认 0.05，与
        ``GRIPPER_CLOSED_THRESHOLD`` 对齐），夹到该角度即视为完全闭合。

    Args:
        env: 环境实例
        reach_threshold: 距离阈值（米），在这个距离内算"接近"
        ee_cfg, gripper_cfg, target_cfg: 实体配置
        grasp_offset: 抓取点相对 end_effector body 原点的偏移
        gripper_open_pos: 夹爪完全张开的关节角（惩罚最大的那一端）
        gripper_full_close_pos: 视为"完全闭合"的关节角（惩罚为0的那一端）

    Returns:
        shape (num_envs,)，范围 [0, 1]，配合负权重使用
        - 接近目标 且 夹爪打开 -> 1.0 (配合负权重 → 扣分)
        - 接近目标 且 夹爪闭合 -> 0.0 (不扣分)
        - 远离目标 -> 0.0 (不扣分)
    """
    robot: Articulation = env.scene[ee_cfg.name]

    # 距离门控：扩大感受野，使用更平缓的衰减
    dist = _ee_to_target_distance(env, ee_cfg, target_cfg, grasp_offset)
    # 改为线性门控 + clip，而非窄高斯核
    # 在 reach_threshold 内线性衰减：0m→1.0, reach_threshold→0.0
    proximity = torch.clamp(1.0 - dist / reach_threshold, 0.0, 1.0)

    # 张开程度归一化到 [0, 1]：full_close -> 0 (不惩罚)，open -> 1 (最大惩罚)
    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids[0]]
    span = max(gripper_open_pos - gripper_full_close_pos, 1e-6)
    open_amount = torch.clamp((gripper_pos - gripper_full_close_pos) / span, 0.0, 1.0)

    return proximity * open_amount


def gripper_closure_bonus(
    env: ManagerBasedRLEnv,
    reach_threshold: float,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["end_effector_joint"]),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    grasp_offset: tuple[float, float, float] | None = None,
    gripper_open_pos: float = 1.0,
    gripper_full_close_pos: float = 0.05,
    # 渐进式"夹住物体"门控参数
    gate_blend: float = 0.0,
    gate_pos_min: float = 0.18,
    gate_pos_max: float = 0.32,
    gate_effort_threshold: float = 0.1,
) -> torch.Tensor:
    """正向奖励：接近目标时闭合夹爪 + 可选的渐进式"夹住物体"门控

    前期（gate_blend=0）：proximity × close_amount，抓空也给分，让策略学会大胆闭爪。
    后期（gate_blend=1）：加入"真的夹住"判据（位置卡在物块厚度 + 正在施力），
                         抓空不给分，逼策略区分夹住/夹空。
    中间（gate_blend ∈ (0,1)）：线性插值，平滑过渡。

    Args:
        env: 环境实例
        reach_threshold: 距离阈值（米），在这个距离内算"接近"
        ee_cfg, gripper_cfg, target_cfg: 实体配置
        grasp_offset: 抓取点相对 end_effector body 原点的偏移
        gripper_open_pos: 夹爪完全张开的关节角
        gripper_full_close_pos: 视为"完全闭合"的关节角
        gate_blend: 门控混合系数 [0,1]，由 curriculum 控制（默认 0=不门控）
        gate_pos_min: 夹住物体时的最小夹爪位置（如 0.18，物块厚度下界）
        gate_pos_max: 夹住物体时的最大夹爪位置（如 0.32，物块厚度上界）
        gate_effort_threshold: 扭矩阈值（Nm），大于此值算"正在施力"

    Returns:
        shape (num_envs,)，范围 [0, 1]，配合正权重使用
        - gate_blend=0: 接近且闭合 -> 1.0 (抓空也给)
        - gate_blend=1: 接近且闭合且夹住 -> 1.0，抓空 -> 0
    """
    robot: Articulation = env.scene[ee_cfg.name]

    # 距离门控。开方是为了让窗口内部有足够的部分信用：
    # 线性门控下抓取点在 0.7*thresh 处只剩 0.3 的系数，即使完全闭合，
    # 乘上权重后仍小于 ee_reach 的量级，闭爪的信号会被淹没。
    # 开方把 0.3 抬到 0.55，同时保持"越近越好"的单调梯度和 thresh 处归零的硬边界
    # （硬边界是防止在远处白拿闭爪奖励的关键，不能去掉）。
    dist = _ee_to_target_distance(env, ee_cfg, target_cfg, grasp_offset)
    proximity = torch.sqrt(torch.clamp(1.0 - dist / reach_threshold, 0.0, 1.0))

    # 闭合程度归一化到 [0, 1]：full_close -> 1 (最大奖励)，open -> 0 (无奖励)
    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids[0]]
    span = max(gripper_open_pos - gripper_full_close_pos, 1e-6)
    close_amount = torch.clamp(1.0 - (gripper_pos - gripper_full_close_pos) / span, 0.0, 1.0)

    # 基础奖励（抓空也给）
    base_reward = proximity * close_amount

    # 渐进式"夹住物体"门控
    if gate_blend > 1e-6:
        # 位置门控：夹爪卡在物块厚度范围内（闭到底=0.05 说明抓空）
        pos_in_range = (gripper_pos >= gate_pos_min) & (gripper_pos <= gate_pos_max)

        # 力门控：正在施力
        if robot.data.applied_torque is not None and robot.data.applied_torque.numel() > 0:
            gripper_effort = robot.data.applied_torque[:, gripper_cfg.joint_ids[0]]
            is_applying_force = torch.abs(gripper_effort) > gate_effort_threshold
        elif robot.data.joint_effort_target is not None and robot.data.joint_effort_target.numel() > 0:
            gripper_effort = robot.data.joint_effort_target[:, gripper_cfg.joint_ids[0]]
            is_applying_force = torch.abs(gripper_effort) > gate_effort_threshold
        else:
            # 降级：速度接近 0 说明被卡住
            gripper_vel = robot.data.joint_vel[:, gripper_cfg.joint_ids[0]]
            is_applying_force = torch.abs(gripper_vel) < 0.02

        is_holding = pos_in_range & is_applying_force

        # 线性混合：gate_blend=0 忽略门控，gate_blend=1 完全门控
        # 抓空时 is_holding=0，奖励从 base_reward 衰减到 base_reward×(1-gate_blend)
        return base_reward * (1.0 - gate_blend * (1.0 - is_holding.float()))
    else:
        return base_reward


def ee_lift_when_near(
    env: ManagerBasedRLEnv,
    reach_threshold: float,
    gripper_closed_threshold: float,
    lift_height_target: float = 0.35,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["end_effector_joint"]),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    grasp_offset: tuple[float, float, float] | None = None,
    use_link5_com: bool = False,
    link5_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """距离近且夹爪闭合时，提起末端获得奖励（第三阶段稠密引导）

    当末端接近目标且夹爪已闭合时，鼓励策略提升末端高度。
    末端越高（接近 lift_height_target），奖励越高。

    这是一个稠密引导奖励，在主抓取奖励（GraspBonusLift）之前使用，
    帮助策略学会"接近 -> 闭合 -> 提起"的完整动作序列。

    Args:
        env: 环境实例
        reach_threshold: 距离阈值（米），小于此距离才激活
        gripper_closed_threshold: 夹爪闭合阈值，小于此值算闭合
        lift_height_target: 目标提升高度（米）
        ee_cfg: 末端执行器实体配置
        gripper_cfg: 夹爪关节实体配置
        target_cfg: 目标物体实体配置
        grasp_offset: 抓取点偏移
        use_link5_com: 是否使用 link5 质心
        link5_cfg: link5 实体配置

    Returns:
        shape (num_envs,)，范围约 [0, 1]
        - 接近 且 闭合 且 提起 -> ~1.0
        - 不满足任何条件 -> 0.0
    """
    robot: Articulation = env.scene[ee_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # 门控1：距离是否接近
    dist = _ee_to_target_distance(env, ee_cfg, target_cfg, grasp_offset, use_link5_com, link5_cfg)
    near = (dist < reach_threshold).float()

    # 门控2：夹爪是否闭合
    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids[0]]
    closed = (gripper_pos < gripper_closed_threshold).float()

    # 门控：距离近 且 夹爪闭合
    gate = near * closed

    # 末端高度（使用与距离计算相同的参考点）
    if use_link5_com and link5_cfg is not None:
        ee_z = robot.data.body_com_pos_w[:, link5_cfg.body_ids[0], 2]
    else:
        ee_pos_w = _grasp_point_w(robot, ee_cfg, grasp_offset, use_link5_com, link5_cfg)
        ee_z = ee_pos_w[:, 2]

    # 提升奖励：高度越接近目标高度，奖励越高
    # 目标初始在地面约 0.02m，鼓励提升到 lift_height_target
    lift_progress = torch.clamp((ee_z - 0.02) / (lift_height_target - 0.02), 0.0, 1.0)

    return gate * lift_progress


class GraspBonusLift(ManagerTermBase):
    """低速举升至最低高度给基础奖励，靠近目标高度区间再叠加奖励。

    高度、抓取点三维距离和速度必须同时达标，达标即给基础奖励 1。
    额外奖励在最低高度到区间下界之间从 0 线性递增到 1，区间内为 1。
    超过区间上界后按距离指数衰减，衰减尺度为区间下界与最低高度之差，
    使抬得较高的目标向区间回落时也能持续获得更高奖励。
    保持计时器只记录连续达标时间，不延迟奖励；重置时清零。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 每个 env 独立的提起保持计数器（秒）
        self._lift_counter = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Episode 重置时归零计数器"""
        if env_ids is None:
            self._lift_counter.zero_()
        else:
            self._lift_counter[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_height_threshold: float,
        lift_dwell_time: float,
        target_cfg: SceneEntityCfg,
        ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
        grasp_offset: tuple[float, float, float] | None = None,
        hold_dist: float = 0.2,
        max_speed: float = 3.6,
        lift_target_height_threshold1: float = 0.14,
        lift_target_height_threshold2: float = 0.16,
    ) -> torch.Tensor:
        """
        Args:
            env: 环境实例
            lift_height_threshold: 目标质心必须达到的高度（米）
            lift_dwell_time: 保留兼容现有配置；即时奖励不使用保持时间门槛。
            target_cfg: 目标物体实体配置
            ee_cfg: 抓取点参考连杆配置
            grasp_offset: 抓取点相对参考连杆的局部偏移
            hold_dist: 物块到抓取点的最大三维距离（米）
            max_speed: 目标最大线速度（米/秒）
            lift_target_height_threshold1: 目标高度区间下界，须大于最低举升高度（米）
            lift_target_height_threshold2: 目标高度区间上界，须不小于下界（米）

        Returns:
            shape (num_envs,) 的奖励张量，范围 [0, 2]；基础奖励和区间奖励各占 1。
        """
        if not (
            math.isfinite(lift_height_threshold)
            and math.isfinite(lift_target_height_threshold1)
            and math.isfinite(lift_target_height_threshold2)
            and lift_height_threshold < lift_target_height_threshold1 <= lift_target_height_threshold2
        ):
            raise ValueError("举升高度须为有限值且满足 lift_height_threshold < 目标区间下界 <= 上界")

        target: RigidObject = env.scene[target_cfg.name]

        target_z = target.data.root_pos_w[:, 2]
        held_near = _ee_to_target_distance(env, ee_cfg, target_cfg, grasp_offset) < hold_dist
        slow = torch.linalg.vector_norm(target.data.root_lin_vel_w, dim=1) < max_speed
        lifted = (target_z >= lift_height_threshold) & held_near & slow

        # 脱手或高速甩动会打断连续保持，重新满足条件后必须重新计时。
        dt = env.step_dt
        self._lift_counter = torch.where(
            lifted, self._lift_counter + dt, torch.zeros_like(self._lift_counter)
        )

        shaping_width = lift_target_height_threshold1 - lift_height_threshold
        approach = torch.clamp((target_z - lift_height_threshold) / shaping_width, min=0.0, max=1.0)
        overshoot = torch.clamp(target_z - lift_target_height_threshold2, min=0.0)
        height_bonus = approach * torch.exp(-overshoot / shaping_width)
        return lifted.float() * (1.0 + height_bonus)


# 导出名称供 cfg.py 使用
grasp_bonus_lift = GraspBonusLift

def retract_bonus_lift(
    env: ManagerBasedRLEnv,
    lift_height_threshold: float,
    std: float,
    arm_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """抓取后回收奖励：提起物体后，机械臂回到 nominal 姿态（基于提起高度判定）

    只有在"已提起"状态下（目标高于阈值），机械臂关节越接近默认姿态奖励越高（高斯核）。

    Args:
        env: 环境实例
        lift_height_threshold: 目标质心必须达到的高度（米）
        std: 平滑参数，控制奖励曲线陡峭程度
        arm_cfg: 机械臂关节实体配置
        target_cfg: 目标物体实体配置

    Returns:
        shape (num_envs,) 的奖励张量，范围 [0, 1]
    """
    robot: Articulation = env.scene[arm_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # 判断是否已提起
    target_z = target.data.root_pos_w[:, 2]
    lifted = (target_z >= lift_height_threshold).float()

    # 机械臂关节偏离 nominal 的距离（L2）
    arm_joint_ids = arm_cfg.joint_ids
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]  # (num_envs, 5)
    joint_pos_default = robot.data.default_joint_pos[:, arm_joint_ids]
    error = torch.norm(joint_pos - joint_pos_default, dim=-1)

    reward = lifted * torch.exp(-0.5 * (error / std) ** 2)
    return reward


def arm_comfort(
    env: ManagerBasedRLEnv,
    arm_cfg: SceneEntityCfg,
    std: float = 1.0,
) -> torch.Tensor:
    """机械臂舒适度奖励：鼓励机械臂保持接近 nominal 姿态

    通过惩罚关节偏离默认位置，间接引导底盘停在"让机械臂舒服工作"的位置。
    使用高斯核而非 L2 距离，避免远离 nominal 时惩罚线性增长（那样会和 ee_reach 对抗）。

    **为什么需要这一项**：
        当前奖励设计中，底盘和末端的目标可能冲突：
        - `base_approach` 要求底盘停在 standoff 距离外
        - `ee_reach` 要求末端尽可能靠近目标

        如果 standoff 距离设置不当（太近或太远），策略会发现：
        1. 底盘冲到目标旁边 → 末端能靠近 → ee_reach 高分
        2. 但机械臂被迫扭曲到极限位置 → 实际上够不到

        加入 `arm_comfort` 后，策略会权衡：
        - 底盘停在让机械臂舒服伸展的位置（关节接近 nominal）
        - 而不是为了缩短末端距离，把机械臂逼到极限

    **与 `joint_pos_limits` 的区别**：
        - `joint_pos_limits`：只惩罚超出软限位（行程最外 5%）的部分，中间区域不管
        - `arm_comfort`：全程鼓励接近 nominal，越偏离越不舒服（但用高斯核平滑）

        两者互补：`joint_pos_limits` 是硬约束（别压限位），`arm_comfort` 是软引导（别离太远）。

    Args:
        env: 环境实例
        arm_cfg: 机械臂关节实体配置（不包括夹爪，夹爪需要开合）
        std: 高斯核宽度（弧度），控制"多偏就算不舒服"。
            std=1.0 时，偏离 1 rad（57°）奖励降到 0.6；
            std=0.5 时更严格，偏离 0.5 rad（28°）就降到 0.6。

    Returns:
        shape (num_envs,) 的奖励张量，范围 [0, 1]，越接近 nominal 越高
    """
    robot: Articulation = env.scene[arm_cfg.name]

    # 机械臂关节偏离 nominal 的距离（L2 范数）
    arm_joint_ids = arm_cfg.joint_ids
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]  # (num_envs, N)
    joint_pos_default = robot.data.default_joint_pos[:, arm_joint_ids]
    error = torch.norm(joint_pos - joint_pos_default, dim=-1)

    # 高斯核奖励：偏离 0 时为 1，偏离越大越小
    reward = torch.exp(-0.5 * (error / std) ** 2)
    return reward


def grasp_posture_guide(
    env: ManagerBasedRLEnv,
    target_joint_pos: dict[str, float],
    std: float,
    reach_threshold: float,
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="link_005"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    grasp_offset: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """当末端接近目标时，引导机械臂采用特定的抓取姿态

    只有在末端接近目标（距离 < reach_threshold）时才激活，避免在底盘导航阶段干扰。
    使用高斯核奖励偏离目标姿态的程度，鼓励策略在抓取前摆出"容易成功"的姿态。

    **与 arm_comfort 的区别**：
    - arm_comfort: 全程开启，鼓励保持 nominal（收起）姿态，避免过度伸展
    - grasp_posture_guide: 只在接近目标时开启，鼓励切换到抓取（伸展）姿态

    典型用法：
        grasp_posture_guide_sched = CurrTerm(
            func=mdp.reward_weight_schedule,
            params={"term_name": "grasp_posture_guide", "schedule": [(0, 0.0), (30000, 2.0)]},
        )

    Args:
        env: 环境实例
        target_joint_pos: 目标关节角度字典（来自 EGGTART_GRASP_JOINT_POS）
        std: 高斯核标准差（弧度），偏离多少算可接受
        reach_threshold: 距离阈值（米），末端在此距离内才激活
        arm_cfg: 机械臂实体配置
        ee_cfg: 末端执行器实体配置
        target_cfg: 目标物体实体配置
        grasp_offset: 抓取点偏移

    Returns:
        shape (num_envs,) 的奖励，范围 [0, 1]
        - 姿态完全匹配且接近目标 -> 1.0
        - 姿态偏离或距离远 -> 0.0
    """
    robot: Articulation = env.scene[arm_cfg.name]

    # 距离门控：只在接近目标时才激活
    dist = _ee_to_target_distance(env, ee_cfg, target_cfg, grasp_offset)
    proximity = torch.clamp(1.0 - dist / reach_threshold, 0.0, 1.0)

    # 提取机械臂关节（排除夹爪和轮子）
    arm_joint_ids = arm_cfg.joint_ids
    current_pos = robot.data.joint_pos[:, arm_joint_ids]

    # 构建目标姿态张量（只包含机械臂关节）
    target_pos_list = []
    for joint_name in arm_cfg.joint_names:
        if joint_name in target_joint_pos:
            target_pos_list.append(target_joint_pos[joint_name])
        else:
            # 如果没有指定，使用当前 nominal 值（fallback）
            target_pos_list.append(robot.data.default_joint_pos[0, robot.find_joints(joint_name)[0][0]].item())

    target_pos = torch.tensor(target_pos_list, device=env.device, dtype=torch.float32)
    target_pos = target_pos.unsqueeze(0).expand(env.num_envs, -1)

    # 计算姿态误差（L2 norm）
    error = torch.norm(current_pos - target_pos, dim=1)

    # 高斯核奖励：error=0 -> 1.0, error=std -> 0.6, error=2*std -> 0.14
    posture_reward = torch.exp(-0.5 * torch.square(error / std))

    # 只在接近目标时才给奖励
    return proximity * posture_reward


def gripper_holding_object(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["end_effector_joint"]),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    grasp_offset: tuple[float, float, float] | None = None,
    pos_min: float = 0.18,
    pos_max: float = 0.32,
    effort_threshold: float = 0.1,
    distance_threshold: float = 0.07,
) -> torch.Tensor:
    """检测夹爪是否真正夹住物体（力控夹爪专用）

    力控夹爪的关键特征：
    - 位置：gripper_pos 停在 0.18-0.32（被物体挡住）
    - 力矩：正在施加闭合力矩（effort_target > threshold 或 applied_torque > threshold）
    - **距离门控**：夹爪必须接近目标（< 7cm）才计入奖励

    Args:
        env: 环境实例
        gripper_cfg: 夹爪关节配置
        target_cfg: 目标物体配置
        ee_cfg: 末端执行器配置
        grasp_offset: 抓取点偏移
        pos_min: 夹住物体时的最小位置（弧度）
        pos_max: 夹住物体时的最大位置（弧度）
        effort_threshold: 力矩阈值（Nm），绝对值大于此值算"正在施力"
        distance_threshold: 距离阈值（米），夹爪必须在此距离内才给奖励

    Returns:
        shape (num_envs,) 的奖励，范围 [0, 1]
        - 位置在范围内 且 正在施加力 且 接近目标 -> 1.0
        - 否则 -> 0.0
    """
    robot: Articulation = env.scene[gripper_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    gripper_joint_idx = gripper_cfg.joint_ids[0]

    # 0. 距离门控：夹爪必须接近目标
    from isaaclab.utils.math import quat_apply
    ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids[0], :]
    ee_quat_w = robot.data.body_quat_w[:, ee_cfg.body_ids[0], :]

    if grasp_offset is not None:
        grasp_off = torch.tensor(grasp_offset, device=env.device, dtype=ee_pos_w.dtype)
        grasp_off = grasp_off.unsqueeze(0).expand(ee_pos_w.shape[0], -1)  # (num_envs, 3)
        grasp_pos_w = ee_pos_w + quat_apply(ee_quat_w, grasp_off)
    else:
        grasp_pos_w = ee_pos_w

    target_pos = target.data.root_pos_w
    dist = torch.norm(grasp_pos_w - target_pos, dim=1)
    is_near = dist < distance_threshold

    # 1. 检查夹爪位置：在合理范围内（被物体挡住的位置）
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]
    pos_in_range = (gripper_pos >= pos_min) & (gripper_pos <= pos_max)

    # 2. 检查夹爪是否正在施加力矩（力控的关键）
    # 优先使用 applied_torque（实际施加的力矩），fallback 到 joint_effort_target（指令力矩）
    if robot.data.applied_torque is not None:
        gripper_effort = robot.data.applied_torque[:, gripper_joint_idx]
    elif robot.data.joint_effort_target is not None:
        gripper_effort = robot.data.joint_effort_target[:, gripper_joint_idx]
    else:
        # Fallback: 如果没有力矩数据，用速度接近0判断
        gripper_vel = robot.data.joint_vel[:, gripper_joint_idx]
        is_applying_force = torch.abs(gripper_vel) < 0.02
        is_holding = pos_in_range & is_applying_force & is_near
        return is_holding.float()

    # 力矩绝对值大于阈值，说明正在施力（闭合或张开都算）
    is_applying_force = torch.abs(gripper_effort) > effort_threshold

    # 3. 组合判断：位置正确 且 正在施力 且 接近目标
    is_holding = pos_in_range & is_applying_force & is_near

    return is_holding.float()


def target_lift_progress(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["end_effector_joint"]),
    target_height: float = 0.14,
    std: float = 0.05,
    gripper_closed_threshold: float = 0.35,
    gripper_open_pos: float = 1.0,  # 张开端；必须 > gripper_closed_threshold
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    grasp_offset: tuple[float, float, float] | None = None,
    hold_dist: float = 0.2,
    max_speed: float = 3.6,
) -> torch.Tensor:
    """渐进式目标提升奖励：物块须靠近抓取点且低速运动，并按夹爪闭合度门控。

    **重要修改**：使用渐进式闭合门控（连续值），而非二值门控。
    防止"推目标"欺骗行为的同时，提供密集梯度引导夹爪逐渐闭合。

    不是二值的 0/1 稀疏奖励，而是高斯核平滑奖励，提供密集梯度。
    与 grasp_bonus_lift（稀疏奖励）互补：这个提供探索期的引导，那个提供成功时的大奖。

    典型用法：
        target_lift_progress = RewTerm(
            func=mdp.target_lift_progress,
            weight=20.0,
            params={
                "target_height": 0.12,
                "std": 0.05,
                "target_cfg": SceneEntityCfg("target"),
                "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
                "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
                "gripper_open_pos": EGGTART_GRIPPER_OPEN,  # 1.0，张开端
            },
        )

    Args:
        env: 环境实例
        target_cfg: 目标物体配置
        gripper_cfg: 夹爪关节配置
        target_height: 目标高度（m），提到此高度时奖励最大
        std: 高斯核标准差（m），控制奖励的平滑度
        gripper_closed_threshold: 夹爪闭合阈值，小于此值算完全闭合
        gripper_open_pos: 夹爪完全**张开**的关节角（须 > gripper_closed_threshold）
        ee_cfg: 抓取点参考连杆配置
        grasp_offset: 抓取点相对参考连杆的局部偏移
        hold_dist: 物块到抓取点的最大三维距离（米）
        max_speed: 目标最大线速度（米/秒）

    Returns:
        奖励张量，夹爪闭合度 × 提升奖励，范围 [0, 1]
        - 完全张开 → 0（无奖励）
        - 闭合一半 → 0.5×提升奖励
        - 完全闭合且提到目标高度 → 1.0
    """
    robot: Articulation = env.scene[gripper_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # 渐进式门控：夹爪闭合度（连续值 0~1，提供密集梯度）
    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids[0]]
    # span 必须为正：gripper_open_pos 是**张开**端（大值，如 1.0），
    # gripper_closed_threshold 是闭合端（小值，如 0.35）。传反会让整个门控反号——
    # 夹爪张开时反而拿满分，权重又比 gripper_closure_reward 大，
    # 策略会稳定收敛到"永久张开"。这里显式报错而不是静默算错。
    span = gripper_open_pos - gripper_closed_threshold  # 张开到闭合的行程
    if span <= 0:
        raise ValueError(
            f"target_lift_progress: gripper_open_pos ({gripper_open_pos}) 必须大于 "
            f"gripper_closed_threshold ({gripper_closed_threshold})，否则闭合门控反号。"
        )
    gripper_closure = torch.clamp((gripper_open_pos - gripper_pos) / span, 0.0, 1.0)
    # gripper_closure = 0: 完全张开，无奖励
    # gripper_closure = 0.5: 闭合一半，奖励减半
    # gripper_closure = 1.0: 完全闭合，全额奖励

    current_height = target.data.root_pos_w[:, 2]

    # 地面高度（目标初始在地面约 0.015-0.020m）
    ground_height = 0.015

    # 提升量
    lift_amount = current_height - ground_height

    # 高斯核：提到 target_height 时最大（1.0），远离时平滑衰减
    # exp(-0.5 * ((lift - target) / std)^2)
    reward = torch.exp(-0.5 * torch.square((lift_amount - target_height) / std))

    # 如果提升量太小（<5mm，仍在地面），不给奖励
    reward = torch.where(lift_amount > 0.005, reward, torch.zeros_like(reward))

    # 与稀疏奖励一致，脱手或高速甩动时不给举升进度奖励。
    held_near = _ee_to_target_distance(env, ee_cfg, target_cfg, grasp_offset) < hold_dist
    slow = torch.linalg.vector_norm(target.data.root_lin_vel_w, dim=1) < max_speed
    reward = torch.where(held_near & slow, reward, torch.zeros_like(reward))

    # 渐进式门控：夹爪闭合越多，奖励越大（提供密集梯度）
    return gripper_closure * reward
