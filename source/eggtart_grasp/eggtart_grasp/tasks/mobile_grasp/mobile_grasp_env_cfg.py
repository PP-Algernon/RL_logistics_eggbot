"""Eggtart 移动抓取任务的基础 ManagerBasedRLEnvCfg 配置

场景：带麦轮的移动机械臂在地面上追逐并抓取移动目标物体（小浮动立方体，代表蛋挞载荷）。

机器人本身的 articulation 配置在此处留空（MISSING），由具体配置文件
``config/eggtart/grasp_env_cfg.py`` 填充。
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import eggtart_grasp.tasks.mobile_grasp.mdp as mdp
from eggtart_grasp.assets.eggtart import (
    EGGTART_ARM_JOINT_NAMES,
    EGGTART_BASE_FORWARD_AXIS,
    EGGTART_EE_GRASP_OFFSET,
    EGGTART_EE_GRASP_DERECT_OFFSET,
    EGGTART_GRIPPER_JOINT_NAME,
    EGGTART_WHEEL_JOINT_BODY_REGEX,
    EGGTART_WHEEL_JOINT_NAMES,
    EGGTART_GRASP_JOINT_POS,
)

# ---------------------------------------------------------------------------
# Tunable task constants
# ---------------------------------------------------------------------------
# 抓取成功判定：目标物体被提起到的高度阈值
# 目标初始生成高度在地面附近(0.015-0.020 m)，降低门槛让策略更容易获得正反馈
# 从地面（0.015m）提升 10cm（到 0.115m）就算成功，鼓励探索
LIFT_HEIGHT_THRESHOLD = 0.12  # m (目标质心高度，降低到 12cm)

# 提起后必须保持在高度阈值以上这么久才算稳定抓取
# 降低到 0.1s（约 3 步），避免"碰一下就掉"但不要求太久
LIFT_DWELL_TIME = 0.1  # s

# 目标移动速度范围（用于随机初始速度和周期性速度变化）
TARGET_VELOCITY_RANGE = 0.10  # m/s (±range，降低到原来的40%)

# 抓取点落在这个距离内算"到达目标"
# 调整建议：如果EE一直到不了，可以放宽到0.08甚至0.10；等学会了再收紧
GRASP_REACH_THRESHOLD = 0.05      # m (放宽让policy更容易触发grasp)

# 夹爪关节低于此角度算"闭合"
#
# **改成力控后这个值必须跟着改**：位置控制时空夹能压到硬限位 -0.2，所以阈值 -0.15
# 合理；但力控下夹住物体时夹爪**被物体挡住**，根本到不了 -0.2——
# 实测 30 mm 立方体停在 q≈0.29，40 mm 停在 q≈0.40。
# 如果还用 -0.15，"夹住了"这个条件永远不成立，grasp / retract 奖励恒为 0。
#
# 0.35 的依据：目标立方体 30 mm 停在 0.29，留一点余量；同时 0.35 对应开口约 44 mm，
# 比物体宽——也就是"明显在往里夹但还没夹到"不会误判成已闭合。
# 换目标尺寸要重算：开口-角度对应关系见 assets/eggtart.py 的注释表。
GRIPPER_CLOSED_THRESHOLD = 0.35

# 抓取点必须在 GRASP_REACH_THRESHOLD 内**连续停留**这么久，闭爪才算有效抓取
# env.step_dt = (1/120) * 4 = 1/30 s，所以 0.3 s ≈ 9 步，0.6 s ≈ 18 步
# 调整建议：如果grasp一直是0，先降到0.2让policy能拿到奖励，再逐步提高要求
GRASP_DWELL_TIME = 0.2  # s (降低难度，让policy先学会基本动作)


##
# 场景定义
##
@configclass
class MobileGraspSceneCfg(InteractiveSceneCfg):
    """场景配置：地面、光照、移动机械臂和移动目标"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 机器人 -- 由具体配置填充
    robot: ArticulationCfg = MISSING

    # 移动目标：小立方体，恢复重力使其能落地并被抓起
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  # 恢复重力，允许物体落地和被提起
                linear_damping=0.5,      # 增加阻尼，降低滑动
                angular_damping=0.5,     # 增加角阻尼，稳定旋转
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.75, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.1)),
    )

    # 光照
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP 设置
##
@configclass
class ActionsCfg:
    """动作项配置：3 维全向底盘速度 + 机械臂关节位置 + 夹爪"""

    base_velocity = mdp.HolonomicBaseActionCfg(
        asset_name="robot",
        max_lin_vel_x=0.6,
        max_lin_vel_y=0.6,
        max_ang_vel_z=1.5,
    )
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=EGGTART_ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )
    # 夹爪改为**力控**：策略输出夹持力，最终停在哪由接触自然决定，自适应物体体积。
    # 位置控制下策略必须精确命令一个关节角，而"物体多大就该停在哪个角度"是几何决定的
    # （实测 20/30/40 mm 物体分别停在 q=0.116/0.290/0.400），位置控制要么闭不到位、
    # 要么把物体挤穿。详见 actions.py 的 GripperForceAction。
    # max_effort 实测：0.3 Nm 能稳定夹住 30 mm 立方体，3.0 Nm 会把它挤穿。
    gripper_action = mdp.GripperForceActionCfg(
        asset_name="robot",
        joint_names=[EGGTART_GRIPPER_JOINT_NAME],
        max_effort=1.0,
        damping=0.05,
    )


@configclass
class ObservationsCfg:
    """MDP 的观测规格配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        # 本体感知
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        # 任务相关
        target_position_b = ObsTerm(
            func=mdp.target_position_in_base_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")},
        )
        # 使用 link_005 作为参考点（夹爪运动不影响 link5 坐标系）
        ee_to_target_b = ObsTerm(
            func=mdp.ee_to_target_vector_base_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
                "target_cfg": SceneEntityCfg("target"),
                "grasp_offset": EGGTART_EE_GRASP_OFFSET,
            },
        )
        target_lin_vel = ObsTerm(
            func=mdp.target_lin_vel_w, params={"target_cfg": SceneEntityCfg("target")}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """重置和间隔事件配置"""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )
    reset_target = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 在机器人前方/周围生成目标，地面高度（让它自然落地）
            "pose_range": {"x": (0.4, 1.2), "y": (-0.6, 0.6), "z": (0.015, 0.020)},  # z降低到接近地面（立方体高0.03m，质心在0.015m）
            "velocity_range": {
                "x": (-TARGET_VELOCITY_RANGE, TARGET_VELOCITY_RANGE),
                "y": (-TARGET_VELOCITY_RANGE, TARGET_VELOCITY_RANGE),
            },
            "asset_cfg": SceneEntityCfg("target"),
        },
    )

    # 逆向课程学习：三阶段目标行为
    # 阶段 1 (iter 0-375, step 0-9000): 目标主动靠近末端方向点（辅助学习）
    target_approach_stage1 = EventTerm(
        func=mdp.target_approach_ee_direction,
        mode="interval",
        interval_range_s=(0.1, 0.2),  # 高频更新，实时跟踪末端
        params={
            "approach_speed": 0.05,  # 5 cm/s 缓慢靠近
            "activation_distance": 0.1,  # 抓取点进入 10cm 内才触发
            "stage1_end_step": 64000,  # 阶段1结束步数
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
            "direction_offset": EGGTART_EE_GRASP_DERECT_OFFSET,
        },
    )

    # 三阶段目标移动控制：
    # - 阶段1 (step < 64000): 随机移动（配合 target_approach）
    # - 阶段2 (64000 <= step < 120000): 静止
    # - 阶段3 (step >= 120000): 恢复随机移动
    randomize_target_velocity = EventTerm(
        func=mdp.randomize_target_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "velocity_range": {
                "x": (-TARGET_VELOCITY_RANGE, TARGET_VELOCITY_RANGE),
                "y": (-TARGET_VELOCITY_RANGE, TARGET_VELOCITY_RANGE),
            },
            "stage2_start_step": 64000,   # 阶段2开始
            "stage3_start_step": 120000,  # 阶段3开始
            "asset_cfg": SceneEntityCfg("target"),
        },
    )


@configclass
class RewardsCfg:
    """分阶段奖励项配置（权重可调整）"""

    # 阶段 1: 底盘运动
    # base_cfg 必须显式传：底盘位置要用轮心（几何中心），不能用 root_pos_w。
    # standoff: 参考点停在目标外 0.5 m，留出机械臂伸展的空间（不是越近越好）。
    # arm_link1_cfg: 使用机械臂 link1 作为参考点，比底盘中心更能反映机械臂工作空间
    base_approach = RewTerm(
        func=mdp.base_to_target_xy_tanh,
        weight=2.5,
        params={
            "std": 0.5,
            "standoff": 0.5,
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
            "base_cfg": SceneEntityCfg("robot", body_names=EGGTART_WHEEL_JOINT_BODY_REGEX),
            "arm_link1_cfg": SceneEntityCfg("robot", body_names="link_001"),
        },
    )
    # 底盘朝向（工作面对准目标）
    # std 是夹角误差的高斯核宽度（弧度），0.6 rad ≈ 34°
    # forward_axis 是底盘工作面在 base_link 系中的方向，本机器人为 -Y（见 rewards.py 说明）
    base_facing = RewTerm(
        func=mdp.base_facing_target,
        weight=1.2,
        params={
            "std": 0.6,
            "forward_axis": EGGTART_BASE_FORWARD_AXIS,
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
            "base_cfg": SceneEntityCfg("robot", body_names=EGGTART_WHEEL_JOINT_BODY_REGEX),
        },
    )
    # 机械臂舒适度：鼓励关节保持接近 nominal 姿态
    # 全程开启，引导底盘停在"让机械臂舒服工作"的位置，避免为了缩短末端距离而让机械臂扭曲
    arm_comfort = RewTerm(
        func=mdp.arm_comfort,
        weight=1.0,  # 正奖励，越接近 nominal 越高
        params={
            "std": 1.0,  # 偏离 1 rad（57°）时奖励降到 0.6
            "arm_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES),
        },
    )
    # 阶段 2: 末端执行器到达
    #
    ee_reach = RewTerm(
        func=mdp.ee_to_target_tanh,
        weight=2.0,  # 提高权重，让末端靠近成为主要目标
        params={
            "std": 0.2,  # 从 0.1 → 0.2 (20cm)：进一步扩大感受野，距离 0.3-0.4m 也有明显梯度
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    # 夹爪朝向对准目标（夹持方向应该指向目标）
    ee_orientation = RewTerm(
        func=mdp.ee_grasp_direction_alignment,
        weight=2.5,
        params={
            "std": 0.5,  # 角度容差约 30 度
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
            "grasp_direction_offset": EGGTART_EE_GRASP_DERECT_OFFSET,
        },
    )
    # 末端执行器与目标之间的距离
    ee_distance = RewTerm(
        func=mdp.ee_to_target_distance_l2,
        weight=-1.0,  # 从 -1.0 → -3.0：加大惩罚，持续推动靠近
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    # 阶段 3: 抓取
    # 用带停留计时的版本：抓取点必须在阈值内**连续停留** GRASP_DWELL_TIME 秒，
    # 之后闭爪才算有效。原来的瞬时判定让策略学会边冲边提前闭爪——闭爪零成本，
    # 早闭还能提高"某一帧恰好同时满足两条件"的概率，于是夹爪总是夹早了。
    # 计数器是 per-env 的，离开阈值立刻归零（要求连续，不是累计），
    # episode 重置时由 RewardManager 自动清零（类式奖励项才有这个能力）。
    # 引导惩罚：接近目标时不闭合夹爪会被惩罚（配合负权重）
    gripper_close_guide = RewTerm(
        func=mdp.gripper_close_when_near,
        weight=-1.5,  # 加大惩罚：接近时不闭合 → 扣分，逼策略必须闭合
        params={
            "reach_threshold": GRASP_REACH_THRESHOLD,
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
            "gripper_full_close_pos": GRIPPER_CLOSED_THRESHOLD,  # 传正确的闭合阈值
        },
    )
    # 主动惩罚"还没到位就闭爪"。
    # 只加停留计时是消极约束（拿不到分但也不亏），策略可能保持闭爪的习惯；
    # 这一项让提前闭爪真的要花钱，"张爪接近 -> 到位稳住 -> 再闭合"才最优。
    gripper_early = RewTerm(
        func=mdp.gripper_premature_close,
        weight=-0.5,  # 加大惩罚：远离时闭合也要扣分，避免"一直闭着"
        params={
            "reach_threshold": GRASP_REACH_THRESHOLD,
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    # 引导奖励：接近且闭合时鼓励提起末端（稠密奖励，完整动作序列）
    ee_lift_guide = RewTerm(
        func=mdp.ee_lift_when_near,
        weight=7.0,  # 提高权重：让"提起"的吸引力 > "不闭合"的惩罚
        params={
            "reach_threshold": GRASP_REACH_THRESHOLD,
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "lift_height_target": LIFT_HEIGHT_THRESHOLD,
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    # 稀疏抓取奖励（基于提起高度判定，不再依赖夹爪角度）
    # 目标必须被提起到指定高度并保持一段时间才算成功抓取
    grasp = RewTerm(
        func=mdp.grasp_bonus_lift,
        weight=15.0,  # 提高权重，因为这是真正的任务目标
        params={
            "lift_height_threshold": LIFT_HEIGHT_THRESHOLD,
            "lift_dwell_time": LIFT_DWELL_TIME,
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # 渐进式目标提升奖励：提供密集梯度，在稀疏的 grasp 之前引导策略探索
    # 目标提升得越高，奖励越多（高斯核平滑），不是二值的 0/1
    # 重要：使用渐进式闭合门控，防止"推目标"欺骗行为的同时提供密集梯度
    target_lift_progress = RewTerm(
        func=mdp.target_lift_progress,
        weight=20.0,  # 高权重，提供强烈的提升信号
        params={
            "target_height": LIFT_HEIGHT_THRESHOLD,  # 提到门槛高度时奖励最大
            "std": 0.05,  # 5cm 的平滑窗口
            "target_cfg": SceneEntityCfg("target"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "gripper_open_pos": 0.05,  # 夹爪完全张开位置
        },
    )

    # 惩罚项
    # 机械臂舒适度：鼓励关节保持接近 nominal 姿态
    # 全程开启，引导底盘停在"让机械臂舒服工作"的位置，避免为了缩短末端距离而让机械臂扭曲
    arm_comfort = RewTerm(
        func=mdp.arm_comfort,
        weight=1.0,  # 正奖励，越接近 nominal 越高
        params={
            "std": 1.0,  # 偏离 1 rad（57°）时奖励降到 0.6
            "arm_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES),
        },
    )

    # 抓取姿态引导：接近目标时，鼓励机械臂摆出特定的抓取姿态
    # 只在末端接近目标时激活，与 arm_comfort 互补（远离时保持 nominal，接近时切换到抓取姿态）
    grasp_posture_guide = RewTerm(
        func=mdp.grasp_posture_guide,
        weight=2.0,  # 正奖励，接近目标时摆出抓取姿态
        params={
            "target_joint_pos": EGGTART_GRASP_JOINT_POS,  # 目标抓取姿态
            "std": 1.0,  # 0.5→1.0，放宽容差（偏离 1 rad 时仍有 0.6 奖励）
            "reach_threshold": GRASP_REACH_THRESHOLD * 3,  # 2→3，距离 <15cm 时激活（进一步放宽）
            "arm_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES),
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )

    # action_rate 权重从 -0.001 提到 -0.02。
    # 实测（恒定动作跑 200 步）关节位置峰峰值只有 ~1e-6 rad，物理侧完全干净，
    # 所以抖动是**策略在输出 bang-bang 动作**，不是仿真数值问题。
    # 旧权重下：臂+爪 6 维在 ±1 之间来回跳 -> action_rate_l2 ≈ 6*(2^2) = 24,
    # 代价 24*0.001 = 0.024/步，而 ee_reach 靠近目标时单步就有 ~2.0，
    # 抖动只花掉主奖励的 ~1%，几乎免费。-0.02 让同样的抖动代价约 0.48，量级才可比。
    action_rate = RewTerm(
        func=mdp.action_rate_l2, weight=-0.02
    )

    # joint_vel 权重 -0.0001 -> -0.002，并把**夹爪**纳入（原来只有 5 个臂关节，
    # 夹爪疯狂开合完全不受罚）。这一项罚的是关节实际速度，和 action_rate 互补：
    # action_rate 管指令的跳变，joint_vel 管真实的高频运动。
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=EGGTART_ARM_JOINT_NAMES + [EGGTART_GRIPPER_JOINT_NAME]
            )
        },
    )

    # 关节到极限位置的惩罚（用 Isaac Lab 自带项）
    # joint_pos_limits 罚的是超出 **软限位** 的部分：软限位 = soft_joint_pos_limit_factor(0.95)
    # × URDF 硬限位，也就是行程最外侧 5% 那一圈。关节在中间区域时该项恒为 0，
    # 只有压到边界才产生代价，所以不会干扰正常运动。
    # 只作用在机械臂 + 夹爪上：四个轮子关节是 continuous（URDF 里没有 limit），无极限可言。
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,  # 从 -0.5 提高到 -1.0，让机械臂更怕压限位
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=EGGTART_ARM_JOINT_NAMES + [EGGTART_GRIPPER_JOINT_NAME]
            )
        },
    )

    # 底盘速度惩罚（治"追到目标后绕着转"）
    # 现有的 action_rate / joint_vel 治不了这个毛病：
    #   - action_rate 罚动作**变化量**，匀速绕圈时动作近似恒定，几乎不花钱；
    #   - joint_vel 只覆盖机械臂关节，而且 HolonomicBaseAction 是直接写根节点速度的，
    #     轮子关节速度恒为 0，基于关节的惩罚根本约束不到底盘。
    # 绕圈的根因是奖励退化：base_approach 在"距离==standoff"最大、base_facing 在"对准"最大，
    # 这两个条件在半径 0.35 m 的**整个圆周**上同时满足，沿切向漂移不损失奖励。
    # 该项带"到位 × 对准"门控，只在停好之后才罚速度，不和 base_approach 对抗
    # （目标每 2~4 s 重随机速度，底盘本来就得重新追）。详见 rewards.py 的 base_velocity_l2。
    base_vel = RewTerm(
        func=mdp.base_velocity_l2,
        weight=-0.05,
        params={
            "standoff": 0.35,  # 与 base_approach 保持一致
            "arrive_tol": 0.15,
            "align_tol": 0.6,  # 与 base_facing 的 std 保持一致
            "ang_vel_scale": 0.3,
            "forward_axis": EGGTART_BASE_FORWARD_AXIS,
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
            "base_cfg": SceneEntityCfg("robot", body_names=EGGTART_WHEEL_JOINT_BODY_REGEX),
        },
    )


@configclass
class CurriculumCfg:
    """课程学习配置：按训练步数分阶段打开各奖励项

    时间表用 ``common_step_counter``（每次 env.step 加 1，与 num_envs 无关）计时。
    本项目 num_steps_per_env = 24，所以 step = 迭代数 × 24：

        阶段 1 (iter 0-1000,   step 0-24000)  : 只学底盘接近 + 朝向（让底盘导航习惯固化）
        阶段 2 (iter 1000-1500, step 24000-36000): 加入末端执行器到达（底盘+机械臂协同）
        阶段 3 (iter 1500-2000, step 36000-48000): 完整任务（抓取 + 提起）
        阶段 4 (iter 2000+,     step 48000+)    : 加入回收奖励

    上面 RewardsCfg 里写的权重是**最终阶段**的值，课程会在前期把还没到的阶段压成 0。
    改 num_steps_per_env 要同步改这里的阈值。
    每一项的当前权重会记到 TensorBoard 的 ``Curriculum/<name>``，可以直接看切换时机。
    """

    # 底盘两项全程开启，是后面所有阶段的基础
    # 阶段2后降低权重，给末端探索让路
    base_approach_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_approach", "schedule": [(0, 2.5), (24000, 1.5)]},  # 阶段2后降低，避免过度约束
    )
    base_facing_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_facing", "schedule": [(0, 1.2), (24000, 0.8)]},  # 阶段2后降低
    )
    # 末端执行器：阶段 2 打开（延长到 24000 步，让底盘先学稳 1000 迭代）
    # 使用渐进式权重增长，而非阶跃，让策略平滑过渡
    ee_reach_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_reach", "schedule": [(0, 0.0), (24000, 0.5), (30000, 2.0), (36000, 5.0)]},  # 渐进式：0→0.5→2.0→5.0
    )
    ee_distance_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_distance", "schedule": [(0, 0.0), (24000, -0.1), (30000, -0.3)]},  # 渐进式，权重较小避免驱动底盘
    )
    ee_orientation_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_orientation", "schedule": [(0, 0.0), (36000, 2.5)]},  # 延后到 36000，先让末端靠近再管朝向
    )

    # 抓取：阶段 3 打开（36000 步 = 1500 迭代后，确保底盘+末端协同已稳定）
    # 引导惩罚项，用于引导夹爪正确闭合
    gripper_close_guide_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "gripper_close_guide", "schedule": [(0, 0.0), (48000, -1.5)]},  # 24000→36000
    )
    # "提前闭爪"的惩罚，但权重降低避免过度抑制
    gripper_early_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "gripper_early", "schedule": [(0, 0.0), (48000, -0.5)]},  # 24000→36000
    )
    # 引导奖励：接近且闭合时鼓励提起末端（稠密奖励，完整动作序列）
    ee_lift_guide_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_lift_guide", "schedule": [(0, 0.0), (48000, 15.0)]},  # 7.0→15.0，与 grasp 同量级
    )
    # 稀疏抓取奖励：基于提起高度判定
    grasp_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "grasp", "schedule": [(0, 0.0), (48000, 30.0)]},  # 15.0→30.0，大幅提高
    )

    # 渐进式提升奖励：比稀疏奖励更早激活，提供密集引导
    target_lift_progress_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "target_lift_progress", "schedule": [(0, 0.0), (36000, 20.0)]},  # 阶段3开始
    )

    # 机械臂舒适度：全程开启，从一开始就引导底盘停在让机械臂舒服的位置
    # 阶段2后降低权重，允许机械臂为了够目标而伸展
    arm_comfort_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "arm_comfort", "schedule": [(0, 1.0), (24000, 0.3)]},  # 阶段2后大幅降低，给末端探索让路
    )

    # 抓取姿态引导：阶段2开始激活，引导机械臂在接近目标时摆出正确姿态
    grasp_posture_guide_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "grasp_posture_guide", "schedule": [(0, 0.0), (30000, 8.0)]},  # 4.0→8.0，进一步提高
    )

    # 关节限位惩罚全程开启：从一开始就不该往限位上顶
    joint_limits_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "joint_limits", "schedule": [(0, -1.0)]},  # -0.5→-1.0
    )
    
    # 底盘速度惩罚： 一开始底盘还不会走，就罚它动会拖慢学习；等它大致学会接近了再要求"停住"。
    base_vel_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_vel", "schedule": [(0, 0.0), (9000, -0.5)]},
    )
    

@configclass
class TerminationsCfg:
    """MDP 终止条件配置"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_tipped = DoneTerm(func=mdp.base_tipped, params={"min_up_proj": 0.5})


##
# 环境配置
##
@configclass
class MobileGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Eggtart 移动抓取环境的基础配置（带移动目标）"""

    # 场景
    scene: MobileGraspSceneCfg = MobileGraspSceneCfg(num_envs=2048, env_spacing=3.0)
    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 通用设置
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (4.0, 4.0, 3.0)
        # 仿真设置
        self.sim.dt = 1.0 / 120.0


@configclass
class MobileGraspEnvStaticCfg(MobileGraspEnvCfg):
    """Eggtart 移动抓取环境配置（静止目标版本）

    继承基础配置，但禁用目标的随机移动：
    - 初始速度设为 0
    - 禁用周期性速度随机化（通过设置超长间隔）
    """

    def __post_init__(self):
        super().__post_init__()

        # 覆盖目标初始速度为 0（静止）
        self.events.reset_target.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}

        # 禁用周期性速度随机化：设置超长间隔（1小时），实际episode只有10秒
        self.events.randomize_target_velocity.interval_range_s = (3600.0, 3600.0)
