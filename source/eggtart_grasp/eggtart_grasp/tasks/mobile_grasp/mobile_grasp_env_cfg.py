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
    EGGTART_GRIPPER_OPEN,
    EGGTART_WHEEL_JOINT_BODY_REGEX,
    EGGTART_WHEEL_JOINT_NAMES,
    EGGTART_GRASP_JOINT_POS,
)

# ---------------------------------------------------------------------------
# Tunable task constants
# ---------------------------------------------------------------------------
# 抓取成功判定：目标物体被提起到的高度阈值
LIFT_HEIGHT_THRESHOLD = 0.12  # m (目标质心高度，降低到 12cm)

# 提起后必须保持在高度阈值以上这么久才算稳定抓取
LIFT_DWELL_TIME = 0.1  # s

# 目标移动速度范围（用于随机初始速度和周期性速度变化）
TARGET_VELOCITY_RANGE = 0.10  # m/s (±range，降低到原来的40%)

# 抓取点落在这个距离内算"到达目标"
GRASP_REACH_THRESHOLD = 0.1      # m (放宽让policy更容易触发grasp)

# 夹爪关节低于此角度算"闭合"
GRIPPER_CLOSED_THRESHOLD = 0.35

# 抓取点必须在 GRASP_REACH_THRESHOLD 内**连续停留**这么久，闭爪才算有效抓取
GRASP_DWELL_TIME = 0.2  # s (降低难度，让policy先学会基本动作)

# 课程学习阶段划分：阶段1/2/3的起始步数
CURRICULUM_STAGE1_START_ITER = 0
CURRICULUM_STAGE2_START_ITER = 750
CURRICULUM_STAGE3_START_ITER = 2000
CURRICULUM_STAGE4_START_ITER = 4000  

# 逆向课程学习：三阶段目标初始位置和行为控制
# 阶段 1 (step 0-36000, 1500 iter): 固定正前方 + 主动靠近夹爪（最简单）
# 阶段 2 (step 36000-96000, 4000 iter): 固定正前方 + 静止（中等难度）
# 阶段 3 (step 96000+, 4000+ iter): 随机位置 + 静止（完整任务）
ANTI_CURRICULUM_STAGE1_START_ITER = 0
ANTI_CURRICULUM_STAGE2_START_ITER = 4000
ANTI_CURRICULUM_STAGE3_START_ITER = 6000

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
            size=(0.032, 0.032, 0.032),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  # 恢复重力，允许物体落地和被提起
                linear_damping=0.8,      # 增加阻尼，降低滑动
                angular_damping=0.8,     # 增加角阻尼，稳定旋转
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 接触摩擦由材质决定，刚体阻尼只会衰减运动速度。
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
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
        func=mdp.reset_target_curriculum,
        mode="reset",
        params={
            # 阶段1: 固定在正前方
            "stage1_pose_range": {"x": (0.6, 0.7), "y": (-0.05, 0.05), "z": (0.015, 0.020)},
            # 阶段2: 固定在正前方（同阶段1位置）
            "stage2_pose_range": {"x": (0.6, 0.7), "y": (-0.05, 0.05), "z": (0.015, 0.020)},
            # 阶段3: 随机位置
            "stage3_pose_range": {"x": (0.4, 1.2), "y": (-0.6, 0.6), "z": (0.015, 0.020)},
            "velocity_range": {
                "x": (-TARGET_VELOCITY_RANGE, TARGET_VELOCITY_RANGE),
                "y": (-TARGET_VELOCITY_RANGE, TARGET_VELOCITY_RANGE),
            },
            "robot_cfg": SceneEntityCfg("robot"),
            "stage2_start_step": ANTI_CURRICULUM_STAGE2_START_ITER*24,  # 阶段2开始：1500 iter
            "stage3_start_step": ANTI_CURRICULUM_STAGE3_START_ITER*24,  # 阶段3开始：4000 iter（延后）
            "asset_cfg": SceneEntityCfg("target"),
        },
    )
    # 阶段1专用：目标主动靠近夹爪（辅助学习精准抓取）
    target_approach_stage1 = EventTerm(
        func=mdp.target_approach_ee_direction,
        mode="interval",
        interval_range_s=(0.1, 0.2),  # 高频更新，实时跟踪末端
        params={
            "approach_speed": 0.05,  # 5 cm/s 缓慢靠近
            "activation_distance": 0.15,  # 抓取点进入 15cm 内才触发
            "stage1_end_step": ANTI_CURRICULUM_STAGE2_START_ITER*24,  # 阶段1结束步数：1500 iter
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
            "direction_offset": EGGTART_EE_GRASP_DERECT_OFFSET,
        },
    )

    # 禁用随机移动，让模型专注于精准抓取
    randomize_target_velocity = EventTerm(
        func=mdp.randomize_target_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "velocity_range": {
                "x": (-TARGET_VELOCITY_RANGE, TARGET_VELOCITY_RANGE),
                "y": (-TARGET_VELOCITY_RANGE, TARGET_VELOCITY_RANGE),
            },
            "stage2_start_step": 999999999,  # 永远不激活随机移动
            "stage3_start_step": 999999999,
            "asset_cfg": SceneEntityCfg("target"),
        },
    )


@configclass
class RewardsCfg:
    """简化的奖励项配置"""

    # ========== 阶段 1: 底盘导航（全程） ==========
    base_approach = RewTerm(
        func=mdp.base_to_target_xy_tanh,
        weight=2.5,
        params={
            "std": 0.5,
            "standoff": 0.4,
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
            "base_cfg": SceneEntityCfg("robot", body_names=EGGTART_WHEEL_JOINT_BODY_REGEX),
            "arm_link1_cfg": SceneEntityCfg("robot", body_names="link_001"),
        },
    )
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
    arm_comfort = RewTerm(
        func=mdp.arm_comfort,
        weight=0.3,  # 降低权重，给探索空间
        params={
            "std": 1.0,
            "arm_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES),
        },
    )


    # ========== 阶段 2: 机械臂到达 ==========
    # 双核 tanh：宽核(0.3)保远场引导，窄核(0.08)给近场精修。
    # 单核 std=0.2 时 0.1m 处梯度已很平，实测策略推不动最后 10cm——
    # 上一次训练抓取点最好也只到 0.099m，卡在 closure 门控边界外拿不到分，
    # 随后（2900 迭代起）主动放弃伸手退回刷底盘分，ee_reach 从 2.71 崩到 0.29。
    ee_reach = RewTerm(
        func=mdp.ee_to_target_tanh,
        weight=3.0,
        params={
            "std": 0.3,
            "narrow_std": 0.08,
            "wide_weight": 0.6,
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    # 近场精修（阶段 3 才由 curriculum 拉起权重，见 ee_precision_sched）
    # ee_reach 的最窄核 std=0.08 在最后 2cm 内已经饱和：d 从 0.02 收到 0.005
    # 只涨 0.019 分，被 action_rate / joint_vel 的噪声淹没，策略没动力再往里挤。
    # 这一项用 std=0.02 的高斯核把陡峭区间挪到 0 附近（同样一段涨 0.36），
    # 并在 support 处连续归零，5cm 外恒为 0，不改变已训好的远场引导形状。
    ee_precision = RewTerm(
        func=mdp.ee_to_target_precision,
        weight=5.0,
        params={
            "std": 0.02,
            # 与 GRASP_REACH_THRESHOLD 对齐：精修的作用域正好是"算到达"的那个球。
            # 需满足 support >= 2*std（0.05 >= 0.04 ✓），否则截断点落在核还很陡处，
            # 归一化会把近场梯度压平。
            "support": GRASP_REACH_THRESHOLD,
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    ee_distance = RewTerm(
        func=mdp.ee_to_target_distance_l2,
        weight=-0.3,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    # 保留：末端朝向
    ee_orientation = RewTerm(
        func=mdp.ee_grasp_direction_alignment,
        weight=2.5,
        params={
            "std": 0.5,
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
            "grasp_direction_offset": EGGTART_EE_GRASP_DERECT_OFFSET,
        },
    )
    # 保留：抓取姿态引导
    grasp_posture_guide = RewTerm(
        func=mdp.grasp_posture_guide,
        weight=4.0,  # 提高权重
        params={
            "target_joint_pos": EGGTART_GRASP_JOINT_POS,
            "std": 1.0,
            "reach_threshold": GRASP_REACH_THRESHOLD * 3,
            "arm_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES),
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )

    # ========== 阶段 3: 抓取 ==========
    # 闭爪基础奖励（接近+闭爪就给，纯正向）
    # gate_blend 由 curriculum 控制：前期=0（抓空也给分，大胆闭），后期渐进到 1（必须夹住）
    gripper_closure_reward = RewTerm(
        func=mdp.gripper_closure_bonus,
        weight=15.0,
        params={
            # 10cm → 15cm。上次训练抓取点最好只到 0.099m，在 R=0.10 下
            # proximity=sqrt(1-0.099/0.10)≈0.10，闭合满分也只得 1.5 分，
            # 被 ee_reach(w=5) 完全淹没，closure 全程 5999 点恒为 0。
            # R=0.15 让同样的 0.099m 得 proximity=0.583（满分 8.75），
            # 足以压过"退回原地刷 base_approach"的局部最优。
            # 放宽带来的"稍远处也能拿分"由 gate_blend 后期收紧（必须真夹住）来补。
            "reach_threshold": 0.15,
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
            "gripper_full_close_pos": GRIPPER_CLOSED_THRESHOLD,
            "gripper_open_pos": EGGTART_GRIPPER_OPEN,
            # 门控参数（由 gripper_gate_blend_sched 动态控制）
            "gate_blend": 0.0,  # 初始值，会被 curriculum 覆盖
            "gate_pos_min": 0.18,
            "gate_pos_max": 0.32,
            "gate_effort_threshold": 0.1,
        },
    )
    # 提升奖励
    # gripper_open_pos 必须是**张开**端的关节角（1.0），不是闭合端。
    # 之前误传 0.05（闭合端）导致 span = 0.05-0.35 = -0.3 为负，闭合门控整体反号：
    # 夹爪全张开时 closure=1.0 拿满分、闭合时 closure=0.0，权重 30 的这一项
    # 实际在奖励"张开"，压过权重 15 的 gripper_closure_reward，
    # 策略于 ~1735 迭代收敛到永久张开（TensorBoard 里 closure 此后恒为 0）。
    target_lift_progress = RewTerm(
        func=mdp.target_lift_progress,
        weight=30.0,
        params={
            "target_height": LIFT_HEIGHT_THRESHOLD,
            "std": 0.05,
            "target_cfg": SceneEntityCfg("target"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "gripper_open_pos": EGGTART_GRIPPER_OPEN,
        },
    )
    # 稀疏抓取奖励
    grasp = RewTerm(
        func=mdp.grasp_bonus_lift,
        weight=30.0,
        params={
            "lift_height_threshold": LIFT_HEIGHT_THRESHOLD,
            "lift_dwell_time": LIFT_DWELL_TIME,
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # ========== 约束项（全程） ==========
    # 治 bang-bang 抖动：罚动作指令的跳变。
    # 臂+爪 6 维在 ±1 间来回跳时 action_rate_l2 ≈ 24，-0.02 让代价约 0.48，

    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.02
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES)},
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=EGGTART_ARM_JOINT_NAMES + [EGGTART_GRIPPER_JOINT_NAME]
            )
        },
    )
    base_vel = RewTerm(
        func=mdp.base_velocity_l2,
        weight=-0.5,
        params={
            "standoff": 0.5,  # 与 base_approach 保持一致
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
    """课程学习配置3阶段
    阶段 1 (0-18000 步): 底盘导航
    阶段 2 (18000-36000 步): 机械臂到达
    阶段 3 (36000+ 步): 抓取
    """
    # ========== 阶段 1: 底盘导航 ==========
    base_approach_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_approach", "schedule": [(CURRICULUM_STAGE1_START_ITER*24, 2.5)]},
    )
    base_facing_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_facing", "schedule": [(CURRICULUM_STAGE1_START_ITER*24, 1.2)]},
    )
    arm_comfort_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "arm_comfort", "schedule": [(0, 0.3), (CURRICULUM_STAGE2_START_ITER*24, 0.1)]},
    )

    # ========== 阶段 2: 机械臂到达 ==========
    ee_reach_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_reach", "schedule": [(0, 0.0), (CURRICULUM_STAGE2_START_ITER*24, 3.0)]},
    )
    ee_orientation_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_orientation", "schedule": [(0, 0.0), (CURRICULUM_STAGE2_START_ITER*24, 2.5)]},
    )
    grasp_posture_guide_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "grasp_posture_guide", "schedule": [(0, 0.0), (CURRICULUM_STAGE2_START_ITER/2*24, 4.0)]},
    )
    ee_distance_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_distance", "schedule": [(0, 0.0), (CURRICULUM_STAGE3_START_ITER*24, -0.6)]},
    )

    # ========== 阶段 3: 抓取 ==========
    ee_precision_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_precision", "schedule": [(0, 0.0), (CURRICULUM_STAGE3_START_ITER*24, 10.0)]},
    )
    gripper_closure_reward_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "gripper_closure_reward", "schedule": [(0, 0.0), (CURRICULUM_STAGE3_START_ITER*24, 15.0)]},
    )
    # 渐进式"夹住物体"门控：blend 0（抓空也给分）→ 1（必须真夹住）
    # 上次配置在 84000 步（3500 迭代）直接跳到 1，但 closure 当时恒为 0——等于策略还没学会闭爪就先上严格门控。
    # 这次分三段爬升，且起点推后到 96000 步（4000 迭代），给 closure 生效后留足 36000步（1500 迭代）的自由探索期。
    gripper_gate_blend_sched = CurrTerm(
        func=mdp.reward_param_schedule,
        params={
            "term_name": "gripper_closure_reward",
            "param_name": "gate_blend",
            "schedule": [(0, 0.0), (96000, 0.3), (120000, 0.6), (144000, 1.0)],
        },
    )
    target_lift_progress_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "target_lift_progress", "schedule": [(0, 0.0), (CURRICULUM_STAGE3_START_ITER*24, 30.0)]},
    )
    # 稀疏抓取奖励
    grasp_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "grasp", "schedule": [(0, 0.0), (CURRICULUM_STAGE3_START_ITER*24, 30.0)]},
    )

    # ========== 约束项 ==========
    joint_limits_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "joint_limits", "schedule": [(0, -0.1)]},
    )
    joint_vel_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "joint_vel", "schedule": [(0, -0.001), (CURRICULUM_STAGE2_START_ITER*24, -0.0001)]},
    )
    action_rate_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "action_rate", "schedule": [(0, -0.005), (CURRICULUM_STAGE2_START_ITER*24, -0.001)]},
    )
    base_vel_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_vel", "schedule": [(0, -0.5), (CURRICULUM_STAGE4_START_ITER*24, -1.0)]},
    )

@configclass
class BCCurriculumCfg:
    """BC 预训练后的路线 B 奖励课程，与普通环境共用 RewardsCfg。

    第 0 步即启用接近、姿态、举升、抓取和两项正则，不重新等待导航课程。
    其余奖励保留定义，权重设为零；不继承普通课程，避免后期重新启用。
    目标逆向课程仍由 EventCfg 按 ANTI_CURRICULUM_* 阈值独立控制。
    """

    base_approach_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_approach", "schedule": [(0, 1.0)]},
    )

    base_facing_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_facing", "schedule": [(0, 0.0)]},
    )

    arm_comfort_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "arm_comfort", "schedule": [(0, 0.0)]},
    )

    ee_reach_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_reach", "schedule": [(0, 2.0)]},
    )

    ee_orientation_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_orientation", "schedule": [(0, 0.0)]},
    )

    grasp_posture_guide_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "grasp_posture_guide", "schedule": [(0, 1.5)]},
    )

    ee_distance_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_distance", "schedule": [(0, 0.0)]},
    )

    ee_precision_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_precision", "schedule": [(0, 0.0)]},
    )

    gripper_closure_reward_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "gripper_closure_reward", "schedule": [(0, 0.0)]},
    )

    target_lift_progress_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "target_lift_progress", "schedule": [(0, 10.0)]},
    )

    grasp_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "grasp", "schedule": [(0, 30.0)]},
    )

    joint_limits_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "joint_limits", "schedule": [(0, -0.3)]},
    )

    joint_vel_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "joint_vel", "schedule": [(0, 0.0)]},
    )

    action_rate_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "action_rate", "schedule": [(0, -0.01)]},
    )

    base_vel_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_vel", "schedule": [(0, 0.0)]},
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

