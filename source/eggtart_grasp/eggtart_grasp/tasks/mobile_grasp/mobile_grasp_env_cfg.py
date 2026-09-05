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
    # 逆向课程学习：三阶段目标初始位置和行为控制
    # 阶段 1 (step 0-36000, 1500 iter): 固定正前方 + 主动靠近夹爪（最简单）
    # 阶段 2 (step 36000-96000, 4000 iter): 固定正前方 + 静止（中等难度）
    # 阶段 3 (step 96000+, 4000+ iter): 随机位置 + 静止（完整任务）
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
            "stage2_start_step": 36000,  # 阶段2开始：1500 iter
            "stage3_start_step": 96000,  # 阶段3开始：4000 iter（延后）
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
            "stage1_end_step": 36000,  # 阶段1结束步数：1500 iter
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
            "direction_offset": EGGTART_EE_GRASP_DERECT_OFFSET,
        },
    )

    # 目标移动控制：全程静止（阶段1虽然有主动靠近，但初始速度为0）
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
            "standoff": 0.5,
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

    # ========== 阶段 2: 机械臂到达 ==========
    ee_reach = RewTerm(
        func=mdp.ee_to_target_tanh,
        weight=5.0,
        params={
            "std": 0.2,
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
    gripper_closure_reward = RewTerm(
        func=mdp.gripper_closure_bonus,
        weight=15.0,
        params={
            "reach_threshold": 0.08,  # 8cm 内
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
            "gripper_full_close_pos": GRIPPER_CLOSED_THRESHOLD,
        },
    )
    # 提升奖励
    target_lift_progress = RewTerm(
        func=mdp.target_lift_progress,
        weight=30.0,
        params={
            "target_height": LIFT_HEIGHT_THRESHOLD,
            "std": 0.05,
            "target_cfg": SceneEntityCfg("target"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "gripper_open_pos": 0.05,
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
    arm_comfort = RewTerm(
        func=mdp.arm_comfort,
        weight=0.3,  # 降低权重，给探索空间
        params={
            "std": 1.0,
            "arm_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES),
        },
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES)},
    )
    # 底盘速度惩罚：带"到位 × 对准"门控，只在停好之后才罚速度（治绕圈退化）
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
    """简化的课程学习配置：3阶段

    阶段 1 (0-18000 步): 底盘导航
    阶段 2 (18000-36000 步): 机械臂到达
    阶段 3 (36000+ 步): 抓取
    """

    # ========== 阶段 1: 底盘导航 ==========
    base_approach_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_approach", "schedule": [(0, 2.5)]},
    )
    base_facing_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_facing", "schedule": [(0, 1.2)]},
    )

    # ========== 阶段 2: 机械臂到达 ==========
    arm_comfort_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "arm_comfort", "schedule": [(0, 0.3)]},
    )
    ee_reach_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_reach", "schedule": [(0, 0.0), (18000, 5.0)]},
    )
    ee_distance_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_distance", "schedule": [(0, 0.0), (18000, -0.3)]},
    )
    ee_orientation_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_orientation", "schedule": [(0, 0.0), (24000, 2.5)]},
    )
    grasp_posture_guide_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "grasp_posture_guide", "schedule": [(0, 0.0), (24000, 4.0)]},
    )

    # ========== 阶段 3: 抓取 ==========
    # 正向奖励：闭爪就给分
    gripper_closure_reward_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "gripper_closure_reward", "schedule": [(0, 0.0), (36000, 15.0)]},
    )
    # 提升奖励
    target_lift_progress_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "target_lift_progress", "schedule": [(0, 0.0), (36000, 30.0)]},
    )
    # 稀疏抓取奖励
    grasp_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "grasp", "schedule": [(0, 0.0), (48000, 30.0)]},
    )

    # ========== 约束项（全程） ==========
    joint_limits_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "joint_limits", "schedule": [(0, -1.0)]},
    )
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
