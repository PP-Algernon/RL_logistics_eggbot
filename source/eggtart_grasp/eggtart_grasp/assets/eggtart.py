"""Eggtart 移动机械臂的 Isaac Lab ``ArticulationCfg`` 配置

机器人规格（从 ``urdf/robot.urdf`` 读取）：
  - 移动底盘: 4 个麦轮 -> 连续关节 ``wheel_001_joint`` .. ``wheel_004_joint``
        (力矩限制 10 Nm, 速度限制 5 rad/s)
  - 5 轴机械臂: 旋转关节 ``link_001_joint`` .. ``link_005_joint`` (力矩 10 Nm, 速度 5 rad/s)
  - 夹爪: ``end_effector_joint`` (1 自由度舵机轴，建模为夹爪开合自由度)
  - 末端执行器刚体: ``end_effector``;  底盘刚体: ``base_link``

轮子几何（各 body 在 base_link 系下的实测坐标，仿真里打印得到）：
    wheel_001 (-0.4137, -0.3750)   wheel_002 (-0.2578, -0.3792)
    wheel_003 (-0.2524, -0.1894)   wheel_004 (-0.4083, -0.1848)

    底盘前进方向是 **Y**，不是 X —— 四个 ``wheel_00*_joint`` 的 ``axis`` 都沿 ±X，
    轮子绕 X 转，所以滚动方向是 Y。因此：
        轮距（左右，沿 X）≈ 0.161 m    轴距（前后，沿 Y）≈ 0.194 m
        轮子半径 r ≈ 0.043 m（轮轴高度）
    按 -Y 为前，则 FL = wheel_002, FR = wheel_001, RL = wheel_003, RR = wheel_004。

    注意：麦轮运动学的实际参数已在 actions.py 中实测并硬编码；上面的值供参考。
    ``MecanumBaseAction`` 已弃用（现在用 HolonomicBaseAction），其 FL/FR/RL/RR
    顺序常量 :data:`EGGTART_WHEEL_JOINT_NAMES` 沿用旧的 +X 为前的假设，
    如果要重新启用麦轮模式需要先复核这个顺序。

重要提示：
    URDF 中的轮子是简单圆柱体 —— 没有麦轮滚子几何，因此不会物理仿真真实的全向侧滑。
    :class:`HolonomicBaseAction` 直接控制底盘根节点体速度，实现理想全向移动，
    这是可训练的起点，不是高保真模型。如果需要忠实的侧向运动，可以添加滚子网格
    或将底盘改为全向根速度驱动。
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# ---------------------------------------------------------------------------
# Joint / body name groups (imported by the env + action configs)
# ---------------------------------------------------------------------------
EGGTART_ARM_JOINT_NAMES = [
    "link_001_joint",
    "link_002_joint",
    "link_003_joint",
    "link_004_joint",
    "link_005_joint",
]
EGGTART_GRIPPER_JOINT_NAME = "end_effector_joint"
# Ordered FL, FR, RL, RR for the Mecanum kinematics in MecanumBaseAction.
EGGTART_WHEEL_JOINT_NAMES = [
    "wheel_003_joint",  # front-left
    "wheel_002_joint",  # front-right
    "wheel_004_joint",  # rear-left
    "wheel_001_joint",  # rear-right
]
EGGTART_EE_BODY_NAME = "end_effector"
EGGTART_BASE_BODY_NAME = "base_link"

# 真实抓取点（两爪夹持面的开口中心）相对 ``link_005`` body 原点的偏移，
# 在 end_effector 局部系下，单位米。
EGGTART_EE_GRASP_OFFSET = (-0.00092977, -0.0237725, -0.05601)
# 
EGGTART_EE_GRASP_DERECT_OFFSET = (0.00002977, -0.0305, -0.0701)

# 四个轮子 body 的名字正则，用来算底盘几何中心。
# **重要**：base_link 原点不在车身上。仿真里实测（各 body 在 base_link 系下的坐标）：
#     wheel_001 (-0.4137, -0.3750)   wheel_002 (-0.2578, -0.3792)
#     wheel_003 (-0.2524, -0.1894)   wheel_004 (-0.4083, -0.1848)
#     -> 轮心 = (-0.333, -0.282)，距原点 0.435 m
# 所以底盘位置/方位相关的奖励必须用轮心，不能用 root_pos_w
EGGTART_WHEEL_JOINT_BODY_REGEX = "wheel_.*"

# 底盘"工作面"在 base_link 系中的方向（机械臂伸出的那一侧）。
# 这不是 +X：CAD 导出的 base_link 坐标轴决定了实际朝向。
# 依据（URDF 推导 + 仿真实测 body 坐标，两者一致）：
#   - 四个 wheel_00*_joint 的 axis 都是 ±X -> 轮子绕 X 转，滚动前进方向是 Y
#   - 轮心 (base_link 系) = (-0.333, -0.282, 0.042)
#   - 零位姿下末端执行器 = (-0.315, -0.474, 0.280)，相对轮心的水平方向 = (0.094, -0.996)
#   - 机械臂基座 link_001 相对轮心 = (-0.001, -0.119, 0.144)，同样偏 -Y
# 重新导出 URDF 后需要复核此值。
EGGTART_BASE_FORWARD_AXIS = (0.0, -1.0, 0.0)

# ---------------------------------------------------------------------------
# Path resolution -- URDF lives next to this file under ``urdf/``
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EGGTART_URDF_PATH = os.path.join(_THIS_DIR, "urdf", "robot.urdf")

# ---------------------------------------------------------------------------
# Nominal joint positions (rad)
# ---------------------------------------------------------------------------
# 夹爪的开合角。**必须在 end_effector_joint 的硬限位 (-0.2, 1.57) 之内**。
# 注意 CLOSED = -0.2 正好压在下限上：任何"要求夹爪角度低于 -0.2"的判据都不可能成立
# （见 mobile_grasp_env_cfg.py 里 GRIPPER_CLOSED_THRESHOLD 的说明）。
EGGTART_GRIPPER_OPEN = 1.0
EGGTART_GRIPPER_CLOSED = -0.2  # = 硬下限，闭到底

# **重要**：nominal 姿态必须落在软限位（0.95 × 硬限位）之内，否则 joint_pos_limits
# 惩罚项在默认姿态下就恒为正，机械臂被持续往内推，和任务奖励对抗。
# 而且 reset_joints_by_scale 是按 (0.9, 1.1) **乘**默认值来随机的，所以：
#   - 值为 0 的关节缩放后仍是 0 —— 对 link_004 这种软限位区间 (-3.061, -0.079)
#     不含 0 的关节，0 是恒定违规的（旧配置就是这个 bug）；
#   - 值必须让 0.9×q 和 1.1×q 都还在软限位内。
# 下面的值已按新 URDF 限位（2026-08-30 更新）核算过，留了 15% 区间余量。
EGGTART_NOMINAL_JOINT_POS = {
    "link_001_joint": 0.0,      # soft (-1.491, +1.491)，0 合法
    "link_002_joint": -0.66,    # soft (-3.822, -0.098)，不能用 -0.1（1.1×=-0.11 越界在边上）
    "link_003_joint": 0.6,      # soft (+0.079, +3.061)
    "link_004_joint": -0.53,    # soft (-3.061, -0.079)，**不能是 0**
    "link_005_joint": 0.0,      # soft (-1.491, +1.491)，0 合法
    "end_effector_joint": EGGTART_GRIPPER_OPEN,  # soft (-0.156, +1.526)
    "wheel_.*_joint": 0.0,
}

# ---------------------------------------------------------------------------
# ArticulationCfg
# ---------------------------------------------------------------------------
EGGTART_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=EGGTART_URDF_PATH,
        # USD cache is written next to the URDF on first conversion.
        usd_dir=os.path.join(_THIS_DIR, "urdf", "usd_cache"),
        fix_base=False,  # mobile base -- free-floating root
        merge_fixed_joints=True,
        # Default converter drive gains; per-joint control is set by the actuators below.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=40.0,
                damping=2.0,
            ),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.06),  # base spawn height; TODO(tune): match wheel radius so wheels rest on ground
        joint_pos=EGGTART_NOMINAL_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 5-axis arm: position-controlled (stiff PD).
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["link_00[1-5]_joint"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=40.0,
            damping=2.0,
        ),
        # Gripper servo axis: position-controlled (open/close).
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["end_effector_joint"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=30.0,
            damping=1.5,
        ),
        # Mecanum wheels: velocity-controlled (stiffness=0 -> pure velocity tracking).
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_00[1-4]_joint"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=0.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
