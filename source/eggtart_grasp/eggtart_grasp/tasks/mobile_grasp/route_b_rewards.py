"""Route B Reward Configuration for BC + PPO Fine-tuning

This module provides a simplified reward configuration for Route B:
- BC pretraining provides initial exploration capability
- Simplified dense rewards guide fine-tuning (only 4-5 core terms)
- Reward weights are reduced by ~50% compared to full dense rewards
- Curriculum is simplified (no reward weight scheduling, only target distribution)

Usage:
    In your grasp_env_cfg.py, replace RewardsCfg with RouteBRewardsCfg
    or create a variant environment that uses this configuration.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import eggtart_grasp.tasks.mobile_grasp.mdp as mdp
from eggtart_grasp.assets.eggtart import (
    EGGTART_ARM_JOINT_NAMES,
    EGGTART_BASE_FORWARD_AXIS,
    EGGTART_EE_GRASP_OFFSET,
    EGGTART_EE_GRASP_DERECT_OFFSET,
    EGGTART_GRIPPER_JOINT_NAME,
    EGGTART_GRIPPER_OPEN,
    EGGTART_WHEEL_JOINT_BODY_REGEX,
    EGGTART_GRASP_JOINT_POS,
)

# Import constants from base config
from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import (
    LIFT_HEIGHT_THRESHOLD,
    LIFT_DWELL_TIME,
    GRIPPER_CLOSED_THRESHOLD,
    GRASP_REACH_THRESHOLD,
)


@configclass
class RouteBRewardsCfg:
    """Route B Simplified Rewards (Configuration 1: Sparse Success + Minimal Dense Guidance)

    Keeps only 4-5 core reward terms with reduced weights:
    1. base_approach: Guide base toward target (weight: 2.5 -> 1.0)
    2. ee_reach: Guide end-effector to target (weight: 5.0 -> 2.0)
    3. grasp_posture_guide: Guide arm to grasp posture (weight: 4.0 -> 1.5)
    4. target_lift_progress: Reward lifting progress (weight: 30.0 -> 10.0)
    5. grasp: Sparse success reward (weight: 30.0, unchanged)

    All other dense rewards are disabled. BC warm-start provides initial exploration,
    so we don't need heavy dense shaping.
    """

    # ========== Core Guidance Terms ==========

    # 1. Base approach (navigation)
    base_approach = RewTerm(
        func=mdp.base_to_target_xy_tanh,
        weight=1.0,  # Reduced from 2.5
        params={
            "std": 0.5,
            "standoff": 0.4,
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
            "base_cfg": SceneEntityCfg("robot", body_names=EGGTART_WHEEL_JOINT_BODY_REGEX),
            "arm_link1_cfg": SceneEntityCfg("robot", body_names="link_001"),
        },
    )

    # 2. End-effector reach (arm extension)
    ee_reach = RewTerm(
        func=mdp.ee_to_target_tanh,
        weight=2.0,  # Reduced from 5.0 (considering both cores)
        params={
            "std": 0.3,
            "narrow_std": 0.08,
            "wide_weight": 0.6,
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )

    # 3. Grasp posture guidance (arm configuration)
    grasp_posture_guide = RewTerm(
        func=mdp.grasp_posture_guide,
        weight=1.5,  # Reduced from 4.0
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

    # 4. Lift progress (grasping and lifting)
    target_lift_progress = RewTerm(
        func=mdp.target_lift_progress,
        weight=10.0,  # Reduced from 30.0
        params={
            "target_height": LIFT_HEIGHT_THRESHOLD,
            "std": 0.05,
            "target_cfg": SceneEntityCfg("target"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "gripper_open_pos": EGGTART_GRIPPER_OPEN,
        },
    )

    # 5. Sparse success reward (unchanged)
    grasp = RewTerm(
        func=mdp.grasp_bonus_lift,
        weight=30.0,  # Keep original weight for success signal
        params={
            "lift_height_threshold": LIFT_HEIGHT_THRESHOLD,
            "lift_dwell_time": LIFT_DWELL_TIME,
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # ========== Minimal Regularization ==========
    # Keep these to prevent pathological behaviors, but with very low weights

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,  # Reduced from -0.02
    )

    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.3,  # Reduced from -0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES)},
    )


@configclass
class RouteBRewardsCfg_Aggressive:
    """Route B Configuration 2: More Aggressive (Closer to Pure Sparse)

    Only keeps:
    1. grasp: Sparse success reward
    2. ee_reach: Minimal temperature term to guide toward target

    This is closer to the paper's pure sparse formulation.
    Use this AFTER BC has proven to work well with Configuration 1.
    """

    # Sparse success
    grasp = RewTerm(
        func=mdp.grasp_bonus_lift,
        weight=30.0,
        params={
            "lift_height_threshold": LIFT_HEIGHT_THRESHOLD,
            "lift_dwell_time": LIFT_DWELL_TIME,
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # Minimal temperature term
    ee_reach = RewTerm(
        func=mdp.ee_to_target_tanh,
        weight=0.5,  # Very low weight, just a hint
        params={
            "std": 0.3,
            "narrow_std": 0.08,
            "wide_weight": 0.6,
            "ee_cfg": SceneEntityCfg("robot", body_names="link_005"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )

    # Minimal regularization
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.005,
    )


# ============================================================================
# Simplified Curriculum for Route B
# ============================================================================

from isaaclab.managers import CurriculumTermCfg as CurrTerm


@configclass
class RouteBCurriculumCfg:
    """Route B Simplified Curriculum

    Key difference from base curriculum:
    - NO reward weight scheduling (all rewards active from step 0)
    - ONLY target distribution curriculum (Stage 1 -> 2 -> 3)
    - BC warm-start means the policy already "knows how to grasp"
    """

    # Target distribution curriculum is handled in EventCfg (reset_target event)
    # No reward weight scheduling needed - all terms are active from the start
    pass


# ============================================================================
# Route B PPO Configuration (Lower Learning Rate)
# ============================================================================

@configclass
class RouteBPPOAlgorithmCfg:
    """PPO configuration for Route B fine-tuning

    Key changes:
    - Lower learning rate (1e-3 -> 3e-4) to prevent catastrophic forgetting
    - Keep other hyperparameters the same as base config
    """

    # Lowered from 1e-3 to prevent forgetting BC knowledge
    learning_rate = 3e-4

    # Standard PPO hyperparameters (same as base)
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 5
    num_mini_batches = 4
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


# ============================================================================
# Documentation
# ============================================================================

"""
USAGE INSTRUCTIONS:

1. Create a new environment config variant:

```python
@configclass
class MobileGraspEnvStaticRouteBCfg(MobileGraspEnvStaticCfg):
    '''Route B variant: BC + PPO with simplified rewards'''

    def __post_init__(self):
        super().__post_init__()

        # Replace rewards with Route B configuration
        from eggtart_grasp.tasks.mobile_grasp.route_b_rewards import (
            RouteBRewardsCfg,
            RouteBCurriculumCfg,
        )
        self.rewards = RouteBRewardsCfg()
        self.curriculum = RouteBCurriculumCfg()
```

2. Register the new environment in __init__.py

3. Train with BC checkpoint:

```bash
# Step 1: Collect demonstrations
./isaaclab.sh -p scripts/collect_demo.py \\
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \\
    --num_envs 2048 --max_steps 2000 --noise_scale 0.05

# Step 2: BC pretraining
./isaaclab.sh -p scripts/bc_pretrain.py \\
    --data datasets/eggtart_demo.hdf5 \\
    --output checkpoints/bc_pretrained.pt \\
    --epochs 100

# Step 3: PPO fine-tuning with Route B rewards
./isaaclab.sh -p scripts/rsl_rl/train.py \\
    --task Isaac-Mobile-Grasp-Eggtart-Static-RouteB-v0 \\
    --num_envs 2048 --max_iterations 4000 \\
    --resume --load_checkpoint checkpoints/bc_pretrained.pt
```

EXPECTED RESULTS:
- BC baseline (no fine-tuning): 50-70% success rate
- After PPO fine-tuning: 80%+ success rate (static target)
- Should see non-zero grasp rewards within first 100-200 iterations
  (contrast with from-scratch PPO which takes 1000+ iterations)

TROUBLESHOOTING:
- If BC success rate < 50%: Improve demonstration quality or collect more data
- If catastrophic forgetting (success rate drops after PPO): Lower learning rate further (1e-4)
- If no improvement over BC: Add back more dense guidance terms or check curriculum alignment
"""
