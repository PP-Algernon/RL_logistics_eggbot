"""Concrete Eggtart mobile-grasp environment config (train + play variants + static target)."""

from __future__ import annotations

from isaaclab.utils import configclass

from eggtart_grasp.assets.eggtart import EGGTART_CFG
from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import (
    MobileGraspEnvCfg,
    MobileGraspEnvStaticCfg,
)


@configclass
class EggtartMobileGraspEnvCfg(MobileGraspEnvCfg):
    """Training configuration for the Eggtart mobile-grasp task (moving target)."""

    def __post_init__(self):
        super().__post_init__()
        # attach the Eggtart robot to the scene
        self.scene.robot = EGGTART_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class EggtartMobileGraspEnvStaticCfg(MobileGraspEnvStaticCfg):
    """Training configuration for the Eggtart mobile-grasp task (static target)."""

    def __post_init__(self):
        super().__post_init__()
        # attach the Eggtart robot to the scene
        self.scene.robot = EGGTART_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class EggtartMobileGraspEnvCfg_PLAY(EggtartMobileGraspEnvCfg):
    """Reduced-scale config for visualisation / evaluation."""

    def __post_init__(self):
        super().__post_init__()
        # fewer, more-spaced environments and no observation noise
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
