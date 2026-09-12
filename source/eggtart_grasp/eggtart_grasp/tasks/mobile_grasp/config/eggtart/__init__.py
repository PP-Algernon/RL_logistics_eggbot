"""Register the standard PPO and BC/PPO Eggtart environments."""

import gymnasium as gym

from . import agents, grasp_env_cfg


def _register(task_id, cfg):
    gym.register(
        id=task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": cfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:EggtartMobileGraspPPORunnerCfg",
        },
    )


_register("Isaac-Mobile-Grasp-Eggtart-v0", "eggtart_grasp.tasks.mobile_grasp.config.eggtart.grasp_env_cfg:EggtartMobileGraspEnvCfg")
_register("Isaac-Mobile-Grasp-Eggtart-BCPPO-v0", "eggtart_grasp.tasks.mobile_grasp.config.eggtart.grasp_env_cfg:EggtartMobileGraspEnvBCPPOCfg")
