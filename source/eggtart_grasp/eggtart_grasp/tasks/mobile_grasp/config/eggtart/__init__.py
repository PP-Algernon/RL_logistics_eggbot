"""Gym environment registrations for the Eggtart mobile-grasp task."""

import gymnasium as gym

from . import agents, grasp_env_cfg

##
# Register Gym environments.
##

# Moving target version (original)
gym.register(
    id="Isaac-Mobile-Grasp-Eggtart-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": grasp_env_cfg.EggtartMobileGraspEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:EggtartMobileGraspPPORunnerCfg",
    },
)

# Static target version (easier training)
gym.register(
    id="Isaac-Mobile-Grasp-Eggtart-Static-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": grasp_env_cfg.EggtartMobileGraspEnvStaticCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:EggtartMobileGraspPPORunnerCfg",
    },
)

# Play/evaluation version
gym.register(
    id="Isaac-Mobile-Grasp-Eggtart-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": grasp_env_cfg.EggtartMobileGraspEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:EggtartMobileGraspPPORunnerCfg",
    },
)

