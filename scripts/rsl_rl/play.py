"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Mobile-Grasp-Eggtart-v0", help="Name of the task.")
# --- 脚本化抓取接管（演示用）---
parser.add_argument(
    "--scripted_grasp",
    action="store_true",
    default=False,
    help="目标进入可夹范围但策略长时间不闭爪时，由脚本接管执行固定的闭爪+收回序列（仅演示，不影响训练）。",
)
parser.add_argument(
    "--grasp_patience", type=float, default=1.0, help="接管前的等待时间（秒），目标在范围内且策略未闭爪才计时。"
)
parser.add_argument("--grasp_close_time", type=float, default=0.3, help="脚本闭爪动作的持续时间（秒）。")
parser.add_argument("--grasp_hold_time", type=float, default=0.4, help="闭合后保持的时间（秒）。")
parser.add_argument("--grasp_retract_time", type=float, default=1.0, help="收回（臂回 nominal 姿态）的时间（秒）。")
parser.add_argument("--quiet_grasp", action="store_true", default=False, help="不打印脚本接管的状态切换。")
# --- 逆课程学习阶段选择（回放用）---
parser.add_argument(
    "--curriculum_stage",
    type=int,
    default=None,
    choices=[1, 2, 3],
    help="选择目标分布阶段（1/2/3）；位置范围和移动行为遵循环境配置。",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import importlib.metadata as metadata

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import parse_env_cfg

# Import extensions to set up environment tasks
import eggtart_grasp.tasks  # noqa: F401
from eggtart_grasp.assets.eggtart import EGGTART_EE_GRASP_OFFSET
from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import (
    GRASP_REACH_THRESHOLD,
    GRIPPER_CLOSED_THRESHOLD,
)
from eggtart_grasp.utils import ScriptedGraspOverride


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # migrate deprecated stochastic/init_noise_std to distribution_cfg for rsl_rl >= 5.0
    installed_version = metadata.version("rsl-rl-lib")
    # agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = cli_args.resolve_checkpoint(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # Evaluation uses the same config with observation noise disabled.
    env_cfg.observations.policy.enable_corruption = False

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 手动覆盖逆课程学习阶段（用于回放观察不同阶段的行为）
    if args_cli.curriculum_stage is not None:
        stage = args_cli.curriculum_stage
        print(f"[INFO] 手动设置逆课程学习阶段: {stage}")

        # Both reward curricula share the target curriculum configured in EventCfg.
        target_params = env_cfg.events.reset_target.params
        _locked_step = {
            1: 0,
            2: target_params["stage2_start_step"],
            3: target_params["stage3_start_step"],
        }[stage]
        env.unwrapped.common_step_counter = _locked_step
        print(f"  → 目标课程计数设为 {_locked_step}；目标行为遵循环境配置")
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Export the bundled rsl_rl 3.1.2 actor through Isaac Lab exporters.
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        policy=ppo_runner.alg.policy,
        normalizer=getattr(ppo_runner, "empirical_normalization", None),
        path=export_model_dir,
        filename="policy.pt"
    )
    export_policy_as_onnx(
        policy=ppo_runner.alg.policy,
        normalizer=getattr(ppo_runner, "empirical_normalization", None),
        path=export_model_dir,
        filename="policy.onnx"
    )

    # 脚本化抓取接管（仅演示；训练不受影响）
    grasp_override = None
    if args_cli.scripted_grasp:
        grasp_override = ScriptedGraspOverride(
            env.unwrapped,
            reach_threshold=GRASP_REACH_THRESHOLD,
            gripper_closed_threshold=GRIPPER_CLOSED_THRESHOLD,
            patience_time=args_cli.grasp_patience,
            close_time=args_cli.grasp_close_time,
            hold_time=args_cli.grasp_hold_time,
            retract_time=args_cli.grasp_retract_time,
            grasp_offset=EGGTART_EE_GRASP_OFFSET,
            verbose=not args_cli.quiet_grasp,
        )

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # episode 长度计数：用来在 episode 边界清掉脚本状态机
    prev_episode_len = env.unwrapped.episode_length_buf.clone()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # 脚本接管（该夹却不夹时）
            if grasp_override is not None:
                actions = grasp_override(actions)
            # env stepping
            obs, _, _, _ = env.step(actions)

            # 锁定课程阶段：如果用户指定了阶段，每次 step 后恢复 common_step_counter
            if args_cli.curriculum_stage is not None:
                env.unwrapped.common_step_counter = _locked_step

            # episode_length_buf 变小 = 该 env 刚被重置，清掉它的脚本状态
            if grasp_override is not None:
                cur_len = env.unwrapped.episode_length_buf
                reset_ids = torch.nonzero(cur_len < prev_episode_len, as_tuple=False).flatten()
                if reset_ids.numel() > 0:
                    grasp_override.reset(reset_ids)
                prev_episode_len = cur_len.clone()
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
