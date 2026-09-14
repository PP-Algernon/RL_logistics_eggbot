"""Evaluate a BC/PPO checkpoint with complete episodes and continuous lift success.

Example (from isaac-lab/):
    ./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/evaluate.py" --headless \
        --checkpoint checkpoints/bc_pretrained.pt --num_episodes 1000 --num_envs 64
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

import cli_args  # selects the project's RSL-RL fork before importing the runner


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="Isaac-Mobile-Grasp-Eggtart-BCPPO-v0")
    parser.add_argument("--num_envs", type=int, default=64, help="并行环境数")
    parser.add_argument("--num_episodes", type=int, default=1000, help="评估的完整回合总数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum_stage", type=int, choices=(2, 3), default=2,
                        help="2：前方固定初始位置；3：随机初始位置。两者均关闭目标跟踪")
    parser.add_argument("--stochastic", action="store_true", help="按 checkpoint 保存的动作噪声采样；默认确定性动作")
    parser.add_argument("--observation_noise", action="store_true", help="开启训练中的观测噪声；默认关闭")
    parser.add_argument("--lift_height", type=float, help="成功举升的世界高度（米），默认任务 LIFT_HEIGHT_THRESHOLD")
    parser.add_argument("--dwell_time", type=float, help="连续举升保持时间（秒），默认 LIFT_SUCCESS_DWELL_TIME")
    parser.add_argument(
        "--success_timeout_s", "--episode_length_s", dest="episode_length_s", type=float,
        help="每个实例未成功时自动重置的时限（仿真秒）；默认当前任务时限，旧参数名仍可用",
    )
    parser.add_argument("--output", type=Path, help="JSON 报告路径；默认 checkpoint 同级 evaluation/ 目录")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.num_episodes <= 0:
        parser.error("--num_envs 和 --num_episodes 必须为正整数")
    for name in ("lift_height", "dwell_time", "episode_length_s"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name} 必须为有限正数")
    if args.output is not None and args.output.suffix.lower() != ".json":
        parser.error("--output 必须是 .json 文件路径")
    return args


def configure_episode_termination(cfg, *, height, dwell, lift_dwell_time, timeout_s=None):
    """Enable success and per-instance timeout resets on the evaluation config.

    Isaac Lab resets only the finished instances inside env.step(), including
    their robot/target states, episode clocks and stateful termination terms.
    """
    from isaaclab.envs.mdp import time_out
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
    from eggtart_grasp.tasks.mobile_grasp.mdp.terminations import LiftSuccess

    timeout = cfg.episode_length_s if timeout_s is None else timeout_s
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--success_timeout_s 必须是有限正数（仿真秒）")
    if not math.isfinite(dwell) or dwell <= lift_dwell_time:
        raise ValueError(f"--dwell_time 必须大于 LIFT_DWELL_TIME（{lift_dwell_time} s）")
    if timeout < dwell:
        raise ValueError(f"未成功重置时限 {timeout} s 小于成功保持时间 {dwell} s")
    cfg.episode_length_s = timeout
    # Install explicitly even if the selected training config disables time_out.
    cfg.terminations.time_out = TerminationTermCfg(func=time_out, time_out=True)
    cfg.terminations.lift_success = TerminationTermCfg(
        func=LiftSuccess, time_out=False, params={
            "lift_height_threshold": height, "dwell_time": dwell,
            "lift_dwell_time": lift_dwell_time, "target_cfg": SceneEntityCfg("target"),
        },
    )


def evaluate(args, app, stop_requested):
    # Simulator-dependent imports must follow AppLauncher.
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import eggtart_grasp.tasks  # noqa: F401
    from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import (
        LIFT_HEIGHT_THRESHOLD, LIFT_DWELL_TIME, LIFT_SUCCESS_DWELL_TIME,
    )
    from evaluation_metrics import EpisodeStatistics

    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
    checkpoint = Path(cli_args.resolve_checkpoint(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or checkpoint.parent / "evaluation" / f"{checkpoint.stem}_{stamp}.json").resolve()
    num_envs = min(args.num_envs, args.num_episodes)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    cfg.seed = args.seed
    cfg.observations.policy.enable_corruption = args.observation_noise
    cfg.events.target_approach_stage1 = None
    height = LIFT_HEIGHT_THRESHOLD if args.lift_height is None else args.lift_height
    dwell = LIFT_SUCCESS_DWELL_TIME if args.dwell_time is None else args.dwell_time
    configure_episode_termination(
        cfg, height=height, dwell=dwell, lift_dwell_time=LIFT_DWELL_TIME,
        timeout_s=args.episode_length_s,
    )
    target_params = cfg.events.reset_target.params
    locked_step = target_params[f"stage{args.curriculum_stage}_start_step"]
    pose_range = target_params[f"stage{args.curriculum_stage}_pose_range"]
    print(f"[EVAL] checkpoint: {checkpoint}", flush=True)
    print(f"[EVAL] 当前任务配置；阶段 {args.curriculum_stage}；目标跟踪关闭；"
          f"动作={'随机采样' if args.stochastic else '确定性'}；观测噪声={args.observation_noise}", flush=True)
    print(f"[EVAL] 成功条件：物块世界高度 >= {height:.3f} m 连续 {dwell:.3f} s（仅按高度）；"
          f"{num_envs} 环境 / {args.num_episodes} 回合", flush=True)
    print(f"[EVAL] 每个实例 {cfg.episode_length_s:.3f} 仿真秒内未成功即记为超时并自动重置；"
          "成功当步即重置，其他实例继续运行", flush=True)

    env = gym.make(args.task, cfg=cfg)
    try:
        base_env = env.unwrapped
        # Set the distribution before the wrapper's initial reset, and reset
        # again after loading so all measured episodes start at length zero.
        base_env.common_step_counter = locked_step
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_optimizer=False)
        deterministic_policy = runner.get_inference_policy(device=base_env.device)
        policy = runner.alg.policy.act if args.stochastic else deterministic_policy
        wrapped.seed(args.seed)
        with torch.inference_mode():
            obs, _ = wrapped.reset()
        statistics = EpisodeStatistics(num_envs, args.num_episodes, base_env.device)
        manager = base_env.termination_manager
        term_names = manager.active_terms
        failure_terms = {name for name in term_names if name != "lift_success"
                         and not manager.get_term_cfg(name).time_out}
        output.parent.mkdir(parents=True, exist_ok=True)
        env_config_path = output.with_suffix(".env.yaml")
        dump_yaml(str(env_config_path), cfg)
        settings = {
            "checkpoint": str(checkpoint), "task": args.task, "seed": args.seed,
            "num_envs": num_envs, "curriculum_stage": args.curriculum_stage,
            "target_pose_range": pose_range, "target_tracking": False,
            "stochastic_actions": args.stochastic, "observation_noise": args.observation_noise,
            "success_criterion": "target world z >= lift_height continuously for dwell_time; height only",
            "lift_height": height, "dwell_time": dwell,
            "effective_dwell_time": math.ceil(dwell / base_env.step_dt) * base_env.step_dt,
            "episode_length_s": cfg.episode_length_s, "step_dt": base_env.step_dt,
            "success_timeout_s": cfg.episode_length_s,
            "effective_timeout_s": base_env.max_episode_length * base_env.step_dt,
            "timeout_clock": "per-episode simulation time",
            "env_config": str(env_config_path), "configuration_source": "current task configuration",
        }
        start = last_report = time.monotonic()
        stop_reason = "completed"
        try:
            with torch.inference_mode():
                while not statistics.finished:
                    if stop_requested():
                        stop_reason = "interrupted"
                        break
                    if not app.is_running():
                        stop_reason = "simulator_closed"
                        break
                    base_env.common_step_counter = locked_step
                    obs, rewards, dones, _ = wrapped.step(policy(obs))
                    # Get the terminal flags, not the already-reset object pose.
                    # TerminationManager retains this step's flags across reset.
                    flags = {name: manager.get_term(name) for name in term_names}
                    statistics.update(rewards, dones, flags, failure_terms, base_env.step_dt)
                    runner.alg.policy.reset(dones)
                    now = time.monotonic()
                    if now - last_report >= 10.0 or statistics.finished:
                        summary = statistics.summary()
                        rate = summary["success_rate"]
                        text_rate = f"{100 * rate:.2f}%" if rate is not None else "待首个完整回合"
                        print(f"[EVAL] {summary['completed_episodes']}/{args.num_episodes} 回合；"
                              f"成功 {summary['successes']}；超时重置 {summary['outcomes']['time_out']}；"
                              f"成功率 {text_rate}", flush=True)
                        last_report = now
        except KeyboardInterrupt:
            stop_reason = "interrupted"

        summary = statistics.summary()
        report = {
            "settings": settings, "summary": summary,
            "stop_reason": stop_reason, "elapsed_s": time.monotonic() - start,
            "episodes": statistics.episodes,
        }
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        rate = summary["success_rate"]
        rate_text = f"{100 * rate:.2f}%" if rate is not None else "无完整回合"
        print(f"[EVAL] {'完成' if summary['complete'] else '未完成（仅统计已结束回合）'}："
              f"成功率 {rate_text} ({summary['successes']}/{summary['completed_episodes']})", flush=True)
        print(f"[EVAL] 终止分类：{summary['outcomes']}", flush=True)
        print(f"[EVAL] 报告：{output}", flush=True)
        return 0 if summary["complete"] else 130
    finally:
        env.close()


def main():
    args = parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    # Native simulator callbacks can swallow KeyboardInterrupt. Set a flag and
    # finish the current step, preserving complete episode accounting.
    interrupted = False

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    exit_code = 1
    try:
        exit_code = evaluate(args, app, lambda: interrupted)
    except Exception:
        # Print before SimulationApp.close(), which can otherwise hide errors.
        traceback.print_exc()
    finally:
        import omni.kit.app

        omni.kit.app.get_app().post_quit(exit_code)
        app.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
