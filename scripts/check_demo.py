#!/usr/bin/env python
"""成功轨迹回放：恢复同一条轨迹的机器人/目标初态并执行录制动作。

用法（远程，非 headless——isaaclab.sh 默认带窗口，可直接观看）:
    ./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/check_demo.py \
        --task Isaac-Mobile-Grasp-Eggtart-v0 \
        --data datasets/eggtart_demo.hdf5 \
        --episode 0

参数:
    --episode      回放哪条轨迹：整数索引 / 'random' / 'all'（默认 0）
    --frame_skip   每 N 步保存一帧，不跳过动作或物理步（默认 1）
    --save_frames  保存渲染帧到目录（可选，用 opencv 写 PNG，之后可合成视频）
    --post_wait    回放结束后保持窗口（默认 5 秒）
    --no-realtime  关闭实时播放；默认按仿真时间逐帧播放

说明:
    - 依赖新版采集数据（hdf5 含 init_robot_root_pos 等初始状态），
      回放时恢复录制的世界坐标，不分别随机生成机器人或物块。
    - 不含初始状态的数据不能用于验证原轨迹；脚本会报错，避免误播随机场景。
    - 接触轨迹对物理参数敏感，修改环境配置后可能无法复现录制结果。
    - 回放时限自动覆盖选中轨迹；训练成功/失败终止项关闭，完整播放后再切换轨迹。
"""

import argparse
import os
import sys
import time
import traceback

# When launched through a Kit Python that does not have Isaac Lab installed as
# an editable package, isaaclab.sh still exposes ISAACLAB_PATH.  Add its source
# tree before importing AppLauncher; otherwise the script fails before Isaac
# Sim can load ``omni.timeline``.
_isaaclab_root = os.environ.get("ISAACLAB_PATH")
if _isaaclab_root:
    _isaaclab_src = os.path.join(_isaaclab_root, "source", "isaaclab")
    if os.path.isdir(_isaaclab_src) and _isaaclab_src not in sys.path:
        sys.path.insert(0, _isaaclab_src)

import h5py
import numpy as np
import torch

from demo_dataset import canonical_demo_task

# Isaac Lab/Omniverse extensions (for example ``omni.timeline``) are loaded
# only after SimulationApp starts.  This replay script used to import the task
# configuration before launching the app, which works for plain data inspection
# but fails as soon as the task package imports Isaac Lab managers.
from isaaclab.app import AppLauncher

_ACTION_DIM = 9


def _load_episode_indices(arg: str, num_episodes: int) -> list[int]:
    if num_episodes <= 0:
        raise ValueError("数据集没有成功轨迹可供回放")
    if arg == "random":
        return [int(np.random.randint(0, num_episodes))]
    if arg == "all":
        return list(range(num_episodes))
    idx = int(arg)
    if idx < 0 or idx >= num_episodes:
        raise ValueError(f"episode {idx} 越界（共 {num_episodes} 条轨迹）")
    return [idx]


def _configure_replay_terminations(env_cfg, max_episode_steps: int) -> None:
    """Keep the final recorded state visible, then let the replay loop reset.

    Training success/failure conditions must not teleport the scene partway
    through an archived demonstration. Retain only a timeout beyond its end.
    """
    from isaaclab.envs.mdp import time_out
    from isaaclab.managers import TerminationTermCfg

    step_dt = env_cfg.sim.dt * env_cfg.decimation
    previous_limit = env_cfg.episode_length_s
    # time_out uses >=, so allow one extra step even for the longest episode.
    env_cfg.episode_length_s = max(previous_limit, (max_episode_steps + 1) * step_dt)
    disabled = []
    for name, term in vars(env_cfg.terminations).items():
        if name != "time_out" and isinstance(term, TerminationTermCfg):
            setattr(env_cfg.terminations, name, None)
            disabled.append(name)
    env_cfg.terminations.time_out = TerminationTermCfg(func=time_out, time_out=True)
    print(f"[回放] 最长选中轨迹 {max_episode_steps} 步（{max_episode_steps * step_dt:.3f} s）；"
          f"回放时限 {previous_limit:.3f} -> {env_cfg.episode_length_s:.3f} s")
    if disabled:
        print(f"[回放] 已关闭训练终止项: {', '.join(disabled)}；每条录制轨迹完整播放后切换")


def main() -> None:
    ap = argparse.ArgumentParser(description="成功轨迹回放")
    ap.add_argument("--task", default=None, help="默认使用数据集记录的 env_name")
    ap.add_argument("--data", default="datasets/eggtart_demo.hdf5")
    ap.add_argument("--episode", default="0", help="轨迹索引 / random / all")
    ap.add_argument("--frame_skip", type=int, default=1, help="每 N 步保存一帧，不跳过动作")
    ap.add_argument("--save_frames", default="", help="保存渲染帧的目录（可选）")
    ap.add_argument(
        "--post_wait", type=float, default=5.0,
        help="回放结束后保持窗口的秒数（默认 5；设为 0 立即退出）",
    )
    ap.add_argument(
        "--realtime", action=argparse.BooleanOptionalAction, default=True,
        help="按仿真时间播放（默认开启）；--no-realtime 全速播放",
    )
    AppLauncher.add_app_launcher_args(ap)
    args_cli = ap.parse_args()

    if args_cli.frame_skip < 1 or args_cli.post_wait < 0:
        ap.error("frame_skip 必须 >= 1，post_wait 必须 >= 0")

    # ---------- 加载 hdf5 ----------
    with h5py.File(args_cli.data, "r") as f:
        act = f["action"][:]
        ep_lens = f["episode_lengths"][:]
        init_keys = ("init_robot_root_pos", "init_robot_root_quat", "init_robot_joint_pos",
                     "init_target_root_pos", "init_target_root_quat")
        missing = [key for key in init_keys if key not in f]
        if missing:
            raise ValueError(f"数据集缺少初始状态 {missing}，无法对应原轨迹。请使用包含初始状态的数据集。")
        init_rp, init_rq, init_jp, init_tp, init_tq = [f[key][:] for key in init_keys]
        for key, array, width in zip(init_keys, (init_rp, init_rq, init_jp, init_tp, init_tq), (3, 4, None, 3, 4)):
            if array.ndim != 2 or len(array) != len(ep_lens) or (width is not None and array.shape[1] != width):
                raise ValueError(f"{key} 的形状 {array.shape} 与轨迹数量/状态维度不匹配")
            if not np.isfinite(array).all():
                raise ValueError(f"{key} 包含非有限数值")
        recorded_task = f.attrs.get("env_name", "Isaac-Mobile-Grasp-Eggtart-v0")
        lift_height = float(f.attrs.get("lift_height", 0.15))
        has_obs = "obs" in f

    if act.ndim != 2 or act.shape[1] != _ACTION_DIM or not np.isfinite(act).all():
        raise ValueError(f"action 应为有限值的 [N, {_ACTION_DIM}] 数组，实际形状为 {act.shape}")
    if ep_lens.ndim != 1 or not np.issubdtype(ep_lens.dtype, np.integer) or (ep_lens <= 0).any() or ep_lens.sum() != len(act):
        raise ValueError("episode_lengths 必须为正整数，其总和必须等于 action 样本数")
    args_cli.task = args_cli.task or canonical_demo_task(recorded_task)
    if args_cli.task != recorded_task:
        print(f"[提示] 录制任务是 {recorded_task}，本次使用 {args_cli.task}；物理配置需要保持一致。")
    print(f"数据: {args_cli.data}  成功轨迹: {len(ep_lens)} 条  样本: {len(act):,}")

    ep_indices = _load_episode_indices(args_cli.episode, len(ep_lens))
    print(f"将回放 {len(ep_indices)} 条轨迹: {ep_indices}\n")

    if args_cli.save_frames:
        args_cli.enable_cameras = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        _replay(args_cli, simulation_app, act, ep_lens, ep_indices,
                (init_rp, init_rq, init_jp, init_tp, init_tq), lift_height, has_obs)
    except BaseException:
        # Kit shutdown can terminate Python before an unhandled exception is
        # printed. Report it before closing the application.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()


def _replay(args_cli, simulation_app, act, ep_lens, ep_indices, init_states, lift_height, has_obs):
    init_rp, init_rq, init_jp, init_tp, init_tq = init_states

    # ---------- 建环境（与采集同配置） ----------
    import eggtart_grasp.tasks  # noqa: F401 -- register the task after AppLauncher
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.events.target_approach_stage1 = None
    env_cfg.events.randomize_target_velocity = None
    _configure_replay_terminations(env_cfg, max(int(ep_lens[ep]) for ep in ep_indices))

    import gymnasium as gym

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.save_frames else None)
    try:
        base_env = env.unwrapped
        device = base_env.device
        robot = base_env.scene["robot"]
        target = base_env.scene["target"]
        if init_jp.shape[1] != robot.num_joints:
            raise ValueError(f"录制关节数 {init_jp.shape[1]} 与当前机器人 {robot.num_joints} 不一致")
        # 目标相对底盘的位置观测没有噪声，可逐步核对回放是否偏离采集。
        target_slice = None
        column = 0
        manager = base_env.observation_manager
        for name, shape in zip(manager.active_terms["policy"], manager.group_obs_term_dim["policy"]):
            width = int(np.prod(shape))
            if name == "target_position_b":
                target_slice = slice(column, column + width)
            column += width
        print(
            f"场景已创建: robot_pos={robot.data.root_pos_w[0].detach().cpu().numpy().round(3)} "
            f"target_pos={target.data.root_pos_w[0].detach().cpu().numpy().round(3)}"
        )

        if args_cli.save_frames:
            os.makedirs(args_cli.save_frames, exist_ok=True)

        # 累计起点（每条轨迹在拼接数组中的偏移）
        ep_starts = np.concatenate([[0], np.cumsum(ep_lens)[:-1]])

        for ep in ep_indices:
            if not simulation_app.is_running():
                break
            s, e = int(ep_starts[ep]), int(ep_starts[ep] + ep_lens[ep])
            print(f"=== 回放轨迹 {ep} (长度 {ep_lens[ep]} 步) ===")

            # 同一 ep 同时索引动作、机器人和目标初态。保留录制的世界坐标，
            # 不能只把其中一个物体移回 env_0 原点。
            env.reset()
            idx0 = torch.tensor([0], device=device)
            rp = torch.tensor(init_rp[ep], device=device).unsqueeze(0)
            rq = torch.tensor(init_rq[ep], device=device).unsqueeze(0)
            jp = torch.tensor(init_jp[ep], device=device).unsqueeze(0)
            tp = torch.tensor(init_tp[ep], device=device).unsqueeze(0)
            tq = torch.tensor(init_tq[ep], device=device).unsqueeze(0)
            # root_state = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            zero6 = torch.zeros(1, 6, device=device)
            robot.write_root_state_to_sim(torch.cat([rp, rq, zero6], dim=-1), env_ids=idx0)
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=idx0)
            target.write_root_state_to_sim(torch.cat([tp, tq, zero6], dim=-1), env_ids=idx0)
            print(f"  已恢复初始状态: robot_pos={init_rp[ep].round(3)} target_pos={init_tp[ep].round(3)}")

            base_env.sim.forward()
            obs = base_env.observation_manager.compute(update_history=False)
            reference = None
            if has_obs and target_slice is not None:
                with h5py.File(args_cli.data, "r") as f:
                    if f["obs"].shape == (len(act), column):
                        reference = torch.as_tensor(f["obs"][s:e, target_slice], device=device)
                if reference is not None:
                    initial_error = torch.linalg.vector_norm(obs["policy"][0, target_slice] - reference[0]).item()
                    print(f"  初始目标相对机器人位置误差: {initial_error:.6f} m")
                    if initial_error > 1e-4:
                        raise ValueError("初始状态与该轨迹首帧观测不对应，停止回放")
            max_position_error = torch.zeros((), device=device)

            try:
                center = 0.5 * (robot.data.root_pos_w[0] + target.data.root_pos_w[0])
                center[2] = 0.15
                eye = center + torch.tensor([1.2, 1.5, 0.9], device=device)
                base_env.sim.set_camera_view(eye=eye.tolist(), target=center.tolist())
            except Exception as exc:  # noqa: BLE001
                print(f"[警告] 无法按实体位置聚焦相机: {exc}")

            base_env.sim.render()
            # env.step 内部已刷新视口；直接 app.update() 会额外推进物理，
            # 改变每个录制动作的执行时长。额外刷新只能调用 sim.render()。
            for t in range(e - s):
                if not simulation_app.is_running():
                    break
                frame_start = time.perf_counter()
                if reference is not None:
                    position_error = torch.linalg.vector_norm(obs["policy"][0, target_slice] - reference[t])
                    max_position_error = torch.maximum(max_position_error, position_error)
                a = torch.tensor(act[s + t], device=device).unsqueeze(0)
                obs, _, terminated, truncated, _ = env.step(a)
                if (terminated | truncated).any():
                    reasons = [name for name in base_env.termination_manager.active_terms
                               if base_env.termination_manager.get_term(name).any()]
                    raise RuntimeError(
                        f"轨迹 {ep} 在第 {t + 1}/{e - s} 步意外重置，触发项: {reasons}；"
                        f"回放时限为 {base_env.max_episode_length} 步"
                    )

                if (t % args_cli.frame_skip == 0) and args_cli.save_frames:
                    try:
                        rgb = env.render()
                        if rgb is not None:
                            import cv2
                            cv2.imwrite(os.path.join(args_cli.save_frames, f"ep{ep}_t{t:04d}.png"),
                                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(f"保存轨迹 {ep} 第 {t} 帧失败") from exc
                if args_cli.realtime and base_env.sim.has_gui():
                    time.sleep(max(0.0, float(base_env.step_dt) - (time.perf_counter() - frame_start)))

            final_z = target.data.root_pos_w[0, 2].item()
            print(f"  轨迹 {ep} 回放结束，目标中心高度: {final_z:.3f} m "
                  f"（地面中心高度 + 抬升门槛: {env_cfg.scene.target.spawn.size[2] / 2 + lift_height:.3f} m）")
            if reference is not None:
                print(f"  全程目标相对位置最大偏差: {max_position_error.item():.6f} m\n")

        if args_cli.post_wait > 0 and base_env.sim.has_gui():
            print(f"保持窗口 {args_cli.post_wait:.1f} 秒（可用 --post_wait 0 关闭）")
            end_time = time.perf_counter() + args_cli.post_wait
            while simulation_app.is_running() and time.perf_counter() < end_time:
                base_env.sim.render()
                time.sleep(0.02)

    finally:
        env.close()
    print("回放结束。画面为 Isaac Lab viewport 实时显示；"
          "若使用 --save_frames，可用 ffmpeg 合成视频: "
          "ffmpeg -framerate 30 -i ep0_t%04d.png out.mp4")


if __name__ == "__main__":
    main()
