"""Check evaluation timeout flags, per-instance resets and success at the deadline.

Run with isaaclab.sh -p tests/check_evaluation_timeout.py --headless.
"""
import argparse
import math
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
env = None

try:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import eggtart_grasp.tasks  # noqa: F401
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
    from evaluate import configure_episode_termination, parse_args
    from evaluation_metrics import EpisodeStatistics

    for flag in ("--success_timeout_s", "--episode_length_s"):
        with patch.object(sys, "argv", ["evaluate.py", flag, "6"]):
            assert parse_args().episode_length_s == 6.0

    task = "Isaac-Mobile-Grasp-Eggtart-BCPPO-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=4)
    cfg.events.target_approach_stage1 = None
    cfg.terminations.time_out = None  # Evaluation must restore the guard itself.
    for timeout in (0.0, -1.0, float("inf"), float("nan"), 0.05):
        try:
            configure_episode_termination(cfg, height=0.1, dwell=0.07, lift_dwell_time=0.01,
                                          timeout_s=timeout)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid timeout accepted: {timeout}")
    configure_episode_termination(cfg, height=0.1, dwell=0.07, lift_dwell_time=0.01,
                                  timeout_s=0.105)
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    manager = env.termination_manager
    success_term = manager.get_term_cfg("lift_success").func
    assert manager.get_term_cfg("time_out").time_out
    assert env.max_episode_length == math.ceil(0.105 / env.step_dt)

    # Two instances reach the deadline next step. Only env 1 succeeds on it;
    # env 2 reaches its own deadline one step later and env 3 is still running.
    target = env.scene["target"]
    state = target.data.root_state_w.clone()
    state[:, :2] = env.scene.env_origins[:, :2] + 1.0
    state[:, 2] = torch.tensor([0.016, 0.25, 0.016, 0.016], device=env.device)
    state[:, 7:] = 0.0
    target.write_root_state_to_sim(state)
    success_term._lift_steps[:] = torch.tensor(
        [1, math.ceil(0.07 / env.step_dt) - 1, 0, 0], device=env.device
    )
    env.episode_length_buf[:] = torch.tensor(
        [env.max_episode_length - 1, env.max_episode_length - 1, env.max_episode_length - 2, 0],
        device=env.device,
    )
    statistics = EpisodeStatistics(4, 6, env.device)
    statistics.lengths[:] = env.episode_length_buf

    def step():
        _, rewards, terminated, truncated, _ = env.step(torch.zeros(4, 9, device=env.device))
        statistics.update(rewards, terminated | truncated,
                          {name: manager.get_term(name) for name in manager.active_terms},
                          {"base_tipped", "target_dropped"}, env.step_dt)
        return terminated, truncated

    terminated, truncated = step()
    assert terminated.tolist() == [False, True, False, False]
    assert truncated.tolist() == [True, True, False, False]
    assert env.episode_length_buf.tolist() == [0, 0, env.max_episode_length - 1, 1]
    assert not success_term._lift_steps[:2].any()
    assert not manager.get_term_cfg("target_dropped").func._was_lifted[:2].any()
    torch.testing.assert_close(target.data.root_pos_w[:2, 2], torch.full((2,), 0.1, device=env.device))
    torch.testing.assert_close(target.data.root_pos_w[2:, :2], state[2:, :2])
    assert [episode["outcome"] for episode in statistics.episodes] == ["time_out", "success"]
    assert statistics.episodes[0]["duration_s"] >= 0.105
    assert statistics.episodes[0]["duration_s"] < 0.105 + env.step_dt

    terminated, truncated = step()
    assert not terminated.any()
    assert truncated.tolist() == [False, False, True, False]
    assert env.episode_length_buf.tolist() == [1, 1, 0, 2]
    assert statistics.summary()["outcomes"]["time_out"] == 2
    assert statistics.summary()["success_rate"] == 1 / 3
    assert statistics.episodes[-1]["env_id"] == 2
    print("PASS: CLI aliases, timeout validation, deadline success, independent resets and failure accounting", flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
