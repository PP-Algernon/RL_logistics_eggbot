"""Check continuous lift success, reset isolation and Gym termination flags.

Run with isaaclab.sh -p, --headless and either Eggtart --task.
"""
import argparse
import math
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Mobile-Grasp-Eggtart-BCPPO-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
env = None

try:
    import gymnasium as gym
    import torch

    import eggtart_grasp.tasks  # noqa: F401
    from eggtart_grasp.tasks.mobile_grasp.mdp.terminations import LiftSuccess
    from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import (
        LIFT_DWELL_TIME,
        LIFT_HEIGHT_THRESHOLD,
        LIFT_SUCCESS_DWELL_TIME,
    )
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=4)
    # Evaluation enables this term even when training intentionally disables it.
    cfg.terminations.lift_success = TerminationTermCfg(
        func=LiftSuccess, time_out=False, params={
            "lift_height_threshold": LIFT_HEIGHT_THRESHOLD,
            "dwell_time": LIFT_SUCCESS_DWELL_TIME,
            "lift_dwell_time": LIFT_DWELL_TIME,
            "target_cfg": SceneEntityCfg("target"),
        },
    )
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    manager = env.termination_manager
    term = manager.get_term_cfg("lift_success")
    assert isinstance(term.func, LiftSuccess)
    assert not term.time_out
    assert term.params["lift_height_threshold"] == LIFT_HEIGHT_THRESHOLD
    assert term.params["dwell_time"] == LIFT_SUCCESS_DWELL_TIME > LIFT_DWELL_TIME
    assert term.params["lift_dwell_time"] == env.cfg.rewards.grasp.params["lift_dwell_time"]
    assert env.max_episode_length_s > term.params["dwell_time"]
    steps = math.ceil(term.params["dwell_time"] / env.step_dt)
    target = env.scene["target"]
    low, high = LIFT_HEIGHT_THRESHOLD - 0.001, LIFT_HEIGHT_THRESHOLD

    def set_heights(heights):
        state = target.data.root_state_w.clone()
        # Keep targets clear of robot contacts during the automatic-reset check.
        state[:, :2] = env.scene.env_origins[:, :2] + 1.0
        state[:, 2] = torch.as_tensor(heights, device=env.device)
        state[:, 7:] = 0.0
        target.write_root_state_to_sim(state)

    def check(expected):
        manager.compute()
        torch.testing.assert_close(
            manager.get_term("lift_success"),
            torch.tensor(expected, dtype=torch.bool, device=env.device),
        )
        assert not manager.time_outs.any()

    # Time spent below threshold never contributes, even for a full dwell period.
    set_heights(low)
    for _ in range(steps + 1):
        check([False] * 4)

    # Exact height qualifies. A one-step interruption must restart the entire dwell.
    for step in range(1, steps + 1):
        set_heights([high, low, low if step == steps // 2 else high, high])
        if step == steps // 2:
            env._reset_idx(torch.tensor([3], device=env.device))
            set_heights([high, low, low, high])
        check([step == steps, False, False, False])
    assert manager.terminated.tolist() == [True, False, False, False]
    print(f"PASS: {args.task} height boundary / continuous dwell / 2.5 s does not terminate", flush=True)

    # Partial resets leave all other environments' accumulated duration intact.
    env.reset()
    set_heights(high)
    for _ in range(steps - 1):
        check([False] * 4)
    env._reset_idx(torch.tensor([1], device=env.device))
    set_heights(high)
    check([True, False, True, True])
    env.reset()
    set_heights(high)
    check([False] * 4)
    print(f"PASS: {args.task} partial / full episode reset isolation", flush=True)

    # A non-integral number of control steps rounds up, never terminating early.
    term.func.reset()
    params = dict(term.params, dwell_time=term.params["dwell_time"] + env.step_dt / 2)
    for _ in range(steps):
        assert not term.func(env, **params).any()
    assert term.func(env, **params).all()
    for dwell in (0.0, LIFT_DWELL_TIME - 0.1, LIFT_DWELL_TIME, float("nan"), float("inf")):
        try:
            term.func(env, **dict(term.params, dwell_time=dwell))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid success dwell accepted: {dwell}")
    print(f"PASS: {args.task} fractional duration / strict dwell validation", flush=True)

    # The final physics step returns terminated (not truncated), logs success,
    # and resets the successful environment's timer before the next episode.
    env.reset()
    set_heights([high + 0.15, low, low, low])
    for _ in range(steps - 1):
        check([False] * 4)
    _, reward, terminated, truncated, extras = env.step(torch.zeros(4, 9, device=env.device))
    assert terminated.tolist() == [True, False, False, False]
    assert not truncated.any()
    assert torch.isfinite(reward).all()
    assert env.episode_length_buf.tolist() == [0, 1, 1, 1]
    assert not term.func._lift_steps.any()
    assert extras["log"]["Episode_Termination/lift_success"] > 0.0
    # The evaluator must see the success flag despite the timer and pose reset.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
    from evaluation_metrics import EpisodeStatistics

    statistics = EpisodeStatistics(4, 4, env.device)
    statistics.update(
        reward, terminated | truncated,
        {name: manager.get_term(name) for name in manager.active_terms},
        {"base_tipped", "target_dropped"}, env.step_dt,
    )
    assert statistics.summary()["successes"] == 1
    set_heights(high)
    check([False] * 4)
    print(f"PASS: {args.task} success flag / automatic reset / logging / next episode", flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
