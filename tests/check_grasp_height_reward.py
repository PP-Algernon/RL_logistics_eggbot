"""Validate the grasp height reward using controlled states in Isaac Sim.

Run with isaaclab.sh -p, --headless and either registered Eggtart --task.
"""
import argparse
import math
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
    from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import (
        LIFT_HEIGHT_THRESHOLD,
        LIFT_TARGET_HEIGHT_THRESHOLD1,
        LIFT_TARGET_HEIGHT_THRESHOLD2,
    )
    from isaaclab.utils.math import quat_apply
    from isaaclab_tasks.utils import parse_env_cfg

    heights = [0.119, 0.12, 0.125, 0.13, 0.135, 0.14, 0.15, 0.16, 0.165, 0.17, 0.175, 0.18, 0.24]
    expected = [0.0, 1.0, 1.25, 1.5, 1.75, 2.0, 2.0, 2.0,
                1 + math.exp(-0.25), 1 + math.exp(-0.5), 1 + math.exp(-0.75),
                1 + math.exp(-1.0), 1 + math.exp(-4.0)]
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=len(heights))
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    term = env.reward_manager.get_term_cfg("grasp")
    assert term.params["lift_height_threshold"] == LIFT_HEIGHT_THRESHOLD == 0.12
    assert term.params["lift_target_height_threshold1"] == LIFT_TARGET_HEIGHT_THRESHOLD1 == 0.14
    assert term.params["lift_target_height_threshold2"] == LIFT_TARGET_HEIGHT_THRESHOLD2 == 0.16
    robot, target = env.scene["robot"], env.scene["target"]
    ee = term.params["ee_cfg"].body_ids[0]
    offset = torch.tensor(term.params["grasp_offset"], device=env.device).expand(env.num_envs, -1)

    def grasp_positions():
        return robot.data.body_pos_w[:, ee] + quat_apply(robot.data.body_quat_w[:, ee], offset)

    # Align the actual grasp point and target at each height, without advancing physics.
    heights = torch.tensor(heights, device=env.device)
    root = robot.data.root_state_w.clone()
    root[:, 2] += heights - grasp_positions()[:, 2]
    root[:, 7:] = 0.0
    robot.write_root_state_to_sim(root)
    state = target.data.root_state_w.clone()
    state[:, :3] = grasp_positions()
    state[:, 2] = heights
    state[:, 7:] = 0.0
    target.write_root_state_to_sim(state)
    expected = torch.tensor(expected, device=env.device)
    term.func.reset()
    actual = term.func(env, **term.params)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
    # Repeated calls award holding at the same height immediately and consistently.
    torch.testing.assert_close(term.func(env, **term.params), expected, atol=2e-6, rtol=0)
    print(f"PASS: {args.task} threshold jump / approach / interval plateau / overshoot", flush=True)

    # Both the base and height bonuses must disappear if the object is detached or fast.
    invalid = state.clone()
    invalid[5, 0] += term.params["hold_dist"] + 0.05
    invalid[6, 7] = term.params["max_speed"] + 1.0
    invalid[7, 2] += term.params["hold_dist"] + 0.05
    target.write_root_state_to_sim(invalid)
    gated = expected.clone()
    gated[5:8] = 0.0
    torch.testing.assert_close(term.func(env, **term.params), gated, atol=2e-6, rtol=0)
    assert (term.func._lift_counter[5:8] == 0).all()
    target.write_root_state_to_sim(state)
    torch.testing.assert_close(term.func(env, **term.params), expected, atol=2e-6, rtol=0)
    term.func.reset(env_ids=torch.tensor([5], device=env.device))
    assert term.func._lift_counter[5] == 0 and term.func._lift_counter[6] > 0
    print(f"PASS: {args.task} distance / speed gates / recovery / partial reset", flush=True)

    for lower, upper in ((0.12, 0.16), (0.16, 0.14), (float("nan"), 0.16), (0.14, float("inf"))):
        params = dict(term.params, lift_target_height_threshold1=lower, lift_target_height_threshold2=upper)
        try:
            term.func(env, **params)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid height interval accepted: {(lower, upper)}")
    print(f"PASS: {args.task} invalid interval validation", flush=True)
    print(f"PASS: {args.task} grasp height reward checks complete", flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
