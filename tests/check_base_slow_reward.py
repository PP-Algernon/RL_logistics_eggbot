"""Check grasp-gated base slowdown using controlled Isaac Sim states.

Run with isaaclab.sh -p, --headless and either Eggtart --task.
"""
import argparse
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
    from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import CURRICULUM_STAGE3_START_ITER
    from isaaclab.utils.math import quat_apply
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=12)
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    name = "base_slow_after_grasp"
    is_bc = args.task.endswith("-BCPPO-v0")
    assert env.reward_manager.get_term_cfg(name).weight == (5.0 if is_bc else 0.0)
    for step in (CURRICULUM_STAGE3_START_ITER * 24 - 1, CURRICULUM_STAGE3_START_ITER * 24):
        env.common_step_counter = step
        env.reset()
        assert env.reward_manager.get_term_cfg(name).weight == (
            5.0 if is_bc or step == CURRICULUM_STAGE3_START_ITER * 24 else 0.0
        )
    term = env.reward_manager.get_term_cfg(name)
    robot, target = env.scene["robot"], env.scene["target"]
    grasp = env.reward_manager.get_term_cfg("grasp")
    for key in ("lift_height_threshold", "hold_dist", "grasp_offset"):
        assert term.params[key] == grasp.params[key]
    assert term.params["max_target_speed"] == grasp.params["max_speed"]
    ee = term.params["ee_cfg"].body_ids[0]
    offset = torch.tensor(term.params["grasp_offset"], device=env.device).expand(env.num_envs, -1)

    def grasp_positions():
        return robot.data.body_pos_w[:, ee] + quat_apply(robot.data.body_quat_w[:, ee], offset)

    root = robot.data.root_state_w.clone()
    root[:, 2] += 0.15 - grasp_positions()[:, 2]
    root[:, 7:] = 0.0
    robot.write_root_state_to_sim(root)
    state = target.data.root_state_w.clone()
    state[:, :3] = grasp_positions()
    state[:, 7:] = 0.0
    state[9, 2] = 0.119
    state[10, 0] += term.params["hold_dist"] + 0.05
    state[11, 7] = term.params["max_target_speed"] + 1.0
    target.write_root_state_to_sim(state)

    velocity = torch.zeros(12, 6, device=env.device)
    velocity[1, 0], velocity[2, 0], velocity[3, 0] = 0.15, 0.30, -0.15
    velocity[4, 5], velocity[5, 5] = 0.3, -0.3
    velocity[6, 0], velocity[6, 5] = 0.15, 0.3
    velocity[7, 1] = 0.15
    velocity[8, 0], velocity[8, 5] = 0.6, 1.5
    robot.write_root_velocity_to_sim(velocity)
    expected = torch.tensor([1, 0.5, 0.2, 0.5, 0.5, 0.5, 1/3, 0.5, 1/42, 0, 0, 0], device=env.device)
    torch.testing.assert_close(term.func(env, **term.params), expected)
    print(f"PASS: {args.task} curriculum / translation / rotation / grasp gates", flush=True)

    # Actual slowdown increases the reward; staying stopped remains rewarded.
    robot.write_root_velocity_to_sim(velocity * 0.5)
    slower = term.func(env, **term.params)
    assert (slower[1:9] > expected[1:9]).all()
    robot.write_root_velocity_to_sim(torch.zeros_like(velocity))
    for _ in range(2):
        result = term.func(env, **term.params)
        torch.testing.assert_close(result[:9], torch.ones(9, device=env.device))
        assert (result[9:] == 0).all()
    # Losing the object immediately disables the reward, even when stopped.
    state[0, 2] = 0.04
    target.write_root_state_to_sim(state)
    assert term.func(env, **term.params)[0] == 0
    state[0, :3] = grasp_positions()[0]
    target.write_root_state_to_sim(state)
    assert term.func(env, **term.params)[0] == 1
    print(f"PASS: {args.task} slowing / sustained stop / drop / regrasp", flush=True)

    for key, value in (("lin_vel_scale", 0.0), ("ang_vel_scale", -1.0), ("lin_vel_scale", float("nan"))):
        try:
            term.func(env, **dict(term.params, **{key: value}))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid speed scale accepted: {key}={value}")
    env.reset()
    _, reward, _, _, _ = env.step(torch.zeros(12, 9, device=env.device))
    assert torch.isfinite(reward).all()
    print(f"PASS: {args.task} invalid scales / environment step", flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
