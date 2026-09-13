"""Check lift-then-drop termination, episode isolation and automatic resets.

Run through isaaclab.sh -p with --headless and either Eggtart --task.
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
    from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import LIFT_HEIGHT_THRESHOLD
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=4)
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    manager = env.termination_manager
    term = manager.get_term_cfg("target_dropped")
    assert not term.time_out
    assert term.params["lift_height_threshold"] == LIFT_HEIGHT_THRESHOLD == 0.12
    assert term.params["drop_height_threshold"] == 0.05
    target = env.scene["target"]

    def set_heights(heights):
        state = target.data.root_state_w.clone()
        # Keep targets away from the robot during the physics-step check below.
        state[:, :2] = env.scene.env_origins[:, :2] + 1.0
        state[:, 2] = torch.as_tensor(heights, device=env.device)
        state[:, 7:] = 0.0
        target.write_root_state_to_sim(state)

    def check_drops(expected):
        manager.compute()
        expected = torch.tensor(expected, dtype=torch.bool, device=env.device)
        torch.testing.assert_close(manager.get_term("target_dropped"), expected)
        torch.testing.assert_close(manager.terminated, expected)
        assert not manager.time_outs.any()

    # Falling from the spawn height to the floor must not end a fresh episode.
    for height in (0.10, 0.016, 0.119, 0.016):
        set_heights(height)
        check_drops([False] * 4)
    print(f"PASS: {args.task} initial free fall ignored", flush=True)

    # Exactly 0.12 arms the condition; exactly 0.05 does not trigger a drop.
    set_heights([0.12, 0.13, 0.119, 0.14])
    check_drops([False] * 4)
    set_heights([0.05, 0.049, 0.016, 0.06])
    check_drops([False, True, False, False])
    # A new episode in env 1 must not clear the history of envs 0 and 3.
    env._reset_idx(torch.tensor([1], device=env.device))
    set_heights([0.049, 0.04, 0.016, 0.04])
    check_drops([True, False, False, True])
    env.reset()
    set_heights(0.016)
    check_drops([False] * 4)
    print(f"PASS: {args.task} thresholds / per-episode history / partial and full resets", flush=True)

    # Verify gym step returns a failure and automatically clears the dropped episode.
    set_heights([0.13, 0.10, 0.10, 0.10])
    check_drops([False] * 4)
    set_heights(0.016)
    _, _, terminated, truncated, _ = env.step(torch.zeros(4, 9, device=env.device))
    assert terminated.tolist() == [True, False, False, False]
    assert not truncated.any()
    assert env.episode_length_buf.tolist() == [0, 1, 1, 1]
    assert not term.func._was_lifted.any()
    set_heights(0.016)
    check_drops([False] * 4)
    print(f"PASS: {args.task} terminated flag / automatic reset / next episode", flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
