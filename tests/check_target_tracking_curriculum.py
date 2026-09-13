"""Check tracking assistance, decay, release gates and unassisted stages in Isaac Sim.

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
    from eggtart_grasp.tasks.mobile_grasp.mdp.curriculums import target_tracking_curriculum
    from eggtart_grasp.tasks.mobile_grasp.mdp.target import target_tracking_strength
    from isaaclab.utils.math import quat_apply
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=8)
    assert cfg.events.randomize_target_velocity is None
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    event = env.event_manager.get_term_cfg("target_approach_stage1")
    assert env.event_manager.active_terms["interval"] == ["target_approach_stage1"]
    assert event.interval_range_s == (0.0, 0.0)
    params = event.params
    end = cfg.events.reset_target.params["stage2_start_step"]
    assert params["stage1_end_step"] == end
    assert target_tracking_strength(0, end) == 1.0
    assert target_tracking_strength(end // 2, end) == 0.5
    assert target_tracking_strength(end, end) == 0.0
    robot, target = env.scene["robot"], env.scene["target"]
    ee = params["ee_cfg"].body_ids[0]
    gripper = params["gripper_cfg"].joint_ids[0]
    offset = torch.tensor(params["grasp_offset"], device=env.device).expand(env.num_envs, -1)
    joints = robot.data.default_joint_pos.clone()
    joints[:, gripper] = 1.0
    joints[3, gripper] = params["gripper_closed_threshold"]
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))

    def grasp_positions():
        return robot.data.body_pos_w[:, ee] + quat_apply(robot.data.body_quat_w[:, ee], offset)

    # Test event behavior without advancing physics; move the actual reference body.
    root = robot.data.root_state_w.clone()
    root[:, 2] += 0.06 - grasp_positions()[:, 2]
    root[5, 2] += 0.20  # Raised gripper must not attract the target.
    robot.write_root_state_to_sim(root)
    state = target.data.root_state_w.clone()
    state[:, :3] = grasp_positions()
    state[:, 0] += 0.10
    state[:, 2] = 0.016
    state[1, 0] = grasp_positions()[1, 0] + 0.005
    state[2, :2] = grasp_positions()[2, :2]  # Aligned: no constant-speed overshoot.
    state[4, 2] = params["lift_height_threshold"]
    state[6, 0] += 1.0  # Out of activation range.
    state[:, 7:] = 0.0
    state[:, 9] = -0.07  # Preserve gravity-driven descent.
    state[:, 10:13] = torch.tensor([0.1, -0.2, 0.3], device=env.device)

    def apply_at(step, env_ids=None):
        env.common_step_counter = step
        target.write_root_state_to_sim(state)
        event.func(env, env_ids, **params)
        return target.data.root_vel_w.clone()

    full = apply_at(0)
    torch.testing.assert_close(full[0, :2], torch.tensor([-0.20, 0.0], device=env.device), atol=2e-5, rtol=0)
    assert 0 < -full[1, 0] < 0.04
    torch.testing.assert_close(full[2, :2], torch.zeros(2, device=env.device))
    torch.testing.assert_close(full[:, 2:], state[:, 9:13])
    torch.testing.assert_close(full[3:7], state[3:7, 7:13])
    torch.testing.assert_close(target.data.root_pos_w, state[:, :3])
    for step, strength in ((end // 4, 0.75), (end // 2, 0.5), (end - 1, 1 / end)):
        velocity = apply_at(step)
        torch.testing.assert_close(velocity[0, :2], full[0, :2] * strength, atol=2e-6, rtol=0)
        assert target_tracking_curriculum(env, None) == strength
    print(f"PASS: {args.task} strong tracking / close-range slowdown / linear fade / gravity preserved", flush=True)

    # Partial event calls must not touch other environments; no per-episode history is used.
    selected = apply_at(0, torch.tensor([7], device=env.device))
    torch.testing.assert_close(selected[:7], state[:7, 7:13])
    torch.testing.assert_close(selected[7], full[7])
    for step in (end, end + 1, cfg.events.reset_target.params["stage3_start_step"]):
        # Neither overwrite natural motion nor stop the object after assistance ends.
        state[:, 7:9] = torch.tensor([0.11, -0.08], device=env.device)
        torch.testing.assert_close(apply_at(step), state[:, 7:13])
        assert target_tracking_curriculum(env, None) == 0.0
    state[:, 7:9] = 0.0
    print(f"PASS: {args.task} closed/lifted/high/far gates / partial IDs / stages 2 and 3 untouched", flush=True)

    # The real interval dispatcher applies tracking each action step without waiting for a reset.
    env.common_step_counter = 0
    target.write_root_state_to_sim(state)
    env.event_manager.apply(mode="interval", dt=env.step_dt)
    torch.testing.assert_close(target.data.root_vel_w, full)
    env.common_step_counter = end
    target.write_root_state_to_sim(state)
    env.event_manager.apply(mode="interval", dt=env.step_dt)
    torch.testing.assert_close(target.data.root_vel_w, state[:, 7:13])
    # Collection/replay explicitly disable the event: its curriculum metric is then zero.
    cfg.events.target_approach_stage1 = None
    env.common_step_counter = 0
    assert target_tracking_curriculum(env, None) == 0.0
    cfg.events.target_approach_stage1 = event
    env.reset()
    _, rewards, _, _, _ = env.step(torch.zeros(8, 9, device=env.device))
    assert torch.isfinite(rewards).all()
    print(f"PASS: {args.task} interval integration / disabled helper / normal environment step", flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
