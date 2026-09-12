"""Isaac integration check: single registration, Route B rewards, curriculum stages.

Run through isaaclab.sh -p tests/check_mobile_grasp_env.py --headless.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch

    import eggtart_grasp.tasks  # noqa: F401
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from isaaclab_tasks.utils import parse_env_cfg

    task = "Isaac-Mobile-Grasp-Eggtart-v0"
    bc_task = "Isaac-Mobile-Grasp-Eggtart-BCPPO-v0"
    tasks = {name for name in gym.registry if name.startswith("Isaac-Mobile-Grasp-Eggtart")}
    assert tasks == {task, bc_task}, tasks
    cfg = parse_env_cfg(task, device=args.device, num_envs=4)
    assert type(cfg).__name__ == "EggtartMobileGraspEnvCfg"
    assert cfg.scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    weights = {"base_approach": 1.0, "ee_reach": 2.0, "grasp_posture_guide": 1.5,
               "target_lift_progress": 10.0, "grasp": 30.0, "action_rate": -0.01, "joint_limits": -0.3}
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        assert set(env.reward_manager.active_terms) == set(weights)
        assert len(env.curriculum_manager.active_terms) > 0
        params = cfg.events.reset_target.params
        second, third = params["stage2_start_step"], params["stage3_start_step"]
        robot, target = env.scene["robot"], env.scene["target"]
        for step, expected in ((0, 1), (second - 1, 1), (second, 2), (third - 1, 2), (third, 3)):
            env.common_step_counter = step
            obs, _ = env.reset()
            assert obs["policy"].shape == (4, 44)
            assert env.action_manager.total_action_dim == 9
            assert torch.allclose(target.data.root_vel_w, torch.zeros_like(target.data.root_vel_w))
            local = quat_apply_inverse(robot.data.root_quat_w, target.data.root_pos_w - robot.data.root_pos_w)
            ranges = cfg.events.reset_target.params["stage1_pose_range"] if expected < 3 else cfg.events.reset_target.params["stage3_pose_range"]
            for dim, axis in enumerate("xyz"):
                lo, hi = ranges[axis]
                assert ((local[:, dim] >= lo - 1e-5) & (local[:, dim] <= hi + 1e-5)).all(), local
            for name, weight in weights.items():
                assert env.reward_manager.get_term_cfg(name).weight == weight
            # Place targets inside the activation radius: stage 1 must assist,
            # while stage 2/3 must leave even nearby targets to contact physics.
            event = env.event_manager.get_term_cfg("target_approach_stage1")
            body = event.params["ee_cfg"].body_ids[0]
            offset = torch.tensor(event.params["grasp_offset"], device=env.device).expand(4, -1)
            grasp = robot.data.body_pos_w[:, body] + quat_apply(robot.data.body_quat_w[:, body], offset)
            state = target.data.root_state_w.clone()
            state[:, :3] = grasp + torch.tensor([0.02, 0.02, 0.0], device=env.device)
            state[:, 7:13] = 0.0
            target.write_root_state_to_sim(state)
            event.func(env, torch.arange(4, device=env.device), **event.params)
            speed = torch.linalg.vector_norm(target.data.root_vel_w[:, :2], dim=1)
            expected_speed = event.params["approach_speed"] if expected == 1 else 0.0
            torch.testing.assert_close(speed, torch.full_like(speed, expected_speed), atol=1e-5, rtol=1e-5)
            result = env.step(torch.zeros(4, 9, device=env.device))
            assert torch.isfinite(result[0]["policy"]).all()
            assert torch.isfinite(result[1]).all()
            print(f"PASS: step={step}, stage={expected}, obs=44, action=9, all seven rewards enabled", flush=True)
        print("PASS: unified environment and curriculum integration", flush=True)
    finally:
        env.close()
    bc_cfg = parse_env_cfg(bc_task, device=args.device, num_envs=2)
    assert type(bc_cfg).__name__ == "EggtartMobileGraspEnvBCPPOCfg"
    bc_env = gym.make(bc_task, cfg=bc_cfg).unwrapped
    try:
        assert len(bc_env.curriculum_manager.active_terms) == len(bc_cfg.curriculum.__dict__)
        assert set(bc_env.reward_manager.active_terms) == set(weights)
        print("PASS: BCPPO uses BCCurriculumCfg with Route B weights", flush=True)
    finally:
        bc_env.close()
finally:
    app.close()
