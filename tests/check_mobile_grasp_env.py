"""Isaac integration check for task registration, rewards and collection-aligned resets.

Run through isaaclab.sh -p with --task and --headless, once per task.
"""
import argparse
import traceback

from isaaclab.app import AppLauncher

STANDARD_TASK = "Isaac-Mobile-Grasp-Eggtart-v0"
BC_TASK = "Isaac-Mobile-Grasp-Eggtart-BCPPO-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=(STANDARD_TASK, BC_TASK), default=BC_TASK)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch

    import eggtart_grasp.tasks  # noqa: F401
    from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import BCCurriculumCfg, CurriculumCfg
    from isaaclab.utils.math import quat_apply
    from isaaclab_tasks.utils import parse_env_cfg

    tasks = {name for name in gym.registry if name.startswith("Isaac-Mobile-Grasp-Eggtart")}
    assert tasks == {STANDARD_TASK, BC_TASK}, tasks
    standard_cfg = parse_env_cfg(STANDARD_TASK, device=args.device, num_envs=16)
    bc_cfg = parse_env_cfg(BC_TASK, device=args.device, num_envs=16)
    assert type(standard_cfg.curriculum) is CurriculumCfg
    assert type(bc_cfg.curriculum) is BCCurriculumCfg
    cfg = bc_cfg if args.task == BC_TASK else standard_cfg
    assert cfg.events.target_approach_stage1 is None
    assert cfg.events.randomize_target_velocity is None
    weights = {
        "base_approach": 1.0, "ee_reach": 2.0, "grasp_posture_guide": 1.5,
        "target_lift_progress": 10.0, "grasp": 30.0, "action_rate": -0.01, "joint_limits": -0.3,
    }
    env = gym.make(args.task, cfg=cfg).unwrapped
    params = env.event_manager.get_term_cfg("reset_target").params
    second, third = params["stage2_start_step"], params["stage3_start_step"]
    robot, target = env.scene["robot"], env.scene["target"]
    link = params["reference_cfg"].body_ids[0]
    assert len(env.reward_manager.active_terms) == 15
    assert env.action_manager.total_action_dim == 9
    assert "interval" not in env.event_manager.available_modes

    for step, stage in ((0, 1), (second - 1, 1), (second, 2), (third - 1, 2), (third, 3)):
        env.common_step_counter = step
        obs, _ = env.reset()
        assert obs["policy"].shape == (16, 44)
        torch.testing.assert_close(target.data.root_vel_w, torch.zeros_like(target.data.root_vel_w))
        link_pos = robot.data.body_pos_w[:, link]
        link_quat = robot.data.body_quat_w[:, link]
        if stage < 3:
            # Independent copy of the original teacher's placement formula.
            offset = torch.tensor([0.0, -0.5, 0.0], device=env.device).expand(16, -1)
            expected = link_pos + quat_apply(link_quat, offset)
            expected[:, 2] = 0.10
            torch.testing.assert_close(target.data.root_pos_w, expected, atol=1e-5, rtol=0)
        else:
            # Z is world height, so recover local XY through its horizontal projection.
            unit_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(16, -1)
            unit_y = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(16, -1)
            basis = torch.stack((quat_apply(link_quat, unit_x)[:, :2],
                                 quat_apply(link_quat, unit_y)[:, :2]), dim=-1)
            local_xy = torch.linalg.solve(basis, (target.data.root_pos_w - link_pos)[:, :2])
            coordinates = torch.cat((local_xy, target.data.root_pos_w[:, 2:3]), dim=1)
            for dim, axis in enumerate("xyz"):
                lo, hi = params["stage3_pose_range"][axis]
                assert ((coordinates[:, dim] >= lo - 1e-5) & (coordinates[:, dim] <= hi + 1e-5)).all()
            assert local_xy[:, 0].max() - local_xy[:, 0].min() > 0.1
            assert local_xy[:, 1].max() - local_xy[:, 1].min() > 0.1

        if args.task == BC_TASK:
            actual = {name: env.reward_manager.get_term_cfg(name).weight
                      for name in env.reward_manager.active_terms
                      if env.reward_manager.get_term_cfg(name).weight != 0}
            assert actual == weights, actual
        else:
            assert env.reward_manager.get_term_cfg("grasp").weight == (0.0 if step == 0 else 30.0)
        result = env.step(torch.zeros(16, 9, device=env.device))
        assert torch.isfinite(result[0]["policy"]).all()
        assert torch.isfinite(result[1]).all()
        print(f"PASS: {args.task}, step={step}, stage={stage}, placement/velocity/rewards", flush=True)

    # A partial reset must not teleport targets belonging to other episodes.
    before = target.data.root_state_w.clone()
    env.common_step_counter = 0
    ids = torch.tensor([1, 5, 11], device=env.device)
    env._reset_idx(ids)
    untouched = torch.ones(16, dtype=torch.bool, device=env.device)
    untouched[ids] = False
    torch.testing.assert_close(target.data.root_state_w[untouched], before[untouched])
    offset = torch.tensor([0.0, -0.5, 0.0], device=env.device).expand(len(ids), -1)
    expected = robot.data.body_pos_w[ids, link] + quat_apply(robot.data.body_quat_w[ids, link], offset)
    expected[:, 2] = 0.10
    torch.testing.assert_close(target.data.root_pos_w[ids], expected, atol=1e-5, rtol=0)
    assert standard_cfg.curriculum.grasp_sched.params["schedule"][0] == (0, 0.0)
    assert bc_cfg.curriculum.grasp_sched.params["schedule"] == [(0, 30.0)]
    print(f"PASS: {args.task} integration complete, including partial resets", flush=True)
    env.close()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
