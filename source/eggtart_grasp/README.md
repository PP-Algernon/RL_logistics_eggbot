# eggtart_grasp

Isaac Lab extension for an omnidirectional mobile manipulator with a 5-axis arm and force-controlled gripper. Documentation checked against the workspace on 2026-09-14.

Install from the Isaac Lab root:

```bash
./isaaclab.sh -p -m pip install -e <project>/source/eggtart_grasp
```

Two tasks are registered: `Isaac-Mobile-Grasp-Eggtart-v0` uses `CurriculumCfg`, and `Isaac-Mobile-Grasp-Eggtart-BCPPO-v0` uses `BCCurriculumCfg`. Both share [MobileGraspEnvCfg](eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py).

The first two target stages start 0.5 m forward of link_001, with zero lateral offset, world Z 0.10 m and zero initial velocity. Training stage 1 includes fading target tracking assistance; stage 2 disables it and stage 3 randomizes the initial position. Collection and automated evaluation disable assistance. The current default episode duration is 5.5 s.

See the [project README](../../README.md), [documentation index](../../.doc/README.md), [configuration reference](../../.doc/CONFIGURATION.md), [training guide](../../.doc/TODO_BC_PPO_DAPG.md), and [evaluation guide](../../.doc/EVALUATION.md). Earlier guides and experiment notes are in the [archive](../../.doc/archive/README.md).
