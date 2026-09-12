# eggtart_grasp

Isaac Lab extension for the Eggtart mobile manipulator: an omnidirectional base,
5-axis arm and force-controlled gripper grasping a ground target.

Install from the Isaac Lab root:

```bash
./isaaclab.sh -p -m pip install -e <project>/source/eggtart_grasp
```

The registered tasks are `Isaac-Mobile-Grasp-Eggtart-v0` and
`Isaac-Mobile-Grasp-Eggtart-BCPPO-v0`. Both use
`eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py:MobileGraspEnvCfg`
with the standard reward schedule in `CurriculumCfg` or the Route B weights in
`BCCurriculumCfg`. The first two target stages match collection: 0.5 m forward
from link_001, zero lateral offset, world Z 0.10 m and zero initial velocity.
Stage 3 randomizes the position. Target motion assistance is disabled.

See the project [README](../../README.md) and
[training checklist](../../.doc/TODO_BC_PPO_DAPG.md) for commands and migration notes.
