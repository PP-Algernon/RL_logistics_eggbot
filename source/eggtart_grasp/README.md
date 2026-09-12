# eggtart_grasp

Isaac Lab extension for the Eggtart mobile manipulator: an omnidirectional base,
5-axis arm and force-controlled gripper grasping a ground target.

Install from the Isaac Lab root:

```bash
./isaaclab.sh -p -m pip install -e <project>/source/eggtart_grasp
```

The only registered task is `Isaac-Mobile-Grasp-Eggtart-v0`. It uses
`eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py:MobileGraspEnvCfg`
for Route B rewards and the reverse target curriculum, shared by collection,
PPO/DAPG training and policy playback.

See the project [README](../../README.md) and
[training checklist](../../.doc/TODO_BC_PPO_DAPG.md) for commands and migration notes.
