# eggtart_grasp

Isaac Lab external extension for the Eggtart mobile manipulator: a 4-wheel omnidirectional base
+ 5-axis arm + force-controlled gripper learning to navigate to a cube on the ground, grasp it,
and lift it.

Install (from the Isaac Lab root):

```bash
./isaaclab.sh -p -m pip install -e <path>/Eggtart-logistics-robot/source/eggtart_grasp
```

Registered tasks:

- `Isaac-Mobile-Grasp-Eggtart-v0` — training (target gets a small random initial velocity)
- `Isaac-Mobile-Grasp-Eggtart-Static-v0` — training with a fully static target (easier)
- `Isaac-Mobile-Grasp-Eggtart-Play-v0` — visualisation / evaluation

See the project root `README.md` for full usage.
