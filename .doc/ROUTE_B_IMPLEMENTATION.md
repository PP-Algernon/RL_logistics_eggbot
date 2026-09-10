# Route B Implementation Guide: BC Pretraining + PPO Fine-tuning

**Status**: ✅ Implementation Complete  
**Date**: 2026-09-10  
**Reference**: `.doc/技术路线/TECH_ROADMAP_BC.md`

---

## Overview

Route B uses **Behavior Cloning (BC) pretraining + PPO fine-tuning** to improve training efficiency without changing the algorithm stack (keeping rsl_rl/PPO).

**Key Idea**:
1. Use scripted demonstrations to teach the policy basic grasping behavior (BC)
2. Fine-tune with PPO using simplified rewards to improve robustness

**Advantages over from-scratch PPO**:
- Faster convergence (grasp rewards appear in 100-200 iters vs 1000+ iters)
- Higher success rate baseline (50-70% from BC alone)
- No algorithm changes needed (compatible with existing rsl_rl pipeline)

---

## File Structure

```
Eggtart-logistics-robot/
├── scripts/
│   ├── collect_demo.py          # NEW: Stage 0 - Demonstration collection
│   ├── bc_pretrain.py            # NEW: Stage 1 - BC pretraining
│   └── rsl_rl/
│       ├── train.py              # Existing: Use with --resume flag
│       └── play.py               # Existing: Evaluate BC/PPO policies
├── source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/
│   ├── route_b_rewards.py        # NEW: Stage 2 - Simplified reward configs
│   └── mobile_grasp_env_cfg.py   # Existing: Base environment config
├── datasets/
│   └── eggtart_demo.hdf5         # Generated: Demonstration data
└── checkpoints/
    └── bc_pretrained.pt          # Generated: BC checkpoint
```

---

## Implementation Stages

### Stage 0: Demonstration Data Collection

**Script**: `scripts/collect_demo.py`

**Purpose**: Collect (observation, action) pairs using a scripted teacher policy.

**Teacher Policy**:
- **Navigation**: P-controller to approach target (compute base velocity from target position)
- **Reaching**: Move arm to grasp posture when target is close
- **Grasping**: Close gripper when target is within reach threshold
- **Lifting**: Retract arm after grasping

**Key Parameters**:
- Environment: `Isaac-Mobile-Grasp-Eggtart-Static-v0` (static target, easiest)
- Num envs: 2048 (parallel collection)
- Max steps: 2000 (yields ~4M samples)
- Action noise: 0.05 (adds exploration diversity)

**Usage**:
```bash
cd /home/pu/isaac-sim
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/collect_demo.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 2048 \
    --max_steps 2000 \
    --noise_scale 0.05 \
    --output datasets/eggtart_demo.hdf5 \
    --headless
```

**Expected Output**:
- Dataset: `datasets/eggtart_demo.hdf5` (~50k-100k samples)
- Demonstration success rate: ≥80%
- If success rate < 80%: Tune teacher policy or reduce noise

**Validation Criteria** (from roadmap §2.4):
- [ ] Demonstration success rate ≥ 80%
- [ ] Trajectory length reasonable (5-10s, no stalling)
- [ ] Action variance non-zero (noise is working)
- [ ] Data quantity ≥ 50k (s,a) pairs

---

### Stage 1: BC Pretraining

**Script**: `scripts/bc_pretrain.py`

**Purpose**: Train actor network using supervised learning on demonstration data.

**Network Structure**:
- **Critical**: Must match rsl_rl's `ActorCritic` exactly
- Actor: `[256, 128, 64]` hidden dims, ELU activation
- Critic: Same structure but stays randomly initialized (PPO will train it)
- Only actor parameters are optimized during BC

**Training Details**:
- Loss: MSE between predicted and demonstrated actions
- Optimizer: Adam, lr=3e-4
- Epochs: 100 (with early stopping)
- Batch size: 4096
- Train/val split: 90/10

**Observation Normalization**:
- Compute mean/std from training data
- Saved in checkpoint metadata for consistency with PPO

**Usage**:
```bash
cd /home/pu/isaac-sim
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/bc_pretrain.py \
    --data datasets/eggtart_demo.hdf5 \
    --output checkpoints/bc_pretrained.pt \
    --actor_hidden_dims 256 128 64 \
    --critic_hidden_dims 256 128 64 \
    --activation elu \
    --lr 3e-4 \
    --epochs 100 \
    --batch_size 4096
```

**Expected Output**:
- Checkpoint: `checkpoints/bc_pretrained.pt` (rsl_rl format)
- Training loss: Should converge to low value (< 0.1)
- Validation loss: Should not diverge (overfitting check)

**Testing BC Policy**:
```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 32 \
    --checkpoint checkpoints/bc_pretrained.pt
```

**Validation Criteria** (from roadmap §3.3):
- [ ] BC policy success rate > 0% (ideally ≥50%)
- [ ] Checkpoint loads successfully into rsl_rl
- [ ] No errors when running play.py with BC weights

---

### Stage 2: PPO Fine-tuning

**Configuration**: `source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/route_b_rewards.py`

**Purpose**: Fine-tune BC policy with PPO using simplified rewards.

**Reward Configuration 1** (Recommended - `RouteBRewardsCfg`):
Keep only 5 core terms with reduced weights:

| Term | Original Weight | Route B Weight | Purpose |
|------|----------------|----------------|---------|
| `base_approach` | 2.5 | 1.0 | Base navigation |
| `ee_reach` | 5.0 | 2.0 | End-effector reaching |
| `grasp_posture_guide` | 4.0 | 1.5 | Arm configuration |
| `target_lift_progress` | 30.0 | 10.0 | Lifting progress |
| `grasp` (sparse) | 30.0 | 30.0 | Success reward |

All other dense rewards disabled (arm_comfort, ee_precision, base_facing, etc.)

**Reward Configuration 2** (Aggressive - `RouteBRewardsCfg_Aggressive`):
- Only `grasp` (sparse success) + minimal `ee_reach` (weight=0.5)
- Use after Configuration 1 proves successful
- Closer to pure sparse formulation from papers

**Curriculum Changes**:
- **No reward weight scheduling** (all active from step 0)
- BC warm-start means policy already knows how to grasp
- Keep target distribution curriculum (Stage 1→2→3)

**PPO Hyperparameters**:
- **Learning rate**: **3e-4** (reduced from 1e-3 to prevent forgetting)
- Other params unchanged: entropy=0.005, clip=0.2, etc.

**Usage**:
```bash
cd /home/pu/isaac-sim
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 2048 \
    --max_iterations 4000 \
    --resume \
    --load_checkpoint /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/checkpoints/bc_pretrained.pt \
    --headless
```

**Note**: Currently uses base rewards. To use Route B rewards, create environment variant:
1. Add new config in `config/eggtart/grasp_env_cfg.py` that uses `RouteBRewardsCfg`
2. Register as `Isaac-Mobile-Grasp-Eggtart-Static-RouteB-v0`
3. Use that task name in training command

**Expected Behavior**:
- Grasp rewards appear **early** (100-200 iters vs 1000+ from scratch)
- Success rate starts from BC baseline (~50-70%)
- Monotonic improvement (no catastrophic forgetting)
- Final success rate ≥80% on static targets

**Validation Criteria** (from roadmap §4.5):
- [ ] Success rate ≥ 80% (static target)
- [ ] Better than pure PPO from scratch
- [ ] Better than BC without fine-tuning
- [ ] No catastrophic forgetting (success rate doesn't drop)

---

## Quick Start Guide

### Complete Pipeline (3 Commands)

```bash
# Terminal should be in /home/pu/isaac-sim
# Use conda env: my_isaac_env

# 1. Collect demonstrations (~10 min)
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/collect_demo.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 2048 --max_steps 2000 --noise_scale 0.05 \
    --output /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/datasets/eggtart_demo.hdf5 \
    --headless

# 2. BC pretraining (~30 min)
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/bc_pretrain.py \
    --data /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/datasets/eggtart_demo.hdf5 \
    --output /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/checkpoints/bc_pretrained.pt \
    --epochs 100

# 3. PPO fine-tuning (~4-8 hours for 4000 iters)
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 2048 --max_iterations 4000 \
    --resume --load_checkpoint /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/checkpoints/bc_pretrained.pt \
    --headless
```

---

## Monitoring & Debugging

### Key TensorBoard Metrics

**Early Training (0-500 iters)**:
- `Rewards/grasp`: Should be **non-zero early** (BC advantage)
- `Rewards/ee_reach`: Should be positive and stable
- `Rewards/base_approach`: Should guide navigation
- `Train/learning_rate`: Confirm it's 3e-4 (not 1e-3)

**Mid Training (500-2000 iters)**:
- Success rate: Should be climbing from BC baseline
- `Rewards/grasp`: Should increase over time
- No sudden drops (catastrophic forgetting check)

**Late Training (2000-4000 iters)**:
- Success rate: Target ≥80%
- All reward terms: Should be stable or improving

### Common Issues

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| Demo success rate < 80% | Teacher policy broken | Check target distribution, P-controller gains |
| BC success rate < 50% | Poor demo quality or network mismatch | Collect more/better data, verify network structure |
| Catastrophic forgetting | Learning rate too high | Lower to 1e-4 or 5e-5 |
| No improvement over BC | Insufficient guidance | Use Configuration 1 instead of 2 |
| Grasp reward stays zero | Observation normalization mismatch | Check BC saved obs_mean/std |

---

## Comparison: Route B vs From-Scratch PPO

| Metric | From-Scratch PPO | Route B (BC+PPO) |
|--------|------------------|------------------|
| First grasp event | ~1000 iters | ~100 iters |
| 50% success rate | ~2000 iters | ~500 iters (starts near this) |
| 80% success rate | ~4000 iters (if reached) | ~2000 iters |
| Data efficiency | Low (on-policy only) | High (demos + on-policy) |
| Implementation | No changes | +2 scripts, +1 config |

---

## Next Steps

### After Stage 2 Success (Static Target ≥80%)

1. **Transfer to Moving Target**:
   - Collect demos with `Isaac-Mobile-Grasp-Eggtart-v0` (moving target)
   - Re-run BC + PPO pipeline
   - Target: ≥60% success rate (harder task)

2. **Route B Advanced (Optional)**:
   - Try Configuration 2 (aggressive sparse rewards)
   - Implement DAPG regularization (see §5 in roadmap)
   - Add BC loss term to PPO objective to prevent forgetting

3. **Route A (If More Improvement Needed)**:
   - Switch to off-policy (TD3/SAC via skrl)
   - Use demonstration data in replay buffer
   - Pure sparse rewards
   - See `.doc/技术路线/TECH_ROADMAP_OFFLINE2ONLINE.md`

---

## References

- Roadmap: `.doc/技术路线/TECH_ROADMAP_BC.md`
- DAPG Paper: https://arxiv.org/abs/1709.10087
- rsl_rl: https://github.com/leggedrobotics/rsl_rl
- Isaac Lab: https://github.com/isaac-sim/IsaacLab

---

## Changelog

- **2026-09-10**: Initial implementation complete
  - ✅ Stage 0: `collect_demo.py`
  - ✅ Stage 1: `bc_pretrain.py`
  - ✅ Stage 2: `route_b_rewards.py`
  - ⏳ Stage 3: DAPG regularization (optional, not yet implemented)
