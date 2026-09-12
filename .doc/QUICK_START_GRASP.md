> 历史记录：当前保留 `Isaac-Mobile-Grasp-Eggtart-v0` 和 `Isaac-Mobile-Grasp-Eggtart-BCPPO-v0`，分别使用普通奖励课程和 `BCCurriculumCfg`。本文的旧配置与命令请以[训练清单](TODO_BC_PPO_DAPG.md)为准。

# 快速开始：新的提起式抓取训练

## 核心变化

1. ✅ 目标恢复重力，现在会落地
2. ✅ 移动速度降低 60% (±0.25 → ±0.10 m/s)
3. ✅ 抓取判定改为：**提起到 0.35m 并保持 0.3 秒**
4. ✅ 提供两个版本：静止目标 vs 移动目标

## 立即开始训练

### 方案 A: 静止目标（推荐新手）

```bash
conda activate my_isaac_env
cd /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot
source /home/pu/isaac-sim/setup_conda_env.sh

./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 2048 \
    --max_iterations 1500
```

### 方案 B: 移动目标（完整任务）

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --num_envs 2048 \
    --max_iterations 2500
```

## 需要注册新任务

在你的任务注册文件中（通常是 `agents/__init__.py`），添加：

```python
from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import (
    MobileGraspEnvCfg,
    MobileGraspEnvStaticCfg,
)

# 如果文件中已有类似的注册代码，找到并添加：
gym.register(
    id="Eggtart-Mobile-Grasp-Static-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": MobileGraspEnvStaticCfg},
)
```

或者测试不注册直接使用（用于调试）：

```bash
python3 -c "
import gymnasium as gym
from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import MobileGraspEnvStaticCfg
print('Config loaded successfully!')
print('LIFT_HEIGHT_THRESHOLD:', MobileGraspEnvStaticCfg.LIFT_HEIGHT_THRESHOLD)
"
```

## 监控关键指标

启动 TensorBoard:
```bash
tensorboard --logdir logs/rsl_rl/
```

关注：
- `Rewards/grasp`: 提起奖励（iter 500 后启用，应该逐渐上升）
- `Rewards/ee_reach`: EE 到达奖励（应该在 iter 375 左右启用）
- `Curriculum/*_sched`: 各阶段权重变化

## 参数调整

如果训练不理想，可以在 `mobile_grasp_env_cfg.py` 中调整：

```python
# 降低高度要求（更容易成功）
LIFT_HEIGHT_THRESHOLD = 0.25  # 默认 0.35

# 减少保持时间（更容易触发）
LIFT_DWELL_TIME = 0.2  # 默认 0.3

# 进一步降低目标速度（更容易追踪）
TARGET_VELOCITY_RANGE = 0.05  # 默认 0.10
```

## 从静止迁移到移动

先用静止目标训练到收敛（~1000 iters），然后加载权重继续训练移动目标：

```bash
# 步骤 1: 训练静止版本
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --max_iterations 1000

# 步骤 2: 迁移到移动版本
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --resume \
    --load_run <static_run_name> \
    --max_iterations 2000
```

## 文件修改清单

✅ 已修改：
- `mobile_grasp_env_cfg.py`: 主配置文件
- `mdp/rewards.py`: 新增 `GraspBonusLift` 类

📝 需要检查：
- `agents/__init__.py`: 添加任务注册（如果还没有）

## 完整文档

详细说明见 `CHANGES_GRASP_REWARD.md`
