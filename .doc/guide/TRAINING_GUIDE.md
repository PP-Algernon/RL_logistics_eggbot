# 纯状态训练快速启动指南

> **已归档（2026-09-14）**  
> 原路径：`.doc/TRAINING_GUIDE.md`。本文保留当时方案、参数与实验记录，当前操作请见[文档索引](../README.md)和[训练指南](../TODO_BC_PPO_DAPG.md)。

> **历史记录**  
> 当前保留 `Isaac-Mobile-Grasp-Eggtart-v0` 和 `Isaac-Mobile-Grasp-Eggtart-BCPPO-v0`，分别使用普通奖励课程和 `BCCurriculumCfg`。本文的旧配置与命令请以[训练清单](../TODO_BC_PPO_DAPG.md)为准。

本指南帮助你使用**纯状态观测**（无视觉传感器）开始训练移动抓取任务。


## 🚀 开始训练

### 步骤 1: 验证环境已安装

```bash
cd /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot
conda activate my_isaac_env

# 检查环境是否注册
./isaaclab.sh -p scripts/list_envs.py | grep Eggtart
```

应该看到：
```
Isaac-Mobile-Grasp-Eggtart-v0
Isaac-Mobile-Grasp-Eggtart-Play-v0
```

### 步骤 2: 快速可视化测试

先用少量环境测试是否正常运行：

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Play-v0 \
    --num_envs 4
```

**观察**：
- 机器人是否正常加载？
- 目标物体是否出现在合理位置？
- 底盘和机械臂能否正常运动？
- 有无报错？

### 步骤 3: 开始训练（小规模测试）

先用少量环境验证训练流程：

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --num_envs 256 \
    --max_iterations 100 \
    --headless
```

**监控输出**：
- Episode length 是否合理（< 10s）？
- 奖励是否在变化？
- 有无 NaN 或异常值？

### 步骤 4: 全规模训练

确认无误后，启动完整训练：

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --num_envs 2048 \
    --headless
```

**训练日志位置**：
```
logs/rsl_rl/eggtart_mobile_grasp/<timestamp>/
├── config.yaml          # 完整配置
├── model_*.pt           # 模型 checkpoints
└── summaries/           # TensorBoard 日志
```

### 步骤 5: 监控训练进度

启动 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl/eggtart_mobile_grasp
```

在浏览器打开 `http://localhost:6006`

**关键指标**：
- `Loss/value_function`: 应逐渐下降
- `Policy/mean_reward`: 应逐渐上升
- `Policy/mean_episode_length`: 观察是否有增长趋势

---

## ⚙️ 训练配置参数

当前训练超参数（RSL-RL PPO）：

**位置**: [config/eggtart/agents/rsl_rl_ppo_cfg.py](../../source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/config/eggtart/agents/rsl_rl_ppo_cfg.py)

```python
# 训练设置
num_steps_per_env = 24          # 每个环境采样步数
max_iterations = 1500           # 最大迭代次数
save_interval = 50              # 保存间隔

# 网络结构
actor_hidden_dims = [256, 128, 64]
critic_hidden_dims = [256, 128, 64]
activation = "elu"

# PPO 超参数
learning_rate = 1.0e-3
clip_param = 0.2
entropy_coef = 0.005
gamma = 0.99
lam = 0.95
```

---

## 🔧 可能需要的调整

### 如果训练不稳定（NaN/发散）

1. **降低学习率**：
   ```python
   # rsl_rl_ppo_cfg.py
   learning_rate = 5.0e-4  # 从 1e-3 改为 5e-4
   ```

2. **检查奖励尺度**：
   ```bash
   # 在训练开始时观察各项奖励的数值
   # 如果某项特别大，可能需要调整权重
   ```

3. **减小动作幅度**：
   ```python
   # mobile_grasp_env_cfg.py - ActionsCfg
   arm_action = mdp.JointPositionActionCfg(
       scale=0.3,  # 从 0.5 改为 0.3，让机械臂动作更平滑
   )
   ```

## 📋 任务注册说明

### 需要注册新任务

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

---

## 📈 训练监控详解

### 启动训练

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --num_envs 2048
```

### 查看训练曲线（TensorBoard）

训练日志保存在 `logs/rsl_rl/eggtart_mobile_grasp/<时间戳>/`

**启动 TensorBoard**：
```bash
tensorboard --logdir logs/rsl_rl/eggtart_mobile_grasp
```

在浏览器打开 `http://localhost:6006`

**重点关注的指标**：

1. **Loss（Scalars）**：
   - `Loss/value_function`：价值函数拟合误差，应逐渐下降
   - `Loss/surrogate`：策略梯度损失，前期波动较大，后期趋于稳定

2. **Reward（Scalars）**：
   - `Train/mean_reward`：平均总奖励，应逐渐上升
   - `Train/mean_episode_length`：平均 episode 长度，成功任务会提前终止
   - `Reward/<各奖励项名称>`：查看各个奖励项的贡献（如 `base_approach`, `ee_reach`, `grasp` 等）

3. **Policy（Scalars）**：
   - `Policy/mean_std`：动作标准差，初期探索高，后期应下降
   - `Policy/entropy`：策略熵，衡量探索程度

4. **Misc（Scalars）**：
   - `Perf/total_time`：训练时间
   - `Perf/fps`：仿真帧率

### 判断训练效果

- **正常收敛**：
  - `mean_reward` 持续上升并趋于平稳
  - `value_function` loss 逐渐下降到较低值（< 1.0）
  - 各阶段奖励按顺序激活：`base_approach` → `ee_reach` → `grasp` → `retract`

- **需要调整**：
  - `mean_reward` 停滞不前或下降 → 检查奖励权重是否合理
  - 某个奖励项始终为 0 → 该行为未被触发，可能需要降低前置条件难度
  - `Policy/mean_std` 过早降到 0 → 探索不足，可能陷入局部最优

---

## 🎛️ 超参数调优

**文件位置**：`source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/config/eggtart/agents/rsl_rl_ppo_cfg.py`

**常用调整项**：

```python
# 学习率
learning_rate = 1e-3  # 如果训练不稳定可降低到 5e-4

# Clip range
clip_param = 0.2  # PPO clip 范围，影响策略更新幅度

# Episode 长度
# 在 mobile_grasp_env_cfg.py 的 __post_init__ 中
self.episode_length_s = 10.0  # 如果任务太难可延长到 15.0

# 奖励权重
# 在 RewardsCfg 中调整各项 weight
```

---

## 🎨 奖励设计教程

### 原理

强化学习通过奖励函数引导智能体学习期望行为。好的奖励函数应该：

1. **分阶段**：任务分解为子目标（接近 → 到达 → 抓取 → 回收）
2. **密集反馈**：每一步都有信号，避免稀疏奖励
3. **形状良好**：使用 tanh/exp 等平滑函数，避免阶跃
4. **权重平衡**：各阶段奖励量级相当，避免某项过度主导

### 实现流程

#### Step 1: 定义奖励函数

在 `source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/rewards.py` 中添加：

```python
def my_custom_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """你的奖励函数描述
    
    Args:
        env: 环境实例
        std: 平滑参数（用于 tanh/exp 形状）
        robot_cfg: 机器人实体配置
        target_cfg: 目标物体配置
    
    Returns:
        shape (num_envs,) 的奖励张量
    """
    # 1. 获取需要的状态
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    robot_pos = robot.data.root_pos_w  # (num_envs, 3)
    target_pos = target.data.root_pos_w  # (num_envs, 3)
    
    # 2. 计算度量（如距离、角度等）
    distance = torch.norm(robot_pos - target_pos, dim=1)  # (num_envs,)
    
    # 3. 形状函数（返回 [0, 1] 或 [-1, 1]）
    reward = torch.exp(-distance / std)  # 距离越近奖励越高
    
    return reward
```

#### Step 2: 导出函数

在 `rewards.py` 最后添加到 `__all__`：

```python
__all__ = [
    "base_to_target_xy_tanh",
    "ee_to_target_tanh",
    # ... 其他已有的
    "my_custom_reward",  # 你的新函数
]
```

#### Step 3: 添加到环境配置

在 `mobile_grasp_env_cfg.py` 的 `RewardsCfg` 类中添加：

```python
@configclass
class RewardsCfg:
    # ... 其他奖励项
    
    my_custom = RewTerm(
        func=mdp.my_custom_reward,
        weight=1.0,  # 权重，可以是负数（惩罚）
        params={
            "std": 0.5,
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
        },
    )
```

#### Step 4: 测试和调整

1. **启动训练**，观察 TensorBoard 中 `Reward/my_custom` 的曲线
2. **调整 `weight`**：
   - 如果该项始终为 0 → 检查计算逻辑或前置条件
   - 如果该项过度主导总奖励 → 降低权重
   - 如果该项对总奖励无影响 → 提高权重
3. **调整 `std`**：
   - 太小 → 奖励变化陡峭，容易陷入局部最优
   - 太大 → 奖励变化平缓，学习信号弱

### 常用形状函数

```python
# 1. Tanh（距离奖励，对称）
reward = torch.tanh(-distance / std)  # 范围 [-1, 1]

# 2. Exp（距离奖励，非对称）
reward = torch.exp(-distance / std)  # 范围 [0, 1]

# 3. 二值触发（达到阈值）
reward = (distance < threshold).float()  # 0 或 1

# 4. 线性惩罚
reward = -torch.abs(value)  # 越偏离 0 惩罚越大

# 5. 组合（分段奖励）
close_bonus = torch.where(distance < 0.1, 1.0, 0.0)
approach_reward = torch.exp(-distance / std)
reward = approach_reward + close_bonus
```

### 示例：角度对齐奖励

```python
def angle_alignment_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """奖励机器人朝向目标方向"""
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    # 机器人朝向（+X 轴在世界系中的方向）
    robot_quat = robot.data.root_quat_w
    robot_forward = quat_rotate(robot_quat, torch.tensor([1, 0, 0], device=env.device))
    
    # 目标方向
    to_target = target.data.root_pos_w - robot.data.root_pos_w
    to_target = to_target / (torch.norm(to_target, dim=1, keepdim=True) + 1e-6)
    
    # 余弦相似度（点积）
    alignment = torch.sum(robot_forward * to_target, dim=1)  # [-1, 1]
    
    return alignment  # 1 = 完全对齐，-1 = 背对
```

---

## 📌 下一步工作

1. **训练并观察曲线**
   - 启动训练，运行至少 500 iterations
   - 查看各奖励项是否按预期激活
   
2. **调整奖励权重**
   - 如果底盘不朝向目标 → 增加 `base_facing` 权重
   - 如果抓取成功率低 → 增加 `grasp` 和 `ee_reach` 权重
   - 如果动作抖动严重 → 增加 `action_rate` 惩罚权重

3. **添加更多奖励项**（参考上面的教程）

4. **超参数扫描**（可选）
   - 不同学习率：`5e-4`, `1e-3`, `2e-3`
   - 不同 episode 长度：`8s`, `10s`, `15s`
   - 使用 wandb 或手动记录对比

---

## ❓ 常见问题

**Q: 训练很慢怎么办？**  
A: 
- 降低 `num_envs`（如 1024），减少内存占用
- 检查 CPU/GPU 负载，确保资源充足
- 使用 `--headless` 跳过渲染

**Q: 策略不收敛？**  
A:
- 检查奖励函数是否有 NaN/Inf（打印 reward 张量）
- 降低学习率到 `5e-4`
- 延长 episode 长度，给更多探索时间
- 简化任务（如先固定目标位置，不随机移动）

**Q: 如何恢复训练？**  
A:
```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --load_run <RUN_NAME> \
    --checkpoint <ITERATION>
```

**Q: 如何导出模型到真实机器人？**  
A: 训练完成后，在 `logs/.../exported/` 目录会自动生成：
- `policy.pt`：JIT 模型
- `policy.onnx`：ONNX 模型（可用于 C++ 推理）

参考 sim2sim 方案文档进行部署。
---

祝训练顺利！🚀


### 如果训练太慢

1. **减少环境数量**（牺牲样本效率）：
   ```bash
   --num_envs 1024  # 从 2048 降到 1024
   ```

2. **减少仿真精度**（牺牲物理准确性）：
   ```python
   # mobile_grasp_env_cfg.py
   self.sim.dt = 1.0 / 60.0  # 从 120Hz 改为 60Hz
   self.decimation = 2       # 从 4 改为 2
   ```
---

## 🎬 回放训练好的模型

训练完成后，评估策略：

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Play-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/eggtart_mobile_grasp/<timestamp>/model_1500.pt
```

**评估指标**：
- 成功率：多少个环境成功抓取？
- 平均时长：从 reset 到抓取用了多少秒？
- 收臂行为：抓取后是否正确收回？

---

## 🐛 常见问题排查

### Q1: 报错 "Robot asset is MISSING"
**A**: 检查 `eggtart_grasp` 是否安装：
```bash
./isaaclab.sh -p -m pip list | grep eggtart
```

### Q2: 底盘不动 / 机械臂不动
**A**: 检查动作空间配置，确认 `ActionsCfg` 正确。

### Q3: 目标物体掉落或飞走
**A**: 目标已禁用重力且有初速，这是预期行为（移动目标）。

### Q4: 训练曲线震荡剧烈
**A**: 
1. 降低学习率
2. 增加 `num_mini_batches`
3. 检查奖励尺度是否合理

---

## 📝 训练前检查清单

在开始训练前，确认：

- [ ] 环境已安装并注册
- [ ] 可视化测试通过（play.py）
- [ ] 小规模训练无报错（256 envs, 100 iter）
- [ ] TensorBoard 日志正常记录
- [ ] 有足够的磁盘空间（至少 10GB）
- [ ] GPU 显存足够（2048 envs 约需 8-10GB）

---

