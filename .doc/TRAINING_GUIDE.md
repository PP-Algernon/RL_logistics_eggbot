> 历史记录：2026-09-12 配置合并后，本文的旧环境名、配置类和奖励调度不再适用。当前唯一环境为 `Isaac-Mobile-Grasp-Eggtart-v0`，请以[训练清单](TODO_BC_PPO_DAPG.md)为准。

# 纯状态训练快速启动指南

本指南帮助你使用**纯状态观测**（无视觉传感器）开始训练移动抓取任务。

---

## **✅ 当前奖励设计**

你的奖励函数已经配置完善，包含 4 个阶段性奖励 + 2 个惩罚项：

### **阶段性奖励**

| 阶段 | 奖励项 | 权重 | 说明 |
|------|--------|------|------|
| ① | `base_approach` | 1.0 | 底盘在 XY 平面接近目标（tanh shaping） |
| ② | `ee_reach` | 2.0 | 末端执行器接近目标（tanh shaping） |
| ② | `ee_distance` | -0.1 | 末端到目标的 L2 距离惩罚 |
| ③ | `grasp` | 5.0 | 抓取 bonus（末端靠近 **且** 夹爪闭合） |
| ④ | `retract` | 2.0 | 抓取后机械臂收回到 home 姿态 |

### **正则化惩罚**

| 惩罚项 | 权重 | 说明 |
|--------|------|------|
| `action_rate` | -0.001 | 动作平滑性惩罚（相邻帧动作差） |
| `joint_vel` | -0.0001 | 机械臂关节速度惩罚 |

**位置**: [mobile_grasp_env_cfg.py:212-242](eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py#L212-L242)

---

## **🎯 观测空间（纯状态）**

当前观测包含以下项（**无图像**）：

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `joint_pos` | N | 所有关节的相对位置 |
| `joint_vel` | N | 所有关节的相对速度 |
| `base_lin_vel` | 3 | 底盘线速度（体坐标系） |
| `base_ang_vel` | 3 | 底盘角速度（体坐标系） |
| `target_position_b` | 3 | 目标在底盘坐标系下的位置 |
| `ee_to_target_b` | 3 | 末端到目标的向量（底盘系） |
| `target_lin_vel` | 3 | 目标的世界系线速度 |
| `actions` | M | 上一步的动作 |

**所有观测都在底盘坐标系下**，策略对底盘世界位姿不变。

**位置**: [mobile_grasp_env_cfg.py:106-139](eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py#L106-L139)

---

## **🚀 开始训练**

### **步骤 1: 验证环境已安装**

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

### **步骤 2: 快速可视化测试**

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

### **步骤 3: 开始训练（小规模测试）**

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

### **步骤 4: 全规模训练**

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

### **步骤 5: 监控训练进度**

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

## **⚙️ 训练配置参数**

当前训练超参数（RSL-RL PPO）：

**位置**: [config/eggtart/agents/rsl_rl_ppo_cfg.py](eggtart_grasp/tasks/mobile_grasp/config/eggtart/agents/rsl_rl_ppo_cfg.py)

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

## **🔧 可能需要的调整**

### **如果训练不稳定（NaN/发散）**

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

### **如果训练太慢**

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

### **如果机器人学不会抓取**

1. **增大抓取奖励权重**：
   ```python
   # mobile_grasp_env_cfg.py - RewardsCfg
   grasp = RewTerm(..., weight=10.0)  # 从 5.0 改为 10.0
   ```

2. **放宽抓取判定阈值**：
   ```python
   # mobile_grasp_env_cfg.py
   GRASP_REACH_THRESHOLD = 0.08  # 从 0.05 改为 0.08（更宽松）
   ```

3. **使用课程学习**（见 TODO_GUIDE.md）

---

## **📊 预期训练效果**

### **训练时间**

| 环境数 | GPU | 迭代/分钟 | 收敛时间（估算） |
|--------|-----|----------|----------------|
| 2048 | RTX 3090 | ~3-5 | 1-2 小时 |
| 1024 | RTX 3070 | ~2-3 | 2-3 小时 |
| 512 | RTX 2080 Ti | ~1-2 | 3-4 小时 |

### **训练阶段**

1. **前 200 iter**: 学习底盘移动，靠近目标
2. **200-600 iter**: 学习末端伸展，接近目标
3. **600-1000 iter**: 学习夹爪闭合，触发抓取 bonus
4. **1000+ iter**: 学习抓取后收臂，完整任务

---

## **🎬 回放训练好的模型**

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

## **🐛 常见问题排查**

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

## **📝 训练前检查清单**

在开始训练前，确认：

- [ ] 环境已安装并注册
- [ ] 可视化测试通过（play.py）
- [ ] 小规模训练无报错（256 envs, 100 iter）
- [ ] TensorBoard 日志正常记录
- [ ] 有足够的磁盘空间（至少 10GB）
- [ ] GPU 显存足够（2048 envs 约需 8-10GB）

---

## **🎯 下一步优化方向**

训练出基础策略后，可以考虑：

1. **标定麦克纳姆轮几何参数**（TODO 1）
   - 运行 `calibrate_mecanum.py`
   - 提升底盘运动准确性

2. **调整奖励权重**（TODO 5）
   - 根据训练曲线微调权重
   - 尝试课程学习

3. **添加更多观测项**
   - 底盘到目标的距离
   - 夹爪状态（开/合程度）

4. **考虑视觉传感器**（可选）
   - 参考 VISION_TRAINING.md
   - 为 sim-to-real 做准备

---

## **📚 相关文档**

- [TODO_GUIDE.md](TODO_GUIDE.md) - 完成待办事项指南
- [RELATED_PROJECTS.md](source/eggtart_grasp/RELATED_PROJECTS.md) - 相关开源项目
- [README.md](README.md) - 项目总览

---

祝训练顺利！🚀
