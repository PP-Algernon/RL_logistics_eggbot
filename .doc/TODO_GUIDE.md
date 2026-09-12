> 历史记录：2026-09-12 配置合并后，本文的旧环境名、配置类和奖励调度不再适用。当前唯一环境为 `Isaac-Mobile-Grasp-Eggtart-v0`，请以[训练清单](TODO_BC_PPO_DAPG.md)为准。

# TODO Guide - Eggtart Mobile Grasp 训练优化

## 当前状态

✅ **已完成**：
- URDF 加载和可视化（wheel_001 几何问题已修复）
- 全向底盘控制（`HolonomicBaseAction`，不再模拟麦轮动力学）
- 基础环境配置（场景、观测、动作、奖励、终止条件）
- 底盘质量调整（8.0 kg 防止倾倒）
- Play 脚本可以正常加载 checkpoint 运行

## 当前阶段：调参优化

### 1. 奖励函数优化 🔧

**目标**：细化奖励项，引导策略学习期望行为

#### 待添加的奖励项

- [x] **底盘朝向目标奖励**（已实现，见下文）
- [ ] 机械臂关节限位惩罚
- [ ] 底盘加速度平滑惩罚（避免抖动）
- [ ] 抓取成功后保持目标高度奖励
- [ ] 碰撞惩罚（如果需要）

#### 已实现：底盘朝向目标奖励

**文件位置**：`source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/rewards.py`

新增函数：`base_facing_target()`

**使用方法**：在 `mobile_grasp_env_cfg.py` 的 `RewardsCfg` 中添加：
```python
base_facing = RewTerm(
    func=mdp.base_facing_target,
    weight=0.5,
    params={
        "std": 0.3,
        "robot_cfg": SceneEntityCfg("robot"),
        "target_cfg": SceneEntityCfg("target"),
    },
)
```

---

### 2. 训练监控

#### 启动训练

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --num_envs 2048
```

#### 查看训练曲线（TensorBoard）

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

#### 判断训练效果

- **正常收敛**：
  - `mean_reward` 持续上升并趋于平稳
  - `value_function` loss 逐渐下降到较低值（< 1.0）
  - 各阶段奖励按顺序激活：`base_approach` → `ee_reach` → `grasp` → `retract`

- **需要调整**：
  - `mean_reward` 停滞不前或下降 → 检查奖励权重是否合理
  - 某个奖励项始终为 0 → 该行为未被触发，可能需要降低前置条件难度
  - `Policy/mean_std` 过早降到 0 → 探索不足，可能陷入局部最优

---

### 3. 超参数调优

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

## 奖励设计教程

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

## 下一步工作

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

## 常见问题

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
