# 路线 B 实施指南：BC 预训练 + PPO 微调

**状态**：✅ 实施完成
**日期**：2026-09-10
**参考**：`.doc/技术路线/TECH_ROADMAP_BC.md`

---

## 概述

路线 B 采用**行为克隆（BC）预训练 + PPO 微调**来提升训练效率，且不更换算法栈（沿用 rsl_rl/PPO）。

**核心思路**：
1. 用脚本化演示教会策略基本的抓取行为（BC）；
2. 使用精简奖励用 PPO 微调，提升策略鲁棒性。

**相较从零开始训练 PPO 的优势**：
- 收敛更快（grasp 奖励在 100–200 迭代内出现，而从零训练需 1000+ 迭代）；
- 成功率起点更高（仅 BC 即可达到 50–70%）；
- 无需改动算法（与现有 rsl_rl 管线完全兼容）。

---

## 文件结构

```
Eggtart-logistics-robot/
├── scripts/
│   ├── collect_demo.py          # 新增：阶段 0 - 演示数据采集
│   ├── bc_pretrain.py            # 新增：阶段 1 - BC 预训练
│   └── rsl_rl/
│       ├── train.py              # 现有：配合 --resume 参数使用
│       └── play.py               # 现有：评估 BC/PPO 策略
├── source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/
│   ├── route_b_rewards.py        # 新增：阶段 2 - 精简奖励配置
│   └── mobile_grasp_env_cfg.py   # 现有：基础环境配置
├── datasets/
│   └── eggtart_demo.hdf5         # 生成：演示数据
└── checkpoints/
    └── bc_pretrained.pt          # 生成：BC 预训练权重
```

---

## 实施阶段

### 阶段 0：演示数据采集

**脚本**：`scripts/collect_demo.py`

**目的**：使用脚本化老师策略采集 (observation, action) 样本对。

**老师策略**：
- **导航**：P 控制器接近目标（根据目标位置计算底盘速度指令）；
- **到达**：目标接近时，将机械臂移动至抓取姿态；
- **抓取**：目标进入可抓取阈值范围内时闭合夹爪；
- **提起**：抓取成功后回收机械臂。

**关键参数**：
- 环境：` Isaac-Mobile-Grasp-Eggtart-BCPPO-v0`（静止目标，难度最低）
- 环境数量：2048（并行采集）
- 最大步数：2000（约产生 400 万条样本）
- 动作噪声：0.05（增加探索多样性）

**用法**：
```bash
cd /home/pu/isaac-sim
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/collect_demo.py \
    --task  Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 2048 \
    --max_steps 2000 \
    --noise_scale 0.05 \
    --output datasets/eggtart_demo.hdf5 \
    --headless
```

**预期输出**：
- 数据集：`datasets/eggtart_demo.hdf5`（约 5 万–10 万条样本）
- 演示成功率：≥80%
- 若成功率 < 80%：调整老师策略或降低噪声

**验收标准**（依据路线文档 §2.4）：
- [ ] 演示成功率 ≥ 80%
- [ ] 轨迹长度合理（5–10 秒，无明显停滞）
- [ ] 动作方差非零（噪声生效）
- [ ] 数据量 ≥ 5 万条 (s,a) 样本对

---

### 阶段 1：BC 预训练

**脚本**：`scripts/bc_pretrain.py`

**目的**：在演示数据上用监督学习训练 actor 网络。

**网络结构**：
- **关键点**：必须与 rsl_rl 的 `ActorCritic` 结构完全一致
- Actor：`[256, 128, 64]` 隐藏层，ELU 激活
- Critic：结构相同，但保持随机初始化（由 PPO 负责训练）
- BC 阶段仅优化 actor 参数

**训练细节**：
- 损失函数：预测动作与演示动作之间的 MSE
- 优化器：Adam，lr=3e-4
- 轮数：100（含早停）
- 批大小：4096
- 训练/验证集划分：90/10

**观测归一化**：
- 根据训练数据计算均值/标准差
- 随 checkpoint 元数据一并保存，保证与 PPO 一致

**用法**：
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

**预期输出**：
- Checkpoint：`checkpoints/bc_pretrained.pt`（rsl_rl 格式）
- 训练损失：应收敛到较低水平（< 0.1）
- 验证损失：不应发散（过拟合检查）

**测试 BC 策略**：
```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 32 \
    --checkpoint checkpoints/bc_pretrained.pt
```

**验收标准**（依据路线文档 §3.3）：
- [ ] BC 策略成功率 > 0%（理想 ≥50%）
- [ ] checkpoint 能被 rsl_rl 成功加载
- [ ] 用 BC 权重运行 play.py 无报错

---

### 阶段 2：PPO 微调

**配置**：`source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/route_b_rewards.py`

**目的**：用精简奖励对 BC 策略进行 PPO 微调。

**奖励配置 1**（推荐 - `RouteBRewardsCfg`）：
仅保留 5 个核心奖励项并降低权重：

| 奖励项 | 原始权重 | 路线 B 权重 | 作用 |
|--------|----------|-------------|------|
| `base_approach` | 2.5 | 1.0 | 底盘导航 |
| `ee_reach` | 5.0 | 2.0 | 末端到达 |
| `grasp_posture_guide` | 4.0 | 1.5 | 机械臂姿态 |
| `target_lift_progress` | 30.0 | 10.0 | 提起进度 |
| `grasp`（稀疏） | 30.0 | 30.0 | 成功奖励 |

其余稠密奖励项全部禁用（arm_comfort、ee_precision、base_facing 等）。

**奖励配置 2**（激进 - `RouteBRewardsCfg_Aggressive`）：
- 仅保留 `grasp`（稀疏成功奖励）+ 极小权重 `ee_reach`（权重=0.5）
- 待配置 1 验证成功后使用
- 更接近论文中的纯稀疏奖励设定

**课程调整**：
- **取消奖励权重调度**（所有项从第 0 步起生效）
- BC warm-start 意味着策略已具备抓取能力
- 保留目标分布课程（阶段 1→2→3）

**PPO 超参数**：
- **学习率**：**3e-4**（由 1e-3 调低，防止灾难性遗忘）
- 其余参数不变：entropy=0.005、clip=0.2 等

**用法**：
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

**注意**：当前默认使用基础奖励。要使用路线 B 奖励，需创建环境变体：
1. 在 `config/eggtart/grasp_env_cfg.py` 中新增使用 `RouteBRewardsCfg` 的配置
2. 注册为 `Isaac-Mobile-Grasp-Eggtart-Static-RouteB-v0`
3. 训练命令中使用该任务名

**预期表现**：
- grasp 奖励**提前**出现（100–200 迭代，对比从零训练的 1000+ 迭代）
- 成功率从 BC 基线起步（约 50–70%）
- 单调提升（无灾难性遗忘）
- 静止目标最终成功率 ≥80%

**验收标准**（依据路线文档 §4.5）：
- [ ] 成功率 ≥ 80%（静止目标）
- [ ] 优于从零训练的纯 PPO
- [ ] 优于不做微调的 BC
- [ ] 无灾难性遗忘（成功率不回落）

---

## 快速开始

### 完整管线（3 条命令）

```bash
# 终端需位于 /home/pu/isaac-sim
# 使用 conda 环境：my_isaac_env

# 1. 采集演示数据（约 10 分钟）
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/collect_demo.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 2048 --max_steps 2000 --noise_scale 0.05 \
    --output /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/datasets/eggtart_demo.hdf5 \
    --headless

# 2. BC 预训练（约 30 分钟）
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/bc_pretrain.py \
    --data /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/datasets/eggtart_demo.hdf5 \
    --output /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/checkpoints/bc_pretrained.pt \
    --epochs 100

# 3. PPO 微调（4000 迭代约 4–8 小时）
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 2048 --max_iterations 4000 \
    --resume --load_checkpoint /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/checkpoints/bc_pretrained.pt \
    --headless
```

---

## 监控与调试

### 关键 TensorBoard 指标

**训练早期（0–500 迭代）**：
- `Rewards/grasp`：应**早期即非零**（BC 带来的优势）
- `Rewards/ee_reach`：应为正且稳定
- `Rewards/base_approach`：应引导导航
- `Train/learning_rate`：确认是 3e-4（而非 1e-3）

**训练中期（500–2000 迭代）**：
- 成功率：应从 BC 基线持续爬升
- `Rewards/grasp`：应随时间上升
- 无突然下降（灾难性遗忘检查）

**训练后期（2000–4000 迭代）**：
- 成功率：目标 ≥80%
- 所有奖励项：应稳定或持续改善

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 演示成功率 < 80% | 老师策略有缺陷 | 检查目标分布、P 控制器增益 |
| BC 成功率 < 50% | 演示质量差或网络结构不匹配 | 采集更多/更优数据，核对网络结构 |
| 灾难性遗忘 | 学习率过高 | 调低至 1e-4 或 5e-5 |
| 相比 BC 无提升 | 引导信号不足 | 改用配置 1 而非配置 2 |
| grasp 奖励恒为零 | 观测归一化不匹配 | 核对 BC 保存的 obs_mean/std |

---

## 对比：路线 B vs 从零训练 PPO

| 指标 | 从零训练 PPO | 路线 B（BC+PPO） |
|------|--------------|------------------|
| 首次出现 grasp 事件 | 约 1000 迭代 | 约 100 迭代 |
| 达到 50% 成功率 | 约 2000 迭代 | 约 500 迭代（起点即接近） |
| 达到 80% 成功率 | 约 4000 迭代（若能达到） | 约 2000 迭代 |
| 数据效率 | 低（仅 on-policy） | 高（演示数据 + on-policy） |
| 实现改动 | 无 | +2 个脚本，+1 个配置 |

---

## 后续步骤

### 阶段 2 达标后（静止目标 ≥80%）

1. **迁移至移动目标**：
   - 使用 `Isaac-Mobile-Grasp-Eggtart-v0`（移动目标）采集演示
   - 重跑 BC + PPO 管线
   - 目标：≥60% 成功率（任务难度更高）

2. **路线 B 进阶（可选）**：
   - 尝试配置 2（激进稀疏奖励）
   - 实现 DAPG 正则化（见路线文档 §5）
   - 在 PPO 目标中加入 BC 损失项以防遗忘

3. **路线 A（如需进一步提升）**：
   - 切换至 off-policy 算法（通过 skrl 使用 TD3/SAC）
   - 将演示数据用于 replay buffer
   - 纯稀疏奖励
   - 参见 `.doc/技术路线/TECH_ROADMAP_OFFLINE2ONLINE.md`

---

## 参考资料

- 路线文档：`.doc/技术路线/TECH_ROADMAP_BC.md`
- DAPG 论文：https://arxiv.org/abs/1709.10087
- rsl_rl：https://github.com/leggedrobotics/rsl_rl
- Isaac Lab：https://github.com/isaac-sim/IsaacLab

---

## 更新日志

- **2026-09-10**：初始实施完成
  - ✅ 阶段 0：`collect_demo.py`
  - ✅ 阶段 1：`bc_pretrain.py`
  - ✅ 阶段 2：`route_b_rewards.py`
  - ⏳ 阶段 3：DAPG 正则化（可选，尚未实现）
