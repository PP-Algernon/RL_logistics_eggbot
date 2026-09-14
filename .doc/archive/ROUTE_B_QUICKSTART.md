> 已归档（2026-09-14）。原路径：`.doc/ROUTE_B_QUICKSTART.md`。本文保留当时方案、参数与实验记录，当前操作请见[文档索引](../README.md)和[训练指南](../TODO_BC_PPO_DAPG.md)。

# Route B 快速启动指南

## 环境已注册完成 ✅

新环境名称：`Isaac-Mobile-Grasp-Eggtart-BCPPO-v0`

这个环境使用 Route B 的简化奖励配置，专门为 BC 预训练 + PPO 微调设计。

---

## 完整流程（3 步）

### 步骤 0：收集演示数据

```bash
cd /home/pu/isaac-lab

# 使用新注册的 BC+PPO 环境
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/collect_demo.py \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 2048 \
    --max_steps 2000 \
    --noise_scale 0.05 \
    --output /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/datasets/eggtart_demo.hdf5 \
    --headless
```

**预期输出**：
- 文件：`datasets/eggtart_demo.hdf5`
- 样本数：~4M (2000 步 × 2048 环境)
- 成功率：≥ 80%

---

### 步骤 1：BC 预训练

```bash
cd /home/pu/isaac-lab

./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/bc_pretrain.py \
    --data /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/datasets/eggtart_demo.hdf5 \
    --output /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/checkpoints/bc_pretrained.pt \
    --actor_hidden_dims 256 128 64 \
    --critic_hidden_dims 256 128 64 \
    --activation elu \
    --lr 3e-4 \
    --epochs 100 \
    --batch_size 4096
```

**预期输出**：
- 文件：`checkpoints/bc_pretrained.pt`
- 训练损失：收敛到 < 0.1
- 时间：~30 分钟

**测试 BC 策略**：
```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 32 \
    --checkpoint /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/checkpoints/bc_pretrained.pt
```

**预期**：成功率 50-70%

---

### 步骤 2：PPO 微调

```bash
cd /home/pu/isaac-lab

./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 2048 \
    --max_iterations 4000 \
    --resume \
    --load_checkpoint /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/checkpoints/bc_pretrained.pt \
    --headless
```

**预期结果**：
- 抓取奖励在 100-200 迭代内出现（对比纯 PPO 的 1000+）
- 成功率从 BC 基线（~50-70%）开始攀升
- 最终成功率 ≥ 80%（约 2000 迭代）

---

## 监控指标（TensorBoard）

```bash
tensorboard --logdir logs/rsl_rl
```

**关键指标**：
- `Rewards/grasp`：应该早期就非零（BC 优势）
- `Rewards/base_approach`：底盘导航引导
- `Rewards/ee_reach`：末端到达引导
- `Rewards/target_lift_progress`：提起进度
- `Train/learning_rate`：确认是 3e-4（不是 1e-3）

---

## Route B vs 标准配置对比

| 方面 | 标准环境 (Static-v0) | Route B (BCPPO-v0) |
|------|---------------------|-------------------|
| **奖励项数量** | ~15 项稠密奖励 | 5 项核心奖励 |
| **奖励课程** | 3 阶段权重调度 | 无（全程激活） |
| **学习率** | 1e-3 | 3e-4（防遗忘） |
| **初始化** | 随机 | BC 预训练 |
| **首次抓取** | ~1000 迭代 | ~100 迭代 |
| **80% 成功率** | ~4000 迭代 | ~2000 迭代 |

---

## Route B 使用的简化奖励

只保留 5 个核心项（权重已减半）：

1. **`base_approach`** (1.0)：底盘接近目标
2. **`ee_reach`** (2.0)：末端到达目标
3. **`grasp_posture_guide`** (1.5)：抓取姿态引导
4. **`target_lift_progress`** (10.0)：提起进度
5. **`grasp`** (30.0)：稀疏成功奖励

禁用的项：`base_facing`, `arm_comfort`, `ee_precision`, `ee_distance`, `ee_orientation`, `gripper_closure_reward`, `joint_vel`, `base_vel` 等

---

## 故障排查

### 问题：演示成功率 < 80%
**原因**：脚本化老师策略参数需要调整
**解决**：
- 检查 `approach_threshold` (默认 0.15m)
- 检查 `reach_threshold` (默认 0.05m)
- 降低 `noise_scale` (默认 0.05)

### 问题：BC 成功率 < 50%
**原因**：演示数据质量不够或网络结构不匹配
**解决**：
- 收集更多数据（增加 `max_steps`）
- 确认网络结构与 rsl_rl 配置一致
- 检查观测归一化

### 问题：PPO 微调后成功率下降（灾难性遗忘）
**原因**：学习率太高
**解决**：
- 降低学习率到 1e-4 或 5e-5
- 检查 TensorBoard 确认 lr 是 3e-4
- 考虑实现 DAPG 正则化（阶段 3）

### 问题：PPO 没有提升
**原因**：奖励信号不足
**解决**：
- 检查 `RouteBRewardsCfg` 是否正确加载
- 可以暂时增加奖励权重（如 `ee_reach` 从 2.0 → 3.0）
- 或使用标准环境 `Isaac-Mobile-Grasp-Eggtart-Static-v0` 的奖励

---

## 文件位置

```
Eggtart-logistics-robot/
├── scripts/
│   ├── collect_demo.py                          # ✅ 演示收集
│   ├── bc_pretrain.py                            # ✅ BC 预训练
│   └── rsl_rl/train.py                          # ✅ PPO 训练
├── source/eggtart_grasp/eggtart_grasp/
│   └── tasks/mobile_grasp/
│       ├── route_b_rewards.py                   # ✅ Route B 奖励
│       └── config/eggtart/
│           ├── __init__.py                      # ✅ 环境注册
│           └── grasp_env_cfg.py                 # ✅ BCPPO 配置
├── datasets/
│   └── eggtart_demo.hdf5                        # 生成：演示数据
├── checkpoints/
│   └── bc_pretrained.pt                         # 生成：BC 权重
└── .doc/
    ├── ROUTE_B_IMPLEMENTATION.md                # 📖 完整指南
    ├── ROUTE_B_STATUS.md                        # 📖 状态与故障排查
    └── ROUTE_B_QUICKSTART.md                    # 📖 本文件
```

---

## 下一步（可选进阶）

### 1. 移动目标迁移
收集移动目标的演示数据并重新训练：
```bash
# 使用标准移动目标环境（需要先创建对应的 BCPPO 移动版）
# 或直接在静态训练后迁移到移动环境
```

### 2. 激进稀疏奖励（Configuration 2）
编辑 `grasp_env_cfg.py`，将 `RouteBRewardsCfg` 替换为 `RouteBRewardsCfg_Aggressive`

### 3. DAPG 正则化
实现 BC 正则项防止遗忘（需要 fork rsl_rl）

### 4. 切换到 Route A
如果需要更高的成功率上限，考虑 off-policy 算法（TD3/SAC）

---

## 支持

参考文档：
- `.doc/ROUTE_B_IMPLEMENTATION.md` - 详细实现指南
- `.doc/ROUTE_B_STATUS.md` - 当前状态与故障排查
- `.doc/技术路线/TECH_ROADMAP_BC.md` - 原始技术路线

所有实现已完成，现在可以开始运行完整流程！
