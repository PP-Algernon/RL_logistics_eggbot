> 历史记录：2026-09-12 配置合并后，本文的旧环境名、配置类和奖励调度不再适用。当前唯一环境为 `Isaac-Mobile-Grasp-Eggtart-v0`，请以[训练清单](../TODO_BC_PPO_DAPG.md)为准。

# 路线 B 实施手册：BC 预训练 + PPO 微调（Eggtart 移动抓取）

> 定位：在**不动算法栈**（RSL-RL / PPO）的前提下，用演示数据做行为克隆（Behavior Cloning）预训练 warm-start，再用 PPO 微调，把成功率拉起来。
> 配套文档：《TECH_ROADMAP_OFFLINE2ONLINE.md》总路线（路线 B 章节）＋本手册（实施细节）
> 生成日期：2026-09-09

---

## 1. 路线 B 定位与总体流程

**适用前提**：暂时不想换 off-policy 算法（skrl/TD3 等），希望改动最小、先在现有 PPO 管线上看到成功率提升。

**核心机制**：
1. **BC 解决探索**：脚本老师先"把任务做成功"，用 (obs, action) 对监督训练一个策略——策略从一开始就会"接近→对准→闭爪→提起"，不再从零随机摸；
2. **PPO 微调提升鲁棒性**：在 BC 权重基础上继续 PPO 训练，让策略适应随机扰动、目标分布变化，并优化时序/成功率。

**与路线 A 的差异（诚实对比）**：

| 维度 | 路线 A（离线+稀疏+off-policy） | 路线 B（BC+PPO） |
|---|---|---|
| 算法栈 | 换 skrl/rl_games（TD3/SAC） | 保持 rsl_rl（PPO） |
| 奖励设计 | 纯稀疏（成功/每步/翻车） | 稀疏成功 + 精简稠密引导（on-policy 需要） |
| 数据利用 | 专家数据进 replay buffer，可反复利用 | BC 一次性使用，之后靠 PPO 在线 |
| 成功率上限 | 高（论文 sim 近 100%） | 中（依赖 BC 质量与 PPO 防遗忘） |
| 改动量 | 大（新训练入口 + 数据管线） | 小（2 个新脚本 + 配置改动） |
| 主要风险 | 超参敏感、数据量要求 | 灾难性遗忘、数据覆盖窄 |

**总体流程**：

```
阶段0 数据采集          阶段1 BC 预训练         阶段2 PPO 微调          阶段3 进阶（可选）
┌────────────────┐  (s,a)  ┌─────────────────┐ 权重  ┌────────────────┐    ┌─────────────────┐
│ scripted 老师  │ ──────▶ │ bc_pretrain.py  │ ───▶ │ rsl_rl train.py│──▶│ DAPG BC 正则     │
│ + 噪声/随机化   │  HDF5   │ 对齐 ActorCritic│  .pt  │ --resume 加载  │    │ （fork rsl_rl）  │
└────────────────┘         └─────────────────┘       └────────────────┘    └─────────────────┘
```

---

## 2. 阶段 0：演示数据采集

### 2.1 老师策略

复用现有资产，不需要新写控制器：
- **导航**：规则脚本——用目标在底盘系下的位置（观测里已有）算体速度指令 `(vx, vy, ωz)`，全向底盘是理想速度控制，P 控制即可；
- **接近与对准**：当抓取点与目标距离 < `GRASP_REACH_THRESHOLD`（0.05 m）且朝向对准后，进入状态机（复用 `utils/scripted_grasp.py` 的 闭爪→保持→收臂 逻辑）；
- **夹爪**：`max_effort≈1.0`（沿用现值）。

### 2.2 关键配置

| 项 | 建议值 | 说明 |
|---|---|---|
| 环境 | `Isaac-Mobile-Grasp-Eggtart-Static-v0` | 静止目标，成功率最高，先跑通 |
| 目标分布 | 固定正前方 x∈[0.6,0.7]（课程阶段 1） | 数据集中，BC 好学好收 |
| 动作噪声 | 高斯 σ≈0.05–0.1（叠加在指令上） | 增加数据多样性，防 BC 过拟合到单一轨迹 |
| 初始随机化 | 目标初速 0；机器人初始位姿小扰动 | 覆盖面够用即可，别一上来太散 |
| 数据量 | 先 5–10 万条 (s,a) | BC 是监督学习，量不需要像路线 A 那么多 |
| 输出格式 | HDF5（`obs` / `action` 两个数组 + 元信息） | 对齐 Isaac Lab 生态，也方便后续换路线 A |

### 2.3 脚本骨架 `scripts/collect_demo.py`

```python
"""采集 BC 训练数据：scripted 老师 + 动作噪声，并行 rollout。"""
import gymnasium as gym
import h5py, torch, numpy as np

# 1. 创建环境（同 train.py 的构建方式）
env = gym.make("Isaac-Mobile-Grasp-Eggtart-Static-v0", num_envs=2048)

# 2. 实现 scripted 老师：输入 obs，输出 9 维 action 张量
def scripted_teacher(obs: torch.Tensor) -> torch.Tensor:
    # 底盘体速度 (vx, vy, wz)：P 控制朝目标，到位后归零
    # 机械臂关节 5 维：按脚本姿态插值
    # 夹爪力 1 维：目标进入可夹范围后输出夹持力
    ...  # 复用 scripted_grasp.py 的状态机逻辑
    return actions

obs, _ = env.reset()
obs_buf, act_buf = [], []
noise_scale = 0.05
for step in range(MAX_STEPS):
    actions = scripted_teacher(obs)
    noisy = actions + torch.randn_like(actions) * noise_scale
    obs_buf.append(obs.cpu().numpy()); act_buf.append(noisy.cpu().numpy())
    obs, rew, terminated, truncated, infos = env.step(noisy)
    # 只保留成功 episode 的轨迹（抓取并提起成功的 env 索引）

# 3. 写 HDF5：datasets/eggtart_demo.hdf5
with h5py.File("datasets/eggtart_demo.hdf5", "w") as f:
    f.create_dataset("obs", data=np.concatenate(obs_buf))
    f.create_dataset("action", data=np.concatenate(act_buf))
```

### 2.4 验收（阶段 0 完成标准）

- [ ] 演示成功率 ≥ 80%（成功 = 触发 grasp 事件：提到 12 cm 保持 0.1 s）
- [ ] 轨迹长度分布合理（5–10 s，无明显卡死）
- [ ] 动作多样性：加入噪声后 action 方差非零、且不导致成功率骤降
- [ ] 数据量 ≥ 5 万条 (s,a)

---

## 3. 阶段 1：BC 预训练

### 3.1 网络结构对齐 rsl_rl（关键！）

PPO 微调要能加载 BC 权重，BC 网络必须与 rsl_rl 的 `ActorCritic` **结构完全一致**：

- 从 `rsl_rl_ppo_cfg.py` 读取网络配置（当前：actor/critic 均为 `[256,128,64]` ELU）；
- rsl_rl 的 ActorCritic 由三部分组成：`actor`（MLP，输出 action mean）、`log_std`（可训练参数）、`critic`（MLP，输出 value）；
- **BC 只训练 actor**：`log_std` 保持默认初始化，`critic` 保持随机初始化（PPO 会从头学 value）。

> ⚠️ 以你安装的 rsl_rl 版本源码为准对齐接口（`rsl_rl.networks.ActorCritic`、`rsl_rl.algorithms.ppo`）；Isaac Lab 的 rsl_rl 版本 ≥5.0 时 checkpoint 布局有变化，见阶段 1.4。

### 3.2 Loss 与训练细节

```python
"""bc_pretrain.py：监督训练 actor，输出 rsl_rl 兼容 checkpoint。"""
import torch, torch.nn as nn, h5py
from rsl_rl.networks import ActorCritic   # 对齐你的版本

# 1. 构建与 PPO 相同的网络（从 rsl_rl_ppo_cfg.py 取结构）
ac = ActorCritic(num_obs=OBS_DIM, num_actions=ACT_DIM,
                 actor_hidden_dims=[256,128,64], critic_hidden_dims=[256,128,64],
                 activation="elu")

# 2. 载入数据（观测建议做 running normalization，与 PPO 一致）
data = h5py.File("datasets/eggtart_demo.hdf5", "r")
obs, act = data["obs"][:], data["action"][:]

# 3. 只优化 actor 参数（含 log_std 之外的 actor 权重）
opt = torch.optim.Adam(ac.actor.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()   # 或负对数似然 NLL(高斯)
for epoch in range(100):
    for batch in batches(obs, act, bs=4096):
        mean = ac.actor(batch_obs)          # 归一化后输入
        loss = loss_fn(mean, batch_act)
        opt.zero_grad(); loss.backward(); opt.step()
    print(epoch, loss.item())

# 4. 保存为 rsl_rl 可加载的 checkpoint
torch.save({"model_state_dict": ac.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "iter": 0, "infos": {}}, "checkpoints/bc_pretrained.pt")
```

要点：
- **输入归一化**：rsl_rl 训练时观测做了 running normalization（`obs_normalizer`），BC 阶段要么也用相同归一化，要么微调时从 `--load_checkpoint` 加载后前几步让 PPO 重新适应；建议在 BC 脚本里先统计数据集 mean/std 并保存，微调时对齐。
- **Loss 选择**：MSE 简单稳定；NLL（高斯）与 rsl_rl 的策略输出分布更一致，效果通常略好，先 MSE 起步。
- **epochs**：100 左右（数据 5–10 万条足够收敛），监控验证集 loss 防过拟合。

### 3.3 验收（阶段 1 完成标准）

- [ ] BC 策略在 `play.py`（或评估脚本）中直接部署，**成功率 > 0，目标 ≥ 50%**（BC 都没学会 → 数据或老师有问题，先修阶段 0）
- [ ] checkpoint 能被 rsl_rl `--resume` 成功加载（先跑 1 次 10 迭代的冒烟测试）

---

## 4. 阶段 2：PPO 微调

### 4.1 奖励设计（二选一，建议先配置 1）

**配置 1（推荐起步）：稀疏成功 + 精简稠密引导**
从现有奖励里**只保留 4–5 个核心项，权重减半**，其余全部关掉：

| 保留项 | 原权重 | 建议权重 | 作用 |
|---|---|---|---|
| `base_approach` | 2.5 | 1.0 | 底盘接近 |
| `ee_reach` | 5.0 | 2.0 | 末端到达 |
| `grasp_posture_guide` | 4.0 | 1.5 | 抓取姿态引导 |
| `target_lift_progress` | 30.0 | 10.0 | 提起进度 |
| `grasp`（稀疏成功） | 30.0 | 30.0 | 任务成功 |

**配置 2（激进，逼近论文稀疏）**：只留 `grasp` 成功奖励 + 一个权重 ≤0.01 的"到目标距离"温度计项。

> 为什么路线 B 不能完全纯稀疏：PPO 是 on-policy，每轮采样的成功样本占比低，纯稀疏下 advantage 信号太稀；BC warm-start 让策略"会做"之后，配置 2 可行但更敏感，建议配置 1 先出结果、再尝试配置 2。

### 4.2 课程与训练配置

- **目标分布课程保留**：阶段 1（固定正前方）→ 阶段 2 → 阶段 3（随机），与数据采集分布衔接；
- **奖励权重课程关闭**：配置 1 的项从 step 0 就全开（BC 已解决探索，不再需要"逐步放开"）；
- **防遗忘**：`learning_rate` 从 1e-3 调到 **3e-4～1e-4**；`entropy_coef` 保持 0.005；可加 `max_grad_norm` 保持默认；
- **训练量**：4000 迭代上限不变，观察成功率曲线；通常 BC 后几百迭代内就能看到 grasp 事件。

### 4.3 启动命令

```bash
cd /home/pu/isaac-lab
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 2048 \
    --max_iterations 4000 \
    --resume \
    --load_checkpoint /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/checkpoints/bc_pretrained.pt \
    --headless
```

> `--load_checkpoint` 直接指向 BC 权重文件（rsl_rl 的 resume 机制）；`--load_run` 是按 run 名称查找。若你的 Isaac Lab 版本参数名不同，以 `python scripts/rsl_rl/train.py --help` 为准。

### 4.4 监控指标

| TensorBoard 指标 | 预期 |
|---|---|
| `Rewards/grasp` | BC 后早期就应非 0（对比纯 PPO 从零的长期为 0） |
| `Rewards/` 各保留项 | 平稳上升，无突然崩塌（崩塌 = 遗忘信号） |
| `Train/learning_rate` | 确认已调低 |
| 成功率（自定义 log） | 从 BC 基线往上走，最终 ≥ 80% |

### 4.5 验收（阶段 2 完成标准）

- [ ] 成功率 ≥ 80%（静态目标）
- [ ] 明显优于两个对照：纯 PPO 从零（现状）、BC 不微调
- [ ] 无灾难性遗忘（成功率单调上升或平台，不回落）

---

## 5. 阶段 3（可选进阶）：DAPG 式 BC 正则

BC warm-start 后 PPO 微调的最大风险是**遗忘**。DAPG 的做法：PPO 的 policy gradient loss 上加一项 BC 正则：

```
L_total = L_PPO − λ(t) · E_(s,a)~D_demo [ log π(a|s) ]
λ(t) = λ_0 · (1 − t/T_max)      # 线性退火
```

实现路径（按工作量排序）：
1. **fork rsl_rl**：在 `rsl_rl/algorithms/ppo.py` 的 loss 计算处加入 BC 项（每个 epoch 额外采样一批演示数据计算 `-log π(a_demo|s_demo)`），λ 按迭代退火；改动集中在 20–30 行；
2. 或干脆切换到 **skrl 的 PPO**（Isaac Lab 支持），在其自定义 `loss` 里加 BC 正则，顺带为将来路线 A 铺路。

> 先跑通阶段 2 再看是否需要阶段 3；如果阶段 2 成功率已经达标，跳过。

---

## 6. 文件改动清单

| 文件 | 动作 | 说明 |
|---|---|---|
| `scripts/collect_demo.py` | 新增 | 数据采集（§2.3 骨架） |
| `scripts/bc_pretrain.py` | 新增 | BC 训练 + checkpoint 输出（§3.2 骨架） |
| `source/.../config/eggtart/agents/rsl_rl_ppo_cfg.py` | 修改 | 微调阶段 lr 调低；网络结构不动 |
| `source/.../tasks/mobile_grasp/mobile_grasp_env_cfg.py` | 修改 | 新增"路线 B 奖励"配置（配置 1 精简项）；关闭奖励权重课程或替换为精简版 |
| `source/.../tasks/mobile_grasp/mdp/curriculums.py` | 修改 | 奖励课程改为精简版（或整体关闭，仅留目标分布课程） |
| `scripts/rsl_rl/train.py` | 不动 | 直接用 `--resume --load_checkpoint` |
| `checkpoints/bc_pretrained.pt` | 生成 | BC 权重产物 |
| `datasets/eggtart_demo.hdf5` | 生成 | 演示数据集 |

> 建议：新增奖励配置做成开关（如 `env.cfg.use_route_b_rewards = True`），不动原有稠密奖励代码，方便回退与对照实验。

---

## 7. 实验矩阵与验收标准

| 编号 | 实验 | 配置 | 目的 |
|---|---|---|---|
| 基线1 | 纯 PPO 从零（现状） | 现有稠密+课程 | 对照：当前成功率 |
| 基线2 | BC 预训练后直接部署 | 不微调 | 看 BC 本身有多强 |
| 实验1 | **BC + PPO 微调（配置 1）** | 精简稠密 + 低 lr | **主实验** |
| 实验2 | BC + PPO 微调（配置 2） | 稀疏 + 温度计 | 逼近论文，检验数据质量 |
| 实验3 | BC + PPO + DAPG 正则 | 阶段 3 | 防遗忘，冲上限 |

**验收总目标**：实验 1 成功率 ≥ 80%（静态）；并显著高于基线 1 与基线 2。
**迁移**：实验 1 稳定后，把 BC 数据换到 `Isaac-Mobile-Grasp-Eggtart-v0`（移动目标）重跑 0→2 阶段，目标成功率 ≥ 60%（移动目标难度更高，先不追 80%）。

---

## 8. 学习清单（路线 B 专项）

### 8.1 概念
- [ ] 行为克隆（BC）：监督学习策略、MSE vs 负对数似然、输入归一化的重要性
- [ ] 灾难性遗忘：为什么 BC→RL 微调会忘、低 lr / KL / BC 正则三种防护
- [ ] DAPG：BC 预训练 + λ 退火的 BC 正则项（公式与直觉）
- [ ] rsl_rl 内部：`ActorCritic` 三件套（actor / log_std / critic）、PPO loss 结构

### 8.2 工程
- [ ] 读 rsl_rl 源码 `networks.py` 与 `algorithms/ppo.py`（确认网络结构与 checkpoint 格式）
- [ ] 跑通一次 `--resume --load_checkpoint` 冒烟测试（10 迭代）
- [ ] 学会看 TensorBoard 的 `Rewards/grasp` 与成功率曲线判断"是否学到"
- [ ] 用 `play.py` 直接部署 BC 权重评估成功率（不经过训练）

---

## 9. 风险与常见坑

| 风险 | 现象 | 对策 |
|---|---|---|
| 老师数据覆盖窄 | BC 成功率低、只会单一姿势 | 加大动作噪声、扩展目标位置分布；检查老师本身成功率 |
| 灾难性遗忘 | 微调后成功率先升后崩 | lr 降到 1e-4；或启用阶段 3 DAPG 正则；缩短微调长度 |
| 观测归一化不匹配 | BC 好、加载后 PPO 表现突变 | 统一 running normalization；加载后前几百步用小 lr 适应 |
| rsl_rl 版本差异 | checkpoint 加载报 key 不匹配 | 对照你安装版本的 `ActorCritic.state_dict()` 的 key 命名；Isaac Lab 有 `handle_deprecated_rsl_rl_checkpoint` 转换 |
| 奖励项残留干扰 | 微调学歪（如绕圈） | 严格按配置 1 只保留 5 项；对照原稠密配置逐项排查 |
| 移动目标追不上 | Static 达标、Mobile 崩 | 先降低目标速度随机化；数据里加入带初速的轨迹再 BC |

---

## 10. 参考资料（路线 B 相关）

| 资料 | 链接 | 用途 |
|---|---|---|
| DAPG（原论文） | https://arxiv.org/abs/1709.10087 | BC 预训练 + BC 正则的理论与公式 |
| 演示引导 RL 综述 | https://arxiv.org/abs/2303.13489 | 路线 B 在整个方法谱系中的位置 |
| rsl_rl 官方仓库 | https://github.com/leggedrobotics/rsl_rl | 读 `networks.py` / `algorithms/ppo.py` / `runners/on_policy_runner.py` |
| Isaac Lab rsl_rl 训练入口 | https://github.com/isaac-sim/IsaacLab/blob/main/scripts/reinforcement_learning/rsl_rl/train.py | `--resume / --load_run / --load_checkpoint` 参数 |
| Isaac Lab 演示数据录制（Mimic/robomimic） | https://github.com/isaac-sim/IsaacLab/tree/main/scripts/tools | HDF5 演示数据格式参考（即使不用遥操作） |
| 总路线文档 | `E:\Learning\RLworkspace\RL_Logistics_Eggbot\.doc\TECH_ROADMAP_OFFLINE2ONLINE.md` | 路线 A/B 对比、三条红线、背景 |

---

## 11. 一句话总结

**路线 B = 用脚本老师产 (s,a) → BC 预训练出一个"会抓"的策略 → 用低 lr 的 PPO + 精简奖励微调提升鲁棒性**；先跑通 `collect_demo.py → bc_pretrain.py → train.py --resume` 三连，再决定要不要加 DAPG 正则或升级路线 A。
