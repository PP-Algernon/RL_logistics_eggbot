> 历史记录：当前保留 `Isaac-Mobile-Grasp-Eggtart-v0` 和 `Isaac-Mobile-Grasp-Eggtart-BCPPO-v0`，分别使用普通奖励课程和 `BCCurriculumCfg`。本文的旧配置与命令请以[训练清单](../TODO_BC_PPO_DAPG.md)为准。

# Eggtart 移动抓取：从"纯 RL + 稠密奖励"转向"离线数据 + 稀疏奖励 RL"技术路线

> 依据 arXiv:2608.12063（SMPC 演示 + 稀疏 offline-to-online RL，Spot/G1 实物验证）
> 生成日期：2026-09-09 ｜ 配套项目：RL_Logistics_Eggbot（Isaac Lab + RSL-RL/PPO）

---

## 0. 问题定义

**现状**：全向底盘 + 5 轴臂 + 力控夹爪的移动抓取任务，采用 PPO（on-policy）+ 稠密手工奖励塑形 + 课程延迟稀疏奖励（grasp 奖励到 step 48000 才开启）。

**瓶颈**：
1. 手工稠密奖励每次微调都要重跑数小时训练才能验证，迭代极慢；
2. 稀疏成功奖励在 on-policy 下从零探索几乎摸不到成功状态（抓取→提起是长时序接触任务，随机探索命中率≈0）；
3. 课程学习（把目标喂到夹爪前）是"informed reset"思路，缓解了前半程，但探索问题仍压在 RL 自己身上，成功率上不去。

**核心判断（论文结论）**：纯稀疏奖励从零训练复杂操作任务，学不出来；"靠稠密 shaping 硬扛"正是要绕开的瓶颈。解法是**先用一个容易调的"老师"（控制器/脚本）在仿真里产大量成功演示数据，用数据解决探索，RL 只负责用稀疏任务奖励学出超越老师的行为**。

---

## 1. 技术路线总览

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐    ┌─────────────┐
│  老师(Teacher) │ → │ 离线数据集     │ → │ 稀疏奖励 off-policy RL │ → │ 在线微调/部署 │
│ SMPC/脚本状态机 │    │ (obs,act,rew, │    │ TD3/SAC + buffer 预填 │    │ 成功率~100%  │
│ 分钟级可调     │    │  next,done)   │    │ 50%专家数据 + phase-out│    │ 超越老师      │
└──────────────┘    └──────────────┘    └──────────────────────┘    └─────────────┘
```

**核心思路三句话**：
1. 老师负责"把任务做成功"，在仿真里批量、并行地产数据（论文：单 GPU ≈0.5× 实时，约 100 万样本/小时）；
2. 离线数据填进 replay buffer，让 critic 先对"成功轨迹"有正确的价值估计，解决从零探索问题；
3. RL 只用**稀疏奖励**（成功/每步/翻车三项）训练，不依赖手工 shaping；经验成功率到 ~10% 后逐步把专家数据踢出 buffer（phase-out），转为纯在线学习，最终策略可超越老师。

---

## 2. 路线 A（主推）：离线数据 + 稀疏奖励 + off-policy 算法

### 2.1 数据采集层（老师）

**老师选型（按成本从低到高）**：
| 方案 | 说明 | 适配度 |
|---|---|---|
| **现有 scripted_grasp 状态机**（首选） | `utils/scripted_grasp.py` 已有"闭爪→保持→收臂"序列；全向底盘是理想速度控制，导航部分脚本化最好写 | ★★★ 零新增依赖，先跑通管线 |
| 手写规则策略 + 动作噪声 | 在状态机基础上加高斯噪声/扰动，扩大数据覆盖 | ★★★ 数据多样性提升 |
| SMPC / Predictive Sampling | 采样型 MPC（见参考资料 §6.3），分钟级调稠密 cost，自动产出平滑轨迹 | ★★ 工作量最大，但最接近论文 |

**数据采集要点**：
- 并行 rollout：复用现有环境，2048 envs 并行，脚本策略 + 扰动，存 `(obs, action, reward, next_obs, done)`；
- **单模态约束（论文红线，必须遵守）**：老师固定一种行为模式——"正面接近 → 对准 → 闭爪 → 垂直提起"，禁止混入推、蹭、绕等多种都能成功的抓法；
- Domain randomization：目标初始位置（利用现有 `reset_target_curriculum` 阶段 3 的随机范围）、物体质量/摩擦/尺寸、目标初速（Static 版为 0）；
- 数据量预期：论文最难任务需约 400 万样本（4 GPU 小时）；你的任务介于"简单导航"与"滚轮胎"之间，**建议先攒 20–50 万条验证管线**，再按需扩量；
- 存储格式：HDF5（对齐 Isaac Lab `record_demos.py` / robomimic 生态）或 pickle/`.npz`（自用最快）。

### 2.2 算法层（off-policy）

**算法选择**：TD3（对齐论文 FastTD3）或 SAC；**必须离开 rsl_rl（只支持 on-policy PPO）**，改用 Isaac Lab 预集成的：
- **skrl**（推荐，官方支持 TD3/SAC，配置为 YAML）；
- 或 rl_games（SAC）、torchrl、FlashSAC（GitHub 开源 off-policy 框架，支持 IsaacLab）。

**稀疏奖励设计**（论文式，只有三项）：
```
r = 0                      到达目标（抓到并提到 LIFT_HEIGHT_THRESHOLD 以上保持 LIFT_DWELL_TIME）
  = -2/(1-γ) ≈ -200       翻车/终止（底盘倾覆）
  = -1                     其它
```
可选：保留一个极小（权重 ≤0.01）的"到目标距离"项当温度计，帮助收敛但不得主导。

**论文关键超参（可直接作为起点，Table 1）**：
| 参数 | 论文值 | 说明 |
|---|---|---|
| Buffer size | 2^24 ≈ 16.7M | 大 buffer |
| 专家数据初始占比 | 50% | 预填 replay buffer |
| Phase-out 阈值 η | 0.1 | 经验成功率 ≥10% 后逐步剔除专家数据 |
| Batch size | 8192 | 大批量 |
| 折扣 γ | 0.99 | |
| Actor/Critic lr | 3e-4 / 1e-4 | |
| Policy delay | 2 | TD3 标准 |
| Smoothing noise | 0.05 | TD3 目标平滑 |
| Action regularization | 5e-4 | |
| Reward scale | 0.01 | 让 Q 值贴近 0 |

**稳定性工程细节（照抄）**：
- **有界 critic**：Q 值理论上下界为 `[-2/(1-γ), 0]`，critic 输出层接 tanh 再缩放，初始化偏置向 0；显著稳定训练、放宽超参敏感度；
- **Asymmetric actor-critic**：actor 吃带噪声观测，critic 吃仿真真值；
- **动作 delta 化**：策略输出"相对当前目标的增量"（Δ速度/Δ关节位置），显式限制最大加速度/速度，行为更安全、可部署。

### 2.3 训练与验证

1. 先用 `Static-v0`（静止目标）验证整条管线：数据 → 注入 → 稀疏训练 → 成功率曲线；
2. 成功率稳定后再切移动目标（`v0`），或加载 checkpoint 迁移；
3. 评价指标：成功率（grasp 事件计数）、平均任务完成时间（对比脚本老师基线）、策略完成时间方差。

---

## 3. 路线 B（备选，改动最小）：BC 预训练 + PPO 微调

保持现有 PPO 管线，用演示数据做行为克隆（BC）预训练 warm-start，再稀疏奖励微调；或 DAPG 式在 PPO loss 里加 BC 正则项（前期 λ 大，之后退火）。

- 优点：不动算法栈（rsl_rl 支持加载预训练权重），实现快；
- 缺点：on-policy 对离线数据利用效率远低于 off-policy（论文选择 TD3 的原因），成功率上限低于路线 A；
- 用途：先花 1–2 天验证"数据质量是否够好"，再决定是否投入路线 A。

---

## 4. 三条红线（论文消融结论，直接对应本项目）

| # | 红线 | 论文证据 | 对本项目的含义 |
|---|---|---|---|
| 1 | **演示数据必须单模态** | 多模态数据（踢/推/踩都算成功）训练直接失败，即使成功率更高 | 脚本老师固定一种抓法，不混入多解行为 |
| 2 | **专家数据要按时退出** | 一直留在 buffer 后期拖慢收敛（分布偏移）；η=10% 即开始 phase-out；阈值 ≥25% 明显变慢 | 监控经验成功率，达到 ~10% 后逐步降低专家采样比例 |
| 3 | **数据量随任务复杂度上升** | 简单导航少量即可；高协调任务需数百万样本覆盖状态分布 | 先 20–50 万验证，抓不住再扩量 + 加随机化 |

---

## 5. 学习清单（按学习顺序）

### 5.1 概念与方法（先学，理解为什么这么设计）

- [ ] **Offline-to-online RL 范式**：Ball et al. ICML 2023《Efficient Online Reinforcement Learning with Offline Data》——专家数据混合、phase-out 的奠基工作（参考资料 §6.2）
- [ ] **TD3 三处改进**：twin critic、delayed policy update、target policy smoothing；与 DDPG 的差异
- [ ] **SAC 最大熵框架**：可选，作为 TD3 的对照选择
- [ ] **FastTD3 规模化配方**：并行仿真 + 大批量 + 多梯度步数，为什么 off-policy 在 GPU 并行下反超 PPO
- [ ] **行为克隆与 DAPG**：BC 预训练 + BC 正则项如何引导 on-policy 探索（路线 B 的理论基础）
- [ ] **SMPC / Predictive Sampling 原理**：采样型 MPC 如何"不训练、分钟级调参"地产轨迹（只需理解，不必实现）
- [ ] **有界 critic / asymmetric actor-critic**：Q 值爆炸与 deadly triad，tanh 输出层的设计动机

### 5.2 工具与工程

- [ ] **skrl 框架**：TD3/SAC agent 的 YAML 配置、replay buffer 预填、checkpoint 保存/恢复
- [ ] **Isaac Lab skrl workflow**：`scripts/reinforcement_learning/skrl/train.py` 的使用、`--algorithm` 参数、agent cfg 目录结构
- [ ] **Isaac Lab 数据录制**：`scripts/tools/record_demos.py`、HDF5 数据集格式、robomimic 生态（即使不用遥操作，格式可复用）
- [ ] **replay buffer 预填实现**：如何把离线数据集加载进 skrl/torchrl 的 buffer（`init_replay_buffer` / `load_replay_buffer` 类接口）
- [ ] **本项目现有资产**：`scripted_grasp.py` 状态机逻辑、`CurrriculumManager`、`mobile_grasp_env_cfg.py` 奖励项，评估哪些可复用为数据采集器

### 5.3 论文精读清单

- [ ] 精读 arXiv:2608.12063（本篇）：§3.2 稀疏奖励公式与 phase-out 设计、§3.3 SMPC 数据采集、§4 四条消融（Q1–Q4）
- [ ] FastTD3（arXiv:2505.22642）：算法配方与超参
- [ ] Ball et al. 2023（arXiv:2302.02948）：离线数据在线微调的坑与解法
- [ ] DAPG（arXiv:1709.10087）：演示引导策略梯度的正则形式
- [ ] Predictive Sampling（arXiv:2212.00541）：SMPC 老师的原理（可选）

---

## 6. 落地实施清单（按阶段，含检查项）

### Phase 0：数据采集（预估 1–3 天）
- [ ] 写 `scripts/collect_demo.py`：加载现有环境（Static-v0），跑脚本老师 + 动作噪声 + 目标随机化，2048 envs 并行 rollout
- [ ] 固定单模态：正面接近 → 对准 → 闭爪 → 垂直提起；夹爪力沿用 `max_effort≈1.0`
- [ ] 输出 `datasets/eggtart_demo.hdf5`（obs/act/rew/next_obs/done + 元信息）
- [ ] **验证**：统计演示成功率（应 ≥80%）、轨迹长度分布、动作多样性；用 `diagnose_grasp.py` 抽查轨迹质量
- [ ] 目标：20–50 万条样本

### Phase 1：off-policy 训练入口（预估 1–2 天）
- [ ] 为环境配置 skrl（TD3）agent cfg（参考 Isaac Lab 官方 `skrl/train.py` 与 YAML 示例）
- [ ] 保留现有 obs/action 定义，action 改为 delta 化（Δ体速度、Δ关节位置增量、夹爪力）
- [ ] **验证**：能在 `Static-v0` 上跑通 100 迭代、TensorBoard 有曲线

### Phase 2：稀疏奖励 + 数据注入（预估 2–3 天）
- [ ] 新增稀疏奖励项（成功/每步/翻车三项），权重课程全部移除或退化为极小温度计项
- [ ] 实现 buffer 预填：训练前把 HDF5 数据加载进 replay buffer，专家:随机 = 50%:50%
- [ ] 实现有界 critic（tanh + 缩放 `[-2/(1-γ), 0]`）、reward scale 0.01
- [ ] **验证**：critic 初始 Q 值接近 0；前 1–2 万步内成功样本能稳定采样

### Phase 3：phase-out + 在线微调（预估 2–3 天）
- [ ] 监控经验成功率，≥10% 后逐步降低专家数据采样比例（论文：η=0.1）
- [ ] 专家数据完全退出后继续纯在线训练至收敛
- [ ] **验证**：成功率曲线持续上升而非平台期；对比"不注入数据"的对照组（应学不出来，作为基线证明数据有效）

### Phase 4：迁移与评估（预估 1–2 天）
- [ ] `play.py` 评估成功率与任务完成时间，对比脚本老师基线
- [ ] 从 Static 迁移到移动目标（v0），必要时加 domain randomization（物体质量/摩擦/尺寸）
- [ ] 记录：成功率、平均完成时间、方差、失败模式分布

### 里程碑验收
- M1：Phase 0 完成，数据集质量达标（成功率 ≥80%、轨迹多样）
- M2：Phase 2 完成，稀疏训练在 1 万步内出现成功信号（论文：注入数据后几小时内见效）
- M3：Phase 3 完成，成功率 ≥80%（静态目标）
- M4：Phase 4 完成，成功率 ≥80%（移动目标），且完成时间优于脚本老师

---

## 7. 参考资料

### 7.1 核心论文
| 资料 | 链接 | 用途 |
|---|---|---|
| **SMPC 演示 + 稀疏 offline-to-online RL（本篇）** | https://arxiv.org/abs/2608.12063 | 路线总纲；§3.2 稀疏奖励/phase-out、§3.3 数据采集、§4 消融 |
| FastTD3 | https://arxiv.org/abs/2505.22642 | off-policy 算法配方；代码 https://github.com/younggyoseo/FastTD3 |
| Efficient Online RL with Offline Data (Ball et al., ICML 2023) | https://arxiv.org/abs/2302.02948 | 离线数据在线微调奠基工作（phase-out/数据混合理论） |
| Learning Complex Dexterous Manipulation with DRL and Demonstrations (DAPG) | https://arxiv.org/abs/1709.10087 | 路线 B：BC + 演示正则策略梯度 |
| Boosting RL and Planning with Demonstrations: A Survey | https://arxiv.org/abs/2303.13489 | 演示引导 RL 方法全景综述（含 DAPG 公式） |

### 7.2 SMPC / 老师技术
| 资料 | 链接 | 用途 |
|---|---|---|
| Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo (MJPC) | https://arxiv.org/abs/2212.00541 | SMPC 老师原理；代码 https://github.com/google-deepmind/mujoco_mpc |
| Judo: Sampling-Based MPC 开源包 | https://arxiv.org/abs/2506.17184 | 采样型 MPC 的现代开源实现（可选） |
| mjlab: GPU-Accelerated Robot Learning (MuJoCo Warp) | https://arxiv.org/abs/2601.22074 | 论文使用的框架（Isaac Lab API + MuJoCo Warp）；代码 https://github.com/fabio-amadio/mjlab |

### 7.3 落地实现（Isaac Lab / skrl / 数据）
| 资料 | 链接 | 用途 |
|---|---|---|
| Isaac Lab 官方 skrl 训练脚本 | https://github.com/isaac-sim/IsaacLab/blob/main/scripts/reinforcement_learning/skrl/train.py | off-policy 训练入口模板 |
| skrl 官方文档（TD3/SAC 配置） | https://skrl.readthedocs.io | agent YAML 配置、buffer 接口 |
| skrl 社区：Run TD3 in Isaac Lab | https://github.com/Toni-SM/skrl/discussions/261 | Isaac Lab + TD3 完整配置示例 |
| Isaac Lab 遥操作与模仿学习（Mimic/robomimic） | https://github.com/isaac-sim/IsaacLab/tree/main/scripts/tools（record_demos.py） | HDF5 演示数据格式与录制 |
| FlashSAC（off-policy，支持 IsaacLab 等多仿真器） | https://github.com/lorenwel/FlashSAC | 备选 off-policy 框架；论文 https://arxiv.org/abs/2604.04539 |
| SAC-from-scratch in Isaac Lab | https://github.com/DavidH2802/SAC-from-scratch | GPU 并行 SAC 最小实现参考 |
| torchrl + Isaac Lab 集成文档 | https://docs.pytorch.org/rl/stable/reference/isaaclab.html | GPU 常驻 replay buffer 与预填方案 |
| NVIDIA Isaac Lab 官网 | https://developer.nvidia.com/isaac/lab | 框架总览 |

### 7.4 周边参考（可选深挖）
| 资料 | 链接 | 用途 |
|---|---|---|
| awesome-offline-rl（精选清单） | https://github.com/CoderHaoranLi/awesome-offline-rl | 离线 RL 论文地图 |
| Reverse Forward Curriculum Learning (RFCL) | https://arxiv.org/abs/2405.03379 | 反向课程 + 少演示学习，与你现有课程机制对照 |

---

## 8. 风险与注意点

1. **老师质量不足**：脚本状态机在目标位置随机化后成功率可能下降 → 先固定目标分布采集，逐步放开；必要时引入 MPC 或增强状态机（加视觉/接近检测）。
2. **多模态污染**：任何"多种成功行为并存"的演示都会破坏离线初始化 → 采集时强制单一行为模式，定期抽看轨迹。
3. **off-policy 超参敏感**：TD3 在稀疏奖励 + 大批量下容易发散 → 照抄论文有界 critic + reward scale 0.01 + 大 buffer；先小步调 lr。
4. **观测噪声与 asymmetric 设置**：actor 用带噪观测、critic 用真值，需在环境里提供"无噪声观测"通道（Isaac Lab 有现成开关）。
5. **不要回到"继续堆稠密奖励"的老路**：它正是论文要绕开的瓶颈；路线 A/B 验证失败后再评估老师质量，而不是加奖励项。
