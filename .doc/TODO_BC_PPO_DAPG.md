# Eggtart 移动抓取：BC → PPO → DAPG

路线 B 任务：`Isaac-Mobile-Grasp-Eggtart-BCPPO-v0`；普通任务：`Isaac-Mobile-Grasp-Eggtart-v0`。更新：2026-09-12。

## 两种环境配置（2026-09-12）

两种环境共用 `mobile_grasp_env_cfg.py:MobileGraspEnvCfg` 和原有 `RewardsCfg`。
普通环境使用原 `CurriculumCfg` 分阶段调整奖励；BCPPO 使用新的 `BCCurriculumCfg`，从第 0 步启用路线 B 的七项奖励，其他项保持零权重。
目标位置课程独立放在 `EventCfg.reset_target`，按 96,000 / 144,000 环境步切换阶段。
前两阶段与采集的初始位置和速度一致，第三阶段泛化位置；所有阶段关闭主动靠近和随机速度。

只注册上述两个任务，具体配置位于 `config/eggtart/grasp_env_cfg.py`；不再依赖 Static 配置和独立的 Route B 奖励配置。
`BCCurriculumCfg` 位于 `mobile_grasp_env_cfg.py`，路线 B 仅通过它切换奖励权重。PPO 网络和优化器参数仍在
`config/eggtart/agents/rsl_rl_ppo_cfg.py`；机器人资产参数仍在 `assets/eggtart.py`。

Static 和 Play 旧任务名不再接受作为 `--task`；读取 HDF5 时分别映射到 BCPPO 和普通任务，BCPPO 标签保持原样，
不修改原文件或数据指纹。旧模型权重可继续加载，奖励课程由所选任务决定。

## 当前检查结论

当前数据量已更新；下表 BC 权重和 50 次续训成绩来自 9 月 11 日的旧版数据。

BC 的数据、模型接口和 checkpoint 衔接已修正，已完成小规模运行验证。**正式 BC 预训练、策略抓取成功率评估和 1500 次更新的正式训练尚未完成。**

| 项目 | 当前状态 |
|---|---|
| 成功演示 | 7,090 条完整轨迹，1,488,705 个样本（2026-09-12 当前文件），观测 44 维、动作 9 维 |
| 采集成功率 | 54.79%；文件只保留成功轨迹，可以先用现有数据，不以提高教师成功率为前置任务 |
| 数据划分 | 按整条轨迹划分；seed=42 时训练 6,381 条 / 1,339,532 帧，验证 709 条 / 149,173 帧 |
| BC 输入与网络 | 原始观测；actor/critic `[256,128,64]`、ELU，与 PPO 一致 |
| BC 试跑 | 9 月 11 日旧版数据（3,541 条）的 2 epoch 验证 MSE 从 1.786904 降到 0.012915；当前扩大后的数据未重新做正式 BC |
| 权重加载 | 真实 Isaac 环境中 BC → PPO、BC → DAPG 均完成 32 环境、2 次更新 |
| DAPG 实现 | 可显式启用；演示 MSE、指数退火、保存恢复、独立记录参数已实现 |
| 自动化回归 | 5 项通过：轨迹隔离/数据身份、无效数据、旧任务标签迁移、BC/PPO/DAPG 衔接、checkpoint 参数 |
| DAPG 续训短跑 | 512 环境、50 次更新通过；恢复前已有 2 次更新，结束时计数为 52 |
| BC 策略播放入口 | 2 环境、阶段 2、2 帧录制运行完成，JIT/ONNX 导出成功；只验证入口，不评估抓取 |

原清单有以下问题，已修正：

- 入口实际为 `scripts/pretrain_bc.py`，不是 `scripts/bc_pretrain.py`。
- fork 实际为 `third_party/rsl_rl_lib-3.1.2/rsl_rl`，三个入口统一优先导入它。
- rsl_rl 3.1.2 使用 `TensorDict`；原 BC 用旧接口，原 DAPG 补丁也对应其他版本，不能直接粘贴。
- 原 BC 归一化观测，而 PPO 关闭归一化；现在统一原始观测。
- 原 BC 随机按帧划分，邻接帧可能进入训练集和验证集；现在按完整轨迹划分。
- BC 只训练 actor，其优化器不能恢复到 PPO 的 actor+critic+std 优化器。现在自动识别 BC checkpoint，仅加载模型并创建新的 PPO 优化器；PPO/DAPG 续训则恢复其优化器。
- `--resume` 改为开关；`--checkpoint` 支持文件路径，`--load_checkpoint` 为同义参数。
- DAPG 仅在提供 `--demo_data` 时启用，不会因为目录中存在 HDF5 自动开启。

## 路径和环境

| 项目 | 实际路径 |
|---|---|
| 项目 | `/home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot` |
| Isaac Lab | `/home/pu/isaac-lab` |
| 当前数据 | `/home/pu/isaac-lab/datasets/eggtart_demo.hdf5` |
| 建议正式 BC 输出 | `/home/pu/isaac-lab/checkpoints/bc_pretrained.pt` |
| PPO 配置 | `source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/config/eggtart/agents/rsl_rl_ppo_cfg.py` |
| 训练日志 | 启动目录下的 `logs/rsl_rl/eggtart_mobile_grasp/<时间戳>_<run_name>/` |

以下命令在同一个 shell 中执行：

```bash
conda activate my_isaac_env
cd /home/pu/isaac-lab
PROJECT=/home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot
```

BC 不创建 Isaac Sim 应用，但需要 `torch`、`tensordict`、`h5py`、`numpy`。本次检查中，单独调用 conda Python 找不到 torch；使用 Isaac Sim 自带 Python，加上 conda 的 site-packages，可以完成训练。下面 BC 命令采用已验证的启动方式，无需改动全局安装。

## 阶段 1：正式 BC 预训练（待执行）

```bash
PYTHONPATH=/home/pu/miniconda3/envs/my_isaac_env/lib/python3.11/site-packages \
python $PROJECT/scripts/pretrain_bc.py \
    --data datasets/eggtart_demo.hdf5 \
    --output checkpoints/bc_pretrained.pt \
    --epochs 100 --batch_size 4096 --device cuda --seed 42
```

运行前确认输出路径；脚本会覆盖指定路径下的最佳 checkpoint。它最多运行 100 epoch，连续 20 epoch 验证 MSE 无改善则提前停止，只保存验证表现最好的权重，并生成 `bc_pretrained.pt.metrics.json`。

验收：

- [ ] 验证 MSE 下降并趋于平稳，分别查看 base/arm/gripper MSE。
- [ ] 正式 checkpoint 和 metrics JSON 生成；`infos.bc_pretrain=True`，`obs_mean/obs_std=None`。
- [ ] 确定性策略在仿真中独立完成抓取；低 MSE 本身不代表成功率达标。

动作保留采集时的原始尺度。机械臂动作按 `target_q = action × 0.5 + default_q` 转成关节位置目标，出现小于 -1 的值是正常的，不能把整个动作向量裁剪到 `[-1,1]`。BC 训练 actor 的均值；critic 和动作标准差未经过 BC 学习，PPO 初始 `std=1.0` 会引入探索噪声。

## 阶段 2：PPO / DAPG 短跑

### BC → 普通 PPO

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/train.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 512 --max_iterations 50 --headless \
    --resume --checkpoint checkpoints/bc_pretrained.pt \
    --run_name bc_ppo_check
```

确认日志显示本地 fork 路径、BC 初始化信息和 `Standard PPO: demonstration loss disabled`。此步骤验证加载与更新，不预设前 100 次更新的 reward 必须高于从零训练。

### BC → DAPG

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/train.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 512 --max_iterations 50 --headless \
    --resume --checkpoint checkpoints/bc_pretrained.pt \
    --demo_data datasets/eggtart_demo.hdf5 \
    --bc_coef 0.1 --bc_decay 0.95 --bc_min_coef 0.001 \
    --demo_batch_size 1024 --run_name bc_dapg_check
```

这是 **PPO 加演示 MSE 正则项的 DAPG 变体**：

`loss = PPO_loss + max(0.001, 0.1 × 0.95^k) × MSE(actor(demo_obs), demo_action)`

- `k` 是已完成的 DAPG update 次数；首次系数 0.1，第二次 0.095。
- 每个 PPO minibatch 另抽 1,024 个演示样本；整个演示集保存在 CPU。
- 演示损失调用 `act_inference`，不覆盖 PPO rollout 的动作分布。
- 使用 BC 的训练轨迹，验证轨迹继续留出。数据摘要和划分随 checkpoint 保存；文件内容变化会报错，避免悄悄混用数据。
- 从零启用 DAPG 也按轨迹划分数据，默认训练比例 90%。
- 日志记录 `Loss/dapg_bc`、`Loss/dapg_coefficient`，配置写入 `params/dapg.yaml`。
- `--bc_decay 1.0` 可使用固定权重；不需要注释源码。

验收：无异常，损失有限，系数正确衰减，验证轨迹没有被用于演示更新。当前实现没有论文中的 advantage 加权，不预设加速倍数或成功率增益。

### DAPG 断点续训

将 `--checkpoint` 换为已有 DAPG 的 `model_*.pt`，仍显式传入 `--demo_data`。程序恢复优化器、学习率、退火进度和数据划分；保存的退火参数优先于命令行默认值。省略 `--demo_data` 则执行普通 PPO。

`--max_iterations` 表示本次继续进行的更新次数。checkpoint 不保存物理仿真状态；重新启动后环境会重置，环境侧课程计数也从头开始，不能当作逐步完全复现的续训。

## 阶段 3：正式训练前的环境与评估检查

前两阶段的初始场景已与采集对齐。成功口径仍须在正式策略评估时统一：

| 条件 | 采集教师 | 当前两种环境 |
|---|---|---|
| 目标靠近辅助 / 随机速度 | 关闭 / 关闭 | 全阶段关闭 / 关闭 |
| 初始目标位置 | 沿 link_001 前方 0.5 m，世界 z=0.10 m，随后落地 | 前两阶段相同，共用放置函数，初速度为零 |
| 第三阶段位置 | 教师仍使用固定位置 | link_001 局部 x=±0.6 m、y=-1.2 至 -0.4 m，世界 z=0.015-0.020 m |
| 成功口径 | 相对闭爪前抬升 0.15 m，收回后连续保持 0.5 s | 奖励采用目标质心高度 0.12 m、保持 0.1 s |
| 课程 | 教师固定采集流程 | 目标分布阶段 2/3 在 4000/6000 iteration 对应步数切换 |

`CurriculumCfg` 和 `BCCurriculumCfg` 管理各自的奖励权重，目标位置由重置事件管理。训练 1500 次更新仍处于目标分布阶段 1；第 6000 次更新对应的步数之后，下一次重置开始使用第三阶段位置。

后续评估应先在与演示一致的初始状态、静止目标条件下检查 BC，再评估目标位置变化后的泛化。统计学习策略成功率时不要启用 `--scripted_grasp`，也不要把 episode reward 当作成功率。教师约 60% 的采集成功率与学习策略的实际成功率是不同指标。

采集的 `approach_threshold=0.35` 是**进入伸手阶段的距离阈值，单位米**，不是夹爪闭合角度。当前采集入口 close/hold/retract 为 2.00/1.80/1.50 s，抓取就绪 xy/z 容差为 0.008/0.02 m。

## 阶段 4：正式 DAPG 训练（待执行）

完成正式 BC 和上面的闭环检查后：

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/train.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 2048 --max_iterations 1500 --headless \
    --resume --checkpoint checkpoints/bc_pretrained.pt \
    --demo_data datasets/eggtart_demo.hdf5 \
    --bc_coef 0.1 --bc_decay 0.95 --bc_min_coef 0.001 \
    --run_name bc_dapg
```

当前真正注册的 PPO 参数为 lr=1e-3 adaptive、entropy=0.005、每次 24 步、5 epoch、4 minibatch。未接入的 lr=3e-4 配置类已删除，调参只修改实际 runner 配置。

观察 value/surrogate/entropy、演示 MSE、系数，以及单独统计的抓取成功率。系数约在第 90 次更新达到 0.001 下限。

## 阶段 5：评估与对照（待执行）

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/play.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 16 --checkpoint checkpoints/bc_pretrained.pt
```

`--curriculum_stage 1/2/3` 现在从环境配置读取阶段阈值，避免旧硬编码选错阶段。该命令运行确定性策略，可用于观察行为；不提供自动的统一成功率统计。`check_demo.py` 回放的是采集动作，用它看到成功不能证明 BC 策略成功。

分别比较从零 PPO、BC→PPO、BC→DAPG，使用相同种子、环境、训练步数和成功判据。每次调一个参数，先检查动作噪声和学习率是否导致 BC 行为快速退化。

## 本次运行记录

以下 2026-09-11 记录来自合并前的 BCPPO 配置，均为临时验证产物，不是正式模型：

- BC：`/tmp/bc_pipeline_smoke.pt`、`/tmp/bc_pipeline_smoke.pt.metrics.json`、`/tmp/bc_pipeline_smoke.log`。
- 回归测试：`/tmp/bc_ppo_tests.log`。
- 普通 PPO：`/tmp/eggtart_ppo_smoke.log`。
- DAPG 初始短跑：`/tmp/eggtart_dapg_smoke.log`。
- DAPG 续训验证：`/tmp/eggtart_dapg_resume_smoke.log`。
- BC 播放入口：`/tmp/eggtart_bc_play_smoke.log`；导出文件在 `/tmp/exported/`，两帧测试视频在 `/tmp/videos/play/`。
- 50 次续训结果：损失和模型参数均有限，系数从 0.09025 衰减至约 0.00731；末次 reward=19.61，但抓取奖励为 0，不能算抓取验收通过。
- 仿真测试 checkpoint：`/tmp/logs/rsl_rl/eggtart_mobile_grasp/` 下名称含 `pipeline_*_smoke` 的目录。

复跑回归测试：

```bash
cd "$PROJECT"
PYTHONPATH=/home/pu/miniconda3/envs/my_isaac_env/lib/python3.11/site-packages \
/home/pu/isaac-lab/_isaac_sim/python.sh -m unittest discover -s tests -v
```

`init_*` 用于还原演示初始场景；BC 监督训练主要依赖正确对应的 obs/action 和轨迹边界。缺少 init 状态不意味着数据一定不能训练，但会限制物理回放。本次文件含有 init 状态，无需因之前回放显示问题重新采集。

2026-09-12 先前单环境版本的历史验证记录（不代表当前两种配置的验收）：

- `/tmp/eggtart_unified_env.log`：真实环境检查阶段边界、位置采样、零初速、44/9 维接口和七项奖励。
- `/tmp/eggtart_unified_bc_tests.log`：旧数据标签迁移与 BC/PPO/DAPG 回归。
- `/tmp/eggtart_unified_train.log`：唯一任务加载旧演示的 DAPG 两次更新。
- `/tmp/eggtart_unified_replay.log`：读取旧任务标签并回放 episode 0，目标最终高度 0.216 m，相对位置最大偏差显示为 0。
- `/tmp/eggtart_unified_collect.log`：16 环境、350 步小批采集，成功 10/16，保存 2,124 个样本至临时文件。
- `/tmp/eggtart_unified_play.log`：唯一任务的策略播放、阶段 3 选择与两帧视频/模型导出检查。

2026-09-12 当前两种环境与采集初态对齐验证：

- `/tmp/eggtart_spawn_bc.log`、`/tmp/eggtart_spawn_standard.log`：两种环境均通过阶段边界、随机朝向下的教师位置一致性、零初速、第三阶段范围、局部重置和各自奖励课程检查。
- `/tmp/eggtart_spawn_unit.log`：5 项 BC/PPO 回归测试通过。
- `/tmp/eggtart_spawn_collect.log`：16 环境、350 步采集，成功 9/16，仅保存 9 条成功轨迹、1,859 个样本至 `/tmp/eggtart_spawn_collect.hdf5`。这是教师采集流程检查，不是学习策略成功率评估。

## 里程碑

- [x] BC 数据、模型接口、原始观测与权重加载检查
- [x] 按轨迹留出验证集，保存数据身份与划分
- [x] DAPG 代码、保存恢复与基础回归测试
- [x] BC→PPO / BC→DAPG 的真实仿真短跑
- [ ] M1 正式 BC 预训练完成
- [ ] M2 正式 BC 策略独立抓取评估，统一环境与成功判据
- [ ] M3 正式 PPO/DAPG 训练完成
- [ ] M4 学习策略成功率与行为评估、同条件 baseline 对照
