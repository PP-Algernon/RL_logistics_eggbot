# Eggtart Logistics Robot — Isaac Lab 移动抓取

全向底盘 + 5 轴机械臂 + 力控夹爪，在地面抓取立方体并举升。训练采用 **BC 预训练 → PPO / DAPG 微调**。

注册两个任务：`Isaac-Mobile-Grasp-Eggtart-v0` 和 `Isaac-Mobile-Grasp-Eggtart-BCPPO-v0`。两者共用
[mobile_grasp_env_cfg.py](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py)
中的 `MobileGraspEnvCfg`。普通环境使用 `CurriculumCfg` 分阶段调整奖励，BCPPO 使用 `BCCurriculumCfg`，从第 0 步启用路线 B 的奖励权重。场景、动作、观测、奖励函数和目标位置课程共用。

## 代码入口

| 文件 | 用途 |
|---|---|
| `scripts/collect_demo.py` | 脚本教师采集，仅保存成功轨迹和对应初始状态 |
| `scripts/check_demo.py` | 恢复初态、执行同一条轨迹的录制动作 |
| `scripts/pretrain_bc.py` | 按完整轨迹留出验证集，训练 BC actor |
| `scripts/demo_dataset.py` | 数据校验、旧数据标签迁移、轨迹划分 |
| `scripts/rsl_rl/train.py` | PPO；显式提供 `--demo_data` 时启用 DAPG |
| `scripts/rsl_rl/play.py` | 确定性策略播放，关闭观测噪声并导出 JIT/ONNX |
| `source/eggtart_grasp/eggtart_grasp/assets/eggtart.py` | 机器人资产、关节名称与几何常量 |
| `source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/` | 动作、观测、奖励、目标事件和课程的实现函数 |
| `source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/config/eggtart/` | 两种任务注册与 PPO 网络/优化器配置 |

## 运行

```bash
conda activate my_isaac_env
cd /home/pu/isaac-lab
PROJECT=/home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot
./isaaclab.sh -p -m pip install -e "$PROJECT/source/eggtart_grasp"
```

BC 预训练、数据检查、恢复训练及验证步骤见
[BC / PPO / DAPG 清单](.doc/TODO_BC_PPO_DAPG.md)。正式 BC 权重准备好后：

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/train.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 2048 --max_iterations 1500 --headless \
    --resume --checkpoint checkpoints/bc_pretrained.pt \
    --demo_data datasets/eggtart_demo.hdf5
```

省略 `--demo_data` 为普通 PPO；再省略 `--resume --checkpoint ...` 为从零 PPO。
训练和策略播放默认普通环境；BC 微调请显式指定 BCPPO。采集默认 BCPPO，数据回放默认读取文件中的任务名。

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/play.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 16 --checkpoint checkpoints/bc_pretrained.pt --curriculum_stage 2

./isaaclab.sh -p "$PROJECT/scripts/check_demo.py" \
    --data datasets/eggtart_demo.hdf5 --episode 0
```

旧的 `Static-v0`、`Play-v0` 环境入口已删除。读取旧 HDF5 时，Static 标签映射到 BCPPO，Play 标签映射到普通环境，BCPPO 标签保持原样；不修改原文件、数据指纹或验证集划分。checkpoint 的模型权重仍可加载；旧环境配置 pickle 不再作为当前配置入口。

## 奖励和逆向课程

`RewardsCfg` 保留原有奖励函数。BCPPO 的 `BCCurriculumCfg` 使以下七项奖励**从第 0 步全部启用**，其余课程控制项权重固定为零；普通环境保留原奖励调度：

| 项目 | 权重 |
|---|---:|
| 底盘接近 `base_approach` | 1.0 |
| 末端接近 `ee_reach` | 2.0 |
| 抓取姿态 `grasp_posture_guide` | 1.5 |
| 举升进度 `target_lift_progress` | 10.0 |
| 稳定举升 `grasp` | 30.0 |
| 动作变化 `action_rate` | -0.01 |
| 关节限位 `joint_limits` | -0.3 |

两种环境另有固定权重 `5.0` 的 `gripper_holding_object`，从第 0 步启用，不受课程调度影响。它要求夹爪关节角在 0.18–0.32 rad、力矩绝对值大于 0.1 Nm、目标距 `link_005` 的抓取点小于 0.07 m，满足时原始奖励为 1。Isaac Lab 按权重乘动作时间步累计，当前每步贡献约 `5 / 30 = 0.167`。

两种环境的目标位置课程均由 `EventCfg.reset_target` 控制，阶段边界使用 `ANTI_CURRICULUM_*` 常量：

| 阶段 | 起始环境步 | 行为 |
|---|---:|---|
| 1 | 0 | 与采集一致：link_001 前方 0.5 m，侧向 0，世界高度 0.10 m |
| 2 | 96,000 | 保持阶段 1 的初始位置 |
| 3 | 144,000 | 前方 0.4-1.2 m，侧向 ±0.6 m，世界高度 0.015-0.020 m |

位置范围中的 X/Y 是 link_001 局部坐标，前方为 -Y，侧向为 X；Z 是世界高度。采集和训练共用放置函数，所有阶段初速度为零，关闭主动靠近和随机速度，后续受重力和接触影响。阶段在环境重置时更新。按每次 PPO 更新 24 步计算，阶段 2/3 分别从 4000/6000 次更新开始；1500 次更新仍在阶段 1。`play.py --curriculum_stage` 使用同一组阈值。

教师采集和录制动作回放会关闭目标靠近事件，保持静止目标条件。策略成功率评估应统一初始分布与成功口径；不要启用演示用的 `--scripted_grasp` 来统计策略成功率。

## 物理与接口

- 观测：44 维，训练时有观测噪声，BC/PPO 都不归一化。
- 动作：9 维，底盘体速度 3 + 臂关节位置指令 5 + 夹爪力矩系数 1。
- 臂关节目标：`target_q = 0.5 * action + default_q`；不能把整个动作向量裁剪到 `[-1,1]`。
- 底盘使用 `HolonomicBaseAction` 直接控制根速度；保留的麦轮逆运动学动作未启用。
- 夹爪最大力矩 1.0 Nm；目标边长 0.032 m、质量 0.01 kg，摩擦 1.0 / 0.8。
- `grasp` 奖励要求目标质心高度至少 0.12 m、距抓取点小于 0.2 m、线速度小于 3.6 m/s，达标即给基础奖励 1。额外高度奖励从 0.12 到 0.14 m 线性递增至 1，在 0.14–0.16 m 内保持为 1，超过 0.16 m 后按 `exp(-(z - 0.16) / 0.02)` 衰减。因此 0.13 m 时总奖励为 1.5，区间内为 2，0.17 m 约 1.607，0.18 m 约 1.368；任一门控不满足则为 0。以上为乘权重和时间步之前的原始值；`lift_dwell_time` 当前不延迟奖励。教师筛选仍使用相对举升 0.15 m、保持 0.5 s。
- 掉落终止 `target_dropped`：本回合目标质心曾达到 `LIFT_HEIGHT_THRESHOLD`（当前 0.12 m），之后降到 0.05 m 以下即判失败终止（`terminated`，非超时）。初始自然下落不触发；局部重置只清除对应环境的举升记录。
- 物理时间步 1/120 s，动作间隔 4 步，episode 10 s。

## 验证

```bash
./isaaclab.sh -p "$PROJECT/tests/check_mobile_grasp_env.py" --headless
./isaaclab.sh -p "$PROJECT/tests/check_mobile_grasp_env.py" --task Isaac-Mobile-Grasp-Eggtart-v0 --headless
./isaaclab.sh -p "$PROJECT/tests/check_grasp_height_reward.py" --headless
./isaaclab.sh -p "$PROJECT/tests/check_grasp_height_reward.py" --task Isaac-Mobile-Grasp-Eggtart-v0 --headless
./isaaclab.sh -p "$PROJECT/tests/check_target_drop_termination.py" --headless
./isaaclab.sh -p "$PROJECT/tests/check_target_drop_termination.py" --task Isaac-Mobile-Grasp-Eggtart-v0 --headless
```

该检查分别创建两种真实环境，核对任务注册、44/9 维接口、三个阶段的位置/速度、课程边界、奖励权重和局部重置。BC/PPO/DAPG 的 CPU 回归测试命令见训练清单。
