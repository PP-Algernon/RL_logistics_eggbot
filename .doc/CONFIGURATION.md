# 环境配置说明


## 配置入口

| 内容 | 源码 |
|---|---|
| 两个注册 ID | [config/eggtart/__init__.py](../source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/config/eggtart/__init__.py) |
| 机器人绑定和课程选择 | [grasp_env_cfg.py](../source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/config/eggtart/grasp_env_cfg.py) |
| 场景、奖励参数、课程、终止 | [mobile_grasp_env_cfg.py](../source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py) |
| 奖励 / 目标运动 / 终止实现 | [rewards.py](../source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/rewards.py)、[target.py](../source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/target.py)、[terminations.py](../source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/terminations.py) |
| 网络、PPO 参数 | [rsl_rl_ppo_cfg.py](../source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/config/eggtart/agents/rsl_rl_ppo_cfg.py) |

普通任务 `Isaac-Mobile-Grasp-Eggtart-v0` 使用 `CurriculumCfg`，BCPPO 任务 `Isaac-Mobile-Grasp-Eggtart-BCPPO-v0` 使用 `BCCurriculumCfg`。它们共用 `MobileGraspEnvCfg` 和奖励函数。内部 PLAY 辅助类仍存在，但没有单独注册 Play 任务；Static 和独立 Route B 奖励配置已不作为入口。

## 场景与控制

| 参数 | 当前值 |
|---|---|
| 观测 / 动作维度 | 44 / 9；训练有观测噪声，actor/critic 均不归一化 |
| 动作组成 | 底盘速度 3 + 臂关节位置 5 + 夹爪力矩系数 1 |
| 臂目标换算 | `target_q = 0.5 × action + default_q` |
| 底盘控制 | `HolonomicBaseAction` 写根速度，线速度上限 X/Y 各 0.6 m/s，角速度 1.5 rad/s |
| 夹爪 | 力控，最大力矩 1.0 Nm，阻尼 0.05 |
| 目标物块 | 边长 0.032 m，质量 0.01 kg，静/动摩擦 1.0/0.8，重力开启 |
| 仿真 / 动作时间步 | 1/120 s，decimation=4，即 30 Hz 动作 |
| 默认回合时限 | 5.5 s，即 165 个动作步 |
| 默认并行数 / 间距 | 2048 / 3 m |

机器人资产和几何常量位于 [eggtart.py](../source/eggtart_grasp/eggtart_grasp/assets/eggtart.py)。保留的麦轮逆运动学动作未在当前任务启用。

## 目标位置与跟踪课程

X/Y 使用 `link_001` 局部坐标，前方是 -Y，侧向是 X；Z 直接指定世界高度。每回合机器人根位置 X/Y 各随机 ±0.1 m、yaw 随机约 ±π，关节默认角按 0.9–1.1 倍重置。目标跟随重置后的参考连杆放置，采集和环境共用放置函数。

| 阶段 | 起始环境步 / PPO 更新数 | 初始目标与辅助 |
|---|---|---|
| 1 | 0 / 0 | 局部 X=0、Y=-0.5，世界 Z=0.10 m；跟踪强度由 1 线性降至 0 |
| 2 | 96,000 / 4,000 | 与阶段 1 相同位置，关闭跟踪 |
| 3 | 144,000 / 6,000 | 局部 X=±0.6、Y=-1.2 至 -0.4，世界 Z=0.015–0.020 m；关闭跟踪 |

所有阶段初速度为零，随机速度事件关闭。第一阶段跟踪在每个动作步混合物块自然水平速度与朝向抓取点的速度，比例增益 6/s，上限 0.20 m/s。仅在距离 <0.35 m、抓取点高度 ≤0.10 m、物块高度 <0.10 m、夹爪角度 >0.35 rad 时生效；保留竖直和角速度。闭爪、举起或进入第二阶段后停止写入速度。

强度为 `clamp((96000 - step) / 96000, 0, 1)`，训练 1500 次更新时仍有 62.5% 辅助。位置在重置时切换，奖励课程也在重置时求值；`common_step_counter` 每个动作步加 1，不乘并行环境数。改变 runner 的 24 步 rollout 长度时，需要同步检查课程换算。

采集和动作回放关闭辅助。策略播放默认关闭，显式 `--curriculum_stage 1` 才预览辅助；自动评估仅支持无辅助阶段 2/3。初始位置相同不代表第一阶段训练与采集的动态条件相同。

## 奖励权重

权重由课程表维护，环境初始化从所选课程读取第 0 步权重。下表是源码中的完整权重表；`iter` 按每次 PPO 更新 24 步换算。

| 奖励项 | 普通课程 | BCPPO 课程 |
|---|---|---|
| `base_approach` | 0 iter: 2.5 | 0 iter: 0 |
| `base_facing` | 0 iter: 1.2 | 0 iter: 0 |
| `arm_comfort` | 0 iter: 0.3；750 iter: 0.1 | 0 iter: 0.3 |
| `ee_reach` | 0 iter: 0；750 iter: 3 | 0 iter: 2 |
| `ee_orientation` | 0 iter: 0；750 iter: 2.5 | 0 iter: 0 |
| `grasp_posture_guide` | 0 iter: 0；375 iter: 4 | 0 iter: 0 |
| `ee_distance` | 0 iter: 0；2000 iter: -0.6 | 0 iter: 0 |
| `gripper_holding_object` | 0 iter: 5 | 0 iter: 0 |
| `ee_precision` | 0 iter: 0；2000 iter: 10 | 0 iter: 3 |
| `gripper_closure_reward` | 0 iter: 0；2000 iter: 15 | 0 iter: 7 |
| `target_lift_progress` | 0 iter: 0；2000 iter: 30 | 0 iter: 7 |
| `grasp` | 0 iter: 0；2000 iter: 30 | 0 iter: 30 |
| `retract_bonus_lift` | 0 iter: 0；2000 iter: 5 | 0 iter: 45 |
| `base_slow_after_grasp` | 0 iter: 0；2000 iter: 5 | 0 iter: 5 |
| `joint_limits` | 0 iter: -0.1 | 0 iter: -0.3 |
| `joint_vel` | 0 iter: -0.001；750 iter: -0.0001 | 0 iter: -0.015 |
| `action_rate` | 0 iter: -0.005；750 iter: -0.001 | 0 iter: -0.01 |
| `base_vel` | 0 iter: -0.5；4000 iter: -1 | 0 iter: -0.05 |

普通课程另将 `gripper_closure_reward.gate_blend` 在 0 / 4000 / 5000 / 6000 iter 设为 0 / 0.3 / 0.6 / 1。**BCPPO 没有这项参数调度，保持 `gate_blend=0`**，且当前闭爪奖励权重为 7；不能把它当作“必须真夹住”的约束。BCPPO 的 `gripper_holding_object` 当前权重为 0，也不再是旧文档中的固定 5。

Isaac Lab 将原始奖励乘权重和动作时间步后累计。例如权重 30、原始奖励 1 在一个动作步贡献约 1。

## 举升与终止的实际语义

| 项目 | 当前行为 |
|---|---|
| `grasp` | 目标 Z≥0.10 m，距 `link_005 + EGGTART_EE_GRASP_OFFSET` <0.20 m，目标线速度 <3.6 m/s；达标即给基础奖励 1 |
| `grasp` 额外高度奖励 | Z 从 0.10 到 0.14 m 时线性增加到 1；0.14–0.26 m 为 1；超过 0.26 m 后按 `exp(-(z-0.26)/0.04)` 衰减；总原始奖励范围 0–2 |
| `lift_dwell_time` | 当前传入 2.5 s，但 `GraspBonusLift` 只记录保持时间，不等待此时长才给奖励 |
| `target_lift_progress` | 距离 <0.20 m、速度 <3.6 m/s，并乘夹爪闭合度；高斯核以相对地面 0.14 m 举升量为中心（世界 Z≈0.155 m），std=0.07 |
| `retract_bonus_lift` | 只要求目标 Z≥0.10 m，再按臂关节接近默认姿态给分；没有距离或速度门控 |
| `base_slow_after_grasp` | Z≥0.10 m、距离 <0.20 m、目标速度 <3.6 m/s 时鼓励底盘停稳，线/角速度尺度为 0.15 m/s、0.3 rad/s |
| `target_dropped` | 本回合曾观察到 Z≥0.10 m，之后 Z<0.05 m 即失败终止；各环境独立记录并重置 |
| `base_tipped` | 底盘局部 Z 轴向上投影 <0.5 时失败终止 |
| `time_out` | 默认 5.5 s 超时，返回 truncated |
| `LiftSuccess` | 仅评估脚本启用；仅按 Z≥0.10 m 连续保持 3.0 s 计成功，不检查持物距离和速度 |

`LiftSuccess` 要求保持时间严格大于 `LIFT_DWELL_TIME`，当前即 >2.5 s。训练默认没有成功终止，可以继续奖励持物与回收。距离、速度门控是近似约束，不是接触传感器确认的夹持状态。

当前生成高度与掉落项举升阈值同为 0.10 m；掉落项没有单独的“初始下落豁免”，是否记录取决于它采样时是否达到阈值。相关待核对项见[训练待办](TODO_BC_PPO_DAPG.md#待核对与后续工作)。

## PPO 参数

当前 runner：actor/critic `[256,128,64]`、ELU，学习率 `7e-5`、adaptive，entropy=0.005，gamma=0.99，lambda=0.95，clip=0.2，每次采集 24 步、更新 5 epoch、4 minibatch，默认 1500 次更新、每 50 次保存。

从零 PPO 初始动作标准差为 1.0；[train.py](../scripts/rsl_rl/train.py) 对 BC checkpoint 单独初始化为 0.07。PPO checkpoint 恢复已有噪声与课程步数；环境物理配置始终以当前源码和脚本覆盖为准。


核对日期：2026-09-14。返回[文档索引](README.md)。本页数值对应当前源码；运行时的实际配置还受脚本及命令行覆盖影响。