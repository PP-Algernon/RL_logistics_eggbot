# Eggtart Logistics Robot — Isaac Lab 训练项目

全向移动底盘 + 5 轴机械臂的**移动抓取**强化学习项目，基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 与 RSL-RL (PPO)。

任务：底盘与机械臂协同，导航到一个放在地面上的小立方体（蛋挞载荷），用夹爪把它**夹住并提起**。
抓取成功的判据是物理量——目标质心被提到阈值高度以上并保持一段时间，不依赖夹爪关节角。

以外部扩展（external extension）的形式接入 Isaac Lab。

## 目录结构

```
Eggtart-logistics-robot/
├── pyproject.toml                     # 顶层 isort/pyright 配置
├── scripts/
│   ├── list_envs.py                   # 列出已注册环境
│   ├── diagnose_grasp.py              # 诊断 grasp 奖励为何为 0（打印权重/距离/夹爪角）
│   └── rsl_rl/{train,play,cli_args}.py
├── .doc/                              # 训练笔记、调参记录、结构图
└── source/eggtart_grasp/              # 可安装的扩展包
    ├── config/extension.toml
    ├── scripts/calibrate_grasp_offset*.py   # 交互式标定抓取点偏移
    └── eggtart_grasp/
        ├── assets/
        │   ├── eggtart.py             # ArticulationCfg + 关节名/几何常量
        │   └── urdf/{robot.urdf, meshes/, usd_cache/}
        ├── utils/scripted_grasp.py    # 演示用脚本化抓取接管（不参与训练）
        └── tasks/mobile_grasp/
            ├── mobile_grasp_env_cfg.py      # 场景 / MDP / 奖励 / 课程 的基础配置
            ├── mdp/                   # actions·observations·rewards·terminations·target·curriculums
            └── config/eggtart/        # 具体 env cfg + gym 注册 + PPO 配置
```

## 机器人概览

- **底盘**：4 个轮关节（`wheel_001..004_joint`）。策略输出 3 维体速度 `(vx, vy, ωz)`，
  由 [`HolonomicBaseAction`](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/actions.py) **直接写入根节点速度**
  实现理想全向移动（上限 0.6 / 0.6 m/s、1.5 rad/s）。
  基于麦轮逆运动学的 `MecanumBaseAction` 仍在代码里，但已弃用——URDF 的轮子是纯圆柱体、
  没有滚子几何，物理上不产生真实侧滑。
- **机械臂**：5 个旋转关节（`link_001..005_joint`），位置控制（`scale=0.5`，`use_default_offset=True`，
  即 action=0 就是回 nominal 姿态）。
- **夹爪**：`end_effector_joint`，**力控**（[`GripperForceAction`](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/actions.py)）。
  把该关节的 stiffness 置 0 退化为"阻尼 + 力矩指令"，策略只输出夹持力，最终停在哪由接触决定，
  自适应物体尺寸。`max_effort=1.0` Nm（实测 0.3 Nm 能夹住 30 mm 立方体，3.0 Nm 会把它挤穿）。
- **参考 body**：末端用 `link_005`（夹爪开合不影响它的坐标系），底盘用四个 `wheel_*` 的几何中心。

### 几张必须记住的坑

| 项 | 值 / 说明 |
|---|---|
| `base_link` 原点 | **不在车身上**，距轮心 0.435 m。底盘位置/朝向类奖励必须用轮心（`EGGTART_WHEEL_JOINT_BODY_REGEX`），不能用 `root_pos_w` |
| 底盘前进方向 | `EGGTART_BASE_FORWARD_AXIS = (0, -1, 0)`，是 **-Y** 不是 +X（轮关节 axis 沿 ±X，轮子绕 X 转） |
| 真实抓取点 | 在两爪之间，相对 `link_005` 原点偏 `EGGTART_EE_GRASP_OFFSET`（≈ 5.6 cm，主要沿 -Z）；所有距离奖励都加这个偏移 |
| 夹爪闭合阈值 | `GRIPPER_CLOSED_THRESHOLD = 0.35`。力控下夹住物体时夹爪被**物体挡住**，到不了硬限位 −0.2（30 mm 物体停在 q≈0.29）。阈值设太小 grasp/lift 奖励恒为 0 |
| nominal 姿态 | 必须落在软限位（0.95×硬限位）内，且 `reset_joints_by_scale` 是按 (0.9, 1.1) **乘**默认值随机的——`link_004` 的软区间不含 0，所以它不能是 0 |

## 安装

```bash
conda activate my_isaac_env
cd /home/pu/isaac-lab
./isaaclab.sh -p -m pip install -e /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/source/eggtart_grasp
```

验证环境已注册：

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/list_envs.py
```

## 已注册环境

| Task ID | 说明 |
|---|---|
| `Isaac-Mobile-Grasp-Eggtart-v0` | 训练主环境。目标 reset 时带 ±0.10 m/s 随机初速（有重力+阻尼，很快停下） |
| `Isaac-Mobile-Grasp-Eggtart-Static-v0` | 目标初速为 0，完全静止；上手/调试推荐 |
| `Isaac-Mobile-Grasp-Eggtart-Play-v0` | 回放/评估，50 环境、关闭观测噪声 |

> episode 中周期性改变目标速度的事件当前**已禁用**（`stage*_start_step` 设为极大值），
> 让策略先专注精准抓取。要恢复"追移动目标"，改 `EventCfg.randomize_target_velocity` 的阶段阈值。

## 训练

```bash
cd /home/pu/isaac-lab
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 2048 \
    --max_iterations 4000 \
    --headless
```

PPO 配置见 [rsl_rl_ppo_cfg.py](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/config/eggtart/agents/rsl_rl_ppo_cfg.py)：
`num_steps_per_env=24`，actor/critic 均为 `[256,128,64]` ELU，`lr=1e-3` adaptive，`entropy_coef=0.005`。

日志落在 `logs/rsl_rl/eggtart_mobile_grasp/<时间戳>/`，用 `tensorboard --logdir logs/rsl_rl` 看。
课程权重会作为 `Curriculum/<term>` 曲线记录，可以直接确认阶段切换时机。

> ⚠️ `runner.learn()` 只能调**一次**。历史上这里写过 `for` 循环 + `learn(1)` 来手动更新课程，
> 那会让 rsl_rl 反复重跑迭代 0（TensorBoard 全平、checkpoint 反复覆盖 `model_0.pt`）。
> 课程现在完全在环境侧的 `CurriculumManager` 里。

## 回放 / 可视化

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Play-v0 \
    --num_envs 16
```

play 脚本会顺带把策略导出为 `exported/policy.pt` 与 `policy.onnx`（在 checkpoint 同级目录）。
额外参数：

- `--curriculum_stage {1,2,3}`：锁定逆课程阶段，单独观察某一阶段的目标行为。
- `--scripted_grasp`：目标已进入可夹范围但策略迟迟不闭爪时，由状态机接管执行
  闭爪 → 保持 → 收臂序列（仅演示，不影响训练）。相关时长用
  `--grasp_patience / --grasp_close_time / --grasp_hold_time / --grasp_retract_time` 调。
  策略只要自己闭了爪，计时器清零、永不接管。

## 任务设计（MDP）

**Observations**（单组 `policy`，拼接，带均匀噪声）：关节位置/速度（相对默认值）、底盘线/角速度、
目标在底盘系下的位置、抓取点→目标向量（以 `link_005` + grasp offset 为参考）、目标世界线速度、上一步动作。

**Actions**（9 维）：底盘体速度 3 + 机械臂关节位置 5 + 夹爪力 1。

**Terminations**：超时（`episode_length_s = 10`）；底盘倾覆（base z 轴向上投影 < 0.5）。

**关键常量**（都在 [mobile_grasp_env_cfg.py](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py) 顶部）：

```python
LIFT_HEIGHT_THRESHOLD   = 0.12   # m，提到这个高度算成功
LIFT_DWELL_TIME         = 0.1    # s，需保持的时间（≈3 步）
GRASP_REACH_THRESHOLD   = 0.05   # m，抓取点算"到达"的距离
GRIPPER_CLOSED_THRESHOLD= 0.35   # rad，力控下的"已闭合"判据
TARGET_VELOCITY_RANGE   = 0.10   # m/s
```

`sim.dt = 1/120`，`decimation = 4` → `step_dt = 1/30 s`。

### 奖励项

| 组 | 项 | 权重 | 作用 |
|---|---|---|---|
| 导航 | `base_approach` | 2.5 | 轮心到目标的水平距离 tanh，带 `standoff=0.5` m（停在目标前方而非压上去） |
| | `base_facing` | 1.2 | 底盘 -Y 轴对准目标 |
| 到达 | `ee_reach` | 5.0 | 抓取点到目标距离 tanh（`std=0.2`） |
| | `ee_distance` | −0.3 | L2 距离惩罚，压紧最后几厘米 |
| | `ee_orientation` | 2.5 | 夹爪开口方向对准目标 |
| | `grasp_posture_guide` | 4.0 | 靠近后向 `EGGTART_GRASP_JOINT_POS` 姿态收敛 |
| 抓取 | `gripper_closure_reward` | 15.0 | 8 cm 内闭爪的**纯正向**奖励（接近度 × 闭合度） |
| | `target_lift_progress` | 30.0 | 提升高度高斯核 × 连续闭合门控，密集梯度；<5 mm 不给分，防"推着走"骗奖励 |
| | `grasp` | 30.0 | 稀疏成功奖励：提到 12 cm 且连续保持 0.1 s |
| 约束 | `arm_comfort` | 0.3 | 关节靠近舒适区 |
| | `joint_limits` | −1.0 | 越软限位惩罚 |
| | `base_vel` | −0.5 | 底盘速度惩罚，带"到位 × 对准"门控——**只在停好之后才罚**，用来治绕圈退化 |

> 绕圈退化：`base_approach` + `base_facing` 在以目标为心的整个圆周上同时取最大，底盘会学会绕圈刷分。
> 无门控的 `action_rate` / `joint_vel` 惩罚治不了它；`base_vel` 的到位门控才是解法。

### 课程学习

两条正交的课程同时在跑，都以 `env.common_step_counter` 计时（每次 `env.step()` +1，与 `num_envs` 无关；
`迭代数 = step / 24`）。

**① 奖励权重课程**（`CurriculumCfg` + [`reward_weight_schedule`](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/curriculums.py)）：
每个奖励项写一条 `[(起始步, 权重), ...]` 时间表，逐步放开难度。

```
step 0      : 只有 base_approach / base_facing / arm_comfort / joint_limits
step 9000   : base_vel 惩罚上线 (-0.5)
step 18000  : ee_reach (5.0)、ee_distance (-0.3)
step 24000  : ee_orientation (2.5)、grasp_posture_guide (4.0)
step 36000  : gripper_closure_reward (15.0)、target_lift_progress (30.0)
step 48000  : grasp (30.0)
```

**② 目标行为课程**（[`reset_target_curriculum`](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/target.py) +
`target_approach_ee_direction`）：目标初始位置从易到难，且早期会主动"喂"到夹爪跟前。

| 阶段 | 步数 | 目标初始位置（机器人系） | 额外行为 |
|---|---|---|---|
| 1 | 0–36000 | 固定正前方 x∈[0.6,0.7] | 抓取点进入 15 cm 内时，目标以 5 cm/s **主动靠近**夹爪开口方向 |
| 2 | 36000–96000 | 同上，静止 | — |
| 3 | 96000+ | 随机 x∈[0.4,1.2], y∈[±0.6] | — |

> `CurriculumManager.compute()` 只在 `_reset_idx()` 里被调用，也就是**有环境重置时**才求值，
> 权重切换最多滞后一个 episode（≈300 步）。并行环境多时可忽略。

## 调试工具

```bash
# 打印各奖励项当前权重、抓取点距离、夹爪角，定位 grasp 奖励为何为 0
./isaaclab.sh -p scripts/diagnose_grasp.py --num_envs 16

# 交互式标定抓取点偏移（WASD/QE 移动，O/C 开合夹爪，P 打印偏移）
./isaaclab.sh -p source/eggtart_grasp/scripts/calibrate_grasp_offset.py
```


