# Eggtart 移动操作机器人 — Isaac Lab 训练项目初始化

## Context（背景）

当前项目 `Eggtart-logistics-robot/` 只有机器人模型（`USDs/` 下两个 zip：URDF+STL / MJCF+STL），没有任何训练代码。
目标是参考 `guguji_simulation/guguji_isaaclab`（基于 IsaacLabExtensionTemplate 的外置扩展），
搭建一套可直接 `train.py` 启动的 Isaac Lab 训练工程。

**机器人**（读自 `robot.urdf`）：4 麦克纳姆轮底盘 + 5 轴机械臂，是**轮式移动操作机器人**。
- 机械臂：6 个 revolute 关节 `link_001_joint`~`link_005_joint`（五轴）+ `end_effector_joint`（夹爪舵机轴，1-DOF 夹爪），effort=10 / vel=5
- 底盘：4 个 continuous 麦轮关节 `wheel_001_joint`~`wheel_004_joint`，effort=10 / vel=5
- 链路：`base_link` → 5 臂 link → `end_effector`；4 轮直接挂在 `base_link`

**已确认的设计决策**：
1. **整体控制**：策略同时控制底盘（麦轮）+ 机械臂
2. **抓取判定**：`end_effector_joint` 当作夹爪舵机；成功 = 末端接近移动目标 → 夹爪闭合夹住 → 收回机械臂
3. **RL 框架**：rsl_rl（PPO）

**环境**：Isaac Lab **2.3.2** 在 `/home/pu/isaac-lab`；自带 `reach`/`lift` manipulation 模板可借鉴 MDP。
guguji 是 locomotion（步态/速度跟踪），任务内容**不可直接复用**，只复用目录骨架与训练脚本。

> 注意：guguji 任务是双足行走，本项目是移动抓取。因此**复用其工程骨架（pyproject/setup/extension.toml/scripts）**，
> 而 MDP（reward/obs/action/termination）按 `reach` 模板风格 + 本任务需求**全新编写**。

---

## 目标目录结构

```
Eggtart-logistics-robot/
├── README.md                         # 安装/训练/play 说明
├── pyproject.toml                    # isort/pyright 顶层配置（改 known_firstparty）
├── .gitignore                        # 复用 guguji 版本
├── scripts/rsl_rl/                   # 直接复用 guguji 的 rsl_rl 脚本（通用）
│   ├── train.py
│   ├── play.py
│   └── cli_args.py
└── source/eggtart_grasp/
    ├── config/extension.toml         # 扩展元信息 + isaaclab 依赖
    ├── pyproject.toml
    ├── setup.py                      # name="eggtart_grasp"
    └── eggtart_grasp/
        ├── __init__.py               # from .tasks import *
        ├── assets/
        │   ├── __init__.py           # 导出 EGGTART_CFG
        │   ├── eggtart.py            # ArticulationCfg（UrdfFileCfg 加载 robot.urdf）
        │   └── urdf/                 # 从 zip 解压：robot.urdf + meshes/
        └── tasks/
            ├── __init__.py           # import_packages（同 guguji）
            └── mobile_grasp/
                ├── __init__.py
                ├── mobile_grasp_env_cfg.py        # 基类 env：Scene/Obs/Action/Reward/...
                ├── mdp/
                │   ├── __init__.py                # re-export stock mdp + 本地模块
                │   ├── actions.py                 # MecanumBaseAction（自定义）
                │   ├── observations.py            # 目标相对位姿、夹爪状态等
                │   ├── rewards.py                 # 接近/对齐/抓取/收回 分阶段奖励
                │   ├── terminations.py            # 抓取成功 / 翻倒 / 超时
                │   └── target.py                  # 移动目标的运动 event
                └── config/eggtart/
                    ├── __init__.py                # gym.register（2 个 task id）
                    ├── grasp_env_cfg.py           # 具体 env cfg + _PLAY
                    └── agents/
                        ├── __init__.py
                        └── rsl_rl_ppo_cfg.py      # PPO 超参
```

注册的 task id：`Isaac-Mobile-Grasp-Eggtart-v0` 和 `Isaac-Mobile-Grasp-Eggtart-Play-v0`。

---

## 关键实现要点

### 1. 机器人资产 `assets/eggtart.py`
- 用 `sim_utils.UrdfFileCfg` 加载解压后的 `assets/urdf/robot.urdf`，`fix_base=False`（移动底盘），`merge_fixed_joints=True`，`activate_contact_sensors=True`，首次运行生成 USD 缓存到 `assets/urdf/usd_cache/`。
- **执行器分组**（`ImplicitActuatorCfg`）：
  - `arm`：`link_00[1-5]_joint`，position 控制，stiffness/damping 适配 effort=10。
  - `gripper`：`end_effector_joint`，position 控制（开/合）。
  - `wheels`：`wheel_00[1-4]_joint`，**velocity 控制**（stiffness=0, damping>0），effort=10/vel=5。
- 给一组合理的初始关节角 `init_state`（臂收拢、夹爪张开、轮速 0）。
- ⚠️ 麦轮建模说明写进文件注释：URDF 轮子是普通圆柱，没有麦轮滚子几何，**侧向力学不真实**。本方案用「车体速度 → 4 轮转速」的麦轮运动学映射（见下）做近似，作为可训练起点；若需高保真侧移再补滚子或换 holonomic 直驱。

### 2. 自定义动作 `mdp/actions.py` — `MecanumBaseAction`
仿照 guguji 的 `ReferenceGaitAction`（`ActionTerm` + `@configclass ...Cfg`）写法。
- 策略输出 3 维车体速度指令 `(v_x, v_y, ω_z)`，经麦轮逆运动学 `wheel_vel = J⁻¹ · [v_x, v_y, ω_z]` 映射为 4 个轮关节速度目标，写入 articulation 的 velocity target。
- 参数：轮距 `L_x`、`L_y`、轮半径 `r`（从 URDF 量取，注释标 TODO 校准）、速度缩放上限。
- **总动作空间** = 3（底盘）+ 6（5 臂 + 夹爪）= 9 维。臂/夹爪用 stock `JointPositionActionCfg`。

### 3. 场景与移动目标 `mobile_grasp_env_cfg.py` + `mdp/target.py`
- Scene：`GroundPlaneCfg` 地面、`DomeLightCfg` 光、机器人、**目标物体**（`RigidObjectCfg`，小立方体/圆柱代表蛋挞）。无桌子（地面移动场景）。
- 移动目标：`mdp/target.py` 写一个 `interval`/每步 event，按可配置轨迹（直线/圆周 + 随机速度）更新目标 root 位姿或速度，reset 时随机化起点与速度。
- Commands：不用 stock 位姿命令；目标本身就是被追踪对象，相对量进 observation。

### 4. 观测 `mdp/observations.py`
- 机器人本体：`joint_pos_rel`、`joint_vel_rel`、`last_action`（stock）。
- 底盘状态：base 线/角速度（stock `base_lin_vel`/`base_ang_vel`）。
- 任务相关（自定义）：目标相对 base 的位置、相对末端的位置/距离、目标速度、夹爪开合度、是否已夹住标志。

### 5. 奖励 `mdp/rewards.py`（分阶段，借鉴 reach 的 tanh kernel）
- `base_approach`：底盘水平距离目标越近越好（tanh）。
- `ee_reach`：末端到目标距离（tanh，借鉴 `position_command_error_tanh`）。
- `ee_align`：末端朝向对齐（可选）。
- `grasp_success`：末端足够近 + 夹爪闭合 → 大正奖励 + 置「已夹住」标志。
- `retract`：已夹住后，臂关节回到收拢位姿 + 目标随末端 → 奖励。
- 惩罚项：`action_rate_l2`、`joint_vel_l2`、底盘能耗、翻倒。
- 用 `CurriculumCfg` 后期加大平滑性惩罚（同 reach）。

### 6. 终止 `mdp/terminations.py`
- `time_out`（stock）。
- `grasp_done`：成功夹住并收回后判成功。
- `base_tipped`：车体倾覆/姿态异常提前终止。

### 7. 具体 env + PPO 配置
- `grasp_env_cfg.py`：`EggtartMobileGraspEnvCfg(EggtartMobileGraspEnvCfg base)` 装配 `EGGTART_CFG`、`num_envs=2048`、`env_spacing`、episode 时长；`_PLAY` 版本 `num_envs=50`、关闭噪声/外力。
- `rsl_rl_ppo_cfg.py`：仿 guguji `GugujiFlatPPORunnerCfg`，`ActorCritic` MLP `[256,256,128]`，`num_steps_per_env=24`，`max_iterations≈3000`，`experiment_name="eggtart_grasp"`。

### 8. 训练脚本
- `scripts/rsl_rl/{train,play,cli_args}.py` 从 guguji **原样复制**（Isaac Lab 通用 rsl_rl 脚本，不依赖具体包名）。

### 9. 顶层配置
- `pyproject.toml`：复制 guguji，`known_firstparty="eggtart_grasp"`。
- `extension.toml`/`setup.py`：包名、title、module 改成 `eggtart_grasp`。
- `README.md`：写明用 `/home/pu/isaac-lab/isaaclab.sh -p` 或 conda `my_isaac_env` 跑，安装 `pip install -e source/eggtart_grasp`，以及 train/play 命令。

---

## 验证（end-to-end）

1. **解压 & 资产可加载**：解压 URDF zip 到 `assets/urdf/`，确认 `robot.urdf` + `meshes/` 就位。
2. **装扩展**：`cd Eggtart-logistics-robot && /home/pu/isaac-lab/isaaclab.sh -p -m pip install -e source/eggtart_grasp`（或在 `my_isaac_env` 内）。
3. **环境注册**：`isaaclab.sh -p scripts/list_envs.py`（可选拷贝该脚本）应能看到 `Isaac-Mobile-Grasp-Eggtart-v0`。
4. **冒烟训练**：
   `isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Mobile-Grasp-Eggtart-v0 --num_envs 64 --headless --max_iterations 5`
   —— 能跑通几轮、loss 有输出、无报错即视为骨架成功（不要求收敛）。
5. **可视化 play**：训练出 checkpoint 后
   `isaaclab.sh -p scripts/rsl_rl/play.py --task Isaac-Mobile-Grasp-Eggtart-Play-v0 --num_envs 16`。

> 麦轮逆运动学参数、奖励权重、目标运动轨迹均标注 `TODO 调参`——骨架先保证「能注册、能加载机器人、能起训」，调到真能抓取是后续迭代。

## 待确认的小默认值（无异议即按此执行）
- Python 包名：`eggtart_grasp`；Task id：`Isaac-Mobile-Grasp-Eggtart-v0`。
- URDF 解压目标：`source/eggtart_grasp/eggtart_grasp/assets/urdf/`（随包走，路径稳定）。
- 运行用 `/home/pu/isaac-lab/isaaclab.sh`（其内置 python 环境）。
