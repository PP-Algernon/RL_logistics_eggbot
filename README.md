# Eggtart Logistics Robot — Isaac Lab 训练项目

Mecanum 轮移动底盘 + 5 轴机械臂的**移动抓取**强化学习项目，基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 与 RSL-RL (PPO)。
任务：机器人整体协同控制（底盘 + 机械臂），追踪并抓取一个**运动的目标物体**，抓取后将机械臂收回。
作为外部扩展（external extension）接入 Isaac Lab。

## 目录结构

```
Eggtart-logistics-robot/
├── pyproject.toml                 # 顶层 isort/pyright 配置
├── scripts/                       # 训练 / 回放 / 列举环境脚本（通用）
│   ├── list_envs.py
│   └── rsl_rl/{train,play,cli_args}.py
└── source/eggtart_grasp/          # 可安装的扩展包
    ├── config/extension.toml
    ├── setup.py · pyproject.toml
    └── eggtart_grasp/
        ├── assets/                # 机器人资产配置 + URDF
        │   ├── eggtart.py         # ArticulationCfg（关节/执行器/麦轮分组）
        │   └── urdf/{robot.urdf, meshes/}
        └── tasks/mobile_grasp/
            ├── mobile_grasp_env_cfg.py   # 基础 ManagerBasedRLEnvCfg
            ├── mdp/               # 自定义 actions/observations/rewards/terminations/target
            └── config/eggtart/    # 具体环境 cfg + gym 注册 + PPO 配置
```

## 机器人概览

- **底盘**：4 个麦克纳姆轮（`wheel_001..004_joint`，连续关节）。策略输出 3 维体速度 `(vx, vy, ωz)`，
  由 [`MecanumBaseAction`](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/actions.py) 经麦轮逆运动学映射到 4 个轮速目标。
- **机械臂**：5 个旋转关节（`link_001..005_joint`），位置控制。
- **夹爪**：`end_effector_joint`，位置控制（开/合）。
- **末端 body**：`end_effector`；**底盘 body**：`base_link`。

> ⚠️ **建模说明**：URDF 中的轮子是纯圆柱体、无麦轮滚子几何，因此横向（vy）平移在物理上不严格成立——
> `MecanumBaseAction` 是**可训练的近似**。几何参数（轮半径 `r`、半轴距 `lx`、半轮距 `ly`、各轮旋向 `wheel_spin_sign`）
> 均已在代码中标注 `TODO` 待按实物标定。

## 安装

```bash
cd /home/pu/isaac-lab
./isaaclab.sh -p -m pip install -e /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/source/eggtart_grasp
```

验证环境已注册：

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/list_envs.py
# 应能看到 Isaac-Mobile-Grasp-Eggtart-v0 / -Play-v0
```

## 训练

```bash
cd /home/pu/isaac-lab
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task=Isaac-Mobile-Grasp-Eggtart-v0 \
    --max_iterations 2000 \
    --num_envs=2048 \
    --headless
```

## 回放 / 可视化

```bash
cd /home/pu/isaac-lab
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/play.py \
    --task=Isaac-Mobile-Grasp-Eggtart-Play-v0 \
    --num_envs=16
```

## 任务设计（MDP）

| 项 | 说明 |
|---|---|
| **Observations** | 关节位置/速度、底盘线/角速度、目标在底盘系下的位置、末端→目标向量、目标世界速度、上一步动作 |
| **Actions** | 底盘体速度 (3) + 机械臂关节位置 (5) + 夹爪 (1) |
| **Rewards** | ①底盘接近目标 ②末端 reach ③抓取 bonus（末端靠近且夹爪闭合）④抓取后收臂；附加动作平滑/关节速度惩罚 |
| **Terminations** | 超时；底盘倾覆 |
| **Target** | 关闭重力的小立方体，赋初速后匀速滑行；reset 随机化位姿与速度，间隔事件周期性改变航向 |

## 待办（TODO，按需标定 / 升级）

- [ ] 标定麦轮几何与旋向（`assets/eggtart.py`、`mdp/actions.py`），让 `(vx, vy, ωz)` 与真实运动一致。
- [ ] 标定 `init_state.pos` 的离地高度与机械臂 home 位姿、夹爪开/合角度。
- [ ] 真实抓取：当前“抓住”为瞬时条件判定，目标未刚性附着到夹爪。
      可加 latch 状态 + 物理 attach 约束（自定义 env 子类）实现真正的 pick-and-carry。
- [ ] 为轮子添加滚子几何（或改用 holonomic 根速度驱动）以获得真实横移。
- [ ] 调权重/课程（reward weights、curriculum）。
```
