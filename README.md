# Eggtart Logistics Robot — Isaac Lab 移动抓取

全向底盘 + 5 轴机械臂 + 力控夹爪，在地面抓取立方体并举升。训练采用 **BC 预训练 → PPO / DAPG 微调**。

注册两个任务：`Isaac-Mobile-Grasp-Eggtart-v0` 和 `Isaac-Mobile-Grasp-Eggtart-BCPPO-v0`。两者共用是
[mobile_grasp_env_cfg.py](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py)
中的 `MobileGraspEnvCfg`，包含机器人场景、动作、观测、奖励、事件、终止条件和 `CurriculumCfg`。

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
| `source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/config/eggtart/` | 唯一任务注册与 PPO 网络/优化器配置 |

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
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --num_envs 2048 --max_iterations 1500 --headless \
    --resume --checkpoint checkpoints/bc_pretrained.pt \
    --demo_data datasets/eggtart_demo.hdf5
```

省略 `--demo_data` 为普通 PPO；再省略 `--resume --checkpoint ...` 为从零 PPO。
训练、策略播放、采集脚本默认任务均为唯一任务名。

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/play.py" \
    --num_envs 16 --checkpoint checkpoints/bc_pretrained.pt --curriculum_stage 2

./isaaclab.sh -p "$PROJECT/scripts/check_demo.py" \
    --data datasets/eggtart_demo.hdf5 --episode 0
```

旧的 `Static-v0`、`Play-v0` 环境入口已删除；`BCPPO-v0` 作为路线 B 环境保留。已有 HDF5 内的旧任务标签会在读取时映射到唯一任务，不修改原文件、数据指纹或验证集划分。checkpoint 的模型权重仍可加载；旧环境配置 pickle 不再作为当前配置入口。

## 奖励和逆向课程

`RewardsCfg` 使用路线 B 的五项任务奖励和两项正则项，**从第 0 步全部启用**：

| 项目 | 权重 |
|---|---:|
| 底盘接近 `base_approach` | 1.0 |
| 末端接近 `ee_reach` | 2.0 |
| 抓取姿态 `grasp_posture_guide` | 1.5 |
| 举升进度 `target_lift_progress` | 10.0 |
| 稳定举升 `grasp` | 30.0 |
| 动作变化 `action_rate` | -0.01 |
| 关节限位 `joint_limits` | -0.3 |

`CurriculumCfg.target_difficulty` 集中定义阶段边界，`EventCfg` 定义位置范围和靠近速度：

| 阶段 | 起始环境步 | 行为 |
|---|---:|---|
| 1 | 0 | 小范围位置采样，目标进入 0.15 m 范围后以 0.05 m/s 辅助靠近 |
| 2 | 96,000 | 相同位置范围，关闭目标靠近辅助 |
| 3 | 144,000 | 扩大初始位置范围，继续关闭辅助 |

所有阶段目标初始速度为零，后续受重力和接触影响。课程在环境重置时更新，TensorBoard 记录 `Curriculum/target_difficulty`。按每次 PPO 更新 24 步计算，阶段 2/3 分别从 4000/6000 次更新开始；1500 次更新仍在阶段 1。`play.py --curriculum_stage` 使用同一组阈值。

教师采集和录制动作回放会关闭目标靠近事件，保持静止目标条件。策略成功率评估应统一初始分布与成功口径；不要启用演示用的 `--scripted_grasp` 来统计策略成功率。

## 物理与接口

- 观测：44 维，训练时有观测噪声，BC/PPO 都不归一化。
- 动作：9 维，底盘体速度 3 + 臂关节位置指令 5 + 夹爪力矩系数 1。
- 臂关节目标：`target_q = 0.5 * action + default_q`；不能把整个动作向量裁剪到 `[-1,1]`。
- 底盘使用 `HolonomicBaseAction` 直接控制根速度；已删除弃用的麦轮逆运动学动作。
- 夹爪最大力矩 1.0 Nm；目标边长 0.032 m、质量 0.01 kg，摩擦 1.0 / 0.8。
- 当前奖励成功判据：目标质心高度至少 0.12 m，连续保持 0.1 s。教师筛选仍使用相对举升 0.15 m、保持 0.5 s。
- 物理时间步 1/120 s，动作间隔 4 步，episode 10 s。

## 验证

```bash
./isaaclab.sh -p "$PROJECT/tests/check_mobile_grasp_env.py" --headless
```

该检查创建真实环境，核对唯一任务注册、44/9 维接口、三个阶段的位置/速度、课程边界和七个奖励项。BC/PPO/DAPG 的 CPU 回归测试命令见训练清单。
