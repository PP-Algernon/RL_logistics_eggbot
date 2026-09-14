# Eggtart Logistics Robot

基于 Isaac Lab 的移动抓取工程：全向底盘、5 轴机械臂和力控夹爪，采用 **BC 预训练 → PPO / DAPG 微调**。文档按 2026-09-14 工作区代码核对。

## 文档入口

| 文档 | 内容 |
|---|---|
| [文档索引](.doc/README.md) | 当前文档、阅读顺序与维护约定 |
| [训练指南与待办](.doc/TODO_BC_PPO_DAPG.md) | 采集、回放、BC、PPO/DAPG、续训和待核对项 |
| [环境配置说明](.doc/CONFIGURATION.md) | 两种任务、奖励权重、目标课程、物理参数 |
| [评估与验证](.doc/EVALUATION.md) | 成功口径、评估命令、输出与测试入口 |
| [历史归档](.doc/archive/README.md) | 旧指南、路线方案、早期实验记录 |

## 环境与安装

| 任务 ID | 奖励课程 |
|---|---|
| `Isaac-Mobile-Grasp-Eggtart-v0` | `CurriculumCfg`，分阶段启用奖励 |
| `Isaac-Mobile-Grasp-Eggtart-BCPPO-v0` | `BCCurriculumCfg`，BC 后微调权重 |

两者共用 [MobileGraspEnvCfg](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py)。训练与策略播放默认普通任务；采集和自动评估默认 BCPPO。

```bash
conda activate my_isaac_env
cd /home/pu/isaac-lab
PROJECT=/home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot
./isaaclab.sh -p -m pip install -e "$PROJECT/source/eggtart_grasp"
```

有 BC checkpoint 后，使用路线 B 的 PPO 微调：

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/train.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 512 --max_iterations 50 --headless \
    --resume --checkpoint checkpoints/bc_pretrained.pt \
    --run_name bc_ppo_check
```

增加 `--demo_data datasets/eggtart_demo.hdf5` 启用 DAPG 演示损失。模型抓取效果使用 [evaluate.py](scripts/rsl_rl/evaluate.py) 评估；训练奖励和演示回放成功都不能直接当作策略成功率。

## 代码入口

| 路径 | 用途 |
|---|---|
| [collect_demo.py](scripts/collect_demo.py) / [check_demo.py](scripts/check_demo.py) | 成功演示采集 / 对应初态与动作回放 |
| [pretrain_bc.py](scripts/pretrain_bc.py) / [demo_dataset.py](scripts/demo_dataset.py) | BC 训练 / 数据校验与整轨迹划分 |
| [train.py](scripts/rsl_rl/train.py) / [play.py](scripts/rsl_rl/play.py) | PPO/DAPG 训练 / 策略可视化 |
| [evaluate.py](scripts/rsl_rl/evaluate.py) | 完整回合成功率评估 |
| [mobile_grasp_env_cfg.py](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py) | 场景、奖励参数、课程与终止条件 |
| [eggtart.py](source/eggtart_grasp/eggtart_grasp/assets/eggtart.py) | 机器人资产、关节与抓取点常量 |
| [rsl_rl_ppo_cfg.py](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/config/eggtart/agents/rsl_rl_ppo_cfg.py) | 网络与 PPO 优化器 |

当前训练第一阶段有渐退的目标跟踪辅助；采集、自动评估关闭辅助。默认回合时限为 **5.5 秒**。奖励、评估和教师筛选使用不同判据，具体差异与已知限制见[配置说明](.doc/CONFIGURATION.md)和[训练待办](.doc/TODO_BC_PPO_DAPG.md#待核对与后续工作)。
