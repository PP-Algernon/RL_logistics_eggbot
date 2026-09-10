# RSL-RL 脚本使用手册

本文档整理 `scripts/rsl_rl` 下三个脚本的实际用法：

```text
scripts/rsl_rl/
├── train.py       PPO 训练入口
├── play.py        加载 checkpoint 回放、评估和导出策略
└── cli_args.py    RSL-RL 参数注册与配置覆盖工具
```

所有命令都应在 Isaac Lab 根目录执行，例如：

```bash
cd /home/pu/isaac-lab
conda activate my_isaac_env
```

推荐使用 Isaac Lab 的启动器：

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py ...
```

直接运行 `python scripts/rsl_rl/train.py` 只有在当前 Python 环境已经正确配置 Isaac Lab、Isaac Sim、RSL-RL 和本项目扩展时才可行。

## 1. 当前可用任务

| 任务名 | 用途 | 默认特点 |
|---|---|---|
| `Isaac-Mobile-Grasp-Eggtart-v0` | 正式训练 | 移动目标版本 |
| `Isaac-Mobile-Grasp-Eggtart-Static-v0` | 调试、预训练、课程学习前期 | 静止目标，更容易收敛 |
| `Isaac-Mobile-Grasp-Eggtart-Play-v0` | 回放和评估 | 使用 PLAY 配置，默认 50 个环境 |

任务名必须和注册名完全一致，大小写也要一致。检查任务是否注册：

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/list_envs.py
```

## 2. 参数来源

命令行参数有三层来源：

1. `train.py` 或 `play.py` 自己定义的参数。
2. `cli_args.py` 注册的 RSL-RL 参数。
3. `AppLauncher.add_app_launcher_args(parser)` 注入的 Isaac Lab / Isaac Sim 启动参数。

`train.py` 使用 `parse_known_args()`，未知参数会继续交给 Hydra；`play.py` 使用严格的 `parse_args()`，拼写错误会直接报错。

## 3. `train.py`：PPO 训练

### 3.1 基本命令

静止目标小规模冒烟测试：

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 256 \
    --max_iterations 100 \
    --headless
```

正式训练移动目标：

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --num_envs 2048 \
    --max_iterations 4000 \
    --headless
```

### 3.2 `train.py` 自定义参数

| 参数 | 类型 | 默认值 | 作用 |
|---|---:|---:|---|
| `--task` | `str` | `None` | Gym 任务名；实际训练必须指定 |
| `--num_envs` | `int` | 配置文件默认值 | 覆盖并行环境数量 |
| `--seed` | `int` | `None` | 覆盖 RSL-RL 和环境随机种子 |
| `--max_iterations` | `int` | 配置文件默认值 `1500` | PPO 训练迭代次数 |
| `--video` | flag | 关闭 | 训练期间录制视频 |
| `--video_length` | `int` | `200` | 每段视频的环境 step 数 |
| `--video_interval` | `int` | `2000` | 每隔多少环境 step 触发一次录制 |

当前 PPO 配置中的 `num_steps_per_env=24`，所以训练迭代与环境 step 的关系近似为：

```text
common_step_counter = iteration * 24
```

这里的 `num_envs` 是并行环境数量，不应乘入课程学习的 step 计算。

视频训练示例：

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 256 \
    --max_iterations 1000 \
    --video \
    --video_interval 500 \
    --video_length 300 \
    --headless
```

训练视频保存在本次训练目录下：

```text
logs/rsl_rl/eggtart_mobile_grasp/<timestamp>/videos/train/
```

### 3.3 RSL-RL 参数

这些参数由 `cli_args.py` 注册：

| 参数 | 类型 | 默认值 | 作用 |
|---|---:|---:|---|
| `--experiment_name` | `str` | 配置值 `eggtart_mobile_grasp` | 参数已注册，但当前 `cli_args.py` 没有把它覆盖到配置；实际日志根目录仍取配置值 |
| `--run_name` | `str` | 配置值 | `train.py` 中作为当前 run 目录后缀；`play.py` 不用它定位 checkpoint |
| `--resume` | `bool` | `None` | 训练脚本是否加载旧 checkpoint |
| `--load_run` | `str` | `None` | 要加载的 run 目录名 |
| `--checkpoint` | `str` | `None` | checkpoint 文件名，例如 `model_1000.pt` |
| `--logger` | `wandb/tensorboard/neptune` | 配置值 | 覆盖日志后端 |
| `--log_project_name` | `str` | `None` | WandB 或 Neptune 项目名 |

`--load_run`、`--checkpoint` 是定位 checkpoint 的关键参数。`--experiment_name` 当前不能通过命令行改变训练日志根目录；如果需要改变实验名，应修改 `rsl_rl_ppo_cfg.py` 中的 `experiment_name`。

当前代码中 `--resume` 是 `type=bool`，因此严格按现状使用时应写成：

```bash
--resume True
```

而不是只写一个没有值的 `--resume`。此外，Python 的 `bool("False")` 仍然是 `True`，因此不要写 `--resume False` 来关闭恢复。若不需要恢复，直接省略 `--resume`。

恢复训练示例：

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 2048 \
    --max_iterations 2000 \
    --resume True \
    --load_run 2026-09-09_14-37-10 \
    --checkpoint model_1000.pt \
    --headless
```

训练脚本会在 `logs/rsl_rl/<experiment_name>/` 下查找 `load_run`，再查找对应的 checkpoint。通常不要把 `--load_run` 写成完整路径；使用日志根目录下的 run 目录名即可。

## 4. `play.py`：回放和评估

### 4.1 加载指定 checkpoint

你原来的命令中参数拼写错误：

```text
错误：--check_point=model_1000.pt
正确：--checkpoint=model_1000.pt
```

修正后的命令：

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 16 \
    --load_run 2026-09-09_14-37-10 \
    --checkpoint model_1000.pt \
    --curriculum_stage 3 \
    --scripted_grasp
```

如果想录制这次回放：

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 16 \
    --load_run 2026-09-09_14-37-10 \
    --checkpoint model_1000.pt \
    --curriculum_stage 3 \
    --scripted_grasp \
    --video \
    --video_length 600
```

回放视频保存在 checkpoint 所在 run 目录：

```text
logs/rsl_rl/eggtart_mobile_grasp/2026-09-09_14-37-10/videos/play/
```

`play.py` 开启视频后会在录制 `video_length` 个环境 step 后自动退出；不开启视频时会持续运行，直到关闭 Isaac Sim 或终止进程。

`play.py` 会直接调用 `get_checkpoint_path(...)` 加载模型，因此不需要 `--resume`；虽然它也会显示并接受这个 RSL-RL 参数，但回放代码不会根据它决定是否加载模型。

### 4.2 `play.py` 自定义参数

| 参数 | 类型 | 默认值 | 作用 |
|---|---:|---:|---|
| `--task` | `str` | `None` | 回放任务名；必须与模型兼容 |
| `--num_envs` | `int` | 任务配置值 | 覆盖并行环境数量 |
| `--video` | flag | 关闭 | 录制一段回放视频 |
| `--video_length` | `int` | `200` | 回放视频长度，单位为环境 step |
| `--disable_fabric` | flag | 关闭 | 调用 `parse_env_cfg` 时禁用 Fabric |
| `--curriculum_stage` | `1/2/3` | `None` | 固定观察某个逆课程阶段 |
| `--scripted_grasp` | flag | 关闭 | 回放中启用脚本抓取接管 |
| `--grasp_patience` | `float` | `1.0` 秒 | 进入抓取范围后等待策略闭合夹爪的时间 |
| `--grasp_close_time` | `float` | `0.3` 秒 | 脚本闭合夹爪持续时间 |
| `--grasp_hold_time` | `float` | `0.4` 秒 | 闭合后保持时间 |
| `--grasp_retract_time` | `float` | `1.0` 秒 | 抓取后收回到 nominal 姿态的时间 |
| `--quiet_grasp` | flag | 关闭 | 不打印脚本抓取状态切换 |

`play.py` 也会接受 `--run_name`、`--logger` 和 `--log_project_name`，但回放使用 `log_dir=None`，这些参数通常不会影响回放结果。定位 checkpoint 时优先使用 `--load_run` 和 `--checkpoint`。

`--scripted_grasp` 只在 `play.py` 回放时接管动作，不参与训练，也不会修改 checkpoint。策略先正常输出动作；当满足脚本条件时，脚本状态机接管夹爪、保持和收臂动作。

### 4.3 `--curriculum_stage`

这个参数把环境的 `common_step_counter` 固定在指定阶段，并在每个环境 step 后恢复该值，因此不会在回放过程中自动切换阶段：

| 参数 | 固定计数值 | 观察内容 |
|---:|---:|---|
| `1` | `0` | 阶段 1，目标主动接近末端执行器 |
| `2` | `36000` | 阶段 2，目标保持静止 |
| `3` | `96000` | 阶段 3，恢复随机移动目标 |

如果只想评估 checkpoint 自己经历的自动课程，不要传 `--curriculum_stage`。

## 5. Isaac Lab / Isaac Sim 启动参数

这部分不是 `train.py` 或 `play.py` 自己实现的，而是由 `AppLauncher.add_app_launcher_args()` 注入。当前环境中可见的参数包括：

| 参数 | 作用 |
|---|---|
| `--headless` | 无 GUI 运行；训练通常建议开启 |
| `--livestream {0,1,2}` | Isaac Sim livestream 模式 |
| `--enable_cameras` | 启用相机；`--video` 会自动设置 |
| `--xr` | 启用 XR |
| `--device DEVICE` | 指定计算设备，例如 `cuda:0` 或 `cpu` |
| `--verbose` | 输出更详细的启动日志 |
| `--info` | 打印启动器信息 |
| `--experience EXPERIENCE` | 指定 Isaac Sim experience 文件 |
| `--rendering_mode {performance,quality,balanced}` | 渲染质量模式 |
| `--kit_args KIT_ARGS` | 传给 Omniverse Kit 的额外参数 |
| `--anim_recording_enabled` | 启用动画录制 |
| `--anim_recording_start_time` | 动画录制开始时间 |
| `--anim_recording_stop_time` | 动画录制结束时间 |

`play.py` 另外定义了 `--disable_fabric`；它不是 AppLauncher 参数，而是回放脚本传给 `parse_env_cfg(..., use_fabric=not args_cli.disable_fabric)` 的配置开关。

常用组合：

```bash
# 训练：无界面、指定 GPU
--headless --device cuda:0

# 回放视频：不要使用 headless，或者确保 Isaac Sim 支持无界面渲染和相机
--video --enable_cameras
```

## 6. 输出目录和 checkpoint

默认 PPO 配置位于：

```text
source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/config/eggtart/agents/rsl_rl_ppo_cfg.py
```

当前默认值包括：

```text
num_steps_per_env = 24
max_iterations    = 1500
save_interval     = 50
experiment_name   = eggtart_mobile_grasp
```

训练输出通常为：

```text
logs/rsl_rl/eggtart_mobile_grasp/<run_name>/
├── model_*.pt
├── params/
│   ├── env.yaml
│   ├── env.pkl
│   ├── agent.yaml
│   └── agent.pkl
├── videos/train/       # train.py 开启 --video 时
└── summaries/          # TensorBoard 等日志，具体取决于 RSL-RL 版本
```

`play.py` 会自动导出策略到所加载 checkpoint 的同级目录：

```text
exported/policy.pt
exported/policy.onnx
```

因此一次成功回放不仅会显示动作，还会生成 JIT 和 ONNX 策略文件。

## 7. 推荐工作流

### 7.1 环境冒烟测试

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Play-v0 \
    --num_envs 4
```

### 7.2 分阶段训练

```bash
# 阶段 1：先使用静止目标和较短训练
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 256 --max_iterations 1000 --video --video_interval 500 --headless

# 阶段 2/3：继续训练或切换正式移动目标配置前，先用 play.py 检查 checkpoint
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 16 --load_run <RUN_NAME> --checkpoint model_1000.pt \
    --curriculum_stage 3 --video --video_length 600
```

### 7.3 TensorBoard

```bash
tensorboard --logdir logs/rsl_rl/eggtart_mobile_grasp
```

浏览器打开：

```text
http://localhost:6006
```

## 8. 常见错误

### `unrecognized arguments: --check_point=...`

参数名拼写错了。代码注册的是：

```bash
--checkpoint model_1000.pt
```

不是：

```bash
--check_point model_1000.pt
```

### `--load_checkpoint` 不识别

本项目的 `cli_args.py` 使用的是 `--checkpoint`。一些 Isaac Lab 官方文档或旧项目使用 `--load_checkpoint`，不能直接套用到当前脚本。

### `--resume` 缺少参数

当前代码把它声明为 `type=bool`，使用：

```bash
--resume True
```

训练恢复时还需要同时提供 `--load_run` 和 `--checkpoint`，否则无法稳定定位模型。

### 找不到 checkpoint

依次检查：

```text
1. --experiment_name 是否和训练时一致
2. --load_run 是否是 logs/rsl_rl/<experiment_name>/ 下的目录名
3. --checkpoint 是否确实存在，例如 model_1000.pt
4. task 对应的 actor/critic 结构和 checkpoint 是否来自同一套配置
```

### 视频没有生成

确认：

```text
1. 命令包含 --video
2. 有可用渲染环境；脚本会自动打开 cameras
3. video_length 大于 0
4. 到达了 RecordVideo 的触发条件
5. 检查对应 run 目录下的 videos/train 或 videos/play
```

### 使用 `--headless` 看不到窗口

这是正常行为。`--headless` 会关闭 GUI；如果要肉眼观察 Isaac Sim 窗口，应去掉 `--headless`。如果只需要 mp4，通常可以使用 headless 加视频录制，但具体表现取决于 Isaac Sim 的渲染配置。

## 9. 参数快速查询

在 Isaac Lab 根目录执行：

```bash
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/train.py --help
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/rsl_rl/play.py --help
```

其中 `play.py --help` 会列出脚本参数、RSL-RL 参数和 AppLauncher 参数。`train.py` 由于使用 Hydra 的 `parse_known_args()`，部分 Hydra 参数可能不会像普通 argparse 参数一样显示在同一份帮助中。
