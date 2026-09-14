# 策略评估与代码验证


## 三个播放与评估入口

| 脚本 | 执行内容 | 用途 |
|---|---|---|
| [check_demo.py](../scripts/check_demo.py) | HDF5 中的录制动作，恢复对应初态 | 检查数据和物理回放 |
| [play.py](../scripts/rsl_rl/play.py) | checkpoint 的确定性策略 | 观察动作、录制视频、导出模型 |
| [evaluate.py](../scripts/rsl_rl/evaluate.py) | checkpoint 策略，完整回合统计 | 输出成功率与终止原因 |

`check_demo.py` 的成功不是 BC/PPO 策略的成功。`play.py --scripted_grasp` 会启用脚本接管，不能用于策略自身的成功率验收。

`check_demo.py` 根据选中最长轨迹自动延长回放时限，并关闭训练成功/失败终止项，播完录制动作后再切换；这只影响数据回放。`evaluate.py` 仍按成功、失败和超时时限重置并统计回合。

## 自动评估

从 Isaac Lab 根目录执行；`PROJECT` 按训练指南设置：

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/evaluate.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --checkpoint checkpoints/bc_pretrained.pt \
    --num_envs 64 --num_episodes 1000 --curriculum_stage 2 \
    --success_timeout_s 6 \
    --seed 42 --headless --output evaluation/bc_stage2.json
```

将 checkpoint 替换为 PPO/DAPG 的 `model_*.pt` 即可评估微调模型；阶段 3 使用 `--curriculum_stage 3` 并更换报告名称。评估只接受阶段 2/3，均关闭跟踪辅助。默认确定性动作、无观测噪声；`--stochastic` 使用 checkpoint 保存的动作噪声，`--observation_noise` 开启训练观测噪声。评估直接加载 BC checkpoint 时不会执行训练入口的 0.07 噪声初始化。

| 参数 | 默认值 / 行为 |
|---|---|
| `--task` | `Isaac-Mobile-Grasp-Eggtart-BCPPO-v0` |
| `--num_envs` / `--num_episodes` | 64 / 1000；环境数不超过待统计回合数 |
| `--curriculum_stage` | 2；可选 2、3 |
| `--seed` | 42 |
| `--lift_height` | 当前任务 `LIFT_HEIGHT_THRESHOLD`，即世界高度 0.10 m |
| `--dwell_time` | 当前 `LIFT_SUCCESS_DWELL_TIME`，即 3.0 s |
| `--success_timeout_s` / `--episode_length_s` | 每个实例未成功时自动重置的仿真时限；两个参数名等价，默认读取任务时限 5.5 s |
| `--output` | checkpoint 同级 `evaluation/` 下带时间戳的 JSON |

例如 `--success_timeout_s 6` 表示每个实例从本回合重置起，6 个仿真秒内未成功就计为 `time_out` 失败并自动重置机器人、物块、回合计时和成功保持计时；其他实例继续运行。默认沿用任务时限。时间按动作步向上取整，与电脑实际运行的墙钟时间无关。评估会显式启用超时终止，即使训练配置关闭了该项也会生效。

如需给慢速抓取留出保持时间，可使用 `--success_timeout_s 10`，同时在所有对照模型上使用相同时限。`--dwell_time` 必须严格大于当前 `LIFT_DWELL_TIME=2.5 s`，时限必须不小于成功保持时间。超时回合计入成功率分母，不能通过重置丢弃失败样本。

## 成功口径与统计

当前 `LiftSuccess` 的成功条件是：**物块世界 Z≥0.10 m 连续保持 3.0 s**。中途低于阈值或回合重置就清零，按动作步向上取整。它仅判断高度，没有抓取点距离、速度或接触门控；应将报告理解为“连续高度达标率”，并结合画面检查持物行为。训练奖励的距离/速度条件不会自动传给成功终止项。

评估配置额外启用 `lift_success`，训练默认没有成功终止。成功和超时同一步出现时计成功；翻倒或掉落与成功同时出现时计失败。统计读取终止当步的标志，避免把自动重置后的物块状态作为结果。

每个并行环境分配固定回合配额，第一回合从零计时，最终恰好统计指定数量的完整回合。报告包含：

- `settings`：checkpoint、任务、阶段、种子、动作与噪声模式、成功阈值、时限、实际配置路径。`success_timeout_s` 记录指定时限，`effective_timeout_s` 记录按动作步取整后的实际时限，`timeout_clock` 说明按每回合仿真时间计时。
- `summary`：完成数、成功数、成功率（0–1）、终止原因和 `complete` 状态。
- `episodes`：逐回合结果；旁边的 `.env.yaml` 保存实际环境配置。

中断时仅统计已完成回合，保存报告并标记 `complete=false`。同一回合可能触发多个终止项，原因计数不一定相加等于回合数。

评估读取当前源码配置，不会从旧 checkpoint 恢复旧物理场景。比较模型时固定阶段、时限、阈值、噪声模式、种子和环境数量，并保存代码版本。第一阶段辅助训练奖励不能直接代表无辅助效果。

## 策略可视化

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/play.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 16 --checkpoint checkpoints/bc_pretrained.pt \
    --curriculum_stage 2
```

`play.py` 默认关闭观测噪声与目标辅助，显式选阶段 1 才启用辅助预览。`--video --video_length 300 --headless` 可录制 300 个动作步。脚本还导出 JIT/ONNX 模型；画面用于诊断，完整成功率使用 `evaluate.py`。

## 代码验证入口

本次整理仅检查文档、命令与源码的一致性，没有重新运行以下仿真测试或模型评估。旧运行结果见[归档](archive/README.md)，不代表最新参数已验收。

CPU 回归：

```bash
cd "$PROJECT"
PYTHONPATH=/home/pu/miniconda3/envs/my_isaac_env/lib/python3.11/site-packages \
/home/pu/isaac-lab/_isaac_sim/python.sh -m unittest discover -s tests -v
```

`test_bc_ppo_pipeline.py` 检查数据隔离、BC/PPO/DAPG 衔接与 checkpoint；`test_evaluation_metrics.py` 检查回合计数和成功/失败优先级。

仿真检查以 `check_*.py` 命名，不会由上面的 unittest discover 自动执行。按修改范围选择对应入口，并对两个任务分别运行，例如：

```bash
cd /home/pu/isaac-lab
./isaaclab.sh -p "$PROJECT/tests/check_grasp_height_reward.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 --headless
./isaaclab.sh -p "$PROJECT/tests/check_grasp_height_reward.py" \
    --task Isaac-Mobile-Grasp-Eggtart-v0 --headless
```

| 测试 | 检查内容 |
|---|---|
| [check_mobile_grasp_env.py](../tests/check_mobile_grasp_env.py) | 注册、接口、阶段、奖励与重置 |
| [check_grasp_height_reward.py](../tests/check_grasp_height_reward.py) | 举升高度区间、距离与速度门控 |
| [check_base_slow_reward.py](../tests/check_base_slow_reward.py) | 举升后底盘减速奖励 |
| [check_target_drop_termination.py](../tests/check_target_drop_termination.py) | 掉落、局部重置与自动重置 |
| [check_lift_success_termination.py](../tests/check_lift_success_termination.py) | 连续高度达标计时与终止标志 |
| [check_evaluation_timeout.py](../tests/check_evaluation_timeout.py) | 未成功超时重置、实例隔离、最后一步成功和失败统计 |
| [check_target_tracking_curriculum.py](../tests/check_target_tracking_curriculum.py) | 跟踪强度衰减、激活条件和无辅助阶段 |

部分检查含有历史数值断言，参数调整后需核对它们是否仍表达期望行为。确认日志中具体 PASS 项及异常输出，不能只根据 Kit 进程退出码判断通过。


核对日期：2026-09-14。返回[文档索引](README.md)，运行环境设置见[训练指南](TODO_BC_PPO_DAPG.md#运行环境)。
