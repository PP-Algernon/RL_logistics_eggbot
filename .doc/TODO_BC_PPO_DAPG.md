# BC → PPO / DAPG 训练指南与待办

核对日期：2026-09-14。返回[文档索引](README.md)。奖励数值见[配置说明](CONFIGURATION.md)，成功率统计见[评估指南](EVALUATION.md)。

## 运行环境

以下命令从 Isaac Lab 根目录执行。数据、checkpoint 和日志的相对路径都相对于启动目录。

```bash
conda activate my_isaac_env
cd /home/pu/isaac-lab
PROJECT=/home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot
./isaaclab.sh -p -m pip install -e "$PROJECT/source/eggtart_grasp"
```

仿真脚本使用 `./isaaclab.sh -p`，先启动 `AppLauncher` 再导入任务。BC 只需 Python、torch、tensordict、h5py 和 numpy，不启动仿真。下面 BC 命令使用本机 Isaac Sim 的 Python，并补入 conda 的依赖路径；换机器时相应调整。

## 1. 采集与检查演示

[collect_demo.py](../scripts/collect_demo.py) 仅写入成功轨迹及对应初态。用独立输出文件进行小批检查：

```bash
./isaaclab.sh -p "$PROJECT/scripts/collect_demo.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 16 --max_steps 350 --headless \
    --output datasets/eggtart_demo_check.hdf5
```

当前教师从 `link_001` 前方 0.5 m、侧向 0、世界高度 0.10 m 放置目标，初速度为零，关闭跟踪辅助。教师要求相对闭爪前举升 0.15 m，收回后保持 0.5 s，成功后继续夹持至重置。文件中的 `success_rate` 是教师尝试成功率；只保存成功轨迹并不意味着采集成功率为 100%。

**当前时限需先核对**：采集直接继承任务的 5.5 s 回合时限，教师还需完成接近、闭爪、保持及回收；这与早期 10 s 采集设置不同，可能在教师完成前超时。先看小批结果是否有完整成功轨迹，再扩大采集；`--max_steps` 增加总运行步数，不会延长单回合。采集目前没有 `--episode_length_s` 参数。

[check_demo.py](../scripts/check_demo.py) 恢复同一条轨迹的机器人根位姿、关节位置、物块位姿，再顺序执行该轨迹的动作：

```bash
./isaaclab.sh -p "$PROJECT/scripts/check_demo.py" \
    --data datasets/eggtart_demo.hdf5 --episode 0
```

`--episode 0` 固定第一条；`--episode random` 随机一条；`--episode all` 顺序全部。默认按仿真时间播放；`--no-realtime --post_wait 0` 可快速检查。`--save_frames` 保存画面，`--frame_skip` 仅改变保存帧频率，不跳过动作。

回放自动将时限扩展到至少“选中最长轨迹 + 1 个动作步”，并关闭训练的成功、掉落、翻倒等终止项，由脚本在录制动作全部完成后切换轨迹。因此旧 10 s 配置采集的轨迹也能在当前任务下完整播放，最后一帧不会被自动重置覆盖。实际运动仍受当前物理配置影响；脚本保留初态校验与全程相对位置偏差报告，完整播完不等于物理复现成功。Static 数据标签映射到 BCPPO，Play 标签映射到普通任务，BCPPO 标签保持原样；旧 Static/Play ID 不能直接作为 `--task`。

## 2. BC 预训练

```bash
PYTHONPATH=/home/pu/miniconda3/envs/my_isaac_env/lib/python3.11/site-packages \
/home/pu/isaac-lab/_isaac_sim/python.sh "$PROJECT/scripts/pretrain_bc.py" \
    --data datasets/eggtart_demo.hdf5 \
    --output checkpoints/bc_pretrained.pt \
    --epochs 100 --batch_size 4096 --device cuda --seed 42
```

默认按整条轨迹划分 90% 训练、10% 验证，避免相邻帧跨集合泄漏。actor/critic 网络结构均为 `[256,128,64]`、ELU；BC 仅训练 actor 均值，使用原始观测与原始动作尺度，学习率 `3e-4`。臂动作满足 `target_q = 0.5 × action + default_q`，不能将整个动作向量裁剪到 `[-1,1]`。

输出为最佳验证 MSE 的 checkpoint 和同名 `.metrics.json`，连续 20 epoch 无改善会早停。指定输出文件会被更新覆盖。checkpoint 包含 `infos.bc_pretrain=True`、网络配置、数据身份和划分；BC 优化器单独保存，PPO 加载后创建自己的优化器。

验收记录应包含总 MSE、base/arm/gripper 分项 MSE，以及无辅助策略评估结果。历史清单中的 7,090 条数据、旧 MSE 和教师成功率是当时的快照，现已放入[归档](archive/TODO_BC_PPO_DAPG_before_2026-09-14.md)，不作为当前数据量或模型成绩。

## 3. PPO / DAPG 微调

使用 BCPPO 环境执行普通 PPO 微调：

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/train.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 512 --max_iterations 50 --headless \
    --resume --checkpoint checkpoints/bc_pretrained.pt \
    --run_name bc_ppo_check
```

`BCPPO` 选择环境奖励课程，是否启用 DAPG 由 `--demo_data` 决定。训练入口识别 BC checkpoint 后将动作标准差初始化为 **0.07**，随后仍由 PPO 学习；普通 PPO checkpoint 续训保留已保存的标准差。

加入演示损失的 DAPG 变体：

```bash
./isaaclab.sh -p "$PROJECT/scripts/rsl_rl/train.py" \
    --task Isaac-Mobile-Grasp-Eggtart-BCPPO-v0 \
    --num_envs 512 --max_iterations 50 --headless \
    --resume --checkpoint checkpoints/bc_pretrained.pt \
    --demo_data datasets/eggtart_demo.hdf5 \
    --bc_coef 0.1 --bc_decay 0.95 --bc_min_coef 0.001 \
    --demo_batch_size 1024 --run_name bc_dapg_check
```

实现为 `PPO_loss + max(0.001, 0.1 × 0.95^k) × MSE(actor(demo_obs), demo_action)`，`k` 为已完成的 DAPG 更新次数。每个 PPO minibatch 另抽演示样本，只使用训练轨迹；数据变化或任务标签不匹配会报错。该实现没有 advantage 加权。`--bc_decay 1.0` 可固定系数。

先核对日志中的本地 RSL-RL 路径、损失有限性、`Loss/dapg_bc`、`Loss/dapg_coefficient` 和评估结果，再扩大环境数及更新次数。从零训练可移除 `--resume --checkpoint`；采用普通奖励课程时将任务换为 `Isaac-Mobile-Grasp-Eggtart-v0`。

## 4. 续训与产物

将 `--checkpoint` 换为 PPO/DAPG 的 `model_*.pt` 并保留 `--resume`；DAPG 续训仍需显式传入 `--demo_data`。恢复优化器、学习率、演示退火进度与数据划分，保存的 DAPG 退火参数优先于命令行默认值。

新 PPO checkpoint 保存环境步数，训练入口恢复目标课程进度；旧 checkpoint 使用 `(iter + 1) × num_steps_per_env` 估计。BC 初始化从第 0 步开始。物理场景重新重置，环境配置读取当前源码；这不是旧轨迹的逐步复现。

`--max_iterations` 表示本次运行的更新次数。日志在 `logs/rsl_rl/eggtart_mobile_grasp/<时间戳>_<run_name>/`；网络、环境及 DAPG 配置与 checkpoint 随运行保存。默认每 50 次更新保存 checkpoint。参数来源见[配置说明](CONFIGURATION.md)。

## 待核对与后续工作

这些条目按当前源码整理，不代表已经完成模型验收。本次文档整理未启动训练或重新评估 checkpoint。

- [ ] 为当前 checkpoint 建立阶段 2、阶段 3 的无辅助评估记录，保存 JSON、配置、种子和代码版本。
- [ ] 统一持物判据：`LiftSuccess` 仅按高度计时，`retract_bonus_lift` 仅用高度门控；BCPPO 的回收奖励权重为 45。它们不能直接证明物块在夹爪控制下。
- [ ] 复核闭爪门控：BCPPO 的 `gripper_closure_reward` 权重为 7，但没有 `gate_blend` 调度，当前保持 0；普通课程最终推进到 1。
- [ ] 复核生成高度与掉落阈值：当前两者都是 0.10 m。掉落项首次观察到 `z >= 0.10` 就会记录已举起，需核对初始下落是否可能被计入。
- [ ] 核对 5.5 s 时限与教师完整动作、评估连续保持 3 s 是否匹配；数据回放已按录制轨迹长度独立配置时限。
- [ ] 对照最新参数运行相关仿真检查，更新旧断言；记录新的结果后再判断是否通过。
- [ ] 在相同配置、预算和评估条件下比较从零 PPO、BC→PPO、BC→DAPG。
