# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""使用 RSL-RL 训练强化学习智能体的脚本"""

"""首先启动 Isaac Sim 模拟器"""

import argparse
import sys

from isaaclab.app import AppLauncher

# 本地导入
import cli_args  # isort: skip


# 添加命令行参数
parser = argparse.ArgumentParser(description="使用 RSL-RL 训练强化学习智能体")
parser.add_argument("--video", action="store_true", default=False, help="训练期间录制视频")
parser.add_argument("--video_length", type=int, default=200, help="录制视频的长度（步数）")
parser.add_argument("--video_interval", type=int, default=2000, help="视频录制间隔（步数）")
parser.add_argument("--num_envs", type=int, default=None, help="要仿真的环境数量")
parser.add_argument("--task", type=str, default=None, help="任务名称")
parser.add_argument("--seed", type=int, default=None, help="环境使用的随机种子")
parser.add_argument("--max_iterations", type=int, default=None, help="强化学习策略训练迭代次数")
# 追加 RSL-RL 命令行参数
cli_args.add_rsl_rl_args(parser)
# 追加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 如果录制视频，始终启用相机
if args_cli.video:
    args_cli.enable_cameras = True

# 清空 sys.argv 供 Hydra 使用
sys.argv = [sys.argv[0]] + hydra_args

# 启动 omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余部分在此之后"""

import gymnasium as gym
import os
import torch
from datetime import datetime
# DAPG: 优先使用项目内 fork 的 rsl_rl（third_party），改源码即时生效
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party"))

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
import pickle
from isaaclab.utils.io import dump_yaml

# 定义 dump_pickle 兼容函数
def dump_pickle(file_path, data):
    """将数据序列化为 pickle 文件

    Args:
        file_path: 文件路径
        data: 要保存的数据
    """
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import importlib.metadata as metadata
installed_version = metadata.version("rsl-rl-lib")
from isaaclab_tasks.utils.hydra import hydra_task_config

# 导入扩展以设置环境任务
import eggtart_grasp.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# 课程学习已移到环境侧（CurriculumManager），见
# source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/curriculums.py
# 和 mobile_grasp_env_cfg.py 的 CurriculumCfg。
#
# 这里原来有个 update_curriculum() + for 循环手动调 runner.learn(1) 的写法，已删除，
# 因为它会破坏 rsl_rl 的日志：learn() 结尾是 current_learning_iteration = it
# （不是 it + 1），而每次调用都从 current_learning_iteration 开始，
# 于是每次 learn(1) 都在重跑第 0 次迭代 —— TensorBoard 所有点写在 step 0，
# checkpoint 反复覆盖 model_0.pt。详细说明见 curriculums.py 的模块 docstring。


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """使用 RSL-RL 智能体进行训练

    Args:
        env_cfg: 环境配置
        agent_cfg: 智能体（PPO）配置
    """
    # 用非 Hydra 命令行参数覆盖配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 设置环境随机种子
    # 注意：某些随机化发生在环境初始化时，因此在此处设置种子
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 指定实验日志目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 实验日志目录: {log_root_path}")
    # 指定运行日志目录: {时间戳}_{运行名称}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # 创建 isaac 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # 包装以进行视频录制
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 训练期间录制视频")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 如果 RL 算法需要，转换为单智能体实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 为 rsl-rl 包装环境
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 从 rsl-rl 创建 runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # 将 git 状态写入日志
    runner.add_git_repo_to_log(__file__)
    # 在创建新 log_dir 之前保存恢复路径
    if agent_cfg.resume:
        # 获取之前 checkpoint 的路径
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: 从以下位置加载模型 checkpoint: {resume_path}")
        # 加载之前训练的模型
        runner.load(resume_path)

    # 将配置转储到日志目录
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # 运行训练（课程学习由环境侧的 CurriculumManager 负责，见 CurriculumCfg）
    print("[INFO] 开始训练（启用课程学习）...")
    print("[INFO] 课程学习策略（step = 迭代数 × num_steps_per_env）:")
    print("  - 阶段 1 - 学习底盘接近 + 朝向")
    print("  - 阶段 2 - 引入末端执行器到达")
    print("  - 阶段 3 - 完整任务（抓取和回收）")


    # learn() 只调一次，让 rsl_rl 自己管迭代计数和日志
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("[INFO] 训练完成！")

    # 关闭模拟器
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭 sim 应用
    simulation_app.close()

