# 相关项目与参照代码（已验证）

本清单中每个 GitHub 仓库都经 GitHub API 确认存在，附抓取时的 star 数与最后推送时间（抓取日期 2026-08-28）。
本地路径均在 `/home/pu/isaac-lab`（IsaacLab **v2.3.2**，commit `37ddf62`）上 `ls` 确认过。

> 未标注"已读源码"的条目，我只核对了仓库存在性与 README/描述，没有逐行读实现。

---

## 一、最高优先级：你本地就有的官方参照

这几个不用下载，直接读。对你的价值高于任何第三方仓库。

### 1. 官方 lift 任务 —— 你的 ③抓取 / ④收臂 的权威对照 ✅已读源码

```
/home/pu/isaac-lab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/
├── mdp/rewards.py        # object_is_lifted / object_ee_distance / object_goal_distance
└── lift_env_cfg.py       # RewardsCfg 权重
```

**两个直接影响你的发现：**

**(a) 官方用「物体被抬起的高度」判定抓取成功，不看夹爪关节角**

```python
def object_is_lifted(env, minimal_height, object_cfg) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)
```

你的 `grasp_bonus` 判定是 `(dist < 0.05) & (gripper_pos < -0.6)`——末端够近 **且** 夹爪关节角小于阈值。
问题在于这两个条件策略都能"作弊"满足：把夹爪一直闭合着、末端凑到物体附近就能白拿 5.0 的奖励，
而物体根本没被夹住。官方的 height 判定是**物理结果**，作弊不了。
你的目标立方体已禁用重力（`disable_gravity=True`），height 判定不能直接照搬，但思路可以借：
用「物体是否跟着末端一起动」作为验证信号，例如物体与末端的相对位置在连续若干步内保持不变。

**(b) 官方的稀疏成功奖励权重远大于 dense shaping**

| | 官方 lift | 你的配置 |
|---|---|---|
| reach (dense) | `reaching_object` **1.0** | `ee_reach` 2.0 |
| 成功 (稀疏) | `lifting_object` **15.0** | `grasp` 5.0 |
| 后续阶段 | `object_goal_tracking` **16.0** | `retract` 2.0 |
| action_rate | **-1e-4** | -1e-3（大 10 倍） |
| joint_vel | -1e-4 | -1e-4（一致） |

官方是 1 : 15 : 16，稀疏奖励绝对主导；你是 2 : 5 : 2，dense 项占比偏高。
dense 项权重相对过大时，策略容易停在"末端悬在物体旁边持续收 shaping 奖励"的局部最优，
而不去冒险尝试真正抓取。这是我看到你配置后最建议先调的一处。
另外你的 `action_rate` 是官方 10 倍，如果训练出来动作发木、不敢动，可以先把它降到 -1e-4。

### 2. 官方 locomanipulation/pick_place —— 移动底盘 + 臂 + 抓取的官方组合 ✅路径已确认

```
/home/pu/isaac-lab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/
├── pick_place/            # G1 人形的 pick-place（locomanipulation_g1_env_cfg.py）
└── tracking/
```

机器人是 G1 人形而非轮式底盘，但**「移动平台 + 上肢抓取」的任务分解、观测组织、奖励分层**是官方级参照，
比我上一版给你列的任何第三方仓库都可靠。我只确认了文件存在，没读内部实现。

### 3. 官方 SurfaceGripper —— 你 TODO 3「真实抓取」的官方解 ✅本地已确认存在

```
/home/pu/isaac-lab/source/isaaclab/isaaclab/assets/surface_gripper/
/home/pu/isaac-lab/source/isaaclab/isaaclab/envs/mdp/actions/surface_gripper_actions.py
```

`SurfaceGripperBinaryAction` 的 docstring（已读）：

```
- [-1, -0.3] --> Gripper is Opening
- [-0.3, 0.3] --> Gripper is Idle (do nothing)
- [0.3,  1  ] --> Gripper is Closing
```

这是吸附式夹爪，闭合时**由物理引擎创建 D6 关节约束**把物体真正连到夹爪上——正是你 TODO 里
"可加 latch 状态 + 物理 attach 约束"想要的效果，且是官方实现，不用自己写 USD 约束。

⚠️ 代价：需要在机器人 USD 里配置 Attachment Points（D6 Joint）。社区反馈这一步容易踩坑，
NVIDIA 论坛有 Isaac Sim 5.1 下自定义 URDF 配 SurfaceGripper 报 D6Joint 错误 / 物理爆炸的案例
（[NVIDIA 开发者论坛](https://forums.developer.nvidia.com/t/title-isaac-sim-5-1-0-surface-gripper-not-working-on-custom-urdf-robot-d6joint-errors-physics-explosion/363946)、
[IsaacSim Discussion #542](https://github.com/isaac-sim/IsaacSim/discussions/542)）。
建议等纯状态策略训通之后再动，别现在引入。

官方教程：[Interacting with a surface gripper](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/run_surface_gripper.html)

### 4. 官方 navigation 任务 —— 分层控制参照 ✅路径已确认

```
/home/pu/isaac-lab/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/
└── mdp/pre_trained_policy_action.py   # 把预训练策略当作 action term
```

`pre_trained_policy_action.py` 这个模式值得知道：高层策略输出速度命令，底层用**已训练好的**策略执行。
如果你后面想把"底盘导航"和"机械臂抓取"分成两级训练（先训底盘跟踪速度命令，再训上层），这就是官方脚手架。

---

## 二、第三方仓库（GitHub API 已验证存在）

### 5. protomota/ogre-lab —— 与你底盘部分最贴合 ✅已读部分源码

| | |
|---|---|
| 链接 | https://github.com/protomota/ogre-lab |
| star | **0**（最后推送 2025-12-09） |
| 内容 | 4 轮麦克纳姆轮机器人的 Isaac Lab 速度跟踪策略，RSL-RL 训练，导出 ONNX + JIT 供 ROS2 Nav2 当局部控制器 |

star 是 0，是个人项目，但**技术栈和你几乎重合**：麦轮 + Isaac Lab + RSL-RL + ONNX 导出
（正好衔接你之前定的 MuJoCo sim2sim 方案）。它是 direct workflow（你是 manager-based），配置组织方式不同。

**(a) 它的轮子标定脚本，就是你 TODO 1 需要的东西**

```
scripts/test_wheel_direction.py   # 不加任何符号修正，直接发轮速，看机器人往哪走
scripts/test_each_wheel.py        # 逐个轮子单独测
scripts/test_forward_only.py
scripts/test_spin_only.py
scripts/print_joint_order.py      # 打印 find_joints 返回的真实顺序
```

`test_wheel_direction.py` 的方法（已读）非常朴素有效——发两组固定轮速，肉眼看 GUI：

```
1. 发 [+6, +6, +6, +6] rad/s 3 秒 → 观察是否原地打转
2. 发 [+6, -6, +6, -6] rad/s 3 秒 → 观察是否直线前进
```

它记录的结论：右侧两轮（FR/RR）URDF 轴向相反，所以前进需要 `[+, -, +, -]`。

**(b) `print_joint_order.py` 指出一个你可能中招的坑**

它源码里写明：`find_joints(["fl_joint","fr_joint","rl_joint","rr_joint"])` 返回的**实际顺序是 FR, RR, RL, FL**，
和传入顺序不一致，所以它在 `_apply_action` 里按 index 0/1 是右侧轮来修正符号。

你的 [actions.py:44-46](source/eggtart_grasp/eggtart_grasp/tasks/mobile_grasp/mdp/actions.py#L44-L46) 用了
`preserve_order=True`，理论上规避了这个坑——但**建议实测确认一次**，因为你的 `_ik` 符号矩阵和
`wheel_spin_sign` 都是按 FL/FR/RL/RR 顺序硬编码的，一旦顺序错位，横移和自转的符号会全错，
而这种错误在训练中不会报错，只会表现为"策略学得很差"，极难排查。

**(c) 它的几何参数写法可对照**

```python
wheel_radius = 0.040      # 40mm    (你: 0.043)
wheelbase    = 0.095      # 95mm    (你: half_wheelbase=0.08 → 全长 0.16)
track_width  = 0.205      # 205mm   (你: half_track=0.097 → 全宽 0.194)
self.L = (self.cfg.wheelbase + self.cfg.track_width) / 2.0
```

注意它存的是**全长**再除 2，你存的是**半长**直接相加（`self._k = half_wheelbase + half_track`）。
两者数学等价，但读代码时容易混淆——它的 `L = (0.095+0.205)/2 = 0.15`，你的 `k = 0.08+0.097 = 0.177`。
体型接近，你的参数不算离谱。

### 6. UWRobotLearning/WheeledLab —— 轮式机器人 + IsaacLab，最活跃的同类项目

| | |
|---|---|
| 链接 | https://github.com/UWRobotLearning/WheeledLab |
| star | **287**（最后推送 2026-08-13，活跃） |
| 内容 | 轮式机器人的环境 / 资产 / 工作流，集成 IsaacLab；华盛顿大学机器人学习实验室出品 |

轮式机器人在 IsaacLab 里的 star 最高、维护最活跃的项目。侧重 sim-to-real 的轮式平台（含漂移、越野等），
不含机械臂。看它的**工程组织方式和 sim2real 流程**价值大于直接抄任务。我未读其源码。

衍生项目 [lbnmahs/labs-for-wheels](https://github.com/lbnmahs/labs-for-wheels)（0 star，2026-02 推送）
把 WheeledLab 重构成了**独立的 external extension**——你的项目正是 external extension 结构，
它的目录组织可作对照。

### 7. TIERS/isaac-marl-mobile-manipulation —— 底盘/臂策略解耦

| | |
|---|---|
| 链接 | https://github.com/TIERS/isaac-marl-mobile-manipulation |
| star | **107**（最后推送 2024-07-03，**已停更约 2 年**） |
| 内容 | 多智能体 RL 做移动操作，研究"把底盘和机械臂的控制策略分开"是否比单一策略更好 |

研究问题和你直接相关（你现在是**单一策略同时输出底盘 3 维 + 臂 5 维 + 夹爪 1 维**，共 9 维）。
如果训练发现底盘和臂互相干扰、学不动，这个仓库的解耦思路值得参考。

⚠️ 2024-07 后停更，很可能基于 **Isaac Sim / Orbit 旧 API**，对不上你的 IsaacLab v2.3.2，别指望能直接跑。当思路参考。

### 8. pearl-robot-lab/rlmmbp —— 学"该在哪抓"

| | |
|---|---|
| 链接 | https://github.com/pearl-robot-lab/rlmmbp |
| 内容 | 用 RL 学移动操作行为：agent 学习**把机器人底盘停在哪个位置**、以及何时激活机械臂 |

⚠️ **更正**：我上一版给的 owner `YinpeiDai/rlmmbp` 是错的。正确 owner 是 `pearl-robot-lab`
（`iROSA-lab/rlmmbp` 会 301 重定向到它，两个写法都能到达）。

"学习底盘该停在哪个位姿再伸手"这个思路对你的 ①→② 阶段过渡有直接启发——
你现在是 dense 奖励让底盘一路贴近目标，但抓取真正需要的是**底盘停在一个臂可达的好位姿**，
而不是"离目标越近越好"（贴太近反而可能进入臂的奇异区或自碰撞）。

### 9. qaz9517532846/zm_robot —— 四麦轮 AGV

| | |
|---|---|
| 链接 | https://github.com/qaz9517532846/zm_robot |
| star | **137**（最后推送 2026-05-19） |
| 内容 | 四麦克纳姆轮 AGV，在 Isaac Sim 下运行，带 2 个 2D 激光雷达、RGB-D 相机、IMU |

主要价值在**麦轮 URDF/USD 建模**（对应你 TODO 4 的滚子几何）。ROS 导航栈项目，不是 RL 项目，
也不是 IsaacLab 扩展。

### 10. leggedrobotics/rsl_rl —— 你正在用的训练库

| | |
|---|---|
| 链接 | https://github.com/leggedrobotics/rsl_rl |
| star | **2919**（最后推送 2026-08-28，**今天还在更新**） |
| 内容 | GPU 加速的 PPO 等算法实现 |

读 `modules/actor_critic.py` 和 `algorithms/ppo.py` 是理解你训练配置里每个超参到底在干什么的最短路径。

### 11. NathanWu7/isaacLab.manipulation —— external extension 模板

| | |
|---|---|
| 链接 | https://github.com/NathanWu7/isaacLab.manipulation |
| star | **314**（最后推送 2025-06-29） |
| 内容 | 独立于 IsaacLab 的操作任务扩展（机械臂 + 灵巧手） |

README 自述基于 **orbit 旧版**，注意 API 代差。价值在项目结构组织。

---

## 三、我没能验证的部分

- **WebFetch 被网络策略拦截**（`Unable to verify if domain github.com is safe to fetch`），
  所以第三方仓库我只能通过 GitHub API 元数据 + `raw.githubusercontent.com` 拉单个文件来核对，
  没有系统读过它们的完整实现。标了"已读源码"的才是我真正看过的。
- **上一版清单里这些条目我这次没能验证，已全部移除**：`debi-ml/Summit_ws`、`Ericonaldo/visual_wholebody`、
  `yubohann/RoboCupVisionRL_IsaacLab_ROS2`、`ByteDance-Seed/manip-as-in-sim-suite`、
  `THU-VCLab/Part-Guided-3D-RL...`、`allenai/MolmoBot`、`leggedrobotics/rsl_rl_rwm`、
  `ethz-asl/moma`、`UT-Austin-RobIn/telemoma`、`RobotecAI/agentic-mobile-manipulator`，
  以及两个 arXiv 编号。其中部分可能真实存在，但既然我上一版是凭印象编的，就不该留在清单里充数。
- **star 数和推送时间是 2026-08-28 的快照**，会变。

---

## 四、给你的行动建议

按投入产出排序：

1. **读官方 lift 的 `rewards.py` + `lift_env_cfg.py`**（本地，10 分钟）。
   重点看权重比例 1:15:16，然后决定要不要调你的 2:5:2。

2. **按 ogre-lab 的方式实测轮子顺序和方向**（半小时）。
   写个最小脚本，不加任何符号修正直接发 `[+6,+6,+6,+6]` 和 `[+6,-6,+6,-6]`，肉眼看 GUI。
   顺便验证 `preserve_order=True` 下 `find_joints` 返回的顺序确实是 FL/FR/RL/RR。
   这一步能排掉一整类"训练不报错但学不会"的隐患。

3. **重新审视 `grasp_bonus` 的判定条件**。
   现在的 `(近) & (夹爪角度小)` 可被策略白拿。参考官方 `object_is_lifted` 的思路，
   换成能反映物理结果的信号。

4. 训通纯状态策略之后，再考虑 SurfaceGripper（TODO 3）和滚子几何（TODO 4）。

---

## 引用来源

搜索结果来源：
[TIERS/isaac-marl-mobile-manipulation](https://github.com/TIERS/isaac-marl-mobile-manipulation) ·
[UWRobotLearning/WheeledLab](https://github.com/UWRobotLearning/WheeledLab) ·
[pearl-robot-lab/rlmmbp](https://github.com/pearl-robot-lab/rlmmbp) ·
[lbnmahs/labs-for-wheels](https://github.com/lbnmahs/labs-for-wheels) ·
[protomota/ogre-lab](https://github.com/protomota/ogre-lab) ·
[qaz9517532846/zm_robot](https://github.com/qaz9517532846/zm_robot) ·
[NathanWu7/isaacLab.manipulation](https://github.com/NathanWu7/isaacLab.manipulation) ·
[isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) ·
[leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) ·
[Isaac Lab: Surface Gripper 教程](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/run_surface_gripper.html) ·
[Isaac Lab: Task Design Workflows](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html) ·
[Isaac Sim: Surface Gripper Extension](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_simulation/ext_isaacsim_robot_surface_gripper.html) ·
[IsaacSim Discussion #542](https://github.com/isaac-sim/IsaacSim/discussions/542) ·
[NVIDIA 论坛: SurfaceGripper D6Joint 问题](https://forums.developer.nvidia.com/t/title-isaac-sim-5-1-0-surface-gripper-not-working-on-custom-urdf-robot-d6joint-errors-physics-explosion/363946)
