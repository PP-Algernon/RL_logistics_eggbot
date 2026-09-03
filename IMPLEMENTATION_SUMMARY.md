# 抓取任务修改实施总结

## 修改完成 ✓

已成功实现以下两项核心修改：

### 1. ✓ 目标物理特性调整
- **恢复重力**: `disable_gravity: True` → `False`
- **降低移动速度**: ±0.25 m/s → **±0.10 m/s** (60% 减速)
- **增加阻尼**: linear/angular damping = 0.5 (防止滑动)
- **调整初始高度**: z = 0.015-0.020 m (在地面，让其自然落地)

### 2. ✓ 基于提起高度的抓取判定
- **新奖励函数**: `GraspBonusLift` 类
- **判定标准**: 目标质心高度 ≥ 0.35m 并保持 0.3s
- **奖励权重**: 提高到 10.0 (原 5.0)
- **移除依赖**: 不再依赖夹爪角度闭合阈值

## 修改的文件

### 核心配置文件
1. **mobile_grasp_env_cfg.py** (主配置)
   - 添加新常量: `LIFT_HEIGHT_THRESHOLD`, `LIFT_DWELL_TIME`, `TARGET_VELOCITY_RANGE`
   - 修改目标物理属性 (恢复重力、降低速度)
   - 更新抓取奖励配置 (使用 `grasp_bonus_lift`)
   - 新增 `MobileGraspEnvStaticCfg` 类 (静止目标版本)

2. **mdp/rewards.py** (奖励函数)
   - 新增 `GraspBonusLift` 类 (基于高度判定)
   - 导出 `grasp_bonus_lift` 函数名

3. **config/eggtart/grasp_env_cfg.py** (具体机器人配置)
   - 添加 `EggtartMobileGraspEnvStaticCfg` 类

4. **config/eggtart/__init__.py** (任务注册)
   - 注册新任务: `Isaac-Mobile-Grasp-Eggtart-Static-v0`

## 两个训练版本

### 版本 A: 移动目标 (完整挑战)
**任务ID**: `Isaac-Mobile-Grasp-Eggtart-v0`

特点:
- 目标初始速度: ±0.10 m/s (随机方向)
- 每 2-4 秒改变运动方向
- 需要学习追踪和预测

```bash
./isaaclab.sh -p source/eggtart_grasp/scripts/rsl_rl_train.py \
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --num_envs 2048 \
    --max_iterations 2500
```

### 版本 B: 静止目标 (简化训练)
**任务ID**: `Isaac-Mobile-Grasp-Eggtart-Static-v0`

特点:
- 目标初始速度: 0 (完全静止)
- 速度随机化被禁用
- 降低任务难度，适合初期训练

```bash
./isaaclab.sh -p source/eggtart_grasp/scripts/rsl_rl_train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --num_envs 2048 \
    --max_iterations 1500
```

## 新增参数说明

### 高度判定参数
```python
LIFT_HEIGHT_THRESHOLD = 0.35  # m
# 理由：目标初始在地面(~0.02m)，需要提升约0.33m才算成功
# 足够高证明真正抓住，但不超出机械臂工作空间

LIFT_DWELL_TIME = 0.3  # s
# 理由：避免"瞬间碰撞"误判，要求稳定保持
# 约9个仿真步 (dt=1/30s)
```

### 速度参数
```python
TARGET_VELOCITY_RANGE = 0.10  # m/s
# 理由：原速度(0.25)对追踪难度过高
# 降低到40%让策略更容易学会基本抓取
```

## 课程学习时间表

| 阶段 | 迭代数 | 步数 | 启用项 |
|------|--------|------|--------|
| 1 | 0-375 | 0-9000 | 底盘接近 + 朝向 |
| 2 | 375-500 | 9000-12000 | + EE到达 + 夹爪引导 |
| 3 | 500+ | 12000+ | + **提起抓取** + 回收 |

注意：提起抓取奖励在 iter 500 (步数 12000) 启用

## 监控指标

启动 TensorBoard:
```bash
tensorboard --logdir logs/rsl_rl/
```

关键指标:
- **`Rewards/grasp`**: 提起奖励 (应逐渐增加，iter 500后启用)
- **`Rewards/ee_reach`**: EE到达奖励
- **`Rewards/gripper_close_guide`**: 夹爪引导奖励
- **`Curriculum/grasp_sched`**: 抓取奖励权重调度

## 预期行为变化

### 训练初期 (iter 0-500)
- 学习底盘接近和朝向目标
- 学习EE到达目标附近
- 开始学习闭合夹爪 (引导奖励)

### 训练中期 (iter 500-1500)
- 抓取奖励启用，学习"夹住→提起"序列
- 目标高度应逐渐增加
- `Rewards/grasp` 从 0 逐渐上升

### 训练后期 (iter 1500+)
- 稳定抓取并提起目标
- 学习回收动作 (机械臂回默认姿态)
- 移动目标版本：学习预测性抓取

## 故障排查

### 如果 `Rewards/grasp` 一直是 0:
1. 检查课程学习: `Curriculum/grasp_sched` 是否已启用 (需 iter > 500)
2. 检查目标高度: 添加自定义观测监控 `target.data.root_pos_w[:, 2]`
3. 降低阈值尝试: `LIFT_HEIGHT_THRESHOLD = 0.25` (在 mobile_grasp_env_cfg.py)
4. 缩短保持时间: `LIFT_DWELL_TIME = 0.2`

### 如果目标一直在地上不动:
1. 确认重力: 检查 `disable_gravity=False`
2. 检查速度设置: 移动版本应有 `velocity_range: ±0.10`
3. 检查事件触发: `randomize_target_velocity` 应每 2-4s 触发

### 如果夹爪夹不住:
1. 检查力控参数: `max_effort=1.0` 应该足够
2. 检查引导奖励: `gripper_close_guide` 应在 iter 500 启用
3. 可能需要调整 PD 参数或接触刚度

## 参数微调建议

如果训练效果不理想，可以调整以下参数:

### 降低难度
```python
# mobile_grasp_env_cfg.py
LIFT_HEIGHT_THRESHOLD = 0.25  # 降低高度要求
LIFT_DWELL_TIME = 0.2         # 减少保持时间
TARGET_VELOCITY_RANGE = 0.05  # 进一步降低速度
```

### 提高难度 (after success with static)
```python
LIFT_HEIGHT_THRESHOLD = 0.40  # 提高高度要求
LIFT_DWELL_TIME = 0.5         # 延长保持时间
TARGET_VELOCITY_RANGE = 0.15  # 提高移动速度
```

### 调整课程学习
```python
# CurriculumCfg
grasp_sched = CurrTerm(
    func=mdp.reward_weight_schedule,
    params={"term_name": "grasp", "schedule": [
        (0, 0.0),      # iter 0: 关闭
        (9000, 10.0)   # iter 375: 提前启用 (原 iter 500)
    ]},
)
```

## 渐进式训练策略

推荐从静止目标开始，逐步过渡到移动目标:

```bash
# 步骤 1: 静止目标 (约1000 iters)
./isaaclab.sh -p source/eggtart_grasp/scripts/rsl_rl_train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Static-v0 \
    --max_iterations 1000

# 步骤 2: 迁移到移动目标 (加载checkpoint)
./isaaclab.sh -p source/eggtart_grasp/scripts/rsl_rl_train.py \
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --resume \
    --load_run <静止版本的run_name> \
    --max_iterations 2000
```

## 文档资源

- **详细修改说明**: [CHANGES_GRASP_REWARD.md](CHANGES_GRASP_REWARD.md)
- **快速开始指南**: [QUICK_START_GRASP.md](QUICK_START_GRASP.md)
- **本总结**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## 下一步工作

可选的进一步改进:
1. **多阶段抓取奖励**: 0.2m (+5), 0.35m (+10), 保持1s (+5)
2. **接触力检测**: 结合接触传感器判断抓取质量
3. **搬运任务**: 提起后移动到指定位置放下
4. **自适应高度**: 根据目标大小动态调整阈值
5. **课程学习**: 从大目标(易抓)到小目标(难抓)

## 验证清单

- [x] 目标恢复重力 (`disable_gravity=False`)
- [x] 移动速度降低 60% (±0.10 m/s)
- [x] 新增静止目标配置 (`MobileGraspEnvStaticCfg`)
- [x] 实现基于高度的抓取判定 (`GraspBonusLift`)
- [x] 更新奖励配置使用新判定方法
- [x] 注册两个训练任务 (移动 + 静止)
- [x] 所有文件语法检查通过
- [x] 创建完整文档

## 代码质量检查

所有修改文件已通过语法检查:
```bash
✓ mobile_grasp_env_cfg.py
✓ mdp/rewards.py
✓ config/eggtart/grasp_env_cfg.py
✓ config/eggtart/__init__.py
✓ mdp/__init__.py
```

实施完成！可以开始训练了。
