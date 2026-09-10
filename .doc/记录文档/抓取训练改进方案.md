# 抓取训练改进方案

## 问题诊断

1. **grasp reward一直是0** - policy从未学会夹取动作
2. **gripper_early一直是0** - 夹爪从不闭合
3. **夹爪不会夹** - 稀疏奖励太难触发，policy没有学习信号

## 已实施的改进

### 1. 放宽抓取条件（更容易触发）

```python
# mobile_grasp_env_cfg.py
GRASP_REACH_THRESHOLD = 0.08  # 从0.05放宽到0.08m，让EE更容易进入"接近"范围
GRASP_DWELL_TIME = 0.2  # 从0.6降到0.2s，降低停留时间要求（约6步）
```

### 2. 提前curriculum时间表（更早学习抓取）

```python
# 原来：
# - ee_reach: iter 500 (step 12000)
# - grasp: iter 1000 (step 24000)

# 现在：
# - ee_reach: iter 250 (step 6000)，权重 5.0
# - gripper_close_guide: iter 375 (step 9000)，权重 2.0  # 新增引导奖励
# - grasp: iter 500 (step 12000)，权重 10.0
```

### 3. 新增引导奖励 `gripper_close_when_near`

这是一个**稠密奖励**，在稀疏的`grasp_bonus_dwell`之前使用：

```python
# rewards.py
def gripper_close_when_near(...):
    """
    奖励"接近目标时闭合夹爪"
    
    工作原理：
    - 计算EE到目标的距离，用高斯门控
    - 在GRASP_REACH_THRESHOLD内，proximity快速上升到1.0
    - 夹爪闭合程度归一化到[0,1]
    - 返回 proximity * close_amount
    
    效果：
    - 远离目标：无论夹爪如何都是0
    - 接近但张开：proximity高但close_amount低，奖励小
    - 接近且闭合：proximity和close_amount都高，奖励接近1.0
    """
```

**为什么需要这个奖励？**
- `grasp_bonus_dwell`太稀疏：需要同时满足(接近 + 停留 + 闭合)
- policy在随机探索时几乎不可能偶然触发
- `gripper_close_guide`提供了一个"温暖"的学习信号，引导policy建立"靠近->闭合"的关联

### 4. 降低gripper_early惩罚

```python
gripper_early_sched: -0.01 -> -0.005  # 降低惩罚，让policy敢于尝试闭合
```

## 学习阶段设计

```
阶段1 (iter 0-250, step 0-6000):
  - base_approach: 学习底盘接近
  - base_facing: 学习底盘朝向
  
阶段2 (iter 250-375, step 6000-9000):
  + ee_reach (5.0): 学习末端执行器接近目标
  + ee_distance (-0.2): 惩罚距离过远
  
阶段2.5 (iter 375-500, step 9000-12000):
  + gripper_close_guide (2.0): 学习"接近时闭合"（新增）
  
阶段3 (iter 500+, step 12000+):
  + grasp (10.0): 稀疏抓取奖励（停留+闭合）
  + gripper_early (-0.005): 惩罚提前闭爪
  + retract (2.0): 抓取后回收
```

## 调优建议

### 如果训练iter < 500
- 当前还在早期阶段，grasp相关奖励还没激活
- **建议：训练到iter 500+再评估grasp效果**

### 如果到了iter 500+，grasp仍然是0

#### 方案A：继续放宽阈值
```python
GRASP_REACH_THRESHOLD = 0.10  # 再放宽
GRASP_DWELL_TIME = 0.1  # 降到3步
```

#### 方案B：提高引导奖励权重
```python
gripper_close_guide_sched: weight 2.0 -> 3.0
ee_reach_sched: weight 5.0 -> 7.0
```

#### 方案C：检查EE是否能到达
运行诊断脚本查看实际距离：
```bash
cd /home/pu/isaac-lab
./isaaclab.sh -p /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot/scripts/diagnose_grasp.py --headless
```

查看输出：
- 如果最小距离 > 0.08m，说明EE到不了 -> 增大`ee_reach`权重
- 如果最小距离 < 0.08m但夹爪从不闭合 -> 增大`gripper_close_guide`权重

### 让夹爪离目标更近的方法

1. **调整grasp offset** - 如果之前的calibration脚本显示GREEN球（自动计算）比RED球（当前offset）更合理：
   ```python
   # eggtart.py
   EGGTART_EE_GRASP_OFFSET = (-0.007327907, -0.017998196, -0.020484457)  # 使用自动计算的值
   ```

2. **增加ee_reach权重**
   ```python
   ee_reach_sched: (6000, 5.0) -> (6000, 8.0)
   ```

3. **降低ee_distance惩罚的std**（让接近曲线更陡）
   ```python
   ee_distance = RewTerm(
       func=mdp.ee_distance_penalty,
       weight=-0.2,
       params={
           "std": 0.1,  # 从默认值降低，让距离惩罚更敏感
           ...
       }
   )
   ```

## 监控指标

训练时关注TensorBoard中的这些曲线：

1. **Curriculum/** - 确认各阶段按时激活
   - `Curriculum/ee_reach` 应该在iter 250跳到5.0
   - `Curriculum/gripper_close_guide` 应该在iter 375跳到2.0
   - `Curriculum/grasp` 应该在iter 500跳到10.0

2. **Episode_Reward/** - 各项奖励累计
   - `Episode_Reward/ee_reach` 应该在阶段2开始上升
   - `Episode_Reward/gripper_close_guide` 应该在阶段2.5开始上升
   - `Episode_Reward/grasp` 应该在阶段3开始出现非零值

3. **Step_Reward/** - 单步奖励，更实时
   - `Step_Reward/gripper_close_guide` 看policy是否在接近时尝试闭合

## 预期训练效果

- **iter 0-250**: 学会底盘移动和朝向，Episode_Reward/base_approach上升
- **iter 250-375**: 学会伸出机械臂接近目标，Episode_Reward/ee_reach上升
- **iter 375-500**: 开始在接近时尝试闭合夹爪，Episode_Reward/gripper_close_guide > 0
- **iter 500+**: 学会完整抓取动作（接近+停留+闭合），Episode_Reward/grasp开始出现非零值

如果到iter 700还没有grasp，说明任务太难，需要进一步调整参数。

## 快速测试

重新训练并观察：
```bash
cd /home/pu/isaac-lab
./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Mobile-Grasp-Eggtart-v0 --num_envs 2048
```

在TensorBoard中重点关注：
1. iter 375时，`Step_Reward/gripper_close_guide`是否开始出现正值
2. iter 500时，`Episode_Reward/grasp`是否开始出现非零值

如果这两个都是0，回来我们再进一步调整。
