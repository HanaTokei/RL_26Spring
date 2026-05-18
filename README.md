# RL_26Spring（二阶平衡车：仿真强化学习 + 真机部署脚手架）

本目录是一个**不依赖 MATLAB** 的“二阶平衡车仿真强化学习”小项目，同时提前把**真机部署脚本**准备好，便于后续直接在实车上验证与对比。

本项目用于完成课程作业的两类任务：
1. **任务一：平衡站立** — 训练 RL 策略使平衡车在无外力辅助下保持平衡
2. **任务二：移动任务** — 在保持平衡的同时跟踪目标轮速（匀速直线或曲线运动）

## 目录结构

- `sim_rl/`：仿真环境 + 训练/评估/基线脚本
- `deploy/`：真机部署脚本（WiFi LQR / WiFi RL）
- `runs/`：训练产物（默认不建议进 git）

---

## 环境与依赖

```bash
python -m pip install -r RL_26Spring/requirements.txt
```

依赖：gymnasium、stable-baselines3、torch、tensorboard

---

## 仿真环境定义

环境代码：`sim_rl/balance_car_env.py`

### 状态（观测）空间

**14 维观测向量：**

| 索引 | 符号 | 含义 | 单位 |
|------|------|------|------|
| 0 | theta_L | 左轮角度 | rad |
| 1 | theta_R | 右轮角度 | rad |
| 2 | theta_L_dot | 左轮角速度 | rad/s |
| 3 | theta_R_dot | 右轮角速度 | rad/s |
| 4 | theta_1 | 车身倾角 | rad |
| 5 | theta1_dot | 车身角速度 | rad/s |
| 6 | theta_2 | 摆杆倾角 | rad |
| 7 | theta2_dot | 摆杆角速度 | rad/s |
| 8 | alpha1 | 车身角加速度（有限差分） | rad/s² |
| 9 | alpha2 | 摆杆角加速度（有限差分） | rad/s² |
| 10 | mean_theta1_dot | 过去10步 theta1_dot 均值 | rad/s |
| 11 | mean_theta2_dot | 过去10步 theta2_dot 均值 | rad/s |
| 12 | std_theta1_dot | 过去10步 theta1_dot 标准差 | rad/s |
| 13 | std_theta2_dot | 过去10步 theta2_dot 标准差 | rad/s |

原始 8 维 + 新增 6 维（角加速度 + 时序统计），帮助策略感知状态变化趋势。

### 动作空间

`[u_L, u_R]`：左右轮控制输入（环境内部乘以 200 映射到 [-200, 200]）。

### 终止条件

- `|theta_1| > 45°`（车身倾倒）
- `|theta_2| > 60°`（摆杆倾倒）
- episode 时间超过 `--max-time`

### 奖励函数

```
r = alive_bonus(20)
  - w_theta1 * theta1² - w_theta1dot * theta1dot²
  - w_theta2 * theta2² - w_theta2dot * theta2dot²
  - w_wheel_dot * (wdot_l² + wdot_r²)
  - w_action * (u[0]² + u[1]²)
  + w_theta1_closeness * 1/(1 + 10*theta1²)   # 靠近直立加分
  + w_theta2_closeness * 1/(1 + 10*theta2²)
  + 0.1 * min(t / max_time_s, 1.0)             # 时间越长加分
  - terminate_penalty（倒地时）
```

---

## 关键突破：VecNormalize

SB3 神经网络输出 [-1, 1]，但环境动作空间是 [-200, 200]。同时观测空间各维度数值范围差异巨大（轮角度 ~0.01，关节角度 ~3.14，角速度 ~10）。

VecNormalize 对观测和奖励做在线归一化：
- 维护 running mean/std，将所有观测标准化到同一量级
- 防止神经网络被大数值特征绑架，忽略小数值特征
- 配合 `clip_reward=10.0` 防止奖励尖峰破坏训练

训练后保存 `vecnormalize.pkl`，评估时加载以保证统计量一致。

---

## 训练与评估

### 训练 SAC 模型（任务一：平衡站立）

```bash
python RL_26Spring/sim_rl/train_sb3.py \
  --algo sac \
  --steps 500000 \
  --n-envs 4 \
  --logdir /root/RL26Spring/runs/sac_v6
```

### 评估模型

```bash
python RL_26Spring/sim_rl/eval_sb3.py \
  --algo sac \
  --model /root/RL26Spring/runs/sac_v6/model.zip \
  --vecnormalize-path /root/RL26Spring/runs/sac_v6/vecnormalize.pkl \
  --episodes 10
```

### 训练移动任务模型（任务二）

```bash
python RL_26Spring/sim_rl/train_sb3.py \
  --algo sac \
  --steps 500000 \
  --n-envs 4 \
  --logdir /root/RL26Spring/runs/sac_moving \
  --v-target 1.0 \
  --max-time 30
```

随机速度训练（鲁棒性）：

```bash
python RL_26Spring/sim_rl/train_sb3.py \
  --algo sac \
  --steps 500000 \
  --n-envs 4 \
  --logdir /root/RL26Spring/runs/sac_moving_random \
  --v-target-random \
  --v-target-range 0.5 2.0 \
  --max-time 30
```

---

## 训练结果

| 模型 | ep_len_mean | 评估表现 |
|------|-------------|---------|
| SAC 初始版 | ~50步 | 1秒即翻 |
| SAC v4（+VecNormalize） | 637步 | 10秒稳定 |
| **SAC v6（+Reward Shaping）** | **981步** | **30秒稳定站立** |

SAC v6 评估结果（5 episodes，max_time=30s）：
```
avg_len: 3001, std_len: 0.0
→ 所有回合均跑满30秒无翻车
```

当前最佳模型：`/root/RL26Spring/runs/sac_v6/model.zip` + `vecnormalize.pkl`

---

## 真机部署

### 上车前检查清单

1. 确认车端固件已烧录（能发状态、能收 u_L/u_R）
2. PC 连上小车 AP（`192.168.4.1`）
3. **先运行 PC 脚本，再按复位键**
4. 首次测试在安全场地、有人扶车，先用 LQR 验证链路

### 1) LQR 基线部署（验证通信链路）

```bash
python RL_26Spring/deploy/wifi_lqr.py --host 192.168.4.1 --port 6390 --u-max 20000
```

### 2) RL 策略部署

```bash
python RL_26Spring/deploy/wifi_rl.py \
  --algo sac \
  --model /root/RL26Spring/runs/sac_v6/model.zip \
  --vecnormalize-path /root/RL26Spring/runs/sac_v6/vecnormalize.pkl \
  --u-max 5000
```

内置安全策略：
- `|theta_1| > fallback_theta1` 时自动与 LQR 混合回退
- 输出限幅 `--u-max`

---

## 待做

- [ ] 完成移动任务（任务二）训练与评估
- [ ] 曲线运动扩展（v_target 随时间变化）
- [ ] 真机验证