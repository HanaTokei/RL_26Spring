# RL_26Spring V3 — 平衡车强化学习 + STM32板载部署

二阶平衡车RL项目，支持平衡站立任务训练与STM32真机部署。

## 项目结构

```
sim_rl/
├── balance_car_env.py    # 仿真环境（8维观测，手动归一化）
├── train_sb3_v3.py       # V3训练脚本（32×32网络，500k步）
├── eval_sb3.py           # 评估脚本
├── rl_model_v3.h          # 导出的C模型（STM32用）
├── matlab_linear_model.py # MATLAB线性模型接口
└── runs/
    └── sac_v3/
        └── model.zip     # 训练好的模型

deploy/
├── wifi_lqr.py           # WiFi LQR控制
└── wifi_rl.py            # WiFi RL控制（实验用，实际部署用板载）
```

---

## 快速开始

### 训练

```bash
cd /root/blockdata/26SpringRL/RL_26Spring_gfm
python sim_rl/train_sb3_v3.py --algo sac --steps 500000 --n-envs 4
```

### 评估（30秒仿真时间）

```bash
python -c "
from stable_baselines3 import SAC
from sim_rl.balance_car_env import BalanceCarEnv, BalanceCarParams

env = BalanceCarEnv(params=BalanceCarParams(max_time_s=30.0), seed=42)
model = SAC.load('runs/sac_v3/model.zip')

obs, _ = env.reset(seed=42)
total_steps = 0
while total_steps < 30000:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, info = env.step(action)
    total_steps += 1
    if term or trunc:
        obs, _ = env.reset()
print(f'Done: {total_steps} steps')
"
```

### 导出到C（STM32）

```bash
python /root/blockdata/26SpringRL/RL_26Spring_lzw/src/deploy/export_to_c.py \
  --algo SAC \
  --model runs/sac_v3/model.zip \
  --quantize int8 \
  --output sim_rl/rl_model_v3.h
```

---

## 仿真环境说明

### 状态（8维）

| 索引 | 符号 | 含义 | 归一化除数 |
|------|------|------|------------|
| 0 | theta_L | 左轮角度 | /10 |
| 1 | theta_R | 右轮角度 | /10 |
| 2 | theta_L_dot | 左轮角速度 | /10 |
| 3 | theta_R_dot | 右轮角速度 | /10 |
| 4 | theta_1 | 车身倾角 | /0.8 (≈45°) |
| 5 | theta_dot_1 | 车身角速度 | /10 |
| 6 | theta_2 | 摆杆倾角 | /1.0 (≈60°) |
| 7 | theta_dot_2 | 摆杆角速度 | /10 |

归一化方法：固定常数（不是在线统计），确保仿真和STM32部署完全一致。

### 动作空间

`[u_L, u_R]`：范围 [-200, 200]，网络输出tanh后乘以200。

### 终止条件

- `|theta_1| > π/4`（约45°）
- `|theta_2| > π/3`（约60°）
- 仿真时间超过 `max_time_s`

---

## STM32部署步骤

### 1. 复制头文件

将 `sim_rl/rl_model_v3.h` 复制到Keil工程目录：

```
MiniBalance/
├── control.c
├── encoder.c
├── rl_model_v3.h   ← 复制到这里
└── ...
```

### 2. 修改control.c

在文件顶部添加：

```c
#include "rl_model_v3.h"
```

找到 `Control_mode == 1` 分支，替换为：

```c
float state[8] = {
    theta_L, theta_R,
    theta_L_dot, theta_R_dot,
    theta_1, theta_dot_1,
    theta_2, theta_dot_2
};
// 手动归一化（与仿真一致）
state[0] /= 10.0f;  state[1] /= 10.0f;
state[2] /= 10.0f;  state[3] /= 10.0f;
state[4] /= 0.8f;   state[5] /= 10.0f;
state[6] /= 1.0f;   state[7] /= 10.0f;

float action[2];
rl_predict(state, action);
u_L = action[0];
u_R = action[1];
```

### 3. 编译烧录

- 网络规模：1410参数，int8量化后1.4KB
- 推理时间：<1ms（STM32F103 @72MHz）
- 内存：约20KB Flash + 2KB RAM

---

## 模型规格

| 项目 | 值 |
|------|-----|
| 输入维度 | 8 |
| 网络结构 | 8 → 32 → 32 → 2 |
| 激活函数 | ReLU（隐层），Tanh（输出） |
| 参数量 | 1410 |
| 存储大小（int8） | 1.4 KB |
| 推理时间 | ~1ms |
| 最大动作 | ±200 |

---

## 训练结果（V3）

| 指标 | 值 |
|------|-----|
| 训练步数 | 500,000 |
| 网络 | 32×32 |
| 归一化 | 手动（固定常数） |
| 30秒Eval步数 | 30,000（全部存活） |
| 平均reward/步 | 34.7 |

---

## 对比旧版

| 版本 | 输入维度 | 归一化 | 网络 | Eval表现 |
|------|---------|--------|------|---------|
| V2 | 8 | 无 | 64×64 | ~36步存活 |
| **V3** | **8** | **手动固定** | **32×32** | **30000步存活** |

关键改进：手动归一化让训练和部署一致，避免VecNormalize的running mean/std在部署时不可用的问题。