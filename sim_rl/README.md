# Python 仿真强化学习（不依赖 MATLAB）

本目录提供一个**完全仿真**的二阶平衡车 Gymnasium 环境，以及基于 Stable-Baselines3 的训练脚本。

设计目标：
- 在电脑上用仿真环境训练（SAC / PPO / TD3）
- 训练收敛后可把策略移植到 PC-WiFi 控制脚本（替换 `wifi3.0.py` 的 LQR 计算）

## 安装依赖

在项目根目录执行：

```powershell
python -m pip install gymnasium stable-baselines3
```

（本机已验证 `gymnasium`、`stable-baselines3` 可安装。）

## 快速训练（SAC）

```powershell
python sim_rl/train_sb3.py --algo sac --steps 200000 --logdir sim_rl/runs/sac
```

训练完成会输出：
- 模型：`sim_rl/runs/sac/model.zip`
- 训练曲线：TensorBoard 日志在 `sim_rl/runs/sac/tb/`

查看 TensorBoard：

```powershell
python -m tensorboard.main --logdir sim_rl/runs/sac/tb
```

## 评估

```powershell
python sim_rl/eval_sb3.py --model sim_rl/runs/sac/model.zip --episodes 20
```

## 环境说明

环境文件：`sim_rl/balance_car_env.py`

- 状态（8维，列向量）：
  `[theta_L, theta_R, theta_L_dot, theta_R_dot, theta_1, theta_dot_1, theta_2, theta_dot_2]`
- 动作（2维，连续）：
  `[u_L, u_R]`（左右轮“等效输入”，可理解为角加速度/力矩的归一化输入）
- 动力学模型：
  - 默认使用课程资料提供的线性化连续模型（由 `inverted_pendulum_on_self_balancing_robot.m` 推导的 A/B），并在 Python 中复现
  - 状态顺序已按环境定义做了置换对齐
- 终止：
  - `|theta_1| > theta1_fail` 或 `|theta_2| > theta2_fail`
  - 或 episode 时间超过上限
- 奖励：
  `alive_bonus - (角度/角速度/轮速/动作 的二次惩罚)`；终止给大负奖励

### 更贴近真车的选项

训练/评估脚本支持注入延迟与执行器滞后（对 PC-WiFi 控制很重要）：

- `--action-delay-steps N`：动作延迟 N 步（N=1 表示延迟一个控制周期）
- `--obs-delay-steps N`：观测延迟 N 步
- `--actuator-alpha a`：执行器一阶滞后系数（1.0=无滞后；0.2~0.5=更滞后）
- `--action-drop-prob p`：以概率 p 丢失本周期动作更新（保持上一周期动作）
- `--dt-jitter-ratio r`：控制周期抖动（每步 dt 乘以 `1+U[-r,+r]`）
- `--use-speed-pi`：启用“速度 PI 内环近似”（模仿固件 `u -> 目标轮速 -> PI -> 执行器` 结构）

示例（模拟 1 步动作延迟 + 观测噪声 + 轻微执行器滞后）：

```powershell
python sim_rl/train_sb3.py --algo sac --steps 300000 --logdir sim_rl/runs/sac_delay --action-delay-steps 1 --obs-noise 0.01 --actuator-alpha 0.3
```

更“像真机 WiFi”的示例（延迟 + 丢包 + 周期抖动 + 执行器滞后）：

```powershell
python sim_rl/train_sb3.py --algo sac --steps 800000 --logdir sim_rl/runs/sac_wifi_like --action-delay-steps 1 --action-drop-prob 0.05 --dt-jitter-ratio 0.2 --obs-noise 0.01 --actuator-alpha 0.3
```

启用速度 PI 内环近似（更接近 `control.c` 的控制结构）：

```powershell
python sim_rl/train_sb3.py --algo sac --steps 800000 --logdir sim_rl/runs/sac_pi --use-speed-pi --speed-kp 2.0 --speed-ki 0.2 --speed-i-max 200
```

### 为什么这种仿真合理

它属于“状态空间模型 + 数值积分”的经典控制仿真做法，适合课程大作业做方法对比与训练曲线分析。
Sim-to-real 需要进一步做域随机化（噪声/参数扰动/延迟），后续可在环境参数中加入。
