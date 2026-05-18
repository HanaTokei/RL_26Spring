from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_are

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_rl.balance_car_env import BalanceCarEnv, BalanceCarParams  # noqa: E402


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Continuous-time LQR gain K for x_dot = A x + B u, u = -K x
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-noise", type=float, default=0.0)
    p.add_argument("--u-max", type=float, default=200.0)
    p.add_argument("--max-time", type=float, default=10.0)
    p.add_argument("--action-delay-steps", type=int, default=0)
    p.add_argument("--obs-delay-steps", type=int, default=0)
    p.add_argument("--actuator-alpha", type=float, default=1.0)
    p.add_argument("--action-drop-prob", type=float, default=0.0)
    p.add_argument("--dt-jitter-ratio", type=float, default=0.0)
    p.add_argument("--use-speed-pi", action="store_true")
    p.add_argument("--speed-kp", type=float, default=2.0)
    p.add_argument("--speed-ki", type=float, default=0.2)
    p.add_argument("--speed-i-max", type=float, default=200.0)
    p.add_argument("--speed-to-waccel", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    params = BalanceCarParams(
        obs_noise_std=float(args.obs_noise),
        u_max=float(args.u_max),
        max_time_s=float(args.max_time),
        action_delay_steps=int(args.action_delay_steps),
        obs_delay_steps=int(args.obs_delay_steps),
        actuator_alpha=float(args.actuator_alpha),
        action_drop_prob=float(args.action_drop_prob),
        dt_jitter_ratio=float(args.dt_jitter_ratio),
        use_speed_pi=bool(args.use_speed_pi),
        speed_kp=float(args.speed_kp),
        speed_ki=float(args.speed_ki),
        speed_i_max=float(args.speed_i_max),
        speed_to_waccel=float(args.speed_to_waccel),
    )
    env = BalanceCarEnv(params=params, seed=args.seed)

    # LQR weights: emphasize keeping theta_1/theta_2 small, lightly penalize wheel rates and actions.
    Q = np.diag([0.1, 0.1, 0.5, 0.5, 20.0, 2.0, 10.0, 1.0]).astype(np.float64)
    R = np.diag([0.5, 0.5]).astype(np.float64)
    K = lqr_gain(env.A, env.B, Q, R)

    lengths = []
    returns = []
    failures = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_ret = 0.0
        steps = 0
        terminated = False
        truncated = False
        while True:
            x = obs.astype(np.float64)
            u = -K @ x
            u = np.clip(u, -params.u_max, params.u_max).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(u)
            ep_ret += float(reward)
            steps += 1
            if terminated or truncated:
                break
        lengths.append(steps)
        returns.append(ep_ret)
        if terminated:
            failures += 1

    print("baseline", "LQR")
    print("episodes", args.episodes)
    print("avg_len", float(np.mean(lengths)), "std_len", float(np.std(lengths)))
    print("avg_return", float(np.mean(returns)), "std_return", float(np.std(returns)))
    print("terminated_failures", failures)


if __name__ == "__main__":
    main()
