from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_rl.balance_car_env import BalanceCarEnv, BalanceCarParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--algo", choices=["sac", "ppo", "td3"], default="sac")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vecnormalize-path", type=str, default=None)
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
    # Moving task (task 2) parameters
    p.add_argument("--v-target", type=float, default=0.0, help="Target wheel velocity for moving task")
    p.add_argument("--v-target-random", action="store_true", help="Randomize v_target at each episode")
    p.add_argument("--v-target-range", type=float, nargs=2, default=[0.5, 2.0])
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
        v_target=float(args.v_target),
        v_target_random=bool(args.v_target_random),
        v_target_range=tuple(args.v_target_range),
    )

    env = BalanceCarEnv(params=params, seed=args.seed)
    vec_env = DummyVecEnv([lambda: env])
    if args.vecnormalize_path:
        vec_env = VecNormalize.load(args.vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_obs = True
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    model_path = Path(args.model)
    if args.algo == "sac":
        model = SAC.load(str(model_path))
    elif args.algo == "td3":
        model = TD3.load(str(model_path))
    else:
        model = PPO.load(str(model_path))

    lengths = []
    returns = []
    failures = 0
    v_errors = []
    for ep in range(args.episodes):
        obs = vec_env.reset()
        ep_ret = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_ret += float(reward)
            steps += 1
            if done:
                failures += 1
                if len(info) > 0 and isinstance(info, list):
                    v_errors.append(abs(float(info[0].get("wheel_speed", 0)) - float(info[0].get("v_target", 0))))
        lengths.append(steps)
        returns.append(ep_ret)

    print("episodes", args.episodes)
    print("avg_len", float(np.mean(lengths)), "std_len", float(np.std(lengths)))
    print("avg_return", float(np.mean(returns)), "std_return", float(np.std(returns)))
    print("terminated_failures", failures)
    if v_errors:
        print("avg_vel_error", float(np.mean(v_errors)), "rad/s")

    vec_env.close()


if __name__ == "__main__":
    main()
