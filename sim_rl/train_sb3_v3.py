from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_rl.balance_car_env import BalanceCarEnv, BalanceCarParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["sac", "ppo", "td3"], default="sac")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--logdir", type=str, default="sim_rl/runs/sac_v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4)

    # Basic domain randomization knobs
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
    p.add_argument("--v-target", type=float, default=0.0, help="Target wheel velocity for moving task (0=disabled)")
    p.add_argument("--v-target-random", action="store_true", help="Randomize v_target at each episode")
    p.add_argument("--v-target-range", type=float, nargs=2, default=[0.5, 2.0])
    p.add_argument("--w-velocity", type=float, default=8.0, help="Weight for velocity tracking reward")
    return p.parse_args()


def make_env(params: BalanceCarParams, seed: int):
    def _thunk():
        env = BalanceCarEnv(params=params, seed=seed)
        env = Monitor(env)
        return env

    return _thunk


def main() -> None:
    args = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

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
        w_velocity=float(args.w_velocity),
    )

    # Save params for reproducibility
    (logdir / "env_params.json").write_text(
        __import__("json").dumps(asdict(params), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    vec_env = make_vec_env(
        make_env(params, args.seed),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=DummyVecEnv,
    )
    vec_env = VecMonitor(vec_env)

    tb_log = str(logdir / "tb")
    checkpoint = CheckpointCallback(
        save_freq=max(10_000 // max(1, args.n_envs), 1),
        save_path=str(logdir / "checkpoints"),
        name_prefix="model",
        save_replay_buffer=(args.algo in ["sac", "td3"]),
        save_vecnormalize=False,
    )

    if args.algo == "sac":
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tb_log,
            seed=args.seed,
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[32, 32]),
            ent_coef="auto",
            target_entropy=1.0,
        )
    elif args.algo == "td3":
        model = TD3(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tb_log,
            seed=args.seed,
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[32, 32]),
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tb_log,
            seed=args.seed,
            learning_rate=3e-4,
            gamma=0.99,
            n_steps=1024,
            batch_size=256,
            policy_kwargs=dict(net_arch=[32, 32]),
        )

    model.learn(total_timesteps=args.steps, callback=checkpoint)
    model.save(str(logdir / "model.zip"))

    vec_env.close()


if __name__ == "__main__":
    main()
