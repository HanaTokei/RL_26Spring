from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_rl.balance_car_env import BalanceCarParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="RL_26Spring/runs/curriculum_sac")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4)

    p.add_argument("--stage-steps", type=int, default=300_000)
    p.add_argument("--stages", type=int, default=4)

    p.add_argument("--u-max", type=float, default=200.0)
    p.add_argument("--max-time", type=float, default=10.0)
    return p.parse_args()


def stage_params(stage: int, base: BalanceCarParams) -> BalanceCarParams:
    """
    Curriculum from easy -> wifi-like.
    Stage 0: no noise, no delay
    Stage 1: small obs noise
    Stage 2: + action delay + actuator lag
    Stage 3: + drop + dt jitter + speed PI
    """
    p = BalanceCarParams(**asdict(base))
    if stage == 0:
        return p
    if stage == 1:
        p.obs_noise_std = 0.005
        return p
    if stage == 2:
        p.obs_noise_std = 0.01
        p.action_delay_steps = 1
        p.actuator_alpha = 0.3
        return p
    # stage >=3
    p.obs_noise_std = 0.01
    p.action_delay_steps = 1
    p.actuator_alpha = 0.3
    p.action_drop_prob = 0.05
    p.dt_jitter_ratio = 0.2
    p.use_speed_pi = True
    return p


def main() -> None:
    args = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    base = BalanceCarParams(u_max=float(args.u_max), max_time_s=float(args.max_time))

    # We call the existing training script as a library would be nicer, but keep it simple:
    # Train sequentially using SB3's ability to continue learning with new env.
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def make_env(params: BalanceCarParams, seed: int):
        from sim_rl.balance_car_env import BalanceCarEnv

        def _thunk():
            env = BalanceCarEnv(params=params, seed=seed)
            return Monitor(env)

        return _thunk

    model: SAC | None = None
    for stage in range(min(args.stages, 4)):
        p = stage_params(stage, base)
        stage_dir = logdir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "env_params.json").write_text(json.dumps(asdict(p), indent=2, ensure_ascii=False), encoding="utf-8")

        vec_env = make_vec_env(
            make_env(p, args.seed + stage * 1000),
            n_envs=args.n_envs,
            seed=args.seed + stage * 1000,
            vec_env_cls=DummyVecEnv,
        )
        vec_env = VecMonitor(vec_env)

        tb_log = str(stage_dir / "tb")
        if model is None:
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
            )
        else:
            model.set_env(vec_env)
            model.tensorboard_log = tb_log

        model.learn(total_timesteps=args.stage_steps, reset_num_timesteps=False)
        model.save(str(stage_dir / "model.zip"))
        vec_env.close()

    if model is not None:
        model.save(str(logdir / "model_final.zip"))


if __name__ == "__main__":
    main()
