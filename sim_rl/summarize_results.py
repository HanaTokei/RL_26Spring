from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, required=True, help="Where to write summary.csv (and optional plots)")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)

    # Environment knobs (must match what you want to evaluate)
    p.add_argument("--obs-noise", type=float, default=0.01)
    p.add_argument("--u-max", type=float, default=200.0)
    p.add_argument("--max-time", type=float, default=10.0)
    p.add_argument("--action-delay-steps", type=int, default=1)
    p.add_argument("--obs-delay-steps", type=int, default=0)
    p.add_argument("--actuator-alpha", type=float, default=0.3)
    p.add_argument("--action-drop-prob", type=float, default=0.05)
    p.add_argument("--dt-jitter-ratio", type=float, default=0.2)
    p.add_argument("--use-speed-pi", action="store_true")

    # Models to compare
    p.add_argument("--rl-model", type=str, default="", help="Path to SB3 model.zip")
    return p.parse_args()


def run_cmd(cmd: list[str]) -> dict:
    out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="ignore")
    result: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] in {"avg_len", "std_len", "avg_return", "std_return", "terminated_failures", "episodes"}:
            result[parts[0]] = parts[1]
        if parts[0] == "baseline":
            result["baseline"] = parts[1]
    return result


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_flags = [
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--obs-noise",
        str(args.obs_noise),
        "--u-max",
        str(args.u_max),
        "--max-time",
        str(args.max_time),
        "--action-delay-steps",
        str(args.action_delay_steps),
        "--obs-delay-steps",
        str(args.obs_delay_steps),
        "--actuator-alpha",
        str(args.actuator_alpha),
        "--action-drop-prob",
        str(args.action_drop_prob),
        "--dt-jitter-ratio",
        str(args.dt_jitter_ratio),
    ]
    if args.use_speed_pi:
        base_flags.append("--use-speed-pi")

    rows = []

    # LQR baseline
    lqr = run_cmd([sys.executable, str(Path(__file__).with_name("baseline_lqr.py")), *base_flags])
    rows.append(
        {
            "name": "LQR",
            "avg_len": float(lqr.get("avg_len", "nan")),
            "std_len": float(lqr.get("std_len", "nan")),
            "avg_return": float(lqr.get("avg_return", "nan")),
            "std_return": float(lqr.get("std_return", "nan")),
            "fail_rate": float(lqr.get("terminated_failures", "0")) / float(lqr.get("episodes", args.episodes)),
        }
    )

    if args.rl_model:
        rl = run_cmd(
            [
                sys.executable,
                str(Path(__file__).with_name("eval_sb3.py")),
                "--algo",
                "sac",
                "--model",
                args.rl_model,
                *base_flags,
            ]
        )
        rows.append(
            {
                "name": "SAC",
                "avg_len": float(rl.get("avg_len", "nan")),
                "std_len": float(rl.get("std_len", "nan")),
                "avg_return": float(rl.get("avg_return", "nan")),
                "std_return": float(rl.get("std_return", "nan")),
                "fail_rate": float(rl.get("terminated_failures", "0")) / float(args.episodes),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "summary.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", str(outdir / "summary.csv"))


if __name__ == "__main__":
    main()

