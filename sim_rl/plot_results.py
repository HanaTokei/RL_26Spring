from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, required=True, help="Directory containing CSV logs produced by summarize_results.py")
    p.add_argument("--out", type=str, default="", help="Output png path (default: <logdir>/summary.png)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logdir = Path(args.logdir)
    csv_path = logdir / "summary.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run summarize_results.py first.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("summary.csv is empty")

    out = Path(args.out) if args.out else (logdir / "summary.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    axes[0].bar(df["name"], df["avg_len"], yerr=df["std_len"])
    axes[0].set_title("Episode length")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(df["name"], df["avg_return"], yerr=df["std_return"])
    axes[1].set_title("Episode return")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(df["name"], df["fail_rate"])
    axes[2].set_title("Failure rate (terminated)")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print("saved", str(out))


if __name__ == "__main__":
    main()

