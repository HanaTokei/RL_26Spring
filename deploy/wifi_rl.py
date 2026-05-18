from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.lqr_controller import LQRController  # noqa: E402
from deploy.protocol import WheeltecPacket, decode_state_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="192.168.4.1")
    p.add_argument("--port", type=int, default=6390)

    p.add_argument("--algo", choices=["sac", "ppo", "td3"], default="sac")
    p.add_argument("--model", type=str, required=True, help="Path to SB3 model.zip")

    # Safety / scaling
    p.add_argument("--u-max", type=float, default=20000.0, help="Clip final u_L/u_R")
    p.add_argument("--fallback-theta1", type=float, default=0.6, help="If |theta_1| exceeds this, fallback to LQR")
    p.add_argument("--mix-beta", type=float, default=0.5, help="When falling back, u = beta*u_lqr + (1-beta)*u_rl")

    p.add_argument("--print-every", type=int, default=20)
    return p.parse_args()


def load_model(algo: str, model_path: str):
    mp = Path(model_path)
    if algo == "sac":
        return SAC.load(str(mp))
    if algo == "td3":
        return TD3.load(str(mp))
    return PPO.load(str(mp))


def main() -> None:
    args = parse_args()
    model = load_model(args.algo, args.model)
    lqr = LQRController()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.host, args.port))

    buffer = ""
    step = 0
    try:
        while True:
            chunk = s.recv(128)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="ignore")

            values = decode_state_frame(buffer)
            if values is None:
                if len(buffer) > 4096:
                    buffer = buffer[-2048:]
                continue

            end = buffer.find("}")
            buffer = buffer[end + 1 :]

            theta_1, theta_dot_1, theta_2, theta_dot_2, theta_L, theta_R, theta_L_dot, theta_R_dot = values[:8]

            # Build observation in the same order as our sim env:
            # [theta_L, theta_R, theta_L_dot, theta_R_dot, theta_1, theta_dot_1, theta_2, theta_dot_2]
            obs = np.array(
                [theta_L, theta_R, theta_L_dot, theta_R_dot, theta_1, theta_dot_1, theta_2, theta_dot_2],
                dtype=np.float32,
            )

            # RL action: assumes policy outputs (u_L, u_R) directly (same semantics as LQR output)
            u_rl, _ = model.predict(obs, deterministic=True)
            u_rl = np.asarray(u_rl, dtype=np.float32).reshape(2)
            u_l_rl, u_r_rl = float(u_rl[0]), float(u_rl[1])

            # LQR baseline (for fallback/mixing)
            u_l_lqr, u_r_lqr = lqr.compute(
                theta_1=theta_1,
                theta_dot_1=theta_dot_1,
                theta_2=theta_2,
                theta_dot_2=theta_dot_2,
                theta_L=theta_L,
                theta_R=theta_R,
                theta_L_dot=theta_L_dot,
                theta_R_dot=theta_R_dot,
            )

            if abs(theta_1) > args.fallback_theta1:
                beta = float(np.clip(args.mix_beta, 0.0, 1.0))
                u_l = beta * u_l_lqr + (1.0 - beta) * u_l_rl
                u_r = beta * u_r_lqr + (1.0 - beta) * u_r_rl
                mode = "MIX"
            else:
                u_l, u_r = u_l_rl, u_r_rl
                mode = "RL"

            # Final safety clip
            u_l = max(-args.u_max, min(args.u_max, u_l))
            u_r = max(-args.u_max, min(args.u_max, u_r))

            s.send(WheeltecPacket(u_l=u_l, u_r=u_r).encode())

            if step % args.print_every == 0:
                print(
                    f"step={step} mode={mode} theta1={theta_1:+.3f} theta2={theta_2:+.3f} "
                    f"u=({u_l:+.1f},{u_r:+.1f}) rl=({u_l_rl:+.1f},{u_r_rl:+.1f})"
                )
            step += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        s.close()


if __name__ == "__main__":
    main()
