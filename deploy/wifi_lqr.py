from __future__ import annotations

import argparse
import socket
import time
from typing import Optional

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.lqr_controller import LQRController  # noqa: E402
from deploy.protocol import WheeltecPacket, decode_state_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="192.168.4.1")
    p.add_argument("--port", type=int, default=6390)
    p.add_argument("--u-max", type=float, default=20000.0, help="Clip u_L/u_R to this range")
    p.add_argument("--print-every", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctrl = LQRController()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.host, args.port))

    buffer = ""
    step = 0
    try:
        while True:
            chunk = s.recv(128)
            if not chunk:
                break
            try:
                buffer += chunk.decode("utf-8", errors="ignore")
            except Exception:
                continue

            values = decode_state_frame(buffer)
            if values is None:
                # keep accumulating
                if len(buffer) > 4096:
                    buffer = buffer[-2048:]
                continue

            # Trim buffer up to end of frame to avoid re-parsing
            end = buffer.find("}")
            buffer = buffer[end + 1 :]

            theta_1, theta_dot_1, theta_2, theta_dot_2, theta_L, theta_R, theta_L_dot, theta_R_dot = values[:8]
            u_l, u_r = ctrl.compute(
                theta_1=theta_1,
                theta_dot_1=theta_dot_1,
                theta_2=theta_2,
                theta_dot_2=theta_dot_2,
                theta_L=theta_L,
                theta_R=theta_R,
                theta_L_dot=theta_L_dot,
                theta_R_dot=theta_R_dot,
            )

            u_l = max(-args.u_max, min(args.u_max, u_l))
            u_r = max(-args.u_max, min(args.u_max, u_r))
            pkt = WheeltecPacket(u_l=u_l, u_r=u_r).encode()
            s.send(pkt)

            if step % args.print_every == 0:
                print(
                    f"step={step} theta1={theta_1:+.3f} theta2={theta_2:+.3f} "
                    f"u=({u_l:+.1f},{u_r:+.1f})"
                )
            step += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        s.close()


if __name__ == "__main__":
    main()
