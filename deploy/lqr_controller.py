from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LQRController:
    """
    LQR gain values copied from the provided reference wifi3.0.py / control.c.
    This is the "real-robot baseline" controller used for deployment and fallback.
    """

    # Left output coefficients
    K11: float = 81.2695
    K12: float = -10.0616
    K13: float = -5492.4061
    K14: float = 18921.7098
    K15: float = 100.3633
    K16: float = 8.0376
    K17: float = 447.3084
    K18: float = 2962.7738

    # Right output coefficients
    K21: float = -10.0616
    K22: float = 81.2695
    K23: float = -5492.4061
    K24: float = 18921.7098
    K25: float = 8.0376
    K26: float = 100.3633
    K27: float = 447.3084
    K28: float = 2962.7738

    def compute(
        self,
        theta_1: float,
        theta_dot_1: float,
        theta_2: float,
        theta_dot_2: float,
        theta_L: float,
        theta_R: float,
        theta_L_dot: float,
        theta_R_dot: float,
        target_theta_L: float = 0.0,
        target_theta_R: float = 0.0,
        target_theta_L_dot: float = 0.0,
        target_theta_R_dot: float = 0.0,
        target_theta_1: float = 0.0,
    ) -> Tuple[float, float]:
        u_L = -(
            self.K11 * (theta_L - target_theta_L)
            + self.K12 * (theta_R - target_theta_R)
            + self.K13 * (theta_1 - target_theta_1)
            + self.K14 * theta_2
            + self.K15 * (theta_L_dot - target_theta_L_dot)
            + self.K16 * (theta_R_dot - target_theta_R_dot)
            + self.K17 * theta_dot_1
            + self.K18 * theta_dot_2
        )
        u_R = -(
            self.K21 * (theta_L - target_theta_L)
            + self.K22 * (theta_R - target_theta_R)
            + self.K23 * (theta_1 - target_theta_1)
            + self.K24 * theta_2
            + self.K25 * (theta_L_dot - target_theta_L_dot)
            + self.K26 * (theta_R_dot - target_theta_R_dot)
            + self.K27 * theta_dot_1
            + self.K28 * theta_dot_2
        )
        return float(u_L), float(u_R)

