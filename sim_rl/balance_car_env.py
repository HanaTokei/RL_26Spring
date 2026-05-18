from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from sim_rl.matlab_linear_model import (
    build_continuous_ab_from_course_params,
    permutation_to_env_order,
)


@dataclass
class BalanceCarParams:
    """
    Lightweight 2nd-order balance car simulator.

    State (8,):
      [theta_L, theta_R, theta_L_dot, theta_R_dot, theta_1, theta_dot_1, theta_2, theta_dot_2]

    Action (2,):
      [u_L, u_R]  (treated as "wheel input"; units are abstracted)

    Dynamics:
      x_dot = A x + B u + nonlinear_gravity_correction(x)
      Integrated with Euler + substeps.

    Notes:
    - This is intentionally a simplified model (good for coursework and algorithm comparison).
    - For better sim-to-real, add domain randomization (params/noise/delay).
    """

    dt: float = 0.01
    substeps: int = 5

    # Termination thresholds (rad)
    theta1_fail: float = float(np.pi / 4)  # ~45 deg
    theta2_fail: float = float(np.pi / 3)  # ~60 deg
    max_time_s: float = 10.0

    # Action limits
    u_max: float = 200.0

    # Reward weights
    alive_bonus: float = 20.0
    w_theta1: float = 5.0
    w_theta1dot: float = 0.2
    w_theta2: float = 2.0
    w_theta2dot: float = 0.1
    w_wheel_dot: float = 0.01
    w_action: float = 0.001
    terminate_penalty: float = 100.0
    w_theta1_closeness: float = 10.0
    w_theta2_closeness: float = 5.0

    # Initial state randomization (rad, rad/s)
    init_theta1_range: float = 0.15
    init_theta2_range: float = 0.15
    init_omega_range: float = 0.2

    # Observation noise
    obs_noise_std: float = 0.0

    # Model choice: "course_linear" uses the A/B derived from the provided MATLAB script.
    model: str = "course_linear"

    # Optional execution/sensing delay simulation (in steps)
    # - action_delay_steps: applied action is from k-action_delay_steps
    # - obs_delay_steps: observation returned is from k-obs_delay_steps
    action_delay_steps: int = 0
    obs_delay_steps: int = 0

    # Optional actuator lag (first-order): u_applied <- u_applied + alpha*(u_cmd-u_applied)
    # Set alpha=1 for no lag. Typical small lag: alpha in [0.1, 0.5]
    actuator_alpha: float = 1.0

    # WiFi/PC-in-loop artifacts
    # - action_drop_prob: with this probability, keep last commanded action (zero-order hold)
    # - dt_jitter_ratio: per-step dt jitter ratio, actual dt = dt * (1 + uniform(-r, +r))
    action_drop_prob: float = 0.0
    dt_jitter_ratio: float = 0.0

    # Optional speed loop approximation (mimic control.c: u -> target wheel speed -> PI -> actuator)
    use_speed_pi: bool = False
    # PI gains on wheel speed (not PWM; abstracted). Keep conservative.
    speed_kp: float = 2.0
    speed_ki: float = 0.2
    # Integrator clamp to avoid windup
    speed_i_max: float = 200.0
    # Map PI output to wheel acceleration (abstract gain)
    speed_to_waccel: float = 1.0

    # Moving task: target wheel velocity for task 2 (rad/s)
    # Set to 0.0 to disable moving task (pure balance)
    v_target: float = 0.0
    # Weight for velocity tracking reward
    w_velocity: float = 5.0
    # Whether to randomize v_target at reset (for robust training)
    v_target_random: bool = False
    v_target_range: tuple[float, float] = (0.5, 2.0)


def _build_ab(params: BalanceCarParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a plausible A,B for an 8-state, 2-input system.

    This is not an exact physical derivation; it's a minimal model that:
    - Couples wheel inputs to body and pole angular accelerations
    - Provides some damping terms
    - Keeps the system stabilizable near upright
    """
    # State ordering:
    # 0 theta_L, 1 theta_R, 2 theta_L_dot, 3 theta_R_dot, 4 theta_1, 5 theta_dot_1, 6 theta_2, 7 theta_dot_2
    A = np.zeros((8, 8), dtype=np.float64)
    B = np.zeros((8, 2), dtype=np.float64)

    # Kinematics
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    A[4, 5] = 1.0
    A[6, 7] = 1.0

    # Damping on wheel rates
    wheel_damp = 2.0
    A[2, 2] = -wheel_damp
    A[3, 3] = -wheel_damp

    # Body/pole damping
    body_damp = 1.5
    pole_damp = 1.0
    A[5, 5] = -body_damp
    A[7, 7] = -pole_damp

    # Coupling: average wheel acceleration affects body and pole
    # body_ddot ≈ c_body * (uL+uR)/2 + k terms from angles (gravity handled separately)
    c_body = 8.0
    c_pole = -6.0

    # Influence of wheel rates on body and pole (rough coupling)
    A[5, 2] = 0.5
    A[5, 3] = 0.5
    A[7, 2] = -0.3
    A[7, 3] = -0.3

    # Wheel inputs directly drive wheel accelerations
    B[2, 0] = 1.0
    B[3, 1] = 1.0

    # Wheel inputs influence body/pole angular accelerations via average input
    B[5, 0] = c_body * 0.5
    B[5, 1] = c_body * 0.5
    B[7, 0] = c_pole * 0.5
    B[7, 1] = c_pole * 0.5

    return A, B


class BalanceCarEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, params: Optional[BalanceCarParams] = None, seed: Optional[int] = None):
        super().__init__()
        self.params = params or BalanceCarParams()
        if self.params.model == "course_linear":
            A_mat, B_mat = build_continuous_ab_from_course_params()
            P, P_inv = permutation_to_env_order()
            self.A = P @ A_mat @ P_inv
            self.B = P @ B_mat
        else:
            self.A, self.B = _build_ab(self.params)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-self.params.u_max, high=self.params.u_max, shape=(2,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._x = np.zeros((8,), dtype=np.float64)
        self._u_applied = np.zeros((2,), dtype=np.float64)
        self._u_cmd_last = np.zeros((2,), dtype=np.float64)
        self._action_queue: list[np.ndarray] = []
        self._obs_queue: list[np.ndarray] = []
        self._speed_i = np.zeros((2,), dtype=np.float64)
        self._theta1dot_history: list[float] = []
        self._theta2dot_history: list[float] = []
        self._theta1dot_prev: float = 0.0
        self._theta2dot_prev: float = 0.0
        self._v_target_current: float = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        p = self.params
        self._t = 0.0
        self._x[:] = 0.0
        self._u_applied[:] = 0.0
        self._u_cmd_last[:] = 0.0
        self._action_queue = []
        self._obs_queue = []
        self._speed_i[:] = 0.0
        self._theta1dot_history.clear()
        self._theta2dot_history.clear()
        self._theta1dot_prev = 0.0
        self._theta2dot_prev = 0.0
        if p.v_target_random:
            self._v_target_current = self._rng.uniform(p.v_target_range[0], p.v_target_range[1])
        else:
            self._v_target_current = p.v_target

        # Randomize initial small tilt near upright
        self._x[4] = self._rng.uniform(-p.init_theta1_range, p.init_theta1_range)  # theta_1
        self._x[6] = self._rng.uniform(-p.init_theta2_range, p.init_theta2_range)  # theta_2
        self._x[5] = self._rng.uniform(-p.init_omega_range, p.init_omega_range)  # theta_dot_1
        self._x[7] = self._rng.uniform(-p.init_omega_range, p.init_omega_range)  # theta_dot_2

        obs = self._observe()
        # Initialize observation delay queue
        if p.obs_delay_steps > 0:
            self._obs_queue = [obs.copy() for _ in range(p.obs_delay_steps)]
        info: dict[str, Any] = {"t": self._t}
        return obs, info

    def _observe(self) -> np.ndarray:
        # Base 8-dim state
        obs = self._x.astype(np.float32, copy=True)

        # Manual normalization — fixed constants, no running stats
        # Dividing by approximate max magnitude of each feature group
        # so all dims are ~[-1, 1]
        obs[0] /= 10.0      # theta_L  ~±10 rad
        obs[1] /= 10.0      # theta_R  ~±10 rad
        obs[2] /= 10.0      # theta_L_dot ~±10 rad/s
        obs[3] /= 10.0      # theta_R_dot ~±10 rad/s
        obs[4] /= 0.8       # theta_1  ~±0.8 rad (45°)
        obs[5] /= 10.0      # theta_dot_1 ~±10 rad/s
        obs[6] /= 1.0       # theta_2  ~±1 rad (60°)
        obs[7] /= 10.0      # theta_dot_2 ~±10 rad/s

        if self.params.obs_noise_std > 0:
            obs += self._rng.normal(0.0, self.params.obs_noise_std, size=obs.shape).astype(np.float32)
        return obs

    def step(self, action: np.ndarray):
        p = self.params
        u_cmd = np.asarray(action, dtype=np.float64).reshape(2)
        u_cmd = np.clip(u_cmd, -p.u_max, p.u_max)

        # Random action drop (hold last command), to mimic packet loss / missed control update
        if p.action_drop_prob > 0.0:
            if self._rng.random() < float(p.action_drop_prob):
                u_cmd = self._u_cmd_last.copy()
            else:
                self._u_cmd_last = u_cmd.copy()
        else:
            self._u_cmd_last = u_cmd.copy()

        # Action delay
        if p.action_delay_steps > 0:
            self._action_queue.append(u_cmd.copy())
            if len(self._action_queue) <= p.action_delay_steps:
                u_cmd_effective = np.zeros_like(u_cmd)
            else:
                u_cmd_effective = self._action_queue.pop(0)
        else:
            u_cmd_effective = u_cmd

        # Actuator lag
        alpha = float(np.clip(p.actuator_alpha, 0.0, 1.0))
        self._u_applied = self._u_applied + alpha * (u_cmd_effective - self._u_applied)
        u = self._u_applied

        # Step dt jitter (simulate variable loop period)
        if p.dt_jitter_ratio > 0.0:
            jitter = float(self._rng.uniform(-p.dt_jitter_ratio, p.dt_jitter_ratio))
            dt_eff = p.dt * (1.0 + jitter)
        else:
            dt_eff = p.dt

        dt_sub = dt_eff / float(p.substeps)
        for _ in range(p.substeps):
            if p.use_speed_pi:
                # Mimic control.c structure:
                #   target_wdot = current_wdot + u * t
                #   PI(wdot_instant, target_wdot) -> command -> wheel accel
                wdot = self._x[2:4]
                target_wdot = wdot + u * dt_eff
                err = target_wdot - wdot
                self._speed_i += float(p.speed_ki) * err * dt_sub
                self._speed_i = np.clip(self._speed_i, -p.speed_i_max, p.speed_i_max)
                pi_out = float(p.speed_kp) * err + self._speed_i
                # Map PI output into wheel acceleration channels
                u_eff = np.clip(pi_out * float(p.speed_to_waccel), -p.u_max, p.u_max)
                xdot = self.A @ self._x + self.B @ u_eff
            else:
                xdot = self.A @ self._x + self.B @ u

            # Nonlinear gravity correction: replace linear approximation with sin(theta) behavior
            # This term "pushes" the system away from upright when angle increases.
            theta1 = float(self._x[4])
            theta2 = float(self._x[6])
            # sin(theta) - theta is ~0 near 0 but adds nonlinearity for larger angles
            xdot[5] += 25.0 * (np.sin(theta1) - theta1)
            xdot[7] += 15.0 * (np.sin(theta2) - theta2)

            self._x += xdot * dt_sub
            self._t += dt_sub

        terminated = bool((abs(self._x[4]) > p.theta1_fail) or (abs(self._x[6]) > p.theta2_fail))
        truncated = bool(self._t >= p.max_time_s)

        reward = self._compute_reward(u, terminated)
        obs_now = self._observe()
        if p.obs_delay_steps > 0:
            self._obs_queue.append(obs_now.copy())
            obs = self._obs_queue.pop(0)
        else:
            obs = obs_now

        info: dict[str, Any] = {
            "t": self._t,
            "terminated": terminated,
            "truncated": truncated,
            "theta_1": float(self._x[4]),
            "theta_2": float(self._x[6]),
            "u_cmd": u_cmd.astype(np.float32),
            "u_applied": u.astype(np.float32),
            "dt_eff": float(dt_eff),
            "v_target": float(self._v_target_current),
            "wheel_speed": float((self._x[2] + self._x[3]) / 2.0),
        }

        return obs, float(reward), terminated, truncated, info

    def _compute_reward(self, u: np.ndarray, terminated: bool) -> float:
        p = self.params
        theta1, theta1dot = float(self._x[4]), float(self._x[5])
        theta2, theta2dot = float(self._x[6]), float(self._x[7])
        wdot_l, wdot_r = float(self._x[2]), float(self._x[3])

        # Base survival bonus (larger to emphasize living)
        r = p.alive_bonus

        # Quadratic state costs
        r -= (
            p.w_theta1 * (theta1 * theta1)
            + p.w_theta1dot * (theta1dot * theta1dot)
            + p.w_theta2 * (theta2 * theta2)
            + p.w_theta2dot * (theta2dot * theta2dot)
            + p.w_wheel_dot * (wdot_l * wdot_l + wdot_r * wdot_r)
            + p.w_action * float(u[0] * u[0] + u[1] * u[1])
        )

        # Progress reward: reward for staying close to upright (closeness to 0)
        r += p.w_theta1_closeness * (1.0 / (1.0 + 10.0 * theta1 * theta1))
        r += p.w_theta2_closeness * (1.0 / (1.0 + 10.0 * theta2 * theta2))

        # Time bonus: small bonus proportional to time alive (encourages speed)
        r += 0.1 * min(self._t / p.max_time_s, 1.0)

        # Velocity tracking reward for task 2 (moving task)
        if self._v_target_current != 0.0:
            avg_wheel_speed = (wdot_l + wdot_r) / 2.0
            vel_error = avg_wheel_speed - self._v_target_current
            r -= p.w_velocity * (vel_error * vel_error)

        if terminated:
            r -= p.terminate_penalty
        return r


if __name__ == "__main__":
    # Quick smoke test
    env = BalanceCarEnv()
    obs, info = env.reset()
    total = 0.0
    for _ in range(1000):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    print("smoke_total_reward", total, "steps", int(info["t"] / env.params.dt))
