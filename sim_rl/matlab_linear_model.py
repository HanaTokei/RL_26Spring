from __future__ import annotations

import numpy as np


def build_continuous_ab_from_course_params() -> tuple[np.ndarray, np.ndarray]:
    """
    Re-implement the linearized continuous-time model from the course-provided MATLAB script:

      D:\\vscode_project\\26SpringRL\\_tmp_matlab_program\\MATLAB程序\\R2022a\\inverted_pendulum_on_self_balancing_robot.m

    The script constructs A,B via:
      temp = p^-1 * q
      A = [ 0 0 0 0 1 0 0 0
            0 0 0 0 0 1 0 0
            0 0 0 0 0 0 1 0
            0 0 0 0 0 0 0 1
            temp(:,1:8) ]
      B = [zeros(4,2); temp(:,9:10)]

    State ordering in that script corresponds to:
      x = [theta_L, theta_R, theta_1, theta_2, theta_L_dot, theta_R_dot, theta_dot_1, theta_dot_2]^T

    Returns:
      A (8x8), B (8x2)
    """
    # Parameters (copied from MATLAB)
    m_1 = 0.9
    m_2 = 0.1
    r = 0.0335
    L_1 = 0.126
    L_2 = 0.390
    l_1 = L_1 / 2
    l_2 = L_2 / 2
    g = 9.8
    I_1 = (1 / 12) * m_1 * L_1**2
    I_2 = (1 / 12) * m_2 * L_2**2

    # p matrix (4x4)
    p_11 = 1
    p_12 = 0
    p_13 = 0
    p_14 = 0
    p_21 = 0
    p_22 = 1
    p_23 = 0
    p_24 = 0
    p_31 = (r / 2) * (m_1 * l_1 + m_2 * L_1)
    p_32 = (r / 2) * (m_1 * l_1 + m_2 * L_1)
    p_33 = m_1 * l_1**2 + m_2 * L_1**2 + I_1
    p_34 = m_2 * L_1 * l_2
    p_41 = (r / 2) * m_2 * l_2
    p_42 = (r / 2) * m_2 * l_2
    p_43 = m_2 * L_1 * l_2
    p_44 = m_2 * l_2**2 + I_2

    p = np.array(
        [
            [p_11, p_12, p_13, p_14],
            [p_21, p_22, p_23, p_24],
            [p_31, p_32, p_33, p_34],
            [p_41, p_42, p_43, p_44],
        ],
        dtype=np.float64,
    )

    # q matrix (4x10)
    q = np.zeros((4, 10), dtype=np.float64)
    q[0, 8] = 1
    q[1, 9] = 1
    q[2, 2] = (m_1 * l_1 + m_2 * L_1) * g
    q[3, 3] = m_2 * g * l_2

    temp = np.linalg.solve(p, q)  # p^-1 * q

    A = np.zeros((8, 8), dtype=np.float64)
    # xdot angles = angular velocities
    A[0, 4] = 1.0
    A[1, 5] = 1.0
    A[2, 6] = 1.0
    A[3, 7] = 1.0
    # xdot velocities = temp * [x; u]
    A[4:8, 0:8] = temp[:, 0:8]

    B = np.zeros((8, 2), dtype=np.float64)
    B[4:8, :] = temp[:, 8:10]

    return A, B


def permutation_to_env_order() -> tuple[np.ndarray, np.ndarray]:
    """
    Our env observation order (for compatibility with earlier README + common robotics logging):
      env_x = [theta_L, theta_R, theta_L_dot, theta_R_dot, theta_1, theta_dot_1, theta_2, theta_dot_2]

    MATLAB model state order:
      mat_x = [theta_L, theta_R, theta_1, theta_2, theta_L_dot, theta_R_dot, theta_dot_1, theta_dot_2]

    This returns permutation matrices P such that:
      env_x = P * mat_x
      mat_x = P_inv * env_x
    """
    # Index mapping: env_i -> mat_index
    env_to_mat = np.array([0, 1, 4, 5, 2, 6, 3, 7], dtype=np.int64)
    P = np.zeros((8, 8), dtype=np.float64)
    for env_i, mat_i in enumerate(env_to_mat):
        P[env_i, mat_i] = 1.0
    P_inv = P.T  # permutation matrix inverse
    return P, P_inv

