"""Symplectic Euler pendulum shared by the fit and the forecast."""

from __future__ import annotations

import numpy as np

# Observation noise uses a Generator. The Gibbs sampler uses a JAX key with this seed.
SEED = 11
TRACE_COUNT = 4
STEP_COUNT = 160
DT = 0.05
GRAVITY_OVER_LENGTH = 9.81
RELEASE_ANGLES = (0.45, 1.05, 1.70, 2.35)
OBSERVATION_STD = np.array([0.04, 0.12])
PROCESS_STD = 0.01
TRUNCATION = 5
DEFAULT_SWEEPS = 300


def simulate_pendulum(seed: int = SEED) -> dict[str, np.ndarray]:
    """Integrate several releases and add observation noise.

    Args:
        seed: Seed for the observation noise and the small process noise.

    Returns:
        Time, the noise-free state, and the noisy observations. Both series
        have shape ``(traces, steps, 2)`` in ``(angle, angular velocity)``.
    """

    rng = np.random.default_rng(seed)
    state = np.empty((TRACE_COUNT, STEP_COUNT, 2))
    stiffness = GRAVITY_OVER_LENGTH
    for trace, release in enumerate(RELEASE_ANGLES):
        angle = float(release)
        velocity = 0.0
        for step in range(STEP_COUNT):
            state[trace, step, 0] = angle
            state[trace, step, 1] = velocity
            velocity = velocity - DT * stiffness * np.sin(angle)
            velocity += float(rng.normal(0.0, PROCESS_STD))
            angle = angle + DT * velocity
    noise = rng.normal(0.0, 1.0, size=state.shape) * OBSERVATION_STD
    time = np.arange(STEP_COUNT) * DT
    return {"time": time, "state": state, "observations": state + noise}


def continue_pendulum(state: np.ndarray, extra_steps: int, seed: int) -> dict[str, np.ndarray]:
    """Integrate ``extra_steps`` past the last stored state.

    The step is the same symplectic Euler update as :func:`simulate_pendulum`.
    The stored state is the pre-update angle and velocity, so the first
    returned frame is one step after ``state[:, -1]``.

    Args:
        state: Noise-free series, shape ``(traces, steps, 2)``.
        extra_steps: Number of new frames.
        seed: Seed for the process disturbance and the observation noise
            on the continuation.

    Returns:
        Noise-free continuation and noisy observations, each of shape
        ``(traces, extra_steps, 2)``.
    """

    rng = np.random.default_rng(seed)
    trace_count = state.shape[0]
    continued = np.empty((trace_count, extra_steps, 2))
    stiffness = GRAVITY_OVER_LENGTH
    for trace in range(trace_count):
        angle = float(state[trace, -1, 0])
        velocity = float(state[trace, -1, 1])
        for step in range(extra_steps):
            velocity = velocity - DT * stiffness * np.sin(angle)
            velocity += float(rng.normal(0.0, PROCESS_STD))
            angle = angle + DT * velocity
            continued[trace, step, 0] = angle
            continued[trace, step, 1] = velocity
    noise = rng.normal(0.0, 1.0, size=continued.shape) * OBSERVATION_STD
    return {"state": continued, "observations": continued + noise}


def small_angle_dynamics() -> np.ndarray:
    """Linear map of the symplectic Euler step at ``theta = 0``."""

    stiffness = DT * GRAVITY_OVER_LENGTH
    return np.array(
        [
            [1.0 - DT * stiffness, DT],
            [-stiffness, 1.0],
        ]
    )
