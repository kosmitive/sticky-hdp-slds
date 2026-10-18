"""Blocked sticky HDP-SLDS fit for the noisy pendulum."""

from __future__ import annotations

import time

import numpy as np

from examples.pendulum.series import OBSERVATION_STD, SEED, TRUNCATION, small_angle_dynamics


def _slds_prior_and_state(observations: np.ndarray):
    """Prior means and an observation-initialized state for the blocked sampler."""

    import jax.numpy as jnp

    from sticky_hdp_slds import SLDSData, SLDSPrior, SLDSState

    latent_dim = 2
    mean_dynamics = jnp.asarray(small_angle_dynamics(), dtype=jnp.float64)
    dynamics = jnp.broadcast_to(mean_dynamics, (TRUNCATION, latent_dim, latent_dim))
    beta = jnp.full(TRUNCATION, 1.0 / TRUNCATION)
    stay = 0.92
    transition = (1.0 - stay) * beta + stay * jnp.eye(TRUNCATION)
    observations_j = jnp.asarray(observations, dtype=jnp.float64)
    state = SLDSState(
        modes=jnp.zeros(observations_j.shape[:2], dtype=jnp.int32),
        latents=observations_j,
        dynamics=dynamics,
        process_cov=jnp.broadcast_to(
            0.15 * jnp.eye(latent_dim), (TRUNCATION, latent_dim, latent_dim)
        ),
        emission_cov=jnp.diag(jnp.asarray(OBSERVATION_STD, dtype=jnp.float64) ** 2),
        beta=beta,
        transition=transition,
        alpha_plus_kappa=jnp.asarray(12.0),
        rho=jnp.asarray(stay),
        gamma=jnp.asarray(1.5),
        column_precision=jnp.full((TRUNCATION, latent_dim), 40.0),
    )
    prior = SLDSPrior(
        alpha_kappa=jnp.asarray([12.0, 1.0]),
        rho=jnp.asarray([24.0, 2.0]),
        gamma=jnp.asarray([3.0, 2.0]),
        dynamics_mean=mean_dynamics,
        dynamics_column_precision=40.0 * jnp.eye(latent_dim),
        dynamics_df=jnp.asarray(8.0),
        dynamics_scale=0.15 * jnp.eye(latent_dim),
        latent_prior=jnp.diag(jnp.asarray([2.0, 16.0])),
        emission_df=jnp.asarray(6.0),
        emission_scale=jnp.diag(jnp.asarray(OBSERVATION_STD, dtype=jnp.float64) ** 2) * 3.0,
        emission=jnp.eye(latent_dim),
        ard=jnp.ones(latent_dim),
    )
    return state, SLDSData(observations=observations_j), prior


def fit(observations: np.ndarray, sweeps: int) -> dict[str, np.ndarray]:
    """Run ``sweeps`` blocked Gibbs iterations and keep the monitored traces.

    Args:
        observations: Noisy pendulum series, shape ``(traces, steps, 2)``.
        sweeps: Number of complete Gibbs sweeps.

    Returns:
        Log joint, occupied-mode counts, ``rho``, latents, modes, and dynamics.
    """

    from jax import random

    from sticky_hdp_slds import sample_slds

    state, data, prior = _slds_prior_and_state(observations)
    started = time.perf_counter()
    trace, loglik = sample_slds(
        random.PRNGKey(SEED),
        state,
        data,
        prior,
        sweeps,
        sequential_modes=False,
        ard=False,
    )
    loglik.block_until_ready()
    elapsed = time.perf_counter() - started
    log_joint = np.asarray(loglik)
    modes = np.asarray(trace.modes)
    used_modes = np.array([np.unique(modes[sweep]).size for sweep in range(sweeps)])
    print(
        f"blocked sweeps {sweeps} in {elapsed:.1f}s  "
        f"log_joint[0] {log_joint[0]:.2f}  log_joint[-1] {log_joint[-1]:.2f}",
        flush=True,
    )
    return {
        "log_joint": log_joint,
        "used_modes": used_modes,
        "rho": np.asarray(trace.rho),
        "latents": np.asarray(trace.latents),
        "modes": modes,
        "dynamics": np.asarray(trace.dynamics),
        "elapsed_sec": np.array(elapsed),
        "prior_emission": np.asarray(prior.emission),
        "process_cov": np.asarray(trace.process_cov),
        "transition": np.asarray(trace.transition),
        "emission_cov": np.asarray(trace.emission_cov),
    }


def convergence_summary(log_joint: np.ndarray, used_modes: np.ndarray) -> dict[str, float]:
    """Summarize whether the second half of the chain is still trending.

    Args:
        log_joint: Complete-data log joint after each sweep.
        used_modes: Number of distinct modes after each sweep.

    Returns:
        Levels, a per-sweep slope on the second half, and that slope measured
        in units of the second-half standard deviation.
    """

    half = max(len(log_joint) // 2, 1)
    second = np.asarray(log_joint[half:], dtype=np.float64)
    centered = np.arange(second.size, dtype=np.float64)
    centered -= centered.mean()
    denom = float(np.dot(centered, centered))
    slope = float(np.dot(centered, second - second.mean()) / denom) if denom > 0.0 else 0.0
    scale = float(np.std(second))
    return {
        "start": float(log_joint[0]),
        "mid": float(log_joint[half]),
        "end": float(log_joint[-1]),
        "second_half_mean": float(np.mean(second)),
        "second_half_std": scale,
        "second_half_slope_per_sweep": slope,
        "slope_over_std": slope / scale if scale > 0.0 else 0.0,
        "modes_min": int(used_modes.min()),
        "modes_max": int(used_modes.max()),
        "modes_last": int(used_modes[-1]),
        "modes_second_half_mean": float(np.mean(used_modes[half:])),
    }
