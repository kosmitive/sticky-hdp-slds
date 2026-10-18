"""Figures for the pendulum fit and the holdout forecast."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.pendulum.series import DT, RELEASE_ANGLES, TRACE_COUNT, TRUNCATION

_MODE_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2")


def _relabel_by_angle(modes: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Order mode ids by mean absolute angle so a figure is readable."""

    present = np.unique(modes)
    centers = []
    for mode in present:
        centers.append(float(np.mean(np.abs(angle[modes == mode]))))
    order = present[np.argsort(centers)]
    relabeled = np.empty_like(modes)
    for new_id, old_id in enumerate(order):
        relabeled[modes == old_id] = new_id
    return relabeled


def save_plots(
    series: dict[str, np.ndarray],
    history: dict[str, np.ndarray],
    destination: Path,
) -> None:
    """Write the convergence, reconstruction, and mode figures."""

    destination.mkdir(parents=True, exist_ok=True)
    sweeps = np.arange(1, history["log_joint"].size + 1)
    half = history["log_joint"].size // 2
    posterior_mean = np.mean(history["latents"][half:], axis=0)
    final_modes = _relabel_by_angle(history["modes"][-1], series["state"][:, :, 0])
    trace = int(np.argmax(RELEASE_ANGLES))
    time_axis = series["time"]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    axes[0].plot(sweeps, history["log_joint"], color="#1f4e79", lw=1.4)
    axes[0].axvline(half + 1, color="#888888", ls="--", lw=0.8)
    axes[0].set_ylabel("complete-data log joint")
    axes[1].plot(sweeps, history["used_modes"], color="#b85c38", lw=1.4)
    axes[1].set_ylabel("occupied modes")
    axes[1].set_xlabel("Gibbs sweep")
    axes[1].set_ylim(0.5, TRUNCATION + 0.5)
    fig.suptitle("Sticky HDP-SLDS on the noisy pendulum")
    fig.tight_layout()
    fig.savefig(destination / "pendulum_convergence.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    names = ("angle (rad)", "angular velocity (rad/s)")
    for axis, name in enumerate(names):
        axes[axis].plot(
            time_axis,
            series["observations"][trace, :, axis],
            color="#9aa0a6",
            lw=0.9,
            label="observed",
        )
        axes[axis].plot(
            time_axis,
            series["state"][trace, :, axis],
            color="#333333",
            lw=1.0,
            ls="--",
            label="noise-free",
        )
        axes[axis].plot(
            time_axis,
            posterior_mean[trace, :, axis],
            color="#1f4e79",
            lw=1.5,
            label="posterior mean",
        )
        axes[axis].set_ylabel(name)
    axes[0].legend(loc="upper right", frameon=False, ncol=3)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"Largest release ({RELEASE_ANGLES[trace]:.2f} rad), second-half mean state")
    fig.tight_layout()
    fig.savefig(destination / "pendulum_reconstruction.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(TRACE_COUNT, 1, figsize=(8.2, 7.2), sharex=True)
    for trace_id, axis in enumerate(axes):
        modes = final_modes[trace_id]
        change = np.flatnonzero(np.diff(modes)) + 1
        bounds = np.concatenate(([0], change, [modes.size]))
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            axis.axvspan(
                time_axis[start] - 0.5 * DT,
                time_axis[end - 1] + 0.5 * DT,
                color=_MODE_COLORS[int(modes[start]) % len(_MODE_COLORS)],
                alpha=0.45,
                lw=0,
            )
        axis.plot(time_axis, series["state"][trace_id, :, 0], color="#222222", lw=1.1)
        axis.set_ylabel(f"trace {trace_id + 1}")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Final mode sample, labels ordered by mean |angle|")
    fig.tight_layout()
    fig.savefig(destination / "pendulum_modes.png", dpi=140)
    plt.close(fig)


def _lead_rmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """RMSE at each lead time, pooled over traces and channels."""

    squared = np.mean((prediction - target) ** 2, axis=(0, 2))
    return np.sqrt(squared)


def save_forecast_plots(
    result: dict[str, np.ndarray],
    destination: Path,
    *,
    short_horizon: int,
) -> None:
    """Write the forecast ribbon and the lead-time score curves."""

    destination.mkdir(parents=True, exist_ok=True)
    futures = result["futures"]
    predictive_mean = np.mean(futures, axis=0)
    lower = np.percentile(futures, 5.0, axis=0)
    upper = np.percentile(futures, 95.0, axis=0)
    trace = int(np.argmax(RELEASE_ANGLES))
    train_time = result["time_train"]
    hold_time = train_time[-1] + DT * np.arange(1, futures.shape[2] + 1)
    split = train_time[-1]
    last = result["observations_train"][trace, -1]
    channel_names = ("angle (rad)", "angular velocity (rad/s)")

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True)
    for channel, axis in enumerate(axes):
        axis.axvspan(split, hold_time[-1] + 0.5 * DT, color="#f4e8d0", zorder=0)
        axis.plot(
            train_time,
            result["observations_train"][trace, :, channel],
            color="#9aa0a6",
            lw=0.8,
            label="fitted observations",
        )
        axis.plot(
            train_time,
            result["posterior_mean"][trace, :, channel],
            color="#1f4e79",
            lw=1.3,
            label="posterior mean",
        )
        axis.fill_between(
            hold_time,
            lower[trace, :, channel],
            upper[trace, :, channel],
            color="#1f4e79",
            alpha=0.22,
            label="90% predictive band",
        )
        axis.plot(
            hold_time,
            predictive_mean[trace, :, channel],
            color="#1f4e79",
            lw=1.5,
            label="predictive mean",
        )
        axis.plot(
            hold_time,
            result["observations_hold"][trace, :, channel],
            color="#b85c38",
            lw=1.1,
            label="held-out observations",
        )
        axis.plot(
            hold_time,
            np.full(hold_time.shape, last[channel]),
            color="#666666",
            lw=1.0,
            ls=":",
            label="last observation",
        )
        axis.axvline(split, color="#888888", lw=0.8)
        axis.set_ylabel(channel_names[channel])
    axes[0].text(0.74, 0.08, "held out", transform=axes[0].transAxes, color="#8a6a32", fontsize=8)
    axes[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"Forecast from the fitted window, release {RELEASE_ANGLES[trace]:.2f} rad")
    fig.tight_layout()
    fig.savefig(destination / "pendulum_forecast.png", dpi=140)
    plt.close(fig)

    holdout = result["observations_hold"]
    baseline = np.repeat(result["observations_train"][:, -1:, :], holdout.shape[1], axis=1)
    model_lead = _lead_rmse(predictive_mean, holdout)
    baseline_lead = _lead_rmse(baseline, holdout)
    inside = (holdout >= lower) & (holdout <= upper)
    coverage_lead = np.mean(inside, axis=(0, 2))
    largest_error = predictive_mean[trace, :, :] - holdout[trace, :, :]
    largest_lead = np.sqrt(np.mean(largest_error**2, axis=1))
    largest_coverage = np.mean(inside[trace], axis=1)
    lead = np.arange(1, holdout.shape[1] + 1)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)
    axes[0].plot(lead, model_lead, color="#1f4e79", lw=1.4, label="all traces")
    axes[0].plot(lead, largest_lead, color="#b85c38", lw=1.3, label="largest release")
    axes[0].plot(lead, baseline_lead, color="#666666", lw=1.2, ls=":", label="last observation")
    axes[0].axvline(short_horizon, color="#888888", lw=0.8, ls="--")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(frameon=False)
    axes[1].plot(lead, coverage_lead, color="#1f4e79", lw=1.4, label="all traces")
    axes[1].plot(lead, largest_coverage, color="#b85c38", lw=1.3, label="largest release")
    axes[1].axhline(0.90, color="#888888", lw=0.8, ls="--")
    axes[1].axvline(short_horizon, color="#888888", lw=0.8, ls="--")
    axes[1].set_ylabel("90% band coverage")
    axes[1].set_xlabel("lead time (steps)")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(frameon=False)
    fig.suptitle("Holdout score by lead time, pooled over traces and channels")
    fig.tight_layout()
    fig.savefig(destination / "pendulum_forecast_horizon.png", dpi=140)
    plt.close(fig)
