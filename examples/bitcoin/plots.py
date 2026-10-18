"""Figures for the Bitcoin rolling-origin study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def save_regime_plots(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    """Write the forecast, horizon, probability, and convergence figures."""

    destination.mkdir(parents=True, exist_ok=True)
    _plot_forecast(name, asset_name, result, destination)
    _plot_horizon(name, asset_name, result, destination)
    _plot_probabilities(name, asset_name, result, destination)
    _plot_convergence(name, asset_name, result, destination)


def _plot_forecast(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    horizon = 20 if 20 in result["actual"] else result["horizons"][-1]
    actual = result["actual"][horizon]
    dates = np.asarray(result["dates"])[: actual.size]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(dates, actual, color="black", lw=1.3, label="realized")
    series = [
        ("regime", "#1f4e79", 1.4),
        ("har", "#b85c38", 1.1),
        ("ewma", "#2a9d8f", 1.0),
        ("garch", "#6d597a", 1.0),
        ("mean", "#888888", 1.0),
    ]
    for key, color, width in series:
        if key == "regime":
            values = result["regime"][horizon]
        else:
            values = result["baseline"][horizon][key]
        ax.plot(dates, values, color=color, lw=width, label=key)
    ax.set_ylabel(f"log mean r^2 over {horizon} days")
    ax.set_xlabel("forecast origin")
    ax.legend(ncol=3, frameon=False)
    ax.set_title(f"{asset_name}: rolling-origin {horizon}-day log variance")
    if np.issubdtype(np.asarray(dates).dtype, np.datetime64) or _is_datetime(dates):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_forecast.png", dpi=140)
    plt.close(fig)


def _is_datetime(dates: np.ndarray) -> bool:
    if dates.size == 0:
        return False
    return hasattr(dates[0], "strftime")


def _plot_horizon(name: str, asset_name: str, result: dict[str, object], destination: Path) -> None:
    horizons = list(result["horizons"])
    models = ["regime", "har", "ewma", "garch", "mean"]
    colors = ["#1f4e79", "#b85c38", "#2a9d8f", "#6d597a", "#888888"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    width = 0.15
    positions = np.arange(len(horizons), dtype=np.float64)
    for offset, (model, color) in enumerate(zip(models, colors, strict=True)):
        heights = []
        for horizon in horizons:
            if model not in result["scores"][horizon]:
                heights.append(np.nan)
            else:
                heights.append(result["scores"][horizon][model]["rmse"])
        ax.bar(positions + (offset - 2) * width, heights, width=width, color=color, label=model)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(horizon) for horizon in horizons])
    ax.set_xlabel("horizon (days)")
    ax.set_ylabel("RMSE of log mean r^2")
    ax.legend(ncol=3, frameon=False)
    ax.set_title(f"{asset_name}: rolling-origin RMSE")
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_horizon.png", dpi=140)
    plt.close(fig)


def _plot_probabilities(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    dates = np.asarray(result["dates"])
    probs = result["probabilities"]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True)
    for regime in range(probs.shape[1]):
        axes[0].plot(dates, probs[:, regime], lw=1.1, label=f"regime {regime}")
    axes[0].set_ylabel("filtered probability")
    axes[0].legend(ncol=3, frameon=False)
    horizon = 1 if 1 in result["actual"] else result["horizons"][0]
    axes[1].plot(
        dates[: result["actual"][horizon].size], result["actual"][horizon], color="black", lw=1.0
    )
    axes[1].plot(
        dates[: result["regime"][horizon].size], result["regime"][horizon], color="#1f4e79", lw=1.0
    )
    axes[1].set_ylabel("log next-day r^2")
    axes[1].set_xlabel("forecast origin")
    axes[0].set_title(f"{asset_name}: filtered regimes, ordered from low to high intercept")
    if _is_datetime(dates):
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_probs.png", dpi=140)
    plt.close(fig)


def _plot_convergence(
    name: str, asset_name: str, result: dict[str, object], destination: Path
) -> None:
    loglik = result["diagnostics"]["loglik_first"]
    if loglik is None:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    sweeps = np.arange(1, loglik.shape[1] + 1)
    half = loglik.shape[1] // 2
    for chain in range(loglik.shape[0]):
        ax.plot(sweeps, loglik[chain], lw=1.0, label=f"chain {chain + 1}")
    ax.axvline(half + 1, color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("Gibbs sweep")
    ax.set_ylabel("complete-data log likelihood")
    ax.legend(frameon=False)
    rhat = float(result["diagnostics"]["refits"][0]["rhat_max"])
    ax.set_title(f"{asset_name}: first refit, max R-hat {rhat:.3f}")
    fig.tight_layout()
    fig.savefig(destination / f"{name}_regime_convergence.png", dpi=140)
    plt.close(fig)
