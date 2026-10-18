"""Fit the sticky HDP-SLDS to a noisy nonlinear pendulum.

uv run python -m examples.pendulum.demo
uv run python -m examples.pendulum.demo 40
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from examples.pendulum.fit import convergence_summary, fit
from examples.pendulum.plots import save_plots
from examples.pendulum.series import (
    DEFAULT_SWEEPS,
    DT,
    SEED,
    STEP_COUNT,
    TRACE_COUNT,
    TRUNCATION,
    simulate_pendulum,
)

_IMG_DIR = Path(__file__).resolve().parents[2] / "docs" / "img"


def _print_regime_diagnostic(
    series: dict[str, np.ndarray],
    history: dict[str, np.ndarray],
) -> None:
    """Print whether the final modes split the swing by absolute angle."""

    final = history["modes"][-1]
    angle = np.abs(series["state"][:, :, 0])
    half = history["log_joint"].size // 2
    quarters = np.array_split(history["log_joint"], 4)
    print("log_joint_quarter_means", " ".join(f"{float(np.mean(part)):.1f}" for part in quarters))
    posterior_mean = np.mean(history["latents"][half:], axis=0)
    truth = series["state"]
    observed = series["observations"]
    print(
        "rmse_posterior_mean",
        float(np.sqrt(np.mean((posterior_mean - truth) ** 2))),
        "rmse_observations",
        float(np.sqrt(np.mean((observed - truth) ** 2))),
    )
    print("final_A")
    for mode in np.unique(final):
        mask = final == mode
        dwells = []
        for trace in final:
            run = 0
            for value in trace:
                if value == mode:
                    run += 1
                elif run:
                    dwells.append(run)
                    run = 0
            if run:
                dwells.append(run)
        print(
            f"  mode {int(mode)}  count {int(mask.sum())}  "
            f"mean_abs_angle {float(np.mean(angle[mask])):.3f}  "
            f"mean_dwell {float(np.mean(dwells)):.2f}"
        )
        print("   ", np.array2string(history["dynamics"][-1, mode], precision=3))


def main() -> None:
    """Simulate the pendulum, run the sampler, and write the three figures."""

    sweeps = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEPS
    series = simulate_pendulum(SEED)
    print(
        f"pendulum demo  seed {SEED}  traces {TRACE_COUNT}  "
        f"steps {STEP_COUNT}  dt {DT}  truncation {TRUNCATION}  sweeps {sweeps}",
        flush=True,
    )
    history = fit(series["observations"], sweeps)
    summary = convergence_summary(history["log_joint"], history["used_modes"])
    save_plots(series, history, _IMG_DIR)
    _print_regime_diagnostic(series, history)
    print("convergence")
    for key, value in summary.items():
        print(f"  {key:28s} {value}")
    print(f"  elapsed_sec                  {float(history['elapsed_sec']):.1f}")
    print(f"  rho_last                     {float(history['rho'][-1]):.3f}")
    print(f"  figures                      {_IMG_DIR}")


if __name__ == "__main__":
    main()
