"""Run the published Bitcoin rolling-origin score and write its figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from examples.bitcoin.plots import save_regime_plots
from sticky_hdp_slds import evaluate_returns, load_closes
from sticky_hdp_slds.volatility import CHAINS, REFIT_YEARS, SWEEPS, datasets

_DATA_DIR = Path(__file__).resolve().parent / "data"
_IMG_DIR = Path(__file__).resolve().parents[2] / "docs" / "img"
_DATASETS = datasets(_DATA_DIR)


def _print_report(name: str, result: dict[str, object]) -> None:
    diagnostics = result["diagnostics"]
    print(f"{name} rolling-origin log mean r^2", flush=True)
    print(
        "  "
        + " ".join(
            f"{key} {diagnostics[key]}"
            for key in (
                "holdout",
                "train_days",
                "sweeps",
                "chains",
                "refit_every",
                "seed",
                "regimes",
                "self_stay_prior",
                "zero_returns",
                "garch_failures",
            )
        ),
        flush=True,
    )
    print(
        f"  rhat_max {diagnostics['rhat_max']:.4f} elapsed_sec {diagnostics['elapsed_sec']:.1f}",
        flush=True,
    )
    for item in diagnostics["refits"]:
        intercepts = " ".join(f"{value:.3f}" for value in item["intercepts"])
        slopes = " ".join(f"{value:.3f}" for value in item["slopes"])
        stay = " ".join(f"{value:.3f}" for value in item["self_stay_by_regime"])
        share = " ".join(f"{value:.3f}" for value in item["occupancy"])
        print(
            f"  refit {item['origin']} rhat {item['rhat_max']:.3f} stay {stay} "
            f"dur {' '.join(f'{value:.1f}' for value in item['duration_days'])} "
            f"c {intercepts} b {slopes} radius {item['spectral_radius']:.3f} "
            f"share {share} ll {item['loglik_second_mean']:.1f} "
            f"ll_std {item['loglik_second_std']:.1f}",
            flush=True,
        )
    for horizon, scores in result["scores"].items():
        parts = [f"h {horizon} n {scores['n']}"]
        for model in ("regime", "har", "ewma", "garch", "mean"):
            if model not in scores:
                continue
            parts.append(f"{model} {scores[model]['rmse']:.6f} bias {scores[model]['bias']:.6f}")
        print("  " + " ".join(parts), flush=True)
        for model, test in scores["dm"].items():
            print(
                f"    dm vs {model} stat {test['stat']:.4f} p {test['pvalue']:.4g} "
                f"fallback {test['nw_fallback']:.0f}",
                flush=True,
            )


def run(sweeps: int | None = None, chains: int | None = None) -> dict[str, object]:
    """Score Bitcoin and write the four regime figures."""

    spec = _DATASETS["btc"]
    snapshot = load_closes(spec["path"])
    closes = snapshot["closes"]
    returns = np.log(closes[1:] / closes[:-1])
    result = evaluate_returns(
        returns,
        holdout=int(spec["holdout"]),
        train_days=REFIT_YEARS * int(spec["annualization_days"]),
        sweeps=SWEEPS if sweeps is None else sweeps,
        chains=CHAINS if chains is None else chains,
        dates=np.asarray(snapshot["dates"]),
    )
    _print_report("btc", result)
    save_regime_plots(spec["prefix"], spec["name"], result, _IMG_DIR)
    print(f"  figures {_IMG_DIR}", flush=True)
    return result
