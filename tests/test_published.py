"""Regression against the published pendulum forecast."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from examples.pendulum.forecast import fit_and_forecast, forecast_metrics

_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "published.json").read_text())
_PENDULUM_RMSE_ATOL = 0.02
_PENDULUM_COVERAGE_ATOL = 0.02


def test_pendulum_forecast_matches_the_published_scores() -> None:
    metrics = forecast_metrics(fit_and_forecast(300, 80))
    reference = _FIXTURE["pendulum"]
    assert metrics["rmse_model_10"] == pytest.approx(reference["rmse_10"], abs=_PENDULUM_RMSE_ATOL)
    assert metrics["rmse_model_full"] == pytest.approx(
        reference["rmse_full"], abs=_PENDULUM_RMSE_ATOL
    )
    assert metrics["coverage_10"] == pytest.approx(
        reference["coverage_10"], abs=_PENDULUM_COVERAGE_ATOL
    )
    assert metrics["coverage_full"] == pytest.approx(
        reference["coverage_full"], abs=_PENDULUM_COVERAGE_ATOL
    )
