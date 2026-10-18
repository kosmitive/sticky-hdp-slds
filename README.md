# Sticky HDP-SLDS

Gibbs sampling for a sticky hierarchical Dirichlet process switching linear dynamical system, following Fox, Sudderth, Jordan, and Willsky, [Bayesian Nonparametric Inference of Switching Linear Dynamical Systems](https://arxiv.org/abs/1003.3829). The sampler is JAX. The result below is an 80-step forecast of a noisy pendulum that the sampler never sees.

## Forecast

![Fitted window, 90% forecast band, and held-out observations for the largest release](docs/img/pendulum_forecast.png)

![RMSE and 90% coverage against lead time](docs/img/pendulum_forecast_horizon.png)

`examples/pendulum/forecast.py` fits the training series with the blocked sampler (seed 11, 300 sweeps, sequential mode updates off) and scores 150 futures from sweeps 151–300. The chain starts at the prior means. Pooled over four traces and both channels, the predictive mean has RMSE 0.162 on the first 10 steps and 1.553 over all 80. The central 90% band covers 97.5% of the first 10 steps and 92.0% of the horizon. Repeating the last observation scores 1.446 and 3.032 on those same windows. An earlier sequential sampler of this holdout scored RMSE 1.456 and 94.5% coverage over the 80 steps.

The 2.35 rad release is the trace in the figure. Its 90% band covers 69.4% of the horizon and 40% of the last 20-step block. The three smaller releases have horizon coverage 99.4%, 99.4%, and 100%.

## Quick start

[uv](https://docs.astral.sh/uv/) installs the locked environment. Python 3.11 or newer is required.

```bash
uv sync
uv run pytest
uv run python -m examples.smoke
uv run python -m examples.pendulum.forecast
uv run pre-commit install
```

`uv sync` installs `sticky_hdp_slds`, the `dev` group (pytest, ruff, pre-commit, and the `examples` group), and the `docs` group. The `examples` group provides `matplotlib`, which is also the `plots` extra. Another project installs the wheel with uv:

```bash
uv add sticky-hdp-slds
uv add "sticky-hdp-slds[plots]"
```

Serve the site:

```bash
uv sync
uv run mkdocs serve
uv build
```

The smoke script builds a two-mode system in two latent dimensions, simulates two sequences of length 12, and runs three blocked Gibbs sweeps. A successful run prints a finite log joint, matching latent and observation shapes, transition rows that sum to one, and a self-transition weight `rho` in `(0, 1)`.

## Layout

| Path | Role |
| --- | --- |
| `src/sticky_hdp_slds/` | Installable package: JAX Gibbs samplers |
| `examples/smoke.py` | Short end-to-end run |
| `examples/pendulum/` | Noisy pendulum fit and 80-step forecast |
| `tests/` | Unit checks and stored-reference regressions |
| `docs/` | This site's source |

## Results

### Noisy pendulum

```bash
uv run python -m examples.pendulum.demo
```

The script integrates a nonlinear pendulum with the symplectic Euler method (`g/l = 9.81`) and fits the sticky HDP-SLDS. Four traces are released from rest at 0.45, 1.05, 1.70, and 2.35 rad. Each trace has 160 steps of `dt = 0.05` s. The observations are the angle and the angular velocity, with standard deviations 0.04 and 0.12, and the velocity also receives a process disturbance of standard deviation 0.01. The emission matrix is the identity, so piecewise-linear modes are asked to track the angle dependence of `sin(theta)`.

The truncation is 5 modes. Prior means are `alpha + kappa = 12`, `rho = 0.92`, and `gamma = 1.5`. The matrix-normal mean is the small-angle map of the same integrator, with column precision `40 I`. Those scales belong to this series. A different length, noise level, or sampling rate needs its own prior.

The figures are the blocked sampler: seed 11, 300 sweeps, sequential mode updates off, 6.6 s. The complete-data log joint starts at 2109 and ends at 3489. The four quarter means are 3462, 3516, 3540, and 3529. Over sweeps 151–300 the mean is 3534, the standard deviation is 43, and the least-squares slope is -0.07 per sweep. The last sample uses 2 modes. The second-half average is 2.05, and the chain sometimes occupies all 5.

The last sample splits the swing into two sticky regimes. The mode whose mean absolute angle is 0.614 has restoring entry `A_{2,1} = -0.424` and a mean dwell of 30.6 steps. The small-angle integrator has `A_{2,1} = -dt * g/l = -0.491`. The mode whose mean absolute angle is 1.877 has `A_{2,1} = -0.217` and a mean dwell of 9.4 steps. The posterior mean of the latent state over the second half of the chain has RMSE 0.056 against the noise-free trajectory. The raw observations have RMSE 0.089.

![Log joint and occupied modes across 300 sweeps](docs/img/pendulum_convergence.png)

![Observed, noise-free, and posterior-mean angle and angular velocity](docs/img/pendulum_reconstruction.png)

![Final mode sample against the noise-free angle](docs/img/pendulum_modes.png)

### Forecast

```bash
uv run python -m examples.pendulum.forecast
```

The forecast uses the same training series, seed, and hyperparameters as the fit. Eighty further steps are integrated with continuation seed 12 and are never shown to the sampler. The chain is 300 blocked sweeps and the sampler took 6.8 s. Sweeps 151–300 each contribute one future. The headline scores and both figures are at the top of this page.

Angle and angular velocity coverage over the full horizon are 91.9% and 92.2%. The three smaller releases have holdout RMSE 0.419, 0.578, and 1.064, with coverage 99.4%, 99.4%, and 100%. The largest release, 2.35 rad, has RMSE 0.223 on the first 10 steps (coverage 100%) and RMSE 2.830 over the horizon (coverage 69.4%). In 20-step blocks its RMSE is 0.241, 1.574, 3.084, and 4.471, and the band coverage is 100%, 75%, 62.5%, and 40%. The prior was not adjusted to the holdout.

The chain mixes inside a band; it does not freeze. The forecast on the 2.35 rad release is the case where a linear mode fitted on the observed swing does not carry the trajectory for the full extra horizon.

## Tests

`uv run pytest` runs the checks in `tests/`. The fast tests cover filter normalization, a three-step filter against exact enumeration, a forced backward sample, conjugate Gaussian and inverse-Wishart moments, the ARD normal equations, and the Schur-Cohn test against companion eigenvalues. The slower test compares the pendulum forecast with the stored values in `tests/fixtures/published.json`.

## License

MIT. See [LICENSE](LICENSE).
