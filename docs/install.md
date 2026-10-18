# Install and usage

This project targets Python 3.11 and later. The import name is `sticky_hdp_slds`. [uv](https://docs.astral.sh/uv/) installs the locked dependencies from `uv.lock` and builds the package with the uv build backend. The Gibbs samplers are JAX. NumPy is used for the price files, the baseline regressions, and the pendulum integrator.

```bash
uv sync
uv run pytest
uv run python -m examples.smoke
```

`uv sync` installs the package, the `dev` group (pytest, ruff, pre-commit, and the `examples` group), and the `docs` group. The `examples` group provides `matplotlib`, the same library as the `plots` extra.

```bash
uv sync
uv run mkdocs serve
uv build
```

A wheel install from another project uses the same extras:

```bash
uv add sticky-hdp-slds
uv add "sticky-hdp-slds[plots]"
```

The smoke script builds a two-mode system in two latent dimensions, simulates two sequences of length 12, and runs three blocked Gibbs sweeps. A successful run prints a finite log joint, matching latent and observation shapes, transition rows that sum to one, and a self-transition weight `rho` in `(0, 1)`.

## Examples

```bash
uv run python -m examples.pendulum.demo
uv run python -m examples.pendulum.forecast
```

`examples.pendulum.demo` and `examples.pendulum.forecast` accept a sweep count as their only argument. The default is 300. The forecast fits the training window only.

## Layout

| Path | Role |
| --- | --- |
| `src/sticky_hdp_slds/` | Installable package: Gibbs samplers and the volatility study |
| `examples/smoke.py` | Short end-to-end run |
| `examples/pendulum/` | Noisy pendulum fit and 80-step forecast |
| `tests/` | Unit checks and stored-reference regressions |
| `docs/` | This site's source |
