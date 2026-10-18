"""Score the Bitcoin holdout with the sticky regime HAR.

uv run python -m examples.bitcoin
uv run python -m examples.bitcoin 40
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

from examples.bitcoin.study import run


def main() -> None:
    """Fit the published Bitcoin protocol, or the sweep count in ``argv``."""

    sweeps = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(sweeps=sweeps)


if __name__ == "__main__":
    main()
