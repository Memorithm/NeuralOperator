#!/usr/bin/env python3
"""Run a small deterministic convergence check for the spectral oracle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import relative_l2_error, spectral_derivative  # noqa: E402


def main() -> None:
    length = 2.0 * np.pi
    rows = []
    for n in (16, 32, 64, 128, 256):
        x = np.arange(n) * length / n
        field = np.sin(3.0 * x) + 0.25 * np.cos(5.0 * x)
        expected = 3.0 * np.cos(3.0 * x) - 1.25 * np.sin(5.0 * x)
        derivative = spectral_derivative(field, length=length)
        rows.append({"n": n, "relative_l2_error": relative_l2_error(derivative, expected)})
    print(json.dumps({"experiment": "smooth_periodic_derivative", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
