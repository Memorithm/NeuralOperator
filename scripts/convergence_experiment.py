#!/usr/bin/env python3
"""Compare spectral and second-order periodic finite-difference derivatives."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    central_difference_periodic,
    relative_l2_error,
    spectral_derivative,
)


def main() -> None:
    length = 2.0 * np.pi
    rows = []
    for n in (16, 32, 64, 128, 256):
        x = np.arange(n) * length / n
        field = np.exp(np.sin(x))
        expected = np.cos(x) * np.exp(np.sin(x))
        spectral_error = relative_l2_error(
            spectral_derivative(field, length=length), expected
        )
        finite_difference_error = relative_l2_error(
            central_difference_periodic(field, length=length), expected
        )
        rows.append(
            {
                "n": n,
                "spectral_relative_l2_error": spectral_error,
                "central_difference_relative_l2_error": finite_difference_error,
            }
        )
    print(json.dumps({"experiment": "smooth_periodic_convergence", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
