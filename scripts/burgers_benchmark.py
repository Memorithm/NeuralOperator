#!/usr/bin/env python3
"""Run a small invariant check for the periodic Burgers reference solver."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    integrate_burgers,
    periodic_energy,
    periodic_mean,
)


def main() -> None:
    n = 64
    length = 2.0 * np.pi
    x = np.arange(n) * length / n
    initial = 0.2 + np.sin(x) + 0.25 * np.cos(2.0 * x)
    trajectory = integrate_burgers(
        initial,
        dt=2.0e-4,
        steps=40,
        viscosity=0.05,
        length=length,
    )
    means = np.asarray([periodic_mean(field) for field in trajectory])
    energies = np.asarray([periodic_energy(field) for field in trajectory])
    print(
        json.dumps(
            {
                "experiment": "periodic_burgers_reference_solver",
                "grid_points": n,
                "steps": 40,
                "dt": 2.0e-4,
                "viscosity": 0.05,
                "initial_mean": float(means[0]),
                "final_mean": float(means[-1]),
                "mean_drift": float(np.max(np.abs(means - means[0]))),
                "initial_energy": float(energies[0]),
                "final_energy": float(energies[-1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
