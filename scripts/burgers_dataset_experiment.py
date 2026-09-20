#!/usr/bin/env python3
"""Generate a deterministic Burgers dataset and report its provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import generate_burgers_dataset  # noqa: E402


def main() -> None:
    dataset = generate_burgers_dataset(
        samples=8,
        points=32,
        dt=2.0e-4,
        steps=5,
        viscosity=0.05,
        seed=20260920,
    )
    input_mean = np.mean(dataset.inputs, axis=1)
    target_mean = np.mean(dataset.targets, axis=1)
    print(
        json.dumps(
            {
                "experiment": "burgers_dataset_generation",
                "samples": dataset.samples,
                "resolution": dataset.resolution,
                "input_shape": list(dataset.inputs.shape),
                "target_shape": list(dataset.targets.shape),
                "length": dataset.length,
                "dt": dataset.dt,
                "steps": dataset.steps,
                "viscosity": dataset.viscosity,
                "seed": dataset.seed,
                "maximum_mean_drift": float(np.max(np.abs(target_mean - input_mean))),
                "maximum_input_target_difference": float(
                    np.max(np.abs(dataset.targets - dataset.inputs))
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
