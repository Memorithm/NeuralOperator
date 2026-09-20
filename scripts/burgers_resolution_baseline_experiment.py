#!/usr/bin/env python3
"""Compare coarse-output interpolation with direct FNO resolution transfer."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    NumpyFNO1D,
    generate_burgers_dataset,
    periodic_linear_interpolate,
    relative_l2_error,
)


def main() -> None:
    train = generate_burgers_dataset(
        samples=4, points=16, dt=2.0e-4, steps=3, viscosity=0.05, seed=1
    )
    coarse = generate_burgers_dataset(
        samples=2, points=16, dt=2.0e-4, steps=3, viscosity=0.05, seed=2
    )
    fine = generate_burgers_dataset(
        samples=2, points=32, dt=2.0e-4, steps=3, viscosity=0.05, seed=2
    )

    model = NumpyFNO1D(width=3, modes=5, seed=4)
    fit = model.fit(train.inputs, train.targets, maxiter=60, tolerance=1.0e-7)
    coarse_prediction = model(coarse.inputs)
    fine_inputs = periodic_linear_interpolate(coarse.inputs[..., 0], 32)[..., None]
    direct_fine_prediction = model(fine_inputs)
    exact_fine_prediction = model(fine.inputs)
    interpolated_prediction = periodic_linear_interpolate(
        coarse_prediction[..., 0], 32
    )[..., None]
    interpolated_truth = periodic_linear_interpolate(coarse.targets[..., 0], 32)[..., None]

    def metrics(prediction, target):
        return {
            "mse": float(((prediction - target) ** 2).mean()),
            "relative_l2": relative_l2_error(prediction, target),
        }

    print(
        json.dumps(
            {
                "experiment": "burgers_resolution_interpolation_baseline",
                "fit": {
                    "initial_loss": fit.initial_loss,
                    "final_loss": fit.final_loss,
                    "success": fit.success,
                },
                "coarse_fno_on_coarse_grid": metrics(coarse_prediction, coarse.targets),
                "direct_fno_on_interpolated_fine_input": metrics(
                    direct_fine_prediction, fine.targets
                ),
                "direct_fno_on_exact_fine_input": metrics(
                    exact_fine_prediction, fine.targets
                ),
                "interpolated_coarse_fno_output": metrics(
                    interpolated_prediction, fine.targets
                ),
                "interpolated_coarse_truth": metrics(interpolated_truth, fine.targets),
                "note": "The interpolation baseline transports a coarse prediction; it does not learn an operator.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
