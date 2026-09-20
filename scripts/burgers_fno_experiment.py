#!/usr/bin/env python3
"""Train the tiny FNO on Burgers and evaluate a held-out resolution."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    NumpyFNO1D,
    generate_burgers_dataset,
    relative_l2_error,
)


def main() -> None:
    train = generate_burgers_dataset(
        samples=4, points=16, dt=2.0e-4, steps=3, viscosity=0.05, seed=1
    )
    held_out = generate_burgers_dataset(
        samples=2, points=16, dt=2.0e-4, steps=3, viscosity=0.05, seed=2
    )
    fine = generate_burgers_dataset(
        samples=2, points=32, dt=2.0e-4, steps=3, viscosity=0.05, seed=3
    )
    model = NumpyFNO1D(width=3, modes=5, seed=4)
    fit = model.fit(train.inputs, train.targets, maxiter=60, tolerance=1.0e-7)
    print(
        json.dumps(
            {
                "experiment": "burgers_fno_reference",
                "fit": {
                    "initial_loss": fit.initial_loss,
                    "final_loss": fit.final_loss,
                    "success": fit.success,
                    "iterations": fit.iterations,
                    "function_evaluations": fit.function_evaluations,
                    "message": fit.message,
                },
                "mse": {
                    "train_resolution_16": model.loss(train.inputs, train.targets),
                    "held_out_resolution_16": model.loss(
                        held_out.inputs, held_out.targets
                    ),
                    "zero_shot_resolution_32": model.loss(fine.inputs, fine.targets),
                },
                "relative_l2": {
                    "held_out_resolution_16": relative_l2_error(
                        model(held_out.inputs), held_out.targets
                    ),
                    "zero_shot_resolution_32": relative_l2_error(
                        model(fine.inputs), fine.targets
                    ),
                },
                "dataset": {
                    "dt": train.dt,
                    "steps": train.steps,
                    "viscosity": train.viscosity,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
