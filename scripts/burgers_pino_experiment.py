#!/usr/bin/env python3
"""Compare data-only and transition-physics-aware reference training."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    NumpyFNO1D,
    burgers_transition_physics_loss,
    fit_fno_burgers_physics,
    generate_burgers_dataset,
)


def main() -> None:
    train = generate_burgers_dataset(
        samples=4, points=16, dt=2.0e-4, steps=3, viscosity=0.05, seed=1
    )
    held_out = generate_burgers_dataset(
        samples=2, points=16, dt=2.0e-4, steps=3, viscosity=0.05, seed=2
    )
    data_only = NumpyFNO1D(width=3, modes=5, seed=4)
    data_fit = data_only.fit(train.inputs, train.targets, maxiter=60, tolerance=1.0e-7)

    physics_aware = NumpyFNO1D(width=3, modes=5, seed=4)
    physics_fit = fit_fno_burgers_physics(
        physics_aware,
        train.inputs,
        train.targets,
        horizon=train.dt * train.steps,
        viscosity=train.viscosity,
        physics_weight=1.0e-3,
        maxiter=60,
        tolerance=1.0e-8,
    )

    def physics_loss(model: NumpyFNO1D) -> float:
        return burgers_transition_physics_loss(
            held_out.inputs[..., 0],
            model(held_out.inputs)[..., 0],
            horizon=held_out.dt * held_out.steps,
            viscosity=held_out.viscosity,
            length=held_out.length,
        )

    print(
        json.dumps(
            {
                "experiment": "burgers_transition_physics_comparison",
                "physics_weight": 1.0e-3,
                "data_only": {
                    "final_data_mse": data_only.loss(train.inputs, train.targets),
                    "held_out_data_mse": data_only.loss(
                        held_out.inputs, held_out.targets
                    ),
                    "held_out_physics_mse": physics_loss(data_only),
                    "optimizer": {
                        "initial_loss": data_fit.initial_loss,
                        "final_loss": data_fit.final_loss,
                        "success": data_fit.success,
                    },
                },
                "physics_aware": {
                    "final_data_mse": physics_aware.loss(train.inputs, train.targets),
                    "held_out_data_mse": physics_aware.loss(
                        held_out.inputs, held_out.targets
                    ),
                    "held_out_physics_mse": physics_loss(physics_aware),
                    "optimizer": {
                        "initial_total_loss": physics_fit.initial_total_loss,
                        "final_total_loss": physics_fit.final_total_loss,
                        "initial_physics_loss": physics_fit.initial_physics_loss,
                        "final_physics_loss": physics_fit.final_physics_loss,
                        "success": physics_fit.success,
                    },
                },
                "warning": "This is a one-transition residual experiment, not a full continuous-time PINO.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
