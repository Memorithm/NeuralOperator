#!/usr/bin/env python3
"""Train a one-transition FNO and measure recursive Burgers rollout stability."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    NumpyFNO1D,
    evaluate_burgers_operator,
    generate_burgers_dataset,
    integrate_burgers,
    random_periodic_fields,
)


def main() -> None:
    train = generate_burgers_dataset(
        samples=4,
        points=16,
        dt=2.0e-4,
        steps=3,
        viscosity=0.05,
        seed=1,
    )
    initial_fields = random_periodic_fields(
        samples=2,
        points=train.resolution,
        modes=4,
        seed=3,
        length=train.length,
    )
    horizon = train.dt * train.steps
    reference = integrate_burgers(
        initial_fields,
        dt=horizon,
        steps=5,
        viscosity=train.viscosity,
        length=train.length,
    )

    model = NumpyFNO1D(width=3, modes=5, seed=4)
    fit = model.fit(train.inputs, train.targets, maxiter=60, tolerance=1.0e-7)
    _, metrics = evaluate_burgers_operator(
        model,
        initial_fields,
        reference,
        horizon=horizon,
        viscosity=train.viscosity,
        length=train.length,
    )

    print(
        json.dumps(
            {
                "experiment": "burgers_recursive_rollout",
                "fit": {
                    "initial_loss": fit.initial_loss,
                    "final_loss": fit.final_loss,
                    "success": fit.success,
                    "iterations": fit.iterations,
                    "function_evaluations": fit.function_evaluations,
                    "message": fit.message,
                },
                "rollout": metrics.as_dict(),
                "dataset": {
                    "resolution": train.resolution,
                    "training_samples": train.samples,
                    "transition_horizon": horizon,
                    "viscosity": train.viscosity,
                    "rollout_steps": 5,
                },
                "warning": (
                    "The recursive rollout is a reference evaluation. It does not "
                    "establish production stability or continuous-time PINO validity."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
