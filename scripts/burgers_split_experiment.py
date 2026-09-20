#!/usr/bin/env python3
"""Compare validation and OOD Burgers rollouts under one fixed protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    BurgersDatasetSpec,
    NumpyFNO1D,
    evaluate_burgers_operator,
    generate_burgers_split,
    integrate_burgers,
)


def main() -> None:
    common = {
        "points": 16,
        "dt": 2.0e-4,
        "steps": 3,
        "length": 2.0 * 3.141592653589793,
    }
    train_spec = BurgersDatasetSpec(
        samples=4,
        viscosity=0.05,
        seed=1,
        modes=4,
        amplitude=0.5,
        **common,
    )
    validation_spec = BurgersDatasetSpec(
        samples=2,
        viscosity=0.05,
        seed=2,
        modes=4,
        amplitude=0.5,
        **common,
    )
    ood_spec = BurgersDatasetSpec(
        samples=2,
        viscosity=0.1,
        seed=3,
        modes=6,
        amplitude=0.35,
        **common,
    )
    split = generate_burgers_split(train_spec, validation_spec, ood_spec)

    model = NumpyFNO1D(width=3, modes=5, seed=4)
    fit = model.fit(
        split.train.inputs,
        split.train.targets,
        maxiter=60,
        tolerance=1.0e-7,
    )

    reports = {}
    for name, dataset in (
        ("validation", split.validation),
        ("ood", split.ood),
    ):
        reference = integrate_burgers(
            dataset.inputs[..., 0],
            dt=dataset.horizon,
            steps=5,
            viscosity=dataset.viscosity,
            length=dataset.length,
        )
        _, metrics = evaluate_burgers_operator(
            model,
            dataset.inputs[..., 0],
            reference,
            horizon=dataset.horizon,
            viscosity=dataset.viscosity,
            length=dataset.length,
        )
        reports[name] = {
            "spec": split.as_dict()[name],
            "rollout": metrics.as_dict(),
        }

    print(
        json.dumps(
            {
                "experiment": "burgers_train_validation_ood",
                "fit": {
                    "initial_loss": fit.initial_loss,
                    "final_loss": fit.final_loss,
                    "success": fit.success,
                },
                "partitions": reports,
                "limitation": (
                    "This protocol varies initial-condition statistics and viscosity; "
                    "forcing and larger PDE families remain future work."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
