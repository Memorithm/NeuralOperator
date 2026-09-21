"""Controlled physics-weight sweep for Darcy FNO 2D training."""

from __future__ import annotations

import json
from time import perf_counter

from neural_operator_reference import (
    TorchFNO2D,
    evaluate_darcy_dataset,
    fit_torch_fno_darcy_physics,
    generate_darcy_dataset,
    trainable_parameter_count,
)


def main() -> None:
    train = generate_darcy_dataset(
        samples=16,
        points_x=9,
        modes=2,
        seed=1,
        log_std=0.45,
    )
    validation = generate_darcy_dataset(
        samples=4,
        points_x=9,
        modes=2,
        seed=101,
        log_std=0.45,
    )
    ood = generate_darcy_dataset(
        samples=4,
        points_x=9,
        modes=2,
        seed=201,
        log_std=0.85,
    )

    physics_weights = [0.0, 1.0e-5, 1.0e-4, 3.0e-4]
    epochs = 120
    learning_rate = 1.0e-2
    runs = []

    for physics_weight in physics_weights:
        model = TorchFNO2D(
            width=8,
            modes_y=4,
            modes_x=4,
            depth=2,
            padding=2,
            hard_dirichlet=True,
            seed=4,
        )
        start = perf_counter()
        fit = fit_torch_fno_darcy_physics(
            model,
            train.inputs,
            train.targets,
            forcing=train.forcing,
            data_weight=1.0,
            physics_weight=physics_weight,
            length_x=train.length_x,
            length_y=train.length_y,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        training_seconds = float(perf_counter() - start)

        runs.append(
            {
                "physics_weight": physics_weight,
                "parameter_count": trainable_parameter_count(model),
                "training_seconds": training_seconds,
                "fit": {
                    "initial_total_loss": fit.initial_total_loss,
                    "final_total_loss": fit.final_total_loss,
                    "initial_data_loss": fit.initial_data_loss,
                    "final_data_loss": fit.final_data_loss,
                    "initial_physics_loss": fit.initial_physics_loss,
                    "final_physics_loss": fit.final_physics_loss,
                    "epochs": fit.epochs,
                },
                "validation": evaluate_darcy_dataset(
                    model,
                    validation,
                    repeats=3,
                ).as_dict(),
                "ood_coefficient_contrast": evaluate_darcy_dataset(
                    model,
                    ood,
                    repeats=3,
                ).as_dict(),
            }
        )

    report = {
        "problem": "Darcy 2D differentiable physics-weight sweep",
        "equation": "-div(k grad u) = f",
        "training_grid": list(train.grid_shape),
        "training_samples": train.samples,
        "optimizer": {
            "name": "Adam",
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "physics_weights": physics_weights,
        "runs": runs,
        "interpretation": [
            "physics_weight=0 is the same training function with no physics contribution",
            "the sweep is sensitivity evidence, not hyperparameter optimization",
            "a single initialization and dataset split cannot establish robustness",
            "raw Darcy residual scale depends on grid spacing and forcing convention",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
