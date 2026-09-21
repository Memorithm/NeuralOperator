"""Controlled Darcy 2D FNO training and generalization benchmark."""

from __future__ import annotations

import json

from neural_operator_reference import (
    TorchFNO2D,
    evaluate_darcy_dataset,
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
    ood_contrast = generate_darcy_dataset(
        samples=4,
        points_x=9,
        modes=2,
        seed=201,
        log_std=0.85,
    )
    fine_resolution = generate_darcy_dataset(
        samples=4,
        points_x=17,
        modes=2,
        seed=101,
        log_std=0.45,
    )

    model = TorchFNO2D(
        width=8,
        modes_y=4,
        modes_x=4,
        depth=2,
        padding=2,
        hard_dirichlet=True,
        seed=4,
    )
    fit = model.fit(
        train.inputs,
        train.targets,
        epochs=60,
        learning_rate=1.0e-2,
    )

    report = {
        "problem": "Darcy 2D: -div(k grad u) = f",
        "boundary_condition": "homogeneous Dirichlet, imposed architecturally",
        "train_grid": list(train.grid_shape),
        "parameter_count": trainable_parameter_count(model),
        "fit": {
            "initial_loss": fit.initial_loss,
            "final_loss": fit.final_loss,
            "epochs": fit.epochs,
        },
        "validation": evaluate_darcy_dataset(
            model,
            validation,
            repeats=3,
        ).as_dict(),
        "ood_coefficient_contrast": {
            "train_log_std": train.log_std,
            "evaluation_log_std": ood_contrast.log_std,
            "metrics": evaluate_darcy_dataset(
                model,
                ood_contrast,
                repeats=3,
            ).as_dict(),
        },
        "zero_shot_resolution": {
            "train_grid": list(train.grid_shape),
            "evaluation_grid": list(fine_resolution.grid_shape),
            "metrics": evaluate_darcy_dataset(
                model,
                fine_resolution,
                repeats=3,
            ).as_dict(),
        },
        "limitations": [
            "small deterministic research dataset",
            "CPU timing is environment-specific",
            "no matched-budget DeepONet or local baseline in this tranche",
            "no production-solver speedup claim",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
