"""Matched-budget FNO, DeepONet and local-convolution Darcy comparison."""

from __future__ import annotations

import json
from time import perf_counter

from neural_operator_reference import (
    TorchDeepONet2D,
    TorchFNO2D,
    TorchLocalConv2D,
    evaluate_darcy_dataset,
    generate_darcy_dataset,
    normalized_query_grid,
    trainable_parameter_count,
)


def timed_fit(callable_fit):
    start = perf_counter()
    result = callable_fit()
    return result, float(perf_counter() - start)


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
    coordinates = normalized_query_grid(9, 9)

    fno = TorchFNO2D(
        width=8,
        modes_y=4,
        modes_x=4,
        depth=2,
        padding=2,
        seed=4,
    )
    deeponet = TorchDeepONet2D(
        sensor_points_y=9,
        sensor_points_x=9,
        hidden_width=40,
        latent_width=64,
        seed=4,
    )
    local = TorchLocalConv2D(width=29, seed=4)

    epochs = 240
    learning_rate = 1.0e-2

    fno_fit, fno_seconds = timed_fit(
        lambda: fno.fit(
            train.inputs,
            train.targets,
            epochs=epochs,
            learning_rate=learning_rate,
        )
    )
    deep_fit, deep_seconds = timed_fit(
        lambda: deeponet.fit(
            train.inputs,
            coordinates,
            train.targets.reshape(train.samples, -1, 1),
            epochs=epochs,
            learning_rate=learning_rate,
        )
    )
    local_fit, local_seconds = timed_fit(
        lambda: local.fit(
            train.inputs,
            train.targets,
            epochs=epochs,
            learning_rate=learning_rate,
        )
    )

    def deeponet_operator(inputs):
        return deeponet(inputs, coordinates).reshape(
            inputs.shape[0],
            9,
            9,
            1,
        )

    models = {
        "fno2d": {
            "model": fno,
            "fit": fno_fit,
            "training_seconds": fno_seconds,
        },
        "deeponet2d": {
            "model": deeponet_operator,
            "parameter_model": deeponet,
            "fit": deep_fit,
            "training_seconds": deep_seconds,
        },
        "local_conv2d": {
            "model": local,
            "fit": local_fit,
            "training_seconds": local_seconds,
        },
    }

    report_models = {}
    parameter_counts = []
    for name, entry in models.items():
        parameter_model = entry.get("parameter_model", entry["model"])
        parameter_count = trainable_parameter_count(parameter_model)
        parameter_counts.append(parameter_count)
        fit = entry["fit"]
        report_models[name] = {
            "parameter_count": parameter_count,
            "training_seconds": entry["training_seconds"],
            "fit": {
                "initial_loss": fit.initial_loss,
                "final_loss": fit.final_loss,
                "epochs": fit.epochs,
            },
            "validation": evaluate_darcy_dataset(
                entry["model"],
                validation,
                repeats=3,
            ).as_dict(),
            "ood_coefficient_contrast": evaluate_darcy_dataset(
                entry["model"],
                ood,
                repeats=3,
            ).as_dict(),
        }

    report = {
        "problem": "Darcy 2D matched-budget operator comparison",
        "grid": list(train.grid_shape),
        "training_samples": train.samples,
        "validation_samples": validation.samples,
        "ood_samples": ood.samples,
        "optimizer": {
            "name": "Adam",
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "parameter_budget": {
            "minimum": min(parameter_counts),
            "maximum": max(parameter_counts),
            "max_over_min": max(parameter_counts) / min(parameter_counts),
        },
        "models": report_models,
        "limitations": [
            "small deterministic research dataset",
            "equal optimizer steps do not imply equal compute cost",
            "CPU wall-clock timing is environment-specific",
            "one coefficient family and one elliptic PDE",
            "results do not establish a universal architecture ranking",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
