"""Matched-budget learned-operator comparison for full-SPD Darcy tensors."""

from __future__ import annotations

import json
from time import perf_counter

from neural_operator_reference import (
    TorchDeepONet2D,
    TorchFNO2D,
    TorchLocalConv2D,
    evaluate_darcy_tensor_dataset,
    generate_darcy_tensor_dataset,
    normalized_query_grid,
    trainable_parameter_count,
)


def timed_fit(callable_fit):
    start = perf_counter()
    result = callable_fit()
    return result, float(perf_counter() - start)


def tensor_family(*, samples, seed, ratio, base_angle, angle_amplitude, angle_modes):
    return generate_darcy_tensor_dataset(
        samples=samples,
        points_x=9,
        modes=2,
        seed=seed,
        log_std=0.45,
        anisotropy_ratio=ratio,
        orientation_base_radians=base_angle,
        orientation_amplitude_radians=angle_amplitude,
        orientation_modes=angle_modes,
    )


def main() -> None:
    train = tensor_family(
        samples=16,
        seed=1,
        ratio=4.0,
        base_angle=0.2,
        angle_amplitude=0.45,
        angle_modes=2,
    )
    validation = tensor_family(
        samples=4,
        seed=101,
        ratio=4.0,
        base_angle=0.2,
        angle_amplitude=0.45,
        angle_modes=2,
    )
    ood_rotation = tensor_family(
        samples=4,
        seed=201,
        ratio=4.0,
        base_angle=1.0,
        angle_amplitude=0.70,
        angle_modes=3,
    )
    ood_ratio = tensor_family(
        samples=4,
        seed=301,
        ratio=16.0,
        base_angle=0.2,
        angle_amplitude=0.45,
        angle_modes=2,
    )
    coordinates = normalized_query_grid(9, 9)

    fno = TorchFNO2D(
        in_channels=3,
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
        in_channels=3,
        hidden_width=22,
        latent_width=64,
        seed=4,
    )
    local = TorchLocalConv2D(
        in_channels=3,
        width=28,
        seed=4,
    )

    epochs = 180
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

    cohorts = {
        "validation_iid": validation,
        "ood_orientation": ood_rotation,
        "ood_principal_ratio": ood_ratio,
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
            "cohorts": {
                cohort_name: evaluate_darcy_tensor_dataset(
                    entry["model"],
                    cohort,
                    repeats=3,
                ).as_dict()
                for cohort_name, cohort in cohorts.items()
            },
        }

    report = {
        "problem": "full-SPD Darcy 2D matched-budget operator comparison",
        "input_channels": ["K_xx", "K_xy", "K_yy"],
        "grid": list(train.grid_shape),
        "training_samples": train.samples,
        "validation_samples": validation.samples,
        "ood_samples_per_family": ood_rotation.samples,
        "training_family": {
            "principal_ratio": 4.0,
            "orientation_base_radians": 0.2,
            "orientation_amplitude_radians": 0.45,
            "orientation_modes": 2,
        },
        "ood_families": {
            "orientation": {
                "principal_ratio": 4.0,
                "orientation_base_radians": 1.0,
                "orientation_amplitude_radians": 0.70,
                "orientation_modes": 3,
            },
            "principal_ratio": {
                "principal_ratio": 16.0,
                "orientation_base_radians": 0.2,
                "orientation_amplitude_radians": 0.45,
                "orientation_modes": 2,
            },
        },
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
        "metric_note": (
            "physics_mse is the Q1 finite-element algebraic residual MSE and "
            "must not be numerically compared to the scalar finite-difference residual"
        ),
        "limitations": [
            "small deterministic research dataset",
            "single 9x9 training resolution",
            "one optimizer and one training budget",
            "equal optimizer steps do not imply equal compute cost",
            "wall-clock timing is environment-specific",
            "OOD families vary orientation and principal ratio separately",
            "results characterize this protocol only and are not a universal architecture ranking",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
