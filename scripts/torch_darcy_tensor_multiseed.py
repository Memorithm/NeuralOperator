"""Multi-seed replication for the matched-budget full-SPD Darcy comparison."""

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
    summarize_scalars,
    trainable_parameter_count,
)


def _timed_fit(fit_callable):
    start = perf_counter()
    result = fit_callable()
    return result, float(perf_counter() - start)


def _tensor_family(
    *,
    samples: int,
    seed: int,
    ratio: float,
    base_angle: float,
    angle_amplitude: float,
    angle_modes: int,
):
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


def _summary(entries, key):
    return summarize_scalars(entry[key] for entry in entries).as_dict()


def _cohort_summary(entries, cohort, key):
    return summarize_scalars(
        entry["cohorts"][cohort][key] for entry in entries
    ).as_dict()


def _ratio_summary(entries, numerator_cohort):
    return summarize_scalars(
        entry["cohorts"][numerator_cohort]["relative_l2"]
        / entry["cohorts"]["validation_iid"]["relative_l2"]
        for entry in entries
    ).as_dict()


def main() -> None:
    replicate_seeds = [2, 4, 6]
    training_samples = 16
    evaluation_samples = 8
    epochs = 180
    learning_rate = 1.0e-2

    # Evaluation cohorts are fixed across every architecture and replicate.
    validation = _tensor_family(
        samples=evaluation_samples,
        seed=9101,
        ratio=4.0,
        base_angle=0.2,
        angle_amplitude=0.45,
        angle_modes=2,
    )
    ood_orientation = _tensor_family(
        samples=evaluation_samples,
        seed=9201,
        ratio=4.0,
        base_angle=1.0,
        angle_amplitude=0.70,
        angle_modes=3,
    )
    ood_ratio = _tensor_family(
        samples=evaluation_samples,
        seed=9301,
        ratio=16.0,
        base_angle=0.2,
        angle_amplitude=0.45,
        angle_modes=2,
    )
    cohorts = {
        "validation_iid": validation,
        "ood_orientation": ood_orientation,
        "ood_principal_ratio": ood_ratio,
    }
    coordinates = normalized_query_grid(9, 9)

    raw_runs = []
    for replicate_seed in replicate_seeds:
        training_seed = 10000 + replicate_seed
        train = _tensor_family(
            samples=training_samples,
            seed=training_seed,
            ratio=4.0,
            base_angle=0.2,
            angle_amplitude=0.45,
            angle_modes=2,
        )

        fno = TorchFNO2D(
            in_channels=3,
            width=8,
            modes_y=4,
            modes_x=4,
            depth=2,
            padding=2,
            seed=replicate_seed,
        )
        deeponet = TorchDeepONet2D(
            sensor_points_y=9,
            sensor_points_x=9,
            in_channels=3,
            hidden_width=22,
            latent_width=64,
            seed=replicate_seed,
        )
        local = TorchLocalConv2D(
            in_channels=3,
            width=28,
            seed=replicate_seed,
        )

        fno_fit, fno_seconds = _timed_fit(
            lambda: fno.fit(
                train.inputs,
                train.targets,
                epochs=epochs,
                learning_rate=learning_rate,
            )
        )
        deep_fit, deep_seconds = _timed_fit(
            lambda: deeponet.fit(
                train.inputs,
                coordinates,
                train.targets.reshape(training_samples, -1, 1),
                epochs=epochs,
                learning_rate=learning_rate,
            )
        )
        local_fit, local_seconds = _timed_fit(
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
            "fno2d": (fno, fno, fno_fit, fno_seconds),
            "deeponet2d": (
                deeponet_operator,
                deeponet,
                deep_fit,
                deep_seconds,
            ),
            "local_conv2d": (local, local, local_fit, local_seconds),
        }

        for (
            architecture,
            (operator, parameter_model, fit, training_seconds),
        ) in models.items():
            raw_runs.append(
                {
                    "architecture": architecture,
                    "replicate_seed": replicate_seed,
                    "training_dataset_seed": training_seed,
                    "parameter_count": trainable_parameter_count(
                        parameter_model
                    ),
                    "training_seconds": training_seconds,
                    "initial_training_loss": fit.initial_loss,
                    "final_training_loss": fit.final_loss,
                    "cohorts": {
                        cohort_name: evaluate_darcy_tensor_dataset(
                            operator,
                            cohort,
                            repeats=1,
                        ).as_dict()
                        for cohort_name, cohort in cohorts.items()
                    },
                }
            )

    summaries = {}
    for architecture in ("fno2d", "deeponet2d", "local_conv2d"):
        entries = [
            entry
            for entry in raw_runs
            if entry["architecture"] == architecture
        ]
        summaries[architecture] = {
            "replicates": len(entries),
            "parameter_count": entries[0]["parameter_count"],
            "training_seconds": _summary(entries, "training_seconds"),
            "initial_training_loss": _summary(
                entries,
                "initial_training_loss",
            ),
            "final_training_loss": _summary(
                entries,
                "final_training_loss",
            ),
            "validation_iid": {
                "relative_l2": _cohort_summary(
                    entries,
                    "validation_iid",
                    "relative_l2",
                ),
                "physics_mse": _cohort_summary(
                    entries,
                    "validation_iid",
                    "physics_mse",
                ),
            },
            "ood_orientation": {
                "relative_l2": _cohort_summary(
                    entries,
                    "ood_orientation",
                    "relative_l2",
                ),
                "physics_mse": _cohort_summary(
                    entries,
                    "ood_orientation",
                    "physics_mse",
                ),
                "relative_l2_over_iid": _ratio_summary(
                    entries,
                    "ood_orientation",
                ),
            },
            "ood_principal_ratio": {
                "relative_l2": _cohort_summary(
                    entries,
                    "ood_principal_ratio",
                    "relative_l2",
                ),
                "physics_mse": _cohort_summary(
                    entries,
                    "ood_principal_ratio",
                    "physics_mse",
                ),
                "relative_l2_over_iid": _ratio_summary(
                    entries,
                    "ood_principal_ratio",
                ),
            },
        }

    counts = {
        entry["architecture"]: entry["parameter_count"]
        for entry in raw_runs
        if entry["replicate_seed"] == replicate_seeds[0]
    }
    report = {
        "problem": "full-SPD Darcy 2D multi-seed matched-budget replication",
        "grid": [9, 9],
        "input_channels": ["K_xx", "K_xy", "K_yy"],
        "training_samples": training_samples,
        "evaluation_samples_per_cohort": evaluation_samples,
        "replicate_seeds": replicate_seeds,
        "training_dataset_seeds": [
            10000 + seed for seed in replicate_seeds
        ],
        "optimizer": {
            "name": "Adam",
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "parameter_budget": {
            "counts": counts,
            "minimum": min(counts.values()),
            "maximum": max(counts.values()),
            "max_over_min": max(counts.values()) / min(counts.values()),
        },
        "training_family": {
            "principal_ratio": 4.0,
            "orientation_base_radians": 0.2,
            "orientation_amplitude_radians": 0.45,
            "orientation_modes": 2,
        },
        "evaluation_families": {
            "validation_iid": {
                "seed": 9101,
                "principal_ratio": 4.0,
                "orientation_base_radians": 0.2,
                "orientation_amplitude_radians": 0.45,
                "orientation_modes": 2,
            },
            "ood_orientation": {
                "seed": 9201,
                "principal_ratio": 4.0,
                "orientation_base_radians": 1.0,
                "orientation_amplitude_radians": 0.70,
                "orientation_modes": 3,
            },
            "ood_principal_ratio": {
                "seed": 9301,
                "principal_ratio": 16.0,
                "orientation_base_radians": 0.2,
                "orientation_amplitude_radians": 0.45,
                "orientation_modes": 2,
            },
        },
        "summaries": summaries,
        "raw_runs": raw_runs,
        "interpretation": [
            (
                "each replicate changes both the training draw and model "
                "initialization using a paired seed across architectures"
            ),
            "all evaluation cohorts are fixed across architectures and replicates",
            "mean, sample standard deviation, minimum and maximum are reported",
            (
                "relative_l2_over_iid is a within-replicate descriptive "
                "generalization ratio, not a statistical effect size"
            ),
            "three replicates are exploratory and do not establish statistical significance",
            (
                "results characterize this 9x9 smooth-tensor protocol only "
                "and are not a universal architecture ranking"
            ),
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
