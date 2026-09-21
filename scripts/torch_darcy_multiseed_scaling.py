"""Multi-seed and sample-scaling study for matched Darcy 2D operators."""

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
    summarize_scalars,
    trainable_parameter_count,
)


def _timed_fit(fit_callable):
    start = perf_counter()
    result = fit_callable()
    return result, float(perf_counter() - start)


def _summary(entries, key):
    return summarize_scalars(entry[key] for entry in entries).as_dict()


def _nested_summary(entries, split, key):
    return summarize_scalars(entry[split][key] for entry in entries).as_dict()


def main() -> None:
    sample_counts = [8, 16, 32]
    replicate_seeds = [2, 4, 6]
    epochs = 120
    learning_rate = 1.0e-2

    validation = generate_darcy_dataset(
        samples=8,
        points_x=9,
        modes=2,
        seed=9001,
        log_std=0.45,
    )
    ood = generate_darcy_dataset(
        samples=8,
        points_x=9,
        modes=2,
        seed=9002,
        log_std=0.85,
    )
    coordinates = normalized_query_grid(9, 9)

    raw_runs = []
    for sample_count in sample_counts:
        for replicate_seed in replicate_seeds:
            train = generate_darcy_dataset(
                samples=sample_count,
                points_x=9,
                modes=2,
                seed=10000 + 100 * sample_count + replicate_seed,
                log_std=0.45,
            )

            fno = TorchFNO2D(
                width=8,
                modes_y=4,
                modes_x=4,
                depth=2,
                padding=2,
                hard_dirichlet=True,
                seed=replicate_seed,
            )
            deep = TorchDeepONet2D(
                sensor_points_y=9,
                sensor_points_x=9,
                hidden_width=40,
                latent_width=64,
                hard_dirichlet=True,
                seed=replicate_seed,
            )
            local = TorchLocalConv2D(
                width=29,
                hard_dirichlet=True,
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
                lambda: deep.fit(
                    train.inputs,
                    coordinates,
                    train.targets.reshape(sample_count, -1, 1),
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

            def deep_operator(inputs):
                return deep(inputs, coordinates).reshape(
                    inputs.shape[0],
                    9,
                    9,
                    1,
                )

            models = {
                "fno2d": (fno, fno, fno_fit, fno_seconds),
                "deeponet2d": (deep_operator, deep, deep_fit, deep_seconds),
                "local_conv2d": (local, local, local_fit, local_seconds),
            }
            for name, (
                operator,
                parameter_model,
                fit,
                training_seconds,
            ) in models.items():
                raw_runs.append(
                    {
                        "sample_count": sample_count,
                        "replicate_seed": replicate_seed,
                        "architecture": name,
                        "parameter_count": trainable_parameter_count(
                            parameter_model
                        ),
                        "training_seconds": training_seconds,
                        "final_training_loss": fit.final_loss,
                        "validation": evaluate_darcy_dataset(
                            operator,
                            validation,
                            repeats=1,
                        ).as_dict(),
                        "ood": evaluate_darcy_dataset(
                            operator,
                            ood,
                            repeats=1,
                        ).as_dict(),
                    }
                )

    summaries = {}
    for sample_count in sample_counts:
        sample_key = str(sample_count)
        summaries[sample_key] = {}
        for architecture in ("fno2d", "deeponet2d", "local_conv2d"):
            entries = [
                entry
                for entry in raw_runs
                if entry["sample_count"] == sample_count
                and entry["architecture"] == architecture
            ]
            summaries[sample_key][architecture] = {
                "replicates": len(entries),
                "parameter_count": entries[0]["parameter_count"],
                "training_seconds": _summary(entries, "training_seconds"),
                "final_training_loss": _summary(
                    entries,
                    "final_training_loss",
                ),
                "validation_relative_l2": _nested_summary(
                    entries,
                    "validation",
                    "relative_l2",
                ),
                "validation_physics_mse": _nested_summary(
                    entries,
                    "validation",
                    "physics_mse",
                ),
                "ood_relative_l2": _nested_summary(
                    entries,
                    "ood",
                    "relative_l2",
                ),
                "ood_physics_mse": _nested_summary(
                    entries,
                    "ood",
                    "physics_mse",
                ),
            }

    report = {
        "problem": "Darcy 2D multi-seed and sample-scaling study",
        "grid": [9, 9],
        "sample_counts": sample_counts,
        "replicate_seeds": replicate_seeds,
        "validation_samples": validation.samples,
        "ood_samples": ood.samples,
        "optimizer": {
            "name": "Adam",
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "summaries": summaries,
        "raw_runs": raw_runs,
        "interpretation": [
            "each replicate changes both the training draw and model initialization",
            "validation and OOD cohorts are fixed across all runs",
            "sample means and sample standard deviations are reported",
            "the study is descriptive and does not imply statistical significance",
            "the benchmark remains limited to one elliptic PDE and coefficient family",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
