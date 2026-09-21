"""Resolution sensitivity study for Darcy PINO physics weighting."""

from __future__ import annotations

import json
from time import perf_counter

from neural_operator_reference import (
    TorchFNO2D,
    evaluate_darcy_dataset,
    fit_torch_fno_darcy_physics,
    generate_darcy_dataset,
    summarize_scalars,
)


def _summary(entries, key):
    return summarize_scalars(entry[key] for entry in entries).as_dict()


def _metric_summary(entries, key):
    return summarize_scalars(
        entry["validation"][key] for entry in entries
    ).as_dict()


def main() -> None:
    grid_sizes = [9, 17]
    physics_weights = [0.0, 1.0e-5, 1.0e-4, 3.0e-4]
    replicate_seeds = [2, 4]
    epochs = 80
    learning_rate = 1.0e-2
    train_samples = 12

    raw_runs = []
    for points in grid_sizes:
        validation = generate_darcy_dataset(
            samples=6,
            points_x=points,
            modes=2,
            seed=15000,
            log_std=0.45,
            spectral_decay=2.0,
        )
        for replicate_seed in replicate_seeds:
            train = generate_darcy_dataset(
                samples=train_samples,
                points_x=points,
                modes=2,
                seed=14000 + replicate_seed,
                log_std=0.45,
                spectral_decay=2.0,
            )
            for physics_weight in physics_weights:
                model = TorchFNO2D(
                    width=8,
                    modes_y=4,
                    modes_x=4,
                    depth=2,
                    padding=2,
                    hard_dirichlet=True,
                    seed=replicate_seed,
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
                validation_metrics = evaluate_darcy_dataset(
                    model,
                    validation,
                    repeats=1,
                ).as_dict()
                raw_runs.append(
                    {
                        "grid_points": points,
                        "replicate_seed": replicate_seed,
                        "physics_weight": physics_weight,
                        "training_seconds": training_seconds,
                        "final_total_loss": fit.final_total_loss,
                        "final_data_loss": fit.final_data_loss,
                        "final_physics_loss": fit.final_physics_loss,
                        "validation": validation_metrics,
                    }
                )

    summaries = {}
    for points in grid_sizes:
        grid_key = str(points)
        summaries[grid_key] = {}
        for physics_weight in physics_weights:
            entries = [
                entry
                for entry in raw_runs
                if entry["grid_points"] == points
                and entry["physics_weight"] == physics_weight
            ]
            summaries[grid_key][str(physics_weight)] = {
                "replicates": len(entries),
                "training_seconds": _summary(entries, "training_seconds"),
                "final_data_loss": _summary(entries, "final_data_loss"),
                "final_physics_loss": _summary(entries, "final_physics_loss"),
                "validation_relative_l2": _metric_summary(
                    entries,
                    "relative_l2",
                ),
                "validation_physics_mse": _metric_summary(
                    entries,
                    "physics_mse",
                ),
            }

    report = {
        "problem": "Darcy PINO resolution-sensitive physics-weight study",
        "grid_sizes": grid_sizes,
        "physics_weights": physics_weights,
        "replicate_seeds": replicate_seeds,
        "training_samples": train_samples,
        "validation_samples": 6,
        "optimizer": {
            "name": "Adam",
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "summaries": summaries,
        "raw_runs": raw_runs,
        "interpretation": [
            "model architecture and low-mode budget are held fixed across grids",
            "the same physics weights are evaluated independently at each resolution",
            "paired seeds preserve optimizer initialization across weight sweeps",
            "coefficient-generator seeds are shared across resolutions but grid normalization is resolution-dependent",
            "the study measures sensitivity and does not define an automatic weight-selection rule",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
