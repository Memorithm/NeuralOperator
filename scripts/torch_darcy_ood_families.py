"""Broader Darcy coefficient-family OOD study for matched-budget operators."""

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


def _metric_summary(entries, family, metric):
    return summarize_scalars(
        entry["families"][family][metric] for entry in entries
    ).as_dict()


def main() -> None:
    replicate_seeds = [2, 4, 6]
    epochs = 120
    learning_rate = 1.0e-2
    train_samples = 16

    family_specs = {
        "iid": {
            "seed": 9100,
            "modes": 2,
            "spectral_decay": 2.0,
            "log_std": 0.45,
            "mean_log_permeability": 0.0,
        },
        "contrast": {
            "seed": 9101,
            "modes": 2,
            "spectral_decay": 2.0,
            "log_std": 0.85,
            "mean_log_permeability": 0.0,
        },
        "rough_high_frequency": {
            "seed": 9102,
            "modes": 3,
            "spectral_decay": 0.5,
            "log_std": 0.45,
            "mean_log_permeability": 0.0,
        },
        "smooth_spectrum": {
            "seed": 9103,
            "modes": 2,
            "spectral_decay": 4.0,
            "log_std": 0.45,
            "mean_log_permeability": 0.0,
        },
        "mean_permeability_high": {
            "seed": 9104,
            "modes": 2,
            "spectral_decay": 2.0,
            "log_std": 0.45,
            "mean_log_permeability": 0.7,
        },
        "mean_permeability_low": {
            "seed": 9105,
            "modes": 2,
            "spectral_decay": 2.0,
            "log_std": 0.45,
            "mean_log_permeability": -0.7,
        },
    }
    families = {
        name: generate_darcy_dataset(
            samples=8,
            points_x=9,
            modes=spec["modes"],
            seed=spec["seed"],
            log_std=spec["log_std"],
            mean_log_permeability=spec["mean_log_permeability"],
            spectral_decay=spec["spectral_decay"],
        )
        for name, spec in family_specs.items()
    }
    coordinates = normalized_query_grid(9, 9)

    raw_runs = []
    for replicate_seed in replicate_seeds:
        train = generate_darcy_dataset(
            samples=train_samples,
            points_x=9,
            modes=2,
            seed=12000 + replicate_seed,
            log_std=0.45,
            mean_log_permeability=0.0,
            spectral_decay=2.0,
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
                train.targets.reshape(train_samples, -1, 1),
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
            return deep(inputs, coordinates).reshape(inputs.shape[0], 9, 9, 1)

        models = {
            "fno2d": (fno, fno, fno_fit, fno_seconds),
            "deeponet2d": (deep_operator, deep, deep_fit, deep_seconds),
            "local_conv2d": (local, local, local_fit, local_seconds),
        }
        for architecture, (operator, parameter_model, fit, training_seconds) in models.items():
            raw_runs.append(
                {
                    "replicate_seed": replicate_seed,
                    "architecture": architecture,
                    "parameter_count": trainable_parameter_count(parameter_model),
                    "training_seconds": training_seconds,
                    "final_training_loss": fit.final_loss,
                    "families": {
                        family: evaluate_darcy_dataset(
                            operator,
                            dataset,
                            repeats=1,
                        ).as_dict()
                        for family, dataset in families.items()
                    },
                }
            )

    summaries = {}
    for architecture in ("fno2d", "deeponet2d", "local_conv2d"):
        entries = [
            entry for entry in raw_runs if entry["architecture"] == architecture
        ]
        summaries[architecture] = {
            "replicates": len(entries),
            "parameter_count": entries[0]["parameter_count"],
            "training_seconds": summarize_scalars(
                entry["training_seconds"] for entry in entries
            ).as_dict(),
            "families": {
                family: {
                    "relative_l2": _metric_summary(
                        entries,
                        family,
                        "relative_l2",
                    ),
                    "physics_mse": _metric_summary(
                        entries,
                        family,
                        "physics_mse",
                    ),
                    "boundary_max_abs": _metric_summary(
                        entries,
                        family,
                        "boundary_max_abs",
                    ),
                }
                for family in family_specs
            },
        }

    report = {
        "problem": "Darcy 2D coefficient-family OOD study",
        "training_family": {
            "grid": [9, 9],
            "samples": train_samples,
            "modes": 2,
            "spectral_decay": 2.0,
            "log_std": 0.45,
            "mean_log_permeability": 0.0,
        },
        "replicate_seeds": replicate_seeds,
        "optimizer": {
            "name": "Adam",
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "family_specs": family_specs,
        "summaries": summaries,
        "raw_runs": raw_runs,
        "interpretation": [
            "IID and OOD cohorts are fixed across model replicates",
            "roughness shift changes spectral decay and available coefficient modes",
            "mean permeability shifts change the coefficient scale without changing forcing",
            "three replicates are descriptive evidence rather than a significance test",
            "results remain specific to the current Darcy generator and 9x9 grid",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
