"""Darcy global-scale equivariance benchmark for matched-budget operators."""

from __future__ import annotations

import json
from time import perf_counter

from neural_operator_reference import (
    TorchDeepONet2D,
    TorchFNO2D,
    TorchLocalConv2D,
    evaluate_darcy_dataset,
    generate_darcy_dataset,
    normalize_darcy_dataset_scale,
    normalize_darcy_permeability_scale,
    normalized_query_grid,
    restore_darcy_pressure_scale,
    summarize_scalars,
    trainable_parameter_count,
)


def _to_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return values


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
    coordinates = normalized_query_grid(9, 9)

    families = {
        "iid": generate_darcy_dataset(
            samples=8,
            points_x=9,
            modes=2,
            seed=16000,
            log_std=0.45,
            mean_log_permeability=0.0,
        ),
        "mean_high": generate_darcy_dataset(
            samples=8,
            points_x=9,
            modes=2,
            seed=16001,
            log_std=0.45,
            mean_log_permeability=0.7,
        ),
        "mean_low": generate_darcy_dataset(
            samples=8,
            points_x=9,
            modes=2,
            seed=16002,
            log_std=0.45,
            mean_log_permeability=-0.7,
        ),
        "mean_high_extreme": generate_darcy_dataset(
            samples=8,
            points_x=9,
            modes=2,
            seed=16003,
            log_std=0.45,
            mean_log_permeability=1.2,
        ),
        "mean_low_extreme": generate_darcy_dataset(
            samples=8,
            points_x=9,
            modes=2,
            seed=16004,
            log_std=0.45,
            mean_log_permeability=-1.2,
        ),
    }

    raw_runs = []
    for seed in replicate_seeds:
        train = generate_darcy_dataset(
            samples=train_samples,
            points_x=9,
            modes=2,
            seed=17000 + seed,
            log_std=0.45,
            mean_log_permeability=0.0,
        )
        normalized_inputs, normalized_targets, _ = normalize_darcy_dataset_scale(
            train
        )

        for mode in ("raw", "scale_equivariant"):
            use_scale = mode == "scale_equivariant"
            fit_inputs = normalized_inputs if use_scale else train.inputs
            fit_targets = normalized_targets if use_scale else train.targets

            fno = TorchFNO2D(
                width=8,
                modes_y=4,
                modes_x=4,
                depth=2,
                padding=2,
                hard_dirichlet=True,
                seed=seed,
            )
            deep = TorchDeepONet2D(
                sensor_points_y=9,
                sensor_points_x=9,
                hidden_width=40,
                latent_width=64,
                hard_dirichlet=True,
                seed=seed,
            )
            local = TorchLocalConv2D(
                width=29,
                hard_dirichlet=True,
                seed=seed,
            )

            fno_fit, fno_seconds = _timed_fit(
                lambda: fno.fit(
                    fit_inputs,
                    fit_targets,
                    epochs=epochs,
                    learning_rate=learning_rate,
                )
            )
            deep_fit, deep_seconds = _timed_fit(
                lambda: deep.fit(
                    fit_inputs,
                    coordinates,
                    fit_targets.reshape(train_samples, -1, 1),
                    epochs=epochs,
                    learning_rate=learning_rate,
                )
            )
            local_fit, local_seconds = _timed_fit(
                lambda: local.fit(
                    fit_inputs,
                    fit_targets,
                    epochs=epochs,
                    learning_rate=learning_rate,
                )
            )

            def raw_deep_operator(inputs):
                return deep(inputs, coordinates).reshape(inputs.shape[0], 9, 9, 1)

            def scale_wrapper(grid_operator):
                def operator(inputs):
                    normalized, scales = normalize_darcy_permeability_scale(inputs)
                    prediction = _to_numpy(grid_operator(normalized))
                    return restore_darcy_pressure_scale(prediction, scales)
                return operator

            def normalized_deep_grid(inputs):
                return deep(inputs, coordinates).reshape(inputs.shape[0], 9, 9, 1)

            if use_scale:
                fno_operator = scale_wrapper(fno)
                deep_operator = scale_wrapper(normalized_deep_grid)
                local_operator = scale_wrapper(local)
            else:
                fno_operator = fno
                deep_operator = raw_deep_operator
                local_operator = local

            models = {
                "fno2d": (fno_operator, fno, fno_fit, fno_seconds),
                "deeponet2d": (deep_operator, deep, deep_fit, deep_seconds),
                "local_conv2d": (local_operator, local, local_fit, local_seconds),
            }
            for architecture, (
                operator,
                parameter_model,
                fit,
                training_seconds,
            ) in models.items():
                raw_runs.append(
                    {
                        "replicate_seed": seed,
                        "mode": mode,
                        "architecture": architecture,
                        "parameter_count": trainable_parameter_count(
                            parameter_model
                        ),
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
        summaries[architecture] = {}
        for mode in ("raw", "scale_equivariant"):
            entries = [
                entry
                for entry in raw_runs
                if entry["architecture"] == architecture and entry["mode"] == mode
            ]
            summaries[architecture][mode] = {
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
                    }
                    for family in families
                },
            }

    report = {
        "problem": "Darcy exact global permeability-scale equivariance",
        "identity": "k = g*k_hat implies u_hat = g*u for fixed forcing",
        "training_family": {
            "samples": train_samples,
            "grid": [9, 9],
            "mean_log_permeability": 0.0,
            "log_std": 0.45,
        },
        "replicate_seeds": replicate_seeds,
        "optimizer": {
            "name": "Adam",
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "summaries": summaries,
        "raw_runs": raw_runs,
        "interpretation": [
            "the scale transform has no trainable parameters",
            "raw and scale-equivariant models use identical architecture seeds",
            "scale-equivariant training removes only the per-sample global geometric-mean coefficient scale",
            "the inverse pressure scaling is exact for scalar global coefficient multiplication",
            "results remain specific to the current shape distribution and Darcy boundary/forcing convention",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
