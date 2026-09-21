#!/usr/bin/env python3
"""Run a controlled PyTorch FNO versus DeepONet Burgers comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    BurgersDatasetSpec,
    TorchDeepONet1D,
    TorchFNO1D,
    evaluate_burgers_dataset,
    evaluate_burgers_operator,
    generate_burgers_split,
    integrate_burgers,
    trainable_parameter_count,
)


try:
    import torch
except ImportError as exc:  # pragma: no cover - script requires optional backend
    raise SystemExit(
        "PyTorch is required; install requirements-autodiff.txt before running"
    ) from exc


def _numpy_operator(model, coordinates=None):
    def operator(inputs):
        with torch.no_grad():
            if coordinates is None:
                output = model(inputs)
            else:
                output = model(inputs, coordinates)
        return output.detach().cpu().numpy()

    return operator


def _rollout_report(operator, dataset, rollout_steps):
    reference = integrate_burgers(
        dataset.inputs[..., 0],
        dt=dataset.horizon,
        steps=rollout_steps,
        viscosity=dataset.viscosity,
        length=dataset.length,
        forcing=dataset.forcing,
    )
    _, metrics = evaluate_burgers_operator(
        operator,
        dataset.inputs[..., 0],
        reference,
        horizon=dataset.horizon,
        viscosity=dataset.viscosity,
        length=dataset.length,
        forcing=dataset.forcing,
    )
    return metrics.as_dict()


def main() -> None:
    torch.set_num_threads(1)
    common = {
        "points": 16,
        "dt": 2.0e-4,
        "steps": 3,
        "length": 2.0 * np.pi,
    }
    split = generate_burgers_split(
        BurgersDatasetSpec(
            samples=12,
            viscosity=0.05,
            seed=101,
            modes=4,
            amplitude=0.5,
            **common,
        ),
        BurgersDatasetSpec(
            samples=4,
            viscosity=0.05,
            seed=102,
            modes=4,
            amplitude=0.5,
            **common,
        ),
        BurgersDatasetSpec(
            samples=4,
            viscosity=0.10,
            seed=103,
            modes=6,
            amplitude=0.35,
            forcing_amplitude=0.2,
            forcing_mode=2,
            **common,
        ),
    )
    coordinates = np.arange(common["points"], dtype=float) * common["length"] / common["points"]
    epochs = 120
    learning_rate = 1.0e-2
    rollout_steps = 5
    inference_repeats = 20

    fno = TorchFNO1D(width=3, modes=5, seed=4)
    deeponet = TorchDeepONet1D(
        sensor_points=common["points"],
        hidden_width=3,
        latent_width=1,
        seed=4,
    )
    parameter_budget = {
        "fno": trainable_parameter_count(fno),
        "deeponet": trainable_parameter_count(deeponet),
    }

    train_start = perf_counter()
    fno_fit = fno.fit(
        split.train.inputs,
        split.train.targets,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    fno_train_seconds = perf_counter() - train_start

    train_start = perf_counter()
    deeponet_fit = deeponet.fit(
        split.train.inputs,
        coordinates,
        split.train.targets,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    deeponet_train_seconds = perf_counter() - train_start

    operators = {
        "fno": _numpy_operator(fno),
        "deeponet": _numpy_operator(deeponet, coordinates),
    }
    fits = {
        "fno": {
            "initial_loss": fno_fit.initial_loss,
            "final_loss": fno_fit.final_loss,
            "train_seconds": float(fno_train_seconds),
        },
        "deeponet": {
            "initial_loss": deeponet_fit.initial_loss,
            "final_loss": deeponet_fit.final_loss,
            "train_seconds": float(deeponet_train_seconds),
        },
    }

    models = {}
    for name, operator in operators.items():
        models[name] = {
            "parameters": parameter_budget[name],
            "fit": fits[name],
            "validation": evaluate_burgers_dataset(
                operator,
                split.validation,
                repeats=inference_repeats,
            ).as_dict(),
            "ood": evaluate_burgers_dataset(
                operator,
                split.ood,
                repeats=inference_repeats,
            ).as_dict(),
            "validation_rollout": _rollout_report(
                operator,
                split.validation,
                rollout_steps,
            ),
            "ood_rollout": _rollout_report(
                operator,
                split.ood,
                rollout_steps,
            ),
        }

    print(
        json.dumps(
            {
                "experiment": "controlled_torch_fno_vs_deeponet_burgers",
                "protocol": {
                    "dataset_split": split.as_dict(),
                    "optimizer": "Adam",
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "model_seed": 4,
                    "rollout_steps": rollout_steps,
                    "inference_repeats": inference_repeats,
                    "parameter_budget_difference": abs(
                        parameter_budget["fno"] - parameter_budget["deeponet"]
                    ),
                },
                "models": models,
                "limitations": [
                    "This is a small CPU reference benchmark, not a production throughput claim.",
                    "Identical optimizer settings do not constitute per-model hyperparameter tuning.",
                    "DeepONet uses fixed 16-point branch sensors; arbitrary trunk query resolution is a separate protocol.",
                    "Wall-clock timings are comparable only within the same runner and process.",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
