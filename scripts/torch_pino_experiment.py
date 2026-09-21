#!/usr/bin/env python3
"""Compare data-only and differentiable spectral-physics PyTorch FNO training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    BurgersDatasetSpec,
    TorchFNO1D,
    evaluate_burgers_dataset,
    fit_torch_fno_burgers_physics,
    generate_burgers_split,
)


try:
    import torch
except ImportError as exc:  # pragma: no cover - script requires optional backend
    raise SystemExit(
        "PyTorch is required; install requirements-autodiff.txt before running"
    ) from exc


def _numpy_operator(model):
    def operator(inputs):
        with torch.no_grad():
            return model(inputs).detach().cpu().numpy()

    return operator


def main() -> None:
    torch.set_num_threads(1)
    common = {
        "points": 16,
        "dt": 2.0e-4,
        "steps": 2,
        "length": 2.0 * 3.141592653589793,
    }
    split = generate_burgers_split(
        BurgersDatasetSpec(
            samples=12,
            viscosity=0.05,
            seed=201,
            modes=4,
            amplitude=0.5,
            **common,
        ),
        BurgersDatasetSpec(
            samples=4,
            viscosity=0.05,
            seed=202,
            modes=4,
            amplitude=0.5,
            **common,
        ),
        BurgersDatasetSpec(
            samples=4,
            viscosity=0.10,
            seed=203,
            modes=6,
            amplitude=0.35,
            forcing_amplitude=0.2,
            forcing_mode=2,
            **common,
        ),
    )

    epochs = 120
    learning_rate = 1.0e-2
    physics_weight = 1.0e-8
    seed = 9

    data_only = TorchFNO1D(width=3, modes=5, seed=seed)
    start = perf_counter()
    data_fit = data_only.fit(
        split.train.inputs,
        split.train.targets,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    data_seconds = perf_counter() - start

    physics_aware = TorchFNO1D(width=3, modes=5, seed=seed)
    start = perf_counter()
    physics_fit = fit_torch_fno_burgers_physics(
        physics_aware,
        split.train.inputs,
        split.train.targets,
        horizon=split.train.horizon,
        viscosity=split.train.viscosity,
        data_weight=1.0,
        physics_weight=physics_weight,
        length=split.train.length,
        forcing=split.train.forcing,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    physics_seconds = perf_counter() - start

    reports = {}
    for name, model in (
        ("data_only", data_only),
        ("physics_aware", physics_aware),
    ):
        operator = _numpy_operator(model)
        reports[name] = {
            "validation": evaluate_burgers_dataset(
                operator,
                split.validation,
                repeats=20,
            ).as_dict(),
            "ood": evaluate_burgers_dataset(
                operator,
                split.ood,
                repeats=20,
            ).as_dict(),
        }

    reports["data_only"]["fit"] = {
        "initial_data_loss": data_fit.initial_loss,
        "final_data_loss": data_fit.final_loss,
        "train_seconds": float(data_seconds),
    }
    reports["physics_aware"]["fit"] = {
        "initial_total_loss": physics_fit.initial_total_loss,
        "final_total_loss": physics_fit.final_total_loss,
        "initial_data_loss": physics_fit.initial_data_loss,
        "final_data_loss": physics_fit.final_data_loss,
        "initial_physics_loss": physics_fit.initial_physics_loss,
        "final_physics_loss": physics_fit.final_physics_loss,
        "train_seconds": float(physics_seconds),
    }

    print(
        json.dumps(
            {
                "experiment": "torch_transition_pino_burgers",
                "protocol": {
                    "dataset_split": split.as_dict(),
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "physics_weight": physics_weight,
                    "model_seed": seed,
                },
                "models": reports,
                "limitations": [
                    "The temporal derivative is a finite transition quotient, not a continuous-time collocation residual.",
                    "The spatial derivatives and de-aliasing are differentiable spectral PyTorch operations.",
                    "No claim of superiority is made without repeated seeds and hyperparameter sweeps.",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
