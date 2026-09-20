#!/usr/bin/env python3
"""Train the optional PyTorch FNO on the same Burgers contract as the oracle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the host
    raise SystemExit(
        "PyTorch is not installed; run `python3 -m pip install "
        "-r requirements-autodiff.txt` first."
    ) from exc

from neural_operator_reference import (  # noqa: E402
    burgers_transition_physics_loss,
    generate_burgers_dataset,
    relative_l2_error,
)
from neural_operator_reference.torch_backend import TorchFNO1D  # noqa: E402


def _numpy_prediction(model: TorchFNO1D, fields: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(fields).detach().cpu().numpy()


def main() -> None:
    train = generate_burgers_dataset(
        samples=4, points=16, dt=2.0e-4, steps=3, viscosity=0.05, seed=1
    )
    held_out = generate_burgers_dataset(
        samples=2, points=16, dt=2.0e-4, steps=3, viscosity=0.05, seed=2
    )
    model = TorchFNO1D(width=3, modes=5, seed=4)
    fit = model.fit(
        train.inputs,
        train.targets,
        epochs=120,
        learning_rate=2.0e-2,
    )
    train_prediction = _numpy_prediction(model, train.inputs)
    held_out_prediction = _numpy_prediction(model, held_out.inputs)
    print(
        json.dumps(
            {
                "experiment": "burgers_fno_torch_autodiff",
                "fit": {
                    "initial_loss": fit.initial_loss,
                    "final_loss": fit.final_loss,
                    "epochs": fit.epochs,
                    "converged": fit.converged,
                    "device": fit.device,
                },
                "mse": {
                    "train": float(np.mean((train_prediction - train.targets) ** 2)),
                    "held_out": float(
                        np.mean((held_out_prediction - held_out.targets) ** 2)
                    ),
                },
                "relative_l2": {
                    "held_out": relative_l2_error(
                        held_out_prediction, held_out.targets
                    )
                },
                "physics_mse": {
                    "held_out": burgers_transition_physics_loss(
                        held_out.inputs[..., 0],
                        held_out_prediction[..., 0],
                        horizon=held_out.dt * held_out.steps,
                        viscosity=held_out.viscosity,
                        length=held_out.length,
                    )
                },
                "dataset": {
                    "dt": train.dt,
                    "steps": train.steps,
                    "viscosity": train.viscosity,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
