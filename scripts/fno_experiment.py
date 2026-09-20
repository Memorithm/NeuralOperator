#!/usr/bin/env python3
"""Train the tiny FNO oracle and evaluate it at two resolutions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import NumpyFNO1D, relative_l2_error  # noqa: E402


def make_dataset(n: int, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.arange(n) * 2.0 * np.pi / n
    inputs, targets = [], []
    for _ in range(samples):
        field = (
            rng.normal() * np.sin(x)
            + rng.normal() * np.cos(x)
            + rng.normal() * np.sin(2.0 * x)
        )
        spectrum = np.fft.rfft(field)
        response = np.zeros_like(spectrum, dtype=float)
        response[:3] = (0.2, 0.5, 0.1)
        inputs.append(field[:, None])
        targets.append(np.fft.irfft(spectrum * response, n=n)[:, None])
    return np.asarray(inputs), np.asarray(targets)


def main() -> None:
    train_inputs, train_targets = make_dataset(12, 4, seed=20260920)
    held_inputs, held_targets = make_dataset(12, 4, seed=20260921)
    fine_inputs, fine_targets = make_dataset(24, 4, seed=20260922)

    model = NumpyFNO1D(width=3, modes=4, seed=4)
    fit = model.fit(train_inputs, train_targets, maxiter=150, tolerance=1.0e-8)
    result = {
        "experiment": "numpy_fno_low_mode_operator",
        "fit": {
            "initial_loss": fit.initial_loss,
            "final_loss": fit.final_loss,
            "success": fit.success,
            "iterations": fit.iterations,
            "function_evaluations": fit.function_evaluations,
            "message": fit.message,
        },
        "mse": {
            "train_resolution_12": model.loss(train_inputs, train_targets),
            "held_out_resolution_12": model.loss(held_inputs, held_targets),
            "zero_shot_resolution_24": model.loss(fine_inputs, fine_targets),
        },
        "relative_l2": {
            "held_out_resolution_12": relative_l2_error(
                model(held_inputs), held_targets
            ),
            "zero_shot_resolution_24": relative_l2_error(
                model(fine_inputs), fine_targets
            ),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
