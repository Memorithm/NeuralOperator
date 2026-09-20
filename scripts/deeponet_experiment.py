#!/usr/bin/env python3
"""Validate a separable DeepONet-style operator on a known 1D map."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    LinearDeepONet1D,
    random_periodic_fields,
    relative_l2_error,
)


def main() -> None:
    sensor_points = 16
    modes = 3
    inputs = random_periodic_fields(
        samples=32, points=sensor_points, modes=4, seed=7
    )[..., None]
    train_inputs, held_out_inputs = inputs[:24], inputs[24:]

    teacher = LinearDeepONet1D(sensor_points=sensor_points, modes=modes, length=2.0 * np.pi)
    rng = np.random.default_rng(9)
    teacher.branch_weight[...] = rng.normal(size=teacher.branch_weight.shape)
    teacher.branch_bias[...] = rng.normal(size=teacher.branch_bias.shape)
    coarse_coordinates = np.arange(sensor_points) * 2.0 * np.pi / sensor_points
    fine_coordinates = np.arange(32) * 2.0 * np.pi / 32
    train_targets = teacher(train_inputs, coordinates=coarse_coordinates)
    held_out_targets = teacher(held_out_inputs, coordinates=coarse_coordinates)
    fine_targets = teacher(held_out_inputs, coordinates=fine_coordinates)

    model = LinearDeepONet1D(sensor_points=sensor_points, modes=modes)
    fit = model.fit(train_inputs, train_targets, coordinates=coarse_coordinates)
    held_out_prediction = model(held_out_inputs, coordinates=coarse_coordinates)
    fine_prediction = model(held_out_inputs, coordinates=fine_coordinates)

    print(
        json.dumps(
            {
                "experiment": "linear_deeponet_coordinate_transfer",
                "fit": {
                    "initial_loss": fit.initial_loss,
                    "final_loss": fit.final_loss,
                    "rank": fit.rank,
                },
                "held_out_coarse": {
                    "mse": float(np.mean((held_out_prediction - held_out_targets) ** 2)),
                    "relative_l2": relative_l2_error(
                        held_out_prediction, held_out_targets
                    ),
                },
                "held_out_fine": {
                    "mse": float(np.mean((fine_prediction - fine_targets) ** 2)),
                    "relative_l2": relative_l2_error(fine_prediction, fine_targets),
                },
                "note": "The trunk is a fixed Fourier basis and the branch is linear; this is a reference contract, not a nonlinear production DeepONet.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
