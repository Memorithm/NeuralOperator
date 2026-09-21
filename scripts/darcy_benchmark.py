"""Reproducible numerical checks for the Darcy 2D reference solver."""

from __future__ import annotations

import json

import numpy as np

from neural_operator_reference.darcy import darcy_residual_2d, solve_darcy_2d
from neural_operator_reference.darcy_datasets import generate_darcy_dataset


def analytic_convergence():
    records = []
    for points in (9, 17, 33, 65):
        axis = np.linspace(0.0, 1.0, points)
        grid_x, grid_y = np.meshgrid(axis, axis)
        exact = np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
        forcing = 2.0 * np.pi**2 * exact
        solution = solve_darcy_2d(np.ones_like(exact), forcing=forcing)
        relative_error = np.linalg.norm(solution - exact) / np.linalg.norm(exact)
        records.append(
            {
                "points": points,
                "relative_l2_error": float(relative_error),
            }
        )

    for index in range(1, len(records)):
        coarse = records[index - 1]
        fine = records[index]
        refinement = (fine["points"] - 1) / (coarse["points"] - 1)
        fine["observed_order"] = float(
            np.log(coarse["relative_l2_error"] / fine["relative_l2_error"])
            / np.log(refinement)
        )
    return records


def dataset_diagnostics():
    dataset = generate_darcy_dataset(
        samples=4,
        points_x=17,
        modes=3,
        seed=21,
        forcing=1.0,
        log_std=0.5,
    )
    residual_maxima = []
    for sample in range(dataset.samples):
        residual = darcy_residual_2d(
            dataset.targets[sample, ..., 0],
            dataset.inputs[sample, ..., 0],
            forcing=dataset.forcing,
            length_x=dataset.length_x,
            length_y=dataset.length_y,
        )
        residual_maxima.append(float(np.max(np.abs(residual))))

    return {
        "samples": dataset.samples,
        "grid_shape": list(dataset.grid_shape),
        "permeability_min": float(np.min(dataset.inputs)),
        "permeability_max": float(np.max(dataset.inputs)),
        "max_abs_discrete_residual": float(max(residual_maxima)),
    }


def main():
    report = {
        "equation": "-div(k grad u) = f",
        "boundary_condition": "homogeneous Dirichlet",
        "analytic_convergence": analytic_convergence(),
        "dataset": dataset_diagnostics(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
