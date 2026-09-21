"""Reproducible diagnostics for diagonal anisotropic Darcy flow."""

from __future__ import annotations

import json

import numpy as np

from neural_operator_reference import (
    darcy_diagonal_residual_2d,
    generate_darcy_diagonal_dataset,
    solve_darcy_diagonal_2d,
)


def analytic_convergence():
    records = []
    kx_value = 2.0
    ky_value = 0.5
    for points in (9, 17, 33, 65):
        axis = np.linspace(0.0, 1.0, points)
        grid_x, grid_y = np.meshgrid(axis, axis)
        exact = np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
        forcing = np.pi**2 * (kx_value + ky_value) * exact
        solution = solve_darcy_diagonal_2d(
            np.full_like(exact, kx_value),
            np.full_like(exact, ky_value),
            forcing=forcing,
        )
        error = np.linalg.norm(solution - exact) / np.linalg.norm(exact)
        records.append(
            {
                "points": points,
                "relative_l2_error": float(error),
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


def ratio_diagnostics():
    records = []
    for ratio in (0.25, 1.0, 4.0):
        dataset = generate_darcy_diagonal_dataset(
            samples=3,
            points_x=17,
            modes=3,
            seed=91,
            log_std=0.45,
            anisotropy_ratio=ratio,
        )
        residual_maxima = []
        for index in range(dataset.samples):
            residual = darcy_diagonal_residual_2d(
                dataset.targets[index, ..., 0],
                dataset.inputs[index, ..., 0],
                dataset.inputs[index, ..., 1],
                forcing=dataset.forcing,
            )
            residual_maxima.append(float(np.max(np.abs(residual))))
        records.append(
            {
                "anisotropy_ratio": ratio,
                "samples": dataset.samples,
                "grid_shape": list(dataset.grid_shape),
                "pressure_rms": float(
                    np.sqrt(np.mean(dataset.targets**2))
                ),
                "max_abs_discrete_residual": max(residual_maxima),
            }
        )
    return records


def main():
    report = {
        "equation": "-d_x(k_x d_x u)-d_y(k_y d_y u)=f",
        "boundary_condition": "homogeneous Dirichlet",
        "analytic_constant_anisotropy": {
            "k_x": 2.0,
            "k_y": 0.5,
            "convergence": analytic_convergence(),
        },
        "ratio_families": ratio_diagnostics(),
        "limitations": [
            "diagonal tensor only",
            "anisotropy ratio is sample-wide and spatially constant",
            "principal axes are aligned with the numerical grid",
            "no learned operator is evaluated in this tranche",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
