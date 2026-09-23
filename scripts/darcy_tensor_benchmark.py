"""Reproducible diagnostics for full SPD tensor Darcy flow."""

from __future__ import annotations

import json

import numpy as np

from neural_operator_reference import (
    darcy_tensor_residual_2d,
    generate_darcy_tensor_dataset,
    solve_darcy_tensor_2d,
)


def rotated_constant_convergence():
    records = []
    kxx_value = 2.0
    kxy_value = 0.6
    kyy_value = 1.0
    for points in (9, 17, 33, 65):
        axis = np.linspace(0.0, 1.0, points)
        grid_x, grid_y = np.meshgrid(axis, axis)
        exact = np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
        forcing = (
            np.pi**2 * (kxx_value + kyy_value) * exact
            - 2.0
            * kxy_value
            * np.pi**2
            * np.cos(np.pi * grid_x)
            * np.cos(np.pi * grid_y)
        )
        kxx = np.full_like(exact, kxx_value)
        kxy = np.full_like(exact, kxy_value)
        kyy = np.full_like(exact, kyy_value)
        solution = solve_darcy_tensor_2d(kxx, kxy, kyy, forcing=forcing)
        residual = darcy_tensor_residual_2d(
            solution,
            kxx,
            kxy,
            kyy,
            forcing=forcing,
        )
        error = np.linalg.norm(solution - exact) / np.linalg.norm(exact)
        records.append(
            {
                "points": points,
                "relative_l2_error": float(error),
                "max_abs_algebraic_residual": float(np.max(np.abs(residual))),
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


def spatial_tensor_diagnostics():
    records = []
    for ratio in (1.0, 4.0, 16.0):
        dataset = generate_darcy_tensor_dataset(
            samples=2,
            points_x=17,
            modes=3,
            seed=109,
            log_std=0.45,
            anisotropy_ratio=ratio,
            orientation_base_radians=0.35,
            orientation_amplitude_radians=0.6,
            orientation_modes=2,
        )
        kxx = dataset.inputs[..., 0]
        kxy = dataset.inputs[..., 1]
        kyy = dataset.inputs[..., 2]
        trace = kxx + kyy
        discriminant = np.sqrt((kxx - kyy) ** 2 + 4.0 * kxy * kxy)
        eigenvalue_max = 0.5 * (trace + discriminant)
        eigenvalue_min = 0.5 * (trace - discriminant)
        expected_ratio = max(ratio, 1.0 / ratio)

        residual_maxima = []
        for index in range(dataset.samples):
            residual = darcy_tensor_residual_2d(
                dataset.targets[index, ..., 0],
                dataset.inputs[index, ..., 0],
                dataset.inputs[index, ..., 1],
                dataset.inputs[index, ..., 2],
                forcing=dataset.forcing,
                length_x=dataset.length_x,
                length_y=dataset.length_y,
            )
            residual_maxima.append(float(np.max(np.abs(residual))))

        records.append(
            {
                "principal_ratio": ratio,
                "max_abs_eigenvalue_ratio_error": float(
                    np.max(np.abs(eigenvalue_max / eigenvalue_min - expected_ratio))
                ),
                "min_tensor_determinant": float(np.min(kxx * kyy - kxy * kxy)),
                "cross_term_std": float(np.std(kxy)),
                "pressure_rms": float(np.sqrt(np.mean(dataset.targets**2))),
                "max_abs_algebraic_residual": max(residual_maxima),
            }
        )
    return records


def main():
    report = {
        "equation": "-div(K grad(u))=f",
        "tensor": "symmetric positive definite 2x2 nodal field",
        "discretization": "Q1 finite elements with 2x2 Gauss quadrature",
        "boundary_condition": "homogeneous Dirichlet",
        "rotated_constant_tensor": {
            "K": [[2.0, 0.6], [0.6, 1.0]],
            "convergence": rotated_constant_convergence(),
        },
        "spatial_tensor_families": spatial_tensor_diagnostics(),
        "limitations": [
            "rectangular geometry only",
            "homogeneous Dirichlet boundary condition only",
            "principal-value ratio is sample-wide while scale and orientation vary spatially",
            "tensor values are represented at grid nodes and bilinearly interpolated",
            "no learned operator is evaluated in this tranche",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
