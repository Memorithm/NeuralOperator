import unittest

import numpy as np

from neural_operator_reference import (
    darcy_diagonal_residual_2d,
    darcy_residual_2d,
    generate_darcy_diagonal_dataset,
    solve_darcy_2d,
    solve_darcy_diagonal_2d,
)


class DarcyAnisotropicReferenceTests(unittest.TestCase):
    def test_isotropic_diagonal_solver_matches_scalar_solver(self):
        axis = np.linspace(0.0, 1.0, 13)
        grid_x, grid_y = np.meshgrid(axis, axis)
        permeability = np.exp(
            0.25 * np.sin(2.0 * np.pi * grid_x)
            + 0.15 * np.cos(2.0 * np.pi * grid_y)
        )
        scalar = solve_darcy_2d(permeability, forcing=1.0)
        diagonal = solve_darcy_diagonal_2d(
            permeability,
            permeability,
            forcing=1.0,
        )
        np.testing.assert_allclose(diagonal, scalar, rtol=0.0, atol=1.0e-14)

        scalar_residual = darcy_residual_2d(
            diagonal,
            permeability,
            forcing=1.0,
        )
        diagonal_residual = darcy_diagonal_residual_2d(
            diagonal,
            permeability,
            permeability,
            forcing=1.0,
        )
        np.testing.assert_allclose(
            diagonal_residual,
            scalar_residual,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_constant_anisotropy_converges_second_order(self):
        errors = []
        kx_value = 2.0
        ky_value = 0.5
        for points in (17, 33):
            axis = np.linspace(0.0, 1.0, points)
            grid_x, grid_y = np.meshgrid(axis, axis)
            exact = np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
            forcing = np.pi**2 * (kx_value + ky_value) * exact
            solution = solve_darcy_diagonal_2d(
                np.full_like(exact, kx_value),
                np.full_like(exact, ky_value),
                forcing=forcing,
            )
            errors.append(
                float(np.linalg.norm(solution - exact) / np.linalg.norm(exact))
            )

        self.assertLess(errors[1], 1.0e-3)
        self.assertLess(errors[1], 0.3 * errors[0])

    def test_dataset_preserves_requested_anisotropy_ratio_and_residual(self):
        dataset = generate_darcy_diagonal_dataset(
            samples=2,
            points_x=9,
            modes=2,
            seed=81,
            log_std=0.4,
            anisotropy_ratio=4.0,
        )
        ratio = dataset.inputs[..., 0] / dataset.inputs[..., 1]
        np.testing.assert_allclose(ratio, 4.0, rtol=1.0e-14, atol=1.0e-14)

        determinant_scale = np.sqrt(
            dataset.inputs[..., 0] * dataset.inputs[..., 1]
        )
        self.assertTrue(np.all(determinant_scale > 0.0))
        for index in range(dataset.samples):
            residual = darcy_diagonal_residual_2d(
                dataset.targets[index, ..., 0],
                dataset.inputs[index, ..., 0],
                dataset.inputs[index, ..., 1],
                forcing=dataset.forcing,
            )
            self.assertLess(float(np.max(np.abs(residual))), 1.0e-10)

    def test_invalid_anisotropy_ratio_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_darcy_diagonal_dataset(
                samples=1,
                points_x=9,
                anisotropy_ratio=0.0,
            )


if __name__ == "__main__":
    unittest.main()
