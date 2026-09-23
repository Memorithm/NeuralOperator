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
        axis_x = np.linspace(0.0, 1.5, 13)
        axis_y = np.linspace(0.0, 0.75, 11)
        grid_x, grid_y = np.meshgrid(axis_x, axis_y)
        permeability = np.exp(
            0.25 * np.sin(2.0 * np.pi * grid_x / 1.5)
            + 0.15 * np.cos(2.0 * np.pi * grid_y / 0.75)
        )
        scalar = solve_darcy_2d(
            permeability,
            forcing=1.0,
            length_x=1.5,
            length_y=0.75,
        )
        diagonal = solve_darcy_diagonal_2d(
            permeability,
            permeability,
            forcing=1.0,
            length_x=1.5,
            length_y=0.75,
        )
        np.testing.assert_allclose(diagonal, scalar, rtol=0.0, atol=1.0e-14)

        scalar_residual = darcy_residual_2d(
            diagonal,
            permeability,
            forcing=1.0,
            length_x=1.5,
            length_y=0.75,
        )
        diagonal_residual = darcy_diagonal_residual_2d(
            diagonal,
            permeability,
            permeability,
            forcing=1.0,
            length_x=1.5,
            length_y=0.75,
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

    def test_dataset_is_deterministic_and_preserves_anisotropy_ratio(self):
        kwargs = dict(
            samples=2,
            points_x=9,
            points_y=11,
            modes=2,
            seed=81,
            log_std=0.4,
            anisotropy_ratio=4.0,
        )
        first = generate_darcy_diagonal_dataset(**kwargs)
        second = generate_darcy_diagonal_dataset(**kwargs)

        np.testing.assert_array_equal(first.inputs, second.inputs)
        np.testing.assert_array_equal(first.targets, second.targets)
        self.assertEqual(first.inputs.shape, (2, 11, 9, 2))
        self.assertEqual(first.targets.shape, (2, 11, 9, 1))
        self.assertEqual(first.grid_shape, (11, 9))

        ratio = first.inputs[..., 0] / first.inputs[..., 1]
        np.testing.assert_allclose(ratio, 4.0, rtol=1.0e-14, atol=1.0e-14)
        determinant_scale = np.sqrt(first.inputs[..., 0] * first.inputs[..., 1])
        self.assertTrue(np.all(np.isfinite(determinant_scale)))
        self.assertTrue(np.all(determinant_scale > 0.0))

        for index in range(first.samples):
            residual = darcy_diagonal_residual_2d(
                first.targets[index, ..., 0],
                first.inputs[index, ..., 0],
                first.inputs[index, ..., 1],
                forcing=first.forcing,
                length_x=first.length_x,
                length_y=first.length_y,
            )
            self.assertLess(float(np.max(np.abs(residual))), 1.0e-10)

    def test_invalid_tensor_coefficients_are_rejected(self):
        coefficient = np.ones((9, 9))
        nonpositive = coefficient.copy()
        nonpositive[4, 4] = 0.0

        with self.assertRaises(ValueError):
            solve_darcy_diagonal_2d(nonpositive, coefficient)
        with self.assertRaises(ValueError):
            solve_darcy_diagonal_2d(coefficient, nonpositive)
        with self.assertRaises(ValueError):
            solve_darcy_diagonal_2d(coefficient, np.ones((9, 8)))
        with self.assertRaises(ValueError):
            darcy_diagonal_residual_2d(
                np.zeros((8, 9)),
                coefficient,
                coefficient,
            )

    def test_invalid_anisotropy_ratio_is_rejected(self):
        for ratio in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    generate_darcy_diagonal_dataset(
                        samples=1,
                        points_x=9,
                        anisotropy_ratio=ratio,
                    )


if __name__ == "__main__":
    unittest.main()
