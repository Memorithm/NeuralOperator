import unittest

import numpy as np

from neural_operator_reference.darcy import darcy_residual_2d, solve_darcy_2d
from neural_operator_reference.darcy_datasets import (
    generate_darcy_dataset,
    random_log_permeability,
)


class DarcyReferenceTests(unittest.TestCase):
    def test_constant_coefficient_solution_converges_second_order(self):
        errors = []
        for points in (17, 33):
            axis = np.linspace(0.0, 1.0, points)
            grid_x, grid_y = np.meshgrid(axis, axis)
            exact = np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
            forcing = 2.0 * np.pi**2 * exact
            solution = solve_darcy_2d(
                np.ones_like(exact),
                forcing=forcing,
            )
            error = np.linalg.norm(solution - exact) / np.linalg.norm(exact)
            errors.append(float(error))

        self.assertLess(errors[1], 1.0e-3)
        self.assertLess(errors[1], 0.3 * errors[0])

    def test_solver_satisfies_its_conservative_stencil(self):
        permeability = random_log_permeability(
            samples=1,
            points_x=17,
            modes=3,
            seed=5,
            log_std=0.5,
        )[0]
        solution = solve_darcy_2d(permeability, forcing=1.0)
        residual = darcy_residual_2d(
            solution,
            permeability,
            forcing=1.0,
        )
        self.assertLess(float(np.max(np.abs(residual))), 1.0e-10)

    def test_dataset_is_deterministic_positive_and_traceable(self):
        first = generate_darcy_dataset(
            samples=2,
            points_x=9,
            modes=2,
            seed=7,
            log_std=0.4,
        )
        second = generate_darcy_dataset(
            samples=2,
            points_x=9,
            modes=2,
            seed=7,
            log_std=0.4,
        )

        np.testing.assert_array_equal(first.inputs, second.inputs)
        np.testing.assert_array_equal(first.targets, second.targets)
        self.assertEqual(first.samples, 2)
        self.assertEqual(first.grid_shape, (9, 9))
        self.assertTrue(np.all(first.inputs > 0.0))
        self.assertTrue(np.all(first.targets[:, 0, :, 0] == 0.0))
        self.assertTrue(np.all(first.targets[:, -1, :, 0] == 0.0))
        self.assertTrue(np.all(first.targets[:, :, 0, 0] == 0.0))
        self.assertTrue(np.all(first.targets[:, :, -1, 0] == 0.0))

        for sample in range(first.samples):
            residual = darcy_residual_2d(
                first.targets[sample, ..., 0],
                first.inputs[sample, ..., 0],
                forcing=first.forcing,
                length_x=first.length_x,
                length_y=first.length_y,
            )
            self.assertLess(float(np.max(np.abs(residual))), 1.0e-10)

    def test_seed_changes_permeability_family(self):
        first = random_log_permeability(1, 9, modes=2, seed=1)
        second = random_log_permeability(1, 9, modes=2, seed=2)
        self.assertFalse(np.array_equal(first, second))

    def test_invalid_coefficients_and_shapes_are_rejected(self):
        permeability = np.ones((9, 9))
        permeability[4, 4] = 0.0
        with self.assertRaises(ValueError):
            solve_darcy_2d(permeability)

        with self.assertRaises(ValueError):
            solve_darcy_2d(np.ones((9, 9)), forcing=np.ones((8, 8)))

        with self.assertRaises(ValueError):
            darcy_residual_2d(
                np.zeros((8, 9)),
                np.ones((9, 9)),
            )


if __name__ == "__main__":
    unittest.main()
