import unittest

import numpy as np

from neural_operator_reference import (
    darcy_tensor_residual_2d,
    generate_darcy_tensor_dataset,
    solve_darcy_tensor_2d,
)


class DarcyTensorReferenceTests(unittest.TestCase):
    def test_rotated_constant_tensor_converges_second_order(self):
        errors = []
        kxx_value = 2.0
        kxy_value = 0.6
        kyy_value = 1.0

        for points in (17, 33):
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
            errors.append(
                float(np.linalg.norm(solution - exact) / np.linalg.norm(exact))
            )
            residual = darcy_tensor_residual_2d(
                solution,
                kxx,
                kxy,
                kyy,
                forcing=forcing,
            )
            self.assertLess(float(np.max(np.abs(residual))), 1.0e-11)

        self.assertLess(errors[1], 1.0e-3)
        self.assertLess(errors[1], 0.3 * errors[0])

    def test_dataset_is_deterministic_spd_and_preserves_principal_ratio(self):
        kwargs = dict(
            samples=2,
            points_x=9,
            points_y=11,
            modes=2,
            seed=81,
            log_std=0.4,
            anisotropy_ratio=4.0,
            orientation_base_radians=0.35,
            orientation_amplitude_radians=0.6,
            orientation_modes=2,
        )
        first = generate_darcy_tensor_dataset(**kwargs)
        second = generate_darcy_tensor_dataset(**kwargs)

        np.testing.assert_array_equal(first.inputs, second.inputs)
        np.testing.assert_array_equal(first.targets, second.targets)
        self.assertEqual(first.inputs.shape, (2, 11, 9, 3))
        self.assertEqual(first.targets.shape, (2, 11, 9, 1))
        self.assertEqual(first.grid_shape, (11, 9))

        kxx = first.inputs[..., 0]
        kxy = first.inputs[..., 1]
        kyy = first.inputs[..., 2]
        determinant = kxx * kyy - kxy * kxy
        self.assertTrue(np.all(determinant > 0.0))
        self.assertGreater(float(np.std(kxy)), 1.0e-3)

        trace = kxx + kyy
        discriminant = np.sqrt((kxx - kyy) ** 2 + 4.0 * kxy * kxy)
        eigenvalue_max = 0.5 * (trace + discriminant)
        eigenvalue_min = 0.5 * (trace - discriminant)
        np.testing.assert_allclose(
            eigenvalue_max / eigenvalue_min,
            4.0,
            rtol=2.0e-14,
            atol=2.0e-14,
        )

        for index in range(first.samples):
            residual = darcy_tensor_residual_2d(
                first.targets[index, ..., 0],
                first.inputs[index, ..., 0],
                first.inputs[index, ..., 1],
                first.inputs[index, ..., 2],
                forcing=first.forcing,
                length_x=first.length_x,
                length_y=first.length_y,
            )
            self.assertLess(float(np.max(np.abs(residual))), 1.0e-11)

    def test_isotropic_tensor_collapses_orientation(self):
        first = generate_darcy_tensor_dataset(
            samples=1,
            points_x=9,
            modes=2,
            seed=19,
            anisotropy_ratio=1.0,
            orientation_base_radians=0.0,
            orientation_amplitude_radians=0.0,
            orientation_modes=2,
        )
        second = generate_darcy_tensor_dataset(
            samples=1,
            points_x=9,
            modes=2,
            seed=19,
            anisotropy_ratio=1.0,
            orientation_base_radians=1.1,
            orientation_amplitude_radians=0.7,
            orientation_modes=2,
        )
        np.testing.assert_allclose(first.inputs, second.inputs, rtol=0.0, atol=2e-15)
        np.testing.assert_allclose(first.targets, second.targets, rtol=0.0, atol=2e-14)
        self.assertLess(float(np.max(np.abs(second.inputs[..., 1]))), 1.0e-15)
        np.testing.assert_allclose(
            second.inputs[..., 0],
            second.inputs[..., 2],
            rtol=0.0,
            atol=2.0e-15,
        )

    def test_invalid_tensor_is_rejected(self):
        coefficient = np.ones((7, 7))
        with self.assertRaises(ValueError):
            solve_darcy_tensor_2d(
                coefficient,
                coefficient,
                coefficient,
            )
        with self.assertRaises(ValueError):
            solve_darcy_tensor_2d(
                coefficient,
                np.zeros((7, 6)),
                coefficient,
            )
        nonfinite = np.zeros((7, 7))
        nonfinite[3, 3] = np.nan
        with self.assertRaises(ValueError):
            solve_darcy_tensor_2d(
                coefficient,
                nonfinite,
                coefficient,
            )

    def test_invalid_dataset_orientation_controls_are_rejected(self):
        invalid_kwargs = (
            {"anisotropy_ratio": 0.0},
            {"anisotropy_ratio": np.inf},
            {"orientation_base_radians": np.nan},
            {"orientation_amplitude_radians": -0.1},
            {"orientation_modes": 0},
            {"orientation_modes": 1.5},
        )
        for extra in invalid_kwargs:
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    generate_darcy_tensor_dataset(
                        samples=1,
                        points_x=9,
                        modes=2,
                        **extra,
                    )


if __name__ == "__main__":
    unittest.main()
