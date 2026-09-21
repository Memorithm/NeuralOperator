import unittest

import numpy as np

from neural_operator_reference import (
    generate_darcy_dataset,
    normalize_darcy_dataset_scale,
    normalize_darcy_permeability_scale,
    normalize_darcy_pressure_scale,
    restore_darcy_pressure_scale,
    solve_darcy_2d,
)


class DarcyScaleEquivarianceTests(unittest.TestCase):
    def test_normalized_permeability_has_unit_geometric_mean(self):
        dataset = generate_darcy_dataset(
            samples=3,
            points_x=9,
            modes=2,
            seed=71,
            log_std=0.45,
            mean_log_permeability=0.4,
        )
        normalized, scales = normalize_darcy_permeability_scale(dataset.inputs)

        geometric_mean = np.exp(
            np.mean(np.log(normalized[..., 0]), axis=(1, 2))
        )
        np.testing.assert_allclose(geometric_mean, 1.0, rtol=0.0, atol=1.0e-14)
        self.assertEqual(scales.shape, (dataset.samples, 1, 1, 1))

    def test_global_permeability_scaling_has_exact_pressure_covariance(self):
        dataset = generate_darcy_dataset(
            samples=2,
            points_x=9,
            modes=2,
            seed=72,
            log_std=0.45,
        )
        factor = np.exp(0.7)
        scaled_inputs = dataset.inputs * factor
        scaled_targets = np.stack(
            [
                solve_darcy_2d(
                    scaled_inputs[index, ..., 0],
                    forcing=dataset.forcing,
                    length_x=dataset.length_x,
                    length_y=dataset.length_y,
                )
                for index in range(dataset.samples)
            ],
            axis=0,
        )[..., None]

        base_inputs, base_scales = normalize_darcy_permeability_scale(
            dataset.inputs
        )
        shifted_inputs, shifted_scales = normalize_darcy_permeability_scale(
            scaled_inputs
        )
        base_targets = normalize_darcy_pressure_scale(
            dataset.targets,
            base_scales,
        )
        shifted_targets = normalize_darcy_pressure_scale(
            scaled_targets,
            shifted_scales,
        )

        np.testing.assert_allclose(
            shifted_inputs,
            base_inputs,
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            shifted_targets,
            base_targets,
            rtol=1.0e-11,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            scaled_targets,
            dataset.targets / factor,
            rtol=1.0e-11,
            atol=1.0e-12,
        )

    def test_dataset_transform_round_trip_restores_pressure(self):
        dataset = generate_darcy_dataset(
            samples=2,
            points_x=9,
            modes=2,
            seed=73,
            log_std=0.45,
            mean_log_permeability=-0.6,
        )
        normalized_inputs, normalized_targets, scales = (
            normalize_darcy_dataset_scale(dataset)
        )
        restored = restore_darcy_pressure_scale(normalized_targets, scales)

        self.assertEqual(normalized_inputs.shape, dataset.inputs.shape)
        np.testing.assert_allclose(restored, dataset.targets, rtol=0.0, atol=1.0e-15)

    def test_nonpositive_permeability_is_rejected(self):
        invalid = np.ones((1, 7, 7, 1))
        invalid[0, 3, 3, 0] = 0.0

        with self.assertRaises(ValueError):
            normalize_darcy_permeability_scale(invalid)


if __name__ == "__main__":
    unittest.main()
