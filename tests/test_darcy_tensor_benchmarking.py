import unittest

import numpy as np

from neural_operator_reference import (
    evaluate_darcy_tensor_dataset,
    generate_darcy_tensor_dataset,
)


class DarcyTensorBenchmarkingTests(unittest.TestCase):
    def test_exact_tensor_targets_have_zero_data_error_and_small_physics_residual(self):
        dataset = generate_darcy_tensor_dataset(
            samples=2,
            points_x=7,
            modes=2,
            seed=31,
            log_std=0.3,
            anisotropy_ratio=4.0,
            orientation_base_radians=0.2,
            orientation_amplitude_radians=0.4,
            orientation_modes=2,
        )

        def exact_operator(inputs):
            self.assertEqual(inputs.shape[-1], 3)
            return dataset.targets

        metrics = evaluate_darcy_tensor_dataset(
            exact_operator,
            dataset,
            repeats=2,
        )
        self.assertEqual(metrics.mse, 0.0)
        self.assertEqual(metrics.relative_l2, 0.0)
        self.assertLess(metrics.physics_mse, 1.0e-28)
        self.assertEqual(metrics.boundary_max_abs, 0.0)
        self.assertEqual(metrics.timing.repeats, 2)

    def test_tensor_evaluator_rejects_wrong_output_shape(self):
        dataset = generate_darcy_tensor_dataset(
            samples=1,
            points_x=7,
            modes=2,
            seed=32,
            anisotropy_ratio=4.0,
        )
        with self.assertRaises(ValueError):
            evaluate_darcy_tensor_dataset(
                lambda inputs: np.zeros_like(inputs),
                dataset,
                repeats=1,
            )

    def test_tensor_evaluator_rejects_invalid_repeats(self):
        dataset = generate_darcy_tensor_dataset(
            samples=1,
            points_x=7,
            modes=2,
            seed=33,
            anisotropy_ratio=4.0,
        )
        with self.assertRaises(ValueError):
            evaluate_darcy_tensor_dataset(
                lambda inputs: dataset.targets,
                dataset,
                repeats=0,
            )


if __name__ == "__main__":
    unittest.main()
