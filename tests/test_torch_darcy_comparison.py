import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    TorchDeepONet2D,
    TorchFNO2D,
    TorchLocalConv2D,
    generate_darcy_dataset,
    normalized_query_grid,
    trainable_parameter_count,
)


try:
    import torch
except ImportError:  # pragma: no cover - installed by optional CI
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch backend is not installed")
class TorchDarcyComparisonTests(unittest.TestCase):
    def test_matched_reference_parameter_budgets_are_within_three_percent(self):
        fno = TorchFNO2D(
            width=8,
            modes_y=4,
            modes_x=4,
            depth=2,
            padding=2,
            seed=4,
        )
        deeponet = TorchDeepONet2D(
            sensor_points_y=9,
            sensor_points_x=9,
            hidden_width=40,
            latent_width=64,
            seed=4,
        )
        local = TorchLocalConv2D(width=29, seed=4)

        counts = [
            trainable_parameter_count(fno),
            trainable_parameter_count(deeponet),
            trainable_parameter_count(local),
        ]

        self.assertEqual(counts, [8449, 8649, 8672])
        self.assertLessEqual(max(counts) / min(counts), 1.03)

    def test_deeponet_hard_boundary_is_exact(self):
        dataset = generate_darcy_dataset(
            samples=2,
            points_x=7,
            modes=2,
            seed=11,
            log_std=0.3,
        )
        coordinates = normalized_query_grid(7, 7)
        model = TorchDeepONet2D(
            sensor_points_y=7,
            sensor_points_x=7,
            hidden_width=16,
            latent_width=12,
            seed=2,
        )
        prediction = model(dataset.inputs, coordinates)[0, :, 0].reshape(7, 7)

        self.assertEqual(float(torch.max(torch.abs(prediction[0, :]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(prediction[-1, :]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(prediction[:, 0]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(prediction[:, -1]))), 0.0)

    def test_local_baseline_hard_boundary_is_exact(self):
        dataset = generate_darcy_dataset(
            samples=2,
            points_x=7,
            modes=2,
            seed=12,
            log_std=0.3,
        )
        model = TorchLocalConv2D(width=8, seed=3)
        prediction = model(dataset.inputs)

        self.assertEqual(float(torch.max(torch.abs(prediction[:, 0]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(prediction[:, -1]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(prediction[:, :, 0]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(prediction[:, :, -1]))), 0.0)

    def test_both_comparators_reduce_reference_darcy_training_loss(self):
        dataset = generate_darcy_dataset(
            samples=4,
            points_x=7,
            modes=2,
            seed=23,
            log_std=0.35,
        )
        coordinates = normalized_query_grid(7, 7)

        deeponet = TorchDeepONet2D(
            sensor_points_y=7,
            sensor_points_x=7,
            hidden_width=16,
            latent_width=12,
            seed=4,
        )
        deep_result = deeponet.fit(
            dataset.inputs,
            coordinates,
            dataset.targets.reshape(dataset.samples, -1, 1),
            epochs=40,
            learning_rate=2.0e-2,
        )

        local = TorchLocalConv2D(width=8, seed=4)
        local_result = local.fit(
            dataset.inputs,
            dataset.targets,
            epochs=30,
            learning_rate=2.0e-2,
        )

        self.assertLess(deep_result.final_loss, deep_result.initial_loss)
        self.assertLess(local_result.final_loss, local_result.initial_loss)

    def test_normalized_query_grid_contract(self):
        coordinates = normalized_query_grid(5, 7)
        self.assertEqual(coordinates.shape, (35, 2))
        np.testing.assert_array_equal(coordinates[0], [0.0, 0.0])
        np.testing.assert_array_equal(coordinates[-1], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
