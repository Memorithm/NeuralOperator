import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    TorchDeepONet2D,
    TorchFNO2D,
    TorchLocalConv2D,
    generate_darcy_tensor_dataset,
    normalized_query_grid,
    trainable_parameter_count,
)


try:
    import torch
except ImportError:  # pragma: no cover - installed by optional CI
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch backend is not installed")
class TorchDarcyTensorComparisonTests(unittest.TestCase):
    def test_three_channel_parameter_budgets_are_within_three_percent(self):
        fno = TorchFNO2D(
            in_channels=3,
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
            in_channels=3,
            hidden_width=22,
            latent_width=64,
            seed=4,
        )
        local = TorchLocalConv2D(in_channels=3, width=28, seed=4)

        counts = [
            trainable_parameter_count(fno),
            trainable_parameter_count(deeponet),
            trainable_parameter_count(local),
        ]
        self.assertEqual(counts, [8465, 8379, 8625])
        self.assertLessEqual(max(counts) / min(counts), 1.03)

    def test_all_models_accept_tensor_channels_and_impose_zero_boundary(self):
        dataset = generate_darcy_tensor_dataset(
            samples=2,
            points_x=7,
            modes=2,
            seed=41,
            log_std=0.3,
            anisotropy_ratio=4.0,
            orientation_base_radians=0.3,
            orientation_amplitude_radians=0.4,
            orientation_modes=2,
        )
        coordinates = normalized_query_grid(7, 7)

        fno = TorchFNO2D(
            in_channels=3,
            width=6,
            modes_y=3,
            modes_x=3,
            depth=1,
            padding=1,
            seed=2,
        )
        deep = TorchDeepONet2D(
            sensor_points_y=7,
            sensor_points_x=7,
            in_channels=3,
            hidden_width=12,
            latent_width=12,
            seed=2,
        )
        local = TorchLocalConv2D(in_channels=3, width=8, seed=2)

        predictions = (
            fno(dataset.inputs),
            deep(dataset.inputs, coordinates).reshape(2, 7, 7, 1),
            local(dataset.inputs),
        )
        for prediction in predictions:
            self.assertEqual(tuple(prediction.shape), (2, 7, 7, 1))
            self.assertTrue(bool(torch.isfinite(prediction).all()))
            pressure = prediction[..., 0]
            self.assertEqual(float(torch.max(torch.abs(pressure[:, 0, :]))), 0.0)
            self.assertEqual(float(torch.max(torch.abs(pressure[:, -1, :]))), 0.0)
            self.assertEqual(float(torch.max(torch.abs(pressure[:, :, 0]))), 0.0)
            self.assertEqual(float(torch.max(torch.abs(pressure[:, :, -1]))), 0.0)


if __name__ == "__main__":
    unittest.main()
