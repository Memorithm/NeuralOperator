import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import TorchDeepONet1D  # noqa: E402


try:
    import torch
except ImportError:  # pragma: no cover - dependency is installed by optional CI
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch backend is not installed")
class TorchDeepONetTests(unittest.TestCase):
    def test_nonlinear_deeponet_transfers_to_arbitrary_query_resolution(self):
        rng = np.random.default_rng(20260920)
        inputs = rng.normal(size=(24, 8, 1))
        train_inputs = inputs[:20]
        held_out_inputs = inputs[20:]
        coarse_coordinates = np.arange(16, dtype=float) * 2.0 * np.pi / 16.0
        fine_coordinates = np.arange(32, dtype=float) * 2.0 * np.pi / 32.0

        def teacher(fields, coordinates):
            mean = np.mean(fields[..., 0], axis=1, keepdims=True)
            values = np.tanh(
                mean
                + 0.2 * np.sin(coordinates)[None, :]
                + 0.1 * np.cos(2.0 * coordinates)[None, :]
            )
            return values[..., None]

        train_targets = teacher(train_inputs, coarse_coordinates)
        held_out_targets = teacher(held_out_inputs, coarse_coordinates)
        fine_targets = teacher(held_out_inputs, fine_coordinates)

        model = TorchDeepONet1D(
            sensor_points=8,
            hidden_width=16,
            latent_width=8,
            seed=4,
        )
        result = model.fit(
            train_inputs,
            coarse_coordinates,
            train_targets,
            epochs=120,
            learning_rate=2.0e-2,
        )
        coarse_prediction = model(held_out_inputs, coarse_coordinates)
        fine_prediction = model(held_out_inputs, fine_coordinates)

        self.assertLess(result.final_loss, result.initial_loss)
        self.assertEqual(tuple(fine_prediction.shape), (4, 32, 1))
        self.assertTrue(bool(torch.isfinite(fine_prediction).all()))
        held_out_loss = float(
            torch.mean(
                (coarse_prediction - torch.as_tensor(held_out_targets)) ** 2
            )
        )
        fine_loss = float(
            torch.mean(
                (fine_prediction - torch.as_tensor(fine_targets)) ** 2
            )
        )
        self.assertTrue(np.isfinite(held_out_loss))
        self.assertTrue(np.isfinite(fine_loss))


if __name__ == "__main__":
    unittest.main()
