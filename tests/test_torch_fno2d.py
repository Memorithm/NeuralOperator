import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    TorchFNO2D,
    evaluate_darcy_dataset,
    generate_darcy_dataset,
)


try:
    import torch
except ImportError:  # pragma: no cover - dependency is installed by optional CI
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch backend is not installed")
class TorchFNO2DTests(unittest.TestCase):
    def test_forward_preserves_grid_and_hard_dirichlet_boundary(self):
        rng = np.random.default_rng(20260921)
        inputs = np.exp(rng.normal(scale=0.2, size=(3, 9, 11, 1)))
        model = TorchFNO2D(
            width=6,
            modes_y=3,
            modes_x=4,
            depth=2,
            padding=2,
            seed=4,
        )

        prediction = model(inputs)

        self.assertEqual(tuple(prediction.shape), inputs.shape)
        self.assertEqual(prediction.dtype, torch.float64)
        self.assertEqual(float(torch.max(torch.abs(prediction[:, 0]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(prediction[:, -1]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(prediction[:, :, 0]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(prediction[:, :, -1]))), 0.0)

    def test_gradients_flow_through_complex_spectral_weights(self):
        rng = np.random.default_rng(8)
        inputs = np.exp(rng.normal(scale=0.2, size=(2, 7, 7, 1)))
        model = TorchFNO2D(
            width=5,
            modes_y=3,
            modes_x=3,
            depth=1,
            padding=1,
            seed=2,
        )

        loss = torch.mean(model(inputs) ** 2)
        loss.backward()

        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_fit_reduces_loss_on_reference_darcy_samples(self):
        dataset = generate_darcy_dataset(
            samples=4,
            points_x=7,
            modes=2,
            seed=17,
            log_std=0.35,
        )
        model = TorchFNO2D(
            width=5,
            modes_y=3,
            modes_x=3,
            depth=1,
            padding=1,
            seed=3,
        )

        result = model.fit(
            dataset.inputs,
            dataset.targets,
            epochs=25,
            learning_rate=2.0e-2,
        )

        self.assertLess(result.final_loss, 0.2 * result.initial_loss)

    def test_same_weights_execute_on_a_finer_grid(self):
        coarse = generate_darcy_dataset(
            samples=2,
            points_x=7,
            modes=2,
            seed=31,
            log_std=0.3,
        )
        fine = generate_darcy_dataset(
            samples=2,
            points_x=13,
            modes=2,
            seed=31,
            log_std=0.3,
        )
        model = TorchFNO2D(
            width=4,
            modes_y=3,
            modes_x=3,
            depth=1,
            padding=1,
            seed=5,
        )
        model.fit(
            coarse.inputs,
            coarse.targets,
            epochs=10,
            learning_rate=2.0e-2,
        )

        prediction = model(fine.inputs)

        self.assertEqual(tuple(prediction.shape), fine.targets.shape)
        self.assertTrue(bool(torch.isfinite(prediction).all()))

    def test_shared_darcy_metrics_include_discrete_residual_and_boundary(self):
        dataset = generate_darcy_dataset(
            samples=2,
            points_x=7,
            modes=2,
            seed=42,
            log_std=0.25,
        )
        model = TorchFNO2D(
            width=4,
            modes_y=3,
            modes_x=3,
            depth=1,
            padding=1,
            seed=6,
        )

        metrics = evaluate_darcy_dataset(model, dataset, repeats=2)

        self.assertGreaterEqual(metrics.mse, 0.0)
        self.assertGreaterEqual(metrics.relative_l2, 0.0)
        self.assertGreaterEqual(metrics.physics_mse, 0.0)
        self.assertEqual(metrics.boundary_max_abs, 0.0)
        self.assertGreaterEqual(metrics.timing.total_seconds, 0.0)

    def test_invalid_shapes_are_rejected(self):
        model = TorchFNO2D(width=4, modes_y=2, modes_x=2, depth=1)
        with self.assertRaises(ValueError):
            model(np.ones((2, 9, 1)))
        with self.assertRaises(ValueError):
            model(np.ones((2, 9, 9, 2)))


if __name__ == "__main__":
    unittest.main()
