import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    TorchFNO2D,
    darcy_residual_2d,
    fit_torch_fno_darcy_physics,
    generate_darcy_dataset,
    torch_darcy_data_physics_loss,
    torch_darcy_residual_2d,
)


try:
    import torch
except ImportError:  # pragma: no cover - installed by optional CI
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch backend is not installed")
class TorchDarcyPhysicsTests(unittest.TestCase):
    def test_differentiable_residual_matches_numpy_oracle(self):
        dataset = generate_darcy_dataset(
            samples=2,
            points_x=7,
            modes=2,
            seed=19,
            log_std=0.3,
        )
        rng = np.random.default_rng(20260921)
        pressure = rng.normal(scale=0.02, size=(2, 7, 7))
        pressure[:, 0, :] = 0.0
        pressure[:, -1, :] = 0.0
        pressure[:, :, 0] = 0.0
        pressure[:, :, -1] = 0.0

        torch_residual = torch_darcy_residual_2d(
            pressure,
            dataset.inputs[..., 0],
            forcing=dataset.forcing,
            length_x=dataset.length_x,
            length_y=dataset.length_y,
        ).detach().cpu().numpy()

        expected = np.stack(
            [
                darcy_residual_2d(
                    pressure[index],
                    dataset.inputs[index, ..., 0],
                    forcing=dataset.forcing,
                    length_x=dataset.length_x,
                    length_y=dataset.length_y,
                )
                for index in range(dataset.samples)
            ],
            axis=0,
        )
        np.testing.assert_allclose(torch_residual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_residual_retains_pressure_gradients(self):
        dataset = generate_darcy_dataset(
            samples=1,
            points_x=7,
            modes=2,
            seed=20,
            log_std=0.25,
        )
        pressure = torch.tensor(
            dataset.targets[..., 0],
            dtype=torch.float64,
            requires_grad=True,
        )
        residual = torch_darcy_residual_2d(
            pressure,
            dataset.inputs[..., 0],
            forcing=dataset.forcing,
        )
        loss = torch.mean((residual + 0.1) ** 2)
        loss.backward()

        self.assertIsNotNone(pressure.grad)
        self.assertTrue(bool(torch.isfinite(pressure.grad).all()))
        self.assertGreater(float(torch.max(torch.abs(pressure.grad))), 0.0)

    def test_data_physics_loss_separates_terms(self):
        dataset = generate_darcy_dataset(
            samples=1,
            points_x=7,
            modes=2,
            seed=21,
            log_std=0.3,
        )
        prediction = torch.tensor(dataset.targets[..., 0], dtype=torch.float64)
        total, data_loss, physics_loss = torch_darcy_data_physics_loss(
            prediction,
            dataset.targets[..., 0],
            dataset.inputs[..., 0],
            forcing=dataset.forcing,
            physics_weight=1.0e-4,
        )

        self.assertEqual(float(data_loss), 0.0)
        self.assertLess(float(physics_loss), 1.0e-18)
        self.assertLess(float(total), 1.0e-22)

    def test_physics_fit_reduces_all_training_terms(self):
        dataset = generate_darcy_dataset(
            samples=4,
            points_x=7,
            modes=2,
            seed=22,
            log_std=0.35,
        )
        model = TorchFNO2D(
            width=5,
            modes_y=3,
            modes_x=3,
            depth=1,
            padding=1,
            seed=4,
        )
        result = fit_torch_fno_darcy_physics(
            model,
            dataset.inputs,
            dataset.targets,
            forcing=dataset.forcing,
            physics_weight=1.0e-4,
            epochs=30,
            learning_rate=2.0e-2,
        )

        self.assertLess(result.final_total_loss, result.initial_total_loss)
        self.assertLess(result.final_data_loss, result.initial_data_loss)
        self.assertLess(result.final_physics_loss, result.initial_physics_loss)

    def test_nonpositive_permeability_is_rejected(self):
        pressure = torch.zeros((1, 7, 7), dtype=torch.float64)
        permeability = torch.ones((1, 7, 7), dtype=torch.float64)
        permeability[:, 3, 3] = 0.0

        with self.assertRaises(ValueError):
            torch_darcy_residual_2d(pressure, permeability)


if __name__ == "__main__":
    unittest.main()
