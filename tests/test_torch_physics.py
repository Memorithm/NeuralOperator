import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    TorchFNO1D,
    burgers_rk4_step,
    burgers_transition_residual,
    fit_torch_fno_burgers_physics,
    generate_burgers_dataset,
    random_periodic_fields,
    torch_burgers_data_physics_loss,
    torch_burgers_transition_residual,
    torch_spectral_derivative,
)


try:
    import torch
except ImportError:  # pragma: no cover - dependency is installed by optional CI
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch backend is not installed")
class TorchSpectralPhysicsTests(unittest.TestCase):
    def test_spectral_derivative_matches_analytic_periodic_field(self):
        points = 32
        length = 2.0 * np.pi
        x = np.arange(points, dtype=float) * length / points
        values = np.sin(3.0 * x) + 0.25 * np.cos(2.0 * x)
        expected = 3.0 * np.cos(3.0 * x) - 0.5 * np.sin(2.0 * x)
        tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)

        derivative = torch_spectral_derivative(
            tensor,
            order=1,
            length=length,
        )

        np.testing.assert_allclose(
            derivative.detach().cpu().numpy(),
            expected,
            rtol=1.0e-11,
            atol=1.0e-11,
        )
        torch.mean(derivative**2).backward()
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(bool(torch.isfinite(tensor.grad).all()))

    def test_transition_residual_matches_numpy_oracle_with_forcing(self):
        points = 16
        length = 2.0 * np.pi
        horizon = 2.0e-4
        viscosity = 0.05
        x = np.arange(points, dtype=float) * length / points
        forcing = 0.2 * np.sin(2.0 * x)
        initial = random_periodic_fields(
            samples=2,
            points=points,
            modes=3,
            seed=31,
            amplitude=0.2,
            length=length,
        )
        predicted = burgers_rk4_step(
            initial,
            dt=horizon,
            viscosity=viscosity,
            length=length,
            forcing=forcing,
        )
        expected = burgers_transition_residual(
            initial,
            predicted,
            horizon=horizon,
            viscosity=viscosity,
            length=length,
            forcing=forcing,
        )
        actual = torch_burgers_transition_residual(
            torch.as_tensor(initial, dtype=torch.float64),
            torch.as_tensor(predicted, dtype=torch.float64),
            horizon=horizon,
            viscosity=viscosity,
            length=length,
            forcing=forcing,
        )

        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_physics_loss_backpropagates_to_fno_parameters(self):
        dataset = generate_burgers_dataset(
            samples=3,
            points=16,
            dt=2.0e-4,
            steps=2,
            viscosity=0.05,
            seed=41,
        )
        model = TorchFNO1D(width=2, modes=4, seed=7)
        prediction = model(dataset.inputs)[..., 0]
        total, data_loss, physics_loss = torch_burgers_data_physics_loss(
            dataset.inputs[..., 0],
            prediction,
            dataset.targets[..., 0],
            horizon=dataset.horizon,
            viscosity=dataset.viscosity,
            data_weight=1.0,
            physics_weight=1.0e-8,
            length=dataset.length,
            forcing=dataset.forcing,
        )
        total.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        )
        self.assertGreater(float(data_loss.detach().cpu()), 0.0)
        self.assertGreater(float(physics_loss.detach().cpu()), 0.0)

    def test_physics_aware_fit_reduces_weighted_training_objective(self):
        dataset = generate_burgers_dataset(
            samples=4,
            points=16,
            dt=2.0e-4,
            steps=2,
            viscosity=0.05,
            seed=51,
        )
        model = TorchFNO1D(width=2, modes=4, seed=8)
        result = fit_torch_fno_burgers_physics(
            model,
            dataset.inputs,
            dataset.targets,
            horizon=dataset.horizon,
            viscosity=dataset.viscosity,
            physics_weight=1.0e-8,
            length=dataset.length,
            forcing=dataset.forcing,
            epochs=20,
            learning_rate=5.0e-3,
        )

        self.assertLess(result.final_total_loss, result.initial_total_loss)
        self.assertTrue(np.isfinite(result.final_physics_loss))


if __name__ == "__main__":
    unittest.main()
