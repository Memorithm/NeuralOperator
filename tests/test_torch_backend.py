"""Optional PyTorch backend tests."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - normal in the base CI image
    torch = None

if torch is not None:
    from neural_operator_reference.torch_backend import TorchFNO1D


@unittest.skipUnless(torch is not None, "optional PyTorch dependency is not installed")
class TorchFNO1DTests(unittest.TestCase):
    def test_forward_preserves_the_operator_contract(self) -> None:
        model = TorchFNO1D(width=3, modes=4, seed=5)
        inputs = np.zeros((2, 16, 1), dtype=float)
        inputs[:, :, 0] = np.sin(np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False))
        prediction = model(inputs)
        self.assertEqual(tuple(prediction.shape), (2, 16, 1))
        self.assertTrue(bool(torch.isfinite(prediction).all()))

    def test_fit_reduces_loss_with_autograd(self) -> None:
        rng = np.random.default_rng(7)
        inputs = rng.normal(size=(4, 16, 1))
        targets = 0.35 * inputs + 0.05
        model = TorchFNO1D(width=4, modes=5, seed=8)
        fit = model.fit(inputs, targets, epochs=60, learning_rate=2.0e-2)
        self.assertEqual(fit.epochs, 60)
        self.assertLess(fit.final_loss, fit.initial_loss)
        self.assertTrue(np.isfinite(fit.final_loss))

    def test_invalid_inputs_are_rejected(self) -> None:
        model = TorchFNO1D(width=3, modes=4)
        with self.assertRaises(ValueError):
            model(np.zeros((2, 16), dtype=float))
        with self.assertRaises(ValueError):
            model.loss(np.zeros((2, 16, 1)), np.zeros((2, 15, 1)))


if __name__ == "__main__":
    unittest.main()
