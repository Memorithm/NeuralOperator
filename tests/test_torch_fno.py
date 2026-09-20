import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import NumpyFNO1D, TorchFNO1D  # noqa: E402


try:
    import torch
except ImportError:  # pragma: no cover - dependency is installed by optional CI
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch backend is not installed")
class TorchFNOTests(unittest.TestCase):
    def test_forward_preserves_the_operator_tensor_contract(self):
        model = TorchFNO1D(width=3, modes=4, seed=4)
        inputs = np.zeros((2, 12, 1), dtype=float)
        output = model(inputs)

        self.assertEqual(tuple(output.shape), (2, 12, 1))
        self.assertEqual(output.dtype, torch.float64)
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_autodiff_training_reduces_reference_loss(self):
        rng = np.random.default_rng(20260920)
        inputs = rng.normal(size=(4, 12, 1))
        targets = (0.4 * inputs + 0.1).astype(float)

        model = TorchFNO1D(width=4, modes=4, seed=4)
        result = model.fit(
            inputs,
            targets,
            epochs=60,
            learning_rate=2.0e-2,
        )

        self.assertLess(result.final_loss, result.initial_loss)
        self.assertLess(model.loss(inputs, targets), result.initial_loss)


    def test_parameters_match_the_numpy_reference(self):
        rng = np.random.default_rng(20260920)
        inputs = rng.normal(size=(2, 12, 1))
        numpy_model = NumpyFNO1D(width=3, modes=4, seed=4)
        torch_model = TorchFNO1D(width=3, modes=4, seed=9)
        torch_model.set_parameter_vector(numpy_model.parameter_vector())

        numpy_output = numpy_model(inputs)
        torch_output = torch_model(inputs).detach().cpu().numpy()

        self.assertTrue(
            np.allclose(torch_output, numpy_output, rtol=1.0e-11, atol=1.0e-11)
        )
        self.assertTrue(
            np.allclose(
                torch_model.parameter_vector(),
                numpy_model.parameter_vector(),
                rtol=0.0,
                atol=0.0,
            )
        )

if __name__ == "__main__":
    unittest.main()
