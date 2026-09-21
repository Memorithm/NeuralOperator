import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    TorchDeepONet1D,
    TorchFNO1D,
    benchmark_inference,
    trainable_parameter_count,
)


try:
    import torch
except ImportError:  # pragma: no cover - dependency is installed by optional CI
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch backend is not installed")
class TorchBenchmarkingTests(unittest.TestCase):
    def test_reference_models_have_nearly_identical_parameter_budgets(self):
        fno = TorchFNO1D(width=3, modes=5, seed=4)
        deeponet = TorchDeepONet1D(
            sensor_points=16,
            hidden_width=3,
            latent_width=1,
            seed=4,
        )

        fno_parameters = trainable_parameter_count(fno)
        deeponet_parameters = trainable_parameter_count(deeponet)

        self.assertEqual(fno_parameters, 67)
        self.assertEqual(deeponet_parameters, 66)
        self.assertLessEqual(abs(fno_parameters - deeponet_parameters), 1)

    def test_timing_contract_accepts_tensor_backends(self):
        rng = np.random.default_rng(20260921)
        inputs = rng.normal(size=(2, 16, 1))
        coordinates = np.arange(16, dtype=float) * 2.0 * np.pi / 16.0
        fno = TorchFNO1D(width=3, modes=5, seed=4)
        deeponet = TorchDeepONet1D(
            sensor_points=16,
            hidden_width=3,
            latent_width=1,
            seed=4,
        )

        fno_prediction, fno_timing = benchmark_inference(fno, inputs, repeats=2)
        deeponet_prediction, deeponet_timing = benchmark_inference(
            lambda values: deeponet(values, coordinates),
            inputs,
            repeats=2,
        )

        self.assertEqual(fno_prediction.shape, inputs.shape)
        self.assertEqual(deeponet_prediction.shape, inputs.shape)
        self.assertGreaterEqual(fno_timing.total_seconds, 0.0)
        self.assertGreaterEqual(deeponet_timing.total_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
