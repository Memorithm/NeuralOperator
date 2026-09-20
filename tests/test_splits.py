import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    BurgersDatasetSpec,
    generate_burgers_split,
)


class BurgersDatasetSplitTests(unittest.TestCase):
    def _specs(self):
        common = {
            "points": 16,
            "dt": 2.0e-4,
            "steps": 2,
            "length": 2.0 * np.pi,
        }
        return (
            BurgersDatasetSpec(
                samples=3,
                viscosity=0.05,
                seed=10,
                modes=3,
                amplitude=0.4,
                **common,
            ),
            BurgersDatasetSpec(
                samples=2,
                viscosity=0.05,
                seed=11,
                modes=4,
                amplitude=0.5,
                **common,
            ),
            BurgersDatasetSpec(
                samples=2,
                viscosity=0.1,
                seed=12,
                modes=6,
                amplitude=0.35,
                **common,
            ),
        )

    def test_split_is_deterministic_and_retains_partition_provenance(self):
        specs = self._specs()
        first = generate_burgers_split(*specs)
        second = generate_burgers_split(*specs)

        self.assertTrue(np.array_equal(first.train.inputs, second.train.inputs))
        self.assertTrue(np.array_equal(first.validation.targets, second.validation.targets))
        self.assertTrue(np.array_equal(first.ood.targets, second.ood.targets))
        self.assertEqual(first.as_dict()["ood"]["viscosity"], 0.1)
        self.assertEqual(first.ood.modes, 6)
        self.assertEqual(first.ood.amplitude, 0.35)
        self.assertEqual(first.train.horizon, 4.0e-4)

    def test_split_rejects_a_different_prediction_grid(self):
        train, validation, ood = self._specs()
        incompatible = BurgersDatasetSpec(
            samples=ood.samples,
            points=32,
            dt=ood.dt,
            steps=ood.steps,
            viscosity=ood.viscosity,
            seed=ood.seed,
            length=ood.length,
            modes=ood.modes,
            amplitude=ood.amplitude,
        )
        with self.assertRaises(ValueError):
            generate_burgers_split(train, validation, incompatible)


if __name__ == "__main__":
    unittest.main()
