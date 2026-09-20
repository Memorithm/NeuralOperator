import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    BurgersDatasetSpec,
    burgers_rhs,
    generate_burgers_dataset,
    generate_burgers_split,
    periodic_forcing_field,
)


class BurgersForcingTests(unittest.TestCase):
    def test_static_forcing_is_added_to_the_reference_rhs(self):
        points = 32
        forcing = periodic_forcing_field(
            points=points,
            amplitude=0.25,
            mode=2,
        )
        actual = burgers_rhs(
            np.zeros(points),
            viscosity=0.05,
            forcing=forcing,
        )
        self.assertTrue(np.allclose(actual, forcing))

    def test_zero_forcing_preserves_the_unforced_dataset(self):
        kwargs = {
            "samples": 2,
            "points": 16,
            "dt": 2.0e-4,
            "steps": 3,
            "viscosity": 0.05,
            "seed": 7,
        }
        unforced = generate_burgers_dataset(**kwargs)
        explicitly_zero = generate_burgers_dataset(
            **kwargs,
            forcing_amplitude=0.0,
        )
        self.assertTrue(np.array_equal(unforced.targets, explicitly_zero.targets))
        self.assertEqual(explicitly_zero.forcing_amplitude, 0.0)
        self.assertEqual(explicitly_zero.forcing_mode, 1)

    def test_ood_split_retains_forcing_provenance(self):
        common = {
            "points": 16,
            "dt": 2.0e-4,
            "steps": 2,
            "length": 2.0 * np.pi,
        }
        train = BurgersDatasetSpec(
            samples=2,
            viscosity=0.05,
            seed=1,
            **common,
        )
        validation = BurgersDatasetSpec(
            samples=2,
            viscosity=0.05,
            seed=2,
            **common,
        )
        ood = BurgersDatasetSpec(
            samples=2,
            viscosity=0.1,
            seed=3,
            modes=6,
            forcing_amplitude=0.2,
            forcing_mode=2,
            **common,
        )
        split = generate_burgers_split(train, validation, ood)

        self.assertEqual(split.as_dict()["ood"]["forcing_amplitude"], 0.2)
        self.assertEqual(split.as_dict()["ood"]["forcing_mode"], 2)
        self.assertTrue(np.any(np.abs(split.ood.forcing) > 0.0))


if __name__ == "__main__":
    unittest.main()
