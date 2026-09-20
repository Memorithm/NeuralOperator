import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    burgers_rk4_step,
    evaluate_burgers_operator,
    integrate_burgers,
    random_periodic_fields,
    rollout_burgers_operator,
)


class BurgersRolloutEvaluationTests(unittest.TestCase):
    def test_exact_reference_transition_has_zero_state_error(self):
        horizon = 2.0e-4
        viscosity = 0.05
        length = 2.0 * np.pi
        initial_fields = random_periodic_fields(
            samples=2,
            points=16,
            modes=3,
            seed=21,
            amplitude=0.2,
            length=length,
        )
        reference = integrate_burgers(
            initial_fields,
            dt=horizon,
            steps=3,
            viscosity=viscosity,
            length=length,
        )

        def exact_operator(inputs):
            return burgers_rk4_step(
                inputs[..., 0],
                dt=horizon,
                viscosity=viscosity,
                length=length,
            )[..., None]

        predicted, metrics = evaluate_burgers_operator(
            exact_operator,
            initial_fields,
            reference,
            horizon=horizon,
            viscosity=viscosity,
            length=length,
        )

        self.assertEqual(predicted.shape, (4, 2, 16))
        self.assertEqual(len(metrics.mse_by_step), 4)
        self.assertEqual(len(metrics.physics_mse_by_transition), 3)
        self.assertLess(metrics.max_relative_l2, 1.0e-14)
        self.assertLess(max(metrics.mean_abs_error_by_step), 1.0e-14)
        self.assertLess(max(metrics.energy_abs_error_by_step), 1.0e-14)

    def test_rollout_rejects_an_operator_with_the_wrong_shape(self):
        initial_fields = np.ones((1, 8), dtype=float)

        def wrong_operator(inputs):
            return inputs[..., 0]

        with self.assertRaises(ValueError):
            rollout_burgers_operator(wrong_operator, initial_fields, steps=1)

    def test_zero_step_rollout_is_the_initial_state(self):
        initial_fields = np.arange(12, dtype=float).reshape(2, 6)
        trajectory = rollout_burgers_operator(
            lambda inputs: inputs,
            initial_fields,
            steps=0,
        )
        self.assertTrue(np.array_equal(trajectory[0], initial_fields))


if __name__ == "__main__":
    unittest.main()
