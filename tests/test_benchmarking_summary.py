import unittest

import numpy as np

from neural_operator_reference import benchmark_inference, summarize_scalars


class ScalarSummaryTests(unittest.TestCase):
    def test_summary_reports_sample_statistics(self):
        summary = summarize_scalars([1.0, 2.0, 4.0])

        self.assertEqual(summary.count, 3)
        self.assertAlmostEqual(summary.mean, 7.0 / 3.0)
        self.assertAlmostEqual(summary.sample_std, np.std([1.0, 2.0, 4.0], ddof=1))
        self.assertEqual(summary.minimum, 1.0)
        self.assertEqual(summary.maximum, 4.0)

    def test_single_value_has_zero_sample_deviation(self):
        summary = summarize_scalars([3.5])

        self.assertEqual(summary.count, 1)
        self.assertEqual(summary.sample_std, 0.0)

    def test_empty_and_nonfinite_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_scalars([])
        with self.assertRaises(ValueError):
            summarize_scalars([1.0, np.nan])


class InferenceTimingContractTests(unittest.TestCase):
    def test_timing_is_labeled_end_to_end_with_variability_and_context(self):
        inputs = np.ones((2, 8, 1), dtype=float)

        prediction, timing = benchmark_inference(
            lambda values: values.copy(),
            inputs,
            repeats=3,
        )

        self.assertTrue(np.array_equal(prediction, inputs))
        self.assertEqual(timing.repeats, 3)
        self.assertEqual(timing.scope, "array_operator_end_to_end")
        self.assertEqual(timing.warmup_calls, 1)
        self.assertTrue(timing.includes_output_conversion)
        self.assertGreaterEqual(timing.sample_std_seconds, 0.0)
        self.assertLessEqual(timing.minimum_seconds, timing.seconds_per_call)
        self.assertGreaterEqual(timing.maximum_seconds, timing.seconds_per_call)
        self.assertAlmostEqual(
            timing.total_seconds,
            timing.seconds_per_call * timing.repeats,
        )
        self.assertIn("platform", timing.context)
        self.assertIn("machine", timing.context)
        self.assertIn("threading_environment", timing.context)
        self.assertIn("backend_threading", timing.context)
        self.assertTrue(
            any("not kernel-only" in item for item in timing.context["limitations"])
        )

    def test_serialized_timing_keeps_measurement_boundaries(self):
        inputs = np.zeros((1, 4, 1), dtype=float)

        _, timing = benchmark_inference(lambda values: values, inputs, repeats=2)
        payload = timing.as_dict()

        self.assertEqual(payload["scope"], "array_operator_end_to_end")
        self.assertEqual(payload["repeats"], 2)
        self.assertIn("sample_std_seconds", payload)
        self.assertIn("minimum_seconds", payload)
        self.assertIn("maximum_seconds", payload)
        self.assertIn("context", payload)


if __name__ == "__main__":
    unittest.main()
