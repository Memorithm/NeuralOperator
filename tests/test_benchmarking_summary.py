import unittest

import numpy as np

from neural_operator_reference import summarize_scalars


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


if __name__ == "__main__":
    unittest.main()
