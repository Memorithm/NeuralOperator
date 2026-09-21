import unittest

import numpy as np

from neural_operator_reference import generate_darcy_dataset, random_log_permeability


class DarcyOODFamilyTests(unittest.TestCase):
    @staticmethod
    def _roughness(permeability):
        log_field = np.log(permeability)
        return float(
            np.mean(np.diff(log_field, axis=1) ** 2)
            + np.mean(np.diff(log_field, axis=2) ** 2)
        )

    def test_default_spectral_decay_preserves_historical_generator(self):
        default = random_log_permeability(
            samples=2,
            points_x=17,
            modes=6,
            seed=3,
            log_std=0.45,
        )
        explicit = random_log_permeability(
            samples=2,
            points_x=17,
            modes=6,
            seed=3,
            log_std=0.45,
            spectral_decay=2.0,
        )
        np.testing.assert_array_equal(default, explicit)

    def test_lower_spectral_decay_generates_rougher_fields(self):
        rough = random_log_permeability(
            samples=2,
            points_x=17,
            modes=6,
            seed=3,
            log_std=0.45,
            spectral_decay=0.5,
        )
        smooth = random_log_permeability(
            samples=2,
            points_x=17,
            modes=6,
            seed=3,
            log_std=0.45,
            spectral_decay=4.0,
        )

        self.assertGreater(self._roughness(rough), 2.0 * self._roughness(smooth))

    def test_dataset_records_spectral_family_provenance(self):
        dataset = generate_darcy_dataset(
            samples=1,
            points_x=9,
            modes=3,
            seed=8,
            spectral_decay=1.0,
        )
        self.assertEqual(dataset.spectral_decay, 1.0)

    def test_negative_spectral_decay_is_rejected(self):
        with self.assertRaises(ValueError):
            random_log_permeability(
                samples=1,
                points_x=9,
                modes=3,
                spectral_decay=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
