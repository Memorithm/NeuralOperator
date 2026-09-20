import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from neural_operator_reference import (  # noqa: E402
    FourierMultiplier1D,
    NumpyFNO1D,
    burgers_rhs,
    generate_burgers_dataset,
    integrate_burgers,
    burgers_residual,
    central_difference_periodic,
    dealias_mask,
    dealiased_product,
    divergence_2d,
    incompressible_velocity_from_streamfunction,
    relative_l2_error,
    spectral_derivative,
    spectral_filter,
)


class SpectralReferenceTests(unittest.TestCase):
    def test_first_and_second_derivatives_of_smooth_periodic_field(self):
        n = 128
        length = 2.0 * np.pi
        x = np.arange(n) * length / n
        field = np.sin(3.0 * x) + 0.25 * np.cos(5.0 * x)
        expected_first = 3.0 * np.cos(3.0 * x) - 1.25 * np.sin(5.0 * x)
        expected_second = -9.0 * np.sin(3.0 * x) - 6.25 * np.cos(5.0 * x)

        first = spectral_derivative(field, order=1, length=length)
        second = spectral_derivative(field, order=2, length=length)

        self.assertLess(relative_l2_error(first, expected_first), 1.0e-12)
        self.assertLess(relative_l2_error(second, expected_second), 1.0e-12)

    def test_batched_axis_is_supported(self):
        n = 64
        x = np.arange(n) * 2.0 * np.pi / n
        fields = np.stack([np.sin(x), np.cos(2.0 * x)])
        expected = np.stack([np.cos(x), -2.0 * np.sin(2.0 * x)])
        actual = spectral_derivative(fields, length=2.0 * np.pi, axis=-1)
        self.assertLess(relative_l2_error(actual, expected), 1.0e-12)

    def test_streamfunction_construction_is_divergence_free(self):
        nx, ny = 64, 48
        lx, ly = 2.0 * np.pi, 2.0 * np.pi
        x = np.arange(nx) * lx / nx
        y = np.arange(ny) * ly / ny
        xx, yy = np.meshgrid(x, y, indexing="ij")
        psi = np.sin(3.0 * xx) * np.cos(2.0 * yy)

        u, v = incompressible_velocity_from_streamfunction(
            psi, lengths=(lx, ly), spatial_axes=(0, 1)
        )
        divergence = divergence_2d(
            u, v, lengths=(lx, ly), spatial_axes=(0, 1)
        )
        self.assertLess(float(np.max(np.abs(divergence))), 1.0e-11)

    def test_burgers_residual_matches_analytic_expression(self):
        n = 96
        length = 2.0 * np.pi
        x = np.arange(n) * length / n
        viscosity = 0.1
        u = np.sin(x)
        expected = np.sin(x) * np.cos(x) + viscosity * np.sin(x)
        actual = burgers_residual(u, np.zeros_like(u), viscosity, length=length)
        self.assertLess(relative_l2_error(actual, expected), 1.0e-12)

    def test_two_thirds_filter_removes_high_mode(self):
        n = 96
        x = np.arange(n) * 2.0 * np.pi / n
        high_mode = np.sin(30.0 * x)
        self.assertTrue(bool(dealias_mask(n)[30]))
        filtered = spectral_filter(np.sin(40.0 * x), keep_ratio=2.0 / 3.0)
        self.assertLess(float(np.max(np.abs(filtered))), 1.0e-12)
        product = dealiased_product(high_mode, high_mode, keep_ratio=2.0 / 3.0)
        self.assertLess(relative_l2_error(product, np.full(n, 0.5)), 1.0e-12)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            spectral_derivative(np.ones(8), order=-1)
        with self.assertRaises(ValueError):
            spectral_filter(np.ones(8), keep_ratio=0.0)
        with self.assertRaises(ValueError):
            burgers_residual(np.ones(8), np.zeros(8), viscosity=-1.0)

    def test_linear_fourier_operator_transfers_to_a_finer_resolution(self):
        length = 2.0 * np.pi
        viscosity_like_alpha = 0.07
        coarse_n, fine_n = 64, 128
        rng = np.random.default_rng(20260920)

        def make_fields(n, samples):
            x = np.arange(n) * length / n
            fields = []
            for _ in range(samples):
                field = np.zeros(n)
                for mode in range(1, 5):
                    field += rng.normal() * np.sin(mode * x)
                    field += rng.normal() * np.cos(mode * x)
                fields.append(field)
            return np.asarray(fields)

        def elliptic_response(fields):
            n = fields.shape[-1]
            modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
            spectrum = np.fft.fft(fields, axis=-1)
            response = np.array(
                [1.0 / (1.0 + viscosity_like_alpha * mode * mode) for mode in modes]
            )
            return np.fft.ifft(spectrum * response, axis=-1).real

        coarse_inputs = make_fields(coarse_n, samples=24)
        coarse_targets = elliptic_response(coarse_inputs)
        operator = FourierMultiplier1D(length=length, max_mode=8).fit(
            coarse_inputs, coarse_targets
        )

        fine_inputs = make_fields(fine_n, samples=8)
        fine_targets = elliptic_response(fine_inputs)
        fine_predictions = operator.predict(fine_inputs)
        self.assertLess(relative_l2_error(fine_predictions, fine_targets), 1.0e-12)

    def test_spectral_derivative_converges_faster_than_centered_difference(self):
        length = 2.0 * np.pi
        errors_spectral = []
        errors_finite_difference = []
        for n in (16, 32, 64):
            x = np.arange(n) * length / n
            field = np.exp(np.sin(x))
            expected = np.cos(x) * np.exp(np.sin(x))
            errors_spectral.append(
                relative_l2_error(spectral_derivative(field, length=length), expected)
            )
            errors_finite_difference.append(
                relative_l2_error(
                    central_difference_periodic(field, length=length), expected
                )
            )

        self.assertLess(errors_spectral[-1], errors_spectral[0] * 1.0e-4)
        self.assertLess(errors_finite_difference[-1], errors_finite_difference[0])
        self.assertLess(errors_spectral[-1], errors_finite_difference[-1])

    def test_fno_reference_learns_a_low_mode_operator(self):
        length = 2.0 * np.pi
        n = 12
        x = np.arange(n) * length / n
        rng = np.random.default_rng(20260920)
        inputs = []
        targets = []
        for _ in range(4):
            field = (
                rng.normal() * np.sin(x)
                + rng.normal() * np.cos(x)
                + rng.normal() * np.sin(2.0 * x)
            )
            inputs.append(field[:, None])
            spectrum = np.fft.rfft(field)
            response = np.zeros_like(spectrum, dtype=float)
            response[:3] = (0.2, 0.5, 0.1)
            targets.append(np.fft.irfft(spectrum * response, n=n)[:, None])
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)

        model = NumpyFNO1D(width=3, modes=4, seed=4)
        result = model.fit(inputs, targets, maxiter=150, tolerance=1.0e-8)

        self.assertLess(result.final_loss, result.initial_loss)
        self.assertLess(result.final_loss, 5.0e-4)

        fine_n = 24
        fine_x = np.arange(fine_n) * length / fine_n
        fine_inputs = []
        fine_targets = []
        for _ in range(2):
            field = (
                rng.normal() * np.sin(fine_x)
                + rng.normal() * np.cos(fine_x)
                + rng.normal() * np.sin(2.0 * fine_x)
            )
            spectrum = np.fft.rfft(field)
            response = np.zeros_like(spectrum, dtype=float)
            response[:3] = (0.2, 0.5, 0.1)
            fine_inputs.append(field[:, None])
            fine_targets.append(np.fft.irfft(spectrum * response, n=fine_n)[:, None])
        fine_inputs = np.asarray(fine_inputs)
        fine_targets = np.asarray(fine_targets)
        self.assertLess(model.loss(fine_inputs, fine_targets), 5.0e-4)

    def test_burgers_solver_preserves_mean_and_dissipates_energy(self):
        n = 64
        length = 2.0 * np.pi
        x = np.arange(n) * length / n
        initial = 0.2 + np.sin(x) + 0.25 * np.cos(2.0 * x)
        trajectory = integrate_burgers(
            initial,
            dt=2.0e-4,
            steps=40,
            viscosity=0.05,
            length=length,
        )

        means = np.mean(trajectory, axis=-1)
        energies = 0.5 * np.mean(trajectory * trajectory, axis=-1)
        self.assertLess(float(np.max(np.abs(means - means[0]))), 1.0e-11)
        self.assertLess(float(energies[-1]), float(energies[0]))
        self.assertLess(float(np.max(np.abs(burgers_rhs(np.ones(n), 0.1)))), 1.0e-12)

    def test_burgers_solver_rejects_invalid_time_parameters(self):
        with self.assertRaises(ValueError):
            integrate_burgers(np.ones(16), dt=0.0, steps=1, viscosity=0.1)
        with self.assertRaises(ValueError):
            integrate_burgers(np.ones(16), dt=1.0e-3, steps=-1, viscosity=0.1)

    def test_burgers_dataset_is_deterministic_and_traceable(self):
        first = generate_burgers_dataset(
            samples=3, points=32, dt=2.0e-4, steps=3, viscosity=0.05, seed=17
        )
        second = generate_burgers_dataset(
            samples=3, points=32, dt=2.0e-4, steps=3, viscosity=0.05, seed=17
        )
        self.assertEqual(first.inputs.shape, (3, 32, 1))
        self.assertEqual(first.targets.shape, (3, 32, 1))
        self.assertEqual(first.resolution, 32)
        self.assertEqual(first.samples, 3)
        self.assertEqual(first.steps, 3)
        self.assertTrue(np.array_equal(first.inputs, second.inputs))
        self.assertTrue(np.array_equal(first.targets, second.targets))
        self.assertLess(float(np.max(np.abs(first.inputs - first.targets))), 0.1)

    def test_fno_learns_a_small_burgers_operator(self):
        train = generate_burgers_dataset(
            samples=4, points=16, dt=2.0e-4, steps=3, viscosity=0.05, seed=1
        )
        held_out = generate_burgers_dataset(
            samples=2, points=16, dt=2.0e-4, steps=3, viscosity=0.05, seed=2
        )
        model = NumpyFNO1D(width=3, modes=5, seed=4)
        result = model.fit(train.inputs, train.targets, maxiter=60, tolerance=1.0e-7)

        self.assertLess(result.final_loss, result.initial_loss)
        self.assertLess(model.loss(held_out.inputs, held_out.targets), 1.0e-3)


if __name__ == "__main__":
    unittest.main()
