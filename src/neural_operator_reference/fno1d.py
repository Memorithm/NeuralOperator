"""Minimal NumPy/SciPy 1D Fourier Neural Operator reference.

This implementation is intentionally small and slow. It exists to validate
the FNO tensor contract and training protocol before introducing a tensor
framework or a Rust backend. The optimizer uses numerical gradients through
SciPy, so this is not a production training implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class FNOFitResult:
    """Stable summary of a reference fit."""

    initial_loss: float
    final_loss: float
    success: bool
    iterations: int
    function_evaluations: int
    message: str


class NumpyFNO1D:
    """A one-block FNO for real-valued periodic 1D fields.

    Input and output layouts are ``(batch, points, channels)``. The Fourier
    block uses ``rfft`` and keeps the first ``modes`` non-negative modes. A
    real mode-mixing matrix is shared with the implicit conjugate modes, which
    guarantees a real inverse transform.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 8,
        modes: int = 8,
        seed: int = 0,
    ) -> None:
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("width", width),
            ("modes", modes),
        ):
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.modes = int(modes)
        rng = np.random.default_rng(seed)

        def init(shape: tuple[int, ...]) -> np.ndarray:
            return rng.normal(0.0, 0.15, size=shape)

        self.lift_weight = init((self.in_channels, self.width))
        self.lift_bias = np.zeros(self.width)
        self.spectral_weight = init((self.modes, self.width, self.width))
        self.pointwise_weight = init((self.width, self.width))
        self.pointwise_bias = np.zeros(self.width)
        self.projection_weight = init((self.width, self.out_channels))
        self.projection_bias = np.zeros(self.out_channels)

    def _validate_inputs(
        self, inputs: np.ndarray, targets: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, points, channels)")
        if inputs.shape[-1] != self.in_channels:
            raise ValueError("input channel count does not match the model")
        if inputs.shape[1] < 2:
            raise ValueError("at least two spatial points are required")
        if targets is None:
            return inputs, None
        targets = np.asarray(targets, dtype=float)
        expected_shape = (inputs.shape[0], inputs.shape[1], self.out_channels)
        if targets.shape != expected_shape:
            raise ValueError(f"targets must have shape {expected_shape}")
        return inputs, targets

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the FNO at the input resolution."""

        inputs, _ = self._validate_inputs(inputs)
        hidden = np.einsum("bnc,cw->bnw", inputs, self.lift_weight)
        hidden = hidden + self.lift_bias

        spectrum = np.fft.rfft(hidden, axis=1)
        filtered_spectrum = np.zeros_like(spectrum)
        usable_modes = min(self.modes, spectrum.shape[1])
        for mode in range(usable_modes):
            filtered_spectrum[:, mode, :] = (
                spectrum[:, mode, :] @ self.spectral_weight[mode]
            )
        spectral = np.fft.irfft(filtered_spectrum, n=inputs.shape[1], axis=1)

        pointwise = np.einsum("bnw,wv->bnv", hidden, self.pointwise_weight)
        pointwise = pointwise + self.pointwise_bias
        hidden = np.tanh(spectral + pointwise)
        return np.einsum("bnw,wo->bno", hidden, self.projection_weight) + self.projection_bias

    __call__ = forward

    def loss(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Return mean squared error for a batch."""

        inputs, targets = self._validate_inputs(inputs, targets)
        prediction = self.forward(inputs)
        return float(np.mean((prediction - targets) ** 2))

    def _parameter_arrays(self) -> List[np.ndarray]:
        return [
            self.lift_weight,
            self.lift_bias,
            self.spectral_weight,
            self.pointwise_weight,
            self.pointwise_bias,
            self.projection_weight,
            self.projection_bias,
        ]

    def parameter_vector(self) -> np.ndarray:
        """Flatten parameters in a deterministic order for SciPy."""

        return np.concatenate([parameter.ravel() for parameter in self._parameter_arrays()])

    def set_parameter_vector(self, vector: np.ndarray) -> None:
        """Restore parameters from :meth:`parameter_vector`."""

        vector = np.asarray(vector, dtype=float)
        expected = sum(parameter.size for parameter in self._parameter_arrays())
        if vector.size != expected:
            raise ValueError(f"parameter vector must contain {expected} values")
        offset = 0
        for parameter in self._parameter_arrays():
            next_offset = offset + parameter.size
            parameter[...] = vector[offset:next_offset].reshape(parameter.shape)
            offset = next_offset

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        maxiter: int = 80,
        tolerance: float = 1.0e-9,
    ) -> FNOFitResult:
        """Fit the reference model with SciPy's finite-difference L-BFGS-B.

        The finite-difference optimizer is deliberately used only for this
        tiny reference. It makes the training path dependency-light while
        keeping the production implementation open for PyTorch or Rust.
        """

        inputs, targets = self._validate_inputs(inputs, targets)
        if not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
            raise ValueError("maxiter must be a positive integer")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and strictly positive")
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("scipy is required to train NumpyFNO1D") from exc

        initial_vector = self.parameter_vector()
        initial_loss = self.loss(inputs, targets)

        def objective(vector: np.ndarray) -> float:
            self.set_parameter_vector(vector)
            return self.loss(inputs, targets)

        result = minimize(
            objective,
            initial_vector,
            method="L-BFGS-B",
            options={
                "maxiter": int(maxiter),
                "ftol": float(tolerance),
                "gtol": float(tolerance),
                "maxls": 20,
            },
        )
        self.set_parameter_vector(result.x)
        return FNOFitResult(
            initial_loss=float(initial_loss),
            final_loss=float(self.loss(inputs, targets)),
            success=bool(result.success),
            iterations=int(getattr(result, "nit", 0)),
            function_evaluations=int(getattr(result, "nfev", 0)),
            message=str(result.message),
        )
