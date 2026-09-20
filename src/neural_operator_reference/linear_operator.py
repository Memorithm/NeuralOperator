"""Closed-form linear Fourier operator used as a pre-FNO baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class FourierMultiplier1D:
    """A fitted diagonal Fourier operator for periodic 1D fields.

    This is intentionally *not* a neural operator. It is a controlled linear
    baseline: each retained Fourier mode receives one learned complex scalar.
    The mode-index representation allows a fitted operator to be evaluated on
    a different uniform resolution, provided the physical length and the
    represented mode range remain valid.
    """

    length: float = 2.0 * np.pi
    max_mode: int = 16
    multipliers: Dict[int, complex] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be finite and strictly positive")
        if not isinstance(self.max_mode, (int, np.integer)) or self.max_mode < 0:
            raise ValueError("max_mode must be a non-negative integer")
        self.length = float(self.length)
        self.max_mode = int(self.max_mode)

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> "FourierMultiplier1D":
        """Fit complex mode multipliers by independent least squares."""

        inputs, targets = np.broadcast_arrays(np.asarray(inputs), np.asarray(targets))
        if inputs.ndim < 2:
            raise ValueError("fit expects samples followed by a spatial axis")
        if not np.isrealobj(inputs) or not np.isrealobj(targets):
            raise ValueError("the baseline currently expects real-valued fields")
        n = inputs.shape[-1]
        if n < 2:
            raise ValueError("the spatial axis must contain at least two points")

        input_spectrum = np.fft.fft(inputs, axis=-1)
        target_spectrum = np.fft.fft(targets, axis=-1)
        mode_indices = np.rint(np.fft.fftfreq(n) * n).astype(int)
        self.multipliers.clear()
        for index, mode in enumerate(mode_indices):
            if abs(int(mode)) > self.max_mode:
                continue
            source = input_spectrum[..., index].reshape(-1)
            target = target_spectrum[..., index].reshape(-1)
            denominator = float(np.vdot(source, source).real)
            if denominator <= np.finfo(float).eps:
                self.multipliers[int(mode)] = 0.0j
            else:
                self.multipliers[int(mode)] = complex(
                    np.vdot(source, target) / denominator
                )
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Apply the fitted operator at the input's current resolution."""

        inputs = np.asarray(inputs)
        if inputs.ndim == 0:
            raise ValueError("inputs must contain a spatial axis")
        if not np.isrealobj(inputs):
            raise ValueError("the baseline currently expects real-valued fields")
        n = inputs.shape[-1]
        if n < 2:
            raise ValueError("the spatial axis must contain at least two points")
        if not self.multipliers:
            raise RuntimeError("fit must be called before predict")

        input_spectrum = np.fft.fft(inputs, axis=-1)
        output_spectrum = np.zeros_like(input_spectrum, dtype=np.complex128)
        mode_indices = np.rint(np.fft.fftfreq(n) * n).astype(int)
        for index, mode in enumerate(mode_indices):
            multiplier = self.multipliers.get(int(mode))
            if multiplier is not None:
                output_spectrum[..., index] = input_spectrum[..., index] * multiplier
        return np.fft.ifft(output_spectrum, axis=-1).real
