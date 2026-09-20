"""Non-neural baselines for periodic resolution-transfer experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def periodic_linear_interpolate(
    field: np.ndarray,
    target_points: int,
    length: float = 2.0 * np.pi,
) -> np.ndarray:
    """Resample a periodic field on a uniform grid by linear interpolation.

    The spatial axis is the last axis. The endpoint at ``length`` is not
    duplicated: interpolation wraps from the last source point to the first
    source point, which preserves periodicity.
    """

    field = np.asarray(field, dtype=float)
    if field.ndim == 0:
        raise ValueError("field must contain a spatial axis")
    source_points = field.shape[-1]
    if source_points < 2:
        raise ValueError("at least two source points are required")
    if not isinstance(target_points, (int, np.integer)) or target_points < 2:
        raise ValueError("target_points must be an integer greater than or equal to 2")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and strictly positive")

    scaled_positions = (
        np.arange(int(target_points), dtype=float) * source_points / int(target_points)
    )
    left = np.floor(scaled_positions).astype(int) % source_points
    fraction = scaled_positions - np.floor(scaled_positions)
    right = (left + 1) % source_points
    return (1.0 - fraction) * field[..., left] + fraction * field[..., right]


@dataclass(frozen=True)
class ConvolutionFitResult:
    """Summary of a closed-form periodic convolution fit."""

    initial_loss: float
    final_loss: float
    rank: int


class PeriodicConv1D:
    """Small local linear convolution baseline with periodic wrapping.

    The stencil is learned on the discrete grid by least squares. Its weights
    are intentionally grid-local, so transferring them to another resolution
    is a baseline rather than a resolution-invariance guarantee.
    """

    def __init__(self, kernel_size: int = 5) -> None:
        if (
            not isinstance(kernel_size, (int, np.integer))
            or kernel_size <= 0
            or kernel_size % 2 == 0
        ):
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = int(kernel_size)
        self.kernel = np.zeros(self.kernel_size, dtype=float)
        self.bias = 0.0

    def _validate_inputs(
        self, inputs: np.ndarray, targets: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 3 or inputs.shape[-1] != 1:
            raise ValueError("inputs must have shape (batch, points, 1)")
        if inputs.shape[1] < self.kernel_size:
            raise ValueError("the grid must contain at least kernel_size points")
        if targets is None:
            return inputs, None
        targets = np.asarray(targets, dtype=float)
        if targets.shape != inputs.shape:
            raise ValueError("targets must have the same shape as inputs")
        return inputs, targets

    def _features(self, inputs: np.ndarray) -> np.ndarray:
        radius = self.kernel_size // 2
        return np.stack(
            [
                np.roll(inputs[..., 0], shift=offset, axis=1)
                for offset in range(-radius, radius + 1)
            ],
            axis=-1,
        )

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        inputs, _ = self._validate_inputs(inputs)
        features = self._features(inputs)
        output = np.einsum("bpk,k->bp", features, self.kernel) + self.bias
        return output[..., None]

    def loss(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        inputs, targets = self._validate_inputs(inputs, targets)
        return float(np.mean((self(inputs) - targets) ** 2))

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> ConvolutionFitResult:
        inputs, targets = self._validate_inputs(inputs, targets)
        initial_loss = self.loss(inputs, targets)
        features = self._features(inputs).reshape(-1, self.kernel_size)
        design = np.column_stack([features, np.ones(features.shape[0])])
        solution, _, rank, _ = np.linalg.lstsq(
            design, targets[..., 0].reshape(-1), rcond=None
        )
        self.kernel[...] = solution[:-1]
        self.bias = float(solution[-1])
        return ConvolutionFitResult(
            initial_loss=float(initial_loss),
            final_loss=float(self.loss(inputs, targets)),
            rank=int(rank),
        )
