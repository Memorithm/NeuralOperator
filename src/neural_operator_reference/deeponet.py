"""A transparent separable DeepONet-style NumPy reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fourier_trunk_features(
    coordinates: np.ndarray,
    modes: int,
    length: float = 2.0 * np.pi,
) -> np.ndarray:
    """Evaluate a constant-plus-Fourier trunk basis at arbitrary coordinates."""

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("coordinates must be a non-empty one-dimensional array")
    if not isinstance(modes, (int, np.integer)) or modes <= 0:
        raise ValueError("modes must be a positive integer")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and strictly positive")

    features = np.empty((coordinates.size, 1 + 2 * int(modes)), dtype=float)
    features[:, 0] = 1.0
    for mode in range(1, int(modes) + 1):
        angle = 2.0 * np.pi * mode * coordinates / float(length)
        features[:, 2 * mode - 1] = np.sin(angle)
        features[:, 2 * mode] = np.cos(angle)
    return features


@dataclass(frozen=True)
class DeepONetFitResult:
    """Summary of a closed-form separable DeepONet fit."""

    initial_loss: float
    final_loss: float
    rank: int


class LinearDeepONet1D:
    """A linear branch plus fixed Fourier trunk DeepONet reference.

    The branch maps fixed sensor values to latent coefficients. The trunk is a
    fixed Fourier basis evaluated at query coordinates, so the fitted operator
    can return a different number of output points without refitting. This is
    deliberately a contract/reference model, not a nonlinear production
    DeepONet.
    """

    def __init__(
        self,
        sensor_points: int,
        modes: int = 4,
        length: float = 2.0 * np.pi,
    ) -> None:
        if not isinstance(sensor_points, (int, np.integer)) or sensor_points < 2:
            raise ValueError("sensor_points must be an integer greater than or equal to 2")
        if not isinstance(modes, (int, np.integer)) or modes <= 0:
            raise ValueError("modes must be a positive integer")
        if modes >= sensor_points // 2:
            raise ValueError("modes must be below the sensor Nyquist limit")
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("length must be finite and strictly positive")
        self.sensor_points = int(sensor_points)
        self.modes = int(modes)
        self.length = float(length)
        self.latent_width = 1 + 2 * self.modes
        self.branch_weight = np.zeros(
            (self.sensor_points, self.latent_width), dtype=float
        )
        self.branch_bias = np.zeros(self.latent_width, dtype=float)

    def _validate_inputs(
        self, inputs: np.ndarray, targets: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 3 or inputs.shape[1:] != (self.sensor_points, 1):
            raise ValueError(
                "inputs must have shape (batch, sensor_points, 1)"
            )
        if targets is None:
            return inputs, None
        targets = np.asarray(targets, dtype=float)
        if targets.ndim != 3 or targets.shape[0] != inputs.shape[0] or targets.shape[-1] != 1:
            raise ValueError("targets must have shape (batch, query_points, 1)")
        return inputs, targets

    def trunk_features(self, coordinates: np.ndarray) -> np.ndarray:
        """Return trunk features for a one-dimensional query grid."""

        return fourier_trunk_features(coordinates, self.modes, length=self.length)

    def _resolve_coordinates(
        self,
        coordinates: np.ndarray | None,
        target_points: int | None,
    ) -> np.ndarray:
        if coordinates is not None and target_points is not None:
            raise ValueError("provide coordinates or target_points, not both")
        if coordinates is not None:
            coordinates = np.asarray(coordinates, dtype=float)
            if coordinates.ndim != 1 or coordinates.size == 0:
                raise ValueError("coordinates must be a non-empty one-dimensional array")
            return coordinates
        points = self.sensor_points if target_points is None else target_points
        if not isinstance(points, (int, np.integer)) or points < 2:
            raise ValueError("target_points must be an integer greater than or equal to 2")
        return np.arange(int(points), dtype=float) * self.length / int(points)

    def __call__(
        self,
        inputs: np.ndarray,
        coordinates: np.ndarray | None = None,
        target_points: int | None = None,
    ) -> np.ndarray:
        inputs, _ = self._validate_inputs(inputs)
        query_coordinates = self._resolve_coordinates(coordinates, target_points)
        branch = np.einsum("bs,sl->bl", inputs[..., 0], self.branch_weight)
        branch = branch + self.branch_bias
        trunk = self.trunk_features(query_coordinates)
        output = np.einsum("bl,pl->bp", branch, trunk)
        return output[..., None]

    def loss(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        coordinates: np.ndarray | None = None,
    ) -> float:
        inputs, targets = self._validate_inputs(inputs, targets)
        prediction = self(inputs, coordinates=coordinates)
        if prediction.shape != targets.shape:
            raise ValueError("target query points do not match the prediction")
        return float(np.mean((prediction - targets) ** 2))

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        coordinates: np.ndarray | None = None,
    ) -> DeepONetFitResult:
        inputs, targets = self._validate_inputs(inputs, targets)
        query_coordinates = self._resolve_coordinates(
            coordinates, None if coordinates is not None else targets.shape[1]
        )
        if query_coordinates.size != targets.shape[1]:
            raise ValueError("coordinates must have one value per target point")
        initial_loss = self.loss(inputs, targets, coordinates=query_coordinates)
        trunk = self.trunk_features(query_coordinates)
        branch_features = np.einsum(
            "bs,pl->bpsl", inputs[..., 0], trunk
        ).reshape(-1, self.sensor_points * self.latent_width)
        bias_features = np.broadcast_to(
            trunk[None, :, :], (inputs.shape[0], trunk.shape[0], trunk.shape[1])
        ).reshape(-1, self.latent_width)
        design = np.column_stack([branch_features, bias_features])
        solution, _, rank, _ = np.linalg.lstsq(
            design, targets[..., 0].reshape(-1), rcond=None
        )
        weight_size = self.sensor_points * self.latent_width
        self.branch_weight[...] = solution[:weight_size].reshape(
            self.sensor_points, self.latent_width
        )
        self.branch_bias[...] = solution[weight_size:]
        return DeepONetFitResult(
            initial_loss=float(initial_loss),
            final_loss=float(self.loss(inputs, targets, coordinates=query_coordinates)),
            rank=int(rank),
        )
