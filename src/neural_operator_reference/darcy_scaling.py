"""Exact global permeability-scale normalization for scalar Darcy flow."""

from __future__ import annotations

import numpy as np

from .darcy_datasets import DarcyDataset


def _validated_permeability_batch(inputs: np.ndarray) -> np.ndarray:
    array = np.asarray(inputs)
    if not np.isrealobj(array):
        raise ValueError("Darcy inputs must be real-valued")
    array = array.astype(float, copy=False)
    if array.ndim != 4 or array.shape[-1] != 1:
        raise ValueError(
            "Darcy inputs must have shape (batch, points_y, points_x, 1)"
        )
    if array.shape[0] == 0 or array.shape[1] < 3 or array.shape[2] < 3:
        raise ValueError("Darcy inputs must contain non-empty spatial grids")
    if not np.all(np.isfinite(array)):
        raise ValueError("Darcy inputs must contain only finite values")
    if np.any(array <= 0.0):
        raise ValueError("Darcy permeability must be strictly positive")
    return array


def _validated_scales(scales: np.ndarray, batch: int) -> np.ndarray:
    array = np.asarray(scales, dtype=float)
    expected = (int(batch), 1, 1, 1)
    if array.shape != expected:
        raise ValueError(f"scales must have shape {expected}")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("scales must be finite and strictly positive")
    return array


def darcy_geometric_mean_scale(inputs: np.ndarray) -> np.ndarray:
    """Return per-sample geometric-mean permeability with broadcastable shape."""

    array = _validated_permeability_batch(inputs)
    log_mean = np.mean(np.log(array[..., 0]), axis=(1, 2), keepdims=True)
    return np.exp(log_mean)[..., None]


def normalize_darcy_permeability_scale(
    inputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the exact per-sample global permeability scale.

    For scalar Darcy flow with fixed forcing and homogeneous Dirichlet pressure,
    writing k = g * k_hat implies u_hat = g * u. The returned coefficient field
    therefore has geometric mean one and is accompanied by g.
    """

    array = _validated_permeability_batch(inputs)
    scales = darcy_geometric_mean_scale(array)
    return array / scales, scales


def normalize_darcy_pressure_scale(
    targets: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Map physical pressure u to normalized pressure u_hat = g * u."""

    targets = np.asarray(targets)
    if not np.isrealobj(targets):
        raise ValueError("Darcy targets must be real-valued")
    targets = targets.astype(float, copy=False)
    if targets.ndim != 4 or targets.shape[-1] != 1:
        raise ValueError(
            "Darcy targets must have shape (batch, points_y, points_x, 1)"
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError("Darcy targets must contain only finite values")
    scale_array = _validated_scales(scales, targets.shape[0])
    return targets * scale_array


def restore_darcy_pressure_scale(
    normalized_pressure: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Restore physical pressure u from normalized pressure u_hat."""

    pressure = np.asarray(normalized_pressure)
    if not np.isrealobj(pressure):
        raise ValueError("normalized pressure must be real-valued")
    pressure = pressure.astype(float, copy=False)
    if pressure.ndim != 4 or pressure.shape[-1] != 1:
        raise ValueError(
            "normalized pressure must have shape "
            "(batch, points_y, points_x, 1)"
        )
    if not np.all(np.isfinite(pressure)):
        raise ValueError("normalized pressure must contain only finite values")
    scale_array = _validated_scales(scales, pressure.shape[0])
    return pressure / scale_array


def normalize_darcy_dataset_scale(
    dataset: DarcyDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scale-normalized inputs/targets and their exact sample scales."""

    if not isinstance(dataset, DarcyDataset):
        raise TypeError("dataset must be a DarcyDataset")
    normalized_inputs, scales = normalize_darcy_permeability_scale(dataset.inputs)
    normalized_targets = normalize_darcy_pressure_scale(
        dataset.targets,
        scales,
    )
    return normalized_inputs, normalized_targets, scales
