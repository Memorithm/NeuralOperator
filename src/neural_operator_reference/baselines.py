"""Non-neural baselines for periodic resolution-transfer experiments."""

from __future__ import annotations

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
