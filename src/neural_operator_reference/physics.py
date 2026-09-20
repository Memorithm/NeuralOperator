"""Physics residuals used to audit operator predictions."""

from __future__ import annotations

import numpy as np

from .burgers import burgers_rhs


def burgers_transition_residual(
    initial_field: np.ndarray,
    predicted_field: np.ndarray,
    horizon: float,
    viscosity: float,
    length: float = 2.0 * np.pi,
) -> np.ndarray:
    """Approximate the Burgers residual over one predicted transition.

    The temporal derivative is approximated by
    ``(predicted_field - initial_field) / horizon`` and evaluated against the
    spatial RHS at the predicted state. This is a diagnostic for a one-step
    operator, not a replacement for a time-continuous PINO formulation.
    """

    initial_field, predicted_field = np.broadcast_arrays(
        np.asarray(initial_field, dtype=float),
        np.asarray(predicted_field, dtype=float),
    )
    if initial_field.ndim == 0:
        raise ValueError("fields must contain a spatial axis")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and strictly positive")
    temporal_derivative = (predicted_field - initial_field) / float(horizon)
    return temporal_derivative - burgers_rhs(
        predicted_field,
        viscosity=viscosity,
        length=length,
    )


def burgers_transition_physics_loss(
    initial_field: np.ndarray,
    predicted_field: np.ndarray,
    horizon: float,
    viscosity: float,
    length: float = 2.0 * np.pi,
) -> float:
    """Return the mean squared transition residual."""

    residual = burgers_transition_residual(
        initial_field,
        predicted_field,
        horizon=horizon,
        viscosity=viscosity,
        length=length,
    )
    return float(np.mean(residual * residual))


def burgers_data_physics_loss(
    initial_field: np.ndarray,
    predicted_field: np.ndarray,
    target_field: np.ndarray,
    horizon: float,
    viscosity: float,
    data_weight: float = 1.0,
    physics_weight: float = 1.0,
    length: float = 2.0 * np.pi,
) -> tuple[float, float, float]:
    """Return ``(total, data_mse, physics_mse)`` for a transition."""

    predicted_field, target_field = np.broadcast_arrays(
        np.asarray(predicted_field, dtype=float), np.asarray(target_field, dtype=float)
    )
    if not np.isfinite(data_weight) or data_weight < 0.0:
        raise ValueError("data_weight must be finite and non-negative")
    if not np.isfinite(physics_weight) or physics_weight < 0.0:
        raise ValueError("physics_weight must be finite and non-negative")
    data_mse = float(np.mean((predicted_field - target_field) ** 2))
    physics_mse = burgers_transition_physics_loss(
        initial_field,
        predicted_field,
        horizon=horizon,
        viscosity=viscosity,
        length=length,
    )
    total = float(data_weight) * data_mse + float(physics_weight) * physics_mse
    return total, data_mse, physics_mse
