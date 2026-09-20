"""Deterministic multi-step evaluation for Burgers operator rollouts.

The routines in this module keep the evaluation contract explicit: operators
receive (batch, points, 1) fields and return one field with the same shape.
The reference solver remains the source of truth for rollout targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .physics import burgers_transition_physics_loss
from .spectral import relative_l2_error


ArrayOperator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BurgersRolloutMetrics:
    """Per-step metrics for a recursive Burgers operator rollout.

    mse_by_step and relative_l2_by_step compare the predicted trajectory with the
    reference trajectory. physics_mse_by_transition evaluates the finite-horizon
    transition residual on consecutive predicted states; it is a discrete
    diagnostic and not a continuous-time PINO loss.
    """

    mse_by_step: tuple[float, ...]
    relative_l2_by_step: tuple[float, ...]
    physics_mse_by_transition: tuple[float, ...]
    mean_abs_error_by_step: tuple[float, ...]
    energy_abs_error_by_step: tuple[float, ...]
    max_relative_l2: float
    final_relative_l2: float

    def as_dict(self) -> dict[str, object]:
        """Return JSON-serializable metric fields."""

        return {
            "mse_by_step": list(self.mse_by_step),
            "relative_l2_by_step": list(self.relative_l2_by_step),
            "physics_mse_by_transition": list(self.physics_mse_by_transition),
            "mean_abs_error_by_step": list(self.mean_abs_error_by_step),
            "energy_abs_error_by_step": list(self.energy_abs_error_by_step),
            "max_relative_l2": self.max_relative_l2,
            "final_relative_l2": self.final_relative_l2,
        }


def rollout_burgers_operator(
    operator: ArrayOperator,
    initial_fields: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Recursively apply a one-transition operator.

    Parameters
    ----------
    operator:
        Callable implementing the reference operator contract. It must accept
        an array with shape (batch, points, 1) and return the same shape.
    initial_fields:
        Batch of periodic fields with shape (batch, points).
    steps:
        Number of recursive applications. Zero returns the initial state only.

    Returns
    -------
    numpy.ndarray
        A trajectory with shape (steps + 1, batch, points).
    """

    if not callable(operator):
        raise TypeError("operator must be callable")
    initial_fields = np.asarray(initial_fields)
    if initial_fields.ndim != 2:
        raise ValueError("initial_fields must have shape (batch, points)")
    if initial_fields.shape[0] == 0 or initial_fields.shape[1] < 3:
        raise ValueError("initial_fields must contain a non-empty batch of 3+ point fields")
    if not np.isrealobj(initial_fields):
        raise ValueError("initial_fields must be real-valued")
    initial_fields = initial_fields.astype(float, copy=False)
    if not np.all(np.isfinite(initial_fields)):
        raise ValueError("initial_fields must contain only finite values")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")

    trajectory = np.empty(
        (int(steps) + 1, initial_fields.shape[0], initial_fields.shape[1]),
        dtype=float,
    )
    trajectory[0] = initial_fields
    current = initial_fields
    expected_shape = (initial_fields.shape[0], initial_fields.shape[1], 1)

    for step in range(int(steps)):
        raw_prediction = np.asarray(operator(current[..., None]))
        if raw_prediction.shape != expected_shape:
            raise ValueError(
                "operator must return an array with shape "
                f"{expected_shape}, got {raw_prediction.shape}"
            )
        if not np.isrealobj(raw_prediction):
            raise ValueError("operator output must be real-valued")
        prediction = raw_prediction.astype(float, copy=False)
        if not np.all(np.isfinite(prediction)):
            raise ValueError("operator output must contain only finite values")
        current = prediction[..., 0]
        trajectory[step + 1] = current

    return trajectory


def evaluate_burgers_rollout(
    predicted_trajectory: np.ndarray,
    reference_trajectory: np.ndarray,
    horizon: float,
    viscosity: float,
    length: float = 2.0 * np.pi,
    forcing: np.ndarray | float | None = None,
) -> BurgersRolloutMetrics:
    """Evaluate a predicted trajectory against a reference trajectory.

    Both trajectories must have shape (time, batch, points). The returned
    conservation-related fields measure absolute mean and energy mismatch with
    the reference at each time step; they do not assert conservation for a
    general Burgers solution.
    """

    predicted_trajectory = np.asarray(predicted_trajectory)
    reference_trajectory = np.asarray(reference_trajectory)
    if predicted_trajectory.ndim != 3 or reference_trajectory.ndim != 3:
        raise ValueError(
            "trajectories must have shape (time, batch, points)"
        )
    if predicted_trajectory.shape != reference_trajectory.shape:
        raise ValueError("predicted and reference trajectories must have the same shape")
    if predicted_trajectory.shape[0] == 0 or predicted_trajectory.shape[1] == 0:
        raise ValueError("trajectories must contain at least one time step and sample")
    if predicted_trajectory.shape[2] < 3:
        raise ValueError("trajectories must contain at least three spatial points")
    if not np.isrealobj(predicted_trajectory) or not np.isrealobj(reference_trajectory):
        raise ValueError("trajectories must be real-valued")
    predicted_trajectory = predicted_trajectory.astype(float, copy=False)
    reference_trajectory = reference_trajectory.astype(float, copy=False)
    if not np.all(np.isfinite(predicted_trajectory)) or not np.all(
        np.isfinite(reference_trajectory)
    ):
        raise ValueError("trajectories must contain only finite values")

    squared_error = (predicted_trajectory - reference_trajectory) ** 2
    mse_by_step = tuple(
        float(value) for value in np.mean(squared_error, axis=(1, 2))
    )
    relative_l2_by_step = tuple(
        relative_l2_error(predicted, reference)
        for predicted, reference in zip(predicted_trajectory, reference_trajectory)
    )

    predicted_mean = np.mean(predicted_trajectory, axis=-1)
    reference_mean = np.mean(reference_trajectory, axis=-1)
    mean_abs_error_by_step = tuple(
        float(value)
        for value in np.mean(np.abs(predicted_mean - reference_mean), axis=1)
    )

    predicted_energy = 0.5 * np.mean(predicted_trajectory**2, axis=-1)
    reference_energy = 0.5 * np.mean(reference_trajectory**2, axis=-1)
    energy_abs_error_by_step = tuple(
        float(value)
        for value in np.mean(np.abs(predicted_energy - reference_energy), axis=1)
    )

    physics_mse_by_transition = tuple(
        burgers_transition_physics_loss(
            predicted_trajectory[step],
            predicted_trajectory[step + 1],
            horizon=horizon,
            viscosity=viscosity,
            length=length,
            forcing=forcing,
        )
        for step in range(predicted_trajectory.shape[0] - 1)
    )
    max_relative_l2 = float(max(relative_l2_by_step))
    return BurgersRolloutMetrics(
        mse_by_step=mse_by_step,
        relative_l2_by_step=relative_l2_by_step,
        physics_mse_by_transition=physics_mse_by_transition,
        mean_abs_error_by_step=mean_abs_error_by_step,
        energy_abs_error_by_step=energy_abs_error_by_step,
        max_relative_l2=max_relative_l2,
        final_relative_l2=float(relative_l2_by_step[-1]),
    )


def evaluate_burgers_operator(
    operator: ArrayOperator,
    initial_fields: np.ndarray,
    reference_trajectory: np.ndarray,
    horizon: float,
    viscosity: float,
    length: float = 2.0 * np.pi,
    forcing: np.ndarray | float | None = None,
) -> tuple[np.ndarray, BurgersRolloutMetrics]:
    """Roll out an operator and evaluate it against a supplied trajectory."""

    reference_trajectory = np.asarray(reference_trajectory)
    if reference_trajectory.ndim != 3:
        raise ValueError("reference_trajectory must have shape (time, batch, points)")
    predicted_trajectory = rollout_burgers_operator(
        operator,
        initial_fields,
        steps=reference_trajectory.shape[0] - 1,
    )
    if predicted_trajectory.shape != reference_trajectory.shape:
        raise ValueError(
            "reference_trajectory batch and spatial shape must match initial_fields"
        )
    metrics = evaluate_burgers_rollout(
        predicted_trajectory,
        reference_trajectory,
        horizon=horizon,
        viscosity=viscosity,
        length=length,
    )
    return predicted_trajectory, metrics
