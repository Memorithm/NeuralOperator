"""Small reference training routines for physics-aware operator experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fno1d import NumpyFNO1D
from .physics import burgers_data_physics_loss


@dataclass(frozen=True)
class PhysicsFitResult:
    """Summary of a weighted data/PDE optimization."""

    initial_total_loss: float
    final_total_loss: float
    initial_data_loss: float
    final_data_loss: float
    initial_physics_loss: float
    final_physics_loss: float
    success: bool
    iterations: int
    function_evaluations: int
    message: str


def fit_fno_burgers_physics(
    model: NumpyFNO1D,
    inputs: np.ndarray,
    targets: np.ndarray,
    horizon: float,
    viscosity: float,
    data_weight: float = 1.0,
    physics_weight: float = 1.0e-3,
    length: float = 2.0 * np.pi,
    maxiter: int = 60,
    tolerance: float = 1.0e-8,
    forcing: float | np.ndarray | None = None,
) -> PhysicsFitResult:
    """Fit a tiny FNO against data and a forced Burgers residual.

    This uses finite-difference optimization and a one-transition temporal
    approximation. It is a reference experiment for loss accounting, not a
    production PINO trainer.
    """

    inputs = np.asarray(inputs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError("inputs and targets must have shape (batch, points, channels)")
    if inputs.shape[-1] != 1 or targets.shape[-1] != 1:
        raise ValueError("the Burgers reference trainer expects one channel")
    model.loss(inputs, targets)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and strictly positive")
    if not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError("maxiter must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive")

    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("scipy is required for physics-aware reference training") from exc

    def evaluate() -> tuple[float, float, float]:
        prediction = model(inputs)[..., 0]
        return burgers_data_physics_loss(
            inputs[..., 0],
            prediction,
            targets[..., 0],
            horizon=horizon,
            viscosity=viscosity,
            data_weight=data_weight,
            physics_weight=physics_weight,
            length=length,
            forcing=forcing,
        )

    initial_total, initial_data, initial_physics = evaluate()
    initial_vector = model.parameter_vector()

    def objective(vector: np.ndarray) -> float:
        model.set_parameter_vector(vector)
        return evaluate()[0]

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
    model.set_parameter_vector(result.x)
    final_total, final_data, final_physics = evaluate()
    return PhysicsFitResult(
        initial_total_loss=float(initial_total),
        final_total_loss=float(final_total),
        initial_data_loss=float(initial_data),
        final_data_loss=float(final_data),
        initial_physics_loss=float(initial_physics),
        final_physics_loss=float(final_physics),
        success=bool(result.success),
        iterations=int(getattr(result, "nit", 0)),
        function_evaluations=int(getattr(result, "nfev", 0)),
        message=str(result.message),
    )
