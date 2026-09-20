"""Reference solver for the periodically forced viscous Burgers equation."""

from __future__ import annotations

import numpy as np

from .spectral import dealiased_product, spectral_derivative


Forcing = float | np.ndarray | None


def _validated_forcing(field: np.ndarray, forcing: Forcing) -> float | np.ndarray:
    """Return a finite real forcing broadcastable to field."""

    if forcing is None:
        return 0.0
    forcing_array = np.asarray(forcing)
    if not np.isrealobj(forcing_array):
        raise ValueError("forcing must be real-valued")
    forcing_array = forcing_array.astype(float, copy=False)
    if not np.all(np.isfinite(forcing_array)):
        raise ValueError("forcing must contain only finite values")
    if forcing_array.ndim == 0:
        return float(forcing_array)
    try:
        return np.broadcast_to(forcing_array, field.shape)
    except ValueError as exc:
        raise ValueError(
            f"forcing shape {forcing_array.shape} is not broadcastable to "
            f"field shape {field.shape}"
        ) from exc


def burgers_rhs(
    field: np.ndarray,
    viscosity: float,
    length: float = 2.0 * np.pi,
    dealias: bool = True,
    forcing: Forcing = None,
) -> np.ndarray:
    """Evaluate u_t = -u u_x + viscosity u_xx + f on a periodic grid.

    The forcing is time-independent for this reference solver and may be a
    scalar or an array broadcastable to the spatial field shape.
    """

    field = np.asarray(field, dtype=float)
    if field.ndim == 0:
        raise ValueError("field must contain a spatial axis")
    if field.shape[-1] < 3:
        raise ValueError("at least three spatial points are required")
    if not np.isfinite(viscosity) or viscosity < 0.0:
        raise ValueError("viscosity must be finite and non-negative")

    field_x = spectral_derivative(field, order=1, length=length, axis=-1)
    if dealias:
        advection = dealiased_product(field, field_x, axis=-1)
    else:
        advection = field * field_x
    diffusion = spectral_derivative(field, order=2, length=length, axis=-1)
    return (
        -advection
        + float(viscosity) * diffusion
        + _validated_forcing(field, forcing)
    )


def burgers_rk4_step(
    field: np.ndarray,
    dt: float,
    viscosity: float,
    length: float = 2.0 * np.pi,
    dealias: bool = True,
    forcing: Forcing = None,
) -> np.ndarray:
    """Advance one explicit fourth-order Runge–Kutta step."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and strictly positive")
    field = np.asarray(field, dtype=float)
    k1 = burgers_rhs(
        field, viscosity, length=length, dealias=dealias, forcing=forcing
    )
    k2 = burgers_rhs(
        field + 0.5 * dt * k1,
        viscosity,
        length=length,
        dealias=dealias,
        forcing=forcing,
    )
    k3 = burgers_rhs(
        field + 0.5 * dt * k2,
        viscosity,
        length=length,
        dealias=dealias,
        forcing=forcing,
    )
    k4 = burgers_rhs(
        field + dt * k3,
        viscosity,
        length=length,
        dealias=dealias,
        forcing=forcing,
    )
    return field + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_burgers(
    initial_field: np.ndarray,
    dt: float,
    steps: int,
    viscosity: float,
    length: float = 2.0 * np.pi,
    dealias: bool = True,
    forcing: Forcing = None,
) -> np.ndarray:
    """Return a trajectory with shape (steps + 1, *initial_field.shape)."""

    initial_field = np.asarray(initial_field, dtype=float)
    if initial_field.ndim == 0:
        raise ValueError("initial_field must contain a spatial axis")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and strictly positive")
    _validated_forcing(initial_field, forcing)

    trajectory = np.empty((int(steps) + 1,) + initial_field.shape, dtype=float)
    trajectory[0] = initial_field
    for step in range(int(steps)):
        trajectory[step + 1] = burgers_rk4_step(
            trajectory[step],
            dt=dt,
            viscosity=viscosity,
            length=length,
            dealias=dealias,
            forcing=forcing,
        )
    return trajectory


def periodic_mean(field: np.ndarray) -> float:
    """Return the spatial mean of a periodic field."""

    field = np.asarray(field, dtype=float)
    if field.ndim == 0:
        raise ValueError("field must contain a spatial axis")
    return float(np.mean(field, axis=-1))


def periodic_energy(field: np.ndarray) -> float:
    """Return the spatial mean of u²/2 for a scalar field."""

    field = np.asarray(field, dtype=float)
    if field.ndim == 0:
        raise ValueError("field must contain a spatial axis")
    return float(0.5 * np.mean(field * field, axis=-1))
