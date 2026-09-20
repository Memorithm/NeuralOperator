"""Small, explicit spectral operators used as a numerical oracle.

The routines in this module assume periodic, uniformly spaced grids. They are
deliberately independent of a neural-network framework so they can be used to
validate a future PyTorch or Rust implementation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


Array = np.ndarray


def _validate_length(length: float) -> float:
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and strictly positive")
    return float(length)


def _validate_axis(axis: int, ndim: int) -> int:
    if ndim == 0:
        raise ValueError("a scalar has no spatial axis")
    if not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be an integer")
    axis = int(axis)
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is outside [-{ndim}, {ndim - 1}]")
    return axis % ndim


def wave_numbers(n: int, length: float = 2.0 * np.pi) -> Array:
    """Return angular Fourier wave numbers for a periodic grid.

    The grid convention is ``x_j = j * length / n`` and the returned values
    satisfy ``d/dx <-> 1j * k``. The ordering is NumPy's FFT ordering.
    """

    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer greater than or equal to 2")
    length = _validate_length(length)
    return 2.0 * np.pi * np.fft.fftfreq(int(n), d=length / int(n))


def spectral_derivative(
    values: Array,
    order: int = 1,
    length: float = 2.0 * np.pi,
    axis: int = -1,
) -> Array:
    """Differentiate a periodic field with a Fourier multiplier.

    For a discretized field ``u`` this computes the derivative represented by
    the discrete Fourier series. It is not a promise that an arbitrary
    non-periodic continuous function has been differentiated exactly.
    """

    values = np.asarray(values)
    if values.ndim == 0:
        raise ValueError("values must contain a spatial axis")
    if not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError("order must be a non-negative integer")
    axis = _validate_axis(axis, values.ndim)
    if order == 0:
        return values.copy()

    n = values.shape[axis]
    k = wave_numbers(n, length)
    multiplier = (1j * k) ** int(order)
    shape = [1] * values.ndim
    shape[axis] = n

    spectrum = np.fft.fft(values, axis=axis)
    differentiated = np.fft.ifft(
        spectrum * multiplier.reshape(shape), axis=axis
    )
    if np.isrealobj(values):
        return differentiated.real
    return differentiated


def dealias_mask(n: int, keep_ratio: float = 2.0 / 3.0) -> Array:
    """Return a boolean Fourier mask for the standard low-mode cutoff.

    For the usual `2/3` rule, modes with integer index ``|m| <= floor(n/3)``
    are retained. The function exposes the ratio so experiments can compare
    alternative filters without changing the filtering implementation.
    """

    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer greater than or equal to 2")
    if not np.isfinite(keep_ratio) or not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    mode_indices = np.fft.fftfreq(int(n)) * int(n)
    cutoff = int(np.floor(int(n) * float(keep_ratio) / 2.0))
    return np.abs(mode_indices) <= cutoff


def spectral_filter(
    values: Array,
    axis: int = -1,
    keep_ratio: float = 2.0 / 3.0,
) -> Array:
    """Remove high Fourier modes along one spatial axis."""

    values = np.asarray(values)
    axis = _validate_axis(axis, values.ndim)
    mask = dealias_mask(values.shape[axis], keep_ratio=keep_ratio)
    shape = [1] * values.ndim
    shape[axis] = values.shape[axis]
    filtered = np.fft.ifft(
        np.fft.fft(values, axis=axis) * mask.reshape(shape), axis=axis
    )
    if np.isrealobj(values):
        return filtered.real
    return filtered


def dealiased_product(
    left: Array,
    right: Array,
    axis: int = -1,
    keep_ratio: float = 2.0 / 3.0,
) -> Array:
    """Multiply two fields and filter the generated high modes."""

    left, right = np.broadcast_arrays(np.asarray(left), np.asarray(right))
    return spectral_filter(left * right, axis=axis, keep_ratio=keep_ratio)


def incompressible_velocity_from_streamfunction(
    streamfunction: Array,
    lengths: Tuple[float, float] = (2.0 * np.pi, 2.0 * np.pi),
    spatial_axes: Tuple[int, int] = (-2, -1),
) -> Tuple[Array, Array]:
    """Build a 2D divergence-free velocity from a scalar stream function.

    With ``u = d psi / dy`` and ``v = -d psi / dx``, the two mixed derivatives
    cancel on a smooth periodic field up to floating-point error.
    """

    if len(lengths) != 2 or len(spatial_axes) != 2:
        raise ValueError("lengths and spatial_axes must contain two entries")
    dx, dy = spatial_axes
    lx, ly = lengths
    dpsi_dx = spectral_derivative(streamfunction, order=1, length=lx, axis=dx)
    dpsi_dy = spectral_derivative(streamfunction, order=1, length=ly, axis=dy)
    return dpsi_dy, -dpsi_dx


def divergence_2d(
    u: Array,
    v: Array,
    lengths: Tuple[float, float] = (2.0 * np.pi, 2.0 * np.pi),
    spatial_axes: Tuple[int, int] = (-2, -1),
) -> Array:
    """Compute ``du/dx + dv/dy`` on a periodic 2D grid."""

    if len(lengths) != 2 or len(spatial_axes) != 2:
        raise ValueError("lengths and spatial_axes must contain two entries")
    dx, dy = spatial_axes
    lx, ly = lengths
    return spectral_derivative(u, 1, lx, dx) + spectral_derivative(v, 1, ly, dy)


def burgers_residual(
    u: Array,
    u_t: Array,
    viscosity: float,
    length: float = 2.0 * np.pi,
    axis: int = -1,
) -> Array:
    """Return the 1D viscous Burgers residual.

    The equation convention is ``u_t + u*u_x - viscosity*u_xx = 0``. The
    temporal derivative is passed by the caller because this module only owns
    spatial spectral differentiation.
    """

    u, u_t = np.broadcast_arrays(np.asarray(u), np.asarray(u_t))
    if not np.isfinite(viscosity) or viscosity < 0.0:
        raise ValueError("viscosity must be finite and non-negative")
    u_x = spectral_derivative(u, order=1, length=length, axis=axis)
    u_xx = spectral_derivative(u, order=2, length=length, axis=axis)
    return u_t + u * u_x - float(viscosity) * u_xx


def relative_l2_error(approximation: Array, reference: Array) -> float:
    """Return ``||approx-reference||_2 / ||reference||_2``."""

    approximation, reference = np.broadcast_arrays(
        np.asarray(approximation), np.asarray(reference)
    )
    difference_norm = np.linalg.norm((approximation - reference).ravel())
    reference_norm = np.linalg.norm(reference.ravel())
    if reference_norm == 0.0:
        return float(difference_norm)
    return float(difference_norm / reference_norm)
