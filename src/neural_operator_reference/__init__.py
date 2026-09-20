"""Numerical reference components for the Neural Operator research program."""

from .spectral import (
    burgers_residual,
    central_difference_periodic,
    dealias_mask,
    dealiased_product,
    divergence_2d,
    incompressible_velocity_from_streamfunction,
    relative_l2_error,
    spectral_derivative,
    spectral_filter,
    wave_numbers,
)
from .linear_operator import FourierMultiplier1D
from .fno1d import FNOFitResult, NumpyFNO1D
from .burgers import (
    burgers_rhs,
    burgers_rk4_step,
    integrate_burgers,
    periodic_energy,
    periodic_mean,
)

__all__ = [
    "burgers_residual",
    "central_difference_periodic",
    "dealias_mask",
    "dealiased_product",
    "divergence_2d",
    "incompressible_velocity_from_streamfunction",
    "relative_l2_error",
    "spectral_derivative",
    "spectral_filter",
    "wave_numbers",
    "FourierMultiplier1D",
    "FNOFitResult",
    "NumpyFNO1D",
    "burgers_rhs",
    "burgers_rk4_step",
    "integrate_burgers",
    "periodic_energy",
    "periodic_mean",
]
