"""Numerical reference components for the Neural Operator research program."""

from .spectral import (
    burgers_residual,
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

__all__ = [
    "burgers_residual",
    "dealias_mask",
    "dealiased_product",
    "divergence_2d",
    "incompressible_velocity_from_streamfunction",
    "relative_l2_error",
    "spectral_derivative",
    "spectral_filter",
    "wave_numbers",
    "FourierMultiplier1D",
]
