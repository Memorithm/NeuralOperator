"""Deterministic datasets for full SPD tensor Darcy flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .darcy_datasets import random_log_permeability
from .darcy_tensor import solve_darcy_tensor_2d


@dataclass(frozen=True)
class DarcyTensorDataset:
    """Traceable full-tensor permeability-to-pressure dataset."""

    inputs: np.ndarray
    targets: np.ndarray
    forcing: float
    length_x: float
    length_y: float
    seed: int
    modes: int
    log_std: float
    mean_log_permeability: float
    spectral_decay: float
    anisotropy_ratio: float
    orientation_base_radians: float
    orientation_amplitude_radians: float
    orientation_modes: int

    @property
    def samples(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.inputs.shape[1]), int(self.inputs.shape[2]))


def _orientation_fields(
    *,
    samples: int,
    points_x: int,
    points_y: int,
    modes: int,
    seed: int,
    length_x: float,
    length_y: float,
    base: float,
    amplitude: float,
) -> np.ndarray:
    if amplitude == 0.0:
        return np.full((samples, points_y, points_x), base, dtype=float)

    x = np.linspace(0.0, length_x, points_x)
    y = np.linspace(0.0, length_y, points_y)
    grid_x, grid_y = np.meshgrid(x, y)
    rng = np.random.default_rng(int(seed) + 104729)
    raw = np.zeros((samples, points_y, points_x), dtype=float)

    for mode_x in range(modes + 1):
        for mode_y in range(modes + 1):
            if mode_x == 0 and mode_y == 0:
                continue
            squared_frequency = mode_x * mode_x + mode_y * mode_y
            scale = 1.0 / float(squared_frequency)
            phase = 2.0 * np.pi * (
                mode_x * grid_x / length_x + mode_y * grid_y / length_y
            )
            raw += scale * rng.normal(size=(samples, 1, 1)) * np.cos(phase)
            raw += scale * rng.normal(size=(samples, 1, 1)) * np.sin(phase)

    raw -= np.mean(raw, axis=(1, 2), keepdims=True)
    max_abs = np.max(np.abs(raw), axis=(1, 2), keepdims=True)
    if np.any(max_abs == 0.0):
        raise FloatingPointError("orientation basis produced zero variation")
    return base + amplitude * raw / max_abs


def generate_darcy_tensor_dataset(
    samples: int,
    points_x: int,
    points_y: int | None = None,
    forcing: float = 1.0,
    modes: int = 3,
    seed: int = 0,
    length_x: float = 1.0,
    length_y: float = 1.0,
    log_std: float = 0.5,
    mean_log_permeability: float = 0.0,
    spectral_decay: float = 2.0,
    anisotropy_ratio: float = 4.0,
    orientation_base_radians: float = 0.0,
    orientation_amplitude_radians: float = 0.0,
    orientation_modes: int = 2,
) -> DarcyTensorDataset:
    """Generate full SPD tensor permeability -> pressure pairs.

    A positive scalar base k supplies the determinant scale. Principal values are
    k*sqrt(r) and k/sqrt(r), then rotated by a deterministic smooth orientation
    field. The resulting three input channels are K_xx, K_xy and K_yy.
    """

    forcing = float(forcing)
    anisotropy_ratio = float(anisotropy_ratio)
    orientation_base_radians = float(orientation_base_radians)
    orientation_amplitude_radians = float(orientation_amplitude_radians)
    if not isinstance(orientation_modes, (int, np.integer)):
        raise ValueError("orientation_modes must be an integer >= 1")
    orientation_modes = int(orientation_modes)

    if not np.isfinite(forcing):
        raise ValueError("forcing must be finite")
    if not np.isfinite(anisotropy_ratio) or anisotropy_ratio <= 0.0:
        raise ValueError("anisotropy_ratio must be finite and strictly positive")
    if not np.isfinite(orientation_base_radians):
        raise ValueError("orientation_base_radians must be finite")
    if (
        not np.isfinite(orientation_amplitude_radians)
        or orientation_amplitude_radians < 0.0
    ):
        raise ValueError(
            "orientation_amplitude_radians must be finite and non-negative"
        )
    if orientation_modes < 1:
        raise ValueError("orientation_modes must be an integer >= 1")

    base = random_log_permeability(
        samples=samples,
        points_x=points_x,
        points_y=points_y,
        modes=modes,
        seed=seed,
        length_x=length_x,
        length_y=length_y,
        log_std=log_std,
        mean_log_permeability=mean_log_permeability,
        spectral_decay=spectral_decay,
    )
    points_y_actual, points_x_actual = base.shape[1:]
    if orientation_modes >= min(points_x_actual, points_y_actual) // 2:
        raise ValueError(
            "orientation_modes must remain below the smallest-grid Nyquist limit"
        )

    angle = _orientation_fields(
        samples=base.shape[0],
        points_x=points_x_actual,
        points_y=points_y_actual,
        modes=orientation_modes,
        seed=seed,
        length_x=float(length_x),
        length_y=float(length_y),
        base=orientation_base_radians,
        amplitude=orientation_amplitude_radians,
    )

    root_ratio = np.sqrt(anisotropy_ratio)
    principal_one = base * root_ratio
    principal_two = base / root_ratio
    cosine = np.cos(angle)
    sine = np.sin(angle)
    permeability_xx = (
        principal_one * cosine * cosine + principal_two * sine * sine
    )
    permeability_yy = (
        principal_one * sine * sine + principal_two * cosine * cosine
    )
    permeability_xy = (principal_one - principal_two) * sine * cosine
    inputs = np.stack(
        (permeability_xx, permeability_xy, permeability_yy),
        axis=-1,
    )

    solutions = np.stack(
        [
            solve_darcy_tensor_2d(
                inputs[index, ..., 0],
                inputs[index, ..., 1],
                inputs[index, ..., 2],
                forcing=forcing,
                length_x=length_x,
                length_y=length_y,
            )
            for index in range(base.shape[0])
        ],
        axis=0,
    )
    return DarcyTensorDataset(
        inputs=inputs,
        targets=solutions[..., None],
        forcing=forcing,
        length_x=float(length_x),
        length_y=float(length_y),
        seed=int(seed),
        modes=int(modes),
        log_std=float(log_std),
        mean_log_permeability=float(mean_log_permeability),
        spectral_decay=float(spectral_decay),
        anisotropy_ratio=anisotropy_ratio,
        orientation_base_radians=orientation_base_radians,
        orientation_amplitude_radians=orientation_amplitude_radians,
        orientation_modes=orientation_modes,
    )
