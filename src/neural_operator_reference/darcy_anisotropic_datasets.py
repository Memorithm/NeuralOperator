"""Deterministic datasets for diagonal anisotropic Darcy flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .darcy_anisotropic import solve_darcy_diagonal_2d
from .darcy_datasets import random_log_permeability


@dataclass(frozen=True)
class DarcyDiagonalDataset:
    """Traceable diagonal permeability-to-pressure dataset."""

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

    @property
    def samples(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.inputs.shape[1]), int(self.inputs.shape[2]))


def generate_darcy_diagonal_dataset(
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
    anisotropy_ratio: float = 1.0,
) -> DarcyDiagonalDataset:
    """Generate diagonal tensor permeability -> pressure pairs.

    A scalar positive base field k is transformed into

        k_x = k * sqrt(r)
        k_y = k / sqrt(r)

    so k_x / k_y = r pointwise while the geometric determinant scale
    sqrt(k_x*k_y) remains the original scalar field.
    """

    forcing = float(forcing)
    anisotropy_ratio = float(anisotropy_ratio)
    if not np.isfinite(forcing):
        raise ValueError("forcing must be finite")
    if not np.isfinite(anisotropy_ratio) or anisotropy_ratio <= 0.0:
        raise ValueError("anisotropy_ratio must be finite and strictly positive")

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
    root_ratio = np.sqrt(anisotropy_ratio)
    permeability_x = base * root_ratio
    permeability_y = base / root_ratio

    solutions = np.stack(
        [
            solve_darcy_diagonal_2d(
                permeability_x[index],
                permeability_y[index],
                forcing=forcing,
                length_x=length_x,
                length_y=length_y,
            )
            for index in range(base.shape[0])
        ],
        axis=0,
    )
    inputs = np.stack((permeability_x, permeability_y), axis=-1)
    return DarcyDiagonalDataset(
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
    )
