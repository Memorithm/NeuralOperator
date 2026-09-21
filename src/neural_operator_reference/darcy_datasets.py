"""Deterministic coefficient-to-solution datasets for Darcy flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .darcy import solve_darcy_2d


@dataclass(frozen=True)
class DarcyDataset:
    """Traceable permeability-to-pressure dataset on a rectangular grid."""

    inputs: np.ndarray
    targets: np.ndarray
    forcing: float
    length_x: float
    length_y: float
    seed: int
    modes: int
    log_std: float
    mean_log_permeability: float

    @property
    def samples(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.inputs.shape[1]), int(self.inputs.shape[2]))


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive_length(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def random_log_permeability(
    samples: int,
    points_x: int,
    points_y: int | None = None,
    modes: int = 3,
    seed: int = 0,
    length_x: float = 1.0,
    length_y: float = 1.0,
    log_std: float = 0.5,
    mean_log_permeability: float = 0.0,
) -> np.ndarray:
    """Generate smooth positive permeability fields with deterministic provenance."""

    samples = _positive_integer(samples, "samples")
    points_x = _positive_integer(points_x, "points_x", minimum=5)
    if points_y is None:
        points_y = points_x
    points_y = _positive_integer(points_y, "points_y", minimum=5)
    modes = _positive_integer(modes, "modes")
    if modes >= min(points_x, points_y) // 2:
        raise ValueError("modes must remain below the smallest-grid Nyquist limit")
    length_x = _positive_length(length_x, "length_x")
    length_y = _positive_length(length_y, "length_y")
    log_std = float(log_std)
    mean_log_permeability = float(mean_log_permeability)
    if not np.isfinite(log_std) or log_std < 0.0:
        raise ValueError("log_std must be finite and non-negative")
    if not np.isfinite(mean_log_permeability):
        raise ValueError("mean_log_permeability must be finite")

    x = np.linspace(0.0, length_x, points_x)
    y = np.linspace(0.0, length_y, points_y)
    grid_x, grid_y = np.meshgrid(x, y)
    rng = np.random.default_rng(seed)
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

    if log_std == 0.0:
        return np.full(
            (samples, points_y, points_x),
            np.exp(mean_log_permeability),
            dtype=float,
        )

    raw -= np.mean(raw, axis=(1, 2), keepdims=True)
    field_std = np.std(raw, axis=(1, 2), keepdims=True)
    if np.any(field_std == 0.0):
        raise FloatingPointError("permeability basis produced zero variance")
    normalized = raw / field_std
    return np.exp(mean_log_permeability + log_std * normalized)


def generate_darcy_dataset(
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
) -> DarcyDataset:
    """Generate permeability -> pressure pairs using the reference solver."""

    forcing = float(forcing)
    if not np.isfinite(forcing):
        raise ValueError("forcing must be finite")
    permeability = random_log_permeability(
        samples=samples,
        points_x=points_x,
        points_y=points_y,
        modes=modes,
        seed=seed,
        length_x=length_x,
        length_y=length_y,
        log_std=log_std,
        mean_log_permeability=mean_log_permeability,
    )
    solutions = np.stack(
        [
            solve_darcy_2d(
                coefficient,
                forcing=forcing,
                length_x=length_x,
                length_y=length_y,
            )
            for coefficient in permeability
        ],
        axis=0,
    )
    return DarcyDataset(
        inputs=permeability[..., None],
        targets=solutions[..., None],
        forcing=forcing,
        length_x=float(length_x),
        length_y=float(length_y),
        seed=int(seed),
        modes=int(modes),
        log_std=float(log_std),
        mean_log_permeability=float(mean_log_permeability),
    )
