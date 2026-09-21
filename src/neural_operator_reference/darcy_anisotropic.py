"""Reference solver for diagonal anisotropic Darcy flow in 2D."""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


Forcing2D = float | np.ndarray


def _validated_grid(field: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(field)
    if not np.isrealobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = array.astype(float, copy=False)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional field")
    if array.shape[0] < 3 or array.shape[1] < 3:
        raise ValueError(f"{name} must contain at least three points per axis")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _validated_length(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _validated_forcing(shape: tuple[int, int], forcing: Forcing2D) -> np.ndarray:
    array = np.asarray(forcing)
    if not np.isrealobj(array):
        raise ValueError("forcing must be real-valued")
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("forcing must contain only finite values")
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=float)
    try:
        return np.broadcast_to(array, shape).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError(
            f"forcing shape {array.shape} is not broadcastable to grid shape {shape}"
        ) from exc


def _harmonic_mean(left, right):
    return 2.0 * left * right / (left + right)


def solve_darcy_diagonal_2d(
    permeability_x: np.ndarray,
    permeability_y: np.ndarray,
    forcing: Forcing2D = 1.0,
    length_x: float = 1.0,
    length_y: float = 1.0,
) -> np.ndarray:
    """Solve -d_x(k_x d_x u)-d_y(k_y d_y u)=f with zero Dirichlet data."""

    kx = _validated_grid(permeability_x, "permeability_x")
    ky = _validated_grid(permeability_y, "permeability_y")
    if kx.shape != ky.shape:
        raise ValueError("permeability_x and permeability_y must share grid shape")
    length_x = _validated_length(length_x, "length_x")
    length_y = _validated_length(length_y, "length_y")
    forcing_field = _validated_forcing(kx.shape, forcing)

    points_y, points_x = kx.shape
    dx = length_x / float(points_x - 1)
    dy = length_y / float(points_y - 1)
    dx2 = dx * dx
    dy2 = dy * dy
    interior_x = points_x - 2
    interior_y = points_y - 2
    unknowns = interior_x * interior_y

    matrix = lil_matrix((unknowns, unknowns), dtype=float)
    rhs = forcing_field[1:-1, 1:-1].reshape(-1).copy()

    def index(row: int, column: int) -> int:
        return (row - 1) * interior_x + (column - 1)

    for row in range(1, points_y - 1):
        for column in range(1, points_x - 1):
            equation = index(row, column)
            east = _harmonic_mean(
                kx[row, column],
                kx[row, column + 1],
            )
            west = _harmonic_mean(
                kx[row, column],
                kx[row, column - 1],
            )
            north = _harmonic_mean(
                ky[row, column],
                ky[row + 1, column],
            )
            south = _harmonic_mean(
                ky[row, column],
                ky[row - 1, column],
            )

            matrix[equation, equation] = (
                (east + west) / dx2 + (north + south) / dy2
            )
            if column + 1 < points_x - 1:
                matrix[equation, index(row, column + 1)] = -east / dx2
            if column - 1 > 0:
                matrix[equation, index(row, column - 1)] = -west / dx2
            if row + 1 < points_y - 1:
                matrix[equation, index(row + 1, column)] = -north / dy2
            if row - 1 > 0:
                matrix[equation, index(row - 1, column)] = -south / dy2

    interior = spsolve(matrix.tocsr(), rhs)
    if not np.all(np.isfinite(interior)):
        raise FloatingPointError("anisotropic Darcy solve produced non-finite values")

    solution = np.zeros_like(kx)
    solution[1:-1, 1:-1] = interior.reshape(interior_y, interior_x)
    return solution


def darcy_diagonal_residual_2d(
    solution: np.ndarray,
    permeability_x: np.ndarray,
    permeability_y: np.ndarray,
    forcing: Forcing2D = 1.0,
    length_x: float = 1.0,
    length_y: float = 1.0,
) -> np.ndarray:
    """Return the interior residual of the diagonal anisotropic stencil."""

    solution = np.asarray(solution)
    if not np.isrealobj(solution):
        raise ValueError("solution must be real-valued")
    solution = solution.astype(float, copy=False)
    if solution.ndim != 2 or solution.shape[0] < 3 or solution.shape[1] < 3:
        raise ValueError("solution must be a 2D grid with at least three points")
    if not np.all(np.isfinite(solution)):
        raise ValueError("solution must contain only finite values")

    kx = _validated_grid(permeability_x, "permeability_x")
    ky = _validated_grid(permeability_y, "permeability_y")
    if solution.shape != kx.shape or solution.shape != ky.shape:
        raise ValueError("solution, permeability_x and permeability_y must share shape")

    length_x = _validated_length(length_x, "length_x")
    length_y = _validated_length(length_y, "length_y")
    forcing_field = _validated_forcing(solution.shape, forcing)
    dx2 = (length_x / float(solution.shape[1] - 1)) ** 2
    dy2 = (length_y / float(solution.shape[0] - 1)) ** 2

    center = solution[1:-1, 1:-1]
    east = _harmonic_mean(kx[1:-1, 1:-1], kx[1:-1, 2:])
    west = _harmonic_mean(kx[1:-1, 1:-1], kx[1:-1, :-2])
    north = _harmonic_mean(ky[1:-1, 1:-1], ky[2:, 1:-1])
    south = _harmonic_mean(ky[1:-1, 1:-1], ky[:-2, 1:-1])

    residual = (
        east * (center - solution[1:-1, 2:])
        + west * (center - solution[1:-1, :-2])
    ) / dx2
    residual += (
        north * (center - solution[2:, 1:-1])
        + south * (center - solution[:-2, 1:-1])
    ) / dy2
    return residual - forcing_field[1:-1, 1:-1]
