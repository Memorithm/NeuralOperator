"""Reference solver for steady variable-coefficient Darcy flow in 2D."""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


Forcing2D = float | np.ndarray


def _validated_grid(
    field: np.ndarray,
    name: str,
    *,
    positive: bool = False,
) -> np.ndarray:
    """Return a finite real 2D grid with at least one interior node."""

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
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _validated_length(value: float, name: str) -> float:
    """Validate a positive physical domain length."""

    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _validated_forcing(shape: tuple[int, int], forcing: Forcing2D) -> np.ndarray:
    """Return a finite real forcing broadcast to the Darcy grid."""

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


def _harmonic_mean(left: np.ndarray | float, right: np.ndarray | float):
    """Return the harmonic mean used for face permeability."""

    return 2.0 * left * right / (left + right)


def solve_darcy_2d(
    permeability: np.ndarray,
    forcing: Forcing2D = 1.0,
    length_x: float = 1.0,
    length_y: float = 1.0,
) -> np.ndarray:
    """Solve -div(k grad u) = f with homogeneous Dirichlet boundaries.

    The permeability is stored at nodal locations. Face values are harmonic
    means of adjacent nodes, yielding a conservative five-point flux stencil
    on the rectangular grid. Boundary values of u are fixed to zero.
    """

    coefficient = _validated_grid(
        permeability, "permeability", positive=True
    )
    length_x = _validated_length(length_x, "length_x")
    length_y = _validated_length(length_y, "length_y")
    forcing_field = _validated_forcing(coefficient.shape, forcing)

    points_y, points_x = coefficient.shape
    dx = length_x / float(points_x - 1)
    dy = length_y / float(points_y - 1)
    interior_x = points_x - 2
    interior_y = points_y - 2
    unknowns = interior_x * interior_y

    matrix = lil_matrix((unknowns, unknowns), dtype=float)
    rhs = forcing_field[1:-1, 1:-1].reshape(-1).copy()
    dx2 = dx * dx
    dy2 = dy * dy

    def index(row: int, column: int) -> int:
        return (row - 1) * interior_x + (column - 1)

    for row in range(1, points_y - 1):
        for column in range(1, points_x - 1):
            equation = index(row, column)
            center = coefficient[row, column]
            east = _harmonic_mean(center, coefficient[row, column + 1])
            west = _harmonic_mean(center, coefficient[row, column - 1])
            north = _harmonic_mean(center, coefficient[row + 1, column])
            south = _harmonic_mean(center, coefficient[row - 1, column])

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

    interior_solution = spsolve(matrix.tocsr(), rhs)
    if not np.all(np.isfinite(interior_solution)):
        raise FloatingPointError("Darcy solve produced non-finite values")

    solution = np.zeros_like(coefficient)
    solution[1:-1, 1:-1] = interior_solution.reshape(interior_y, interior_x)
    return solution


def darcy_residual_2d(
    solution: np.ndarray,
    permeability: np.ndarray,
    forcing: Forcing2D = 1.0,
    length_x: float = 1.0,
    length_y: float = 1.0,
) -> np.ndarray:
    """Return the interior residual of the same conservative Darcy stencil."""

    solution = _validated_grid(solution, "solution")
    coefficient = _validated_grid(
        permeability, "permeability", positive=True
    )
    if solution.shape != coefficient.shape:
        raise ValueError(
            "solution and permeability must use the same grid shape"
        )
    length_x = _validated_length(length_x, "length_x")
    length_y = _validated_length(length_y, "length_y")
    forcing_field = _validated_forcing(coefficient.shape, forcing)

    points_y, points_x = coefficient.shape
    dx2 = (length_x / float(points_x - 1)) ** 2
    dy2 = (length_y / float(points_y - 1)) ** 2

    center_u = solution[1:-1, 1:-1]
    center_k = coefficient[1:-1, 1:-1]
    east = _harmonic_mean(center_k, coefficient[1:-1, 2:])
    west = _harmonic_mean(center_k, coefficient[1:-1, :-2])
    north = _harmonic_mean(center_k, coefficient[2:, 1:-1])
    south = _harmonic_mean(center_k, coefficient[:-2, 1:-1])

    residual = (
        east * (center_u - solution[1:-1, 2:])
        + west * (center_u - solution[1:-1, :-2])
    ) / dx2
    residual += (
        north * (center_u - solution[2:, 1:-1])
        + south * (center_u - solution[:-2, 1:-1])
    ) / dy2
    return residual - forcing_field[1:-1, 1:-1]
