"""Finite-element reference solver for full SPD tensor Darcy flow in 2D."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


Forcing2D = float | np.ndarray


def _validated_real_grid(field: np.ndarray, name: str) -> np.ndarray:
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
    return array


def _validated_tensor(
    permeability_xx: np.ndarray,
    permeability_xy: np.ndarray,
    permeability_yy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kxx = _validated_real_grid(permeability_xx, "permeability_xx")
    kxy = _validated_real_grid(permeability_xy, "permeability_xy")
    kyy = _validated_real_grid(permeability_yy, "permeability_yy")
    if kxx.shape != kxy.shape or kxx.shape != kyy.shape:
        raise ValueError("permeability tensor components must share grid shape")
    determinant = kxx * kyy - kxy * kxy
    if np.any(kxx <= 0.0) or np.any(kyy <= 0.0) or np.any(determinant <= 0.0):
        raise ValueError("permeability tensor must be symmetric positive definite")
    return kxx, kxy, kyy


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


def _assemble_darcy_tensor_system(
    permeability_xx: np.ndarray,
    permeability_xy: np.ndarray,
    permeability_yy: np.ndarray,
    forcing: Forcing2D,
    length_x: float,
    length_y: float,
):
    kxx, kxy, kyy = _validated_tensor(
        permeability_xx,
        permeability_xy,
        permeability_yy,
    )
    length_x = _validated_length(length_x, "length_x")
    length_y = _validated_length(length_y, "length_y")
    forcing_field = _validated_forcing(kxx.shape, forcing)

    points_y, points_x = kxx.shape
    dx = length_x / float(points_x - 1)
    dy = length_y / float(points_y - 1)
    interior_x = points_x - 2
    interior_y = points_y - 2
    unknowns = interior_x * interior_y

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs = np.zeros(unknowns, dtype=float)
    gauss_points = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    jacobian_determinant = dx * dy / 4.0

    def interior_index(row: int, column: int) -> int:
        return (row - 1) * interior_x + (column - 1)

    for cell_row in range(points_y - 1):
        for cell_column in range(points_x - 1):
            nodes = (
                (cell_row, cell_column),
                (cell_row, cell_column + 1),
                (cell_row + 1, cell_column + 1),
                (cell_row + 1, cell_column),
            )
            local_kxx = np.array([kxx[row, column] for row, column in nodes])
            local_kxy = np.array([kxy[row, column] for row, column in nodes])
            local_kyy = np.array([kyy[row, column] for row, column in nodes])
            local_forcing = np.array(
                [forcing_field[row, column] for row, column in nodes]
            )
            stiffness = np.zeros((4, 4), dtype=float)
            load = np.zeros(4, dtype=float)

            for xi in gauss_points:
                for eta in gauss_points:
                    shape = 0.25 * np.array(
                        (
                            (1.0 - xi) * (1.0 - eta),
                            (1.0 + xi) * (1.0 - eta),
                            (1.0 + xi) * (1.0 + eta),
                            (1.0 - xi) * (1.0 + eta),
                        )
                    )
                    dshape_dxi = 0.25 * np.array(
                        (
                            -(1.0 - eta),
                            +(1.0 - eta),
                            +(1.0 + eta),
                            -(1.0 + eta),
                        )
                    )
                    dshape_deta = 0.25 * np.array(
                        (
                            -(1.0 - xi),
                            -(1.0 + xi),
                            +(1.0 + xi),
                            +(1.0 - xi),
                        )
                    )
                    gradients = np.stack(
                        (
                            (2.0 / dx) * dshape_dxi,
                            (2.0 / dy) * dshape_deta,
                        ),
                        axis=1,
                    )
                    tensor = np.array(
                        (
                            (shape @ local_kxx, shape @ local_kxy),
                            (shape @ local_kxy, shape @ local_kyy),
                        )
                    )
                    stiffness += (
                        gradients @ tensor @ gradients.T * jacobian_determinant
                    )
                    load += (
                        shape * float(shape @ local_forcing) * jacobian_determinant
                    )

            for local_row, (row, column) in enumerate(nodes):
                if row in (0, points_y - 1) or column in (0, points_x - 1):
                    continue
                equation = interior_index(row, column)
                rhs[equation] += load[local_row]
                for local_column, (other_row, other_column) in enumerate(nodes):
                    if other_row in (0, points_y - 1) or other_column in (
                        0,
                        points_x - 1,
                    ):
                        continue
                    rows.append(equation)
                    columns.append(interior_index(other_row, other_column))
                    values.append(float(stiffness[local_row, local_column]))

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(unknowns, unknowns),
        dtype=float,
    ).tocsr()
    return matrix, rhs, kxx.shape


def solve_darcy_tensor_2d(
    permeability_xx: np.ndarray,
    permeability_xy: np.ndarray,
    permeability_yy: np.ndarray,
    forcing: Forcing2D = 1.0,
    length_x: float = 1.0,
    length_y: float = 1.0,
) -> np.ndarray:
    """Solve -div(K grad(u)) = f for a full symmetric positive tensor K.

    The domain is rectangular with homogeneous Dirichlet pressure. Tensor
    components are nodal values interpolated bilinearly inside Q1 elements and
    integrated with a 2x2 Gauss rule.
    """

    matrix, rhs, shape = _assemble_darcy_tensor_system(
        permeability_xx,
        permeability_xy,
        permeability_yy,
        forcing,
        length_x,
        length_y,
    )
    interior = spsolve(matrix, rhs)
    if not np.all(np.isfinite(interior)):
        raise FloatingPointError("tensor Darcy solve produced non-finite values")

    solution = np.zeros(shape, dtype=float)
    solution[1:-1, 1:-1] = interior.reshape(shape[0] - 2, shape[1] - 2)
    return solution


def darcy_tensor_residual_2d(
    solution: np.ndarray,
    permeability_xx: np.ndarray,
    permeability_xy: np.ndarray,
    permeability_yy: np.ndarray,
    forcing: Forcing2D = 1.0,
    length_x: float = 1.0,
    length_y: float = 1.0,
) -> np.ndarray:
    """Return the interior algebraic residual of the Q1 tensor discretization."""

    solution = _validated_real_grid(solution, "solution")
    matrix, rhs, shape = _assemble_darcy_tensor_system(
        permeability_xx,
        permeability_xy,
        permeability_yy,
        forcing,
        length_x,
        length_y,
    )
    if solution.shape != shape:
        raise ValueError("solution and permeability tensor must share grid shape")
    boundary = np.concatenate(
        (
            solution[0, :],
            solution[-1, :],
            solution[1:-1, 0],
            solution[1:-1, -1],
        )
    )
    if not np.allclose(boundary, 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("solution must satisfy homogeneous Dirichlet boundary data")
    residual = matrix @ solution[1:-1, 1:-1].reshape(-1) - rhs
    return residual.reshape(shape[0] - 2, shape[1] - 2)
