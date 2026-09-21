"""Differentiable conservative Darcy physics for the optional PyTorch backend."""

from __future__ import annotations

import numpy as np

from .torch_physics import TorchPhysicsFitResult

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for differentiable Darcy physics; "
            "install requirements-autodiff.txt"
        )


def _validated_length(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _forcing_like(field, forcing):
    _require_torch()
    forcing_tensor = torch.as_tensor(
        forcing,
        dtype=field.dtype,
        device=field.device,
    )
    if not bool(torch.isfinite(forcing_tensor).all()):
        raise ValueError("forcing must contain only finite values")
    try:
        _, forcing_tensor = torch.broadcast_tensors(field, forcing_tensor)
    except RuntimeError as exc:
        raise ValueError(
            f"forcing shape {tuple(forcing_tensor.shape)} is not broadcastable "
            f"to field shape {tuple(field.shape)}"
        ) from exc
    return forcing_tensor


def torch_darcy_residual_2d(
    predicted_pressure,
    permeability,
    forcing=1.0,
    length_x: float = 1.0,
    length_y: float = 1.0,
):
    """Return the differentiable conservative interior Darcy residual.

    This mirrors the sparse NumPy/SciPy truth stencil: nodal permeability,
    harmonic face permeability, and homogeneous rectangular geometry. Boundary
    conditions are handled by the pressure field supplied by the model.
    """

    _require_torch()
    if not isinstance(predicted_pressure, torch.Tensor):
        predicted_pressure = torch.as_tensor(
            predicted_pressure,
            dtype=torch.float64,
        )
    permeability = torch.as_tensor(
        permeability,
        dtype=predicted_pressure.dtype,
        device=predicted_pressure.device,
    )
    predicted_pressure, permeability = torch.broadcast_tensors(
        predicted_pressure,
        permeability,
    )
    if predicted_pressure.ndim < 2:
        raise ValueError("pressure must contain two spatial axes")
    if predicted_pressure.shape[-2] < 3 or predicted_pressure.shape[-1] < 3:
        raise ValueError("at least three spatial points are required per axis")
    if not bool(torch.isfinite(predicted_pressure).all()):
        raise ValueError("predicted pressure must contain only finite values")
    if not bool(torch.isfinite(permeability).all()):
        raise ValueError("permeability must contain only finite values")
    if not bool((permeability > 0.0).all()):
        raise ValueError("permeability must be strictly positive")

    length_x = _validated_length(length_x, "length_x")
    length_y = _validated_length(length_y, "length_y")
    dx2 = (length_x / float(predicted_pressure.shape[-1] - 1)) ** 2
    dy2 = (length_y / float(predicted_pressure.shape[-2] - 1)) ** 2

    center_u = predicted_pressure[..., 1:-1, 1:-1]
    center_k = permeability[..., 1:-1, 1:-1]

    east_k = (
        2.0
        * center_k
        * permeability[..., 1:-1, 2:]
        / (center_k + permeability[..., 1:-1, 2:])
    )
    west_k = (
        2.0
        * center_k
        * permeability[..., 1:-1, :-2]
        / (center_k + permeability[..., 1:-1, :-2])
    )
    north_k = (
        2.0
        * center_k
        * permeability[..., 2:, 1:-1]
        / (center_k + permeability[..., 2:, 1:-1])
    )
    south_k = (
        2.0
        * center_k
        * permeability[..., :-2, 1:-1]
        / (center_k + permeability[..., :-2, 1:-1])
    )

    residual = (
        east_k * (center_u - predicted_pressure[..., 1:-1, 2:])
        + west_k * (center_u - predicted_pressure[..., 1:-1, :-2])
    ) / dx2
    residual = residual + (
        north_k * (center_u - predicted_pressure[..., 2:, 1:-1])
        + south_k * (center_u - predicted_pressure[..., :-2, 1:-1])
    ) / dy2

    forcing_tensor = _forcing_like(predicted_pressure, forcing)
    return residual - forcing_tensor[..., 1:-1, 1:-1]


def torch_darcy_data_physics_loss(
    predicted_pressure,
    target_pressure,
    permeability,
    forcing=1.0,
    data_weight: float = 1.0,
    physics_weight: float = 1.0e-4,
    length_x: float = 1.0,
    length_y: float = 1.0,
):
    """Return differentiable total, data-MSE and Darcy-residual-MSE losses."""

    _require_torch()
    if not isinstance(predicted_pressure, torch.Tensor):
        predicted_pressure = torch.as_tensor(
            predicted_pressure,
            dtype=torch.float64,
        )
    target_pressure = torch.as_tensor(
        target_pressure,
        dtype=predicted_pressure.dtype,
        device=predicted_pressure.device,
    )
    permeability = torch.as_tensor(
        permeability,
        dtype=predicted_pressure.dtype,
        device=predicted_pressure.device,
    )
    predicted_pressure, target_pressure, permeability = torch.broadcast_tensors(
        predicted_pressure,
        target_pressure,
        permeability,
    )
    for name, value in (
        ("data_weight", data_weight),
        ("physics_weight", physics_weight),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    data_loss = torch.mean((predicted_pressure - target_pressure) ** 2)
    residual = torch_darcy_residual_2d(
        predicted_pressure,
        permeability,
        forcing=forcing,
        length_x=length_x,
        length_y=length_y,
    )
    physics_loss = torch.mean(residual**2)
    total = float(data_weight) * data_loss + float(physics_weight) * physics_loss
    return total, data_loss, physics_loss


def fit_torch_fno_darcy_physics(
    model,
    inputs,
    targets,
    forcing=1.0,
    data_weight: float = 1.0,
    physics_weight: float = 1.0e-4,
    length_x: float = 1.0,
    length_y: float = 1.0,
    epochs: int = 100,
    learning_rate: float = 1.0e-2,
) -> TorchPhysicsFitResult:
    """Fit a one-channel 2D FNO with data plus conservative Darcy physics."""

    _require_torch()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(epochs, (int, np.integer)) or int(epochs) <= 0:
        raise ValueError("epochs must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and strictly positive")

    try:
        parameter = next(model.parameters())
    except StopIteration as exc:
        raise ValueError("model must expose trainable parameters") from exc

    input_tensor = torch.as_tensor(
        inputs,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    target_tensor = torch.as_tensor(
        targets,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    if (
        input_tensor.ndim != 4
        or target_tensor.ndim != 4
        or input_tensor.shape[-1] != 1
        or target_tensor.shape[-1] != 1
        or input_tensor.shape != target_tensor.shape
    ):
        raise ValueError(
            "inputs and targets must share shape "
            "(batch, points_y, points_x, 1)"
        )
    if not bool(torch.isfinite(input_tensor).all()) or not bool(
        torch.isfinite(target_tensor).all()
    ):
        raise ValueError("inputs and targets must contain only finite values")
    if not bool((input_tensor > 0.0).all()):
        raise ValueError("Darcy permeability inputs must be strictly positive")

    def losses():
        prediction = model(input_tensor)[..., 0]
        return torch_darcy_data_physics_loss(
            prediction,
            target_tensor[..., 0],
            input_tensor[..., 0],
            forcing=forcing,
            data_weight=data_weight,
            physics_weight=physics_weight,
            length_x=length_x,
            length_y=length_y,
        )

    initial_total, initial_data, initial_physics = losses()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = losses()
        total.backward()
        optimizer.step()
    final_total, final_data, final_physics = losses()

    return TorchPhysicsFitResult(
        initial_total_loss=float(initial_total.detach().cpu()),
        final_total_loss=float(final_total.detach().cpu()),
        initial_data_loss=float(initial_data.detach().cpu()),
        final_data_loss=float(final_data.detach().cpu()),
        initial_physics_loss=float(initial_physics.detach().cpu()),
        final_physics_loss=float(final_physics.detach().cpu()),
        epochs=int(epochs),
    )
