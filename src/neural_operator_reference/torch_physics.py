"""Differentiable spectral physics losses for the optional PyTorch backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


@dataclass(frozen=True)
class TorchPhysicsFitResult:
    """Summary of a differentiable data-plus-physics Adam fit."""

    initial_total_loss: float
    final_total_loss: float
    initial_data_loss: float
    final_data_loss: float
    initial_physics_loss: float
    final_physics_loss: float
    epochs: int


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for differentiable spectral physics; "
            "install requirements-autodiff.txt"
        )


def _validated_length(length: float) -> float:
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and strictly positive")
    return float(length)


def _validated_axis(axis: int, ndim: int) -> int:
    if ndim == 0:
        raise ValueError("values must contain a spatial axis")
    if not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be an integer")
    axis = int(axis)
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is outside [-{ndim}, {ndim - 1}]")
    return axis % ndim


def torch_spectral_derivative(
    values,
    order: int = 1,
    length: float = 2.0 * np.pi,
    axis: int = -1,
):
    """Differentiate a periodic tensor with a Fourier multiplier.

    The operation remains inside PyTorch's autograd graph. It mirrors the
    NumPy spectral oracle and is intended for differentiable PDE residuals.
    """

    _require_torch()
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, dtype=torch.float64)
    if values.ndim == 0:
        raise ValueError("values must contain a spatial axis")
    if not (values.is_floating_point() or torch.is_complex(values)):
        values = values.to(dtype=torch.float64)
    if not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError("order must be a non-negative integer")
    axis = _validated_axis(axis, values.ndim)
    length = _validated_length(length)
    if int(order) == 0:
        return values.clone()

    points = int(values.shape[axis])
    if points < 2:
        raise ValueError("at least two spatial points are required")
    real_dtype = values.real.dtype
    wave_numbers = (
        2.0
        * np.pi
        * torch.fft.fftfreq(
            points,
            d=length / points,
            device=values.device,
            dtype=real_dtype,
        )
    )
    multiplier = (1j * wave_numbers) ** int(order)
    shape = [1] * values.ndim
    shape[axis] = points
    spectrum = torch.fft.fft(values, dim=axis)
    differentiated = torch.fft.ifft(
        spectrum * multiplier.reshape(shape),
        dim=axis,
    )
    return differentiated if torch.is_complex(values) else differentiated.real


def torch_dealiased_product(
    left,
    right,
    axis: int = -1,
    keep_ratio: float = 2.0 / 3.0,
):
    """Multiply periodic tensors and remove generated high Fourier modes."""

    _require_torch()
    if not isinstance(left, torch.Tensor):
        left = torch.as_tensor(left, dtype=torch.float64)
    if not isinstance(right, torch.Tensor):
        right = torch.as_tensor(
            right,
            dtype=left.dtype,
            device=left.device,
        )
    left, right = torch.broadcast_tensors(left, right)
    axis = _validated_axis(axis, left.ndim)
    if not np.isfinite(keep_ratio) or not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")

    points = int(left.shape[axis])
    real_dtype = left.real.dtype
    mode_indices = torch.fft.fftfreq(
        points,
        d=1.0 / points,
        device=left.device,
        dtype=real_dtype,
    )
    cutoff = int(np.floor(points * float(keep_ratio) / 2.0))
    mask = torch.abs(mode_indices) <= cutoff
    shape = [1] * left.ndim
    shape[axis] = points

    product = left * right
    filtered = torch.fft.ifft(
        torch.fft.fft(product, dim=axis) * mask.reshape(shape),
        dim=axis,
    )
    return filtered if torch.is_complex(product) else filtered.real


def _forcing_like(field, forcing):
    _require_torch()
    if forcing is None:
        return torch.zeros((), dtype=field.dtype, device=field.device)
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


def torch_burgers_rhs(
    field,
    viscosity: float,
    length: float = 2.0 * np.pi,
    forcing=None,
):
    """Evaluate forced viscous Burgers RHS with differentiable spectral terms."""

    _require_torch()
    if not isinstance(field, torch.Tensor):
        field = torch.as_tensor(field, dtype=torch.float64)
    if field.ndim == 0 or field.shape[-1] < 3:
        raise ValueError("field must contain a spatial axis with at least 3 points")
    if not np.isfinite(viscosity) or viscosity < 0.0:
        raise ValueError("viscosity must be finite and non-negative")
    length = _validated_length(length)

    field_x = torch_spectral_derivative(field, order=1, length=length, axis=-1)
    field_xx = torch_spectral_derivative(field, order=2, length=length, axis=-1)
    advection = torch_dealiased_product(field, field_x, axis=-1)
    return (
        -advection
        + float(viscosity) * field_xx
        + _forcing_like(field, forcing)
    )


def torch_burgers_transition_residual(
    initial_field,
    predicted_field,
    horizon: float,
    viscosity: float,
    length: float = 2.0 * np.pi,
    forcing=None,
):
    """Return the differentiable one-transition Burgers residual tensor."""

    _require_torch()
    if not isinstance(predicted_field, torch.Tensor):
        predicted_field = torch.as_tensor(predicted_field, dtype=torch.float64)
    initial_field = torch.as_tensor(
        initial_field,
        dtype=predicted_field.dtype,
        device=predicted_field.device,
    )
    initial_field, predicted_field = torch.broadcast_tensors(
        initial_field,
        predicted_field,
    )
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and strictly positive")
    temporal_derivative = (
        predicted_field - initial_field
    ) / float(horizon)
    return temporal_derivative - torch_burgers_rhs(
        predicted_field,
        viscosity=viscosity,
        length=length,
        forcing=forcing,
    )


def torch_burgers_data_physics_loss(
    initial_field,
    predicted_field,
    target_field,
    horizon: float,
    viscosity: float,
    data_weight: float = 1.0,
    physics_weight: float = 1.0e-3,
    length: float = 2.0 * np.pi,
    forcing=None,
):
    """Return differentiable (total, data MSE, physics MSE) losses."""

    _require_torch()
    if not isinstance(predicted_field, torch.Tensor):
        predicted_field = torch.as_tensor(predicted_field, dtype=torch.float64)
    target_field = torch.as_tensor(
        target_field,
        dtype=predicted_field.dtype,
        device=predicted_field.device,
    )
    initial_field = torch.as_tensor(
        initial_field,
        dtype=predicted_field.dtype,
        device=predicted_field.device,
    )
    initial_field, predicted_field, target_field = torch.broadcast_tensors(
        initial_field,
        predicted_field,
        target_field,
    )
    for name, value in (
        ("data_weight", data_weight),
        ("physics_weight", physics_weight),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    data_loss = torch.mean((predicted_field - target_field) ** 2)
    residual = torch_burgers_transition_residual(
        initial_field,
        predicted_field,
        horizon=horizon,
        viscosity=viscosity,
        length=length,
        forcing=forcing,
    )
    physics_loss = torch.mean(residual**2)
    total = float(data_weight) * data_loss + float(physics_weight) * physics_loss
    return total, data_loss, physics_loss


def fit_torch_fno_burgers_physics(
    model,
    inputs,
    targets,
    horizon: float,
    viscosity: float,
    data_weight: float = 1.0,
    physics_weight: float = 1.0e-3,
    length: float = 2.0 * np.pi,
    forcing=None,
    epochs: int = 100,
    learning_rate: float = 1.0e-2,
) -> TorchPhysicsFitResult:
    """Fit a one-channel FNO through a differentiable spectral physics loss."""

    _require_torch()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
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
        input_tensor.ndim != 3
        or target_tensor.ndim != 3
        or input_tensor.shape[-1] != 1
        or target_tensor.shape[-1] != 1
        or input_tensor.shape != target_tensor.shape
    ):
        raise ValueError(
            "inputs and targets must share shape (batch, points, 1)"
        )
    if not bool(torch.isfinite(input_tensor).all()) or not bool(
        torch.isfinite(target_tensor).all()
    ):
        raise ValueError("inputs and targets must contain only finite values")

    def losses():
        prediction = model(input_tensor)[..., 0]
        return torch_burgers_data_physics_loss(
            input_tensor[..., 0],
            prediction,
            target_tensor[..., 0],
            horizon=horizon,
            viscosity=viscosity,
            data_weight=data_weight,
            physics_weight=physics_weight,
            length=length,
            forcing=forcing,
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
