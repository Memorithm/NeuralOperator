"""Optional PyTorch backend for the one-dimensional Fourier operator.

The NumPy/SciPy implementation in :mod:`fno1d` is the small, dependency-light
reference.  This module keeps the same tensor contract while replacing finite
differences in the optimizer with PyTorch autograd.  PyTorch is deliberately
optional: importing the main package does not require it, and attempting to
instantiate this backend without it produces an actionable error.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import math
from typing import Any

try:  # The base project intentionally does not depend on PyTorch.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised without the extra
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


TORCH_AVAILABLE = torch is not None


@dataclass(frozen=True)
class TorchFNOFitResult:
    """Stable summary of an autograd-backed fit."""

    initial_loss: float
    final_loss: float
    epochs: int
    converged: bool
    device: str


def _missing_torch() -> RuntimeError:
    return RuntimeError(
        "PyTorch is required for TorchFNO1D; install the optional "
        "requirements with `python3 -m pip install -r requirements-autodiff.txt`"
    )


if torch is None:

    class TorchFNO1D:  # pragma: no cover - the class is a helpful import stub
        """Import-safe stub shown when the optional dependency is absent."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise _missing_torch()

else:

    class TorchFNO1D(nn.Module):
        """A one-block autograd-backed FNO for real periodic 1D fields.

        Inputs and outputs use ``(batch, points, channels)``.  The spectral
        matrix is real-valued and applied to the non-negative modes returned by
        ``torch.fft.rfft``; the inverse transform therefore returns a real
        field and follows the same restricted operator as ``NumpyFNO1D``.
        """

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            width: int = 8,
            modes: int = 8,
            seed: int | None = 0,
            dtype: torch.dtype = torch.float64,
            device: str | torch.device = "cpu",
        ) -> None:
            super().__init__()
            for name, value in (
                ("in_channels", in_channels),
                ("out_channels", out_channels),
                ("width", width),
                ("modes", modes),
            ):
                if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
                raise ValueError("dtype must be a floating-point torch.dtype")
            if seed is not None and (
                isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
            ):
                raise ValueError("seed must be a non-negative integer or None")

            self.in_channels = int(in_channels)
            self.out_channels = int(out_channels)
            self.width = int(width)
            self.modes = int(modes)
            self._dtype = dtype
            self._device = torch.device(device)

            # A local CPU generator makes initialization reproducible without
            # mutating PyTorch's process-wide random state.
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(int(seed))

            def parameter(shape: tuple[int, ...]) -> nn.Parameter:
                values = 0.15 * torch.randn(
                    shape, generator=generator, dtype=dtype, device="cpu"
                )
                return nn.Parameter(values)

            self.lift_weight = parameter((self.in_channels, self.width))
            self.lift_bias = nn.Parameter(torch.zeros(self.width, dtype=dtype))
            self.spectral_weight = parameter((self.modes, self.width, self.width))
            self.pointwise_weight = parameter((self.width, self.width))
            self.pointwise_bias = nn.Parameter(torch.zeros(self.width, dtype=dtype))
            self.projection_weight = parameter((self.width, self.out_channels))
            self.projection_bias = nn.Parameter(
                torch.zeros(self.out_channels, dtype=dtype)
            )
            self.to(device=self._device, dtype=dtype)

        @property
        def device(self) -> torch.device:
            """Return the device holding the model parameters."""

            return self.lift_weight.device

        @property
        def dtype(self) -> torch.dtype:
            """Return the floating-point dtype used by the model."""

            return self.lift_weight.dtype

        def _as_tensor(self, value: Any, name: str) -> torch.Tensor:
            tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")
            return tensor

        def _validate_inputs(
            self, inputs: Any, targets: Any | None = None
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            inputs_tensor = self._as_tensor(inputs, "inputs")
            if inputs_tensor.ndim != 3:
                raise ValueError("inputs must have shape (batch, points, channels)")
            if inputs_tensor.shape[-1] != self.in_channels:
                raise ValueError("input channel count does not match the model")
            if inputs_tensor.shape[1] < 2:
                raise ValueError("at least two spatial points are required")
            if targets is None:
                return inputs_tensor, None

            targets_tensor = self._as_tensor(targets, "targets")
            expected_shape = (
                inputs_tensor.shape[0],
                inputs_tensor.shape[1],
                self.out_channels,
            )
            if tuple(targets_tensor.shape) != expected_shape:
                raise ValueError(f"targets must have shape {expected_shape}")
            return inputs_tensor, targets_tensor

        def forward(self, inputs: Any) -> torch.Tensor:
            """Evaluate the operator at the input resolution."""

            inputs_tensor, _ = self._validate_inputs(inputs)
            hidden = torch.einsum(
                "bnc,cw->bnw", inputs_tensor, self.lift_weight
            )
            hidden = hidden + self.lift_bias

            spectrum = torch.fft.rfft(hidden, dim=1)
            usable_modes = min(self.modes, spectrum.shape[1])
            spectral_low = torch.einsum(
                "bmw,mwv->bmv",
                spectrum[:, :usable_modes, :],
                self.spectral_weight[:usable_modes].to(dtype=spectrum.dtype),
            )
            if usable_modes < spectrum.shape[1]:
                zero_tail = torch.zeros_like(spectrum[:, usable_modes:, :])
                filtered_spectrum = torch.cat((spectral_low, zero_tail), dim=1)
            else:
                filtered_spectrum = spectral_low
            spectral = torch.fft.irfft(
                filtered_spectrum, n=inputs_tensor.shape[1], dim=1
            )

            pointwise = torch.einsum(
                "bnw,wv->bnv", hidden, self.pointwise_weight
            )
            pointwise = pointwise + self.pointwise_bias
            hidden = torch.tanh(spectral + pointwise)
            return (
                torch.einsum("bnw,wo->bno", hidden, self.projection_weight)
                + self.projection_bias
            )

        def _loss_tensor(self, inputs: Any, targets: Any) -> torch.Tensor:
            inputs_tensor, targets_tensor = self._validate_inputs(inputs, targets)
            assert targets_tensor is not None
            prediction = self.forward(inputs_tensor)
            return torch.mean((prediction - targets_tensor) ** 2)

        def loss(self, inputs: Any, targets: Any) -> float:
            """Return the mean squared error for a batch."""

            return float(self._loss_tensor(inputs, targets).detach().cpu().item())

        def fit(
            self,
            inputs: Any,
            targets: Any,
            epochs: int = 100,
            learning_rate: float = 1.0e-2,
            tolerance: float = 0.0,
            device: str | torch.device | None = None,
        ) -> TorchFNOFitResult:
            """Fit the model with Adam and automatic differentiation.

            ``tolerance`` is an optional absolute loss-improvement threshold.
            A value of zero disables early stopping.  The full batch is kept
            in memory because this backend is intended as a small reference
            implementation before a production training pipeline is selected.
            """

            if not isinstance(epochs, int) or epochs <= 0:
                raise ValueError("epochs must be a positive integer")
            if (
                isinstance(learning_rate, bool)
                or not isinstance(learning_rate, Real)
                or not math.isfinite(float(learning_rate))
                or learning_rate <= 0
            ):
                raise ValueError("learning_rate must be finite and strictly positive")
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, Real)
                or not math.isfinite(float(tolerance))
                or tolerance < 0
            ):
                raise ValueError("tolerance must be finite and non-negative")
            if device is not None:
                self.to(device=device, dtype=self.dtype)

            inputs_tensor, targets_tensor = self._validate_inputs(inputs, targets)
            assert targets_tensor is not None
            optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
            initial_loss = float(
                self._loss_tensor(inputs_tensor, targets_tensor).detach().cpu().item()
            )
            previous_loss = initial_loss
            converged = False
            completed_epochs = 0
            self.train()
            for epoch in range(1, epochs + 1):
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss_tensor(inputs_tensor, targets_tensor)
                loss.backward()
                optimizer.step()
                completed_epochs = epoch
                current_loss = float(loss.detach().cpu().item())
                if tolerance > 0.0 and abs(previous_loss - current_loss) <= tolerance:
                    converged = True
                    break
                previous_loss = current_loss

            final_loss = float(
                self._loss_tensor(inputs_tensor, targets_tensor).detach().cpu().item()
            )
            return TorchFNOFitResult(
                initial_loss=initial_loss,
                final_loss=final_loss,
                epochs=completed_epochs,
                converged=converged,
                device=str(self.device),
            )
