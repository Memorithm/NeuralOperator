"""Optional PyTorch/autodiff 1D Fourier Neural Operator reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised by the dependency-free path
    torch = None


@dataclass(frozen=True)
class TorchFNOFitResult:
    """Summary of an Adam fit performed by the optional backend."""

    initial_loss: float
    final_loss: float
    epochs: int


if torch is not None:

    class TorchFNO1D(torch.nn.Module):
        """A float64 one-block FNO trained with PyTorch autograd.

        The tensor contract matches NumpyFNO1D: inputs and outputs have shape
        (batch, points, channels). The implementation is intentionally small;
        it is an autodiff reference, not a production training stack.
        """

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            width: int = 8,
            modes: int = 8,
            seed: int = 0,
        ) -> None:
            super().__init__()
            for name, value in (
                ("in_channels", in_channels),
                ("out_channels", out_channels),
                ("width", width),
                ("modes", modes),
            ):
                if not isinstance(value, (int, np.integer)) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if not isinstance(seed, (int, np.integer)):
                raise ValueError("seed must be an integer")

            torch.manual_seed(int(seed))
            self.in_channels = int(in_channels)
            self.out_channels = int(out_channels)
            self.width = int(width)
            self.modes = int(modes)

            def init(shape: tuple[int, ...]) -> torch.Tensor:
                return torch.randn(shape, dtype=torch.float64) * 0.15

            self.lift_weight = torch.nn.Parameter(
                init((self.in_channels, self.width))
            )
            self.lift_bias = torch.nn.Parameter(
                torch.zeros(self.width, dtype=torch.float64)
            )
            self.spectral_weight = torch.nn.Parameter(
                init((self.modes, self.width, self.width))
            )
            self.pointwise_weight = torch.nn.Parameter(
                init((self.width, self.width))
            )
            self.pointwise_bias = torch.nn.Parameter(
                torch.zeros(self.width, dtype=torch.float64)
            )
            self.projection_weight = torch.nn.Parameter(
                init((self.width, self.out_channels))
            )
            self.projection_bias = torch.nn.Parameter(
                torch.zeros(self.out_channels, dtype=torch.float64)
            )

        def _inputs(self, inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
            if isinstance(inputs, torch.Tensor):
                tensor = inputs.to(
                    device=self.lift_weight.device,
                    dtype=torch.float64,
                )
            else:
                tensor = torch.as_tensor(
                    inputs,
                    device=self.lift_weight.device,
                    dtype=torch.float64,
                )
            if tensor.ndim != 3:
                raise ValueError("inputs must have shape (batch, points, channels)")
            if tensor.shape[-1] != self.in_channels:
                raise ValueError("input channel count does not match the model")
            if tensor.shape[1] < 2:
                raise ValueError("at least two spatial points are required")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("inputs must contain only finite values")
            return tensor

        def _targets(
            self,
            inputs: torch.Tensor,
            targets: np.ndarray | torch.Tensor,
        ) -> torch.Tensor:
            if isinstance(targets, torch.Tensor):
                target_tensor = targets.to(
                    device=self.lift_weight.device,
                    dtype=torch.float64,
                )
            else:
                target_tensor = torch.as_tensor(
                    targets,
                    device=self.lift_weight.device,
                    dtype=torch.float64,
                )
            if target_tensor.ndim != 3:
                raise ValueError(
                    "targets must have shape (batch, points, channels)"
                )
            expected_shape = (
                inputs.shape[0],
                inputs.shape[1],
                self.out_channels,
            )
            if tuple(target_tensor.shape) != expected_shape:
                raise ValueError(f"targets must have shape {expected_shape}")
            if not bool(torch.isfinite(target_tensor).all()):
                raise ValueError("targets must contain only finite values")
            return target_tensor

        def forward(self, inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
            """Evaluate the FNO at the input resolution."""

            inputs = self._inputs(inputs)
            hidden = torch.einsum("bnc,cw->bnw", inputs, self.lift_weight)
            hidden = hidden + self.lift_bias

            spectrum = torch.fft.rfft(hidden, dim=1)
            filtered_spectrum = torch.zeros_like(spectrum)
            usable_modes = min(self.modes, spectrum.shape[1])
            complex_weight = self.spectral_weight.to(dtype=spectrum.dtype)
            for mode in range(usable_modes):
                filtered_spectrum[:, mode, :] = (
                    spectrum[:, mode, :] @ complex_weight[mode]
                )
            spectral = torch.fft.irfft(
                filtered_spectrum,
                n=inputs.shape[1],
                dim=1,
            )

            pointwise = torch.einsum(
                "bnw,wv->bnv",
                hidden,
                self.pointwise_weight,
            )
            pointwise = pointwise + self.pointwise_bias
            hidden = torch.tanh(spectral + pointwise)
            return (
                torch.einsum("bnw,wo->bno", hidden, self.projection_weight)
                + self.projection_bias
            )

        def _loss_tensor(
            self,
            inputs: np.ndarray | torch.Tensor,
            targets: np.ndarray | torch.Tensor,
        ) -> torch.Tensor:
            input_tensor = self._inputs(inputs)
            target_tensor = self._targets(input_tensor, targets)
            return torch.mean((self.forward(input_tensor) - target_tensor) ** 2)

        def loss(
            self,
            inputs: np.ndarray | torch.Tensor,
            targets: np.ndarray | torch.Tensor,
        ) -> float:
            """Return mean squared error without constructing a graph."""

            with torch.no_grad():
                return float(self._loss_tensor(inputs, targets).cpu())

        def fit(
            self,
            inputs: np.ndarray | torch.Tensor,
            targets: np.ndarray | torch.Tensor,
            epochs: int = 100,
            learning_rate: float = 1.0e-2,
        ) -> TorchFNOFitResult:
            """Fit the reference model with Adam and PyTorch autograd."""

            if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
                raise ValueError("epochs must be a positive integer")
            if not np.isfinite(learning_rate) or learning_rate <= 0.0:
                raise ValueError(
                    "learning_rate must be finite and strictly positive"
                )
            input_tensor = self._inputs(inputs)
            target_tensor = self._targets(input_tensor, targets)
            optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
            initial_loss = float(
                self._loss_tensor(input_tensor, target_tensor).detach().cpu()
            )
            for _ in range(int(epochs)):
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss_tensor(input_tensor, target_tensor)
                loss.backward()
                optimizer.step()
            final_loss = float(
                self._loss_tensor(input_tensor, target_tensor).detach().cpu()
            )
            return TorchFNOFitResult(
                initial_loss=initial_loss,
                final_loss=final_loss,
                epochs=int(epochs),
            )

else:

    class TorchFNO1D:
        """Placeholder that reports the optional dependency requirement."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required for TorchFNO1D; install requirements-autodiff.txt"
            )
