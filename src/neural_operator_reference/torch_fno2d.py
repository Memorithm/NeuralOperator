"""Optional PyTorch/autodiff 2D Fourier Neural Operator reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn.functional as torch_functional
except ImportError:  # pragma: no cover - exercised by the dependency-free path
    torch = None
    torch_functional = None


@dataclass(frozen=True)
class TorchFNO2DFitResult:
    """Summary of an Adam fit performed by the optional 2D backend."""

    initial_loss: float
    final_loss: float
    epochs: int


if torch is not None:

    class _SpectralConv2D(torch.nn.Module):
        """Low-mode complex Fourier mixing for real-valued 2D fields."""

        def __init__(self, width: int, modes_y: int, modes_x: int) -> None:
            super().__init__()
            self.width = int(width)
            self.modes_y = int(modes_y)
            self.modes_x = int(modes_x)
            scale = 1.0 / np.sqrt(float(width))
            shape = (self.modes_y, self.modes_x, self.width, self.width)
            self.weight_positive_real = torch.nn.Parameter(
                torch.randn(shape, dtype=torch.float64) * scale
            )
            self.weight_positive_imag = torch.nn.Parameter(
                torch.randn(shape, dtype=torch.float64) * scale
            )
            self.weight_negative_real = torch.nn.Parameter(
                torch.randn(shape, dtype=torch.float64) * scale
            )
            self.weight_negative_imag = torch.nn.Parameter(
                torch.randn(shape, dtype=torch.float64) * scale
            )

        @staticmethod
        def _complex_weight(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
            return torch.complex(real, imag)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            spectrum = torch.fft.rfft2(hidden, dim=(1, 2), norm="ortho")
            filtered = torch.zeros_like(spectrum)
            usable_y = min(self.modes_y, hidden.shape[1] // 2)
            usable_x = min(self.modes_x, spectrum.shape[2])
            if usable_y == 0 or usable_x == 0:
                return torch.zeros_like(hidden)

            positive_weight = self._complex_weight(
                self.weight_positive_real[:usable_y, :usable_x],
                self.weight_positive_imag[:usable_y, :usable_x],
            )
            negative_weight = self._complex_weight(
                self.weight_negative_real[:usable_y, :usable_x],
                self.weight_negative_imag[:usable_y, :usable_x],
            )
            filtered[:, :usable_y, :usable_x, :] = torch.einsum(
                "bxyc,xyco->bxyo",
                spectrum[:, :usable_y, :usable_x, :],
                positive_weight,
            )
            filtered[:, -usable_y:, :usable_x, :] = torch.einsum(
                "bxyc,xyco->bxyo",
                spectrum[:, -usable_y:, :usable_x, :],
                negative_weight,
            )
            return torch.fft.irfft2(
                filtered,
                s=(hidden.shape[1], hidden.shape[2]),
                dim=(1, 2),
                norm="ortho",
            )


    class TorchFNO2D(torch.nn.Module):
        """Coordinate-aware 2D FNO with an optional hard Dirichlet constraint.

        Inputs use shape (batch, points_y, points_x, channels). Two normalized
        coordinate channels are appended before lifting. When hard_dirichlet
        is enabled, the final prediction is multiplied by an analytic envelope
        that is exactly zero on every rectangular boundary.
        """

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            width: int = 12,
            modes_y: int = 6,
            modes_x: int = 6,
            depth: int = 3,
            padding: int = 2,
            hard_dirichlet: bool = True,
            seed: int = 0,
        ) -> None:
            super().__init__()
            for name, value in (
                ("in_channels", in_channels),
                ("out_channels", out_channels),
                ("width", width),
                ("modes_y", modes_y),
                ("modes_x", modes_x),
                ("depth", depth),
            ):
                if not isinstance(value, (int, np.integer)) or int(value) <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if not isinstance(padding, (int, np.integer)) or int(padding) < 0:
                raise ValueError("padding must be a non-negative integer")
            if not isinstance(seed, (int, np.integer)):
                raise ValueError("seed must be an integer")

            torch.manual_seed(int(seed))
            self.in_channels = int(in_channels)
            self.out_channels = int(out_channels)
            self.width = int(width)
            self.modes_y = int(modes_y)
            self.modes_x = int(modes_x)
            self.depth = int(depth)
            self.padding = int(padding)
            self.hard_dirichlet = bool(hard_dirichlet)

            self.lift = torch.nn.Linear(self.in_channels + 2, self.width)
            self.spectral_layers = torch.nn.ModuleList(
                [
                    _SpectralConv2D(self.width, self.modes_y, self.modes_x)
                    for _ in range(self.depth)
                ]
            )
            self.pointwise_layers = torch.nn.ModuleList(
                [torch.nn.Linear(self.width, self.width) for _ in range(self.depth)]
            )
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(self.width, self.width),
                torch.nn.Tanh(),
                torch.nn.Linear(self.width, self.out_channels),
            )
            self.double()

        @property
        def device(self) -> torch.device:
            return self.lift.weight.device

        def _inputs(self, inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
            if isinstance(inputs, torch.Tensor):
                tensor = inputs.to(device=self.device, dtype=torch.float64)
            else:
                tensor = torch.as_tensor(
                    inputs,
                    device=self.device,
                    dtype=torch.float64,
                )
            if tensor.ndim != 4:
                raise ValueError(
                    "inputs must have shape (batch, points_y, points_x, channels)"
                )
            if tensor.shape[0] == 0:
                raise ValueError("inputs must contain at least one sample")
            if tensor.shape[-1] != self.in_channels:
                raise ValueError("input channel count does not match the model")
            if tensor.shape[1] < 3 or tensor.shape[2] < 3:
                raise ValueError("at least three spatial points are required per axis")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("inputs must contain only finite values")
            return tensor

        def _targets(
            self,
            inputs: torch.Tensor,
            targets: np.ndarray | torch.Tensor,
        ) -> torch.Tensor:
            if isinstance(targets, torch.Tensor):
                tensor = targets.to(device=self.device, dtype=torch.float64)
            else:
                tensor = torch.as_tensor(
                    targets,
                    device=self.device,
                    dtype=torch.float64,
                )
            expected_shape = (
                inputs.shape[0],
                inputs.shape[1],
                inputs.shape[2],
                self.out_channels,
            )
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"targets must have shape {expected_shape}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("targets must contain only finite values")
            return tensor

        @staticmethod
        def _coordinate_grid(
            points_y: int,
            points_x: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            y = torch.linspace(0.0, 1.0, points_y, device=device, dtype=dtype)
            x = torch.linspace(0.0, 1.0, points_x, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            return torch.stack((grid_x, grid_y), dim=-1)

        def _boundary_envelope(self, inputs: torch.Tensor) -> torch.Tensor:
            coordinates = self._coordinate_grid(
                inputs.shape[1],
                inputs.shape[2],
                device=inputs.device,
                dtype=inputs.dtype,
            )
            x = coordinates[..., 0]
            y = coordinates[..., 1]
            envelope = 16.0 * x * (1.0 - x) * y * (1.0 - y)
            return envelope.unsqueeze(0).unsqueeze(-1)

        def forward(self, inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
            """Evaluate the operator at the input grid resolution."""

            input_tensor = self._inputs(inputs)
            batch, points_y, points_x, _ = input_tensor.shape
            coordinates = self._coordinate_grid(
                points_y,
                points_x,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ).unsqueeze(0).expand(batch, -1, -1, -1)
            hidden = self.lift(torch.cat((input_tensor, coordinates), dim=-1))

            if self.padding:
                hidden = torch_functional.pad(
                    hidden.permute(0, 3, 1, 2),
                    (0, self.padding, 0, self.padding),
                ).permute(0, 2, 3, 1)

            for spectral, pointwise in zip(
                self.spectral_layers,
                self.pointwise_layers,
            ):
                hidden = torch.tanh(spectral(hidden) + pointwise(hidden))

            if self.padding:
                hidden = hidden[:, :points_y, :points_x, :]
            output = self.projection(hidden)
            if self.hard_dirichlet:
                output = output * self._boundary_envelope(input_tensor)
            return output

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
            """Return mean squared error without retaining an autograd graph."""

            with torch.no_grad():
                return float(self._loss_tensor(inputs, targets).cpu())

        def fit(
            self,
            inputs: np.ndarray | torch.Tensor,
            targets: np.ndarray | torch.Tensor,
            epochs: int = 100,
            learning_rate: float = 1.0e-2,
        ) -> TorchFNO2DFitResult:
            """Fit the reference model with Adam and PyTorch autograd."""

            if not isinstance(epochs, (int, np.integer)) or int(epochs) <= 0:
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
            return TorchFNO2DFitResult(
                initial_loss=initial_loss,
                final_loss=final_loss,
                epochs=int(epochs),
            )


else:

    class TorchFNO2D:
        """Placeholder that reports the optional dependency requirement."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required for TorchFNO2D; "
                "install requirements-autodiff.txt"
            )
