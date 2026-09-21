"""Optional local convolution baseline for 2D Darcy operator learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised by dependency-free CI
    torch = None


@dataclass(frozen=True)
class TorchLocalConv2DFitResult:
    """Summary of an Adam fit for the local convolution baseline."""

    initial_loss: float
    final_loss: float
    epochs: int


if torch is not None:

    class TorchLocalConv2D(torch.nn.Module):
        """Coordinate-aware finite-receptive-field baseline on rectangular grids."""

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            width: int = 29,
            hard_dirichlet: bool = True,
            seed: int = 0,
        ) -> None:
            super().__init__()
            for name, value in (
                ("in_channels", in_channels),
                ("out_channels", out_channels),
                ("width", width),
            ):
                if not isinstance(value, (int, np.integer)) or int(value) <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if not isinstance(seed, (int, np.integer)):
                raise ValueError("seed must be an integer")

            torch.manual_seed(int(seed))
            self.in_channels = int(in_channels)
            self.out_channels = int(out_channels)
            self.width = int(width)
            self.hard_dirichlet = bool(hard_dirichlet)

            self.network = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels + 2, self.width, 3, padding=1),
                torch.nn.Tanh(),
                torch.nn.Conv2d(self.width, self.width, 3, padding=1),
                torch.nn.Tanh(),
                torch.nn.Conv2d(self.width, self.out_channels, 3, padding=1),
            )
            self.double()

        @property
        def device(self) -> torch.device:
            return next(self.parameters()).device

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
            expected = (
                inputs.shape[0],
                inputs.shape[1],
                inputs.shape[2],
                self.out_channels,
            )
            if tuple(tensor.shape) != expected:
                raise ValueError(f"targets must have shape {expected}")
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
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            y = torch.linspace(0.0, 1.0, points_y, device=device, dtype=dtype)
            x = torch.linspace(0.0, 1.0, points_x, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            coordinates = torch.stack((grid_x, grid_y), dim=-1)
            return coordinates, grid_x, grid_y

        def forward(self, inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
            """Evaluate the local baseline at the input resolution."""

            input_tensor = self._inputs(inputs)
            batch, points_y, points_x, _ = input_tensor.shape
            coordinates, grid_x, grid_y = self._coordinate_grid(
                points_y,
                points_x,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
            coordinates = coordinates.unsqueeze(0).expand(batch, -1, -1, -1)
            features = torch.cat((input_tensor, coordinates), dim=-1)
            output = self.network(features.permute(0, 3, 1, 2))
            output = output.permute(0, 2, 3, 1)

            if self.hard_dirichlet:
                envelope = (
                    16.0
                    * grid_x
                    * (1.0 - grid_x)
                    * grid_y
                    * (1.0 - grid_y)
                )
                output = output * envelope.unsqueeze(0).unsqueeze(-1)
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
        ) -> TorchLocalConv2DFitResult:
            """Fit the local baseline with Adam."""

            if not isinstance(epochs, (int, np.integer)) or int(epochs) <= 0:
                raise ValueError("epochs must be a positive integer")
            if not np.isfinite(learning_rate) or learning_rate <= 0.0:
                raise ValueError(
                    "learning_rate must be finite and strictly positive"
                )

            input_tensor = self._inputs(inputs)
            target_tensor = self._targets(input_tensor, targets)
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=float(learning_rate),
            )
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
            return TorchLocalConv2DFitResult(
                initial_loss=initial_loss,
                final_loss=final_loss,
                epochs=int(epochs),
            )


else:

    class TorchLocalConv2D:
        """Placeholder that reports the optional dependency requirement."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required for TorchLocalConv2D; "
                "install requirements-autodiff.txt"
            )
