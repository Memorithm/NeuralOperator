"""Optional nonlinear PyTorch DeepONet reference for 2D scalar fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised by dependency-free CI
    torch = None


@dataclass(frozen=True)
class TorchDeepONet2DFitResult:
    """Summary of an Adam fit for the 2D DeepONet reference."""

    initial_loss: float
    final_loss: float
    epochs: int


def normalized_query_grid(points_y: int, points_x: int) -> np.ndarray:
    """Return flattened normalized x/y coordinates for a rectangular grid."""

    for name, value in (("points_y", points_y), ("points_x", points_x)):
        if not isinstance(value, (int, np.integer)) or int(value) < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    y = np.linspace(0.0, 1.0, int(points_y))
    x = np.linspace(0.0, 1.0, int(points_x))
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    return np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1)


if torch is not None:

    class TorchDeepONet2D(torch.nn.Module):
        """Nonlinear branch/trunk operator for 2D scalar Darcy fields."""

        def __init__(
            self,
            sensor_points_y: int,
            sensor_points_x: int,
            in_channels: int = 1,
            hidden_width: int = 40,
            latent_width: int = 64,
            hard_dirichlet: bool = True,
            seed: int = 0,
        ) -> None:
            super().__init__()
            for name, value in (
                ("sensor_points_y", sensor_points_y),
                ("sensor_points_x", sensor_points_x),
                ("in_channels", in_channels),
                ("hidden_width", hidden_width),
                ("latent_width", latent_width),
            ):
                if not isinstance(value, (int, np.integer)) or int(value) <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if int(sensor_points_y) < 2 or int(sensor_points_x) < 2:
                raise ValueError("sensor grid must contain at least two points per axis")
            if not isinstance(seed, (int, np.integer)):
                raise ValueError("seed must be an integer")

            torch.manual_seed(int(seed))
            self.sensor_points_y = int(sensor_points_y)
            self.sensor_points_x = int(sensor_points_x)
            self.in_channels = int(in_channels)
            self.hidden_width = int(hidden_width)
            self.latent_width = int(latent_width)
            self.hard_dirichlet = bool(hard_dirichlet)

            sensor_values = (
                self.sensor_points_y
                * self.sensor_points_x
                * self.in_channels
            )
            self.branch = torch.nn.Sequential(
                torch.nn.Linear(sensor_values, self.hidden_width),
                torch.nn.Tanh(),
                torch.nn.Linear(self.hidden_width, self.latent_width),
            )
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(2, self.hidden_width),
                torch.nn.Tanh(),
                torch.nn.Linear(self.hidden_width, self.latent_width),
            )
            self.output_bias = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.float64)
            )
            self.double()

        @property
        def device(self) -> torch.device:
            return self.output_bias.device

        def _inputs(self, inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
            if isinstance(inputs, torch.Tensor):
                tensor = inputs.to(device=self.device, dtype=torch.float64)
            else:
                tensor = torch.as_tensor(
                    inputs,
                    device=self.device,
                    dtype=torch.float64,
                )
            expected = (
                self.sensor_points_y,
                self.sensor_points_x,
                self.in_channels,
            )
            if tensor.ndim != 4 or tuple(tensor.shape[1:]) != expected:
                raise ValueError(
                    "inputs must have shape "
                    "(batch, sensor_points_y, sensor_points_x, in_channels)"
                )
            if tensor.shape[0] == 0:
                raise ValueError("inputs must contain at least one sample")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("inputs must contain only finite values")
            return tensor

        def _coordinates(
            self,
            coordinates: np.ndarray | torch.Tensor,
        ) -> torch.Tensor:
            if isinstance(coordinates, torch.Tensor):
                tensor = coordinates.to(device=self.device, dtype=torch.float64)
            else:
                tensor = torch.as_tensor(
                    coordinates,
                    device=self.device,
                    dtype=torch.float64,
                )
            if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] != 2:
                raise ValueError(
                    "coordinates must have shape (query_points, 2)"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("coordinates must contain only finite values")
            if not bool(((tensor >= 0.0) & (tensor <= 1.0)).all()):
                raise ValueError("coordinates must lie in the normalized [0, 1] domain")
            return tensor

        def _targets(
            self,
            inputs: torch.Tensor,
            coordinates: torch.Tensor,
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
            expected = (inputs.shape[0], coordinates.shape[0], 1)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"targets must have shape {expected}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("targets must contain only finite values")
            return tensor

        @staticmethod
        def _boundary_envelope(coordinates: torch.Tensor) -> torch.Tensor:
            x = coordinates[:, 0]
            y = coordinates[:, 1]
            return 16.0 * x * (1.0 - x) * y * (1.0 - y)

        def forward(
            self,
            inputs: np.ndarray | torch.Tensor,
            coordinates: np.ndarray | torch.Tensor,
        ) -> torch.Tensor:
            """Evaluate pressure at arbitrary normalized query coordinates."""

            input_tensor = self._inputs(inputs)
            coordinate_tensor = self._coordinates(coordinates)
            flattened = input_tensor.reshape(input_tensor.shape[0], -1)
            branch = self.branch(flattened)
            trunk = self.trunk(coordinate_tensor)
            output = torch.einsum("bl,pl->bp", branch, trunk)
            output = output + self.output_bias
            if self.hard_dirichlet:
                output = output * self._boundary_envelope(coordinate_tensor)[None, :]
            return output.unsqueeze(-1)

        def _loss_tensor(
            self,
            inputs: np.ndarray | torch.Tensor,
            coordinates: np.ndarray | torch.Tensor,
            targets: np.ndarray | torch.Tensor,
        ) -> torch.Tensor:
            input_tensor = self._inputs(inputs)
            coordinate_tensor = self._coordinates(coordinates)
            target_tensor = self._targets(
                input_tensor,
                coordinate_tensor,
                targets,
            )
            return torch.mean(
                (self.forward(input_tensor, coordinate_tensor) - target_tensor) ** 2
            )

        def loss(
            self,
            inputs: np.ndarray | torch.Tensor,
            coordinates: np.ndarray | torch.Tensor,
            targets: np.ndarray | torch.Tensor,
        ) -> float:
            """Return mean squared error without retaining an autograd graph."""

            with torch.no_grad():
                return float(
                    self._loss_tensor(inputs, coordinates, targets).cpu()
                )

        def fit(
            self,
            inputs: np.ndarray | torch.Tensor,
            coordinates: np.ndarray | torch.Tensor,
            targets: np.ndarray | torch.Tensor,
            epochs: int = 100,
            learning_rate: float = 1.0e-2,
        ) -> TorchDeepONet2DFitResult:
            """Fit branch and trunk networks with Adam."""

            if not isinstance(epochs, (int, np.integer)) or int(epochs) <= 0:
                raise ValueError("epochs must be a positive integer")
            if not np.isfinite(learning_rate) or learning_rate <= 0.0:
                raise ValueError(
                    "learning_rate must be finite and strictly positive"
                )

            input_tensor = self._inputs(inputs)
            coordinate_tensor = self._coordinates(coordinates)
            target_tensor = self._targets(
                input_tensor,
                coordinate_tensor,
                targets,
            )
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=float(learning_rate),
            )
            initial_loss = float(
                self._loss_tensor(
                    input_tensor,
                    coordinate_tensor,
                    target_tensor,
                ).detach().cpu()
            )
            for _ in range(int(epochs)):
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss_tensor(
                    input_tensor,
                    coordinate_tensor,
                    target_tensor,
                )
                loss.backward()
                optimizer.step()
            final_loss = float(
                self._loss_tensor(
                    input_tensor,
                    coordinate_tensor,
                    target_tensor,
                ).detach().cpu()
            )
            return TorchDeepONet2DFitResult(
                initial_loss=initial_loss,
                final_loss=final_loss,
                epochs=int(epochs),
            )


else:

    class TorchDeepONet2D:
        """Placeholder that reports the optional dependency requirement."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required for TorchDeepONet2D; "
                "install requirements-autodiff.txt"
            )
