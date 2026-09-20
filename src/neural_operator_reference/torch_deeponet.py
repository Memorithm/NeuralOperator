"""Optional nonlinear PyTorch DeepONet reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised by the dependency-free path
    torch = None


@dataclass(frozen=True)
class TorchDeepONetFitResult:
    """Summary of an Adam fit performed by the optional backend."""

    initial_loss: float
    final_loss: float
    epochs: int


if torch is not None:

    class TorchDeepONet1D(torch.nn.Module):
        """A nonlinear branch/trunk DeepONet with arbitrary query coordinates.

        The branch consumes fixed sensor values with shape
        (batch, sensor_points, 1). The trunk consumes a one-dimensional query
        coordinate array and the output has shape (batch, query_points, 1).
        """

        def __init__(
            self,
            sensor_points: int,
            hidden_width: int = 32,
            latent_width: int = 16,
            length: float = 2.0 * np.pi,
            seed: int = 0,
        ) -> None:
            super().__init__()
            for name, value in (
                ("sensor_points", sensor_points),
                ("hidden_width", hidden_width),
                ("latent_width", latent_width),
            ):
                if not isinstance(value, (int, np.integer)) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if sensor_points < 2:
                raise ValueError(
                    "sensor_points must be an integer greater than or equal to 2"
                )
            if not np.isfinite(length) or length <= 0.0:
                raise ValueError("length must be finite and strictly positive")
            if not isinstance(seed, (int, np.integer)):
                raise ValueError("seed must be an integer")

            torch.manual_seed(int(seed))
            self.sensor_points = int(sensor_points)
            self.hidden_width = int(hidden_width)
            self.latent_width = int(latent_width)
            self.length = float(length)
            self.branch = torch.nn.Sequential(
                torch.nn.Linear(self.sensor_points, self.hidden_width),
                torch.nn.Tanh(),
                torch.nn.Linear(self.hidden_width, self.latent_width),
            )
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(1, self.hidden_width),
                torch.nn.Tanh(),
                torch.nn.Linear(self.hidden_width, self.latent_width),
            )
            self.output_bias = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.float64)
            )
            self.double()

        def _inputs(self, inputs: np.ndarray | torch.Tensor) -> torch.Tensor:
            if isinstance(inputs, torch.Tensor):
                tensor = inputs.to(
                    device=self.output_bias.device,
                    dtype=torch.float64,
                )
            else:
                tensor = torch.as_tensor(
                    inputs,
                    device=self.output_bias.device,
                    dtype=torch.float64,
                )
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (
                self.sensor_points,
                1,
            ):
                raise ValueError(
                    "inputs must have shape (batch, sensor_points, 1)"
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
                tensor = coordinates.to(
                    device=self.output_bias.device,
                    dtype=torch.float64,
                )
            else:
                tensor = torch.as_tensor(
                    coordinates,
                    device=self.output_bias.device,
                    dtype=torch.float64,
                )
            if tensor.ndim != 1 or tensor.shape[0] == 0:
                raise ValueError(
                    "coordinates must be a non-empty one-dimensional array"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("coordinates must contain only finite values")
            return tensor

        def _targets(
            self,
            inputs: torch.Tensor,
            coordinates: torch.Tensor,
            targets: np.ndarray | torch.Tensor,
        ) -> torch.Tensor:
            if isinstance(targets, torch.Tensor):
                tensor = targets.to(
                    device=self.output_bias.device,
                    dtype=torch.float64,
                )
            else:
                tensor = torch.as_tensor(
                    targets,
                    device=self.output_bias.device,
                    dtype=torch.float64,
                )
            expected_shape = (
                inputs.shape[0],
                coordinates.shape[0],
                1,
            )
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"targets must have shape {expected_shape}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("targets must contain only finite values")
            return tensor

        def forward(
            self,
            inputs: np.ndarray | torch.Tensor,
            coordinates: np.ndarray | torch.Tensor,
        ) -> torch.Tensor:
            """Evaluate the nonlinear operator at query coordinates."""

            input_tensor = self._inputs(inputs)
            coordinate_tensor = self._coordinates(coordinates)
            branch = self.branch(input_tensor[..., 0])
            normalized = (coordinate_tensor / self.length).unsqueeze(-1)
            trunk = self.trunk(normalized)
            output = torch.einsum("bl,pl->bp", branch, trunk)
            return (output + self.output_bias).unsqueeze(-1)

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
            """Return mean squared error without constructing a graph."""

            with torch.no_grad():
                return float(
                    self._loss_tensor(inputs, coordinates, targets).cpu()
                )

        def fit(
            self,
            inputs: np.ndarray | torch.Tensor,
            coordinates: np.ndarray | torch.Tensor,
            targets: np.ndarray | torch.Tensor,
            epochs: int = 150,
            learning_rate: float = 1.0e-2,
        ) -> TorchDeepONetFitResult:
            """Fit branch and trunk networks with Adam and autograd."""

            if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
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
            optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
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
            return TorchDeepONetFitResult(
                initial_loss=initial_loss,
                final_loss=final_loss,
                epochs=int(epochs),
            )

else:

    class TorchDeepONet1D:
        """Placeholder that reports the optional dependency requirement."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required for TorchDeepONet1D; "
                "install requirements-autodiff.txt"
            )
