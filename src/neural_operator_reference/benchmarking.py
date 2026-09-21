"""Shared measurement utilities for controlled neural-operator comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from .darcy import darcy_residual_2d
from .darcy_datasets import DarcyDataset
from .datasets import BurgersDataset
from .physics import burgers_transition_physics_loss
from .spectral import relative_l2_error


ArrayOperator = Callable[[np.ndarray], object]


@dataclass(frozen=True)
class ScalarSummary:
    """Descriptive statistics for repeated scalar measurements."""

    count: int
    mean: float
    sample_std: float
    minimum: float
    maximum: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "sample_std": self.sample_std,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def summarize_scalars(values) -> ScalarSummary:
    """Return finite scalar mean, sample deviation and range."""

    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("values must contain at least one scalar")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite scalars")
    sample_std = 0.0 if array.size == 1 else float(np.std(array, ddof=1))
    return ScalarSummary(
        count=int(array.size),
        mean=float(np.mean(array)),
        sample_std=sample_std,
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
    )


@dataclass(frozen=True)
class InferenceTiming:
    """Wall-clock timing for repeated operator evaluations."""

    repeats: int
    total_seconds: float
    seconds_per_call: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "repeats": self.repeats,
            "total_seconds": self.total_seconds,
            "seconds_per_call": self.seconds_per_call,
        }


@dataclass(frozen=True)
class BurgersOneStepMetrics:
    """One-transition accuracy, physics and inference-cost measurements."""

    mse: float
    relative_l2: float
    physics_mse: float
    timing: InferenceTiming

    def as_dict(self) -> dict[str, object]:
        return {
            "mse": self.mse,
            "relative_l2": self.relative_l2,
            "physics_mse": self.physics_mse,
            "timing": self.timing.as_dict(),
        }


@dataclass(frozen=True)
class Darcy2DMetrics:
    """Accuracy, discrete-physics and inference metrics for Darcy fields."""

    mse: float
    relative_l2: float
    physics_mse: float
    boundary_max_abs: float
    timing: InferenceTiming

    def as_dict(self) -> dict[str, object]:
        return {
            "mse": self.mse,
            "relative_l2": self.relative_l2,
            "physics_mse": self.physics_mse,
            "boundary_max_abs": self.boundary_max_abs,
            "timing": self.timing.as_dict(),
        }


def _as_numpy_output(value: object) -> np.ndarray:
    """Convert NumPy- or tensor-like model output without retaining gradients."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    output = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(output)):
        raise ValueError("operator output must contain only finite values")
    return output


def trainable_parameter_count(model: object) -> int:
    """Count trainable scalar parameters of a PyTorch-like model."""

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise TypeError("model must expose a callable parameters() method")
    count = 0
    for parameter in parameters():
        if not bool(getattr(parameter, "requires_grad", True)):
            continue
        numel = getattr(parameter, "numel", None)
        if callable(numel):
            count += int(numel())
        else:
            count += int(np.asarray(parameter).size)
    return count


def benchmark_inference(
    operator: ArrayOperator,
    inputs: np.ndarray,
    repeats: int = 20,
) -> tuple[np.ndarray, InferenceTiming]:
    """Warm up and time repeated inference under the array-operator contract."""

    if not callable(operator):
        raise TypeError("operator must be callable")
    if not isinstance(repeats, (int, np.integer)) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim < 3 or inputs.shape[-1] != 1:
        raise ValueError(
            "inputs must have shape (batch, *spatial_axes, 1)"
        )
    if not np.all(np.isfinite(inputs)):
        raise ValueError("inputs must contain only finite values")

    prediction = _as_numpy_output(operator(inputs))
    if prediction.shape != inputs.shape:
        raise ValueError(
            "operator must preserve the input batch/spatial/channel shape; "
            f"got {prediction.shape} for {inputs.shape}"
        )

    start = perf_counter()
    for _ in range(int(repeats)):
        prediction = _as_numpy_output(operator(inputs))
        if prediction.shape != inputs.shape:
            raise ValueError(
                "operator must preserve the input shape during timing"
            )
    total = perf_counter() - start
    timing = InferenceTiming(
        repeats=int(repeats),
        total_seconds=float(total),
        seconds_per_call=float(total / int(repeats)),
    )
    return prediction, timing


def evaluate_burgers_dataset(
    operator: ArrayOperator,
    dataset: BurgersDataset,
    repeats: int = 20,
) -> BurgersOneStepMetrics:
    """Evaluate one Burgers transition using shared accuracy and physics metrics."""

    if not isinstance(dataset, BurgersDataset):
        raise TypeError("dataset must be a BurgersDataset")
    prediction, timing = benchmark_inference(operator, dataset.inputs, repeats=repeats)
    error = prediction - dataset.targets
    return BurgersOneStepMetrics(
        mse=float(np.mean(error**2)),
        relative_l2=float(relative_l2_error(prediction, dataset.targets)),
        physics_mse=float(
            burgers_transition_physics_loss(
                dataset.inputs[..., 0],
                prediction[..., 0],
                horizon=dataset.horizon,
                viscosity=dataset.viscosity,
                length=dataset.length,
                forcing=dataset.forcing,
            )
        ),
        timing=timing,
    )



def evaluate_darcy_dataset(
    operator: ArrayOperator,
    dataset: DarcyDataset,
    repeats: int = 20,
) -> Darcy2DMetrics:
    """Evaluate Darcy pressure prediction against the sparse truth solver."""

    if not isinstance(dataset, DarcyDataset):
        raise TypeError("dataset must be a DarcyDataset")
    prediction, timing = benchmark_inference(
        operator,
        dataset.inputs,
        repeats=repeats,
    )
    error = prediction - dataset.targets
    residual_squares: list[np.ndarray] = []
    for sample in range(dataset.samples):
        residual = darcy_residual_2d(
            prediction[sample, ..., 0],
            dataset.inputs[sample, ..., 0],
            forcing=dataset.forcing,
            length_x=dataset.length_x,
            length_y=dataset.length_y,
        )
        residual_squares.append(residual * residual)

    pressure = prediction[..., 0]
    boundary_max_abs = max(
        float(np.max(np.abs(pressure[:, 0, :]))),
        float(np.max(np.abs(pressure[:, -1, :]))),
        float(np.max(np.abs(pressure[:, :, 0]))),
        float(np.max(np.abs(pressure[:, :, -1]))),
    )
    return Darcy2DMetrics(
        mse=float(np.mean(error**2)),
        relative_l2=float(relative_l2_error(prediction, dataset.targets)),
        physics_mse=float(
            np.mean(np.concatenate([values.ravel() for values in residual_squares]))
        ),
        boundary_max_abs=boundary_max_abs,
        timing=timing,
    )
