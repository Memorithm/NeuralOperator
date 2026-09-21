"""Numerical reference components for the Neural Operator research program."""

from .spectral import (
    burgers_residual,
    central_difference_periodic,
    dealias_mask,
    dealiased_product,
    divergence_2d,
    incompressible_velocity_from_streamfunction,
    relative_l2_error,
    spectral_derivative,
    spectral_filter,
    wave_numbers,
)
from .linear_operator import FourierMultiplier1D
from .fno1d import FNOFitResult, NumpyFNO1D
from .burgers import (
    burgers_rhs,
    burgers_rk4_step,
    integrate_burgers,
    periodic_energy,
    periodic_mean,
)
from .datasets import (
    BurgersDataset,
    generate_burgers_dataset,
    periodic_forcing_field,
    random_periodic_fields,
)
from .splits import (
    BurgersDatasetSpec,
    BurgersDatasetSplit,
    generate_burgers_split,
)
from .physics import (
    burgers_data_physics_loss,
    burgers_transition_physics_loss,
    burgers_transition_residual,
)
from .training import PhysicsFitResult, fit_fno_burgers_physics
from .baselines import (
    ConvolutionFitResult,
    PeriodicConv1D,
    periodic_linear_interpolate,
)
from .deeponet import DeepONetFitResult, LinearDeepONet1D, fourier_trunk_features
from .evaluation import (
    BurgersRolloutMetrics,
    evaluate_burgers_operator,
    evaluate_burgers_rollout,
    rollout_burgers_operator,
)
from .torch_fno1d import TorchFNO1D, TorchFNOFitResult
from .torch_deeponet import TorchDeepONet1D, TorchDeepONetFitResult
from .benchmarking import (
    BurgersOneStepMetrics,
    InferenceTiming,
    benchmark_inference,
    evaluate_burgers_dataset,
    trainable_parameter_count,
)

__all__ = [
    "burgers_residual",
    "central_difference_periodic",
    "dealias_mask",
    "dealiased_product",
    "divergence_2d",
    "incompressible_velocity_from_streamfunction",
    "relative_l2_error",
    "spectral_derivative",
    "spectral_filter",
    "wave_numbers",
    "FourierMultiplier1D",
    "FNOFitResult",
    "NumpyFNO1D",
    "burgers_rhs",
    "burgers_rk4_step",
    "integrate_burgers",
    "periodic_energy",
    "periodic_mean",
    "BurgersDataset",
    "generate_burgers_dataset",
    "random_periodic_fields",
    "periodic_forcing_field",
    "BurgersDatasetSpec",
    "BurgersDatasetSplit",
    "generate_burgers_split",
    "burgers_data_physics_loss",
    "burgers_transition_physics_loss",
    "burgers_transition_residual",
    "PhysicsFitResult",
    "fit_fno_burgers_physics",
    "periodic_linear_interpolate",
    "ConvolutionFitResult",
    "PeriodicConv1D",
    "DeepONetFitResult",
    "LinearDeepONet1D",
    "fourier_trunk_features",
    "BurgersRolloutMetrics",
    "evaluate_burgers_operator",
    "evaluate_burgers_rollout",
    "rollout_burgers_operator",
    "TorchFNO1D",
    "TorchFNOFitResult",
    "TorchDeepONet1D",
    "TorchDeepONetFitResult",
    "BurgersOneStepMetrics",
    "InferenceTiming",
    "benchmark_inference",
    "evaluate_burgers_dataset",
    "trainable_parameter_count",
]
