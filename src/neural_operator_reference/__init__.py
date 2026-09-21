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
from .darcy import darcy_residual_2d, solve_darcy_2d
from .darcy_datasets import (
    DarcyDataset,
    generate_darcy_dataset,
    random_log_permeability,
)
from .darcy_scaling import (
    darcy_geometric_mean_scale,
    normalize_darcy_dataset_scale,
    normalize_darcy_permeability_scale,
    normalize_darcy_pressure_scale,
    restore_darcy_pressure_scale,
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
from .torch_fno2d import TorchFNO2D, TorchFNO2DFitResult
from .torch_localconv2d import TorchLocalConv2D, TorchLocalConv2DFitResult
from .torch_deeponet import TorchDeepONet1D, TorchDeepONetFitResult
from .torch_deeponet2d import (
    TorchDeepONet2D,
    TorchDeepONet2DFitResult,
    normalized_query_grid,
)
from .benchmarking import (
    BurgersOneStepMetrics,
    Darcy2DMetrics,
    ScalarSummary,
    InferenceTiming,
    benchmark_inference,
    evaluate_burgers_dataset,
    evaluate_darcy_dataset,
    summarize_scalars,
    trainable_parameter_count,
)
from .torch_physics import (
    TorchPhysicsFitResult,
    fit_torch_fno_burgers_physics,
    torch_burgers_data_physics_loss,
    torch_burgers_rhs,
    torch_burgers_transition_residual,
    torch_dealiased_product,
    torch_spectral_derivative,
)
from .torch_darcy_physics import (
    fit_torch_fno_darcy_physics,
    torch_darcy_data_physics_loss,
    torch_darcy_residual_2d,
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
    "darcy_residual_2d",
    "solve_darcy_2d",
    "DarcyDataset",
    "generate_darcy_dataset",
    "random_log_permeability",
    "darcy_geometric_mean_scale",
    "normalize_darcy_dataset_scale",
    "normalize_darcy_permeability_scale",
    "normalize_darcy_pressure_scale",
    "restore_darcy_pressure_scale",
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
    "TorchFNO2D",
    "TorchFNO2DFitResult",
    "TorchLocalConv2D",
    "TorchLocalConv2DFitResult",
    "TorchDeepONet1D",
    "TorchDeepONetFitResult",
    "TorchDeepONet2D",
    "TorchDeepONet2DFitResult",
    "normalized_query_grid",
    "BurgersOneStepMetrics",
    "Darcy2DMetrics",
    "ScalarSummary",
    "InferenceTiming",
    "benchmark_inference",
    "evaluate_burgers_dataset",
    "evaluate_darcy_dataset",
    "summarize_scalars",
    "trainable_parameter_count",
    "TorchPhysicsFitResult",
    "fit_torch_fno_burgers_physics",
    "torch_burgers_data_physics_loss",
    "torch_burgers_rhs",
    "torch_burgers_transition_residual",
    "torch_dealiased_product",
    "torch_spectral_derivative",
    "fit_torch_fno_darcy_physics",
    "torch_darcy_data_physics_loss",
    "torch_darcy_residual_2d",
]
