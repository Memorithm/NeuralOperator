# Differentiable Darcy PINO path — 21 September 2026

## Scope

This tranche turns the existing conservative Darcy truth stencil into a
differentiable PyTorch residual. The purpose is not to replace the sparse truth
solver. The SciPy solver remains the acceptance oracle; the PyTorch form is a
training constraint that must match the same discrete equation.

## Residual

torch_darcy_residual_2d evaluates

    -div(k grad u) - f

on the interior grid with the same harmonic face permeability and rectangular
spacing convention as darcy_residual_2d. The operation remains inside the
autograd graph with respect to predicted pressure.

The FNO architecture already imposes homogeneous Dirichlet pressure boundaries
exactly through its output envelope, so the physics loss does not need a
separate soft boundary penalty for this benchmark.

## Loss contract

torch_darcy_data_physics_loss returns three terms separately:

- total weighted loss;
- pressure data MSE;
- conservative Darcy residual MSE.

fit_torch_fno_darcy_physics reports initial and final values of all three terms.
This keeps a lower total loss from hiding a worse data or physics component.

## Weight sensitivity benchmark

scripts/torch_darcy_pino_experiment.py trains identical FNO 2D initializations
with physics weights

    0, 1e-5, 1e-4, 3e-4

using the same training, validation and OOD coefficient cohorts. A zero physics
weight goes through the same fitting function and therefore acts as the
data-only control.

The report includes validation/OOD data error, discrete physics residual,
training wall time and all optimization terms. The sweep is sensitivity
evidence, not automatic hyperparameter selection.

## Acceptance checks

tests/test_torch_darcy_physics.py verifies:

1. numerical parity between the PyTorch residual and the NumPy/SciPy stencil;
2. finite non-zero gradients through predicted pressure;
3. exact separation of data and physics losses on a truth solution;
4. simultaneous reduction of total, data and physics training terms;
5. rejection of non-positive permeability.

## Limits

The raw residual scale depends on grid spacing, permeability contrast and the
forcing convention. A physics weight that is useful at 9x9 is not assumed to
transfer to another resolution. Multi-resolution normalization or adaptive
weighting must be measured rather than inferred.

The next gate is multi-seed and sample-scaling evidence around the promising
weight range, followed by additional coefficient families and
advection-diffusion.
