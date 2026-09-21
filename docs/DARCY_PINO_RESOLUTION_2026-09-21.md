# Darcy PINO resolution sensitivity — 21 September 2026

## Question

The Darcy PINO loss combines pressure data error with a conservative discrete
PDE residual. The residual uses grid-dependent finite-difference coefficients,
so a physics weight that is useful on one grid must not be assumed to transfer
unchanged to another resolution.

This tranche measures that sensitivity directly.

## Protocol

scripts/torch_darcy_pino_resolution.py evaluates:

- grids: 9x9 and 17x17;
- physics weights: 0, 1e-5, 1e-4, 3e-4;
- paired initialization/training seeds: 2 and 4;
- 12 training samples and 6 held-out samples per resolution;
- the same FNO 2D architecture and Fourier-mode budget at both resolutions;
- Adam for 80 epochs at learning rate 1e-2.

For a given resolution and seed, every physics-weight run starts from the same
model initialization. This makes the weight sweep paired rather than mixing
weight changes with initialization noise.

## Measurements

Every run records:

- final weighted training loss;
- final pressure data loss;
- final conservative physics loss;
- held-out relative L2 pressure error;
- held-out Darcy residual MSE;
- training wall-clock time.

For each grid and weight, the report aggregates the two replicates with mean,
sample standard deviation, minimum and maximum.

## Interpretation

The study does not select a global physics weight. It answers a narrower
question: whether the observed data/physics trade-off changes materially when
the spatial resolution changes while the model architecture is held fixed.

The coefficient generator uses the same random seeds across resolutions, but
its per-sample normalization is evaluated on each discrete grid. The paired
datasets therefore represent closely related generated families, not bitwise
samples of an identical continuous field.

## Limits

Two resolutions and two replicates are sensitivity evidence, not a convergence
theorem. Further work should include finer grids, forcing-scale changes,
anisotropic permeability, and external Darcy datasets before defining any
adaptive physics-weight rule.
