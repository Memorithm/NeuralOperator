# Darcy multi-seed and sample-scaling study — 21 September 2026

## Motivation

Single-seed operator comparisons can be dominated by initialization or by one
small training draw. This tranche repeats the matched-budget Darcy comparison
across multiple replicates and training-set sizes.

## Protocol

scripts/torch_darcy_multiseed_scaling.py evaluates:

- architectures: FNO 2D, nonlinear DeepONet 2D, local convolution;
- training sample counts: 8, 16, 32;
- replicate seeds: 2, 4, 6;
- grid: 9x9;
- optimizer: Adam, 120 epochs, learning rate 1e-2.

Each replicate changes both the generated training draw and the model
initialization. The held-out IID and OOD cohorts remain fixed across every run,
so replicate variability is not mixed with evaluation-set variability.

## Measurements

For every raw run the report stores:

- exact trainable parameter count;
- training wall-clock time;
- final training loss;
- held-out relative L2 error and Darcy residual MSE;
- OOD relative L2 error and Darcy residual MSE.

For each architecture and sample count, summarize_scalars reports:

- number of replicates;
- mean;
- sample standard deviation;
- minimum;
- maximum.

The complete raw-run records remain in the JSON output so aggregate values can
be audited.

## Interpretation

This benchmark is descriptive. Three replicates are enough to expose obvious
seed sensitivity, but not to establish statistical significance or confidence
intervals for a wider population of PDEs or coefficient distributions.

An apparent ordering is therefore reported as measured behavior under this
protocol, not as a universal ranking of FNO, DeepONet or convolutional
operators.

## Next gate

The next evidence should vary coefficient roughness/frequency families,
resolution, and physics-loss scaling. If conclusions remain stable, the program
can then move to advection-diffusion with the same evidence discipline.
