# Matched-budget Darcy 2D comparison — 21 September 2026

## Question

The first Darcy FNO 2D can fit the controlled elliptic problem, but that alone
does not show that spectral mixing is useful. This tranche therefore compares
three trainable operators under a deliberately narrow, reproducible protocol.

## Architectures

The reference configurations are chosen to keep trainable parameter counts
within three percent:

- FNO 2D: 8,449 parameters;
- nonlinear DeepONet 2D: 8,649 parameters;
- local coordinate-aware convolution: 8,672 parameters.

All three models use float64, receive the same permeability samples, and impose
the same homogeneous Dirichlet pressure condition exactly through a hard output
envelope.

DeepONet uses the full 9x9 permeability grid as fixed branch sensors and the
normalized x/y coordinates as trunk queries. The local baseline has two 3x3
hidden convolution stages, so its receptive field remains finite rather than
global.

## Training protocol

scripts/torch_darcy_operator_comparison.py uses:

- 16 deterministic 9x9 training samples;
- 4 held-out IID samples;
- 4 OOD samples with larger log-permeability standard deviation;
- Adam for exactly 240 epochs at learning rate 1e-2 for every architecture.

Equal optimizer steps are not equal compute. The report therefore records
training wall time separately from prediction error and inference timing.

## Metrics

Each model is evaluated with the same shared Darcy metrics:

- mean squared error;
- global relative L2 error;
- conservative discrete PDE residual MSE;
- maximum boundary violation;
- inference wall-clock timing.

The script also records the exact parameter-budget spread.

## Interpretation limits

This is one small synthetic coefficient family, one grid and one elliptic PDE.
A result on this benchmark must not be generalized into an architecture ranking
for neural operators as a whole. Differences can arise from optimization,
normalization, receptive field, sensor representation and sample count.

The next scientific gate is to repeat the comparison across multiple seeds and
dataset sizes, then add a differentiable Darcy residual to compare data-only
training with physics-informed operator learning.
