# Full-SPD Darcy multi-seed replication — 23 September 2026

## Purpose

This experiment tests whether the single-seed observations from
`DARCY_TENSOR_OPERATOR_COMPARISON_2026-09-23.md` persist when both the
training draw and model initialization change.

The comparison remains deliberately descriptive. Three paired replicates are
enough to expose instability but are not a statistical significance study.

## Protocol

All architectures use the same matched parameter budget as the single-seed
baseline:

| model | trainable parameters |
| --- | ---: |
| DeepONet 2D | 8,379 |
| FNO 2D | 8,465 |
| local convolution 2D | 8,625 |

Replicate seeds: `2, 4, 6`.

Within each replicate, the three architectures share the same training dataset.
Training dataset seeds are `10002, 10004, 10006`.

Training family:

- 16 samples;
- 9x9 grid;
- principal ratio 4;
- orientation base 0.2 rad;
- orientation amplitude 0.45 rad;
- orientation modes 2;
- Adam, 180 epochs, learning rate 1e-2.

The evaluation cohorts are fixed across every model and every replicate:

- IID: 8 samples, seed 9101, training-family parameters;
- orientation OOD: 8 samples, seed 9201, ratio 4, base 1.0 rad,
  amplitude 0.70 rad, 3 orientation modes;
- principal-ratio OOD: 8 samples, seed 9301, ratio 16, original orientation
  family.

## Qualification evidence

Qualified code commit:
`66af393a8735d23f8c57daec6ad1b592e5eef09f`.

Standard GitHub tests:
https://github.com/Memorithm/NeuralOperator/actions/runs/35819768370

Thor control-plane run:
https://github.com/Memorithm/RemoteOps/actions/runs/35819805973

Thor environment:

- NumPy 2.5.3;
- SciPy 1.18.1;
- PyTorch 2.14.0+cu130;
- CUDA available on NVIDIA Thor;
- prerequisite tensor/PyTorch tests: 6/6 passed.

Multi-seed JSON SHA-256:
`51628b65fba2dc9f7b76a5fa6528784a1df6ca7a6fd97ccfad76c467d63d2fe6`.

## Relative L2 error

Values are mean ± sample standard deviation over three paired replicates.

| model | IID | OOD orientation | OOD ratio 16 |
| --- | ---: | ---: | ---: |
| FNO 2D | 0.14116 ± 0.00244 | 0.54257 ± 0.14927 | 0.64622 ± 0.08158 |
| DeepONet 2D | 0.20319 ± 0.00986 | 0.22774 ± 0.00821 | 0.75631 ± 0.01609 |
| local convolution 2D | 0.22406 ± 0.03969 | 0.71692 ± 0.24411 | 1.10353 ± 0.25853 |

Ranges:

| model | IID min–max | OOD orientation min–max | OOD ratio 16 min–max |
| --- | ---: | ---: | ---: |
| FNO 2D | 0.13864–0.14350 | 0.39429–0.69281 | 0.55277–0.70320 |
| DeepONet 2D | 0.19212–0.21100 | 0.21844–0.23398 | 0.74448–0.77463 |
| local convolution 2D | 0.17829–0.24890 | 0.43809–0.89217 | 0.90123–1.39480 |

## Q1 algebraic residual MSE

| model | IID mean ± sd | OOD orientation mean ± sd | OOD ratio 16 mean ± sd |
| --- | ---: | ---: | ---: |
| FNO 2D | 2.2945e-4 ± 7.7247e-5 | 1.2917e-3 ± 5.4505e-4 | 5.2751e-3 ± 4.9524e-3 |
| DeepONet 2D | 1.2404e-4 ± 1.1418e-5 | 1.2796e-4 ± 2.0262e-5 | 6.5943e-4 ± 3.2452e-5 |
| local convolution 2D | 9.9782e-4 ± 4.0664e-4 | 4.2648e-3 ± 3.4105e-3 | 2.0622e-2 ± 5.1590e-3 |

## Within-replicate OOD/IID relative-L2 ratios

This ratio is descriptive only; it is not treated as a statistical effect size.

| model | orientation shift | principal-ratio shift |
| --- | ---: | ---: |
| FNO 2D | 3.851 ± 1.106 | 4.573 ± 0.508 |
| DeepONet 2D | 1.121 ± 0.024 | 3.726 ± 0.131 |
| local convolution 2D | 3.130 ± 0.595 | 4.941 ± 0.814 |

## Interpretation

The single-seed observations were not all accidents:

- FNO retains the lowest mean IID relative-L2 error in this protocol and shows
  very low IID dispersion.
- DeepONet is consistently less affected by the orientation-family shift:
  its orientation-OOD relative-L2 distribution remains close to its IID
  distribution across all three paired replicates.
- FNO and the local convolution baseline show substantially larger orientation
  degradation and larger OOD dispersion.
- the principal-ratio shift from 4 to 16 degrades every architecture. The mean
  within-replicate OOD/IID ratio is above 3.7 for all three.
- DeepONet has the lowest Q1 algebraic residual MSE in all three reported
  cohorts under this protocol, while FNO has lower relative-L2 error on IID
  and on the ratio-16 cohort. Data error and discrete-physics error therefore
  remain distinct diagnostics.

These statements characterize this fixed 9x9 smooth-tensor experiment only.
They are not a universal architecture ranking.

## Next falsifiable gate

The common failure under principal-ratio shift justifies testing tensor-aware
representations or normalization, but changes should remain controlled.

The first candidate should preserve exact Darcy covariance under global tensor
scaling rather than add trainable complexity. A separate experiment can then
test whether representing an SPD tensor through rotation-aware invariants
improves ratio/orientation OOD behavior. Both must be compared against the raw
three-channel baseline above with the same paired seeds and fixed cohorts.

Non-rectangular geometry, mixed/Neumann boundaries and discontinuous facies
remain subsequent truth-oracle gates.
