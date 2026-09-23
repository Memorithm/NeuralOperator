# Full-SPD Darcy learned-operator comparison — 23 September 2026

## Question

Can the existing 2D learned-operator references consume the three physical tensor
channels `K_xx`, `K_xy`, and `K_yy` without architecture-specific tensor
preprocessing, and how do matched-size models behave under controlled tensor OOD
shifts?

This tranche compares the existing FNO 2D, DeepONet 2D, and local convolution
baseline. It does not introduce a new learned architecture.

## Protocol

Truth data come from the Q1 full-SPD Darcy oracle documented in
`DARCY_TENSOR_SPD_2026-09-23.md`.

Training:

- grid: 9 x 9;
- samples: 16;
- principal permeability ratio: 4;
- orientation base: 0.2 rad;
- orientation amplitude: 0.45 rad;
- orientation modes: 2;
- Adam: 180 epochs, learning rate 1e-2.

Evaluation cohorts contain four samples each:

- IID validation: same family, independent seed 101;
- OOD orientation: ratio 4, base 1.0 rad, amplitude 0.70 rad, 3 modes;
- OOD principal ratio: ratio 16, original orientation family.

All three models impose homogeneous Dirichlet pressure exactly. The tensor
evaluation path reports data MSE, relative L2 error, boundary maximum, inference
timing and the Q1 algebraic residual MSE.

Parameter counts are deliberately matched:

| model | trainable parameters |
| --- | ---: |
| DeepONet 2D | 8,379 |
| FNO 2D | 8,465 |
| local convolution 2D | 8,625 |

Maximum/minimum parameter ratio: **1.0293591121**, below the 3% budget gate.

## First controlled execution

GitHub optional-autodiff run:
https://github.com/Memorithm/NeuralOperator/actions/runs/35818857298

Qualified code at that stage: `c6cadfce1d3fa1dc14bb7491a6c261f6413bc7a5`.

The later metric correction only changes behavior for predictions that violate
the Dirichlet boundary: boundary error is measured on the raw prediction and
the interior PDE residual is evaluated after projecting the prescribed zero
boundary. These three benchmark models already impose a zero boundary exactly,
so that correction does not change the values below. The corrected evaluator is
separately covered by general tests.

### Relative L2 error

| model | IID | OOD orientation | OOD ratio 16 |
| --- | ---: | ---: | ---: |
| FNO 2D | 0.1639966 | 0.6498026 | 0.7306504 |
| DeepONet 2D | 0.2352718 | 0.2265173 | 0.9464605 |
| local convolution 2D | 0.2383528 | 1.4752528 | 1.1339381 |

### Q1 algebraic residual MSE

| model | IID | OOD orientation | OOD ratio 16 |
| --- | ---: | ---: | ---: |
| FNO 2D | 4.0085933e-4 | 1.9618502e-3 | 1.0959820e-2 |
| DeepONet 2D | 1.3796998e-4 | 1.4030307e-4 | 7.0233113e-4 |
| local convolution 2D | 7.1069514e-4 | 9.4103172e-3 | 1.9432977e-2 |

Every reported boundary maximum is exactly zero because all three references
use their hard Dirichlet construction.

Training losses decreased for all three models:

| model | initial MSE | final MSE | training seconds on this runner |
| --- | ---: | ---: | ---: |
| FNO 2D | 3.3576331e-2 | 9.6484908e-6 | 2.0081 |
| DeepONet 2D | 3.9837742e-2 | 3.9508780e-5 | 0.2189 |
| local convolution 2D | 3.0567851e-3 | 3.2752615e-5 | 0.5426 |

Wall-clock values are runner-specific and are not portable performance claims.

## What this run says

Within this single-seed, small-data protocol:

- FNO has the lowest IID relative L2 error.
- DeepONet changes little under the orientation-family shift in this particular
  cohort, while FNO and especially the local baseline degrade more.
- all three models degrade strongly when the principal ratio moves from 4 to 16;
  this remains an unresolved generalization problem.
- the algebraic residual and relative L2 error do not order all models in the
  same way, so both diagnostics must remain visible.

These are descriptive observations for this run, not architecture rankings or
claims of universal superiority.

## Required next evidence

The next gate is multi-seed replication with fixed evaluation cohorts, reporting
mean, sample standard deviation, minimum and maximum for each architecture and
cohort. Only after that should tensor-specific normalization, equivariance or
architecture changes be tested.

Other remaining limits are unchanged: one grid resolution, rectangular domain,
homogeneous Dirichlet boundary, smooth coefficient fields and no discontinuous
facies.
