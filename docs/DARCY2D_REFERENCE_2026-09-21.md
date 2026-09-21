# Darcy 2D reference solver — 21 September 2026

## Scope

This tranche introduces the second PDE family in NeuralOperator after Burgers.
It implements a deterministic truth solver for the steady variable-coefficient
Darcy equation

    -div(k grad u) = f

on a rectangular nodal grid with homogeneous Dirichlet pressure boundaries.

## Discretization

- permeability k must be finite and strictly positive;
- face permeability is the harmonic mean of adjacent nodal values;
- the interior operator is a conservative five-point flux stencil;
- the sparse linear system is assembled with SciPy and solved with spsolve;
- darcy_residual_2d evaluates the exact same discrete operator used by the solver.

## Dataset contract

random_log_permeability generates smooth positive fields from a seeded,
low-frequency trigonometric basis. Each sample is normalized in log space before
the requested log standard deviation is applied. generate_darcy_dataset stores
permeability inputs, pressure targets, forcing, geometry and generation
parameters.

The generator is a controlled research distribution. It is not presented as a
replacement for external Darcy benchmark datasets.

## Acceptance checks

tests/test_darcy.py verifies:

1. convergence against u(x,y)=sin(pi x) sin(pi y) for constant permeability;
2. an observed refinement ratio consistent with second-order spatial error;
3. discrete residuals near floating-point solver tolerance;
4. deterministic seeded dataset generation and positive permeability;
5. exact homogeneous boundary values and invalid-input rejection.

scripts/darcy_benchmark.py emits the analytic convergence sequence and dataset
residual diagnostics as JSON.

## Limitations

This slice does not yet provide a trainable 2D neural operator, irregular
meshes, non-homogeneous boundary conditions, anisotropic tensor permeability or
a performance comparison with a classical production solver. No speedup or
resolution-invariance claim follows from this reference solver.

## Next gate

Train a controlled 2D operator on the same permeability-to-pressure pairs,
retain this sparse solver as the acceptance oracle, and compare error, PDE
residual, OOD coefficient contrast, resolution transfer, latency and memory.
