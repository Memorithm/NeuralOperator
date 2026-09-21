# Exact Darcy global-scale equivariance — 21 September 2026

## Motivation

The coefficient-family OOD benchmark showed that shifting the global mean
permeability is substantially harder than changing spectral roughness on the
current training distribution.

For scalar Darcy flow this particular shift has an exact algebraic structure
that should not be left for a neural network to rediscover.

## Exact identity

For fixed forcing f and homogeneous Dirichlet pressure,

    -div(k grad u) = f.

Write a positive scalar permeability field as

    k = g * k_hat,

where g is a sample-wide positive constant. Defining

    u_hat = g * u

gives

    -div(k_hat grad u_hat) = f.

Therefore multiplying all permeability values by c divides the physical
pressure by c. This covariance is exact for the discrete reference solver as
well because every face permeability and matrix coefficient scales linearly
with c.

## Transform

darcy_scaling.py uses the per-sample geometric mean

    g = exp(mean(log k))

as the global scale. It exposes:

- darcy_geometric_mean_scale;
- normalize_darcy_permeability_scale;
- normalize_darcy_pressure_scale;
- restore_darcy_pressure_scale;
- normalize_darcy_dataset_scale.

The normalized permeability has geometric mean one. The corresponding training
target is u_hat=g*u. At inference, normalized pressure is divided by g to
recover physical pressure.

The transform has no trainable parameters and is external to the neural
architecture, so the same rule can be used by FNO, DeepONet, local convolution
and future Rust backends.

## Benchmark

scripts/torch_darcy_scale_equivariance.py compares raw and scale-equivariant
training for all three matched-budget architectures over three paired seeds.

The evaluation families include:

- IID mean log permeability 0;
- mean shifts +0.7 and -0.7;
- stronger mean shifts +1.2 and -1.2.

Raw and normalized models use identical architecture seeds and optimizer
settings. Results include relative pressure error, discrete Darcy residual and
training time.

## Acceptance checks

tests/test_darcy_scale_equivariance.py proves that:

1. normalized permeability has unit geometric mean;
2. scaling k by exp(0.7) produces pressure u/exp(0.7) in the sparse solver;
3. normalized coefficient and normalized pressure are invariant to that global
   scale change within numerical tolerance;
4. normalization/restoration round-trips the pressure field;
5. non-positive permeability is rejected.

## Limits

This exact symmetry removes only a sample-wide scalar factor. It does not solve
contrast changes, anisotropy, local discontinuities, geometry shifts or mixed
boundary conditions. Those remain empirical OOD problems.
