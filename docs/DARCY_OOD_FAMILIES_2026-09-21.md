# Darcy coefficient-family OOD study — 21 September 2026

## Motivation

The previous Darcy experiments changed coefficient contrast but retained the
same low-frequency generator family. That is insufficient to characterize
operator robustness when the spatial statistics of permeability change.

This tranche therefore parameterizes the generator spectrum and evaluates
several fixed out-of-distribution coefficient families.

## Generator extension

random_log_permeability and generate_darcy_dataset now accept
spectral_decay. The amplitude assigned to a non-zero Fourier mode with squared
frequency q follows q^(-spectral_decay/2).

The historical generator corresponds exactly to spectral_decay=2.0. That code
path retains the original arithmetic expression so existing deterministic data
remain bitwise unchanged.

Lower spectral decay gives more weight to high spatial frequencies; larger
decay produces smoother coefficient fields. The selected value is stored in
DarcyDataset provenance.

## Evaluation families

scripts/torch_darcy_ood_families.py trains the matched-budget FNO 2D,
DeepONet 2D and local-convolution references on the historical family, then
evaluates fixed eight-sample cohorts:

- IID: modes=2, spectral_decay=2.0, log_std=0.45;
- contrast: log_std=0.85;
- rough/high-frequency: modes=3, spectral_decay=0.5;
- smooth spectrum: spectral_decay=4.0;
- higher mean permeability: mean log permeability +0.7;
- lower mean permeability: mean log permeability -0.7.

Three paired model/training seeds are used. Each family cohort is identical
across the three architecture replicates.

## Metrics

The report retains every raw run and aggregates, per architecture and family:

- relative L2 pressure error;
- conservative Darcy residual MSE;
- hard-boundary maximum absolute violation;
- training wall-clock time.

Mean, sample standard deviation, minimum and maximum use the common benchmark
summary helper introduced by the multi-seed tranche.

## Acceptance checks

tests/test_darcy_ood_families.py verifies that:

1. the default generator is bitwise identical to explicit spectral_decay=2.0;
2. lower spectral decay gives a measurably rougher deterministic log-field;
3. spectral family metadata are preserved in the dataset;
4. invalid negative decay values are rejected.

## Limits

The spectral generator is still synthetic and isotropic. The experiment does
not yet test anisotropic permeability tensors, discontinuous facies,
non-rectangular geometry, Neumann/mixed boundaries or external benchmark
datasets.

The next gate is resolution-aware normalization of the physics loss, then
anisotropic/geometric OOD evidence before advection-diffusion.
