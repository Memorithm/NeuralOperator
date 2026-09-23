# Darcy diagonal anisotropy qualification — 23 September 2026

## Scope

This tranche extends the scalar Darcy truth solver to the diagonal,
grid-aligned symmetric positive tensor

```text
K = diag(k_x, k_y)
```

and solves

```text
-d_x(k_x d_x u) - d_y(k_y d_y u) = f
```

with homogeneous Dirichlet pressure. Face coefficients use harmonic means,
matching the conservative scalar reference discretization.

The deterministic dataset begins from a positive scalar log-permeability field
`k` and uses

```text
k_x = k * sqrt(r)
k_y = k / sqrt(r)
```

so the requested pointwise anisotropy ratio is `k_x/k_y = r` while
`sqrt(k_x*k_y) = k`.

## Reproducible checks

The versioned test module verifies:

- exact isotropic reduction to the existing scalar Darcy solver and residual;
- second-order convergence against an analytic constant-anisotropy solution;
- deterministic two-channel dataset generation;
- exact preservation of the requested anisotropy ratio;
- small conservative discrete residuals;
- rejection of non-positive coefficients, incompatible shapes and invalid ratios.

Run locally with:

```bash
PYTHONPATH=src python3 -m unittest tests.test_darcy_anisotropic -v
PYTHONPATH=src python3 scripts/darcy_anisotropic_benchmark.py
```

## ARM64 qualification

The self-hosted ARM64 run executed commit
`7e1451a3f22026e31e604d0f26752e49b722b1b3`.

Tests: **5/5 passed**.

Isotropic reduction:

```text
max_abs_solution_difference = 0.0
```

Analytic convergence for `k_x=2`, `k_y=0.5`:

| points | relative L2 error | observed order |
| ---: | ---: | ---: |
| 9 | 1.2950746721879309e-2 | — |
| 17 | 3.218964440080193e-3 | 2.008366739525912 |
| 33 | 8.035776793720459e-4 | 2.00208724281546 |
| 65 | 2.0082180969406666e-4 | 2.000521533835277 |

Controlled ratio families:

| k_x/k_y | max ratio error | max abs discrete residual | pressure RMS |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.0 | 3.352873534367973e-14 | 0.03340734820715356 |
| 1 | 0.0 | 3.863576125695545e-14 | 0.03895093863826968 |
| 4 | 0.0 | 4.063416270128073e-14 | 0.031222566349966847 |
| 16 | 0.0 | 4.218847493575595e-14 | 0.01909289637089592 |

## Interpretation and limits

This establishes numerical correctness for the current **diagonal,
grid-aligned** anisotropic truth family. It does not establish performance of a
neural operator on anisotropic data.

The current dataset also keeps one anisotropy ratio constant over each sample.
The next scientifically distinct extensions are spatially varying tensor
anisotropy and rotated principal axes, followed by geometry/boundary-condition
shifts and discontinuous facies. Those cases need their own truth-oracle tests
before learned-model comparisons are interpreted.
