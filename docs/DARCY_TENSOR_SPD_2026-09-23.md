# Darcy full SPD tensor qualification — 23 September 2026

## Scope

This tranche extends the Darcy truth path from grid-aligned diagonal
anisotropy to a full symmetric positive-definite tensor field

```text
K = [[K_xx, K_xy],
     [K_xy, K_yy]]
```

and solves

```text
-div(K grad(u)) = f
```

on a rectangular domain with homogeneous Dirichlet pressure.

The reference discretization uses Q1 finite elements with a 2x2 Gauss rule.
Tensor components and forcing are represented at grid nodes and interpolated
bilinearly inside each element. Nodal SPD validation is sufficient to preserve
SPD under this convex interpolation.

## Deterministic tensor family

The dataset begins from the existing positive scalar log-permeability field
`k`. Principal permeabilities are

```text
lambda_1 = k * sqrt(r)
lambda_2 = k / sqrt(r)
```

and are rotated by a deterministic smooth angle field. The three stored input
channels are `K_xx`, `K_xy`, and `K_yy`.

The angle field has an explicit base angle, bounded amplitude, Fourier-mode
budget, and seed. Therefore magnitude and principal-axis orientation can vary
spatially while the requested principal-value ratio remains controlled.

## ARM64 qualification

Control branch: `research/darcy-tensor-spd`.

Qualified commit:
`7c918dcde8e43f90130f25221c3ecf2f476cf135`.

RemoteOps run:
https://github.com/Memorithm/RemoteOps/actions/runs/35818381772

Runner: `thor-remoteops-arm64-01`, host `tarek`, architecture `aarch64`.

All **5/5** tensor-reference tests passed.

### Rotated constant tensor convergence

For

```text
K = [[2.0, 0.6],
     [0.6, 1.0]]
```

and the analytic solution
`u = sin(pi*x) sin(pi*y)`, the measured relative L2 errors were:

| points | relative L2 error | observed order | max algebraic residual |
| ---: | ---: | ---: | ---: |
| 9 | 1.309422741703436e-2 | — | 8.326672684688674e-16 |
| 17 | 3.2991146490415567e-3 | 1.9887801209560605 | 1.700029006457271e-15 |
| 33 | 8.263844849968481e-4 | 1.9971938409203311 | 2.1371793224034263e-15 |
| 65 | 2.0669683328955815e-4 | 1.9992968824537392 | 3.0826036168107862e-15 |

This is consistent with second-order convergence for the chosen smooth test.

### Spatial tensor families

With a spatially varying orientation field and spatially varying determinant
scale, the controlled families produced:

| principal ratio | cross-term std | max eigenvalue-ratio error | min determinant | max algebraic residual |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 | 0.0 | 1.2485692546532806e-1 | 1.739060284666749e-16 |
| 4 | 4.3542341769969484e-1 | 3.1086244689504383e-15 | 1.2485692546532806e-1 | 1.227316859253591e-16 |
| 16 | 1.0885585442492374 | 4.618527782440651e-14 | 1.2485692546532806e-1 | 1.48318857196017e-16 |

The benchmark JSON SHA-256 recorded by RemoteOps is
`20b0467f61c41d9afbe064ef2bf6ecf80a9527fe69d4da8c8639e891045a7e7f`.

## Interpretation and limits

This establishes a reproducible truth oracle for rotated and spatially varying
SPD permeability tensors. It does **not** establish superiority of a learned
operator.

Current limits:

- rectangular geometry only;
- homogeneous Dirichlet boundary condition only;
- the principal-value ratio is sample-wide, although tensor magnitude and
  orientation vary spatially;
- Q1 nodal representation is the reference discretization for this tranche;
- discontinuous facies and mixed/Neumann boundary conditions remain separate
  validation gates.

The next learned-model experiment may now use the three tensor channels, but
its conclusions must remain conditional on a matched training/evaluation
protocol and the truth-oracle evidence above.
