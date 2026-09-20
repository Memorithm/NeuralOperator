# Optional autodiff backend — 20 September 2026

The repository now contains a small PyTorch implementation, `TorchFNO1D`,
alongside the dependency-light `NumpyFNO1D` reference.

## Scope

- Both implementations use the tensor contract `(batch, points, channels)`.
- Both keep a single real-valued spectral mixing block and preserve the input
  resolution at the output.
- `TorchFNO1D.fit` uses PyTorch automatic differentiation and Adam on a full
  batch. The NumPy/SciPy optimizer remains the transparent oracle and is not
  replaced by this optional path.
- Input, target shape, and finite-value checks are enforced in both backends.

This is an implementation gate, not a performance claim. Numerical parity,
multi-seed comparison, recursive rollout quality, and GPU behavior still need
to be measured before choosing a production backend.

## Installation and smoke test

The base requirements stay unchanged. On a Debian 12 machine, install the
optional dependency in the active environment with:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-autodiff.txt
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/torch_fno_experiment.py
```

When PyTorch is absent, the optional test module is skipped and the base
NumPy/SciPy suite remains runnable. The experiment script exits with an
installation hint rather than silently falling back to the finite-difference
optimizer.

## Next validation gate

Run the PyTorch and NumPy implementations on the same deterministic train,
validation, and out-of-distribution Burgers split; report fit loss, one-step
MSE/relative L2, and recursive rollout metrics with at least three seeds. Only
after that comparison should the autodiff path be used for the nonlinear
DeepONet and the larger PDE families in the roadmap.
