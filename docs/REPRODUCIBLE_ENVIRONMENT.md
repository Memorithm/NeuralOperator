# Reproducible benchmark environments

The optional autodiff backend and the ordinary test workflow use checked-in,
hash-verified lock files for the GitHub-hosted Linux/Python 3.12 campaign:

- `requirements-py312-linux.lock` covers the NumPy/SciPy reference path.
- `requirements-autodiff-py312-linux.lock` covers the reference path plus the
  PyTorch autodiff backend.

The lock files were generated with `uv pip compile --generate-hashes` for
Python 3.12 on `x86_64-manylinux_2_17`. CI installs with pip
`--require-hashes`; it does not upgrade or re-resolve the declared graph.
Workflow actions are pinned to immutable commit SHAs. The setup-python step
selects Python 3.12.8, while the lock files remain valid for the Python 3.12
minor line.

Each workflow publishes `neural-operator-reproducibility` or
`neural-operator-benchmarks` evidence containing a manifest with the source
and checkout SHAs, lockfile SHA-256, Python/platform identity, and explicit
dataset/checkpoint digest fields. When a benchmark supplies
`DATASET_SHA256` or `CHECKPOINT_SHA256`, those values are recorded; otherwise
the manifest says `not-provided` and makes no provenance claim.

The path filters were removed from the optional workflow so a dependency,
benchmark, test, or lockfile change cannot bypass its qualification job.
This records reproducibility metadata; it does not turn benchmark success into
an accuracy, generalization, hardware, or promotion claim.
