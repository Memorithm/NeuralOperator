"""Fixed synthetic parity/cost qualification; does not consume any model holdout."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import scipy

ORACLE_REVISION = "fcd34147c9623ffd8ea49dbb7894e0516fb88be0"
ORACLE_SHA256 = "c72975c02563d8c554bcbfcd419fcd1711c9f4e63491a76e5ef0e676fac39a92"


def load_frozen_oracle(source: Path) -> ModuleType:
    """Verify immutable baseline bytes before importing any oracle code."""
    if hashlib.sha256(source.read_bytes()).hexdigest() != ORACLE_SHA256:
        raise ValueError("Darcy oracle drift: explicit baseline review/repin required")
    # Load the verified file directly, not package __init__ or a replaceable
    # sys.path module which could intercept the independent reference.
    spec = importlib.util.spec_from_file_location("frozen_darcy_oracle", source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load verified Darcy oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    source = Path("src/neural_operator_reference/darcy.py").resolve()
    oracle = load_frozen_oracle(source)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    fixtures = []
    # Frozen before execution: rectangular/square grids, four independent
    # deterministic coefficient/forcing families, supplied and solved states.
    for rows, columns in [(3, 3), (5, 7), (9, 9), (17, 29)]:
        for seed in range(4):
            y, x = np.indices((rows, columns))
            k = 0.25 + ((7 * x + 11 * y + 3 * seed) % 31) / 8.0
            forcing = ((5 * x + 3 * y + seed) % 17) / 4.0 - 1.0
            lx, ly = 1.0 + seed / 4.0, 0.75 + seed / 8.0
            supplied = np.sin(x * 0.125) + np.cos(y * 0.25)
            solved = oracle.solve_darcy_2d(k, forcing, lx, ly)
            for kind, u in [("supplied", supplied), ("solved", solved)]:
                name = f"{rows}x{columns}-s{seed}-{kind}"
                reference = oracle.darcy_residual_2d(u, k, forcing, lx, ly).reshape(-1)
                fixture = (
                    f"{rows} {columns} {lx:.17e} {ly:.17e}\n"
                    + "\n".join(" ".join(f"{v:.17e}" for v in a.reshape(-1)) for a in (u, k, forcing))
                    + "\n"
                ).encode("ascii")
                run = subprocess.run(
                    [str(binary)], input=fixture, capture_output=True,
                    timeout=30, check=True,
                )
                lines = run.stdout.decode("ascii").splitlines()
                if len(lines) != 2:
                    raise AssertionError("unexpected Rust result framing")
                actual = np.fromstring(lines[0], sep=" ")
                if actual.shape != reference.shape or not np.all(np.isfinite(actual)):
                    raise AssertionError("incomplete or nonfinite Rust residual")
                np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-10)
                cost = json.loads(lines[1])
                interior = (rows - 2) * (columns - 2)
                if (cost["grid_points"], cost["interior_points"], cost["scalar_ops"], cost["output_bytes"]) != (
                    rows * columns, interior, 30 * interior + 4, 8 * interior
                ):
                    raise AssertionError("structural cost accounting mismatch")
                (args.output / f"{name}.input.txt").write_bytes(fixture)
                (args.output / f"{name}.rust.txt").write_bytes(run.stdout)
                (args.output / f"{name}.oracle.txt").write_text(
                    " ".join(f"{v:.17e}" for v in reference) + "\n", encoding="ascii"
                )
                fixtures.append(fixture)
                records.append({
                    "case": name, "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
                    "max_abs_error": float(np.max(np.abs(actual - reference))),
                    "cost": cost,
                })
    report = {
        "schema": "neural-operator.rust-darcy-qualification@1",
        "source_head_sha": os.environ["SOURCE_HEAD_SHA"],
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "oracle_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "oracle_revision": ORACLE_REVISION,
        "oracle_pin_verified_before_import": True,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(b"".join(fixtures)).hexdigest(),
        "rtol": 1e-12, "atol": 1e-10,
        "cases": records, "accepted": len(records) == 32,
        "limitations": ["isotropic residual only; no solver/model migration", "synthetic corpus, no final holdout", "hosted CPU timing is not fixed-host performance or superiority evidence"],
    }
    (args.output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(f"Darcy Rust parity and exact structural cost: {len(records)}/32 cases passed")


if __name__ == "__main__":
    main()
