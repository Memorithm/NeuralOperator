"""Write an explicit environment/provenance manifest for a benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command_output(*command: str) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = {
        "schema": "neural-operator.reproducibility-manifest.v1",
        "repository": os.environ.get("GITHUB_REPOSITORY", "local"),
        "source_head_sha": os.environ.get("SOURCE_HEAD_SHA", os.environ.get("GITHUB_SHA", "unavailable")),
        "checkout_sha": os.environ.get("GITHUB_SHA", "unavailable"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "lock_file": str(args.lock),
        "lock_sha256": sha256(args.lock),
        "numpy_scipy_torch_data_sha256": os.environ.get("DATASET_SHA256", "not-provided"),
        "checkpoint_sha256": os.environ.get("CHECKPOINT_SHA256", "not-provided"),
        "rust_or_external_toolchain": os.environ.get("TOOLCHAIN_ID", "not-applicable"),
        "git": command_output("git", "--version"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
