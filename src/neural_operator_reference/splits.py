"""Deterministic train/validation/OOD Burgers dataset specifications."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .datasets import BurgersDataset, generate_burgers_dataset


@dataclass(frozen=True)
class BurgersDatasetSpec:
    """Configuration and provenance for one Burgers experiment partition.

    The split varies the initial-condition family through modes and amplitude,
    viscosity, and a static sinusoidal forcing amplitude/mode. Time-dependent
    forcing is deliberately outside this reference solver contract.
    """

    samples: int
    points: int
    dt: float
    steps: int
    viscosity: float
    seed: int
    length: float = 2.0 * np.pi
    modes: int = 4
    amplitude: float = 0.5
    forcing_amplitude: float = 0.0
    forcing_mode: int = 1

    def build(self) -> BurgersDataset:
        """Generate this partition with the reference solver."""

        return generate_burgers_dataset(
            samples=self.samples,
            points=self.points,
            dt=self.dt,
            steps=self.steps,
            viscosity=self.viscosity,
            seed=self.seed,
            length=self.length,
            modes=self.modes,
            amplitude=self.amplitude,
            forcing_amplitude=self.forcing_amplitude,
            forcing_mode=self.forcing_mode,
        )

    def as_dict(self) -> dict[str, int | float]:
        """Return the complete configuration as JSON-compatible scalars."""

        return {
            "samples": int(self.samples),
            "points": int(self.points),
            "dt": float(self.dt),
            "steps": int(self.steps),
            "viscosity": float(self.viscosity),
            "seed": int(self.seed),
            "length": float(self.length),
            "modes": int(self.modes),
            "amplitude": float(self.amplitude),
            "forcing_amplitude": float(self.forcing_amplitude),
            "forcing_mode": int(self.forcing_mode),
        }


@dataclass(frozen=True)
class BurgersDatasetSplit:
    """Generated train, validation and out-of-distribution partitions."""

    train: BurgersDataset
    validation: BurgersDataset
    ood: BurgersDataset
    train_spec: BurgersDatasetSpec
    validation_spec: BurgersDatasetSpec
    ood_spec: BurgersDatasetSpec

    def as_dict(self) -> dict[str, dict[str, int | float]]:
        """Return partition provenance for experiment manifests."""

        return {
            "train": self.train_spec.as_dict(),
            "validation": self.validation_spec.as_dict(),
            "ood": self.ood_spec.as_dict(),
        }


def _sample_identity(sample: np.ndarray) -> str:
    """Return a canonical identity for one generated initial condition."""

    canonical = np.ascontiguousarray(sample)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(value) for value in canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _assert_disjoint_inputs(partitions: dict[str, BurgersDataset]) -> None:
    """Reject any initial condition shared by two experiment partitions."""

    owners: dict[str, str] = {}
    for name, dataset in partitions.items():
        local: set[str] = set()
        for sample in dataset.inputs:
            identity = _sample_identity(sample)
            if identity in local:
                raise ValueError(f"{name} contains a duplicate input sample")
            local.add(identity)
            previous = owners.get(identity)
            if previous is not None:
                raise ValueError(
                    f"{name} overlaps {previous}: generated input sample {identity}"
                )
            owners[identity] = name


def generate_burgers_split(
    train: BurgersDatasetSpec,
    validation: BurgersDatasetSpec,
    ood: BurgersDatasetSpec,
) -> BurgersDatasetSplit:
    """Generate compatible train, validation and OOD partitions.

    The spatial grid and transition duration must be shared across partitions.
    Initial-condition statistics, viscosity and forcing may differ by design.
    Seeds are role-separated and generated initial conditions are checked by a
    canonical content identity. This prevents silent train/validation/OOD
    leakage as well as mixing numerical-grid changes with distribution changes.
    """

    specs = (train, validation, ood)
    for name, spec in zip(("train", "validation", "ood"), specs):
        if not isinstance(spec, BurgersDatasetSpec):
            raise TypeError(f"{name} must be a BurgersDatasetSpec")

    for attribute in ("points", "dt", "steps", "length"):
        values = [getattr(spec, attribute) for spec in specs]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"train, validation and ood must share {attribute}; got {values}"
            )

    seeds = [spec.seed for spec in specs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "train, validation and ood must use distinct seed domains; "
            f"got {seeds}"
        )

    partitions = {
        "train": train.build(),
        "validation": validation.build(),
        "ood": ood.build(),
    }
    _assert_disjoint_inputs(partitions)

    return BurgersDatasetSplit(
        train=partitions["train"],
        validation=partitions["validation"],
        ood=partitions["ood"],
        train_spec=train,
        validation_spec=validation,
        ood_spec=ood,
    )
