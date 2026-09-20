"""Deterministic train/validation/OOD Burgers dataset specifications."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import BurgersDataset, generate_burgers_dataset


@dataclass(frozen=True)
class BurgersDatasetSpec:
    """Configuration and provenance for one Burgers experiment partition.

    The split currently varies the initial-condition family through modes and
    amplitude, and can vary viscosity. Forcing is deliberately not exposed
    until the reference solver has an explicit forcing contract.
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


def generate_burgers_split(
    train: BurgersDatasetSpec,
    validation: BurgersDatasetSpec,
    ood: BurgersDatasetSpec,
) -> BurgersDatasetSplit:
    """Generate compatible train, validation and OOD partitions.

    The spatial grid and transition duration must be shared across partitions.
    Initial-condition statistics and viscosity may differ by design. This
    prevents an OOD result from silently mixing changes in the numerical grid
    or prediction horizon with changes in the physical/data distribution.
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

    return BurgersDatasetSplit(
        train=train.build(),
        validation=validation.build(),
        ood=ood.build(),
        train_spec=train,
        validation_spec=validation,
        ood_spec=ood,
    )
