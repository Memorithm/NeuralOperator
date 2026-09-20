# Dataset split protocol — 20 September 2026

The Burgers benchmark now has an explicit partition contract.

## Shared quantities

Train, validation and OOD partitions must use the same:

- spatial resolution and periodic domain length;
- solver time step;
- number of solver substeps per learned transition.

This keeps a change in reported error attributable to the data or physical
distribution rather than an unannounced change in the numerical protocol.

## Partitioned quantities

The specifications retain independent:

- random seed;
- initial-condition Fourier bandwidth;
- initial-condition amplitude;
- viscosity.

The current OOD protocol therefore tests a controlled shift in the initial
condition family and viscosity. It does not yet test external forcing; that
requires an explicit forcing contract in the reference solver.

## Reproduction

From the repository root:

    PYTHONPATH=src python3 scripts/burgers_split_experiment.py

The script reports the fit summary plus recursive validation and OOD metrics.
It is an exploratory reference experiment, not evidence of production
generalization.
