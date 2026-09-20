# Optional autodiff backend — 20 September 2026

NeuralOperator now contains an optional PyTorch implementation of the same
small one-block 1D FNO contract as the NumPy/SciPy reference.

## Contract

- inputs and outputs use shape (batch, points, channels);
- parameters use float64 for comparison with the numerical oracle;
- Fourier transforms use torch.fft.rfft and torch.fft.irfft;
- optimization uses Adam and PyTorch autograd;
- the dependency-free NumPy/SciPy implementation remains the acceptance oracle.

## Reproduction

Install the optional dependency set:

    python3 -m pip install -r requirements-autodiff.txt

Run the focused tests:

    PYTHONPATH=src python3 -m unittest discover -s tests -p test_torch_fno.py -v

This backend currently proves the training contract and gradient path. It does
not yet prove parity of trained weights, speedups, GPU behavior, or production
stability. Those comparisons are the next measurement gate.
