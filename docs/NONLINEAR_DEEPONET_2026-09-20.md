# Nonlinear DeepONet backend — 20 September 2026

The optional PyTorch backend now includes a genuinely nonlinear branch/trunk
DeepONet. The branch encodes fixed sensor values and the trunk encodes query
coordinates, so the number of query points can change at inference time.

The acceptance test trains on a controlled nonlinear synthetic operator, checks
the autograd loss reduction, and evaluates both the training query grid and a
denser query grid on held-out inputs.

This does not yet establish superiority over FNO, interpolation, convolution,
or the linear DeepONet. Those comparisons require a shared data split,
parameter budget, seed protocol, rollout horizon and cost measurement.
