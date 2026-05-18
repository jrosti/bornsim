"""Lower-level adjoint correctness checks."""

from __future__ import annotations

import numpy as np
import optax

from bornsim.circuit import Circuit
from bornsim.losses import MMD
from bornsim.trainer import Trainer
from tests.bornsim.conftest import gpu_only


@gpu_only
def test_adjoint_matches_numerical_central_difference() -> None:
    circuit = Circuit(n_qubits=6, n_layers=2, edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)))
    rng = np.random.default_rng(0)
    params = rng.normal(0.0, 0.1, size=(circuit.n_params,)).astype(np.float32)
    target = rng.random(2**6, dtype=np.float32)
    target /= target.sum()
    trainer = Trainer(circuit, MMD(target, bandwidths=(0.25, 0.5, 1.0)), optax.adam(1e-2))
    _loss, grad = trainer.gradient(params)

    eps = 5e-4
    numeric = np.zeros_like(grad)
    for idx in range(circuit.n_params):
        plus = params.copy()
        minus = params.copy()
        plus[idx] += eps
        minus[idx] -= eps
        loss_plus, _ = trainer.gradient(plus)
        loss_minus, _ = trainer.gradient(minus)
        numeric[idx] = np.float32((loss_plus - loss_minus) / (2 * eps))
    np.testing.assert_allclose(grad, numeric, atol=1e-2)
