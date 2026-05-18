"""Integration tests for the bornsim Trainer."""

from __future__ import annotations

import numpy as np
import optax

from bornsim.circuit import Circuit
from bornsim.losses import KL
from bornsim.trainer import Trainer
from tests.bornsim.conftest import gpu_only


@gpu_only
def test_trainer_step_decreases_loss_and_gradient_matches() -> None:
    circuit = Circuit(n_qubits=6, n_layers=2, edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)))
    target = np.full((2**6,), 1.0 / (2**6), dtype=np.float32)
    trainer = Trainer(circuit, KL(target), optax.adam(1e-2))
    params = np.zeros((circuit.n_params,), dtype=np.float32)
    opt_state = trainer.init_opt_state(params)

    losses: list[float] = []
    for _step in range(20):
        params, opt_state, info = trainer.step(params, opt_state)
        losses.append(info["loss_value"])

    assert losses[-1] <= losses[0]
    probs = trainer.simulate(params)
    np.testing.assert_allclose(float(probs.sum().get()), 1.0, atol=1e-5)

    loss_value, grad = trainer.gradient(params)
    _params_next, _opt_state_next, info = trainer.step(params, opt_state)
    np.testing.assert_allclose(float(loss_value), info["loss_value"], atol=1e-5)
    assert grad.shape == params.shape
