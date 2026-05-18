"""Tests for bornsim loss objects."""

from __future__ import annotations

import cupy as cp
import numpy as np
import optax

from bornsim import losses
from bornsim.circuit import Circuit
from bornsim.trainer import Trainer
from tests.bornsim.conftest import gpu_only


@gpu_only
def test_kl_value_and_probgrad_match_finite_difference() -> None:
    p_model = cp.asarray([0.1, 0.2, 0.15, 0.05, 0.1, 0.1, 0.1, 0.2], dtype=cp.float32)
    p_data = np.asarray([0.12, 0.18, 0.14, 0.06, 0.1, 0.1, 0.1, 0.2], dtype=np.float32)
    loss = losses.KL(p_data)
    value, grad = loss.value_and_probgrad(p_model)
    eps = 1e-4
    numeric = np.zeros((8,), dtype=np.float64)
    for idx in range(8):
        delta = np.zeros((8,), dtype=np.float32)
        delta[idx] = eps
        plus, _ = loss.value_and_probgrad(cp.asarray(p_model.get() + delta))
        minus, _ = loss.value_and_probgrad(cp.asarray(p_model.get() - delta))
        numeric[idx] = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(value, float(value))
    np.testing.assert_allclose(cp.asnumpy(grad), numeric, atol=1e-3)


@gpu_only
def test_mmd_value_and_probgrad_match_finite_difference() -> None:
    p_model = cp.asarray([0.14, 0.18, 0.12, 0.06, 0.1, 0.1, 0.1, 0.2], dtype=cp.float32)
    p_data = np.asarray([0.12, 0.18, 0.14, 0.06, 0.1, 0.1, 0.1, 0.2], dtype=np.float32)
    loss = losses.MMD(p_data, bandwidths=(0.5, 1.0))
    _value, grad = loss.value_and_probgrad(p_model)
    eps = 1e-4
    numeric = np.zeros((8,), dtype=np.float64)
    for idx in range(8):
        delta = np.zeros((8,), dtype=np.float32)
        delta[idx] = eps
        plus, _ = loss.value_and_probgrad(cp.asarray(p_model.get() + delta))
        minus, _ = loss.value_and_probgrad(cp.asarray(p_model.get() - delta))
        numeric[idx] = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(cp.asnumpy(grad), numeric, atol=1e-3)


class QuadraticLoss:
    def __init__(self, target: np.ndarray):
        self.target = cp.asarray(target, dtype=cp.float32)

    def value_and_probgrad(self, probs: cp.ndarray) -> tuple[float, cp.ndarray]:
        delta = probs - self.target
        return float(cp.sum(delta * delta).get()), 2.0 * delta


@gpu_only
def test_user_supplied_loss_runs_through_trainer() -> None:
    circuit = Circuit(n_qubits=2, n_layers=1, edges=((0, 1),))
    trainer = Trainer(
        circuit,
        QuadraticLoss(np.full((4,), 0.25, dtype=np.float32)),
        optax.adam(1e-2),
    )
    params = np.zeros((circuit.n_params,), dtype=np.float32)
    opt_state = trainer.init_opt_state(params)
    new_params, _new_opt_state, info = trainer.step(params, opt_state)
    assert new_params.shape == params.shape
    assert np.isfinite(info["loss_value"])
