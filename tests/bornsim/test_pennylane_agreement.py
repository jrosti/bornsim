"""PennyLane agreement tests for bornsim forward and backward passes."""

from __future__ import annotations

import cupy as cp
import numpy as np
import optax
import pennylane as qml
from pennylane import numpy as pnp

from bornsim.circuit import Circuit
from bornsim.losses import MMD
from bornsim.trainer import Trainer
from tests.bornsim.conftest import gpu_only


def _pl_probs(circuit: Circuit, params: np.ndarray) -> np.ndarray:
    dev = qml.device("default.qubit", wires=circuit.n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")  # type: ignore[untyped-decorator]
    def qnode(theta: pnp.ndarray) -> pnp.ndarray:
        for kind, theta_index, targets in circuit.gate_specs():
            if kind == "ry":
                qml.RY(theta[theta_index], wires=targets[0])
            elif kind == "rz":
                qml.RZ(theta[theta_index], wires=targets[0])
            else:
                qml.IsingZZ(theta[theta_index], wires=list(targets))
        return qml.probs(wires=range(circuit.n_qubits))

    return np.asarray(qnode(pnp.asarray(params, dtype=np.float32)), dtype=np.float32)


def _mmd_kernel_matrix(n_qubits: int, bandwidths: tuple[float, ...]) -> np.ndarray:
    states = np.arange(2**n_qubits, dtype=np.uint32)[:, None]
    bit_indices = np.arange(n_qubits - 1, -1, -1, dtype=np.uint32)[None, :]
    bits = ((states >> bit_indices) & 1).astype(np.float32)
    sqdist = np.sum((bits[:, None, :] - bits[None, :, :]) ** 2, axis=-1)
    kernel = np.zeros_like(sqdist, dtype=np.float32)
    for bandwidth in bandwidths:
        kernel += np.exp(-sqdist / (2.0 * bandwidth * bandwidth)).astype(np.float32)
    return kernel


def _pl_loss_grad(
    circuit: Circuit,
    params: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray]:
    dev = qml.device("default.qubit", wires=circuit.n_qubits)
    target_pnp = pnp.asarray(target, dtype=np.float32)
    kernel = pnp.asarray(_mmd_kernel_matrix(circuit.n_qubits, (0.25, 0.5, 1.0)), dtype=np.float32)

    @qml.qnode(dev, diff_method="parameter-shift")  # type: ignore[untyped-decorator]
    def qnode(theta: pnp.ndarray) -> pnp.ndarray:
        for kind, theta_index, targets in circuit.gate_specs():
            if kind == "ry":
                qml.RY(theta[theta_index], wires=targets[0])
            elif kind == "rz":
                qml.RZ(theta[theta_index], wires=targets[0])
            else:
                qml.IsingZZ(theta[theta_index], wires=list(targets))
        return qml.probs(wires=range(circuit.n_qubits))

    def objective(theta: pnp.ndarray) -> pnp.ndarray:
        probs = qnode(theta)
        delta = probs - target_pnp
        return pnp.dot(delta, kernel @ delta)

    params_pnp = pnp.asarray(params, dtype=np.float32, requires_grad=True)
    probs_np = np.asarray(qnode(params_pnp), dtype=np.float32)
    delta_np = probs_np - target
    kernel_np = np.asarray(kernel, dtype=np.float32)
    loss_value = float(delta_np @ (kernel_np @ delta_np))
    grad_value = np.asarray(qml.grad(objective)(params_pnp), dtype=np.float32)
    return loss_value, grad_value


@gpu_only
def test_pennylane_agreement_small_topologies() -> None:
    topologies = [
        Circuit(n_qubits=4, n_layers=2, edges=((0, 1), (1, 2), (2, 3))),
        Circuit(
            n_qubits=6,
            n_layers=3,
            edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)),
        ),
        Circuit(
            n_qubits=8,
            n_layers=2,
            edges=((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)),
        ),
    ]
    scales = [0.0, 0.1, 1.0]
    rng = np.random.default_rng(0)
    for circuit in topologies:
        target = rng.random(2 ** circuit.n_qubits, dtype=np.float32)
        target /= target.sum()
        trainer = Trainer(circuit, MMD(target, bandwidths=(0.25, 0.5, 1.0)), optax.adam(1e-2))
        for scale in scales:
            params = rng.normal(0.0, scale, size=(circuit.n_params,)).astype(np.float32)
            born_probs = cp.asnumpy(trainer.simulate(params))
            pl_probs = _pl_probs(circuit, params)
            np.testing.assert_allclose(born_probs, pl_probs, atol=1e-5)

            _born_loss, born_grad = trainer.gradient(params)
            _pl_loss, pl_grad = _pl_loss_grad(circuit, params, target)
            np.testing.assert_allclose(born_grad, pl_grad, atol=1e-5)
