"""Shared helpers for bornsim performance comparison scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def grid_shape(n_qubits: int) -> tuple[int, int]:
    mapping = {
        10: (2, 5),
        15: (3, 5),
        20: (4, 5),
        28: (4, 7),
    }
    if n_qubits not in mapping:
        raise ValueError(f"unsupported qubit count: {n_qubits}")
    return mapping[n_qubits]


def target_distribution(n_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = rng.random(2**n_qubits, dtype=np.float32)
    target /= target.sum()
    return target.astype(np.float32)


def qiskit_born_ansatz(
    *,
    n_qubits: int,
    n_layers: int,
    coupling_map: list[tuple[int, int]],
) -> tuple[QuantumCircuit, list[Parameter]]:
    n_edges = len(coupling_map)
    n_params = n_layers * (2 * n_qubits + n_edges) + 2 * n_qubits
    circuit = QuantumCircuit(n_qubits)
    params = [Parameter(f"theta_{idx}") for idx in range(n_params)]
    flat_idx = 0

    for _layer in range(n_layers):
        for wire in range(n_qubits):
            circuit.ry(params[flat_idx], wire)
            flat_idx += 1
        for wire in range(n_qubits):
            circuit.rz(params[flat_idx], wire)
            flat_idx += 1
        for src, dst in coupling_map:
            circuit.rzz(params[flat_idx], src, dst)
            flat_idx += 1

    for wire in range(n_qubits):
        circuit.ry(params[flat_idx], wire)
        flat_idx += 1
    for wire in range(n_qubits):
        circuit.rz(params[flat_idx], wire)
        flat_idx += 1

    return circuit, params


def pennylane_apply_born_ansatz(
    theta: pnp.ndarray,
    *,
    n_qubits: int,
    n_layers: int,
    coupling_map: list[tuple[int, int]],
) -> None:
    flat = pnp.reshape(theta, (-1,))
    idx = 0
    for _layer in range(n_layers):
        for wire in range(n_qubits):
            qml.RY(flat[idx], wires=wire)
            idx += 1
        for wire in range(n_qubits):
            qml.RZ(flat[idx], wires=wire)
            idx += 1
        for src, dst in coupling_map:
            qml.IsingZZ(flat[idx], wires=[src, dst])
            idx += 1
    for wire in range(n_qubits):
        qml.RY(flat[idx], wires=wire)
        idx += 1
    for wire in range(n_qubits):
        qml.RZ(flat[idx], wires=wire)
        idx += 1


def save_json(path: Path, payload: object) -> None:
    path.write_text(f"{__import__('json').dumps(payload, indent=2)}\n", encoding="utf-8")
