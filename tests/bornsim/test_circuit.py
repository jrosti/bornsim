"""Tests for the public bornsim Circuit API."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from bornsim.circuit import Circuit


def test_circuit_n_params_formula_matches_hand_count() -> None:
    circuit = Circuit(n_qubits=4, n_layers=2, edges=((0, 1), (1, 2), (2, 3)))
    assert circuit.n_params == 2 * (2 * 4 + 3) + 2 * 4


def test_circuit_param_layout_is_canonical() -> None:
    circuit = Circuit(n_qubits=3, n_layers=2, edges=((0, 1), (1, 2)))
    assert circuit.param_layout == (
        ("ry", 3),
        ("rz", 3),
        ("rzz", 2),
        ("ry", 3),
        ("rz", 3),
        ("rzz", 2),
        ("ry", 3),
        ("rz", 3),
    )


def test_circuit_is_immutable() -> None:
    circuit = Circuit(n_qubits=4, n_layers=1, edges=((0, 1),))
    with pytest.raises(FrozenInstanceError):
        circuit.n_qubits = 5  # type: ignore[misc]
