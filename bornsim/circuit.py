"""Immutable circuit description for the Born-machine ansatz."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GateKind = Literal["ry", "rz", "rzz"]


@dataclass(frozen=True, slots=True)
class Circuit:
    """Immutable description of the RY-RZ-RZZ Born ansatz.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of entangling layers.
        edges: Undirected RZZ coupling map.
        closing_block: Whether to append the final single-qubit RY-RZ block.
    """

    n_qubits: int
    n_layers: int
    edges: tuple[tuple[int, int], ...]
    closing_block: bool = True

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        for src, dst in self.edges:
            if not (0 <= src < self.n_qubits and 0 <= dst < self.n_qubits):
                raise ValueError("edge endpoint out of range")
            if src == dst:
                raise ValueError("self-loops are not allowed")

    @property
    def n_params(self) -> int:
        """Return total scalar parameter count."""
        base = self.n_layers * (2 * self.n_qubits + len(self.edges))
        if self.closing_block:
            base += 2 * self.n_qubits
        return base

    @property
    def param_layout(self) -> tuple[tuple[str, int], ...]:
        """Return the canonical parameter-group layout."""
        layout: list[tuple[str, int]] = []
        for _layer in range(self.n_layers):
            layout.extend((("ry", self.n_qubits), ("rz", self.n_qubits), ("rzz", len(self.edges))))
        if self.closing_block:
            layout.extend((("ry", self.n_qubits), ("rz", self.n_qubits)))
        return tuple(layout)

    def gate_specs(self) -> tuple[tuple[GateKind, int, tuple[int, ...]], ...]:
        """Return the canonical sequence of parameterized gates.

        Returns:
            Tuple of `(kind, theta_index, targets)` entries in application order.
        """
        specs: list[tuple[GateKind, int, tuple[int, ...]]] = []
        theta_index = 0
        for _layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                specs.append(("ry", theta_index, (qubit,)))
                theta_index += 1
            for qubit in range(self.n_qubits):
                specs.append(("rz", theta_index, (qubit,)))
                theta_index += 1
            for src, dst in self.edges:
                specs.append(("rzz", theta_index, (src, dst)))
                theta_index += 1
        if self.closing_block:
            for qubit in range(self.n_qubits):
                specs.append(("ry", theta_index, (qubit,)))
                theta_index += 1
            for qubit in range(self.n_qubits):
                specs.append(("rz", theta_index, (qubit,)))
                theta_index += 1
        return tuple(specs)
