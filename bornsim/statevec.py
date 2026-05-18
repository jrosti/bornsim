"""Forward statevector simulation for the cuStateVec Born-machine backend."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cupy as cp
import numpy as np
from cuquantum import ComputeType, cudaDataType
from cuquantum.bindings import custatevec as cusv

from bornsim import _config
from bornsim.circuit import Circuit, GateKind
from bornsim.kernels import APPLY_RZ_KERNEL, APPLY_RZZ_KERNEL, launch_dims


@dataclass(frozen=True, slots=True)
class PreparedGate:
    """Per-step cached gate data for fast apply/gradient paths."""

    kind: GateKind
    theta_index: int
    targets: tuple[int, ...]
    bits: tuple[int, ...]
    matrix: cp.ndarray | None
    phase_zero: np.complexfloating[Any, Any] | None
    phase_one: np.complexfloating[Any, Any] | None


class CuStateVecRunner:
    """Thin convenience wrapper around the low-level cuStateVec bindings."""

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.handle = cusv.create()
        self._workspace_cache: dict[tuple[int, bool], cp.cuda.Memory | None] = {}
        self._workspace_size_cache: dict[tuple[int, bool], int] = {}
        self._matrix_cache: dict[tuple[str, bool], cp.ndarray] = {
            ("ry", False): cp.empty((2, 2), dtype=_config.COMPLEX_DTYPE),
            ("ry", True): cp.empty((2, 2), dtype=_config.COMPLEX_DTYPE),
            ("rz", False): cp.empty((2, 2), dtype=_config.COMPLEX_DTYPE),
            ("rz", True): cp.empty((2, 2), dtype=_config.COMPLEX_DTYPE),
            ("rzz", False): cp.empty((4, 4), dtype=_config.COMPLEX_DTYPE),
            ("rzz", True): cp.empty((4, 4), dtype=_config.COMPLEX_DTYPE),
        }

    def destroy(self) -> None:
        """Destroy the cuStateVec handle."""
        cusv.destroy(self.handle)

    def make_zero_state(self) -> cp.ndarray:
        """Return the computational zero state."""
        state = cp.zeros((1 << self.n_qubits,), dtype=_config.COMPLEX_DTYPE)
        state[0] = _config.COMPLEX_DTYPE(1.0 + 0.0j)
        return state

    def logical_to_bit(self, logical_qubit: int) -> int:
        """Map logical qubit index to little-endian statevector bit."""
        return self.n_qubits - 1 - logical_qubit

    def targets_to_bits_tuple(self, targets: tuple[int, ...]) -> tuple[int, ...]:
        """Map logical qubits to backend bit positions."""
        return tuple(self.logical_to_bit(target) for target in targets)

    def _workspace(self, matrix: cp.ndarray, *, n_targets: int, adjoint: bool) -> tuple[int, int]:
        key = (n_targets, adjoint)
        if key not in self._workspace_size_cache:
            workspace_size = int(
                cusv.apply_matrix_get_workspace_size(
                    self.handle,
                    cudaDataType.CUDA_C_32F,
                    self.n_qubits,
                    matrix.data.ptr,
                    cudaDataType.CUDA_C_32F,
                    cusv.MatrixLayout.ROW,
                    int(adjoint),
                    n_targets,
                    0,
                    ComputeType.COMPUTE_32F,
                )
            )
            self._workspace_size_cache[key] = workspace_size
            self._workspace_cache[key] = cp.cuda.alloc(workspace_size) if workspace_size else None
        workspace = self._workspace_cache[key]
        workspace_size = self._workspace_size_cache[key]
        return (0 if workspace is None else workspace.ptr, workspace_size)

    def apply_matrix(
        self,
        state: cp.ndarray,
        matrix: cp.ndarray,
        *,
        targets: tuple[int, ...],
        adjoint: bool = False,
    ) -> None:
        """Apply a dense gate matrix in place."""
        bits = self.targets_to_bits_tuple(targets)
        workspace_ptr, workspace_size = self._workspace(
            matrix,
            n_targets=len(targets),
            adjoint=adjoint,
        )
        cusv.apply_matrix(
            self.handle,
            state.data.ptr,
            cudaDataType.CUDA_C_32F,
            self.n_qubits,
            matrix.data.ptr,
            cudaDataType.CUDA_C_32F,
            cusv.MatrixLayout.ROW,
            int(adjoint),
            bits,
            len(bits),
            0,
            0,
            0,
            ComputeType.COMPUTE_32F,
            workspace_ptr,
            workspace_size,
        )

    def matrix_for_gate(self, kind: str, theta: cp.ndarray) -> cp.ndarray:
        """Return a cached dense gate matrix for one parameter."""
        matrix = self._matrix_cache[(kind, False)]
        if kind == "ry":
            c = cp.cos(theta / 2.0).astype(_config.REAL_DTYPE)
            s = cp.sin(theta / 2.0).astype(_config.REAL_DTYPE)
            matrix.fill(0)
            matrix[0, 0] = c
            matrix[0, 1] = -s
            matrix[1, 0] = s
            matrix[1, 1] = c
            return matrix
        raise ValueError(f"matrix_for_gate only supports dense non-diagonal gates, got {kind}")


def prepare_gates(
    circuit: Circuit,
    params: np.ndarray,
    *,
    runner: CuStateVecRunner,
) -> list[PreparedGate]:
    """Prepare gate payloads for repeated forward/backward application."""
    theta_device = cp.asarray(params, dtype=_config.REAL_DTYPE)
    prepared: list[PreparedGate] = []
    for kind, theta_index, targets in circuit.gate_specs():
        theta = theta_device[theta_index]
        bits = runner.targets_to_bits_tuple(targets)
        if kind == "ry":
            prepared.append(
                PreparedGate(
                    kind=kind,
                    theta_index=theta_index,
                    targets=targets,
                    bits=bits,
                    matrix=runner.matrix_for_gate(kind, theta).copy(),
                    phase_zero=None,
                    phase_one=None,
                )
            )
        else:
            phase_zero = np.complex64(cp.exp(-0.5j * theta).item())
            phase_one = np.complex64(cp.exp(0.5j * theta).item())
            prepared.append(
                PreparedGate(
                    kind=kind,
                    theta_index=theta_index,
                    targets=targets,
                    bits=bits,
                    matrix=None,
                    phase_zero=phase_zero,
                    phase_one=phase_one,
                )
            )
    return prepared


def apply_prepared_gate(
    *,
    runner: CuStateVecRunner,
    state: cp.ndarray,
    gate: PreparedGate,
    adjoint: bool,
) -> None:
    """Apply a prepared gate in place."""
    n_amplitudes = state.size
    if gate.kind == "ry":
        if gate.matrix is None:
            raise ValueError("RY gate requires cached matrix")
        runner.apply_matrix(state, gate.matrix, targets=gate.targets, adjoint=adjoint)
        return

    if gate.phase_zero is None or gate.phase_one is None:
        raise ValueError("diagonal gate requires phases")
    phase_zero = gate.phase_one if adjoint else gate.phase_zero
    phase_one = gate.phase_zero if adjoint else gate.phase_one
    grid, block = launch_dims(n_amplitudes)
    if gate.kind == "rz":
        APPLY_RZ_KERNEL(
            grid,
            block,
            (
                state,
                np.int32(gate.bits[0]),
                phase_zero,
                phase_one,
                np.uint64(n_amplitudes),
            ),
        )
        return
    APPLY_RZZ_KERNEL(
        grid,
        block,
        (
            state,
            np.int32(gate.bits[0]),
            np.int32(gate.bits[1]),
            phase_zero,
            phase_one,
            np.uint64(n_amplitudes),
        ),
    )


def simulate_state(circuit: Circuit, params: np.ndarray) -> tuple[cp.ndarray, float]:
    """Simulate the final statevector.

    Args:
        circuit: Circuit description.
        params: Flat parameter vector of shape `(circuit.n_params,)`.

    Returns:
        Final statevector on GPU and forward wall-clock seconds.
    """
    if params.shape != (circuit.n_params,):
        raise ValueError(f"expected params shape {(circuit.n_params,)}, got {params.shape}")
    runner = CuStateVecRunner(circuit.n_qubits)
    try:
        prepared = prepare_gates(circuit, params, runner=runner)
        started = perf_counter()
        state = runner.make_zero_state()
        for gate in prepared:
            apply_prepared_gate(runner=runner, state=state, gate=gate, adjoint=False)
        cp.cuda.Stream.null.synchronize()
        elapsed = perf_counter() - started
        return state, elapsed
    finally:
        runner.destroy()


def probabilities(state: cp.ndarray) -> cp.ndarray:
    """Convert a statevector to a normalized probability vector."""
    abs2 = (state * cp.conj(state)).real.astype(_config.REAL_DTYPE)
    return abs2 / cp.sum(abs2, dtype=_config.REAL_DTYPE)
