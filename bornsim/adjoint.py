"""Reverse-mode gradient computation for bornsim statevectors."""

from __future__ import annotations

from time import perf_counter

import cupy as cp
import numpy as np

from bornsim import _config
from bornsim.circuit import Circuit
from bornsim.kernels import GRAD_RY_KERNEL, GRAD_RZ_KERNEL, GRAD_RZZ_KERNEL, launch_dims
from bornsim.statevec import CuStateVecRunner, PreparedGate, apply_prepared_gate, prepare_gates


def cotangent_from_probgrad(
    state_final: cp.ndarray,
    probs: cp.ndarray,
    grad_probs: cp.ndarray,
) -> cp.ndarray:
    """Map `dL/dp` to a statevector cotangent."""
    abs2 = (state_final * cp.conj(state_final)).real.astype(_config.REAL_DTYPE)
    norm = cp.sum(abs2, dtype=_config.REAL_DTYPE)
    dot_term = cp.dot(grad_probs, probs)
    grad_abs2 = (grad_probs - dot_term) / norm
    return state_final * grad_abs2.astype(_config.COMPLEX_DTYPE)


def _generator_gradient(
    *,
    lam: cp.ndarray,
    psi: cp.ndarray,
    gate: PreparedGate,
    reduce_buffer: cp.ndarray,
) -> cp.ndarray:
    reduce_buffer.fill(0)
    if gate.kind == "ry":
        grid, block = launch_dims(psi.size // 2)
        GRAD_RY_KERNEL(
            grid,
            block,
            (
                lam,
                psi,
                np.int32(gate.bits[0]),
                np.uint64(psi.size // 2),
                reduce_buffer,
            ),
        )
        return reduce_buffer[0]
    if gate.kind == "rz":
        grid, block = launch_dims(psi.size)
        GRAD_RZ_KERNEL(
            grid,
            block,
            (
                lam,
                psi,
                np.int32(gate.bits[0]),
                np.uint64(psi.size),
                reduce_buffer,
            ),
        )
        return reduce_buffer[0]
    grid, block = launch_dims(psi.size)
    GRAD_RZZ_KERNEL(
        grid,
        block,
        (
            lam,
            psi,
            np.int32(gate.bits[0]),
            np.int32(gate.bits[1]),
            np.uint64(psi.size),
            reduce_buffer,
        ),
    )
    return reduce_buffer[0]


def gradient(
    circuit: Circuit,
    params: np.ndarray,
    state_final: cp.ndarray,
    cotangent: cp.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute the reverse-mode gradient for a full circuit.

    Args:
        circuit: Circuit description.
        params: Flat parameter vector.
        state_final: Final statevector from the forward pass.
        cotangent: `dL/dpsi` cotangent state on GPU.

    Returns:
        Host gradient vector and backward wall-clock seconds.
    """
    runner = CuStateVecRunner(circuit.n_qubits)
    try:
        prepared = prepare_gates(circuit, params, runner=runner)
        psi = state_final.copy()
        lam = cotangent.copy()
        grads = cp.empty((circuit.n_params,), dtype=_config.REAL_DTYPE)
        reduce_buffer = cp.zeros((1,), dtype=_config.REAL_DTYPE)
        started = perf_counter()
        for gate in reversed(prepared):
            grads[gate.theta_index] = _generator_gradient(
                lam=lam,
                psi=psi,
                gate=gate,
                reduce_buffer=reduce_buffer,
            )
            apply_prepared_gate(runner=runner, state=psi, gate=gate, adjoint=True)
            apply_prepared_gate(runner=runner, state=lam, gate=gate, adjoint=True)
        cp.cuda.Stream.null.synchronize()
        elapsed = perf_counter() - started
        return cp.asnumpy(grads), elapsed
    finally:
        runner.destroy()
