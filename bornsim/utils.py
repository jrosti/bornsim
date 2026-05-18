"""Probability-vector utilities for bornsim callers."""

from __future__ import annotations

from collections.abc import Sequence

import cupy as cp
import numpy as np


def marginals(probs: cp.ndarray, n_qubits: int, qubits: Sequence[int]) -> cp.ndarray:
    """Return the marginal over a subset of qubits.

    Args:
        probs: Full probability vector of shape `(2**n_qubits,)`.
        n_qubits: Total qubit count.
        qubits: Logical qubit indices to keep.

    Returns:
        Marginal distribution on the same device as `probs`.
    """
    keep = tuple(int(qubit) for qubit in qubits)
    tensor = probs.reshape((2,) * n_qubits)
    axes_to_sum = tuple(axis for axis in range(n_qubits) if axis not in keep)
    reduced = tensor.sum(axis=axes_to_sum)
    return reduced.reshape((-1,))


def conditional(
    probs: cp.ndarray,
    n_qubits: int,
    conditioning_qubits: Sequence[int],
    conditioning_values: Sequence[int],
) -> cp.ndarray:
    """Return `P(rest | conditioning_qubits = values)`.

    Raises:
        ValueError: If the conditioning event has zero probability.
    """
    if len(conditioning_qubits) != len(conditioning_values):
        raise ValueError("conditioning_qubits and conditioning_values length mismatch")
    tensor = probs.reshape((2,) * n_qubits)
    index: list[slice | int] = [slice(None)] * n_qubits
    for qubit, value in zip(conditioning_qubits, conditioning_values, strict=True):
        index[int(qubit)] = int(value)
    conditioned = tensor[tuple(index)]
    total = cp.sum(conditioned)
    if float(total.get()) == 0.0:
        raise ValueError("conditioning event has zero probability")
    return (conditioned / total).reshape((-1,))


def to_numpy(probs: cp.ndarray) -> np.ndarray:
    """Move a probability vector to host memory."""
    return cp.asnumpy(probs)
