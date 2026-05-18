"""Loss objects for full-probability Born-machine training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import cupy as cp
import numpy as np

from bornsim import _config


class Loss(Protocol):
    """Protocol for losses defined on full probability vectors.

    Implementers take a model probability vector on GPU and return the scalar
    loss value plus the gradient with respect to probabilities on the same
    device.
    """

    def value_and_probgrad(self, probs: cp.ndarray) -> tuple[float, cp.ndarray]:
        """Return `(loss_value, dloss_dprobs)` for a model probability vector."""


def _apply_factored_kernel(delta: cp.ndarray, n_qubits: int, bandwidth: float) -> cp.ndarray:
    alpha = cp.float32(np.exp(-1.0 / (2.0 * bandwidth * bandwidth)))
    kernel = cp.asarray([[1.0, alpha], [alpha, 1.0]], dtype=_config.REAL_DTYPE)
    vec = cp.reshape(delta, (2,) * n_qubits)
    for axis in range(n_qubits):
        vec = cp.moveaxis(vec, axis, 0)
        trailing_shape = vec.shape[1:]
        vec = cp.reshape(vec, (2, -1))
        vec = kernel @ vec
        vec = cp.reshape(vec, (2, *trailing_shape))
        vec = cp.moveaxis(vec, 0, axis)
    return cp.reshape(vec, (-1,))


class MMD:
    """Exact mixture-of-RBF MMD loss on `{0,1}^n` full distributions."""

    def __init__(self, p_data: np.ndarray, bandwidths: Sequence[float]):
        self.p_data = cp.asarray(np.asarray(p_data, dtype=np.float32), dtype=_config.REAL_DTYPE)
        self.bandwidths = tuple(float(value) for value in bandwidths)
        n_states = int(self.p_data.size)
        n_qubits = n_states.bit_length() - 1
        if 2**n_qubits != n_states:
            raise ValueError("p_data length must be a power of two")
        self.n_qubits = n_qubits

    def value_and_probgrad(self, probs: cp.ndarray) -> tuple[float, cp.ndarray]:
        delta = probs.astype(_config.REAL_DTYPE) - self.p_data
        transformed_total = cp.zeros_like(delta)
        for bandwidth in self.bandwidths:
            transformed_total += _apply_factored_kernel(delta, self.n_qubits, bandwidth)
        loss = float(cp.dot(delta, transformed_total).get())
        grad_probs = 2.0 * transformed_total
        return loss, grad_probs.astype(_config.REAL_DTYPE)


class KL:
    """Forward KL loss `KL(p_data || p_model)`."""

    def __init__(self, p_data: np.ndarray, eps: float = 1e-12):
        self.p_data = cp.asarray(np.asarray(p_data, dtype=np.float32), dtype=_config.REAL_DTYPE)
        self.eps = float(eps)

    def value_and_probgrad(self, probs: cp.ndarray) -> tuple[float, cp.ndarray]:
        probs_real = probs.astype(_config.REAL_DTYPE)
        probs_clipped = cp.clip(probs_real, self.eps, 1.0)
        target_clipped = cp.clip(self.p_data, self.eps, 1.0)
        terms = target_clipped * (cp.log(target_clipped) - cp.log(probs_clipped))
        loss = float(cp.sum(terms, dtype=_config.REAL_DTYPE).get())
        grad_probs = cp.where(
            probs_real > self.eps,
            -target_clipped / probs_clipped,
            cp.zeros_like(probs_clipped),
        )
        return loss, grad_probs.astype(_config.REAL_DTYPE)
