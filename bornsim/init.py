"""Parameter initialization strategies for bornsim circuits."""

from __future__ import annotations

from functools import cache

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from bornsim.circuit import Circuit


def _group_slices(circuit: Circuit) -> list[tuple[str, slice]]:
    groups: list[tuple[str, slice]] = []
    start = 0
    for kind, count in circuit.param_layout:
        groups.append((kind, slice(start, start + count)))
        start += count
    return groups


def _zero_params(circuit: Circuit) -> np.ndarray:
    return np.zeros((circuit.n_params,), dtype=np.float32)


def warm_start(
    circuit: Circuit,
    marginals: np.ndarray,
    correlations: np.ndarray,
    mixing_alpha: float = np.pi / 8,
    noise_scale: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Compute warm-start parameters from target marginals and correlations.

    Args:
        circuit: Circuit description.
        marginals: Per-qubit `P(qubit=1)` vector.
        correlations: Pairwise phi-correlation matrix.
        mixing_alpha: Layer-1 RY mixing angle.
        noise_scale: Gaussian noise scale for remaining layers.
        seed: Random seed for later-layer noise.

    Returns:
        Flat parameter vector in the circuit's canonical order.
    """
    if marginals.shape != (circuit.n_qubits,):
        raise ValueError("marginals shape mismatch")
    if correlations.shape != (circuit.n_qubits, circuit.n_qubits):
        raise ValueError("correlations shape mismatch")

    params = _zero_params(circuit)
    groups = _group_slices(circuit)

    clipped = np.clip(np.asarray(marginals, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    theta = (2.0 * np.arcsin(np.sqrt(clipped))).astype(np.float32)

    first_ry = groups[0][1]
    first_rz = groups[1][1]
    params[first_ry] = theta
    params[first_rz] = 0.0

    if circuit.n_layers >= 2:
        second_ry = groups[3][1]
        params[second_ry] = np.float32(mixing_alpha)

    rzz_group_idx = 2
    rzz_slice = groups[rzz_group_idx][1]
    rzz_values = np.zeros((len(circuit.edges),), dtype=np.float32)
    for edge_idx, (src, dst) in enumerate(circuit.edges):
        rzz_values[edge_idx] = np.float32(
            fit_rzz_angle(
                float(correlations[src, dst]),
                float(theta[src]),
                float(theta[dst]),
                float(mixing_alpha),
                float(mixing_alpha),
            )
        )
    params[rzz_slice] = rzz_values

    rng = np.random.default_rng(seed)
    start_group = 5 if circuit.n_layers >= 2 else 3
    for _kind, group_slice in groups[start_group:]:
        size = group_slice.stop - group_slice.start
        params[group_slice] = rng.normal(0.0, noise_scale, size=(size,)).astype(np.float32)
    return params


def random_init(
    circuit: Circuit,
    seed: int = 0,
    scale: float = 0.1,
) -> np.ndarray:
    """Return `scale * randn` in the canonical parameter order."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(circuit.n_params,)).astype(np.float32)


def _two_qubit_params(
    theta_i: float,
    theta_j: float,
    alpha_i: float,
    alpha_j: float,
    phi: float,
) -> pnp.ndarray:
    params = np.zeros((10,), dtype=np.float32)
    params[0] = theta_i
    params[1] = theta_j
    params[4] = phi
    params[5] = alpha_i
    params[6] = alpha_j
    return pnp.asarray(params, dtype=np.float32)


@cache
def _two_qubit_qnode() -> qml.QNode:
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method="backprop")  # type: ignore[untyped-decorator]
    def probs(params: pnp.ndarray) -> pnp.ndarray:
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.RZ(params[2], wires=0)
        qml.RZ(params[3], wires=1)
        qml.IsingZZ(params[4], wires=[0, 1])
        qml.RY(params[5], wires=0)
        qml.RY(params[6], wires=1)
        qml.RZ(params[7], wires=0)
        qml.RZ(params[8], wires=1)
        return qml.probs(wires=[0, 1])

    return probs


def _pairwise_correlation_from_probs(probs: np.ndarray) -> float:
    bits = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    means = probs @ bits
    centered = bits - means
    second = centered.T @ (centered * probs[:, None])
    var0 = max(second[0, 0], 1e-12)
    var1 = max(second[1, 1], 1e-12)
    return float(second[0, 1] / np.sqrt(var0 * var1))


def fit_rzz_angle(
    target_corr: float,
    theta_i: float,
    theta_j: float,
    alpha_i: float,
    alpha_j: float,
    *,
    grid_size: int = 257,
) -> float:
    """Fit an RZZ angle to a target two-qubit correlation."""
    if abs(target_corr) < 1e-4:
        return 0.0
    return _fit_rzz_angle_cached(
        round(float(target_corr), 6),
        round(float(theta_i), 6),
        round(float(theta_j), 6),
        round(float(alpha_i), 6),
        round(float(alpha_j), 6),
        grid_size,
    )


@cache
def _fit_rzz_angle_cached(
    target_corr: float,
    theta_i: float,
    theta_j: float,
    alpha_i: float,
    alpha_j: float,
    grid_size: int,
) -> float:
    qnode = _two_qubit_qnode()
    best_phi = 0.0
    best_error = float("inf")
    search_lo = -np.pi
    search_hi = np.pi
    for _pass in range(3):
        for phi in np.linspace(search_lo, search_hi, grid_size, dtype=np.float32):
            probs = np.asarray(
                qnode(_two_qubit_params(theta_i, theta_j, alpha_i, alpha_j, float(phi))),
                dtype=np.float64,
            )
            corr = _pairwise_correlation_from_probs(probs)
            error = abs(corr - target_corr)
            if error < best_error:
                best_error = error
                best_phi = float(phi)
        half_width = (search_hi - search_lo) / 8.0
        search_lo = max(-np.pi, best_phi - half_width)
        search_hi = min(np.pi, best_phi + half_width)
    return best_phi
