"""Parameter initialization strategies for bornsim circuits."""

from __future__ import annotations

from functools import cache

import numpy as np

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


def _ry_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta / 2.0))
    s = float(np.sin(theta / 2.0))
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz_matrix(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.exp(-0.5j * theta), 0.0],
            [0.0, np.exp(0.5j * theta)],
        ],
        dtype=np.complex128,
    )


def _rzz_matrix(theta: float) -> np.ndarray:
    phase_same = np.exp(-0.5j * theta)
    phase_diff = np.exp(0.5j * theta)
    return np.diag([phase_same, phase_diff, phase_diff, phase_same]).astype(np.complex128)


def _apply_single_qubit(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    full_gate = np.kron(gate, np.eye(2, dtype=np.complex128)) if qubit == 0 else np.kron(
        np.eye(2, dtype=np.complex128),
        gate,
    )
    return full_gate @ state


@cache
def _two_qubit_probs(
    theta_i: float,
    theta_j: float,
    alpha_i: float,
    alpha_j: float,
    phi: float,
) -> tuple[float, float, float, float]:
    state = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    state = _apply_single_qubit(state, _ry_matrix(theta_i), 0)
    state = _apply_single_qubit(state, _ry_matrix(theta_j), 1)
    state = _apply_single_qubit(state, _rz_matrix(0.0), 0)
    state = _apply_single_qubit(state, _rz_matrix(0.0), 1)
    state = _rzz_matrix(phi) @ state
    state = _apply_single_qubit(state, _ry_matrix(alpha_i), 0)
    state = _apply_single_qubit(state, _ry_matrix(alpha_j), 1)
    state = _apply_single_qubit(state, _rz_matrix(0.0), 0)
    state = _apply_single_qubit(state, _rz_matrix(0.0), 1)
    probs = np.abs(state) ** 2
    probs /= probs.sum()
    return tuple(float(value) for value in probs)


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
    best_phi = 0.0
    best_error = float("inf")
    search_lo = -np.pi
    search_hi = np.pi
    for _pass in range(3):
        for phi in np.linspace(search_lo, search_hi, grid_size, dtype=np.float32):
            probs = np.asarray(
                _two_qubit_probs(theta_i, theta_j, alpha_i, alpha_j, float(phi)),
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
