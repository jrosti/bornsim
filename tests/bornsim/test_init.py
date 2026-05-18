"""Tests for bornsim initialization helpers."""

from __future__ import annotations

import numpy as np

from bornsim.circuit import Circuit
from bornsim.init import fit_rzz_angle, random_init, warm_start


def test_warm_start_ry_angles_match_target_marginals() -> None:
    circuit = Circuit(n_qubits=4, n_layers=2, edges=((0, 1), (1, 2), (2, 3)))
    marginals = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    corr = np.eye(4, dtype=np.float64)
    params = warm_start(circuit, marginals, corr, noise_scale=0.0)
    recovered = np.sin(params[:4] / 2.0) ** 2
    np.testing.assert_allclose(recovered, marginals, atol=1e-6)


def test_fit_rzz_angle_returns_small_value_for_zero_correlation() -> None:
    phi = fit_rzz_angle(0.0, 0.6, 0.8, float(np.pi / 8.0), float(np.pi / 8.0))
    assert abs(phi) < 1e-6


def test_random_init_is_reproducible_and_scaled() -> None:
    circuit = Circuit(n_qubits=6, n_layers=2, edges=((0, 1), (1, 2)))
    params_a = random_init(circuit, seed=7, scale=0.2)
    params_b = random_init(circuit, seed=7, scale=0.2)
    np.testing.assert_array_equal(params_a, params_b)
    assert abs(float(params_a.mean())) < 0.1
    assert 0.1 < float(params_a.std()) < 0.3
