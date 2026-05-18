"""Tests for bornsim probability utilities."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from bornsim import utils
from tests.bornsim.conftest import gpu_only


@gpu_only
def test_marginals_match_hand_computed_values() -> None:
    probs = cp.asarray([0.05, 0.1, 0.15, 0.2, 0.1, 0.1, 0.1, 0.2], dtype=cp.float32)
    marginal = utils.marginals(probs, 3, [0, 2])
    np.testing.assert_allclose(
        cp.asnumpy(marginal),
        np.array([0.2, 0.3, 0.2, 0.3], dtype=np.float32),
    )


@gpu_only
def test_conditional_normalizes_and_raises_on_zero_mass() -> None:
    probs = cp.asarray([0.05, 0.1, 0.15, 0.2, 0.1, 0.1, 0.1, 0.2], dtype=cp.float32)
    conditional = utils.conditional(probs, 3, [0], [1])
    np.testing.assert_allclose(float(cp.sum(conditional).get()), 1.0, atol=1e-6)
    zero_probs = cp.asarray([1.0, 0.0, 0.0, 0.0], dtype=cp.float32)
    with pytest.raises(ValueError):
        utils.conditional(zero_probs, 2, [0], [1])


@gpu_only
def test_to_numpy_round_trip() -> None:
    probs = cp.asarray([0.2, 0.3, 0.5, 0.0], dtype=cp.float32)
    np.testing.assert_allclose(
        utils.to_numpy(probs),
        np.array([0.2, 0.3, 0.5, 0.0], dtype=np.float32),
    )
