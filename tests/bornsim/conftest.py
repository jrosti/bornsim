"""Shared pytest helpers for bornsim GPU tests."""

from __future__ import annotations

import cupy as cp
import pytest


def has_cuda() -> bool:
    """Return whether a CUDA device is available to CuPy."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


gpu_only = pytest.mark.skipif(not has_cuda(), reason="bornsim tests require CUDA")
