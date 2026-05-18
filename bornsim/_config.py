"""Internal bornsim configuration.

This module centralizes CUDA device selection and default numeric precision
for the cuStateVec backend.
"""

from __future__ import annotations

import cupy as cp

COMPLEX_DTYPE = cp.complex64
REAL_DTYPE = cp.float32


def get_device() -> cp.cuda.Device:
    """Return the active CuPy device handle."""
    return cp.cuda.Device()


def set_precision_for_testing(*, complex128: bool) -> None:
    """Switch module-level dtypes for test-only verification paths."""
    # ruff: noqa: PLW0603
    global COMPLEX_DTYPE
    global REAL_DTYPE
    if complex128:
        COMPLEX_DTYPE = cp.complex128
        REAL_DTYPE = cp.float64
    else:
        COMPLEX_DTYPE = cp.complex64
        REAL_DTYPE = cp.float32
