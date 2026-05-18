"""Custom CUDA kernels for cuStateVec extensions.

Implements fused generator-reduction kernels for RY, RZ, and RZZ gradients,
plus diagonal apply kernels for RZ and RZZ forward/adjoint application.
"""

from __future__ import annotations

import cupy as cp

CUDA_KERNELS = r"""
#include <cuComplex.h>

extern "C" {

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float shared[256];
    shared[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void grad_rz_reduce(
    const cuFloatComplex* lam,
    const cuFloatComplex* psi,
    int bit,
    unsigned long long n_amplitudes,
    float* out
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    float contribution = 0.0f;
    if (i < n_amplitudes) {
        int sign = ((i >> bit) & 1ULL) ? -1 : 1;
        cuFloatComplex product = cuCmulf(cuConjf(lam[i]), psi[i]);
        contribution = cuCimagf(product) * sign;
    }
    float sum = block_sum(contribution);
    if (threadIdx.x == 0) {
        atomicAdd(out, sum);
    }
}

__global__ void grad_rzz_reduce(
    const cuFloatComplex* lam,
    const cuFloatComplex* psi,
    int bit_a,
    int bit_b,
    unsigned long long n_amplitudes,
    float* out
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    float contribution = 0.0f;
    if (i < n_amplitudes) {
        int bit_a_value = (i >> bit_a) & 1ULL;
        int bit_b_value = (i >> bit_b) & 1ULL;
        int sign = (bit_a_value == bit_b_value) ? 1 : -1;
        cuFloatComplex product = cuCmulf(cuConjf(lam[i]), psi[i]);
        contribution = cuCimagf(product) * sign;
    }
    float sum = block_sum(contribution);
    if (threadIdx.x == 0) {
        atomicAdd(out, sum);
    }
}

__global__ void grad_ry_reduce(
    const cuFloatComplex* lam,
    const cuFloatComplex* psi,
    int bit,
    unsigned long long n_pairs,
    float* out
) {
    unsigned long long pair_index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    float contribution = 0.0f;
    if (pair_index < n_pairs) {
        unsigned long long low_mask = (1ULL << bit) - 1ULL;
        unsigned long long low = pair_index & low_mask;
        unsigned long long high = pair_index >> bit;
        unsigned long long i0 = (high << (bit + 1)) | low;
        unsigned long long i1 = i0 | (1ULL << bit);

        cuFloatComplex term_a = cuCmulf(cuConjf(lam[i1]), psi[i0]);
        cuFloatComplex term_b = cuCmulf(cuConjf(lam[i0]), psi[i1]);
        contribution = cuCrealf(cuCsubf(term_a, term_b));
    }
    float sum = block_sum(contribution);
    if (threadIdx.x == 0) {
        atomicAdd(out, sum);
    }
}

__global__ void apply_rz_diagonal(
    cuFloatComplex* state,
    int bit,
    cuFloatComplex phase_zero,
    cuFloatComplex phase_one,
    unsigned long long n_amplitudes
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_amplitudes) {
        return;
    }
    state[i] = cuCmulf(state[i], ((i >> bit) & 1ULL) ? phase_one : phase_zero);
}

__global__ void apply_rzz_diagonal(
    cuFloatComplex* state,
    int bit_a,
    int bit_b,
    cuFloatComplex phase_same,
    cuFloatComplex phase_diff,
    unsigned long long n_amplitudes
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_amplitudes) {
        return;
    }
    int bit_a_value = (i >> bit_a) & 1ULL;
    int bit_b_value = (i >> bit_b) & 1ULL;
    state[i] = cuCmulf(state[i], bit_a_value == bit_b_value ? phase_same : phase_diff);
}

}
"""

KERNEL_THREADS = 256
GRAD_RZ_KERNEL = cp.RawKernel(CUDA_KERNELS, "grad_rz_reduce")
GRAD_RZZ_KERNEL = cp.RawKernel(CUDA_KERNELS, "grad_rzz_reduce")
GRAD_RY_KERNEL = cp.RawKernel(CUDA_KERNELS, "grad_ry_reduce")
APPLY_RZ_KERNEL = cp.RawKernel(CUDA_KERNELS, "apply_rz_diagonal")
APPLY_RZZ_KERNEL = cp.RawKernel(CUDA_KERNELS, "apply_rzz_diagonal")


def launch_dims(n_items: int) -> tuple[tuple[int], tuple[int]]:
    """Return `(grid, block)` launch dimensions for 1D kernels."""
    blocks = (n_items + KERNEL_THREADS - 1) // KERNEL_THREADS
    return (blocks,), (KERNEL_THREADS,)
