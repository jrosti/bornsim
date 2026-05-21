"""Run the README timing sweep on synthetic full-probability KL targets.

Example:
    python examples/performance_test/bornsim_readme_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax-cache")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cupy as cp
import numpy as np
import optax

from bornsim import Circuit, init, losses
from bornsim.topology import grid_coupling_map_rect
from bornsim.trainer import Trainer

from _common import grid_shape, target_distribution


@dataclass(frozen=True, slots=True)
class SweepRow:
    n_qubits: int
    n_layers: int
    n_edges: int
    n_params: int
    repeats: int
    median_forward_seconds: float
    median_backward_seconds: float
    median_total_seconds: float
    mean_total_seconds: float
    std_total_seconds: float
    median_gpu_used_mib: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/readme_sweep"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qubits", type=str, default="10,15,20,28")
    parser.add_argument("--layers", type=str, default="6,12,24,48")
    parser.add_argument("--default-repeats", type=int, default=30)
    parser.add_argument("--heavy-repeats", type=int, default=10)
    return parser.parse_args()


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def gpu_used_mib() -> float:
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return float((total_bytes - free_bytes) / (1024.0 * 1024.0))


def cleanup_cuda_memory() -> None:
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def repeats_for(n_qubits: int, n_layers: int, default_repeats: int, heavy_repeats: int) -> int:
    if n_qubits == 28 and n_layers in {24, 48}:
        return heavy_repeats
    return default_repeats


def run_one(*, n_qubits: int, n_layers: int, seed: int, repeats: int) -> SweepRow:
    rows, cols = grid_shape(n_qubits)
    edges = tuple(grid_coupling_map_rect(rows, cols, connectivity=4))
    target = target_distribution(n_qubits, seed)
    circuit = Circuit(n_qubits=n_qubits, n_layers=n_layers, edges=edges)
    params = init.random_init(circuit, seed=seed, scale=0.1)
    trainer = Trainer(circuit, losses.KL(target, eps=1e-12), optax.adam(1e-2))

    _ = trainer._gradient_info(params)
    cp.cuda.Stream.null.synchronize()

    forward_times: list[float] = []
    backward_times: list[float] = []
    total_times: list[float] = []
    gpu_used: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        info = trainer._gradient_info(params)
        total = perf_counter() - started
        cp.cuda.Stream.null.synchronize()
        forward_times.append(float(info.forward_seconds))
        backward_times.append(float(info.backward_seconds))
        total_times.append(float(total))
        gpu_used.append(gpu_used_mib())

    return SweepRow(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_edges=len(edges),
        n_params=circuit.n_params,
        repeats=repeats,
        median_forward_seconds=float(np.median(forward_times)),
        median_backward_seconds=float(np.median(backward_times)),
        median_total_seconds=float(np.median(total_times)),
        mean_total_seconds=float(np.mean(total_times)),
        std_total_seconds=float(np.std(total_times)),
        median_gpu_used_mib=float(np.median(gpu_used)),
    )


def write_results(results_dir: Path, rows: list[SweepRow]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["# README sweep", ""]
    lines.append("| Q | L | repeats | forward | backward | total | GPU used MiB |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row.n_qubits}` | `{row.n_layers}` | `{row.repeats}` | "
            f"`{row.median_forward_seconds:.4f}s` | `{row.median_backward_seconds:.4f}s` | "
            f"`{row.median_total_seconds:.4f}s` | `{row.median_gpu_used_mib:.1f}` |"
        )
    (results_dir / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows_out: list[SweepRow] = []
    for n_qubits in parse_csv_ints(args.qubits):
        for n_layers in parse_csv_ints(args.layers):
            repeats = repeats_for(n_qubits, n_layers, args.default_repeats, args.heavy_repeats)
            row = run_one(n_qubits=n_qubits, n_layers=n_layers, seed=args.seed, repeats=repeats)
            rows_out.append(row)
            write_results(args.results_dir, rows_out)
            cleanup_cuda_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
