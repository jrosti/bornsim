"""Train the 25-qubit L=6 V6_otsu Born-machine example."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import optax

from bornsim import Circuit, init, losses
from bornsim.topology import king_coupling_map
from bornsim.trainer import Trainer

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax-cache")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", type=Path, default=Path("./data/V6_otsu"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    return parser.parse_args()


def _load_train_images(data_path: Path) -> np.ndarray:
    train_path = data_path / "train.npz"
    if not train_path.exists():
        raise FileNotFoundError(
            f"missing {train_path}; run examples/generate_v6_otsu.py or provide train.npz"
        )
    with np.load(train_path) as data:
        if "images" in data:
            images = data["images"]
        elif "X_bin" in data:
            images = data["X_bin"]
        else:
            raise KeyError("train.npz must contain an 'images' or 'X_bin' array")
    flat = np.asarray(images, dtype=np.uint8).reshape((images.shape[0], -1))
    if flat.shape[1] != 25:
        raise ValueError(f"expected 25 binary features, got shape {flat.shape}")
    return flat


def _empirical_bitstring_distribution(x_bin: np.ndarray) -> np.ndarray:
    powers = (1 << np.arange(x_bin.shape[1] - 1, -1, -1, dtype=np.uint32)).astype(np.uint32)
    indices = x_bin.astype(np.uint32) @ powers
    counts = np.bincount(indices, minlength=2 ** x_bin.shape[1]).astype(np.float32)
    return np.asarray(counts / counts.sum(), dtype=np.float32)


def _pairwise_correlation(x_bin: np.ndarray) -> np.ndarray:
    x = np.asarray(x_bin, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True, dtype=np.float64)
    variances = np.mean(centered * centered, axis=0, dtype=np.float64)
    scales = np.sqrt(np.maximum(variances, 1e-12), dtype=np.float64)
    normalized = centered / scales
    corr = normalized.T @ normalized / float(x.shape[0])
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def main() -> int:
    args = _parse_args()
    try:
        train_images = _load_train_images(args.data_path)
    except Exception as exc:
        raise SystemExit(f"failed to load V6_otsu data: {exc}") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    marginals = train_images.mean(axis=0, dtype=np.float64).astype(np.float32)
    correlations = _pairwise_correlation(train_images)
    p_data = _empirical_bitstring_distribution(train_images)

    circuit = Circuit(n_qubits=25, n_layers=args.depth, edges=tuple(king_coupling_map(5, 5)))
    params = init.warm_start(circuit, marginals, correlations, seed=args.seed, noise_scale=0.05)
    trainer = Trainer(circuit, losses.KL(p_data, eps=1e-12), optax.adam(args.lr))
    opt_state = trainer.init_opt_state(params)

    metrics: list[dict[str, float | int]] = []
    started = perf_counter()
    for step in range(1, args.steps + 1):
        step_started = perf_counter()
        params, opt_state, info = trainer.step(params, opt_state)
        row = {
            "step": step,
            "loss_value": float(info["loss_value"]),
            "grad_norm": float(info["grad_norm"]),
            "prob_sum": float(info["prob_sum"]),
            "forward_seconds": float(info["forward_seconds"]),
            "backward_seconds": float(info["backward_seconds"]),
            "step_seconds": perf_counter() - step_started,
        }
        metrics.append(row)
        if step % 10 == 0 or step == 1 or step == args.steps:
            print(
                f"step={step:4d} train_nll={row['loss_value']:.6f} "
                f"grad_norm={row['grad_norm']:.6f}",
                flush=True,
            )

    total_seconds = perf_counter() - started
    np.savez_compressed(args.output_dir / "final_params.npz", params=params)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    summary = {
        "depth": args.depth,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "n_params": circuit.n_params,
        "final_loss": metrics[-1]["loss_value"] if metrics else None,
        "final_grad_norm": metrics[-1]["grad_norm"] if metrics else None,
        "total_seconds": total_seconds,
    }
    (args.output_dir / "summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
