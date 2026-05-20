"""Probe generic backends against the full-probability target workload."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from bornsim import Circuit, losses
from bornsim.topology import grid_coupling_map_rect
from bornsim.trainer import Trainer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import grid_shape, pennylane_apply_born_ansatz, qiskit_born_ansatz, target_distribution


@dataclass(frozen=True, slots=True)
class ProbeResult:
    backend: str
    status: str
    gradient_kind: str
    notes: str
    forward_seconds: float | None
    backward_seconds: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/backend_alternatives_probe"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--n-layers", type=int, default=6)
    return parser.parse_args()


def bornsim_probe(n_qubits: int, n_layers: int, seed: int) -> ProbeResult:
    import optax
    from bornsim import init

    rows, cols = grid_shape(n_qubits)
    edges = tuple(grid_coupling_map_rect(rows, cols, connectivity=4))
    target = target_distribution(n_qubits, seed)
    circuit = Circuit(n_qubits=n_qubits, n_layers=n_layers, edges=edges)
    params = init.random_init(circuit, seed=seed, scale=0.1)
    trainer = Trainer(circuit, losses.KL(target, eps=1e-12), optax.adam(1e-2))
    started = perf_counter()
    info = trainer._gradient_info(params)
    total = perf_counter() - started
    return ProbeResult(
        backend="bornsim",
        status="ok",
        gradient_kind="manual_adjoint_full_probs",
        notes="Reference implementation for the target workload.",
        forward_seconds=float(info.forward_seconds),
        backward_seconds=float(total - info.forward_seconds),
    )


def pennylane_probe(n_qubits: int, n_layers: int, seed: int) -> ProbeResult:
    import pennylane as qml
    from pennylane import numpy as pnp

    rows, cols = grid_shape(n_qubits)
    edges = grid_coupling_map_rect(rows, cols, connectivity=4)
    n_edges = len(edges)
    n_params = n_layers * (2 * n_qubits + n_edges) + 2 * n_qubits
    target = target_distribution(n_qubits, seed)
    params = np.random.default_rng(seed).normal(0.0, 0.1, size=(n_params,)).astype(np.float32)
    dev = qml.device("lightning.gpu", wires=n_qubits)

    @qml.qnode(dev, diff_method="adjoint")  # type: ignore[untyped-decorator]
    def probs_qnode(theta: pnp.ndarray) -> pnp.ndarray:
        pennylane_apply_born_ansatz(theta, n_qubits=n_qubits, n_layers=n_layers, coupling_map=edges)
        return qml.probs(wires=range(n_qubits))

    def objective(theta: pnp.ndarray) -> pnp.ndarray:
        probs = probs_qnode(theta)
        clipped = qml.math.clip(probs, 1e-12, 1.0)
        return -qml.math.sum(pnp.asarray(target) * qml.math.log(clipped))

    theta = pnp.asarray(params, dtype=np.float32, requires_grad=True)
    try:
        started = perf_counter()
        _ = probs_qnode(theta)
        forward = perf_counter() - started
        started = perf_counter()
        _ = qml.grad(objective)(theta)
        backward = perf_counter() - started
        return ProbeResult("pennylane_lightning_gpu", "ok", "adjoint_full_probs", "Adjoint supported on direct probability path.", forward, backward)
    except Exception as exc:
        return ProbeResult("pennylane_lightning_gpu", "blocked", "adjoint_rejected", f"full-probability adjoint failed: {type(exc).__name__}: {exc}", None, None)


def qiskit_probe(n_qubits: int, n_layers: int, seed: int) -> ProbeResult:
    from qiskit_aer import AerSimulator
    from qiskit_algorithms.gradients import ParamShiftSamplerGradient

    rows, cols = grid_shape(n_qubits)
    edges = grid_coupling_map_rect(rows, cols, connectivity=4)
    circuit, params = qiskit_born_ansatz(n_qubits=n_qubits, n_layers=n_layers, coupling_map=edges)
    values = np.random.default_rng(seed).normal(0.0, 0.1, size=(len(params),)).astype(np.float32)
    bound = circuit.assign_parameters({param: float(value) for param, value in zip(params, values, strict=True)})
    bound.save_probabilities()
    sim = AerSimulator(method="statevector", device="GPU", precision="single")
    started = perf_counter()
    _ = sim.run(bound).result()
    forward = perf_counter() - started
    _ = ParamShiftSamplerGradient(sim)
    return ProbeResult(
        "qiskit_aer",
        "partial",
        "sampler_parameter_shift_only",
        "GPU statevector forward works, but probability gradients are parameter-shift rather than reverse-mode/adjoint.",
        forward,
        None,
    )


def main() -> int:
    args = parse_args()
    rows = [
        asdict(bornsim_probe(args.n_qubits, args.n_layers, args.seed)),
        asdict(pennylane_probe(args.n_qubits, args.n_layers, args.seed)),
        asdict(qiskit_probe(args.n_qubits, args.n_layers, args.seed)),
    ]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
