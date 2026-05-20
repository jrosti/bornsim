"""Small apples-to-oranges Qiskit estimator probe on an expectation task."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import grid_shape, qiskit_born_ansatz
from bornsim.topology import grid_coupling_map_rect


@dataclass(frozen=True, slots=True)
class EstimatorRow:
    backend: str
    setting_name: str
    setting_value: float
    mean_seconds: float
    std_estimate: float
    abs_error_vs_exact: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/estimator_side_probe"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--n-layers", type=int, default=6)
    return parser.parse_args()


def observable_sum_z(n_qubits: int) -> SparsePauliOp:
    return SparsePauliOp.from_list(
        [("".join("Z" if i == wire else "I" for i in range(n_qubits)), 1.0) for wire in range(n_qubits)]
    )


def main() -> int:
    args = parse_args()
    rows, cols = grid_shape(args.n_qubits)
    coupling_map = grid_coupling_map_rect(rows, cols, connectivity=4)
    circuit, parameters = qiskit_born_ansatz(n_qubits=args.n_qubits, n_layers=args.n_layers, coupling_map=coupling_map)
    param_values = np.random.default_rng(args.seed).normal(0.0, 0.1, size=(len(parameters),)).astype(np.float64)
    observable = observable_sum_z(args.n_qubits)
    exact_value = float(StatevectorEstimator(default_precision=0.0, seed=args.seed).run([(circuit, observable, param_values)]).result()[0].data.evs)

    rows_out: list[EstimatorRow] = []
    for precision in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        estimates = []
        timings = []
        for repeat in range(args.repeats):
            estimator = StatevectorEstimator(default_precision=precision, seed=repeat)
            started = perf_counter()
            estimates.append(float(estimator.run([(circuit, observable, param_values)]).result()[0].data.evs))
            timings.append(perf_counter() - started)
        rows_out.append(
            EstimatorRow("qiskit_statevector_estimator", "precision", precision, float(np.mean(timings)), float(np.std(estimates)), float(abs(np.mean(estimates) - exact_value)))
        )
    for shots in (10**2, 10**3, 10**4, 10**5, 10**6):
        estimates = []
        timings = []
        for _ in range(args.repeats):
            estimator = AerEstimator(backend_options={"device": "GPU", "method": "statevector", "precision": "single"}, run_options={"shots": int(shots)})
            started = perf_counter()
            estimates.append(float(estimator.run(circuit, observable, param_values).result().values[0]))
            timings.append(perf_counter() - started)
        rows_out.append(
            EstimatorRow("qiskit_aer_estimator", "shots", float(shots), float(np.mean(timings)), float(np.std(estimates)), float(abs(np.mean(estimates) - exact_value)))
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "results.json").write_text(
        json.dumps({"exact_value": exact_value, "rows": [asdict(row) for row in rows_out]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
