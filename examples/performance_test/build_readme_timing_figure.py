"""Build the README execution-time figure from measured comparison artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SWEEP_JSON = RESULTS_DIR / "bornsim_readme_sweep_results.json"
TENSOR_JSON = RESULTS_DIR / "tensorcircuit_spotcheck.json"
OUTPUT_PNG = HERE / "readme_execution_time.png"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    sweep_rows = load_json(SWEEP_JSON)
    tensor_rows = load_json(TENSOR_JSON)

    plt.rcParams.update(
        {
            "figure.figsize": (11, 4.8),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
        }
    )

    fig, (ax0, ax1) = plt.subplots(1, 2)

    qubits = sorted({row["n_qubits"] for row in sweep_rows})
    for q in qubits:
        rows = sorted(
            [row for row in sweep_rows if row["n_qubits"] == q],
            key=lambda row: row["n_layers"],
        )
        layers = [row["n_layers"] for row in rows]
        totals = [row["median_total_seconds"] for row in rows]
        ax0.plot(layers, totals, marker="o", linewidth=2.2, label=f"Q={q}")

    ax0.set_yscale("log")
    ax0.set_xlabel("Layers L")
    ax0.set_ylabel("Median Total Time (s)")
    ax0.set_title("bornsim Full-Probability KL Sweep")
    ax0.legend(frameon=False, ncols=2)

    labels = [f"Q{row['n_qubits']} L{row['n_layers']}" for row in tensor_rows]
    bornsim_totals = []
    for row in tensor_rows:
        match = next(
            item
            for item in sweep_rows
            if item["n_qubits"] == row["n_qubits"] and item["n_layers"] == row["n_layers"]
        )
        bornsim_totals.append(match["median_total_seconds"])
    tensor_totals = [row["total_seconds"] for row in tensor_rows]

    x = list(range(len(labels)))
    width = 0.38
    ax1.bar([i - width / 2 for i in x], bornsim_totals, width=width, label="bornsim")
    ax1.bar([i + width / 2 for i in x], tensor_totals, width=width, label="TensorCircuit")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Total Time (s)")
    ax1.set_title("Tiny Shared Cases")
    ax1.legend(frameon=False)

    fig.suptitle("Execution-Time Comparison")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
