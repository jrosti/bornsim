# bornsim

bornsim is an adjoint-differentiable statevector simulator for Born machine (parameterized quantum circuit) training, optimized for single NVIDIA GPU hardware. It supports up to 29 qubits exact statevector training on 24 GB of GPU memory and runs at near-peak DRAM bandwidth on RTX 3090. The library provides circuit construction with arbitrary topology and a king-move helper for 2D grids, HNSW-warm-start initialization from empirical data statistics, KL divergence training with Adam, and gradient verification tests against PennyLane references.

For the specific workload this repository targets, the comparison point is exact statevector simulation with a full probability vector and an analytic gradient of a scalar loss built on top of those probabilities. On a 4-neighbor rectangular grid with synthetic full-probability KL, measured median times on the local RTX 3090 were:
Across the measured sweep, runtime scaled close to linearly with depth while memory stayed effectively flat at fixed qubit count. At the large end of the measured range, `Q=28, L=48` took about `24.3s` forward, `71.6s` backward, `96.9s` total, and used about `16.8 GiB` of GPU memory.

![Execution-Time Comparison](examples/performance_test/readme_execution_time.png)

As an apples-to-oranges reference point, a Qiskit Aer shot-based estimator was also tested on an easier expectation-value task (`sum_i Z_i`) at `Q=10, L=6`. Even there, `10^6` shots still gave about `7.8e-4` run-to-run standard deviation and about `1.8e-4` mean absolute error versus the exact expectation, which is useful for observables but not a substitute for exact full-probability gradients.

A transparent lower-bound memory-efficiency estimate at `Q=28` is `payload_bytes_min / peak_bytes`, with `payload_bytes_min = 2 * state_bytes + prob_bytes`. For this workload the lower bound is about `5.0 GiB / 16.4 GiB = 30.5%`.

The main engineering choices behind the gap are:

- manual adjoint reverse pass instead of generic parameter-shift
- no autodiff tape over the full probability vector
- specialized diagonal `RZ` and `RZZ` kernels instead of routing everything through generic dense gate application
- depth-flat adjoint memory use
- fixed circuit family and topology, which removes framework overhead that matters at large `2^Q` state sizes

Install:
```
pip install -e .
```

Generate data (downloads MNIST, binarizes to 5x5 V6_otsu encoding):
```
python examples/generate_v6_otsu.py --output-dir ./data/V6_otsu
```

Train Born machine at L=6 for 500 steps:
```
python examples/train_l6_500.py --depth 6 --steps 500 --data-path ./data/V6_otsu
```

Run gradient agreement tests against PennyLane reference:
```
pip install -e .[test]
pytest tests/
```

Re-run the comparison harnesses:
```
python examples/performance_test/backend_alternatives_probe.py
python examples/performance_test/estimator_side_probe.py
python examples/performance_test/bornsim_readme_sweep.py
python examples/performance_test/build_readme_timing_figure.py
```
