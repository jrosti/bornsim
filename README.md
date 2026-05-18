# bornsim

bornsim is an adjoint-differentiable statevector simulator for Born machine (parameterized quantum circuit) training, optimized for single NVIDIA GPU hardware. It supports up to 29 qubits exact statevector training on 24 GB of GPU memory and runs at near-peak DRAM bandwidth on RTX 3090. The library provides circuit construction with arbitrary topology and a king-move helper for 2D grids, HNSW-warm-start initialization from empirical data statistics, KL divergence training with Adam, and gradient verification tests against PennyLane references.

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
