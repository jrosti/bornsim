"""Generate the 5x5 V6_otsu MNIST encoding used by the example trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from sklearn.datasets import fetch_openml

MNIST_SIDE = 28


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("./data/V6_otsu"))
    return parser.parse_args()


def _crop_to_digit_bbox(image: np.ndarray, *, threshold: float = 0.20) -> np.ndarray:
    mask = image > threshold
    if not np.any(mask):
        return image
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return image[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def _resize_bilinear(image: np.ndarray, *, side: int = 5) -> np.ndarray:
    factors = (side / image.shape[0], side / image.shape[1])
    return np.asarray(zoom(image, factors, order=1, mode="nearest"), dtype=np.float32)


def _otsu_threshold(values: np.ndarray, *, n_bins: int = 256) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    hist, edges = np.histogram(flat, bins=n_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64, copy=False)
    centers = (edges[:-1] + edges[1:]) * 0.5
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    denom = omega * (1.0 - omega)
    sigma_b_sq = np.zeros_like(denom)
    valid = denom > 0.0
    sigma_b_sq[valid] = ((mu[-1] * omega[valid] - mu[valid]) ** 2) / denom[valid]
    best_value = float(sigma_b_sq.max())
    best_indices = np.flatnonzero(sigma_b_sq >= best_value - max(best_value * 1e-12, 1e-12))
    return float(centers[best_indices].mean())


def _preprocess(images: np.ndarray) -> np.ndarray:
    resized = [
        _resize_bilinear(_crop_to_digit_bbox(np.asarray(image, dtype=np.float32)), side=5)
        for image in images
    ]
    return np.stack(resized, axis=0).astype(np.float32, copy=False)


def _save_split(path: Path, images: np.ndarray, labels: np.ndarray) -> None:
    np.savez_compressed(path, images=images, labels=labels)


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    images = np.asarray(mnist.data, dtype=np.float32).reshape(-1, MNIST_SIDE, MNIST_SIDE) / 255.0
    labels = np.asarray(mnist.target, dtype=np.int32)

    train_downsampled = _preprocess(images[:55000])
    val_downsampled = _preprocess(images[55000:60000])
    test_downsampled = _preprocess(images[60000:70000])
    threshold = np.asarray([_otsu_threshold(train_downsampled)], dtype=np.float32)

    _save_split(args.output_dir / "train.npz", (train_downsampled > threshold[0]).astype(np.uint8), labels[:55000])
    _save_split(args.output_dir / "val.npz", (val_downsampled > threshold[0]).astype(np.uint8), labels[55000:60000])
    _save_split(args.output_dir / "test.npz", (test_downsampled > threshold[0]).astype(np.uint8), labels[60000:70000])
    np.savez_compressed(args.output_dir / "metadata.npz", threshold=threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
