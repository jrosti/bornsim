"""Topology helpers for Born-machine circuits."""

from __future__ import annotations


def grid_coupling_map_rect(rows: int, cols: int, *, connectivity: int = 4) -> list[tuple[int, int]]:
    """Return 2D grid edges in row-major qubit order."""
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if col + 1 < cols:
                edges.append((node, node + 1))
            if row + 1 < rows:
                edges.append((node, node + cols))
            if connectivity == 8 and row + 1 < rows and col + 1 < cols:
                edges.append((node, node + cols + 1))
            if connectivity == 8 and row + 1 < rows and col - 1 >= 0:
                edges.append((node, node + cols - 1))
    return edges


def king_coupling_map(rows: int, cols: int) -> list[tuple[int, int]]:
    """Return king-move, 8-neighbor grid edges in row-major qubit order."""
    return grid_coupling_map_rect(rows, cols, connectivity=8)
