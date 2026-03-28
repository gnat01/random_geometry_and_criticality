"""Vicsek cross fractal as a 4-neighbor graph (5 subsquares kept each scale)."""

from __future__ import annotations

import networkx as nx

from .common import grid_positions_square, largest_connected_subgraph

# Keep the cross pattern: center + edge-mid blocks.
# This yields a connected Vicsek-style geometry under 4-neighbor adjacency.
_ALLOWED = {(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)}


def _in_vicsek(i: int, j: int) -> bool:
    while i > 0 or j > 0:
        if (i % 3, j % 3) not in _ALLOWED:
            return False
        i //= 3
        j //= 3
    return True


def build_vicsek(depth: int) -> tuple[nx.Graph, dict, int]:
    """
    Vicsek set on [0, 3^depth)^2 with 4-neighbor edges.

    Returns G, pos_draw, side_length for normalization (3^depth - 1).
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    side = 3**depth
    nodes = [(i, j) for i in range(side) for j in range(side) if _in_vicsek(i, j)]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    s = set(nodes)
    for i, j in nodes:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (i + di, j + dj)
            if nb in s:
                G.add_edge((i, j), nb)

    G = largest_connected_subgraph(G)
    nodes = list(G.nodes())
    denom = float(side - 1) if side > 1 else 1.0
    pos = grid_positions_square(nodes, denom)
    return G, pos, side - 1
