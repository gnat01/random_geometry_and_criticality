"""Sierpiński carpet as a 4-neighbor grid graph (sites survive the usual ternary rule)."""

from __future__ import annotations

import networkx as nx

from .common import grid_positions_square, largest_connected_subgraph


def _in_carpet(i: int, j: int) -> bool:
    """True iff (i,j) is not in any removed middle-third square at any scale."""
    while i > 0 or j > 0:
        if i % 3 == 1 and j % 3 == 1:
            return False
        i //= 3
        j //= 3
    return True


def build_carpet(depth: int) -> tuple[nx.Graph, dict, int]:
    """
    Build carpet on [0, 3^depth)^2 integer lattice with 4-connectivity.

    Returns
    -------
    G, pos_draw, side
        pos_draw maps node -> ndarray [x,y] in [0,1]^2; side = 3^depth - 1 scale for layout.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    side = 3**depth
    nodes = [(i, j) for i in range(side) for j in range(side) if _in_carpet(i, j)]
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
    pos = grid_positions_square(nodes, float(side - 1) if side > 1 else 1.0)
    return G, pos, side - 1
