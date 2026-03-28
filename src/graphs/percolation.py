"""Bond percolation on a 2D grid; return the largest connected component."""

from __future__ import annotations

import networkx as nx
import numpy as np

from .common import grid_positions_square, largest_connected_subgraph


def build_percolation(
    size: int,
    p_open: float,
    rng: np.random.Generator | None = None,
) -> tuple[nx.Graph, dict, int]:
    """
    Start from an (size x size) grid with 4-neighbors; keep each edge with prob p_open.

    Parameters
    ----------
    size : int
        Linear extent (number of nodes along each axis).
    p_open : float
        Bond occupation probability in [0, 1].
    rng : numpy Generator, optional
        Random source for reproducibility.

    Returns
    -------
    G, pos_draw, side
        Largest connected component; positions in [0,1]^2; side = size-1 for layout.
    """
    if size < 2:
        raise ValueError("size must be >= 2")
    if not 0.0 <= p_open <= 1.0:
        raise ValueError("p_open must be in [0, 1]")
    rng = rng or np.random.default_rng()

    G0 = nx.grid_2d_graph(size, size)
    H = nx.Graph()
    H.add_nodes_from(G0.nodes())
    for u, v in G0.edges():
        if rng.random() < p_open:
            H.add_edge(u, v)

    G = largest_connected_subgraph(H)
    nodes = list(G.nodes())
    denom = float(size - 1) if size > 1 else 1.0
    pos = grid_positions_square(nodes, denom)
    return G, pos, size - 1
