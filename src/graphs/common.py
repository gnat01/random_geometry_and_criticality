"""Shared helpers for 2D lattice graphs and layouts."""

from __future__ import annotations

import networkx as nx
import numpy as np


def largest_connected_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    ccs = list(nx.connected_components(G))
    largest = max(ccs, key=len)
    return G.subgraph(largest).copy()


def grid_positions_square(nodes: list[tuple[int, int]], scale: float) -> dict:
    """Map integer lattice nodes (i, j) to [0,1]^2 for plotting."""
    pos = {}
    for n in nodes:
        i, j = n
        pos[n] = np.array([float(i) / scale, float(j) / scale], dtype=float)
    return pos


def mean_degree(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return float(sum(dict(G.degree()).values())) / n
