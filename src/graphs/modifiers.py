"""Graph modifiers: add long-range (Lévy) edges to an existing graph."""

from __future__ import annotations

import networkx as nx
import numpy as np


def add_levy_edges(
    G: nx.Graph,
    pos: dict,
    alpha: float = 1.5,
    p_long: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[nx.Graph, int]:
    """
    Return a copy of G augmented with long-range edges drawn from a power-law kernel.

    For each pair (u, v) not already connected, an edge is added with probability::

        p(u, v) = p_long * (d_nn / d(u, v))^(alpha + 2)

    where d_nn is the median nearest-neighbour edge length (the natural length unit of
    the substrate) and d(u, v) is the Euclidean distance in the embedded positions.

    With this kernel, the probability that a simple random walk step on the augmented
    graph covers distance > r decays as r^{-alpha} — the defining property of a Lévy
    flight with index alpha.

    Parameters
    ----------
    G : nx.Graph
        Base graph (returned unchanged; a copy is augmented).
    pos : dict
        node -> (x, y) positions used to compute pairwise Euclidean distances.
    alpha : float
        Lévy exponent. Must be in (0, 4). Values in (0, 2) give the classic
        heavy-tailed Lévy regime; alpha = 2 recovers Gaussian-like diffusion.
    p_long : float
        Probability of adding an edge at distance d_nn (the nearest-neighbour scale).
        Controls the overall density of long-range edges independently of alpha.
    rng : numpy Generator, optional

    Returns
    -------
    H : nx.Graph
        Copy of G with additional long-range edges.
    n_added : int
        Number of new edges added.
    """
    if not 0 < alpha < 4:
        raise ValueError("alpha must be in (0, 4)")
    if not 0.0 <= p_long <= 1.0:
        raise ValueError("p_long must be in [0, 1]")
    rng = rng or np.random.default_rng()

    nodes = list(G.nodes())
    n = len(nodes)
    if n < 2:
        return G.copy(), 0

    # Median existing edge length sets the natural length scale d_nn.
    nn_dists = []
    for u, v in G.edges():
        pu = np.array(pos[u], dtype=float)
        pv = np.array(pos[v], dtype=float)
        nn_dists.append(float(np.linalg.norm(pu - pv)))
    d_nn = float(np.median(nn_dists)) if nn_dists else 1.0

    coords = np.array([pos[v] for v in nodes], dtype=float)  # (n, 2)
    existing = set(G.edges()) | {(v, u) for u, v in G.edges()}

    H = G.copy()
    n_added = 0
    exponent = alpha + 2.0

    # O(N^2) — manageable for the graph sizes used in this project (< ~3 000 nodes).
    for i in range(n):
        for j in range(i + 1, n):
            u, v = nodes[i], nodes[j]
            if (u, v) in existing or (v, u) in existing:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < 1e-12:
                continue
            p_edge = p_long * (d_nn / d) ** exponent
            if p_edge >= 1.0 or rng.random() < p_edge:
                H.add_edge(u, v)
                n_added += 1

    return H, n_added
