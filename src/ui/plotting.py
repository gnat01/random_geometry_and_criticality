"""Matplotlib helpers for drawing graphs and small multiples."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..theme import BG, EDGE_C, NODE_C, TEXT_C


def draw_graph_2d(
    G: nx.Graph,
    pos: dict,
    ax=None,
    node_size: float = 6.0,
    edge_alpha: float = 0.45,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    segs_x, segs_y = [], []
    for u, v in G.edges():
        segs_x += [pos[u][0], pos[v][0], None]
        segs_y += [pos[u][1], pos[v][1], None]
    ax.plot(segs_x, segs_y, color=EDGE_C, alpha=edge_alpha, lw=0.6, zorder=1)
    xy = np.array([pos[n] for n in G.nodes()])
    ax.scatter(xy[:, 0], xy[:, 1], s=node_size, c=NODE_C, alpha=0.55, linewidths=0, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def pos_walk_square(nodes):
    """Integer lattice embedding for MSD (same metric as drawing up to scale)."""
    return {n: np.array([float(n[0]), float(n[1])], dtype=float) for n in nodes}
