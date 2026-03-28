"""Random walk primitives and stationary measure for simple random walk."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class RandomWalkConfig:
    """Simple random walk: each step picks a uniform random neighbor."""

    graph: nx.Graph
    rng: np.random.Generator


def stationary_distribution_degree(G: nx.Graph) -> dict:
    """Return π(v) = deg(v) / (2|E|) for each node (connected undirected graph)."""
    m = G.number_of_edges()
    if m == 0:
        return {n: 1.0 / max(G.number_of_nodes(), 1) for n in G.nodes()}
    inv2m = 1.0 / (2.0 * m)
    return {n: G.degree(n) * inv2m for n in G.nodes()}


def simple_random_walk(
    G: nx.Graph,
    start,
    steps: int,
    rng: np.random.Generator,
) -> list:
    """Return node trajectory [X_0, ..., X_steps]."""
    adj = {n: list(G.neighbors(n)) for n in G.nodes()}
    out = [start]
    cur = start
    for _ in range(steps):
        nbs = adj[cur]
        if not nbs:
            break
        cur = nbs[rng.integers(len(nbs))]
        out.append(cur)
    return out


def batch_random_steps(
    G: nx.Graph,
    walkers: list,
    rng: np.random.Generator,
) -> list:
    """Advance each walker by one SRW step. Returns new positions."""
    adj = {n: list(G.neighbors(n)) for n in G.nodes()}
    new_w = []
    for w in walkers:
        nbs = adj[w]
        if not nbs:
            new_w.append(w)
            continue
        new_w.append(nbs[rng.integers(len(nbs))])
    return new_w
