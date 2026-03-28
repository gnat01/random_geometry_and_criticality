"""Random walk primitives: simple (SRW) and biased variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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


# ---------------------------------------------------------------------------
# Biased walk primitives
# ---------------------------------------------------------------------------

def biased_random_walk(
    G: nx.Graph,
    start,
    steps: int,
    weight_fn: Callable,
    rng: np.random.Generator,
) -> list:
    """
    Random walk where the probability of stepping to a neighbour v is proportional
    to weight_fn(current_node, v).

    Falls back to uniform SRW if all weights for a node are zero.
    """
    adj = {n: list(G.neighbors(n)) for n in G.nodes()}
    out = [start]
    cur = start
    for _ in range(steps):
        nbs = adj[cur]
        if not nbs:
            break
        w = np.array([max(float(weight_fn(cur, nb)), 0.0) for nb in nbs])
        s = w.sum()
        if s < 1e-30:
            idx = int(rng.integers(len(nbs)))
        else:
            w /= s
            idx = int(rng.choice(len(nbs), p=w))
        cur = nbs[idx]
        out.append(cur)
    return out


def biased_batch_random_steps(
    G: nx.Graph,
    walkers: list,
    weight_fn: Callable,
    rng: np.random.Generator,
) -> list:
    """Advance each walker by one biased step. Returns new positions."""
    adj = {n: list(G.neighbors(n)) for n in G.nodes()}
    new_w = []
    for w in walkers:
        nbs = adj[w]
        if not nbs:
            new_w.append(w)
            continue
        wts = np.array([max(float(weight_fn(w, nb)), 0.0) for nb in nbs])
        s = wts.sum()
        if s < 1e-30:
            idx = int(rng.integers(len(nbs)))
        else:
            wts /= s
            idx = int(rng.choice(len(nbs), p=wts))
        new_w.append(nbs[idx])
    return new_w


# ---------------------------------------------------------------------------
# Weight-function factories (named presets)
# ---------------------------------------------------------------------------

def make_gradient_bias(
    pos: dict,
    target_pos,
    strength: float = 3.0,
    toward: bool = True,
) -> Callable:
    """
    Drift toward (or away from) a fixed target position.

    weight(cur, nb) = exp(±strength * normalised_progress_toward_target)

    Parameters
    ----------
    pos : dict   node → (x, y)
    target_pos : array-like  (x, y) of the attraction / repulsion centre
    strength : float  how sharply the bias grows with directional progress
    toward : bool  True = drift toward target, False = drift away
    """
    t = np.array(target_pos, dtype=float)
    sign = 1.0 if toward else -1.0

    def weight_fn(cur, nb):
        pc = np.array(pos[cur], dtype=float)
        pn = np.array(pos[nb], dtype=float)
        d_cur = np.linalg.norm(pc - t)
        d_nb = np.linalg.norm(pn - t)
        if d_cur < 1e-12:
            return 1.0
        # Positive when stepping closer, negative when moving away.
        delta = (d_cur - d_nb) / (d_cur + 1e-12)
        return float(np.exp(sign * strength * delta))

    return weight_fn


def make_degree_bias(
    G: nx.Graph,
    strength: float = 2.0,
    hub_seeking: bool = True,
) -> Callable:
    """
    Prefer high-degree neighbours (hub_seeking=True) or low-degree ones (False).

    weight(cur, nb) ∝ deg(nb)^(±strength)
    """
    degrees = dict(G.degree())
    exp = strength if hub_seeking else -strength

    def weight_fn(cur, nb):
        d = float(degrees.get(nb, 1))
        return max(d ** exp, 1e-12)

    return weight_fn
