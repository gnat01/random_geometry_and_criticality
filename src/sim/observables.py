"""Chapter B observables: occupation, first passage, traps."""

from __future__ import annotations

import networkx as nx
import numpy as np

from .walks import batch_random_steps, simple_random_walk


def occupation_counts(
    G: nx.Graph,
    start,
    steps: int,
    rng: np.random.Generator,
) -> dict:
    """Empirical visit counts over steps transitions (including start)."""
    traj = simple_random_walk(G, start, steps, rng)
    counts = {n: 0 for n in G.nodes()}
    for v in traj:
        counts[v] = counts.get(v, 0) + 1
    return counts


def first_passage_times(
    G: nx.Graph,
    target_set: set,
    n_trials: int,
    max_steps: int,
    rng: np.random.Generator,
    start_fn,
) -> list[int]:
    """
    Sample hitting times to target_set. If not hit within max_steps, trial is dropped.

    start_fn(rng) -> start node
    """
    adj = {n: list(G.neighbors(n)) for n in G.nodes()}
    times: list[int] = []
    for _ in range(n_trials):
        s0 = start_fn(rng)
        if s0 in target_set:
            times.append(0)
            continue
        cur = s0
        for t in range(1, max_steps + 1):
            nbs = adj[cur]
            if not nbs:
                break
            cur = nbs[rng.integers(len(nbs))]
            if cur in target_set:
                times.append(t)
                break
    return times


def survival_with_traps(
    G: nx.Graph,
    traps: set,
    walkers: list,
    rng: np.random.Generator,
) -> tuple[list, int]:
    """
    One step of SRW with absorbing traps: walkers on traps are removed.

    Returns new walker list and count of survivors.
    """
    if not traps:
        new_w = batch_random_steps(G, walkers, rng)
        return new_w, len(new_w)

    adj = {n: list(G.neighbors(n)) for n in G.nodes()}
    new_w = []
    for w in walkers:
        if w in traps:
            continue
        nbs = adj[w]
        if not nbs:
            new_w.append(w)
            continue
        nxt = nbs[rng.integers(len(nbs))]
        if nxt in traps:
            continue
        new_w.append(nxt)
    return new_w, len(new_w)


def estimate_msd_rms_series(
    G: nx.Graph,
    pos_walk: dict,
    start,
    n_walkers: int,
    n_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel walkers from same start; return step indices, RMS, MSD arrays
    in Euclidean coords given by pos_walk (node -> (x,y)).
    """
    nodes = list(G.nodes())
    adj = {n: list(G.neighbors(n)) for n in nodes}
    s0 = start if start in adj else nodes[0]
    p0 = np.asarray(pos_walk[s0], dtype=float)

    walkers = [s0] * n_walkers
    steps_list = []
    rms_list = []
    msd_list = []

    for step in range(1, n_steps + 1):
        walkers = [
            adj[w][rng.integers(len(adj[w]))] if adj[w] else w for w in walkers
        ]
        vecs = np.array([np.asarray(pos_walk[w], dtype=float) - p0 for w in walkers])
        sq = np.sum(vecs**2, axis=1)
        steps_list.append(step)
        rms_list.append(float(np.sqrt(np.mean(sq))))
        msd_list.append(float(np.mean(sq)))

    return (
        np.asarray(steps_list, dtype=float),
        np.asarray(rms_list, dtype=float),
        np.asarray(msd_list, dtype=float),
    )
