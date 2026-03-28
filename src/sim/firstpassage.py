"""First-passage time measurements on arbitrary graphs."""

from __future__ import annotations

import numpy as np
import networkx as nx


def first_passage_times(
    G: nx.Graph,
    target_node,
    n_walkers: int = 10000,
    t_max: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Release n_walkers from uniformly random starting nodes (excluding target)
    and record the first time each hits target_node.

    Walkers that do not hit within t_max steps are recorded as t_max (censored).

    Returns array of length n_walkers with first-passage times.
    """
    if rng is None:
        rng = np.random.default_rng()

    nodes = list(G.nodes())
    n = len(nodes)
    idx_of = {v: i for i, v in enumerate(nodes)}
    target_idx = idx_of[target_node]

    # Padded adjacency
    degrees = np.array([G.degree(v) for v in nodes], dtype=np.int32)
    max_deg = int(degrees.max())
    adj = np.zeros((n, max_deg), dtype=np.int32)
    for i, v in enumerate(nodes):
        nbs = [idx_of[nb] for nb in G.neighbors(v)]
        adj[i, : len(nbs)] = nbs

    # Start from random nodes (not the target)
    non_target = [i for i in range(n) if i != target_idx]
    start_idx = rng.choice(non_target, size=n_walkers, replace=True).astype(np.int32)
    cur = start_idx.copy()

    hit_time = np.full(n_walkers, t_max, dtype=np.int32)
    active = np.ones(n_walkers, dtype=bool)

    for t in range(1, t_max + 1):
        if not active.any():
            break
        # Step active walkers
        act_idx = np.where(active)[0]
        deg_cur = degrees[cur[act_idx]]
        col = (rng.random(len(act_idx)) * deg_cur).astype(np.int32)
        cur[act_idx] = adj[cur[act_idx], col]
        # Check hits
        just_hit = act_idx[cur[act_idx] == target_idx]
        if len(just_hit):
            hit_time[just_hit] = t
            active[just_hit] = False

    return hit_time


def survival_function(hit_times: np.ndarray, t_max: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical survival function S(t) = P(T_fp > t) from hit times.

    Censored observations (hit_time == t_max) are included — they contribute
    to the right tail but do not count as actual hits.

    Returns (t_values, S_values).
    """
    hit_times = np.asarray(hit_times)
    if t_max is None:
        t_max = int(hit_times.max())

    t_vals = np.arange(1, t_max + 1)
    # S(t) = fraction of walkers with T_fp > t (censored walkers count as > t)
    S = np.array([np.mean(hit_times > t) for t in t_vals])
    return t_vals.astype(float), S


def fit_survival_exponent(
    t_vals: np.ndarray,
    S_vals: np.ndarray,
    t_min: int = 50,
    t_max: int | None = None,
) -> tuple[float, float]:
    """
    Fit S(t) ~ t^{-alpha} on a log-log scale.
    Returns (alpha, R²).  For fractal diffusion alpha = d_s/2.
    """
    if t_max is None:
        t_max = int(t_vals[-1])
    mask = (t_vals >= t_min) & (t_vals <= t_max) & (S_vals > 1e-6)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    lx = np.log(t_vals[mask])
    ly = np.log(S_vals[mask])
    c = np.polyfit(lx, ly, 1)
    alpha = float(-c[0])
    ly_fit = np.polyval(c, lx)
    ss_res = np.sum((ly - ly_fit) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return alpha, r2


def mean_fpt_vs_size(
    graphs: list[tuple],
    target_fn,
    n_walkers: int = 2000,
    t_max: int = 50000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure mean FPT across a family of graphs of increasing size.

    graphs   : list of (G, pos, side) tuples (e.g. from build_carpet at depths 2,3,4,5)
    target_fn: callable(G, pos, side) -> target_node  (e.g. pick the central node)

    Returns (sizes, mean_fpts, std_fpts) where sizes = side length of each graph.
    """
    if rng is None:
        rng = np.random.default_rng()

    sizes, means, stds = [], [], []
    for G, pos, side in graphs:
        target = target_fn(G, pos, side)
        fpts = first_passage_times(G, target, n_walkers=n_walkers, t_max=t_max, rng=rng)
        # Exclude censored walkers from mean (they never hit — would bias estimate high)
        actual = fpts[fpts < t_max]
        sizes.append(side)
        means.append(float(np.mean(actual)) if len(actual) > 0 else float("nan"))
        stds.append(float(np.std(actual)) if len(actual) > 1 else float("nan"))

    return np.array(sizes, dtype=float), np.array(means), np.array(stds)
