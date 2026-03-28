"""
Diffusion-Limited Aggregation (DLA) on graph substrates.

DLA is restricted to percolation graphs: the random substrate gives enough room for
clusters to grow before hitting finite-size saturation, and grids of size ≥ 40 provide
at least one decade of scaling range.

The module exposes two public functions:
  grow_dla_cluster  — one independent DLA realisation on a fixed graph
  dla_ensemble      — N realisations + ensemble-averaged R_g(M) + honest fit in window
"""

from __future__ import annotations

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Single realisation
# ---------------------------------------------------------------------------

def grow_dla_cluster(
    G: nx.Graph,
    pos: dict,
    seed_node=None,
    max_fraction: float = 0.15,
    max_walk_steps: int = 8000,
    rng: np.random.Generator | None = None,
) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Grow one DLA cluster on graph G.

    The seed is placed at the node closest to the graph centroid.  Walkers are
    launched from random non-cluster nodes and perform SRW until they land on a
    node adjacent to the cluster (they stick there) or exceed max_walk_steps
    (the trial is discarded and a fresh walker is released).

    Growth stops when |cluster| / |G| >= max_fraction.

    Parameters
    ----------
    G : nx.Graph
    pos : dict   node → (x, y)  Euclidean embedding for R_g calculation
    seed_node :  optional override for the seed
    max_fraction : float  growth cap as fraction of total nodes
    max_walk_steps : int  walker timeout (discarded, not added to cluster)
    rng : numpy Generator

    Returns
    -------
    growth_order : list  nodes in order of addition (element 0 = seed)
    masses       : np.ndarray  [1, 2, …, len(growth_order)]
    rg_series    : np.ndarray  radius of gyration R_g at each M
    """
    rng = rng or np.random.default_rng()
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return [], np.array([]), np.array([])

    adj = {v: list(G.neighbors(v)) for v in nodes}
    coords_map = {v: np.array(pos[v], dtype=float) for v in nodes}

    # Seed: node whose position is closest to the cloud centroid
    if seed_node is None or seed_node not in G:
        all_coords = np.array([coords_map[v] for v in nodes])
        centroid = all_coords.mean(axis=0)
        dists = np.linalg.norm(all_coords - centroid, axis=1)
        seed_node = nodes[int(np.argmin(dists))]

    cluster: set = {seed_node}
    # Nodes not in the cluster that border at least one cluster node
    cluster_boundary: set = set(adj[seed_node]) - cluster

    growth_order = [seed_node]
    cluster_coords = [coords_map[seed_node].copy()]
    max_size = max(2, int(n * max_fraction))

    rg_series = [0.0]  # R_g at M=1 is 0 by definition
    masses = [1]

    non_cluster_nodes = [v for v in nodes if v not in cluster]

    while len(cluster) < max_size and cluster_boundary and non_cluster_nodes:
        # Launch walker from a random non-cluster node
        launch_idx = int(rng.integers(len(non_cluster_nodes)))
        walker = non_cluster_nodes[launch_idx]

        # Walk until: (a) walker lands on a cluster-boundary node → sticks;
        # (b) max_walk_steps exceeded → discard (no cluster modification).
        stuck_node = None
        for _ in range(max_walk_steps):
            nbs = adj[walker]
            if not nbs:
                break
            walker = nbs[int(rng.integers(len(nbs)))]
            if walker in cluster:
                # Stepped into cluster (can happen near dense boundary).
                # Treat the previous position as the stuck node is awkward without
                # tracking prev, so just discard and release a fresh walker.
                break
            if walker in cluster_boundary:
                stuck_node = walker
                break

        if stuck_node is None:
            continue  # walker timed out or entered cluster — try again

        # Add stuck node to cluster
        cluster.add(stuck_node)
        cluster_coords.append(coords_map[stuck_node].copy())
        growth_order.append(stuck_node)

        # Update boundary
        cluster_boundary.discard(stuck_node)
        for nb in adj[stuck_node]:
            if nb not in cluster:
                cluster_boundary.add(nb)

        # Maintain non_cluster list incrementally
        non_cluster_nodes = [v for v in non_cluster_nodes if v not in cluster]

        # Radius of gyration
        arr = np.array(cluster_coords)
        cm = arr.mean(axis=0)
        rg = float(np.sqrt(np.mean(np.sum((arr - cm) ** 2, axis=1))))
        rg_series.append(rg)
        masses.append(len(cluster))

    return growth_order, np.array(masses, dtype=float), np.array(rg_series, dtype=float)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def dla_ensemble(
    G: nx.Graph,
    pos: dict,
    n_runs: int = 20,
    max_fraction: float = 0.15,
    max_walk_steps: int = 8000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Run n_runs independent DLA realisations on the same graph and return
    ensemble statistics for R_g(M).

    All runs use the same seed node (graph centroid) so that the ensemble
    reflects DLA stochasticity, not substrate variation.

    Parameters
    ----------
    G, pos, max_fraction, max_walk_steps : see grow_dla_cluster
    n_runs : int   number of independent realisations
    rng : Generator

    Returns
    -------
    dict with keys:
      "masses"        : common mass grid (np.ndarray)
      "rg_mean"       : ensemble-mean R_g at each M
      "rg_std"        : ensemble std
      "rg_runs"       : list of per-run rg_series (each interpolated to common grid)
      "growth_orders" : list of per-run growth_order lists
      "d_f_fit"       : estimated fractal dimension from log-log fit
      "d_f_r2"        : R² of the fit
      "fit_lo"        : lower M boundary of fit window
      "fit_hi"        : upper M boundary of fit window
      "max_fraction"  : max_fraction used (for display)
      "n_nodes"       : len(G)
    """
    rng = rng or np.random.default_rng()

    all_rg: list[np.ndarray] = []
    all_masses: list[np.ndarray] = []
    growth_orders: list[list] = []

    for _ in range(n_runs):
        go, masses, rg = grow_dla_cluster(
            G, pos,
            max_fraction=max_fraction,
            max_walk_steps=max_walk_steps,
            rng=rng,
        )
        if len(masses) >= 5:
            all_rg.append(rg)
            all_masses.append(masses)
            growth_orders.append(go)

    if not all_rg:
        return {"error": "No valid DLA runs completed."}

    # Interpolate every run onto the shortest common mass grid
    min_len = min(len(m) for m in all_masses)
    common_masses = all_masses[0][:min_len]  # use first run's mass axis (all start at 1)
    interp_rg = []
    for masses, rg in zip(all_masses, all_rg):
        # Linear interpolation onto common_masses
        interp_rg.append(np.interp(common_masses, masses, rg))

    rg_arr = np.array(interp_rg)  # (n_runs, min_len)
    rg_mean = rg_arr.mean(axis=0)
    rg_std = rg_arr.std(axis=0)

    # Fit window: avoid small-cluster noise at low M and finite-size saturation at high M
    fit_lo, fit_hi = _fit_window(common_masses, rg_mean)

    d_f, r2 = _fit_fractal_dim(common_masses, rg_mean, fit_lo, fit_hi)

    return {
        "masses": common_masses,
        "rg_mean": rg_mean,
        "rg_std": rg_std,
        "rg_runs": interp_rg,
        "growth_orders": growth_orders,
        "d_f_fit": d_f,
        "d_f_r2": r2,
        "fit_lo": fit_lo,
        "fit_hi": fit_hi,
        "max_fraction": max_fraction,
        "n_nodes": G.number_of_nodes(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_window(
    masses: np.ndarray,
    rg_mean: np.ndarray,
    lo_frac: float = 0.05,
    hi_frac: float = 0.60,
) -> tuple[float, float]:
    """
    Choose a fitting window [M_lo, M_hi] that avoids:
      - small-cluster noise  (below lo_frac * max_M)
      - finite-size saturation (above the inflection where growth slows to < 20 % of peak)

    hi_frac caps the window as a safety net even if saturation is not detected.
    """
    max_M = float(masses[-1])
    M_lo = max(5.0, lo_frac * max_M)

    # Detect saturation via smoothed first derivative of R_g
    M_hi_default = hi_frac * max_M
    if len(rg_mean) > 20:
        win = max(3, len(rg_mean) // 8)
        rg_s = np.convolve(rg_mean, np.ones(win) / win, mode="valid")
        drg = np.diff(rg_s)
        if drg.max() > 1e-12:
            # Find first index where growth rate drops below 20 % of its early-phase peak
            early_peak = drg[: max(1, len(drg) // 3)].max()
            threshold = 0.20 * early_peak
            sat_idx = np.searchsorted(-drg, -threshold)
            # Map back to the original mass axis (convolve trims the edges)
            sat_M = float(masses[min(sat_idx + win, len(masses) - 1)])
            M_hi = min(sat_M, M_hi_default)
        else:
            M_hi = M_hi_default
    else:
        M_hi = M_hi_default

    M_hi = max(M_lo + 5.0, M_hi)
    return float(M_lo), float(M_hi)


def _fit_fractal_dim(
    masses: np.ndarray,
    rg_mean: np.ndarray,
    fit_lo: float,
    fit_hi: float,
) -> tuple[float, float]:
    """
    Fit R_g ~ M^(1/d_f) via log-log linear regression in [fit_lo, fit_hi].

    Returns (d_f, R²).  d_f = nan if the window contains fewer than 5 points.
    """
    mask = (masses >= fit_lo) & (masses <= fit_hi) & (rg_mean > 1e-12)
    if mask.sum() < 5:
        return float("nan"), float("nan")

    lm = np.log(masses[mask])
    lr = np.log(rg_mean[mask])
    slope, intercept = np.polyfit(lm, lr, 1)

    # R²
    lr_pred = slope * lm + intercept
    ss_res = np.sum((lr - lr_pred) ** 2)
    ss_tot = np.sum((lr - lr.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-30))

    # slope = 1/d_f  ⟹  d_f = 1/slope
    d_f = float(1.0 / slope) if abs(slope) > 1e-9 else float("nan")
    return d_f, r2
