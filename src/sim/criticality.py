"""
Chapter E: static and dynamic observables near the 2D bond-percolation transition.

The percolation transition is a second-order phase transition at p_c = 1/2 (exact,
by self-duality of the square lattice).  This module provides:

  Static (geometric) observables
  --------------------------------
  order_parameter_sweep     – P_∞(p, L) and susceptibility S(p, L) for several L
  cluster_size_distribution – log-binned n_s at / near / away from p_c
  finite_size_collapse      – rescaled P_∞ for FSS collapse check

  Dynamic observables
  --------------------
  survival_vs_p             – ensemble S(t) with sparse traps; power-law at p_c
  msd_exponent_vs_p         – anomalous diffusion exponent 2/d_w as a function of p

All functions accept a numpy Generator for reproducibility.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from ..graphs.percolation import build_percolation
from ..sim.observables import estimate_msd_rms_series

# Exact 2D bond-percolation critical exponents
P_C: float = 0.5          # critical threshold (exact by duality)
NU: float = 4.0 / 3.0     # correlation-length exponent
BETA: float = 5.0 / 36.0  # order-parameter exponent
TAU: float = 187.0 / 91.0 # Fisher exponent for n_s ~ s^{-τ}
D_W_CRIT: float = 2.87    # walk dimension at p_c (numerical, Havlin & Ben-Avraham)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _component_sizes(size: int, p_open: float, rng: np.random.Generator) -> list[int]:
    """
    Build one percolation instance and return ALL component sizes, sorted descending.
    Isolated nodes count as size-1 components.
    """
    G0 = nx.grid_2d_graph(size, size)
    H = nx.Graph()
    H.add_nodes_from(G0.nodes())
    for u, v in G0.edges():
        if rng.random() < p_open:
            H.add_edge(u, v)
    return sorted([len(c) for c in nx.connected_components(H)], reverse=True)


# ---------------------------------------------------------------------------
# Static observables
# ---------------------------------------------------------------------------

def order_parameter_sweep(
    sizes: list[int],
    p_values: np.ndarray,
    n_samples: int = 40,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Compute P_∞(p) and susceptibility S(p) for each system size L.

    P_∞ = mean |LCC| / L²   (order parameter; vanishes below p_c)
    S   = Σ_{s ≠ s_max} s² n_s / L²  (mean finite-cluster size; peaks at p_c)

    Returns
    -------
    dict: L -> {
        "p":        p_values,
        "P_inf":    mean P_∞,
        "P_inf_std": std P_∞,
        "S":        mean S,
        "S_std":    std S,
    }
    """
    rng = rng or np.random.default_rng()
    results: dict = {}

    for L in sizes:
        N = L * L
        P_mat = np.zeros((len(p_values), n_samples))
        S_mat = np.zeros((len(p_values), n_samples))

        for ip, p in enumerate(p_values):
            for s in range(n_samples):
                sizes_s = _component_sizes(L, float(p), rng)
                giant = sizes_s[0] if sizes_s else 0
                P_mat[ip, s] = giant / N
                rest = sizes_s[1:] if len(sizes_s) > 1 else []
                S_mat[ip, s] = sum(c ** 2 for c in rest) / N if rest else 0.0

        results[L] = {
            "p": p_values,
            "P_inf": P_mat.mean(axis=1),
            "P_inf_std": P_mat.std(axis=1),
            "S": S_mat.mean(axis=1),
            "S_std": S_mat.std(axis=1),
        }

    return results


def cluster_size_distribution(
    size: int,
    p_values: list[float],
    n_samples: int = 150,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Log-binned cluster-size distribution n_s at each p value.

    n_s is defined as (number of s-clusters per lattice site), excluding
    the giant component (which would dominate and obscure the power-law tail).

    At p_c: n_s ~ s^{-τ},  τ = 187/91 ≈ 2.05
    Off-critical: exponential cutoff at s* ~ |p - p_c|^{-1/σ}

    Returns
    -------
    dict: p -> {"centers": bin centers (s), "ns": n_s values}
    """
    rng = rng or np.random.default_rng()
    N = size * size
    bin_edges = np.unique(
        np.round(np.logspace(0, np.log10(N), 35)).astype(int)
    ).astype(float)
    bin_edges = np.concatenate([[0.5], bin_edges])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    results = {}
    for p in p_values:
        counts = np.zeros(len(bin_centers))
        for _ in range(n_samples):
            comp = _component_sizes(size, p, rng)
            for s in (comp[1:] if len(comp) > 1 else []):  # exclude giant
                idx = int(np.searchsorted(bin_edges[1:], s, side="left"))
                if 0 <= idx < len(counts):
                    counts[idx] += 1

        # n_s = counts / (n_samples * N * bin_width)  [units: per site per unit size]
        ns = counts / (n_samples * N * bin_widths)
        mask = ns > 0
        results[float(p)] = {
            "centers": bin_centers[mask],
            "ns": ns[mask],
        }

    return results


def finite_size_collapse(
    sweep_results: dict,
    p_c: float = P_C,
    nu: float = NU,
    beta: float = BETA,
) -> dict:
    """
    Rescale P_∞(p, L) curves for a finite-size scaling collapse.

    With exact 2D bond-percolation exponents:
        x = (p - p_c) · L^{1/ν}   [scaling variable]
        y = P_∞ · L^{β/ν}          [rescaled order parameter]

    A successful collapse means all curves fall on a single universal function,
    confirming that the only relevant length scale near p_c is ξ ~ |p - p_c|^{-ν}.

    Returns
    -------
    dict: L -> {"x": rescaled p, "y": rescaled P_inf, "y_std": rescaled std}
    """
    results = {}
    for L, data in sweep_results.items():
        x = (data["p"] - p_c) * L ** (1.0 / nu)
        scale = L ** (beta / nu)
        results[L] = {
            "x": x,
            "y": data["P_inf"] * scale,
            "y_std": data["P_inf_std"] * scale,
        }
    return results


# ---------------------------------------------------------------------------
# Dynamic observables
# ---------------------------------------------------------------------------

def survival_vs_p(
    size: int,
    p_values: list[float],
    trap_density: float = 0.05,
    n_walkers: int = 200,
    n_steps: int = 500,
    n_graphs: int = 5,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Ensemble survival fraction S(t) for SRW with sparse absorbing traps.

    Traps are placed at random on a fraction trap_density of cluster nodes.
    S(t) is averaged across n_graphs independent percolation realisations.

    At p_c:        S(t) ~ t^{-d_s/2}  (power-law, d_s = spectral dimension ≈ 1.33)
    Above p_c:     S(t) ~ exp(−t/τ)    (exponential, τ depends on trap density)

    Returns
    -------
    dict: p -> np.ndarray of shape (n_steps,)
    """
    rng = rng or np.random.default_rng()
    results: dict = {}

    for p in p_values:
        accumulated = np.zeros(n_steps)
        graphs_used = 0

        for _ in range(n_graphs):
            G, _, _ = build_percolation(size, p, rng=rng)
            nodes = list(G.nodes())
            if len(nodes) < 20:
                continue

            adj = {n: list(G.neighbors(n)) for n in nodes}
            n_traps = max(1, int(len(nodes) * trap_density))
            trap_idx = rng.choice(len(nodes), size=n_traps, replace=False)
            traps = {nodes[int(i)] for i in trap_idx}

            non_trap = [n for n in nodes if n not in traps]
            if not non_trap:
                continue
            n_w = min(n_walkers, len(non_trap))
            start_idx = rng.choice(len(non_trap), size=n_w, replace=False)
            walkers = [non_trap[int(i)] for i in start_idx]

            survival: list[float] = []
            for _ in range(n_steps):
                new_w = []
                for w in walkers:
                    nbs = adj[w]
                    if not nbs:
                        new_w.append(w)
                        continue
                    nxt = nbs[int(rng.integers(len(nbs)))]
                    if nxt not in traps:
                        new_w.append(nxt)
                walkers = new_w
                survival.append(len(walkers) / n_w)
                if not walkers:
                    survival.extend([0.0] * (n_steps - len(survival)))
                    break

            while len(survival) < n_steps:
                survival.append(0.0)

            accumulated += np.array(survival[:n_steps])
            graphs_used += 1

        results[float(p)] = accumulated / graphs_used if graphs_used else np.zeros(n_steps)

    return results


def msd_exponent_vs_p(
    size: int,
    p_values: list[float],
    n_walkers: int = 60,
    n_steps: int = 600,
    n_graphs: int = 5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the MSD anomalous exponent β = 2/d_w as a function of p.

    MSD(N) ~ N^β where:
      β → 1 for large p (normal diffusion on a well-connected grid)
      β ≈ 0.70 at p_c (anomalous diffusion on the fractal incipient cluster)

    Returns
    -------
    p_arr   : np.ndarray  p values
    beta_mean : np.ndarray  mean β at each p
    beta_std  : np.ndarray  std β across graph realisations
    """
    rng = rng or np.random.default_rng()
    beta_all: list[list[float]] = []

    for p in p_values:
        betas: list[float] = []
        for _ in range(n_graphs):
            G, _, _ = build_percolation(size, p, rng=rng)
            nodes = list(G.nodes())
            if len(nodes) < 10:
                continue
            pos_walk = {n: np.array([float(n[0]), float(n[1])]) for n in nodes}
            start = nodes[int(rng.integers(len(nodes)))]
            steps, _, msd = estimate_msd_rms_series(
                G, pos_walk, start, n_walkers, n_steps, rng
            )
            mask = (steps > 25) & (msd > 1e-12)
            if mask.sum() < 8:
                continue
            slope, _ = np.polyfit(np.log(steps[mask]), np.log(msd[mask]), 1)
            betas.append(float(slope))
        beta_all.append(betas if betas else [float("nan")])

    beta_mean = np.array([float(np.nanmean(b)) for b in beta_all])
    beta_std = np.array([
        float(np.nanstd(b)) if len([x for x in b if not np.isnan(x)]) > 1 else 0.0
        for b in beta_all
    ])
    return np.array(p_values, dtype=float), beta_mean, beta_std
