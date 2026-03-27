"""Spectral analysis of fractal graphs: Laplacian eigenspectrum and return probability."""

from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.sparse
import scipy.sparse.linalg


# ---------------------------------------------------------------------------
# Laplacian eigenspectrum
# ---------------------------------------------------------------------------

def laplacian_eigenvalues(G: nx.Graph, k: int = 300) -> np.ndarray:
    """
    Return up to k smallest non-zero eigenvalues of the graph Laplacian,
    sorted ascending.

    The zero eigenvalue (constant eigenvector) is always dropped.
    """
    n = G.number_of_nodes()
    k_req = min(k + 1, n - 1)  # eigsh requires k < n
    L = nx.laplacian_matrix(G).astype(float)
    vals = scipy.sparse.linalg.eigsh(
        L, k=k_req, which="SM", tol=1e-8, return_eigenvectors=False
    )
    vals = np.sort(np.abs(vals))          # abs guards against tiny negative floats
    vals = vals[vals > 1e-8]              # drop zero mode
    return vals[:k]


def fit_spectral_dim_from_eigenvalues(
    eigenvalues: np.ndarray,
    fit_frac: float = 0.25,
) -> tuple[float, float]:
    """
    Fit d_s from N(omega) ~ omega^{d_s} (integrated density of states).

    Uses the bottom `fit_frac` fraction of eigenvalues where the power law holds.
    Returns (d_s, R²).
    """
    N = np.arange(1, len(eigenvalues) + 1, dtype=float)
    n_fit = max(10, int(len(eigenvalues) * fit_frac))
    lx = np.log(eigenvalues[:n_fit])
    ly = np.log(N[:n_fit])
    c = np.polyfit(lx, ly, 1)
    d_s = float(c[0])
    ly_fit = np.polyval(c, lx)
    ss_res = np.sum((ly - ly_fit) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return d_s, r2


# ---------------------------------------------------------------------------
# Return probability (vectorised)
# ---------------------------------------------------------------------------

def return_probability(
    G: nx.Graph,
    n_walkers: int = 5000,
    n_steps: int = 3000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure P(t) = fraction of walkers at their starting node at time t.

    Uses a padded-adjacency vectorised walk for speed.
    Returns (t_array, P_array) both of length n_steps+1.
    """
    if rng is None:
        rng = np.random.default_rng()

    nodes = list(G.nodes())
    n = len(nodes)
    idx_of = {v: i for i, v in enumerate(nodes)}

    # Build padded adjacency (n × max_degree)
    degrees = np.array([G.degree(v) for v in nodes], dtype=np.int32)
    max_deg = int(degrees.max())
    adj = np.zeros((n, max_deg), dtype=np.int32)
    for i, v in enumerate(nodes):
        nbs = [idx_of[nb] for nb in G.neighbors(v)]
        adj[i, : len(nbs)] = nbs

    # Random starting positions
    start = rng.integers(0, n, size=n_walkers, dtype=np.int32)
    cur = start.copy()

    P = np.empty(n_steps + 1)
    P[0] = 1.0

    for t in range(1, n_steps + 1):
        deg_cur = degrees[cur]                            # (n_walkers,)
        col = (rng.random(n_walkers) * deg_cur).astype(np.int32)
        cur = adj[cur, col]
        P[t] = np.mean(cur == start)

    return np.arange(n_steps + 1, dtype=float), P


def fit_spectral_dim_from_return_prob(
    t_arr: np.ndarray,
    P_arr: np.ndarray,
    t_min: int = 100,
    t_max: int | None = None,
) -> tuple[float, float]:
    """
    Fit d_s from P(t) ~ t^{-d_s/2}.
    Returns (d_s, R²).
    """
    if t_max is None:
        t_max = int(t_arr[-1])
    mask = (t_arr >= t_min) & (t_arr <= t_max) & (P_arr > 1e-12)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    lx = np.log(t_arr[mask])
    ly = np.log(P_arr[mask])
    c = np.polyfit(lx, ly, 1)
    d_s = float(-2.0 * c[0])             # slope = -d_s/2
    ly_fit = np.polyval(c, lx)
    ss_res = np.sum((ly - ly_fit) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return d_s, r2
