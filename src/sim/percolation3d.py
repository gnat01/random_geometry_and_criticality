"""3-D cubic-lattice bond percolation and finite-size scaling."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# -----------------------------------------------------------------------
# Critical exponents for 3-D bond percolation (cubic lattice)
# -----------------------------------------------------------------------
P_C_3D: float = 0.2488   # bond percolation threshold
NU_3D: float = 0.876      # correlation-length exponent
BETA_3D: float = 0.418    # order-parameter exponent
GAMMA_3D: float = 1.793   # susceptibility exponent


def bond_percolation_3d(
    L: int,
    p: float,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    """
    Periodic 3-D cubic lattice bond percolation.

    Parameters
    ----------
    L : linear size
    p : bond occupation probability
    rng : NumPy Generator

    Returns
    -------
    n_comp : number of connected components
    labels : integer cluster label for each site, shape (L^3,)
    """
    N = L ** 3
    idx = np.arange(N, dtype=np.int32).reshape(L, L, L)

    src_x = idx.ravel()
    dst_x = np.roll(idx, -1, axis=0).ravel()
    src_y = idx.ravel()
    dst_y = np.roll(idx, -1, axis=1).ravel()
    src_z = idx.ravel()
    dst_z = np.roll(idx, -1, axis=2).ravel()

    src = np.concatenate([src_x, src_y, src_z])
    dst = np.concatenate([dst_x, dst_y, dst_z])

    open_ = rng.random(len(src)) < p
    src, dst = src[open_], dst[open_]

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(N, N))

    n_comp, labels = connected_components(adj, directed=False)
    return n_comp, labels


def _cluster_stats(L: int, labels: np.ndarray) -> tuple[float, float]:
    """Return (P_inf, susceptibility) for one realisation."""
    N = L ** 3
    sizes = np.bincount(labels)
    s_max = int(sizes.max())
    P_inf = s_max / N
    # susceptibility = mean finite-cluster size (excl. giant)
    s_finite = sizes[sizes < s_max]
    if len(s_finite) == 0:
        chi = 0.0
    else:
        chi = float((s_finite ** 2).sum()) / N
    return P_inf, chi


def order_parameter_sweep_3d(
    sizes: list[int],
    p_values: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> dict[int, dict]:
    """
    Sweep p for several system sizes.

    Returns
    -------
    dict mapping L -> {"p": array, "P_inf": array, "chi": array}
    """
    result: dict[int, dict] = {}
    for L in sizes:
        P_inf_arr = np.zeros(len(p_values))
        chi_arr = np.zeros(len(p_values))
        for ip, p in enumerate(p_values):
            P_samples = []
            chi_samples = []
            for _ in range(n_samples):
                _, labels = bond_percolation_3d(L, p, rng)
                Pi, chi = _cluster_stats(L, labels)
                P_samples.append(Pi)
                chi_samples.append(chi)
            P_inf_arr[ip] = np.mean(P_samples)
            chi_arr[ip] = np.mean(chi_samples)
        result[L] = {"p": p_values.copy(), "P_inf": P_inf_arr, "chi": chi_arr}
    return result


def finite_size_collapse_3d(sweep: dict[int, dict]) -> dict[int, dict]:
    """
    Apply 3-D FSS rescaling:
        x = (p - p_c) * L^{1/ν}
        y = P_inf * L^{β/ν}
    """
    result: dict[int, dict] = {}
    for L, data in sweep.items():
        x = (data["p"] - P_C_3D) * L ** (1.0 / NU_3D)
        y = data["P_inf"] * L ** (BETA_3D / NU_3D)
        result[L] = {"x": x, "y": y}
    return result


def lcc_positions_3d(
    L: int, p: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Return (x, y, z) coordinates of the largest connected component.

    Returns
    -------
    xyz : ndarray of shape (n_lcc, 3), values in {0, …, L-1}
    """
    _, labels = bond_percolation_3d(L, p, rng)
    sizes = np.bincount(labels)
    lcc_label = int(sizes.argmax())
    mask = labels == lcc_label
    flat_idx = np.where(mask)[0]
    z = flat_idx % L
    y = (flat_idx // L) % L
    x = flat_idx // (L * L)
    return np.column_stack([x, y, z]).astype(np.float32)
