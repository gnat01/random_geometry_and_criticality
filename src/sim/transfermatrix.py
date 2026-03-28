"""Ising strip transfer matrix: correlation length and T_c extrapolation."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
T_C_ISING: float = 2.0 / np.log(1.0 + np.sqrt(2.0))   # ≈ 2.2692


def _spin_configs(W: int) -> np.ndarray:
    """
    All 2^W spin configurations for a strip of width W.

    Returns
    -------
    cfgs : ndarray of shape (2^W, W) with values in {-1, +1}
    """
    n = 2 ** W
    bits = (np.arange(n)[:, None] >> np.arange(W - 1, -1, -1)[None, :]) & 1
    return (2 * bits - 1).astype(np.int8)


def build_transfer_matrix(W: int, K: float) -> np.ndarray:
    """
    Build the 2^W × 2^W transfer matrix for the Ising strip of width W.

    T[α,β] = exp(K * Σ_i σ_i^α σ_i^β  + (K/2) * (h_α + h_β))
    where h_α = Σ_i σ_i^α σ_{i+1}^α  (horizontal bonds, periodic in W).

    Parameters
    ----------
    W : strip width (number of spins per row)
    K : J/k_B T  (coupling / temperature)

    Returns
    -------
    T_mat : ndarray of shape (2^W, 2^W)
    """
    cfgs = _spin_configs(W)                     # (n, W)
    # Horizontal bond energy per configuration (periodic)
    h = np.sum(cfgs * np.roll(cfgs, 1, axis=1), axis=1).astype(np.float64)
    # Vertical bond energy between every pair of configurations
    v = cfgs.astype(np.float64) @ cfgs.astype(np.float64).T   # (n, n)
    T_mat = np.exp(K * v + 0.5 * K * (h[:, None] + h[None, :]))
    return T_mat


def correlation_length(W: int, T_temp: float) -> float:
    """
    Correlation length ξ = 1 / ln(λ₁/λ₂) for an Ising strip of width W at temperature T.

    Uses scipy.linalg.eigh (symmetric matrix → real eigenvalues).
    """
    K = 1.0 / T_temp
    T_mat = build_transfer_matrix(W, K)
    # eigh returns eigenvalues in ascending order
    evals = eigh(T_mat, eigvals_only=True)
    lam1 = evals[-1]
    lam2 = evals[-2]
    ratio = lam1 / (lam2 + 1e-300)
    if ratio <= 1.0:
        return np.inf
    return 1.0 / np.log(ratio)


def correlation_length_scan(
    widths: list[int],
    T_values: np.ndarray,
) -> dict[int, dict]:
    """
    Compute ξ(T) for several strip widths.

    Returns
    -------
    dict mapping W -> {"T": array, "xi": array, "xi_over_W": array}
    """
    result: dict[int, dict] = {}
    for W in widths:
        xi_arr = np.zeros(len(T_values))
        for iT, T in enumerate(T_values):
            xi_arr[iT] = correlation_length(W, T)
        result[W] = {
            "T": T_values.copy(),
            "xi": xi_arr,
            "xi_over_W": xi_arr / W,
        }
    return result


def pseudo_critical_temps(
    widths: list[int],
    T_values: np.ndarray,
) -> dict[int, float]:
    """
    Estimate T_c(W) as the temperature where ξ/W is maximised.

    Returns
    -------
    dict mapping W -> T_c(W)
    """
    result: dict[int, float] = {}
    for W in widths:
        xi = np.array([
            correlation_length(W, T) for T in T_values
        ])
        idx = int(np.argmax(xi / W))
        result[W] = float(T_values[idx])
    return result
