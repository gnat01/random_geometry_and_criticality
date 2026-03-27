"""Continuous-time random walk (CTRW) on the 2-D integer lattice with Pareto waiting times."""

from __future__ import annotations

import numpy as np


_JUMPS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)


def ctrw_positions(
    n_walkers: int,
    n_jumps: int,
    alpha: float,
    t_eval: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    CTRW on the 2-D integer lattice with Pareto(alpha) waiting times.

    Waiting times are drawn as w = u^{-1/alpha} where u ~ Uniform(0,1),
    giving the heavy-tailed law P(W > t) = t^{-alpha} for t >= 1.

    Parameters
    ----------
    n_walkers : int
        Number of independent walkers.
    n_jumps : int
        Number of jump events per walker.
    alpha : float
        Tail exponent in (0, 1) for subdiffusion.
    t_eval : ndarray of shape (T,)
        Physical times at which positions are recorded.
    rng : np.random.Generator

    Returns
    -------
    positions : ndarray of shape (n_walkers, len(t_eval), 2)
    """
    t_eval = np.asarray(t_eval, dtype=np.float64)
    T = len(t_eval)
    positions = np.zeros((n_walkers, T, 2), dtype=np.float64)

    for w in range(n_walkers):
        # Waiting times: Pareto via inversion
        u = rng.random(n_jumps)
        u = np.clip(u, 1e-14, 1.0)
        waiting = u ** (-1.0 / alpha)

        # Cumulative time at each jump (starting at 0 before any jump)
        cum_t = np.empty(n_jumps + 1, dtype=np.float64)
        cum_t[0] = 0.0
        cum_t[1:] = np.cumsum(waiting)

        # Jump directions
        dirs = rng.integers(0, 4, size=n_jumps)
        dx = _JUMPS[dirs]  # (n_jumps, 2)

        # Position array: pos_at_jump[k] = position after k jumps
        pos_jumps = np.empty((n_jumps + 1, 2), dtype=np.float64)
        pos_jumps[0] = 0.0
        pos_jumps[1:] = np.cumsum(dx, axis=0)

        # For each evaluation time, find the last jump index before t_eval
        # np.searchsorted(cum_t, t, side='right') - 1 gives the index of the
        # last cum_t <= t. We then clip to [0, n_jumps].
        idx = np.searchsorted(cum_t, t_eval, side="right") - 1
        idx = np.clip(idx, 0, n_jumps)

        positions[w] = pos_jumps[idx]

    return positions


def brownian_positions(
    n_walkers: int,
    t_eval: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Regular 2-D lattice random walk (BM comparison for CTRW).

    One step per unit time. Positions are recorded at t_eval (integer-valued).

    Parameters
    ----------
    n_walkers : int
    t_eval : ndarray of integer-valued times
    rng : np.random.Generator

    Returns
    -------
    positions : ndarray of shape (n_walkers, len(t_eval), 2)
    """
    t_eval = np.asarray(t_eval, dtype=np.float64)
    t_int = t_eval.astype(np.int64)
    T_total = int(t_int[-1]) if len(t_int) > 0 else 0
    T = len(t_eval)

    positions = np.zeros((n_walkers, T, 2), dtype=np.float64)

    dirs = rng.integers(0, 4, size=(n_walkers, T_total))
    dx = _JUMPS[dirs]  # (n_walkers, T_total, 2)

    # cumulative positions at each integer step
    pos_all = np.concatenate(
        [np.zeros((n_walkers, 1, 2), dtype=np.float64), np.cumsum(dx, axis=1)],
        axis=1,
    )  # (n_walkers, T_total+1, 2)

    for k, ti in enumerate(t_int):
        positions[:, k, :] = pos_all[:, ti, :]

    return positions
