"""1+1-D directed bond percolation on a strip lattice."""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------
# Critical constants for 1+1-D directed percolation (DP)
# -----------------------------------------------------------------------
P_C_DP: float = 0.6447     # bond percolation threshold
BETA_DP: float = 0.2765    # order-parameter exponent
NU_PERP: float = 1.097     # correlation length exponent (spatial)
NU_PAR: float = 1.734      # correlation time exponent (temporal)


def run_dp_spacetime(
    L: int,
    T: int,
    p: float,
    rng: np.random.Generator,
    seed_full: bool = True,
) -> np.ndarray:
    """
    1+1-D directed bond percolation on an L-wide, T-step strip.

    Each active site at (t, x) tries to activate (t+1, x) and (t+1, (x+1) % L)
    via bonds that are open with probability p.

    Parameters
    ----------
    L : spatial size (periodic boundaries)
    T : number of time steps
    p : bond opening probability
    rng : NumPy Generator
    seed_full : if True, seed entire t=0 row as active; else only centre site

    Returns
    -------
    activity : boolean ndarray of shape (T, L)
    """
    activity = np.zeros((T, L), dtype=bool)
    if seed_full:
        activity[0, :] = True
    else:
        activity[0, L // 2] = True

    for t in range(1, T):
        prev = activity[t - 1]
        if not prev.any():
            break
        # Straight bond: (t-1, x) -> (t, x)
        bond_s = rng.random(L) < p
        # Diagonal bond: (t-1, x) -> (t, (x+1) % L)
        bond_d = rng.random(L) < p
        # Site t is active if reached via straight bond from x, or diagonal bond from x-1
        activity[t] = (prev & bond_s) | (np.roll(prev, 1) & bond_d)

    return activity


def order_parameter_dp(activity: np.ndarray) -> float:
    """Fraction of active sites in the final row."""
    return float(activity[-1].mean())


def survival_probability_dp(activity: np.ndarray) -> float:
    """Fraction of time steps with at least one active site."""
    return float(activity.any(axis=1).mean())


def density_scan(
    L: int,
    T: int,
    p_values: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> dict:
    """
    Sweep over p and measure order parameter and survival probability.

    Returns
    -------
    dict with keys "p", "rho" (mean final density), "surv" (survival prob)
    """
    rho = np.zeros(len(p_values))
    surv = np.zeros(len(p_values))

    for ip, p in enumerate(p_values):
        rhos, survs = [], []
        for _ in range(n_samples):
            act = run_dp_spacetime(L, T, p, rng, seed_full=True)
            rhos.append(order_parameter_dp(act))
            survs.append(survival_probability_dp(act))
        rho[ip] = np.mean(rhos)
        surv[ip] = np.mean(survs)

    return {"p": p_values.copy(), "rho": rho, "surv": surv}


def density_decay(
    L: int,
    T: int,
    p: float,
    n_samples: int,
    rng: np.random.Generator,
) -> dict:
    """
    At a fixed p, measure the mean active density ρ(t) averaged over realisations.

    Returns
    -------
    dict with keys "t", "rho_t"
    """
    rho_sum = np.zeros(T)
    for _ in range(n_samples):
        act = run_dp_spacetime(L, T, p, rng, seed_full=True)
        rho_sum += act.mean(axis=1)
    return {"t": np.arange(T, dtype=np.float64), "rho_t": rho_sum / n_samples}
