"""2-D Ising model: Wolff cluster algorithm and observables."""

from __future__ import annotations

from collections import deque

import numpy as np

# -----------------------------------------------------------------------
# Exact critical constants (square lattice, zero field)
# -----------------------------------------------------------------------
T_C: float = 2.0 / np.log(1.0 + np.sqrt(2.0))   # ≈ 2.2692
NU_ISING: float = 1.0
BETA_ISING: float = 0.125     # 1/8
GAMMA_ISING: float = 1.75     # 7/4


def _random_spins(L: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice([-1, 1], size=(L, L)).astype(np.int8)


def wolff_step(
    spins: np.ndarray,
    T: float,
    rng: np.random.Generator,
) -> int:
    """
    One Wolff cluster flip on an L×L lattice with periodic boundaries.

    Returns
    -------
    cluster_size : int
    """
    L = spins.shape[0]
    p_add = 1.0 - np.exp(-2.0 / T)

    i0 = int(rng.integers(0, L))
    j0 = int(rng.integers(0, L))
    s0 = int(spins[i0, j0])

    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[i0, j0] = True
    queue: deque[tuple[int, int]] = deque()
    queue.append((i0, j0))
    size = 1

    while queue:
        i, j = queue.popleft()
        for ni, nj in (
            ((i - 1) % L, j),
            ((i + 1) % L, j),
            (i, (j - 1) % L),
            (i, (j + 1) % L),
        ):
            if not in_cluster[ni, nj] and int(spins[ni, nj]) == s0:
                if rng.random() < p_add:
                    in_cluster[ni, nj] = True
                    queue.append((ni, nj))
                    size += 1

    spins[in_cluster] = -s0
    return size


def wolff_sweep(
    spins: np.ndarray,
    T: float,
    n_sweeps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Run n_sweeps Wolff steps (each flip roughly O(L²) spins on average at T_c).

    Returns the final spin configuration (in-place modification + return).
    """
    for _ in range(n_sweeps):
        wolff_step(spins, T, rng)
    return spins


def _observables(spins: np.ndarray) -> dict[str, float]:
    L = spins.shape[0]
    N = L * L
    m = float(spins.sum()) / N
    return {"m": m, "m2": m * m, "m4": m ** 4}


def temperature_scan(
    sizes: list[int],
    T_values: np.ndarray,
    n_therm: int,
    n_meas: int,
    rng: np.random.Generator,
) -> dict[int, dict]:
    """
    Sweep temperature for several system sizes using Wolff.

    Returns
    -------
    dict mapping L -> {
        "T": array,
        "M": array,       mean |m|
        "chi": array,     susceptibility  L^2 * (<m²> - <|m|>²)
        "U4": array,      Binder cumulant 1 - <m⁴>/(3<m²>²)
    }
    """
    result: dict[int, dict] = {}

    for L in sizes:
        M_arr = np.zeros(len(T_values))
        chi_arr = np.zeros(len(T_values))
        U4_arr = np.zeros(len(T_values))
        N = L * L

        for iT, T in enumerate(T_values):
            spins = _random_spins(L, rng)
            # Thermalise
            wolff_sweep(spins, T, n_therm, rng)
            # Measure
            ms, m2s, m4s = [], [], []
            for _ in range(n_meas):
                wolff_step(spins, T, rng)
                obs = _observables(spins)
                ms.append(abs(obs["m"]))
                m2s.append(obs["m2"])
                m4s.append(obs["m4"])
            m_mean = float(np.mean(ms))
            m2_mean = float(np.mean(m2s))
            m4_mean = float(np.mean(m4s))
            M_arr[iT] = m_mean
            chi_arr[iT] = N * (m2_mean - m_mean ** 2)
            U4_arr[iT] = 1.0 - m4_mean / (3.0 * m2_mean ** 2 + 1e-30)

        result[L] = {
            "T": T_values.copy(),
            "M": M_arr,
            "chi": chi_arr,
            "U4": U4_arr,
        }

    return result


def finite_size_collapse_ising(sweep: dict[int, dict]) -> dict[int, dict]:
    """
    Apply Ising 2-D FSS rescaling:
        x = (T - T_c) * L^{1/ν}
        y = M * L^{β/ν}
    """
    result: dict[int, dict] = {}
    for L, data in sweep.items():
        x = (data["T"] - T_C) * L ** (1.0 / NU_ISING)
        y = data["M"] * L ** (BETA_ISING / NU_ISING)
        result[L] = {"x": x, "y": y}
    return result
