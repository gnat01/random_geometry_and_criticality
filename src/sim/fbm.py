"""Fractional Brownian motion via Davies-Harte circulant embedding."""

from __future__ import annotations

import numpy as np


def _fgn_1d(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    # Autocovariance of fGn
    k = np.arange(n + 1, dtype=np.float64)
    gamma = 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))

    # Circulant row of length 2n
    row = np.empty(2 * n, dtype=np.float64)
    row[:n + 1] = gamma
    row[n + 1:] = gamma[n - 1:0:-1]

    # Eigenvalues via real FFT; clamp to non-negative
    lam = np.real(np.fft.fft(row))
    lam = np.maximum(lam, 0.0)

    # Build complex Gaussian noise W in spectral domain
    W = np.empty(2 * n, dtype=np.complex128)
    z0 = rng.standard_normal()
    zn = rng.standard_normal()
    a = rng.standard_normal(n - 1)
    b = rng.standard_normal(n - 1)

    W[0] = np.sqrt(lam[0]) * z0
    W[n] = np.sqrt(lam[n]) * zn
    W[1:n] = np.sqrt(lam[1:n] / 2.0) * (a + 1j * b)
    W[n + 1:] = np.conj(W[n - 1:0:-1])

    # Inverse FFT and scale
    fgn_full = np.real(np.fft.ifft(W) * np.sqrt(2 * n))
    return fgn_full[:n]


def fbm_positions(
    n_walkers: int,
    n_steps: int,
    H: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    2-D fractional Brownian motion via Davies-Harte method.

    Parameters
    ----------
    n_walkers : int
    n_steps : int
        Number of increments. Positions include t=0 (zero) through t=n_steps.
    H : float
        Hurst exponent in (0, 1).
    rng : np.random.Generator

    Returns
    -------
    positions : ndarray of shape (n_walkers, n_steps+1, 2)
        positions[:, 0, :] == 0. positions[:, t, :] is cumulative sum of t increments.
    """
    positions = np.zeros((n_walkers, n_steps + 1, 2), dtype=np.float64)

    for w in range(n_walkers):
        inc_x = _fgn_1d(n_steps, H, rng)
        inc_y = _fgn_1d(n_steps, H, rng)
        positions[w, 1:, 0] = np.cumsum(inc_x)
        positions[w, 1:, 1] = np.cumsum(inc_y)

    return positions
