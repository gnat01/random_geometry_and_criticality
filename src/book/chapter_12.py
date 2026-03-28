"""Chapter XII: The Transfer Matrix."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..sim.transfermatrix import (
    T_C_ISING,
    correlation_length,
    correlation_length_scan,
    pseudo_critical_temps,
)
from ..theme import BG, THEO_C, TEXT_C, PANEL, MSD_C, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited

_COLORS = ["#ff6b6b", "#a78bfa", "#4a9fd4", "#4caf50", "#f9a825", "#26c6da"]


def render() -> None:
    mark_visited("chapter_12")
    chapter_header(
        "XII",
        "The Transfer Matrix",
        "Exact finite-strip solution, correlation length, and noise-free T_c extraction",
    )

    st.markdown(r"""
Monte Carlo measures the Ising model with statistical noise. The **transfer matrix** method
is exact on a finite strip — no noise, no thermalisation.

Consider an Ising strip of width $W$ (periodic in the width direction) and infinite length.
The partition function factorises row by row:

$$Z = \mathrm{Tr}\, \mathbf{T}^N,$$

where $\mathbf{T}$ is the $2^W \times 2^W$ transfer matrix with elements

$$T_{\alpha\beta} = \exp\!\left(K \sum_i \sigma_i^\alpha \sigma_i^\beta
  + \tfrac{K}{2}\bigl(h_\alpha + h_\beta\bigr)\right),$$

$K = J/k_BT$, and $h_\alpha = \sum_i \sigma_i^\alpha \sigma_{i+1}^\alpha$ is the horizontal
bond energy within row $\alpha$.

As $N \to \infty$, the largest eigenvalue $\lambda_1$ dominates the free energy.  The
**correlation length** along the strip is:

$$\xi(W, T) = \frac{1}{\ln(\lambda_1 / \lambda_2)}.$$

**Key property.** At criticality, $\xi$ grows like $W$ — the system has correlations on all
length scales.  Away from $T_c$, $\xi$ saturates to a finite value.  The crossing of
$\xi/W$ curves for different strip widths gives a clean, noise-free estimate of $T_c$.

**Computational cost.** Building and diagonalising a $2^W \times 2^W$ matrix scales as $O(2^{3W})$.
Strips up to $W \approx 12$ are feasible on a laptop.
""")

    definition_box(
        "Transfer matrix",
        r"A matrix whose largest eigenvalue gives the partition function of an infinite strip. "
        r"The gap between the two largest eigenvalues sets the inverse correlation length. "
        r"Transfer matrices provide exact finite-strip results and are the foundation of "
        r"conformal field theory extrapolations to the thermodynamic limit.",
    )

    st.markdown("""
---

### Experiment A — Correlation length ξ(T) for several strip widths
""")
    _section_a()

    st.markdown(r"""
---

### Experiment B — ξ/W crossing: estimating T_c

At $T_c$, the correlation length grows proportionally to $W$, so $\xi/W$ curves for different
widths cross at the same point. Plotting $\xi/W$ vs $T$ yields a precise estimate of $T_c$
that converges to the exact value as $W \to \infty$.
""")
    _section_b()

    st.markdown("""
---

### Experiment C — Transfer matrix at a glance
""")
    _section_c()

    key_result(
        r"Transfer matrix on an Ising strip: $\xi = 1/\ln(\lambda_1/\lambda_2)$. "
        r"$\xi/W$ curves cross at $T_c$ for all $W$. "
        r"Extrapolation of $T_c(W) \to T_c(\infty)$ converges to the exact value "
        r"$T_c = 2/\ln(1+\sqrt{2}) \approx 2.2692$. "
        r"The method is noise-free — no Monte Carlo statistics needed."
    )

    st.markdown("---")
    prev_next("chapter_12")


# ---------------------------------------------------------------------------
# Section A — ξ(T) curves
# ---------------------------------------------------------------------------

def _section_a() -> None:
    col1, col2 = st.columns(2)
    with col1:
        W_str = st.selectbox("Strip widths W", ["2, 3, 4, 5", "3, 4, 5, 6"], index=0, key="ch12a_W")
    with col2:
        n_T = st.slider("Temperature points", 20, 80, 40, step=10, key="ch12a_nT")

    if st.button("▶ Compute ξ(T)", key="ch12a_run"):
        widths = [int(w.strip()) for w in W_str.split(",")]
        T_values = np.linspace(1.5, 3.5, n_T)
        with st.spinner(f"Building transfer matrices for W = {widths}…"):
            scan = correlation_length_scan(widths, T_values)
        st.session_state["ch12a_scan"] = scan
        _show_a(scan)
    elif "ch12a_scan" in st.session_state:
        _show_a(st.session_state["ch12a_scan"])


def _show_a(scan: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    for i, (W, data) in enumerate(sorted(scan.items())):
        c = _COLORS[i % len(_COLORS)]
        # Cap ξ for display
        xi_plot = np.clip(data["xi"], 0, 200)
        axes[0].plot(data["T"], xi_plot, color=c, lw=1.8, label=f"$W={W}$")
        axes[1].plot(data["T"], data["xi_over_W"], color=c, lw=1.8, label=f"$W={W}$")

    for ax in axes:
        ax.axvline(T_C_ISING, color=THEO_C, lw=1.5, ls="--", alpha=0.7,
                   label=f"$T_c \\approx {T_C_ISING:.4f}$")
        ax.set_xlabel("$T$", color=TEXT_C, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("$\\xi(W, T)$", color=TEXT_C, fontsize=12)
    axes[0].set_title("Correlation length", color=TEXT_C, fontsize=11)
    axes[1].set_ylabel("$\\xi / W$", color=TEXT_C, fontsize=12)
    axes[1].set_title("$\\xi/W$ (should cross at $T_c$)", color=TEXT_C, fontsize=11)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section B — T_c(W) estimates
# ---------------------------------------------------------------------------

def _section_b() -> None:
    if st.button("▶ Estimate T_c(W) from ξ/W crossings", key="ch12b_run"):
        widths = [2, 3, 4, 5, 6]
        T_values = np.linspace(2.0, 2.6, 60)
        with st.spinner("Computing ξ/W maxima for W = 2 … 6…"):
            t_c_w = pseudo_critical_temps(widths, T_values)
        st.session_state["ch12b_tc"] = t_c_w
        _show_b(t_c_w)
    elif "ch12b_tc" in st.session_state:
        _show_b(st.session_state["ch12b_tc"])


def _show_b(t_c_w: dict) -> None:
    Ws = sorted(t_c_w.keys())
    Tcs = [t_c_w[W] for W in Ws]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.plot(Ws, Tcs, "o-", color=MSD_C, lw=1.8, ms=7, label="$T_c(W)$ from $\\xi/W$ max")
    ax.axhline(T_C_ISING, color=THEO_C, lw=1.5, ls="--", alpha=0.8,
               label=f"Exact $T_c = {T_C_ISING:.4f}$")
    ax.set_xlabel("Strip width $W$", color=TEXT_C, fontsize=12)
    ax.set_ylabel("$T_c(W)$", color=TEXT_C, fontsize=12)
    ax.set_title("Pseudo-critical temperature vs strip width", color=TEXT_C, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    cols = st.columns(len(Ws) + 1)
    for col, (W, Tc) in zip(cols[:len(Ws)], zip(Ws, Tcs)):
        with col:
            st.metric(f"$W={W}$", f"{Tc:.4f}",
                      delta=f"{(Tc - T_C_ISING)*1000:.1f}×10⁻³",
                      delta_color="off")
    with cols[-1]:
        st.metric("Exact $T_c$", f"{T_C_ISING:.4f}")


# ---------------------------------------------------------------------------
# Section C — Matrix viewer
# ---------------------------------------------------------------------------

def _section_c() -> None:
    st.markdown("""
Visualise the transfer matrix entries for a small strip width at a chosen temperature.
Each cell shows the Boltzmann weight for transitioning between two spin row configurations.
""")
    col1, col2 = st.columns(2)
    with col1:
        W_view = st.slider("Strip width W", 2, 4, 2, step=1, key="ch12c_W")
    with col2:
        T_view = st.slider("Temperature T", 1.5, 3.5, float(round(T_C_ISING, 2)), step=0.05,
                           key="ch12c_T")

    from ..sim.transfermatrix import build_transfer_matrix
    K = 1.0 / T_view
    T_mat = build_transfer_matrix(W_view, K)
    n = T_mat.shape[0]

    fig, ax = plt.subplots(figsize=(min(n * 0.7 + 1.5, 7), min(n * 0.7 + 1.5, 7)), facecolor=BG)
    ax.set_facecolor(PANEL)
    im = ax.imshow(T_mat, cmap="YlOrRd", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Label axes with spin configurations
    from ..sim.transfermatrix import _spin_configs
    cfgs = _spin_configs(W_view)
    labels = ["".join("+" if s > 0 else "-" for s in row) for row in cfgs]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, color=TEXT_C, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, color=TEXT_C, fontsize=8)
    ax.set_title(f"Transfer matrix: $W={W_view}$, $T={T_view:.2f}$, $K={K:.3f}$",
                 color=TEXT_C, fontsize=10)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    xi_val = correlation_length(W_view, T_view)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matrix size", f"{n} × {n}")
    with col2:
        st.metric(f"$\\xi(W={W_view}, T={T_view:.2f})$", f"{xi_val:.4f}")
    with col3:
        st.metric("$T_c$ (exact)", f"{T_C_ISING:.4f}")
