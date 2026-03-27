"""Chapter IX: Three-Dimensional Percolation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..sim.percolation3d import (
    P_C_3D, NU_3D, BETA_3D, GAMMA_3D,
    order_parameter_sweep_3d,
    finite_size_collapse_3d,
    lcc_positions_3d,
)
from ..theme import BG, THEO_C, TEXT_C, PANEL, MSD_C, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited

_COLORS = ["#ff6b6b", "#a78bfa", "#4a9fd4", "#4caf50"]


def render() -> None:
    mark_visited("chapter_9")
    chapter_header(
        "IX",
        "Three-Dimensional Percolation",
        "From the 2-D exact solution to 3-D numerics — universality without exact exponents",
    )

    st.markdown(r"""
Chapters III and IV studied bond percolation on the **2-D square lattice**, where exact results
are available: $p_c = 1/2$, $\nu = 4/3$, $\beta = 5/36$. Moving to 3-D removes the luxury of
exact solutions, but the critical structure remains — power laws, scale invariance, and a
universal finite-size-scaling collapse. Only the numbers change.

The **simple cubic lattice** has coordination number $z = 6$. The threshold is lower than in 2-D
because each site has more neighbours, so connectivity percolates at a smaller bond probability:

$$p_c^{\mathrm{3D}} \approx 0.2488 \quad (\text{bond, cubic lattice}).$$

The exponents are **not** rational fractions. They are determined numerically via high-statistics
Monte Carlo:

| Exponent | 2-D (exact) | 3-D (numerical) |
|---|---|---|
| $\nu$ | $4/3 \approx 1.333$ | $\approx 0.876$ |
| $\beta$ | $5/36 \approx 0.139$ | $\approx 0.418$ |
| $\gamma$ | $43/18 \approx 2.389$ | $\approx 1.793$ |

The larger $\beta$ in 3-D means the order parameter turns on **faster** after the transition —
the giant component grows more steeply once $p > p_c$. The smaller $\nu$ means correlations
diverge more slowly, reflecting the larger effective dimension.

**Upper critical dimension.** Percolation has upper critical dimension $d_u = 6$. For $d \geq 6$,
mean-field exponents apply ($\nu = 1/2$, $\beta = 1$). So 3-D sits between 2-D and the
mean-field regime — it is firmly in a non-trivial universality class.
""")

    definition_box(
        "Universality class",
        r"Two systems are in the same universality class if they share identical critical "
        r"exponents. The exponents depend only on the spatial dimension $d$ and the symmetry "
        r"of the order parameter — not on microscopic details like the lattice type or bond "
        r"vs site percolation.",
    )

    st.markdown("""
---

### Experiment A — Order parameter and susceptibility sweep
""")
    _section_a()

    st.markdown(r"""
---

### Experiment B — Finite-size scaling collapse

The same FSS logic from Chapter IV applies in 3-D with the 3-D exponents:

$$P_\infty(p, L) = L^{-\beta/\nu}\, f\!\left((p - p_c)\, L^{1/\nu}\right).$$

Plotting $P_\infty \cdot L^{\beta/\nu}$ against $(p - p_c) \cdot L^{1/\nu}$ should collapse
all system sizes onto a single curve — but with $\nu \approx 0.876$ instead of $4/3$.
""")
    _section_b()

    st.markdown("""
---

### Experiment C — 3-D largest connected component
""")
    _section_c()

    key_result(
        r"3-D cubic bond percolation: $p_c \approx 0.2488$, $\nu \approx 0.876$, "
        r"$\beta \approx 0.418$. The exponents are irrational — determined numerically. "
        r"FSS collapse still works with the correct 3-D exponents, confirming universality "
        r"in three dimensions."
    )

    st.markdown("---")
    prev_next("chapter_9")


# ---------------------------------------------------------------------------
# Section A
# ---------------------------------------------------------------------------

def _section_a() -> None:
    col_a, col_b = st.columns(2)
    with col_a:
        L_list_str = st.selectbox(
            "System sizes L", ["6, 8, 10", "8, 10, 12"], index=0, key="ch9_sizes"
        )
    with col_b:
        n_samples = st.slider("Samples per (L, p)", 5, 30, 10, step=5, key="ch9_ns")

    if st.button("▶ Run order parameter sweep", key="ch9a_run"):
        sizes = [int(x.strip()) for x in L_list_str.split(",")]
        p_values = np.linspace(0.15, 0.35, 18)
        rng = np.random.default_rng(42)
        with st.spinner(f"Sweeping p for L = {sizes}…"):
            sweep = order_parameter_sweep_3d(sizes, p_values, n_samples, rng)
        st.session_state["ch9a_sweep"] = sweep
        _show_a(sweep)
    elif "ch9a_sweep" in st.session_state:
        _show_a(st.session_state["ch9a_sweep"])


def _show_a(sweep: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    for i, (L, data) in enumerate(sorted(sweep.items())):
        c = _COLORS[i % len(_COLORS)]
        axes[0].plot(data["p"], data["P_inf"], "o-", color=c, lw=1.8, ms=5, label=f"$L={L}$")
        axes[1].plot(data["p"], data["chi"], "o-", color=c, lw=1.8, ms=5, label=f"$L={L}$")

    for ax, title, ylabel in [
        (axes[0], "Order parameter $P_\\infty(p)$", "$P_\\infty$"),
        (axes[1], "Susceptibility $\\chi(p)$", "$\\chi$"),
    ]:
        ax.axvline(P_C_3D, color=THEO_C, lw=1.5, ls="--", alpha=0.7, label=f"$p_c \\approx {P_C_3D}$")
        ax.set_xlabel("$p$", color=TEXT_C, fontsize=12)
        ax.set_ylabel(ylabel, color=TEXT_C, fontsize=12)
        ax.set_title(title, color=TEXT_C, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Susceptibility peak locations
    cols = st.columns(len(sweep))
    for col, (L, data) in zip(cols, sorted(sweep.items())):
        with col:
            peak_idx = int(np.argmax(data["chi"]))
            peak_p = float(data["p"][peak_idx])
            st.metric(f"$\\chi$ peak, $L={L}$", f"$p = {peak_p:.3f}$",
                      delta=f"{(peak_p - P_C_3D)*1000:.1f}×10⁻³ from $p_c$",
                      delta_color="off")


# ---------------------------------------------------------------------------
# Section B
# ---------------------------------------------------------------------------

def _section_b() -> None:
    if st.button("▶ Run FSS collapse", key="ch9b_run"):
        sizes = [6, 8, 10]
        p_values = np.linspace(0.15, 0.35, 22)
        rng = np.random.default_rng(77)
        with st.spinner("Running FSS for L = 6, 8, 10…"):
            sweep = order_parameter_sweep_3d(sizes, p_values, 12, rng)
            collapse = finite_size_collapse_3d(sweep)
        st.session_state["ch9b_sweep"] = sweep
        st.session_state["ch9b_collapse"] = collapse
        _show_b(sweep, collapse)
    elif "ch9b_sweep" in st.session_state:
        _show_b(st.session_state["ch9b_sweep"], st.session_state["ch9b_collapse"])


def _show_b(sweep: dict, collapse: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    for i, (L, data) in enumerate(sorted(sweep.items())):
        c = _COLORS[i % len(_COLORS)]
        axes[0].plot(data["p"], data["P_inf"], "o-", color=c, lw=1.8, ms=5, label=f"$L={L}$")

    axes[0].axvline(P_C_3D, color=THEO_C, lw=1.5, ls="--", alpha=0.7, label=f"$p_c \\approx {P_C_3D}$")
    axes[0].set_xlabel("$p$", color=TEXT_C, fontsize=12)
    axes[0].set_ylabel("$P_\\infty$", color=TEXT_C, fontsize=12)
    axes[0].set_title("Before collapse (raw)", color=TEXT_C, fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for i, (L, cdata) in enumerate(sorted(collapse.items())):
        c = _COLORS[i % len(_COLORS)]
        axes[1].plot(cdata["x"], cdata["y"], "o-", color=c, lw=1.8, ms=5, label=f"$L={L}$")

    axes[1].set_xlabel(
        f"$(p - p_c) \\cdot L^{{1/\\nu}}$  ($\\nu \\approx {NU_3D}$)",
        color=TEXT_C, fontsize=10,
    )
    axes[1].set_ylabel(
        f"$P_\\infty \\cdot L^{{\\beta/\\nu}}$  ($\\beta \\approx {BETA_3D}$)",
        color=TEXT_C, fontsize=10,
    )
    axes[1].set_title(
        f"FSS collapse ($\\nu={NU_3D}$, $\\beta={BETA_3D}$)",
        color=TEXT_C, fontsize=11,
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        "Collapse quality is limited by the small system sizes needed for browser speed. "
        "With $L \\geq 20$ and thousands of samples the collapse is very tight."
    )


# ---------------------------------------------------------------------------
# Section C
# ---------------------------------------------------------------------------

def _section_c() -> None:
    st.markdown("""
Visualise the largest connected component (LCC) of a single 3-D percolation realisation.
Colour encodes the z-coordinate (depth). Near $p_c$ the LCC is a fractal; well above it fills
the volume.
""")

    col1, col2 = st.columns(2)
    with col1:
        L_vis = st.slider("Lattice size L", 8, 20, 12, step=2, key="ch9c_L")
    with col2:
        p_vis = st.slider("Bond probability p", 0.10, 0.50, float(P_C_3D), step=0.01,
                          key="ch9c_p")

    if st.button("▶ Generate 3-D LCC", key="ch9c_run"):
        rng = np.random.default_rng(99)
        with st.spinner("Running percolation…"):
            xyz = lcc_positions_3d(L_vis, p_vis, rng)
        st.session_state["ch9c_xyz"] = xyz
        st.session_state["ch9c_meta"] = (L_vis, p_vis)
        _show_c(xyz, L_vis, p_vis)
    elif "ch9c_xyz" in st.session_state:
        _show_c(
            st.session_state["ch9c_xyz"],
            *st.session_state["ch9c_meta"],
        )


def _show_c(xyz: np.ndarray, L: int, p: float) -> None:
    if len(xyz) == 0:
        st.warning("LCC is empty — try a higher p.")
        return

    fig = plt.figure(figsize=(7, 6), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)

    frac = float(len(xyz)) / L ** 3
    z_norm = xyz[:, 2] / (L - 1 + 1e-9)
    ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=z_norm, cmap="plasma", s=8, alpha=0.6, edgecolors="none",
    )
    ax.set_title(
        f"LCC: {len(xyz)} / {L**3} sites ({frac*100:.1f}%)  |  p = {p:.3f}",
        color=TEXT_C, fontsize=10,
    )
    ax.tick_params(colors=TEXT_C, labelsize=7)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("LCC size", f"{len(xyz):,}")
    with col2:
        st.metric("Fraction of volume", f"{frac*100:.1f}%")
    with col3:
        st.metric("$p_c$ reference", f"{P_C_3D:.4f}")
