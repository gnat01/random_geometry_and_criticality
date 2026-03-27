"""Chapter IV: The Critical Point."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..sim.criticality import (
    P_C, NU, BETA, TAU,
    order_parameter_sweep,
    cluster_size_distribution,
    finite_size_collapse,
)
from ..theme import BG, MSD_C, THEO_C, TEXT_C, PANEL, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited


_COLORS = ["#ff6b6b", "#a78bfa", "#4a9fd4", "#4caf50"]


def render() -> None:
    mark_visited("chapter_4")
    chapter_header(
        "IV",
        "The Critical Point",
        "Phase transitions, exact exponents, and scaling collapse",
    )

    st.markdown("""
Percolation is not just a curiosity about random graphs. It is a prototype for *all*
second-order phase transitions. The story of the critical point is the story of scale
invariance: near $p_c$, the system looks the same on every length scale, which is why
power laws appear everywhere.

The order parameter is $P_\\infty(p)$, the fraction of nodes in the largest connected
component in the thermodynamic limit. Below $p_c$, $P_\\infty = 0$. Above $p_c$, it
grows as:

$$P_\\infty(p) \\sim (p - p_c)^\\beta, \\quad \\beta = \\frac{5}{36} \\approx 0.139.$$

This tiny exponent means the order parameter turns on very slowly after the transition.
The correlation length diverges on both sides:

$$\\xi(p) \\sim |p - p_c|^{-\\nu}, \\quad \\nu = \\frac{4}{3}.$$

On a finite system of size $L$, the transition is rounded: $P_\\infty$ and the
susceptibility $S(p)$ peak at a pseudo-critical point $p_c(L)$ that shifts toward
the true $p_c$ as $L \\to \\infty$.
""")

    definition_box(
        "Susceptibility",
        r"$S(p) = \sum_{s \neq s_\mathrm{max}} s^2 n_s / L^2$ — "
        "the mean finite-cluster size, excluding the giant. "
        "It peaks at $p_c$ where clusters of all sizes are present simultaneously."
    )

    st.markdown("""
---

### Sub-experiment A: order parameter and susceptibility
""")
    _section_a()

    st.markdown("""
---

### Sub-experiment B: cluster-size distribution

At $p_c$, the number of clusters of size $s$ per lattice site follows a power law:

$$n_s \\sim s^{-\\tau}, \\quad \\tau = \\frac{187}{91} \\approx 2.055.$$

Away from $p_c$, this power law is cut off exponentially at a characteristic size
$s^* \\sim |p - p_c|^{-1/\\sigma}$.
""")
    _section_b()

    st.markdown("""
---

### Sub-experiment C: finite-size scaling collapse

The cleanest test of universality is the **data collapse**. Near $p_c$, the only
relevant length scale is $\\xi \\sim |p - p_c|^{-\\nu}$. When $\\xi \\sim L$, the system
"feels" the boundary, so all the finite-$L$ curves share a universal scaling form:

$$P_\\infty(p, L) = L^{-\\beta/\\nu} \\, f\\!\\left((p - p_c) L^{1/\\nu}\\right).$$

Plotting $y = P_\\infty \\cdot L^{\\beta/\\nu}$ against $x = (p - p_c) \\cdot L^{1/\\nu}$
should collapse all curves onto a single universal function $f$.
""")
    _section_c()

    key_result(
        "2D bond percolation exact exponents: $p_c = 1/2$, $\\nu = 4/3$, "
        "$\\beta = 5/36 \\approx 0.139$, $\\tau = 187/91 \\approx 2.055$. "
        "These are universal — they don't change if you switch lattice geometry."
    )

    st.markdown("---")
    prev_next("chapter_4")


# ---------------------------------------------------------------------------
# Sub-experiment A
# ---------------------------------------------------------------------------

def _section_a() -> None:
    st.markdown("""
Run a sweep of bond probability $p$ for three system sizes and watch how $P_\\infty$
and the susceptibility $S$ sharpen as $L$ increases. The susceptibility peak locates $p_c$.
""")

    if st.button("▶ Run order parameter sweep", key="ch4a_run"):
        with st.spinner("Sweeping p for L = 15, 20, 25…"):
            rng = np.random.default_rng(42)
            p_values = np.linspace(0.30, 0.70, 20)
            sweep = order_parameter_sweep(
                sizes=[15, 20, 25],
                p_values=p_values,
                n_samples=20,
                rng=rng,
            )
        st.session_state["ch4a_sweep"] = sweep
        _show_a(sweep)
    elif "ch4a_sweep" in st.session_state:
        _show_a(st.session_state["ch4a_sweep"])


def _show_a(sweep: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    for i, (L, data) in enumerate(sorted(sweep.items())):
        c = _COLORS[i % len(_COLORS)]
        p = data["p"]
        P = data["P_inf"]
        S = data["S"]
        axes[0].plot(p, P, "o-", color=c, lw=1.8, ms=5, label=f"$L = {L}$")
        axes[1].plot(p, S, "o-", color=c, lw=1.8, ms=5, label=f"$L = {L}$")

    for ax, title, ylabel in [
        (axes[0], "Order parameter $P_\\infty(p)$", "$P_\\infty$"),
        (axes[1], "Susceptibility $S(p)$", "$S$"),
    ]:
        ax.axvline(P_C, color=THEO_C, lw=1.5, ls="--", alpha=0.7, label=f"$p_c = {P_C}$")
        ax.set_xlabel("$p$", color=TEXT_C, fontsize=12)
        ax.set_ylabel(ylabel, color=TEXT_C, fontsize=12)
        ax.set_title(title, color=TEXT_C, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Locate susceptibility peaks
    peak_info = []
    for L, data in sorted(sweep.items()):
        peak_idx = int(np.argmax(data["S"]))
        peak_p = float(data["p"][peak_idx])
        peak_info.append((L, peak_p))

    cols = st.columns(len(peak_info))
    for col, (L, pp) in zip(cols, peak_info):
        with col:
            st.metric(f"$S$ peak at $L={L}$", f"$p = {pp:.3f}$",
                      delta=f"{(pp - P_C)*1000:.1f} × 10⁻³ from $p_c$",
                      delta_color="off")


# ---------------------------------------------------------------------------
# Sub-experiment B
# ---------------------------------------------------------------------------

def _section_b() -> None:
    st.markdown("""
The slope on the log-log plot below should approach $-\\tau \\approx -2.055$ at $p_c$.
Off-critical distributions show an exponential cutoff.
""")

    if st.button("▶ Run cluster-size distribution", key="ch4b_run"):
        with st.spinner("Sampling cluster distributions at p = 0.40, 0.50, 0.60…"):
            rng = np.random.default_rng(99)
            dist = cluster_size_distribution(
                size=22,
                p_values=[0.40, 0.50, 0.60],
                n_samples=80,
                rng=rng,
            )
        st.session_state["ch4b_dist"] = dist
        _show_b(dist)
    elif "ch4b_dist" in st.session_state:
        _show_b(st.session_state["ch4b_dist"])


def _show_b(dist: dict) -> None:
    colors_b = {0.40: MSD_C, 0.50: THEO_C, 0.60: RMS_C}
    labels_b = {0.40: "$p = 0.40$ (below $p_c$)", 0.50: "$p_c = 0.50$", 0.60: "$p = 0.60$ (above $p_c$)"}

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(PANEL)

    for p_val, data in dist.items():
        c = colors_b.get(p_val, TEXT_C)
        lbl = labels_b.get(p_val, f"$p = {p_val:.2f}$")
        ax.scatter(data["centers"], data["ns"], s=12, color=c, alpha=0.8, label=lbl, zorder=3)
        ax.plot(data["centers"], data["ns"], color=c, alpha=0.35, lw=1.2, zorder=2)

    # Theory power law at p_c
    s_arr = np.logspace(0, 3.5, 100)
    ref_p = 0.50
    if ref_p in dist and len(dist[ref_p]["centers"]) > 0:
        ref_s = dist[ref_p]["centers"][len(dist[ref_p]["centers"]) // 2]
        ref_ns = dist[ref_p]["ns"][len(dist[ref_p]["ns"]) // 2]
        A = ref_ns * ref_s**TAU
        ax.plot(s_arr, A * s_arr**(-TAU), "--", color=THEO_C, lw=2, alpha=0.6,
                label=f"$n_s \\sim s^{{-\\tau}},\\ \\tau = {TAU:.3f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cluster size $s$", color=TEXT_C, fontsize=12)
    ax.set_ylabel("$n_s$ (per site per unit size)", color=TEXT_C, fontsize=12)
    ax.set_title("Cluster-size distribution", color=TEXT_C, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        f"At $p_c$, the power-law tail should follow $n_s \\sim s^{{-{TAU:.3f}}}$. "
        "You can see the other two curves cut off exponentially — they lack scale-free clusters."
    )


# ---------------------------------------------------------------------------
# Sub-experiment C
# ---------------------------------------------------------------------------

def _section_c() -> None:
    st.markdown("""
If the collapse works, curves at $L = 15, 22, 30$ should all overlap — confirming that
the correlation-length exponent $\\nu = 4/3$ is correct and that the only scale near $p_c$
is $\\xi$.
""")

    if st.button("▶ Run finite-size collapse", key="ch4c_run"):
        with st.spinner("Running FSS for L = 15, 22, 30 with 25 p-values…"):
            rng = np.random.default_rng(77)
            p_values = np.linspace(0.30, 0.70, 25)
            sweep = order_parameter_sweep(
                sizes=[15, 22, 30],
                p_values=p_values,
                n_samples=20,
                rng=rng,
            )
            collapse = finite_size_collapse(sweep)
        st.session_state["ch4c_sweep"] = sweep
        st.session_state["ch4c_collapse"] = collapse
        _show_c(sweep, collapse)
    elif "ch4c_sweep" in st.session_state:
        _show_c(st.session_state["ch4c_sweep"], st.session_state["ch4c_collapse"])


def _show_c(sweep: dict, collapse: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    for i, (L, data) in enumerate(sorted(sweep.items())):
        c = _COLORS[i % len(_COLORS)]
        axes[0].plot(data["p"], data["P_inf"], "o-", color=c, lw=1.8, ms=5, label=f"$L = {L}$")

    axes[0].axvline(P_C, color=THEO_C, lw=1.5, ls="--", alpha=0.7, label=f"$p_c = {P_C}$")
    axes[0].set_xlabel("$p$", color=TEXT_C, fontsize=12)
    axes[0].set_ylabel("$P_\\infty(p, L)$", color=TEXT_C, fontsize=12)
    axes[0].set_title("Before collapse (raw)", color=TEXT_C, fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for i, (L, cdata) in enumerate(sorted(collapse.items())):
        c = _COLORS[i % len(_COLORS)]
        axes[1].plot(cdata["x"], cdata["y"], "o-", color=c, lw=1.8, ms=5, label=f"$L = {L}$")

    axes[1].set_xlabel(
        f"$(p - p_c) \\cdot L^{{1/\\nu}} = (p - {P_C}) \\cdot L^{{3/4}}$",
        color=TEXT_C, fontsize=10,
    )
    axes[1].set_ylabel(
        f"$P_\\infty \\cdot L^{{\\beta/\\nu}} = P_\\infty \\cdot L^{{5/48}}$",
        color=TEXT_C, fontsize=10,
    )
    axes[1].set_title(
        f"After FSS collapse ($\\nu = {NU:.3f},\\ \\beta = {BETA:.3f}$)",
        color=TEXT_C, fontsize=11,
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        "The right panel is the collapse. If it works, all three curves overlap "
        "in the centre of the plot. Any residual spread comes from finite-size corrections "
        "that vanish as $L \\to \\infty$."
    )
