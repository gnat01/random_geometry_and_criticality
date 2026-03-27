"""Chapter XVIII: Directed Percolation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..sim.directed_perc import (
    P_C_DP, BETA_DP, NU_PERP, NU_PAR,
    run_dp_spacetime,
    density_scan,
    density_decay,
)
from ..theme import BG, THEO_C, TEXT_C, PANEL, MSD_C, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited

_COLORS = ["#ff6b6b", "#a78bfa", "#4a9fd4", "#4caf50"]


def render() -> None:
    mark_visited("chapter_18")
    chapter_header(
        "XVIII",
        "Directed Percolation",
        "Time-directed connectivity, the DP universality class, and epidemic spreading",
    )

    st.markdown(r"""
Standard percolation treats space isotropically — bonds can carry information in any direction.
**Directed percolation** (DP) breaks this symmetry: bonds only point in one preferred direction,
which we identify with **time**. At each step, an active site at $(t, x)$ can activate $(t+1, x)$
(straight bond) or $(t+1, x+1)$ (diagonal bond), each with probability $p$.

This is not just a technical modification. Directed percolation describes a broad class of
physical phenomena where *time has an arrow*:

- Epidemic spreading (can a disease persist forever, or go extinct?)
- Forest-fire fronts (does a fire spread or die out?)
- Interface depinning (does a driven interface advance or get stuck?)
- Reaction-diffusion systems at the active-absorbing transition

All these systems belong to the **DP universality class**, characterised by exponents:

| Exponent | Symbol | Value (1+1D) |
|---|---|---|
| Order parameter | $\beta$ | $\approx 0.2765$ |
| Spatial correlation length | $\nu_\perp$ | $\approx 1.097$ |
| Temporal correlation length | $\nu_\parallel$ | $\approx 1.734$ |

At $p_c \approx 0.6447$: the active density $\rho(t) \to 0$ as a power law $\rho \sim t^{-\delta}$
with $\delta = \beta / \nu_\parallel \approx 0.160$.

**Why $p_c > 0.5$?** Because the diagonal bond shifts the connectivity rightward; the system
needs a higher occupation probability to maintain a percolating cluster compared to undirected
percolation.

**The absorbing state.** Once all sites are inactive, the system cannot recover — activity is
**absorbed**. This distinguishing feature places DP in a different universality class from
equilibrium critical points: detailed balance is broken, and the absorbing state is non-generic.
""")

    definition_box(
        "Directed percolation universality class",
        r"An active-absorbing phase transition belongs to the DP class when: (1) the "
        r"absorbing state is unique, (2) there is no additional conservation law or symmetry, "
        r"(3) the order parameter couples to fluctuations of the control parameter. "
        r"The Janssen–Grassberger conjecture states that all such transitions are in the DP class "
        r"unless a special symmetry (like particle-hole symmetry) is present.",
    )

    st.markdown("""
---

### Experiment A — Spacetime diagram
""")
    _section_a()

    st.markdown(r"""
---

### Experiment B — Order parameter sweep: active density vs p

The mean active density in the final row $\rho = \langle \text{active fraction at }t=T\rangle$
is the order parameter. It vanishes below $p_c$ and grows as $\rho \sim (p - p_c)^\beta$
above it.
""")
    _section_b()

    st.markdown(r"""
---

### Experiment C — Critical density decay $\rho(t) \sim t^{-\delta}$

At $p = p_c$, the active density decays to zero as a power law. Below $p_c$ it decays
exponentially. Above $p_c$ it saturates to a finite value.
""")
    _section_c()

    key_result(
        r"1+1D DP threshold $p_c \approx 0.6447$. Order parameter exponent $\beta \approx 0.277$. "
        r"Spatial exponent $\nu_\perp \approx 1.097$, temporal exponent $\nu_\parallel \approx 1.734$. "
        r"Critical density decay $\rho(t) \sim t^{-\delta}$ with $\delta = \beta/\nu_\parallel \approx 0.160$. "
        r"The DP universality class governs epidemic spreading, interface depinning, and "
        r"any active-absorbing transition without additional symmetry."
    )

    st.markdown("---")
    prev_next("chapter_18")


# ---------------------------------------------------------------------------
# Section A — Spacetime diagram
# ---------------------------------------------------------------------------

def _section_a() -> None:
    st.markdown("""
Each column is a spatial site, each row is a time step. White = active, black = inactive.
Watch how the active cluster spreads, contracts, or dies depending on p.
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        L_st = st.slider("Spatial size L", 50, 200, 100, step=50, key="ch18a_L")
    with col2:
        T_st = st.slider("Time steps T", 50, 300, 150, step=50, key="ch18a_T")
    with col3:
        p_st = st.slider("Bond probability p", 0.30, 0.90, float(round(P_C_DP, 2)), step=0.01,
                         key="ch18a_p")

    seed_mode = st.radio("Initial condition", ["Full row", "Single site"], horizontal=True,
                         key="ch18a_seed")
    seed_full = seed_mode == "Full row"

    if st.button("▶ Generate spacetime", key="ch18a_run"):
        rng = np.random.default_rng(42)
        with st.spinner("Running DP…"):
            activity = run_dp_spacetime(L_st, T_st, p_st, rng, seed_full=seed_full)
        st.session_state["ch18a_act"] = activity
        st.session_state["ch18a_p"] = p_st
        _show_a(activity, p_st)
    elif "ch18a_act" in st.session_state:
        _show_a(st.session_state["ch18a_act"], st.session_state["ch18a_p"])


def _show_a(activity: np.ndarray, p: float) -> None:
    T, L = activity.shape
    rho_t = activity.mean(axis=1)
    final_density = float(rho_t[-1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG,
                             gridspec_kw={"width_ratios": [3, 1]})
    for ax in axes:
        ax.set_facecolor(PANEL)

    axes[0].imshow(
        activity.astype(np.float32), cmap="binary_r",
        aspect="auto", interpolation="nearest",
        extent=[0, L, T, 0],
    )
    phase = "supercritical" if p > P_C_DP + 0.02 else ("subcritical" if p < P_C_DP - 0.02 else "critical")
    axes[0].set_title(f"Spacetime ($p={p:.3f}$, {phase})", color=TEXT_C, fontsize=10)
    axes[0].set_xlabel("Site $x$", color=TEXT_C, fontsize=10)
    axes[0].set_ylabel("Time $t$", color=TEXT_C, fontsize=10)

    axes[1].plot(rho_t, np.arange(T), color=THEO_C, lw=1.5)
    axes[1].axvline(0, color=TEXT_C, lw=0.5, alpha=0.3)
    axes[1].set_xlabel("$\\rho(t)$", color=TEXT_C, fontsize=10)
    axes[1].set_ylim(T, 0)
    axes[1].set_title("Active density", color=TEXT_C, fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final density $\\rho(T)$", f"{final_density:.3f}")
    with col2:
        st.metric("Survived?", "Yes" if final_density > 0 else "Extinct")
    with col3:
        st.metric("$p_c$ reference", f"{P_C_DP:.4f}")


# ---------------------------------------------------------------------------
# Section B — Order parameter sweep
# ---------------------------------------------------------------------------

def _section_b() -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        L_sw = st.slider("L", 80, 200, 120, step=40, key="ch18b_L")
    with col2:
        T_sw = st.slider("T", 100, 400, 200, step=100, key="ch18b_T")
    with col3:
        n_samp = st.slider("Samples per p", 5, 30, 10, step=5, key="ch18b_ns")

    if st.button("▶ Run order parameter sweep", key="ch18b_run"):
        p_values = np.linspace(0.45, 0.85, 20)
        rng = np.random.default_rng(42)
        with st.spinner(f"Sweeping p ({len(p_values)} values × {n_samp} samples)…"):
            result = density_scan(L_sw, T_sw, p_values, n_samp, rng)
        st.session_state["ch18b_res"] = result
        _show_b(result)
    elif "ch18b_res" in st.session_state:
        _show_b(st.session_state["ch18b_res"])


def _show_b(result: dict) -> None:
    p = result["p"]
    rho = result["rho"]
    surv = result["surv"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    axes[0].plot(p, rho, "o-", color=THEO_C, lw=1.8, ms=5, label="$\\rho$ (final density)")
    axes[0].axvline(P_C_DP, color=RMS_C, lw=1.5, ls="--", alpha=0.7, label=f"$p_c \\approx {P_C_DP}$")
    axes[0].set_xlabel("$p$", color=TEXT_C, fontsize=12)
    axes[0].set_ylabel("Active density $\\rho$", color=TEXT_C, fontsize=12)
    axes[0].set_title("Order parameter: $\\rho$ vs $p$", color=TEXT_C, fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(p, surv, "o-", color=MSD_C, lw=1.8, ms=5, label="Survival probability")
    axes[1].axvline(P_C_DP, color=RMS_C, lw=1.5, ls="--", alpha=0.7, label=f"$p_c \\approx {P_C_DP}$")
    axes[1].set_xlabel("$p$", color=TEXT_C, fontsize=12)
    axes[1].set_ylabel("Survival probability", color=TEXT_C, fontsize=12)
    axes[1].set_title("Fraction of time with activity", color=TEXT_C, fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Find empirical threshold
    above = rho > 0.01
    if above.any():
        p_thresh = float(p[above][0])
        st.metric("Empirical $p_c$ (first $\\rho > 0.01$)", f"{p_thresh:.3f}",
                  delta=f"{(p_thresh - P_C_DP)*1000:.1f}×10⁻³ from $p_c$",
                  delta_color="off")


# ---------------------------------------------------------------------------
# Section C — Critical decay
# ---------------------------------------------------------------------------

def _section_c() -> None:
    col1, col2 = st.columns(2)
    with col1:
        L_dc = st.slider("L", 100, 400, 200, step=100, key="ch18c_L")
    with col2:
        T_dc = st.slider("T", 200, 800, 400, step=200, key="ch18c_T")

    n_samp_dc = st.slider("Ensemble size", 10, 60, 20, step=10, key="ch18c_ns")

    if st.button("▶ Run density decay (p_c and flanks)", key="ch18c_run"):
        p_vals = [P_C_DP - 0.04, P_C_DP, P_C_DP + 0.04]
        rng = np.random.default_rng(77)
        results = {}
        for p_dc in p_vals:
            with st.spinner(f"Decay at p = {p_dc:.4f}…"):
                results[p_dc] = density_decay(L_dc, T_dc, p_dc, n_samp_dc, rng)
        st.session_state["ch18c_res"] = results
        _show_c(results)
    elif "ch18c_res" in st.session_state:
        _show_c(st.session_state["ch18c_res"])


def _show_c(results: dict) -> None:
    colors_dc = {
        min(results.keys()): MSD_C,
        P_C_DP: THEO_C,
        max(results.keys()): RMS_C,
    }
    # Match by closest key
    keys = sorted(results.keys())
    color_list = [MSD_C, THEO_C, RMS_C]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    for (p_dc, res), c in zip(sorted(results.items()), color_list):
        t = res["t"]
        rho_t = res["rho_t"]
        label = f"$p = {p_dc:.4f}$" + (" ($p_c$)" if abs(p_dc - P_C_DP) < 0.001 else "")
        # Linear
        axes[0].plot(t, rho_t, color=c, lw=1.5, label=label)
        # Log-log (only where rho > 0)
        valid = rho_t > 1e-6
        if valid.sum() > 2:
            axes[1].loglog(t[valid], rho_t[valid], color=c, lw=1.5, label=label)

    # Theory power law at p_c
    t_arr = np.logspace(1, np.log10(max(max(r["t"]) for r in results.values())), 50)
    delta_dp = BETA_DP / NU_PAR  # ≈ 0.160
    # Anchor to mid-t value of p_c curve
    pc_key = min(results.keys(), key=lambda k: abs(k - P_C_DP))
    pc_rho = results[pc_key]["rho_t"]
    pc_t = results[pc_key]["t"]
    mid = len(pc_t) // 3
    if mid > 0 and pc_rho[mid] > 1e-6:
        A = pc_rho[mid] * pc_t[mid] ** delta_dp
        axes[1].loglog(t_arr, A * t_arr ** (-delta_dp), "--",
                       color=THEO_C, lw=1.5, alpha=0.6,
                       label=f"$\\sim t^{{-\\delta}},\\ \\delta \\approx {delta_dp:.3f}$")

    axes[0].set_xlabel("$t$", color=TEXT_C, fontsize=12)
    axes[0].set_ylabel("$\\rho(t)$", color=TEXT_C, fontsize=12)
    axes[0].set_title("Active density (linear)", color=TEXT_C, fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("$t$  (log)", color=TEXT_C, fontsize=12)
    axes[1].set_ylabel("$\\rho(t)$  (log)", color=TEXT_C, fontsize=12)
    axes[1].set_title("Log-log: power law at $p_c$", color=TEXT_C, fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("$p_c$ (theory)", f"{P_C_DP:.4f}")
    with col2:
        st.metric("$\\beta$", f"{BETA_DP:.4f}")
    with col3:
        st.metric("$\\delta = \\beta/\\nu_\\parallel$", f"{BETA_DP/NU_PAR:.4f}")
