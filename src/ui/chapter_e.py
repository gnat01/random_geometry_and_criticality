"""Chapter E: critical phenomena at the 2D bond-percolation phase transition."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..sim.criticality import (
    P_C, NU, BETA, TAU, D_W_CRIT,
    order_parameter_sweep,
    cluster_size_distribution,
    finite_size_collapse,
    survival_vs_p,
    msd_exponent_vs_p,
)
from ..theme import TEXT_C, THEO_C, RMS_C, MSD_C, WALK_C, apply_mpl_style, fig_bg


# ---------------------------------------------------------------------------
# Colour helpers — diverging palette centred on p_c
# ---------------------------------------------------------------------------

def _p_colors(p_values: list[float], p_c: float = P_C) -> list:
    """
    Blue for p < p_c, green at p_c, red above.
    Uses a smooth interpolation through the RdYlGn_r colormap.
    """
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=0.35, vmax=0.70)
    return [cmap(norm(p)) for p in p_values]


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _order_parameter_fig(sweep: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=fig_bg())
    sizes = sorted(sweep.keys())
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(sizes) - 1, 1)) for i in range(len(sizes))]

    for color, L in zip(colors, sizes):
        d = sweep[L]
        ax1.plot(d["p"], d["P_inf"], color=color, lw=1.8, label=f"L = {L}")
        ax1.fill_between(d["p"],
                         np.maximum(d["P_inf"] - d["P_inf_std"], 0),
                         d["P_inf"] + d["P_inf_std"],
                         color=color, alpha=0.15)
        ax2.plot(d["p"], d["S"], color=color, lw=1.8, label=f"L = {L}")
        ax2.fill_between(d["p"],
                         np.maximum(d["S"] - d["S_std"], 0),
                         d["S"] + d["S_std"],
                         color=color, alpha=0.15)

    for ax in (ax1, ax2):
        ax.axvline(P_C, color=THEO_C, lw=1.2, ls="--", label=f"p_c = {P_C}")
        ax.set_xlabel("Bond probability p", color=TEXT_C)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    ax1.set_ylabel("P_∞  (giant component fraction)", color=TEXT_C)
    ax1.set_title("Order parameter P_∞(p)", color=TEXT_C, fontsize=11)

    ax2.set_ylabel("S  (mean finite-cluster size / L²)", color=TEXT_C)
    ax2.set_title("Susceptibility proxy S(p)", color=TEXT_C, fontsize=11)
    ax2.set_yscale("log")

    fig.tight_layout()
    return fig


def _cluster_dist_fig(dist: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=fig_bg())
    p_vals = sorted(dist.keys())
    colors = _p_colors(p_vals)

    # Theory slope at p_c
    ref_s = np.logspace(0, 3, 50)
    ref_ns = ref_s[0] ** TAU * ref_s ** (-TAU)  # normalised to first point

    for color, p in zip(colors, p_vals):
        d = dist[p]
        if len(d["centers"]) == 0:
            continue
        ax.loglog(d["centers"], d["ns"], "o-", color=color, ms=4,
                  lw=1.6, label=f"p = {p:.2f}")

    # Reference slope
    ax.loglog(ref_s, ref_ns * dist[p_vals[len(p_vals) // 2]]["ns"][0] /
              (ref_s[0] ** (-TAU)) * ref_s[0] ** (-TAU),
              "--", color=THEO_C, lw=1.3,
              label=f"slope −τ = −{TAU:.3f}")

    ax.set_xlabel("Cluster size s", color=TEXT_C)
    ax.set_ylabel("n_s  (clusters per site per unit s)", color=TEXT_C)
    ax.set_title("Cluster-size distribution n_s(s)", color=TEXT_C, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    return fig


def _collapse_fig(collapsed: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=fig_bg())
    sizes = sorted(collapsed.keys())
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(sizes) - 1, 1)) for i in range(len(sizes))]

    for color, L in zip(colors, sizes):
        d = collapsed[L]
        ax.plot(d["x"], d["y"], "o-", color=color, ms=3, lw=1.6, label=f"L = {L}")
        ax.fill_between(d["x"],
                        np.maximum(d["y"] - d["y_std"], 0),
                        d["y"] + d["y_std"],
                        color=color, alpha=0.12)

    ax.axvline(0, color=THEO_C, lw=1.0, ls="--", label="p = p_c")
    ax.set_xlabel(f"(p − p_c) · L^{{1/ν}},  ν = {NU:.3f}", color=TEXT_C)
    ax.set_ylabel(f"P_∞ · L^{{β/ν}},  β = {BETA:.4f}", color=TEXT_C)
    ax.set_title("Finite-size scaling collapse", color=TEXT_C, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _survival_fig(surv: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=fig_bg())
    p_vals = sorted(surv.keys())
    colors = _p_colors(p_vals)
    steps = np.arange(1, len(next(iter(surv.values()))) + 1)

    for color, p in zip(colors, p_vals):
        s = surv[p]
        mask = s > 0
        label = f"p = {p:.2f}"
        ax1.semilogy(steps[mask], s[mask], color=color, lw=1.8, label=label)
        if mask.sum() > 5:
            ax2.loglog(steps[mask], s[mask], color=color, lw=1.8, label=label)

    # Annotate expected power-law slope at p_c
    d_s = 2.0 * D_W_CRIT / (D_W_CRIT + 1.0)  # spectral dim ≈ 2d_f/d_w, heuristic
    t_ref = steps[steps > 5]
    s_ref = t_ref ** (-d_s / 2.0)
    # Normalise to middle of the pack for visibility
    mid_key = p_vals[len(p_vals) // 2]
    mid = surv[mid_key]
    mid_mask = mid > 0
    if mid_mask.sum() > 3:
        norm = mid[mid_mask][0] / (steps[mid_mask][0] ** (-d_s / 2.0))
        ax2.loglog(t_ref, norm * s_ref, "--", color=THEO_C, lw=1.2,
                   label=f"t^{{−d_s/2}}, d_s ≈ {d_s:.2f}")

    for ax in (ax1, ax2):
        ax.set_xlabel("Steps t", color=TEXT_C)
        ax.set_ylabel("Survival fraction S(t)", color=TEXT_C)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.2)

    ax1.set_title("Survival: log-linear  (exponential = straight line)", color=TEXT_C, fontsize=10)
    ax2.set_title("Survival: log-log  (power law = straight line)", color=TEXT_C, fontsize=10)
    fig.tight_layout()
    return fig


def _msd_exponent_fig(p_arr, beta_mean, beta_std) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=fig_bg())

    ax.errorbar(p_arr, beta_mean, yerr=beta_std,
                fmt="o-", color=RMS_C, lw=1.8, ms=6, capsize=4, label="Measured β = 2/d_w")

    ax.axvline(P_C, color=THEO_C, lw=1.2, ls="--", label=f"p_c = {P_C}")
    ax.axhline(1.0, color=WALK_C, lw=1.0, ls=":", label="β = 1  (normal diffusion)")
    ax.axhline(2.0 / D_W_CRIT, color=MSD_C, lw=1.0, ls=":",
               label=f"β = 2/d_w ≈ {2/D_W_CRIT:.2f}  (theory at p_c)")

    ax.set_xlabel("Bond probability p", color=TEXT_C)
    ax.set_ylabel("MSD exponent β  (MSD ~ t^β)", color=TEXT_C)
    ax.set_title("Anomalous diffusion exponent vs p", color=TEXT_C, fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_chapter_e():
    apply_mpl_style()
    st.markdown(
        "## Chapter E — Critical phenomena at the percolation transition\n"
        "Bond percolation on the square lattice undergoes a **second-order phase transition** "
        "at p_c = ½ (exact, by self-duality). At criticality, clusters exist at **every size** "
        "with no characteristic length scale, the correlation length diverges, "
        "and diffusion slows to a power law. "
        "Pick a mode to explore each facet."
    )

    _MODES = (
        "order_parameter",
        "cluster_geometry",
        "finite_size_collapse",
        "critical_dynamics",
    )
    _MODE_LABELS = {
        "order_parameter":    "Phase transition (P_∞ & susceptibility)",
        "cluster_geometry":   "Cluster-size distribution n_s",
        "finite_size_collapse": "Finite-size scaling collapse",
        "critical_dynamics":  "Critical slowing down (survival & MSD)",
    }
    _MODE_ENV = os.environ.get("FRACTAL_E_MODE", "order_parameter")
    default_mode = _MODE_ENV if _MODE_ENV in _MODES else "order_parameter"

    with st.sidebar:
        st.markdown("### Mode")
        mode = st.selectbox(
            "Observable",
            _MODES,
            index=_MODES.index(default_mode),
            format_func=lambda m: _MODE_LABELS[m],
        )

        st.markdown("### Lattice")
        perc_size = st.slider(
            "System size L (for dynamics modes)",
            10, 50,
            _env_int("FRACTAL_PERC_SIZE", 28),
            help="Grid side for survival and MSD modes. Sweep modes use L = 15, 25, 35 internally.",
        )
        n_samples = st.slider(
            "Samples per (L, p) point",
            10, 80,
            _env_int("FRACTAL_E_SAMPLES", 30),
            help="More samples → smoother curves, longer runtime.",
        )

    rng = np.random.default_rng(_env_int("FRACTAL_SEED", 42))

    # -----------------------------------------------------------------------
    if mode == "order_parameter":
        st.markdown(
            "Sweeping p from 0.30 to 0.70 for three system sizes. "
            "**Left:** The order parameter P_∞ = |LCC| / L² rises sharply at p_c. "
            "The transition sharpens as L → ∞ (finite-size rounding visible here). "
            "**Right:** The susceptibility proxy S(p) peaks at p_c — the percolation "
            "analog of magnetic susceptibility diverging at a critical point."
        )
        if st.button("▶ Run sweep", type="primary", key="run_e_op"):
            p_vals = np.linspace(0.30, 0.70, 28)
            sizes = [15, 25, 35]
            with st.spinner("Sweeping p for three system sizes…"):
                sweep = order_parameter_sweep(sizes, p_vals, n_samples=n_samples, rng=rng)
            st.pyplot(_order_parameter_fig(sweep))
            plt.close("all")
            st.caption(
                f"L ∈ {sizes}  ·  {len(p_vals)} p-values  ·  {n_samples} samples each  ·  "
                f"p_c = {P_C} (exact by self-duality)  ·  β = {BETA:.4f},  ν = {NU:.4f}"
            )

    # -----------------------------------------------------------------------
    elif mode == "cluster_geometry":
        p_at = st.multiselect(
            "p values to compare",
            [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.70],
            default=[0.45, 0.50, 0.55],
            help="At p_c (= 0.50) the distribution is a power law n_s ~ s^{-τ}.",
        )
        st.markdown(
            f"Log-binned n_s (clusters per site per unit size), giant component excluded. "
            f"At p_c the slope should match **−τ = −{TAU:.3f}** (Fisher exponent). "
            f"Off-critical, an exponential cutoff truncates the power law at s* ~ |p − p_c|^{{−1/σ}}."
        )
        if p_at and st.button("▶ Run", type="primary", key="run_e_cg"):
            with st.spinner("Sampling cluster sizes…"):
                dist = cluster_size_distribution(
                    perc_size, [float(p) for p in p_at],
                    n_samples=max(n_samples * 3, 100), rng=rng
                )
            st.pyplot(_cluster_dist_fig(dist))
            plt.close("all")
            st.caption(
                f"L = {perc_size}  ·  {max(n_samples*3, 100)} samples  ·  "
                f"τ = {TAU:.4f} (exact)"
            )

    # -----------------------------------------------------------------------
    elif mode == "finite_size_collapse":
        st.markdown(
            "Rescaling P_∞(p, L) by the exact 2D exponents collapses all curves onto a "
            "**single universal function** — the hallmark of a second-order phase transition. "
            f"Axes: x = (p − p_c) · L^{{1/ν}}, y = P_∞ · L^{{β/ν}} with "
            f"p_c = {P_C}, ν = {NU:.4f}, β = {BETA:.4f}."
        )
        if st.button("▶ Run collapse", type="primary", key="run_e_fss"):
            p_vals = np.linspace(0.32, 0.68, 30)
            sizes = [15, 22, 30, 40]
            with st.spinner("Building collapse dataset…"):
                sweep = order_parameter_sweep(sizes, p_vals, n_samples=n_samples, rng=rng)
                collapsed = finite_size_collapse(sweep)
            st.pyplot(_collapse_fig(collapsed))
            plt.close("all")
            st.caption(
                f"L ∈ {sizes}  ·  {len(p_vals)} p-values  ·  {n_samples} samples each. "
                f"Residual spread is due to finite-size corrections to scaling."
            )

    # -----------------------------------------------------------------------
    elif mode == "critical_dynamics":
        p_dyn = st.multiselect(
            "p values",
            [0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.70],
            default=[0.45, 0.50, 0.55, 0.70],
            help="Green ≈ p_c; blue below; red above.",
        )
        trap_density = st.slider("Trap density", 0.01, 0.15, 0.05, 0.01,
                                  help="Fraction of cluster nodes that are absorbing traps.")

        st.markdown(
            "**Top:** survival S(t) shown on log-linear (left) and log-log (right) axes. "
            "An exponential decay is a straight line on log-linear; "
            "a power law is straight on log-log. At p_c the log-log plot should be linear. "
            "**Bottom:** MSD anomalous exponent β = 2/d_w across p, "
            f"with theory lines at β = 1 (normal) and β ≈ {2/D_W_CRIT:.2f} (critical)."
        )

        if p_dyn and st.button("▶ Run", type="primary", key="run_e_dyn"):
            p_list = sorted([float(p) for p in p_dyn])
            n_graphs = 5

            with st.spinner("Running survival ensembles…"):
                surv = survival_vs_p(
                    perc_size, p_list,
                    trap_density=trap_density,
                    n_walkers=200,
                    n_steps=500,
                    n_graphs=n_graphs,
                    rng=rng,
                )
            st.pyplot(_survival_fig(surv))
            plt.close("all")

            with st.spinner("Estimating MSD exponents…"):
                p_arr, beta_mean, beta_std = msd_exponent_vs_p(
                    perc_size, p_list,
                    n_walkers=60, n_steps=600,
                    n_graphs=n_graphs, rng=rng,
                )
            st.pyplot(_msd_exponent_fig(p_arr, beta_mean, beta_std))
            plt.close("all")

            st.caption(
                f"L = {perc_size}  ·  {n_graphs} graph realisations per p  ·  "
                f"traps: {trap_density:.0%} of nodes  ·  "
                f"d_w(p_c) ≈ {D_W_CRIT}  →  β ≈ {2/D_W_CRIT:.2f}"
            )
