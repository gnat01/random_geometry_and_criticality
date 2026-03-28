"""Chapter D: DLA on bond-percolation substrates with ensemble analysis."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import streamlit as st

from ..graphs.percolation import build_percolation
from ..sim.dla import dla_ensemble
from ..theme import (
    TEXT_C,
    WALK_C,
    START_C,
    THEO_C,
    RMS_C,
    MSD_C,
    apply_mpl_style,
    fig_bg,
)
from .plotting import draw_graph_2d


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


@st.cache_data
def _build_perc_cached(size: int, p_open: float, seed: int):
    rng = np.random.default_rng(seed)
    return build_percolation(size, p_open, rng=rng)


def _draw_dla_cluster(G, pos, growth_order: list, title: str):
    """Draw the percolation graph with the DLA cluster coloured by growth order."""
    fig, ax = plt.subplots(figsize=(6, 5.8), facecolor=fig_bg())
    draw_graph_2d(G, pos, ax=ax, node_size=4, edge_alpha=0.15)

    if growth_order:
        m = len(growth_order)
        cmap = plt.cm.plasma
        for i, node in enumerate(growth_order):
            x, y = float(pos[node][0]), float(pos[node][1])
            color = cmap(i / max(m - 1, 1))
            ax.scatter([x], [y], s=22, c=[color], linewidths=0, zorder=6)

        # Colour bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=1, vmax=m))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Growth order M", color=TEXT_C, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=TEXT_C)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_C)

    ax.set_title(title, color=TEXT_C, fontsize=11)
    fig.tight_layout()
    return fig


def _draw_ensemble_fig(result: dict) -> plt.Figure:
    """Log-log plot of ensemble R_g(M) with fit window and honest finite-size labels."""
    masses = result["masses"]
    rg_mean = result["rg_mean"]
    rg_std = result["rg_std"]
    rg_runs = result["rg_runs"]
    fit_lo = result["fit_lo"]
    fit_hi = result["fit_hi"]
    d_f = result["d_f_fit"]
    r2 = result["d_f_r2"]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=fig_bg())
    apply_mpl_style()

    # Individual runs — faint
    for rg_r in rg_runs:
        ax.loglog(masses, rg_r, color=WALK_C, alpha=0.18, lw=0.8)

    # Ensemble mean ± 1σ band
    ax.loglog(masses, rg_mean, color=RMS_C, lw=2.0, label="Ensemble mean $R_g$", zorder=4)
    ax.fill_between(
        masses,
        np.maximum(rg_mean - rg_std, 1e-6),
        rg_mean + rg_std,
        color=RMS_C,
        alpha=0.20,
        label="±1σ band",
    )

    # Fit line
    mask = (masses >= fit_lo) & (masses <= fit_hi) & (rg_mean > 1e-12)
    if mask.sum() >= 5 and not np.isnan(d_f):
        lm = np.log(masses[mask])
        slope = 1.0 / d_f
        intercept = np.log(rg_mean[mask]).mean() - slope * lm.mean()
        m_fit = masses[mask]
        ax.loglog(
            m_fit,
            np.exp(intercept + slope * np.log(m_fit)),
            "--",
            color=THEO_C,
            lw=1.8,
            zorder=5,
            label=f"Fit: $d_f$ = {d_f:.3f}  ($R^2$ = {r2:.3f})",
        )

    # Fit-window boundary lines
    ax.axvline(fit_lo, color="#888888", lw=1.0, ls=":")
    ax.axvline(fit_hi, color="#888888", lw=1.0, ls=":")

    # Annotations for honest finite-size commentary
    ymin, ymax = ax.get_ylim()
    y_ann = np.exp(0.15 * (np.log(ymax) - np.log(max(ymin, 1e-6))) + np.log(max(ymin, 1e-6)))
    ax.text(
        fit_lo * 1.05,
        y_ann,
        "← small-cluster\nnoise",
        color="#aaaaaa",
        fontsize=7,
        va="bottom",
    )
    ax.text(
        fit_hi * 1.05,
        y_ann,
        "finite-size\nsaturation →",
        color="#aaaaaa",
        fontsize=7,
        va="bottom",
    )

    ax.set_xlabel("Cluster mass $M$", color=TEXT_C)
    ax.set_ylabel("Radius of gyration $R_g$", color=TEXT_C)
    ax.set_title(
        f"DLA on percolation — $R_g \\sim M^{{1/d_f}}$   ({len(rg_runs)} runs)",
        color=TEXT_C,
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    return fig


def render_chapter_d():
    apply_mpl_style()
    st.markdown(
        "## Chapter D — Diffusion-Limited Aggregation\n"
        "DLA clusters grow by sending random walkers from the graph interior: "
        "a walker sticks when it lands adjacent to the existing cluster. "
        "Running an **ensemble** of independent realisations on the same percolation "
        "substrate reveals the self-similar scaling $R_g \\sim M^{1/d_f}$, with "
        "$d_f \\approx 1.71$ expected for 2D DLA. "
        "Fit quality degrades near the marked finite-size boundaries — "
        "the window shown is the honest scaling regime."
    )

    with st.sidebar:
        st.markdown("### Substrate")
        perc_size = st.slider(
            "Grid size",
            20,
            80,
            _env_int("FRACTAL_PERC_SIZE", 45),
            help="Larger grids give more room before finite-size saturation.",
        )
        p_open = st.slider(
            "Bond open probability p",
            0.40,
            0.80,
            _env_float("FRACTAL_P_OPEN", 0.55),
            0.01,
        )
        seed = st.number_input(
            "Substrate RNG seed", 0, 999999, _env_int("FRACTAL_SEED", 42)
        )

        st.markdown("### DLA parameters")
        n_runs = st.slider(
            "Ensemble runs",
            5,
            50,
            _env_int("FRACTAL_DLA_RUNS", 20),
            help="More runs → smoother mean R_g and narrower σ band.",
        )
        max_frac = st.slider(
            "Max cluster size (fraction of nodes)",
            0.05,
            0.30,
            _env_float("FRACTAL_DLA_MAX_FRAC", 0.15),
            0.01,
            help="Stop each cluster when it reaches this fraction of the substrate. "
                 "Keep low to stay out of saturation.",
        )
        max_walk = st.slider(
            "Walker timeout (steps)",
            1000,
            20000,
            _env_int("FRACTAL_DLA_MAX_WALK", 8000),
            1000,
            help="Discard walkers that haven't stuck after this many steps.",
        )

    G, pos, _ = _build_perc_cached(perc_size, float(p_open), int(seed))
    n_nodes = G.number_of_nodes()

    col1, col2 = st.columns([1.0, 1.1])
    with col1:
        cluster_ph = st.empty()
    with col2:
        ensemble_ph = st.empty()

    # Show empty substrate
    fig0, ax0 = plt.subplots(figsize=(6, 5.8), facecolor=fig_bg())
    draw_graph_2d(G, pos, ax=ax0, node_size=4, edge_alpha=0.15)
    ax0.set_title(f"Percolation substrate · {n_nodes} nodes", color=TEXT_C, fontsize=11)
    fig0.tight_layout()
    cluster_ph.pyplot(fig0)
    plt.close(fig0)

    st.divider()
    max_cluster = int(n_nodes * max_frac)
    st.caption(
        f"Substrate: **{n_nodes}** nodes  ·  target cluster size: **{max_cluster}** nodes  "
        f"({max_frac:.0%} of substrate)  ·  ensemble: **{n_runs}** runs"
    )

    if st.button("▶ Run DLA ensemble", type="primary", key="run_d"):
        rng = np.random.default_rng(seed + 1)
        with st.spinner(f"Growing {n_runs} DLA clusters…"):
            result = dla_ensemble(
                G,
                pos,
                n_runs=n_runs,
                max_fraction=float(max_frac),
                max_walk_steps=max_walk,
                rng=rng,
            )

        if "error" in result:
            st.error(result["error"])
            return

        # Show last cluster on the substrate
        last_order = result["growth_orders"][-1]
        cluster_ph.pyplot(_draw_dla_cluster(G, pos, last_order, "Last DLA cluster"))
        plt.close("all")

        # Show ensemble R_g plot
        ensemble_ph.pyplot(_draw_ensemble_fig(result))
        plt.close("all")

        d_f = result["d_f_fit"]
        r2 = result["d_f_r2"]
        fit_lo = result["fit_lo"]
        fit_hi = result["fit_hi"]
        completed = len(result["growth_orders"])

        st.success(
            f"Ensemble complete — {completed}/{n_runs} valid runs  ·  "
            f"fit window M ∈ [{fit_lo:.0f}, {fit_hi:.0f}]  ·  "
            f"**d_f = {d_f:.3f}** (R² = {r2:.3f})  ·  "
            f"2D DLA theory: d_f ≈ 1.71"
        )

        if r2 < 0.90:
            st.warning(
                "R² < 0.90 — the fit window is narrow or the cluster is too small "
                "relative to the substrate. Try a larger grid size or lower max-fraction."
            )
