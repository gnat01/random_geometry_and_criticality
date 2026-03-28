"""Chapter V: Growth as Fractal (DLA)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import streamlit as st

from ..graphs.percolation import build_percolation
from ..sim.dla import dla_ensemble
from ..theme import BG, THEO_C, TEXT_C, PANEL, NODE_C, EDGE_C
from ..ui.plotting import draw_graph_2d
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited


def render() -> None:
    mark_visited("chapter_5")
    chapter_header(
        "V",
        "Growth as Fractal",
        "Diffusion-limited aggregation and fractal dimension from growth",
    )

    st.markdown("""
So far you have seen fractals that are built by removing parts of a regular object —
the carpet, the Vicsek fractal, percolation clusters. But fractals can also *grow*.

**Diffusion-limited aggregation** (DLA) was introduced by Witten and Sander in 1981.
The rule is simple:

1. Place a seed particle at the centre of the graph.
2. Release a random walker from a distant point.
3. When the walker lands on a node adjacent to the cluster, it sticks there.
4. Repeat.

Nothing in the rule is explicitly fractal. Yet the clusters that grow have fractal
geometry with $d_f \\approx 1.71$ in two-dimensional Euclidean space. The branching
structure emerges purely from the competition between diffusion and the geometric
screening effect: branches extend into open space, shielding the interior from
incoming walkers.

On a fractal substrate — the largest connected component of a percolation cluster —
the effective dimension of the substrate itself is less than 2. The DLA cluster that
grows on this substrate has a fractal dimension *higher* than 1.71, because the substrate
is already constrained.
""")

    definition_box(
        "Radius of gyration",
        "$R_g(M) = \\sqrt{\\frac{1}{M} \\sum_{i=1}^M |\\mathbf{r}_i - \\mathbf{r}_{cm}|^2}$ "
        "— the RMS distance of cluster particles from the centre of mass. "
        "Scaling $R_g \\sim M^{1/d_f}$ defines the fractal dimension of the cluster."
    )

    st.markdown("""
---

### Experiment: DLA ensemble on a percolation substrate

We grow 8 independent DLA clusters on the same percolation graph ($p = 0.6$, $L = 35$)
and measure the ensemble-averaged $R_g(M)$ scaling. The slope of the log-log plot gives
$1/d_f$.
""")

    n_runs = st.slider("Number of DLA runs", 4, 16, 8, step=2, key="ch5_runs")
    size = st.slider("Substrate size $L$", 25, 50, 35, step=5, key="ch5_size")

    if st.button("▶ Run DLA ensemble", key="ch5_run"):
        _run_dla(n_runs, size)
    elif "ch5_dla" in st.session_state:
        _show_dla(st.session_state["ch5_dla"])

    st.markdown("""
---

### Reflection

The fractal dimension you measure will typically land between 1.5 and 2.0 for these
substrate sizes — the precise value depends on how much of the substrate the cluster
fills before the simulation stops. The key qualitative fact is the branching morphology:
DLA clusters are never compact.

Why does diffusion produce fractals? The deeper reason is that the random walk is itself
a fractal object (with $d_w = 2$ in 2D Euclidean space), and the DLA process selects
growth points proportional to the harmonic measure — the probability that a random walker
first hits the cluster boundary at that point. The harmonic measure is highly concentrated
on tips and re-entrant corners, which is why DLA grows outward rather than inward.

This connection to harmonic analysis makes DLA closely related to the Hele-Shaw problem
in fluid mechanics and to dielectric breakdown. The same branching instability appears
in all three settings.
""")

    key_result(
        "DLA in 2D Euclidean space: $d_f \\approx 1.71$ (numerical). "
        "On a fractal substrate (percolation at $p > p_c$), the effective $d_f$ is "
        "modified by the substrate geometry. The branching structure is universal — "
        "it emerges from the diffusion field, not from any built-in rule."
    )

    st.markdown("---")
    prev_next("chapter_5")


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def _run_dla(n_runs: int, size: int) -> None:
    with st.spinner(f"Growing {n_runs} DLA clusters on a {size}×{size} percolation substrate…"):
        rng = np.random.default_rng(42)
        G, pos, _ = build_percolation(size, 0.60, rng)
        rng2 = np.random.default_rng(7)
        result = dla_ensemble(G, pos, n_runs=n_runs, max_fraction=0.12, rng=rng2)
        result["G"] = G
        result["pos"] = pos
    st.session_state["ch5_dla"] = result
    _show_dla(result)


def _show_dla(result: dict) -> None:
    if "error" in result:
        st.error(result["error"])
        return

    masses = result["masses"]
    rg_mean = result["rg_mean"]
    rg_std = result["rg_std"]
    d_f = result["d_f_fit"]
    r2 = result["d_f_r2"]
    fit_lo = result["fit_lo"]
    fit_hi = result["fit_hi"]

    # --- Panel layout ---
    col1, col2 = st.columns([1, 1])

    # Left: one DLA cluster visualised on the substrate
    with col1:
        G = result.get("G")
        pos = result.get("pos")
        if G is not None and pos is not None and result.get("growth_orders"):
            go = result["growth_orders"][0]
            _draw_dla_cluster(G, pos, go)

    # Right: R_g scaling
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
        ax.set_facecolor(PANEL)

        # Individual runs (faint)
        for rg_run in result.get("rg_runs", []):
            ax.plot(masses[:len(rg_run)], rg_run, color="#aaaaaa", alpha=0.2, lw=0.8)

        # Ensemble mean ± std
        ax.plot(masses, rg_mean, color=THEO_C, lw=2.5, label="Ensemble mean $R_g$", zorder=4)
        ax.fill_between(
            masses,
            rg_mean - rg_std,
            rg_mean + rg_std,
            color=THEO_C, alpha=0.15, zorder=3,
        )

        # Fit line
        if not np.isnan(d_f):
            mask = (masses >= fit_lo) & (masses <= fit_hi) & (rg_mean > 1e-12)
            if mask.sum() >= 5:
                lm = np.log(masses[mask])
                lr = np.log(rg_mean[mask])
                slope_fit, int_fit = np.polyfit(lm, lr, 1)
                A_fit = np.exp(int_fit)
                fit_m = masses[mask]
                ax.plot(fit_m, A_fit * fit_m**slope_fit, "--", color="#4caf50", lw=2,
                        label=f"Fit: $R_g \\sim M^{{1/d_f}}$, $d_f = {d_f:.2f}$  ($R^2 = {r2:.2f}$)",
                        zorder=5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cluster mass $M$", color=TEXT_C, fontsize=12)
        ax.set_ylabel("$R_g(M)$", color=TEXT_C, fontsize=12)
        ax.set_title("$R_g$ scaling: slope = $1/d_f$", color=TEXT_C, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Metrics
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Measured $d_f$", f"{d_f:.3f}" if not np.isnan(d_f) else "N/A")
    with col_b:
        st.metric("Reference $d_f$ (2D Euclidean)", "1.710")
    with col_c:
        st.metric("Fit $R^2$", f"{r2:.3f}" if not np.isnan(r2) else "N/A")


def _draw_dla_cluster(G, pos, growth_order: list) -> None:
    """Draw the substrate graph with the DLA cluster coloured by growth order."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=BG)
    ax.set_facecolor(BG)

    # Draw substrate edges (very faint)
    segs_x, segs_y = [], []
    for u, v in G.edges():
        segs_x += [pos[u][0], pos[v][0], None]
        segs_y += [pos[u][1], pos[v][1], None]
    ax.plot(segs_x, segs_y, color=EDGE_C, alpha=0.12, lw=0.4, zorder=1)

    # Background substrate nodes (very faint)
    non_cluster = set(G.nodes()) - set(growth_order)
    if non_cluster:
        nc_xy = np.array([pos[n] for n in non_cluster])
        ax.scatter(nc_xy[:, 0], nc_xy[:, 1], s=2, c="#334455", alpha=0.3, linewidths=0, zorder=2)

    # DLA cluster: colour by growth order
    if growth_order:
        cluster_xy = np.array([pos[n] for n in growth_order])
        cmap = cm.plasma
        colours = cmap(np.linspace(0, 1, len(growth_order)))
        ax.scatter(
            cluster_xy[:, 0], cluster_xy[:, 1],
            s=max(2, 20 - len(growth_order) // 30),
            c=colours, alpha=0.85, linewidths=0, zorder=3,
        )
        # Mark seed
        seed_xy = pos[growth_order[0]]
        ax.scatter([seed_xy[0]], [seed_xy[1]], s=60, c="#00e676",
                   marker="*", zorder=5, label="Seed")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("DLA cluster (colour = growth order)", color=TEXT_C, fontsize=10, pad=6)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
