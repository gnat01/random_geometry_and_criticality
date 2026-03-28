"""Chapter III: Disorder and the Random Substrate."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..graphs.percolation import build_percolation
from ..sim.observables import estimate_msd_rms_series
from ..theme import BG, MSD_C, THEO_C, TEXT_C, PANEL
from ..ui.plotting import draw_graph_2d, pos_walk_square
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited


def render() -> None:
    mark_visited("chapter_3")
    chapter_header(
        "III",
        "Disorder and the Random Substrate",
        "Percolation as a random fractal",
    )

    st.markdown("""
The carpet and Vicsek fractal are perfectly regular: every hole is in exactly the right
place. Real disordered materials are nothing like that. Porous rock, diluted magnets,
polymer gels — they are random. Yet they too can be fractal, and the transition to
fractality is sharp.

**Bond percolation** is the simplest model of a random medium. Take a square lattice.
Keep each bond independently with probability $p$, remove it with probability $1 - p$.
For small $p$, the network falls apart into tiny isolated clusters. For large $p$, most
nodes are connected in one giant component. In between, at a critical value $p_c$, the
system sits exactly at the boundary.

The remarkable fact is that $p_c = \\frac{1}{2}$ *exactly* for bond percolation on the
square lattice. This was proved by Hammersley in 1959 using a self-duality argument: the
dual lattice of a square lattice is again a square lattice, and the dual of a percolation
configuration at bond probability $p$ is a configuration at bond probability $1 - p$.
The critical point must therefore be the self-dual point $p = 1/2$.
""")

    definition_box(
        "Percolation critical point",
        "$p_c = 1/2$ (exact) for bond percolation on the 2D square lattice. "
        "At $p_c$, the largest connected component has fractal geometry with "
        "$d_f = 91/48 \\approx 1.896$ and the same anomalous walk exponent "
        "$d_w \\approx 2.87$ as the Sierpiński carpet — not a coincidence."
    )

    st.markdown("""
---

### Experiment A: three realisations at $p = 0.4$, $0.5$, $0.6$

Below you can see the largest connected component of three independent percolation
instances at bond probabilities below, at, and above $p_c$. The visual change is striking:
the $p = 0.5$ cluster looks like a fractal. The $p = 0.4$ cluster is fragmented and tiny.
The $p = 0.6$ cluster is compact and dense.
""")

    _draw_three_instances()

    st.markdown("""
---

### Experiment B: compare MSD slopes at $p \\approx p_c$ vs $p = 0.70$

At $p_c$, the walk is anomalous. Well above $p_c$, the giant component is nearly a full
lattice, and the walk approaches normal diffusion. Press ▶ Run to see the difference in
MSD slopes directly.
""")

    size = st.slider("Grid size $L$", 12, 35, 22, step=2, key="ch3_size")

    if st.button("▶ Run MSD comparison", key="ch3_run"):
        _run_msd_comparison(size)
    elif "ch3_msd" in st.session_state:
        _show_msd(st.session_state["ch3_msd"])

    st.markdown("""
---

### Reflection

The MSD slope at $p_c$ should come out around $2/d_w \\approx 0.70$, and well above $p_c$
it climbs toward 1. This is the dynamical signature of the geometric transition: the walker
doesn't "know" it's on a fractal by looking at the graph — it discovers it through the
anomaly in its own diffusion.

The key insight here is that the percolation transition is *universal*. The exact exponents
don't depend on the details of the lattice type or the boundary conditions — only on the
space dimension. This is the hallmark of a second-order phase transition, and it's what
makes percolation a prototype for all critical phenomena.
""")

    key_result(
        "At $p_c = 1/2$: the incipient infinite cluster is a random fractal with "
        "$d_f \\approx 1.896$ and anomalous walk dimension $d_w \\approx 2.87$. "
        "The MSD exponent $2/d_w \\approx 0.70$ versus 1.0 for normal diffusion."
    )

    st.markdown("---")
    prev_next("chapter_3")


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _build_three() -> list:
    """Build three percolation instances; cached so the page doesn't redraw on every interaction."""
    results = []
    for p, seed in [(0.4, 1), (0.5, 2), (0.6, 3)]:
        rng = np.random.default_rng(seed)
        G, pos, _ = build_percolation(30, p, rng)
        results.append((p, G, pos))
    return results


def _draw_three_instances() -> None:
    instances = _build_three()
    cols = st.columns(3)
    for col, (p, G, pos) in zip(cols, instances):
        with col:
            label_color = THEO_C if abs(p - 0.5) < 0.01 else TEXT_C
            st.markdown(
                f"<div style='text-align:center; color:{label_color}; font-weight:600;"
                f" font-size:0.95rem;'>$p = {p}$ &nbsp; ({G.number_of_nodes()} nodes)</div>",
                unsafe_allow_html=True,
            )
            fig, ax = plt.subplots(figsize=(4, 4), facecolor=BG)
            draw_graph_2d(G, pos, ax=ax, node_size=4, edge_alpha=0.4)
            if abs(p - 0.5) < 0.01:
                ax.set_title("← critical", color=THEO_C, fontsize=9, pad=4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def _run_msd_comparison(size: int) -> None:
    p_crit = 0.50
    p_high = 0.70
    n_walkers = 50
    n_steps = 400

    results = {}
    with st.spinner(f"Running MSD at p={p_crit} and p={p_high}…"):
        for p in [p_crit, p_high]:
            rng = np.random.default_rng(7)
            G, _, _ = build_percolation(size, p, rng)
            nodes = list(G.nodes())
            if len(nodes) < 10:
                continue
            pos_walk = pos_walk_square(nodes)
            rng2 = np.random.default_rng(13)
            start = nodes[rng2.integers(len(nodes))]
            steps, _, msd = estimate_msd_rms_series(
                G, pos_walk, start, n_walkers, n_steps, rng2
            )
            mask = (steps > 15) & (steps < steps[-1] * 0.75) & (msd > 1e-12)
            slope = float(np.polyfit(np.log(steps[mask]), np.log(msd[mask]), 1)[0]) \
                if mask.sum() >= 6 else float("nan")
            results[p] = {"steps": steps, "msd": msd, "slope": slope}

    st.session_state["ch3_msd"] = results
    _show_msd(results)


def _show_msd(results: dict) -> None:
    if not results:
        st.warning("No results to display.")
        return

    colors = {0.50: THEO_C, 0.70: MSD_C}
    labels = {0.50: "$p = p_c = 0.50$", 0.70: "$p = 0.70$"}

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(PANEL)

    for p, data in results.items():
        steps = data["steps"]
        msd = data["msd"]
        slope = data["slope"]
        c = colors.get(p, TEXT_C)
        lbl = labels.get(p, f"$p = {p}$")
        slope_str = f" (slope {slope:.2f})" if not np.isnan(slope) else ""
        ax.plot(steps, msd, color=c, lw=2, label=lbl + slope_str, alpha=0.9)

        if not np.isnan(slope):
            # Draw the fitted line
            logsteps = np.log(steps[steps > 0])
            logmsd = np.log(msd[steps > 0] + 1e-30)
            intercept = np.mean(logmsd) - slope * np.mean(logsteps)
            fit = np.exp(intercept) * steps**slope
            ax.plot(steps, fit, "--", color=c, alpha=0.4, lw=1.5)

    # Theory reference
    d_w_crit = 2.87
    theory_exp = 2 / d_w_crit
    ref_steps = np.array(list(results.values())[0]["steps"])
    ref_msd = np.array(list(results.values())[0]["msd"])
    A_theory = float(ref_msd[5]) / max(ref_steps[5]**theory_exp, 1e-30)
    ax.plot(ref_steps, A_theory * ref_steps**theory_exp, ":",
            color="#aaaaaa", lw=1.5,
            label=f"Theory $2/d_w={theory_exp:.3f}$ at $p_c$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Steps $t$", color=TEXT_C, fontsize=12)
    ax.set_ylabel(r"$\langle r^2 \rangle$", color=TEXT_C, fontsize=12)
    ax.set_title("MSD at $p_c$ vs above $p_c$: anomalous vs normal diffusion", color=TEXT_C, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    cols = st.columns(len(results))
    for col, (p, data) in zip(cols, results.items()):
        with col:
            slope = data["slope"]
            st.metric(f"Slope at $p = {p}$", f"{slope:.3f}" if not np.isnan(slope) else "N/A")
