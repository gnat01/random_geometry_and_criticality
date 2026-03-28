"""Chapter A: geometry beyond the gasket — carpet, Vicsek, percolation."""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..graphs.carpet import build_carpet
from ..graphs.modifiers import add_levy_edges
from ..graphs.percolation import build_percolation
from ..graphs.vicsek import build_vicsek
from ..sim.walks import (
    batch_random_steps,
    biased_batch_random_steps,
    make_degree_bias,
    make_gradient_bias,
)
from ..theme import (
    MSD_C,
    RMS_C,
    START_C,
    TEXT_C,
    THEO_C,
    WALK_C,
    apply_mpl_style,
    fig_bg,
)
from .plotting import draw_graph_2d, pos_walk_square


def _env_default(key: str, fallback: str) -> str:
    return os.environ.get(key, fallback)


@st.cache_data
def _build_graph_cached(
    graph: str,
    depth: int,
    perc_size: int,
    p_open: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    if graph == "carpet":
        return build_carpet(depth)
    if graph == "vicsek":
        return build_vicsek(depth)
    return build_percolation(perc_size, p_open, rng=rng)


def _make_stats_fig(
    step_hist,
    rms_hist,
    msd_hist,
    loglog: bool,
    d_w: float | None,
    sat_step: float | None,
):
    steps = np.asarray(step_hist, dtype=float)
    rms = np.asarray(rms_hist, dtype=float)
    msd = np.asarray(msd_hist, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), facecolor=fig_bg())
    plot_fn = "loglog" if loglog and len(steps) > 5 else "plot"
    getattr(ax1, plot_fn)(steps, rms, color=RMS_C, lw=1.8, label="RMS", zorder=3)
    ax1.set_xlabel("Steps N")
    ax1.set_ylabel("RMS ⟨r²⟩½")
    ax1.set_title("RMS distance", color=TEXT_C)
    ax1.grid(True)
    getattr(ax2, plot_fn)(steps, msd, color=MSD_C, lw=1.8, label="MSD", zorder=3)
    ax2.set_xlabel("Steps N")
    ax2.set_ylabel("MSD ⟨r²⟩")
    ax2.set_title("Mean squared displacement", color=TEXT_C)
    ax2.grid(True)

    if len(steps) > 20 and sat_step:
        fit_start = 50
        fit_end = max(int(sat_step * 0.25), fit_start + 20)
        mask = (steps >= fit_start) & (steps <= fit_end) & (rms > 0) & (msd > 0)
        if mask.sum() > 5:
            log_s = np.log(steps[mask])
            sr, ir = np.polyfit(log_s, np.log(rms[mask]), 1)
            sv, iv = np.polyfit(log_s, np.log(msd[mask]), 1)
            ds = steps[: np.searchsorted(steps, fit_end * 2.0)]
            getattr(ax1, plot_fn)(
                ds,
                np.exp(ir) * ds**sr,
                "-.",
                color="#00cfff",
                lw=1.2,
                label=f"RMS fit slope {sr:.3f}",
                zorder=4,
            )
            getattr(ax2, plot_fn)(
                ds,
                np.exp(iv) * ds**sv,
                "-.",
                color="#00cfff",
                lw=1.2,
                label=f"MSD fit slope {sv:.3f}",
                zorder=4,
            )
    if d_w:
        end_i = min(len(steps), np.searchsorted(steps, (sat_step or steps[-1]) * 2.0))
        ts = steps[:end_i]
        ai = min(len(steps) // 4, len(steps) - 1)
        cr = rms[ai] / (steps[ai] ** (1.0 / d_w) + 1e-12)
        cv = msd[ai] / (steps[ai] ** (2.0 / d_w) + 1e-12)
        getattr(ax1, plot_fn)(
            ts,
            cr * ts ** (1.0 / d_w),
            "--",
            color=THEO_C,
            lw=1.3,
            label=f"Theory N^(1/d_w), d_w={d_w:.2f}",
        )
        getattr(ax2, plot_fn)(
            ts,
            cv * ts ** (2.0 / d_w),
            "--",
            color=THEO_C,
            lw=1.3,
            label="Theory N^(2/d_w)",
        )
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _draw_walk_overlay(G, pos, walkers, start, step, title: str):
    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=fig_bg())
    draw_graph_2d(G, pos, ax=ax)
    sx, sy = float(pos[start][0]), float(pos[start][1])
    ax.scatter([sx], [sy], s=100, c=START_C, marker="*", zorder=8, linewidths=0)
    if walkers:
        wx = [float(pos[w][0]) for w in walkers]
        wy = [float(pos[w][1]) for w in walkers]
        ax.scatter(wx, wy, s=18, c=WALK_C, alpha=0.75, linewidths=0, zorder=5)
    ax.set_title(f"{title} · step {step}", color=TEXT_C, fontsize=11)
    fig.tight_layout()
    return fig


def render_chapter_a():
    apply_mpl_style()
    st.markdown(
        "## Chapter A — Geometry beyond the gasket\n"
        "Simple random walks on **Sierpiński carpet**, **Vicsek**, or **bond-percolation** grid graphs. "
        "Pick a family, tune depth or percolation parameters, then **Run walk**."
    )

    default_g = _env_default("FRACTAL_GRAPH", "carpet").lower()
    if default_g not in ("carpet", "vicsek", "percolation"):
        default_g = "carpet"

    with st.sidebar:
        st.markdown("### Graph family")
        graph = st.selectbox(
            "Geometry",
            ("carpet", "vicsek", "percolation"),
            index=("carpet", "vicsek", "percolation").index(default_g),
            help="Carpet & Vicsek: deterministic ternary recursion. Percolation: random subgraph of a grid.",
        )
        depth = st.slider(
            "Recursion depth (carpet / Vicsek)",
            2,
            5,
            int(_env_default("FRACTAL_DEPTH", "4")),
            help="Ternary depth; larger ⇒ more nodes (slower).",
        )
        def _ienv(k: str, default: int) -> int:
            try:
                return int(os.environ.get(k, str(default)))
            except ValueError:
                return default

        def _fenv(k: str, default: float) -> float:
            try:
                return float(os.environ.get(k, str(default)))
            except ValueError:
                return default

        perc_size = st.slider(
            "Grid size (percolation)",
            10,
            45,
            _ienv("FRACTAL_PERC_SIZE", 24),
        )
        p_open = st.slider(
            "Bond open probability p",
            0.35,
            0.85,
            _fenv("FRACTAL_P_OPEN", 0.55),
            0.01,
        )
        seed = st.number_input(
            "Percolation RNG seed", 0, 999999, int(_env_default("FRACTAL_SEED", "42"))
        )
        st.markdown("### Walk")
        n_walkers = st.slider("Walkers", 20, 400, 150)
        n_steps = st.slider("Total steps", 200, 12000, 4000)
        update_every = st.slider("Plot refresh every N steps", 5, 200, 50)
        loglog = st.checkbox("Log-log stat plots", value=False)
        show_theory = st.checkbox(
            "Show heuristic anomalous scaling (literature d_w)", value=False
        )

        st.markdown("### Walk modifiers")
        _levy_alpha_default = float(_env_default("FRACTAL_LEVY_ALPHA", "0.0"))
        levy_enabled = st.checkbox(
            "Lévy long-range edges",
            value=(_levy_alpha_default > 0),
        )
        levy_alpha = st.slider(
            "Lévy α (tail exponent)",
            0.3,
            2.0,
            float(_env_default("FRACTAL_LEVY_ALPHA", "1.5")),
            0.05,
            help="Smaller α → heavier tails / more long-range jumps. α ∈ (0,2) is the Lévy regime.",
            disabled=not levy_enabled,
        )
        levy_p = st.slider(
            "Lévy p_long (edge prob at d_nn)",
            0.01,
            0.30,
            float(_env_default("FRACTAL_LEVY_P", "0.05")),
            0.01,
            help="Probability of adding a long-range edge at the nearest-neighbour length scale.",
            disabled=not levy_enabled,
        )

        _bias_options = ("none", "toward-center", "away-center", "hub-seeking", "hub-avoiding")
        _bias_default = _env_default("FRACTAL_BIAS", "none").lower()
        if _bias_default not in _bias_options:
            _bias_default = "none"
        bias_preset = st.selectbox(
            "Walk bias",
            _bias_options,
            index=_bias_options.index(_bias_default),
            help=(
                "none: uniform SRW. "
                "toward/away-center: drift toward or away from graph centroid. "
                "hub-seeking: prefer high-degree neighbours. "
                "hub-avoiding: prefer low-degree neighbours."
            ),
        )
        bias_strength = st.slider(
            "Bias strength",
            0.5,
            6.0,
            float(_env_default("FRACTAL_BIAS_STRENGTH", "3.0")),
            0.5,
            disabled=(bias_preset == "none"),
        )

    G_base, pos, _side = _build_graph_cached(graph, depth, perc_size, float(p_open), int(seed))

    # Apply Lévy modifier if requested (not cached — depends on walk params)
    if levy_enabled:
        levy_rng = np.random.default_rng(int(seed) + 7919)
        G, n_levy = add_levy_edges(G_base, pos, alpha=levy_alpha, p_long=levy_p, rng=levy_rng)
    else:
        G = G_base
        n_levy = 0

    pos_w = pos_walk_square(list(G.nodes()))
    nodes = list(G.nodes())
    adj = {n: list(G.neighbors(n)) for n in nodes}
    n_nodes = G.number_of_nodes()

    d_w = None
    if show_theory:
        if graph == "carpet":
            d_w = 2.88
        elif graph == "vicsek":
            d_w = float(np.log(15) / np.log(3))
        elif graph == "percolation":
            d_w = 2.0

    sat_est = float(n_nodes ** (d_w / 2)) if d_w else float(n_nodes**1.2)

    col1, col2 = st.columns([1.05, 1.0])
    with col1:
        gasket_ph = st.empty()
    with col2:
        stats_ph = st.empty()

    levy_info = f" + {n_levy} Lévy edges (α={levy_alpha})" if levy_enabled and n_levy else ""
    fig0, ax0 = plt.subplots(figsize=(6, 5.5), facecolor=fig_bg())
    draw_graph_2d(G, pos, ax=ax0)
    ax0.set_title(f"{graph} · {n_nodes} nodes{levy_info}", color=TEXT_C, fontsize=11)
    fig0.tight_layout()
    gasket_ph.pyplot(fig0)
    plt.close(fig0)

    # Build bias weight function
    _weight_fn = None
    if bias_preset != "none":
        all_pos_vals = np.array(list(pos.values()), dtype=float)
        centroid = all_pos_vals.mean(axis=0)
        if bias_preset == "toward-center":
            _weight_fn = make_gradient_bias(pos, centroid, strength=bias_strength, toward=True)
        elif bias_preset == "away-center":
            _weight_fn = make_gradient_bias(pos, centroid, strength=bias_strength, toward=False)
        elif bias_preset == "hub-seeking":
            _weight_fn = make_degree_bias(G, strength=bias_strength, hub_seeking=True)
        elif bias_preset == "hub-avoiding":
            _weight_fn = make_degree_bias(G, strength=bias_strength, hub_seeking=False)

    st.divider()
    bias_label = "" if bias_preset == "none" else f"  ·  bias: {bias_preset} (γ={bias_strength})"
    if st.button("▶ Run walk", type="primary", key="run_a"):
        start = nodes[0]
        if (0, 0) in G:
            start = (0, 0)
        p0 = np.asarray(pos_w[start], dtype=float)
        walkers = [start] * n_walkers
        rms_hist: list[float] = []
        msd_hist: list[float] = []
        step_hist: list[int] = []
        rng = np.random.default_rng()
        t0 = time.time()
        progress = st.progress(0.0, text="Walking…")
        for step in range(1, n_steps + 1):
            if _weight_fn is not None:
                walkers = biased_batch_random_steps(G, walkers, _weight_fn, rng)
            else:
                walkers = batch_random_steps(G, walkers, rng)
            vecs = np.array([np.asarray(pos_w[w], dtype=float) - p0 for w in walkers])
            sq = np.sum(vecs**2, axis=1)
            rms_hist.append(float(np.sqrt(np.mean(sq))))
            msd_hist.append(float(np.mean(sq)))
            step_hist.append(step)
            if step % update_every == 0 or step == n_steps:
                gasket_ph.pyplot(
                    _draw_walk_overlay(G, pos, walkers, start, step, graph)
                )
                stats_ph.pyplot(
                    _make_stats_fig(
                        step_hist, rms_hist, msd_hist, loglog, d_w, sat_est
                    )
                )
                plt.close("all")
            progress.progress(step / n_steps, text=f"Step {step} / {n_steps}")
        progress.empty()
        st.success(
            f"Done — {n_steps} steps, {n_walkers} walkers, {time.time() - t0:.1f}s"
            f"{bias_label}"
        )
