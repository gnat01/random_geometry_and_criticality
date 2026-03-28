"""Chapter B — occupation, first passage, traps."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import streamlit as st

from ..graphs.carpet import build_carpet
from ..graphs.percolation import build_percolation
from ..graphs.vicsek import build_vicsek
from ..sim.observables import (
    first_passage_times,
    occupation_counts,
    survival_with_traps,
)
from ..sim.walks import stationary_distribution_degree
from ..theme import START_C, TARGET_C, TEXT_C, TRAP_C, apply_mpl_style, fig_bg
from .plotting import draw_graph_2d


def _env(key: str, fb: str) -> str:
    return os.environ.get(key, fb)


@st.cache_data
def _build_b(
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


def render_chapter_b():
    apply_mpl_style()
    st.markdown(
        "## Chapter B — Observables & first passage\n"
        "**Occupation** — empirical visit frequencies vs. degree stationary measure π. "
        "**First passage** — hitting-time distribution to a target set. "
        "**Traps** — absorbing sites and survival of an ensemble."
    )

    mode_default = _env("FRACTAL_MODE", "occupation").lower()
    if mode_default not in ("occupation", "first_passage", "traps"):
        mode_default = "occupation"
    idx = ("occupation", "first_passage", "traps").index(mode_default)

    with st.sidebar:
        mode = st.radio(
            "Mode",
            ("occupation", "first_passage", "traps"),
            index=idx,
            format_func=lambda x: {
                "occupation": "Occupation vs π",
                "first_passage": "First passage times",
                "traps": "Traps & survival",
            }[x],
        )
        graph = st.selectbox("Graph", ("carpet", "vicsek", "percolation"), index=0)
        depth = st.slider("Depth (carpet / Vicsek)", 2, 5, 4)
        perc_size = st.slider("Grid size (percolation)", 10, 40, 22)
        p_open = st.slider("p (percolation)", 0.4, 0.8, 0.55, 0.01)
        seed = st.number_input("Seed", 0, 999999, 42)

        rng_seed = st.number_input("Simulation RNG seed", 0, 999999, 7)

        occ_steps = st.slider("Occupation steps", 500, 50000, 8000)
        fp_trials = st.slider("First-passage trials", 50, 2000, 400)
        fp_max = st.slider("Max steps per trial", 100, 50000, 8000)
        n_traps = st.slider("Number of traps", 1, 80, 12)
        n_surv_walkers = st.slider("Walkers (traps)", 200, 5000, 1200)
        surv_steps = st.slider("Steps (traps)", 100, 8000, 2000)

    G, pos, _ = _build_b(graph, depth, perc_size, float(p_open), int(seed))
    nodes = list(G.nodes())
    rng = np.random.default_rng(int(rng_seed))

    if mode == "occupation":
        start = (0, 0) if (0, 0) in G else nodes[rng.integers(len(nodes))]
        counts = occupation_counts(G, start, occ_steps, rng)
        total = sum(counts.values())
        pi = stationary_distribution_degree(G)
        emp = []
        theo = []
        for n in nodes:
            emp.append(counts.get(n, 0) / total if total else 0.0)
            theo.append(pi.get(n, 0.0))
        err = np.array(emp) - np.array(theo)
        emax = float(np.max(np.abs(err))) or 1e-12

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=fig_bg())
        ax0, ax1 = axes
        xy = np.array([pos[n] for n in nodes])
        vmax = max(np.max(np.abs(emp)), 1e-9)
        sc = ax0.scatter(
            xy[:, 0],
            xy[:, 1],
            c=emp,
            cmap="viridis",
            s=12,
            vmin=0,
            vmax=vmax,
        )
        plt.colorbar(sc, ax=ax0, fraction=0.046, label="π_emp")
        ax0.set_aspect("equal")
        ax0.axis("off")
        ax0.set_title("Empirical occupation", color=TEXT_C)
        sc2 = ax1.scatter(
            xy[:, 0],
            xy[:, 1],
            c=err,
            cmap="coolwarm",
            norm=mcolors.TwoSlopeNorm(vmin=-emax, vcenter=0.0, vmax=emax),
            s=12,
        )
        plt.colorbar(sc2, ax=ax1, fraction=0.046, label="π_emp − π")
        ax1.set_aspect("equal")
        ax1.axis("off")
        ax1.set_title("Deviation from degree measure", color=TEXT_C)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        tv = np.array(theo)
        ev = np.array(emp)
        mask = tv > 0
        st.metric("χ²-style discrepancy (informal)", f"{np.mean((ev[mask]-tv[mask])**2/tv[mask]):.6f}")

    elif mode == "first_passage":
        # Target: nodes in top-right quadrant of bbox
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        mx, my = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
        target = {n for n in nodes if pos[n][0] >= mx and pos[n][1] >= my}
        if len(target) < 2:
            target = {n for n in nodes if pos[n][0] >= mx}
        if len(target) < 1:
            target = {nodes[-1]}

        def start_fn(rng):
            s = nodes[rng.integers(len(nodes))]
            tries = 0
            while s in target and tries < 50:
                s = nodes[rng.integers(len(nodes))]
                tries += 1
            return s

        times = first_passage_times(
            G, target, fp_trials, fp_max, rng, start_fn
        )

        fig, ax = plt.subplots(figsize=(7, 4), facecolor=fig_bg())
        ax.set_facecolor(fig_bg())
        if not times:
            st.warning(
                "No hits before max steps — raise **Max steps per trial** or enlarge the target set."
            )
        if times:
            ax.hist(
                times,
                bins=min(40, max(8, len(times) // 10)),
                color=TARGET_C,
                alpha=0.85,
                edgecolor="#333",
            )
            ax.axvline(
                np.mean(times),
                color="white",
                ls="--",
                label=f"mean {np.mean(times):.1f}",
            )
        ax.set_xlabel("Steps to hit target")
        ax.set_ylabel("Count")
        ax.set_title("First-passage times (SRW)", color=TEXT_C)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        ex0 = start_fn(rng)
        fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor=fig_bg())
        draw_graph_2d(G, pos, ax=ax2)
        tx = [pos[n][0] for n in target]
        ty = [pos[n][1] for n in target]
        ax2.scatter(tx, ty, s=40, c=TARGET_C, alpha=0.85, edgecolors="none", label="target")
        sx, sy = pos[ex0][0], pos[ex0][1]
        ax2.scatter([sx], [sy], s=120, c=START_C, marker="*", zorder=9, label="example start")
        ax2.legend(loc="upper right")
        fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

        st.caption(f"Target set size: **{len(target)}** · trials recorded: **{len(times)}** / {fp_trials}")

    else:
        traps = set()
        for _ in range(n_traps * 3):
            if len(traps) >= n_traps:
                break
            traps.add(nodes[rng.integers(len(nodes))])
        if not traps:
            traps.add(nodes[0])

        walkers = [nodes[rng.integers(len(nodes))] for _ in range(n_surv_walkers)]
        survivors = [len(walkers)]
        for _ in range(surv_steps):
            walkers, k = survival_with_traps(G, traps, walkers, rng)
            survivors.append(k)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=fig_bg())
        axl, axr = axes
        axl.plot(np.arange(len(survivors)), np.array(survivors) / survivors[0], color=TRAP_C, lw=2)
        axl.set_xlabel("Step")
        axl.set_ylabel("Survival fraction")
        axl.set_title("Survival with traps", color=TEXT_C)
        axl.grid(True, alpha=0.3)
        draw_graph_2d(G, pos, ax=axr)
        tx = [pos[n][0] for n in traps]
        ty = [pos[n][1] for n in traps]
        axr.scatter(tx, ty, s=55, c=TRAP_C, alpha=0.9, linewidths=0, zorder=6)
        axr.set_title("Trap sites", color=TEXT_C)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
