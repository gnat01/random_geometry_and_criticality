"""Chapter VIII: First-Passage Processes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..graphs.carpet import build_carpet
from ..graphs.vicsek import build_vicsek
from ..graphs.percolation import build_percolation
from ..sim.firstpassage import (
    first_passage_times,
    survival_function,
    mean_fpt_vs_size,
)
from ..sim.spectral import fit_spectral_dim_from_return_prob
from ..theme import BG, THEO_C, TEXT_C, PANEL, MSD_C, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited

_THEORY = {
    "Sierpiński carpet":    {"d_s": 2 * np.log(8) / np.log(3) / 2.88,   "d_w": 2.88},
    "Vicsek fractal":       {"d_s": 2 * np.log(5) / np.log(3) / (np.log(15)/np.log(3)), "d_w": np.log(15)/np.log(3)},
    "Regular 2-D grid":     {"d_s": 2.0,  "d_w": 2.0},
    "Critical percolation": {"d_s": 4/3,  "d_w": 2.87},
}


def render() -> None:
    mark_visited("chapter_8")
    chapter_header(
        "VIII",
        "First-Passage Processes",
        "How long does it take a random walker to find its target?",
    )

    st.markdown(r"""
So far we have watched walkers spread — measuring how far they get in time $t$.
Now we ask a different question: given a fixed *target* node $v^*$, how long does
it take a walker released from a random starting point to reach it for the first time?

This is the **first-passage time** $T_{\rm fp}$. It is not just a mathematical curiosity.
In chemistry it is the reaction time: two molecules react when they first meet, and the
reaction rate is $1/\langle T_{\rm fp} \rangle$. In biology it is the search time: a
protein finds its binding site on DNA when it first-passages to it. In finance it is the
hitting time of a barrier — the moment a stock price first crosses a threshold.

The geometry of the substrate enters through the survival function:
$$S(t) = \Pr(T_{\rm fp} > t) \sim t^{-\alpha}$$

where the tail exponent $\alpha$ depends on the spectral dimension $d_s$.
On a Euclidean 2-D lattice, $d_s = 2$ and $\alpha = 1/2$ — the Spitzer-type result.
On a fractal with $d_s < 2$, the exponent is smaller: $\alpha = d_s / 2 < 1$,
meaning the survival function decays more slowly — the walker takes much longer to find
the target. Crucially, when $\alpha \leq 1$ (i.e. $d_s \leq 2$) the mean first-passage
time **diverges** in the thermodynamic limit, even though the walk is recurrent.
""")

    definition_box(
        "First-passage time",
        r"$T_{\rm fp} = \min\{t \geq 1 : X_t = v^*\}$ — the first time the walk "
        r"hits the target node $v^*$. The survival function $S(t) = \Pr(T_{\rm fp} > t)$ "
        r"has a power-law tail $S(t) \sim t^{-\alpha}$ with $\alpha = d_s/2$ on a fractal "
        r"of spectral dimension $d_s$.",
    )

    st.markdown(r"""
---

### The diverging mean: a subtlety worth pausing on

On a finite graph the mean FPT is always finite — the walker must eventually visit every
node. But as the graph grows, $\langle T_{\rm fp} \rangle \sim L^{d_w}$: it grows as the
$d_w$-th power of the linear size $L$. Since $d_w > 2$ on a fractal, this grows *faster*
than the system volume $L^{d_f}$. In the thermodynamic limit the mean diverges.

This has physical consequences. A reaction on a fractal substrate (a porous medium, a gel,
a disordered solid) is slower than on a compact Euclidean medium — not just quantitatively,
but qualitatively: the mean reaction time no longer sets a well-defined timescale.

---

### Experiment 1 — Survival function: fractal vs Euclidean

We release walkers from random starting nodes and plot the empirical survival function
$S(t) = \Pr(T_{\rm fp} > t)$ for the selected fractal **and** for a regular 2-D grid of
the same linear size.  The fractal $S(t)$ lies *above* the grid curve — walkers survive
longer because the fractal topology makes the target harder to reach.

The theory predicts $S(t) \sim t^{-d_s/2}$ for large $t$ on an infinite fractal.
A reference line with the correct theoretical slope is shown as a guide.

> **Finite-size note.** On small finite graphs the mean FPT and the finite-size crossover
> time are both $\sim L^{d_w}$, leaving no clean power-law window.  Extracting $\alpha$
> reliably from a slope fit requires systems orders of magnitude larger.  The quantitative
> exponent is better read from the Laplacian eigenspectrum (Chapter VII) or from the
> mean-FPT scaling in Experiment 2.
""")

    geom1 = st.selectbox(
        "Geometry",
        ["Sierpiński carpet", "Vicsek fractal", "Regular 2-D grid", "Critical percolation"],
        key="ch8_geom1",
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        depth1 = st.slider("Depth / size", 2, 4, 3, key="ch8_depth1")
    with col_b:
        n_walkers1 = st.slider("Walkers", 2000, 20000, 8000, step=1000, key="ch8_w1")
    with col_c:
        t_max1 = st.slider("Max steps $T_{\\rm max}$", 1000, 20000, 5000, step=1000, key="ch8_tmax1")

    if st.button("▶ Run survival experiment", key="ch8_run1"):
        _run_survival(geom1, depth1, n_walkers1, t_max1)
    elif "ch8_surv" in st.session_state:
        _show_survival(st.session_state["ch8_surv"])

    st.markdown(r"""
---

### Experiment 2 — Mean FPT scales as $L^{d_w}$

We grow the same fractal at depths $k = 2, 3, 4$ (and 5 for Vicsek) and measure the
mean FPT from a random start to the central node. The log-log slope against $L = 3^k$
gives the walk dimension $d_w$ — a second independent measurement alongside the MSD.
""")

    geom2 = st.selectbox(
        "Geometry for scaling",
        ["Sierpiński carpet", "Vicsek fractal"],
        key="ch8_geom2",
    )
    col_d, col_e = st.columns(2)
    with col_d:
        nw2 = st.slider("Walkers per graph", 500, 3000, 1000, step=250, key="ch8_w2")
    with col_e:
        tmax2 = st.slider("$T_{\\rm max}$ per walker", 5000, 50000, 20000, step=5000, key="ch8_tmax2")

    if st.button("▶ Run FPT scaling", key="ch8_run2"):
        _run_scaling(geom2, nw2, tmax2)
    elif "ch8_scale" in st.session_state:
        _show_scaling(st.session_state["ch8_scale"])

    st.markdown("---")

    key_result(
        r"On a fractal with spectral dimension $d_s < 2$, the survival function "
        r"$S(t) \sim t^{-d_s/2}$ decays more slowly than on a Euclidean lattice ($d_s=2$, "
        r"$S \sim t^{-1/2}$). The mean FPT diverges in the thermodynamic limit, growing as "
        r"$\langle T_{\rm fp}\rangle \sim L^{d_w}$ with $d_w > 2$. "
        r"This is why diffusion-limited reactions in fractal media are anomalously slow."
    )

    st.markdown(r"""
---

### Relation to the spectral dimension

The connection $\alpha = d_s/2$ can be understood from the spectral representation of the
heat kernel. The probability of being at node $v$ at time $t$ starting from $u$ is

$$p(v,t|u,0) = \frac{1}{n}\sum_k \phi_k(u)\phi_k(v)\, e^{-\lambda_k t}$$

The survival probability is $1 - \int_0^t [\text{probability of first hit at }v^*] dt'$.
For long $t$ the sum is dominated by the $\lambda_k \to 0$ modes; their density
$g(\lambda) \sim \lambda^{d_s/2 - 1}$ (from Chapter VII) controls the long-time tail,
giving $S(t) \sim t^{-d_s/2}$.

This means $\alpha$ is **not** an independent exponent — it is determined by $d_s$, which
in turn is determined by $d_f$ and $d_w$ via the Alexander-Orbach relation. All roads
lead back to the same two numbers.

### Further reading

- Redner, S. (2001). *A Guide to First-Passage Processes.* Cambridge University Press.
- Condamin, S. et al. (2007). First-passage times in complex scale-invariant media. *Nature* 450, 77.
- Bénichou, O. & Voituriez, R. (2014). From first-passage times of random walks in confinement to geometry-controlled kinetics. *Phys. Rep.* 539, 225.
""")

    st.markdown("---")
    prev_next("chapter_8")


# ---------------------------------------------------------------------------
# Experiment 1: survival function
# ---------------------------------------------------------------------------

def _build_graph(geom: str, depth: int, rng):
    import networkx as nx
    if geom == "Sierpiński carpet":
        return build_carpet(depth)
    elif geom == "Vicsek fractal":
        return build_vicsek(depth)
    elif geom == "Regular 2-D grid":
        side = 3 ** depth
        G = nx.grid_2d_graph(side, side)
        pos = {(i, j): np.array([j / (side - 1), i / (side - 1)]) for i, j in G.nodes()}
        return G, pos, side - 1
    else:  # Critical percolation
        G, pos, side = build_percolation(3 ** depth, 0.50, rng)
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()
        return G, pos, side


def _central_node(G, pos, side):
    """Return the node closest to the geometric centre of the graph."""
    centre = np.array([0.5, 0.5])
    return min(G.nodes(), key=lambda v: np.linalg.norm(np.array(pos[v]) - centre))


def _run_survival(geom: str, depth: int, n_walkers: int, t_max: int) -> None:
    rng = np.random.default_rng(42)
    with st.spinner("Building fractal graph…"):
        G, pos, side = _build_graph(geom, depth, rng)
    target = _central_node(G, pos, side)

    # Also build a regular 2-D grid of the same linear size for comparison
    with st.spinner("Building 2-D grid for comparison…"):
        import networkx as nx
        grid_side = side if isinstance(side, int) else int(side)
        Gg = nx.grid_2d_graph(grid_side, grid_side)
        posg = {(i, j): np.array([j / max(grid_side - 1, 1),
                                   i / max(grid_side - 1, 1)]) for i, j in Gg.nodes()}
        target_g = _central_node(Gg, posg, grid_side)

    rng2 = np.random.default_rng(7)
    with st.spinner(f"Running {n_walkers:,} walkers on fractal…"):
        fpts = first_passage_times(G, target, n_walkers=n_walkers, t_max=t_max, rng=rng2)
        t_sv, S = survival_function(fpts, t_max=t_max)

    rng3 = np.random.default_rng(13)
    with st.spinner(f"Running {n_walkers:,} walkers on 2-D grid…"):
        fpts_g = first_passage_times(Gg, target_g, n_walkers=n_walkers, t_max=t_max, rng=rng3)
        t_sv_g, S_g = survival_function(fpts_g, t_max=t_max)

    theory = _THEORY.get(geom, {})
    alpha_theory = theory.get("d_s", float("nan")) / 2.0

    result = {
        "geom": geom, "depth": depth,
        "fpts": fpts, "t_sv": t_sv, "S": S,
        "t_sv_g": t_sv_g, "S_g": S_g, "fpts_g": fpts_g,
        "alpha_theory": alpha_theory,
        "t_max": t_max, "n_nodes": G.number_of_nodes(),
        "n_nodes_g": Gg.number_of_nodes(),
        "n_censored": int(np.sum(fpts == t_max)),
        "n_censored_g": int(np.sum(fpts_g == t_max)),
    }
    st.session_state["ch8_surv"] = result
    _show_survival(result)


def _show_survival(res: dict) -> None:
    t_sv, S = res["t_sv"], res["S"]
    t_sv_g, S_g = res["t_sv_g"], res["S_g"]
    alpha_theory = res["alpha_theory"]
    geom = res["geom"]

    col1, col2 = st.columns([3, 2])

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
        ax.set_facecolor(PANEL)

        ax.plot(t_sv, S, color=THEO_C, lw=1.5,
                label=f"{geom} (fractal)")
        ax.plot(t_sv_g, S_g, color=MSD_C, lw=1.5, alpha=0.8,
                label="Regular 2-D grid")

        # Theory reference slope anchored where fractal S first drops below 0.9
        if not np.isnan(alpha_theory):
            anchor_idx = np.where(S < 0.9)[0]
            if len(anchor_idx):
                t_anchor = float(t_sv[anchor_idx[0]])
                S_anchor = float(S[anchor_idx[0]])
                t_ref = np.array([t_anchor, float(t_sv[-1])], dtype=float)
                S_ref = S_anchor * (t_ref / t_anchor) ** (-alpha_theory)
                ax.plot(t_ref, S_ref, ":", color=RMS_C, lw=1.5,
                        label=f"Theory slope $\\alpha = d_s/2 = {alpha_theory:.3f}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time $t$", color=TEXT_C, fontsize=11)
        ax.set_ylabel(r"$S(t)$", color=TEXT_C, fontsize=11)
        ax.set_title("Survival function comparison", color=TEXT_C, fontsize=11)
        ax.tick_params(colors=TEXT_C)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        # FPT histograms overlay
        fig2, ax2 = plt.subplots(figsize=(4.5, 5), facecolor=BG)
        ax2.set_facecolor(PANEL)
        fpts_hit = res["fpts"][res["fpts"] < res["t_max"]]
        fpts_hit_g = res["fpts_g"][res["fpts_g"] < res["t_max"]]
        for fh, col, label in [
            (fpts_hit,   THEO_C, geom),
            (fpts_hit_g, MSD_C,  "2-D grid"),
        ]:
            if len(fh) > 10:
                bins = np.logspace(np.log10(max(1, fh.min())),
                                   np.log10(fh.max()), 35)
                ax2.hist(fh, bins=bins, color=col, alpha=0.55,
                         edgecolor="none", label=label)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("First-passage time $T_{\\rm fp}$", color=TEXT_C, fontsize=10)
        ax2.set_ylabel("Count", color=TEXT_C, fontsize=10)
        ax2.set_title("FPT distributions", color=TEXT_C, fontsize=10)
        ax2.tick_params(colors=TEXT_C)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # Metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Theory $\\alpha = d_s/2$",
                  f"{alpha_theory:.3f}" if not np.isnan(alpha_theory) else "N/A",
                  help="Asymptotic exponent — only visible on large systems.")
    with mc2:
        fpts_hit = res["fpts"][res["fpts"] < res["t_max"]]
        st.metric("Mean FPT — fractal",
                  f"{np.mean(fpts_hit):.0f}" if len(fpts_hit) else "N/A")
    with mc3:
        fpts_hit_g = res["fpts_g"][res["fpts_g"] < res["t_max"]]
        st.metric("Mean FPT — 2-D grid",
                  f"{np.mean(fpts_hit_g):.0f}" if len(fpts_hit_g) else "N/A")
    with mc4:
        pct = 100 * res["n_censored"] / len(res["fpts"])
        st.metric("Censored (fractal)", f"{pct:.1f}%",
                  help="Walkers that did not hit within T_max — increase T_max to reduce.")


# ---------------------------------------------------------------------------
# Experiment 2: mean FPT scaling with graph size
# ---------------------------------------------------------------------------

def _run_scaling(geom: str, n_walkers: int, t_max: int) -> None:
    depths = [2, 3, 4] if geom == "Sierpiński carpet" else [2, 3, 4, 5]
    builder = build_carpet if geom == "Sierpiński carpet" else build_vicsek
    theory_dw = _THEORY[geom]["d_w"]

    graphs = []
    with st.spinner("Building graphs at multiple depths…"):
        for d in depths:
            G, pos, side = builder(d)
            graphs.append((G, pos, side))

    with st.spinner(f"Running FPT experiments (this may take a minute)…"):
        rng = np.random.default_rng(42)
        sizes, means, stds = mean_fpt_vs_size(
            graphs, _central_node, n_walkers=n_walkers, t_max=t_max, rng=rng
        )

    # Fit d_w from slope
    valid = ~np.isnan(means)
    d_w_fit, r2_fit = float("nan"), float("nan")
    if valid.sum() >= 2:
        lx = np.log(sizes[valid])
        ly = np.log(means[valid])
        c = np.polyfit(lx, ly, 1)
        d_w_fit = float(c[0])
        ly_fit = np.polyval(c, lx)
        ss_res = np.sum((ly - ly_fit) ** 2)
        ss_tot = np.sum((ly - ly.mean()) ** 2)
        r2_fit = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    result = {
        "geom": geom, "depths": depths,
        "sizes": sizes, "means": means, "stds": stds,
        "d_w_fit": d_w_fit, "r2_fit": r2_fit,
        "theory_dw": theory_dw,
    }
    st.session_state["ch8_scale"] = result
    _show_scaling(result)


def _show_scaling(res: dict) -> None:
    sizes, means, stds = res["sizes"], res["means"], res["stds"]
    d_w_fit, r2_fit = res["d_w_fit"], res["r2_fit"]
    theory_dw = res["theory_dw"]

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(PANEL)

    valid = ~np.isnan(means)
    ax.errorbar(sizes[valid], means[valid], yerr=stds[valid],
                fmt="o", color=THEO_C, ms=8, capsize=4, lw=1.5,
                label=r"$\langle T_{\rm fp}\rangle$ (measured)")

    if valid.sum() >= 2 and not np.isnan(d_w_fit):
        lx = np.log(sizes[valid])
        c = np.polyfit(lx, np.log(means[valid]), 1)
        s_fit = np.linspace(sizes[valid].min() * 0.9, sizes[valid].max() * 1.1, 50)
        ax.plot(s_fit, np.exp(np.polyval(c, np.log(s_fit))),
                "--", color=MSD_C, lw=2,
                label=f"Fit: slope = $d_w$ = {d_w_fit:.2f}  ($R^2$ = {r2_fit:.3f})")

    # Theory reference
    if not np.isnan(theory_dw) and valid.sum() >= 1:
        s0, m0 = sizes[valid][0], means[valid][0]
        s_ref = np.array([sizes[valid].min(), sizes[valid].max()], dtype=float)
        ax.plot(s_ref, m0 * (s_ref / s0) ** theory_dw,
                ":", color=RMS_C, lw=1.5,
                label=f"Theory: $d_w$ = {theory_dw:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Linear size $L = 3^k$", color=TEXT_C, fontsize=11)
    ax.set_ylabel(r"$\langle T_{\rm fp}\rangle$", color=TEXT_C, fontsize=11)
    ax.set_title(r"Mean FPT scaling: $\langle T_{\rm fp}\rangle \sim L^{d_w}$",
                 color=TEXT_C, fontsize=11)
    ax.tick_params(colors=TEXT_C)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col2:
        st.metric("Measured $d_w$", f"{d_w_fit:.3f}" if not np.isnan(d_w_fit) else "N/A")
        st.metric("Theory $d_w$", f"{theory_dw:.3f}")
        if not np.isnan(d_w_fit) and not np.isnan(theory_dw):
            err = abs(d_w_fit - theory_dw) / theory_dw * 100
            st.metric("Error", f"{err:.1f}%")
        st.markdown(
            f"**Depths measured:** {res['depths']}\n\n"
            f"**Graph sizes (nodes):** "
            + ", ".join(
                f"{int(s**np.log(8)/np.log(3)):,}" if res['geom'] == "Sierpiński carpet"
                else f"~{int(s**np.log(5)/np.log(3)):,}"
                for s in sizes
            )
        )
