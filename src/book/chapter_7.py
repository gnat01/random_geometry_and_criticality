"""Chapter VII: The Sound of a Fractal (spectral dimension)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..graphs.carpet import build_carpet
from ..graphs.vicsek import build_vicsek
from ..graphs.percolation import build_percolation
from ..sim.spectral import (
    laplacian_eigenvalues,
    fit_spectral_dim_from_eigenvalues,
    return_probability,
    fit_spectral_dim_from_return_prob,
)
from ..theme import BG, THEO_C, TEXT_C, PANEL, MSD_C, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited

# Exact/theoretical reference values
_THEORY = {
    "Sierpiński carpet": {"d_f": np.log(8) / np.log(3), "d_w": 2.88},
    "Vicsek fractal":    {"d_f": np.log(5) / np.log(3), "d_w": np.log(15) / np.log(3)},
    "Critical percolation": {"d_f": 91 / 48, "d_w": 2.87},
}


def render() -> None:
    mark_visited("chapter_7")
    chapter_header(
        "VII",
        "The Sound of a Fractal",
        "Laplacian spectrum, density of states, and the spectral dimension",
    )

    st.markdown(r"""
Every graph has a Laplacian matrix $L = D - A$, where $D$ is the diagonal degree matrix
and $A$ is the adjacency matrix. Its eigenvalues $0 = \lambda_0 \leq \lambda_1 \leq \cdots$
are not just a mathematical curiosity — they are the *vibrational frequencies* of the network.

Imagine every node as a mass and every edge as a spring. The normal modes of vibration
are exactly the eigenvectors of $L$, and their frequencies are $\sqrt{\lambda_k}$. For a
regular solid, the density of low-frequency modes follows Debye's law: $g(\omega) \sim \omega^{d-1}$
in $d$ dimensions. For a fractal, this generalises to

$$g(\omega) \sim \omega^{d_s - 1}$$

where $d_s$ is the **spectral dimension** — a new exponent, distinct from both $d_f$ and $d_w$.
Alexander and Orbach (1982) showed that $d_s = 2 d_f / d_w$, connecting the geometry ($d_f$)
to the dynamics ($d_w$) via the spectrum. For percolation clusters they conjectured
$d_s = 4/3$ universally — a conjecture that is approximately but not exactly correct.

The integrated density of states $N(\omega) = \int_0^\omega g(\omega') d\omega' \sim \omega^{d_s}$
has a direct log-log slope of $d_s$, making it the cleanest way to measure the spectral dimension
from the eigenvalues alone — no walkers required.
""")

    definition_box(
        "Spectral dimension",
        r"$d_s$ is defined by the low-frequency scaling of the vibrational density of states: "
        r"$g(\omega) \sim \omega^{d_s - 1}$, or equivalently $N(\omega) \sim \omega^{d_s}$. "
        r"It is related to the Hausdorff dimension $d_f$ and walk dimension $d_w$ by the "
        r"Alexander-Orbach relation $d_s = 2 d_f / d_w$.",
    )

    st.markdown(r"""
---

### Why $d_s$ appears in the return probability

The return probability — the chance that a walker is back at its starting node at time $t$ —
decays as

$$P(t) \sim t^{-d_s/2}$$

This is a purely dynamical measurement of $d_s$. It follows from the spectral representation
of the heat kernel: $P(t) = \sum_k e^{-\lambda_k t} / n$, which at long times is dominated
by the small-$\lambda$ modes and inherits their $\omega^{d_s}$ density.

For $d_s < 2$ the walk is *recurrent*: it returns to its starting node with probability 1
(just slowly). The carpet has $d_s \approx 1.31 < 2$, so despite the anomalous slow diffusion
it is ergodic — the walker eventually reaches every node.

---

### Experiment: three ways to measure $d_s$

We compute $d_s$ by three independent routes and check they agree:

1. **Eigenspectrum** — slope of $\log N(\omega)$ vs $\log \omega$ (exact, no noise)
2. **Return probability** — slope of $\log P(t)$ vs $\log t$ (dynamical, statistical noise)
3. **Alexander-Orbach** — compute $d_f$ and $d_w$ separately, then $d_s = 2d_f / d_w$
""")

    # --- Controls ---
    col_g, col_d = st.columns([2, 1])
    with col_g:
        geometry = st.selectbox(
            "Geometry",
            ["Sierpiński carpet", "Vicsek fractal", "Critical percolation"],
            key="ch7_geom",
        )
    with col_d:
        if geometry == "Sierpiński carpet":
            depth = st.slider("Depth $k$", 2, 4, 3, key="ch7_depth")
        elif geometry == "Vicsek fractal":
            depth = st.slider("Depth $k$", 2, 5, 4, key="ch7_depth_v")
        else:
            depth = st.slider("Grid size $L$", 20, 50, 35, step=5, key="ch7_perc_L")

    col_w, col_s = st.columns(2)
    with col_w:
        n_walkers = st.slider("Walkers for $P(t)$", 1000, 8000, 3000, step=500, key="ch7_walkers")
    with col_s:
        n_steps = st.slider("Steps per walker", 500, 4000, 2000, step=500, key="ch7_steps")

    if st.button("▶ Compute spectrum + return probability", key="ch7_run"):
        _run_spectral(geometry, depth, n_walkers, n_steps)
    elif "ch7_result" in st.session_state:
        _show_spectral(st.session_state["ch7_result"])

    st.markdown("---")

    key_result(
        r"The Alexander-Orbach relation $d_s = 2d_f / d_w$ connects three independently "
        r"measurable quantities. For the Sierpiński carpet: "
        r"$d_s = 2 \times \log 8/\log 3 \div 2.88 \approx 1.31$. "
        r"For percolation at $p_c$: $d_s \approx 4/3 \approx 1.33$ (Alexander-Orbach conjecture). "
        r"Both values satisfy $d_s < 2$, confirming that walks on these fractals are recurrent."
    )

    st.markdown(r"""
---

### Fractons: the phonons of a fractal

In an amorphous solid or a gel network, the elastic medium is not a smooth continuum — it is
a fractal at the nano-scale. The vibrational modes at frequencies above a crossover $\omega^*$
are not ordinary phonons (which propagate) but **fractons**: localised, non-propagating modes
whose density of states follows $g(\omega) \sim \omega^{d_s - 1}$ with $d_s < d$ (the Euclidean
embedding dimension).

This crossover has been measured in silica aerogels, polymer networks, and proteins. The spectral
dimension $d_s$ of the fractal backbone directly controls the low-temperature specific heat
$C_v \sim T^{d_s}$ of these materials — a measurable thermodynamic signature of fractal geometry.

### Further reading

- Alexander, S. & Orbach, R. (1982). Density of states on fractals: fractons. *J. Phys. Lett.* 43, L625.
- Rammal, R. & Toulouse, G. (1983). Random walks on fractal structures. *J. Phys. Lett.* 44, L13.
- Nakayama, T., Yakubo, K. & Orbach, R. (1994). Dynamical properties of fractal networks. *Rev. Mod. Phys.* 66, 381.
""")

    st.markdown("---")
    prev_next("chapter_7")


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def _run_spectral(geometry: str, depth: int, n_walkers: int, n_steps: int) -> None:
    label = f"{geometry} (depth/size={depth})"
    with st.spinner(f"Building {label}…"):
        rng = np.random.default_rng(42)
        if geometry == "Sierpiński carpet":
            G, _, _ = build_carpet(depth)
        elif geometry == "Vicsek fractal":
            G, _, _ = build_vicsek(depth)
        else:
            G, _, _ = build_percolation(depth, 0.50, rng)
            # Use the largest connected component for percolation
            import networkx as nx
            lcc = max(nx.connected_components(G), key=len)
            G = G.subgraph(lcc).copy()

    n_nodes = G.number_of_nodes()

    k_eig = min(300, n_nodes - 2)
    with st.spinner(f"Computing {k_eig} eigenvalues of {n_nodes}-node Laplacian…"):
        eigenvalues = laplacian_eigenvalues(G, k=k_eig)
        d_s_eig, r2_eig = fit_spectral_dim_from_eigenvalues(eigenvalues, fit_frac=0.25)

    with st.spinner(f"Running {n_walkers:,} walkers for {n_steps:,} steps…"):
        rng2 = np.random.default_rng(7)
        t_arr, P_arr = return_probability(G, n_walkers=n_walkers, n_steps=n_steps, rng=rng2)
        t_min = max(20, n_steps // 20)
        t_max = n_steps // 2
        d_s_ret, r2_ret = fit_spectral_dim_from_return_prob(t_arr, P_arr, t_min=t_min, t_max=t_max)

    # Alexander-Orbach from reference values
    theory = _THEORY.get(geometry, {})
    d_f_ref = theory.get("d_f", float("nan"))
    d_w_ref = theory.get("d_w", float("nan"))
    d_s_ao = 2.0 * d_f_ref / d_w_ref if not (np.isnan(d_f_ref) or np.isnan(d_w_ref)) else float("nan")

    result = {
        "geometry": geometry,
        "n_nodes": n_nodes,
        "eigenvalues": eigenvalues,
        "d_s_eig": d_s_eig,
        "r2_eig": r2_eig,
        "t_arr": t_arr,
        "P_arr": P_arr,
        "d_s_ret": d_s_ret,
        "r2_ret": r2_ret,
        "d_s_ao": d_s_ao,
        "d_f_ref": d_f_ref,
        "d_w_ref": d_w_ref,
        "t_min": t_min,
        "t_max": t_max,
    }
    st.session_state["ch7_result"] = result
    _show_spectral(result)


def _show_spectral(result: dict) -> None:
    geometry = result["geometry"]
    eigenvalues = result["eigenvalues"]
    d_s_eig = result["d_s_eig"]
    r2_eig = result["r2_eig"]
    t_arr = result["t_arr"]
    P_arr = result["P_arr"]
    d_s_ret = result["d_s_ret"]
    r2_ret = result["r2_ret"]
    d_s_ao = result["d_s_ao"]
    t_min = result["t_min"]
    t_max = result["t_max"]

    col1, col2 = st.columns(2)

    # --- Left: integrated density of states ---
    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=BG)
        ax.set_facecolor(PANEL)

        N = np.arange(1, len(eigenvalues) + 1)
        ax.plot(eigenvalues, N, color=THEO_C, lw=1.5, label=r"$N(\omega)$")

        # Fit line over bottom 25%
        n_fit = max(10, len(eigenvalues) // 4)
        lx = np.log(eigenvalues[:n_fit])
        ly = np.log(N[:n_fit].astype(float))
        c = np.polyfit(lx, ly, 1)
        x_fit = eigenvalues[:n_fit]
        ax.plot(
            x_fit, np.exp(np.polyval(c, np.log(x_fit))),
            "--", color=MSD_C, lw=2,
            label=f"Fit: slope = $d_s$ = {d_s_eig:.3f}  ($R^2$ = {r2_eig:.3f})",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Eigenvalue $\omega$", color=TEXT_C, fontsize=11)
        ax.set_ylabel(r"$N(\omega)$  (integrated DOS)", color=TEXT_C, fontsize=11)
        ax.set_title("Laplacian spectrum — $N(\\omega) \\sim \\omega^{d_s}$",
                     color=TEXT_C, fontsize=11)
        ax.tick_params(colors=TEXT_C)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # --- Right: return probability ---
    with col2:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=BG)
        ax.set_facecolor(PANEL)

        ax.plot(t_arr[1:], P_arr[1:], color=RMS_C, lw=1.2, alpha=0.9, label=r"$P(t)$")

        # Fit line
        mask = (t_arr >= t_min) & (t_arr <= t_max) & (P_arr > 1e-12)
        if mask.sum() >= 5 and not np.isnan(d_s_ret):
            lx = np.log(t_arr[mask])
            ly = np.log(P_arr[mask])
            c2 = np.polyfit(lx, ly, 1)
            ax.plot(
                t_arr[mask], np.exp(np.polyval(c2, lx)),
                "--", color=THEO_C, lw=2,
                label=f"Fit: $d_s$ = {d_s_ret:.3f}  ($R^2$ = {r2_ret:.3f})",
            )
            # Shade fit window
            ax.axvspan(t_min, t_max, alpha=0.07, color=THEO_C)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Time $t$", color=TEXT_C, fontsize=11)
        ax.set_ylabel(r"$P(t)$  (return probability)", color=TEXT_C, fontsize=11)
        ax.set_title(r"Return probability — $P(t) \sim t^{-d_s/2}$",
                     color=TEXT_C, fontsize=11)
        ax.tick_params(colors=TEXT_C)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # --- Comparison table ---
    st.markdown("#### Three-way verification of $d_s$")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Eigenspectrum", f"{d_s_eig:.3f}" if not np.isnan(d_s_eig) else "N/A",
                  help="Slope of log N(ω) vs log ω")
    with c2:
        st.metric("Return probability", f"{d_s_ret:.3f}" if not np.isnan(d_s_ret) else "N/A",
                  help="Slope of log P(t) vs log t, divided by -1/2")
    with c3:
        label = (f"2×{result['d_f_ref']:.3f} / {result['d_w_ref']:.3f}"
                 if not np.isnan(result["d_f_ref"]) else "")
        st.metric("Alexander-Orbach", f"{d_s_ao:.3f}" if not np.isnan(d_s_ao) else "N/A",
                  help=f"2 d_f / d_w = {label}")
    with c4:
        st.metric("Graph nodes", f"{result['n_nodes']:,}")

    # Consistency check
    vals = [v for v in [d_s_eig, d_s_ret, d_s_ao] if not np.isnan(v)]
    if len(vals) >= 2:
        spread = max(vals) - min(vals)
        if spread < 0.10:
            st.success(f"All three estimates agree within {spread:.3f} — Alexander-Orbach verified.")
        else:
            st.warning(
                f"Spread across estimates: {spread:.3f}. "
                "Try more walkers or a larger graph for better agreement."
            )
