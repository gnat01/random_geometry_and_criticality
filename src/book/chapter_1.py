"""Chapter I: The Geometry of Fractals."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..graphs.carpet import build_carpet
from ..graphs.vicsek import build_vicsek
from ..theme import BG, THEO_C, TEXT_C
from ..ui.plotting import draw_graph_2d
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited


def render() -> None:
    mark_visited("chapter_1")
    chapter_header(
        "I",
        "The Geometry of Fractals",
        "Self-similarity, Hausdorff dimension, and two canonical examples",
    )

    st.markdown("""
A regular square has area that scales as $L^2$. A line segment has length that scales as $L^1$.
Both statements feel obvious, and in both cases the exponent is an integer.

Now consider the **Sierpiński carpet**. Start with a $3 \\times 3$ grid and remove the centre
cell. Apply the same rule to each surviving cell. Repeat. After $n$ iterations you have a
porous object that lives in the plane but is not the plane. How does its "size" scale with $L$?

At depth $n$, the carpet fits in a box of side $3^n$. It contains $8^n$ unit cells (the factor
of 8 comes from the 8 surviving sub-squares at each step). So:

$$N(L) \\sim L^{d_f}, \\quad d_f = \\frac{\\log 8}{\\log 3} \\approx 1.893.$$

This is the **Hausdorff dimension** — or equivalently the **box-counting dimension** — and it
is not an integer. The carpet is strictly "larger" than a curve ($d_f > 1$) but strictly
"smaller" than a filled square ($d_f < 2$). That is what it means to be a fractal.

The **Vicsek fractal** (a cross-shaped variant) keeps only 5 sub-squares instead of 8:
""")

    st.latex(r"d_f^{\mathrm{Vicsek}} = \frac{\log 5}{\log 3} \approx 1.465.")

    st.markdown("""
Both fractals are **exactly self-similar**: the whole is made of smaller copies of itself.
This is the defining property that makes the Hausdorff dimension well-defined and
non-trivial.

---

### Experiment: measure $d_f$ from node-count scaling

Vary the depth $n$ and watch how the number of graph nodes grows. On a log-log plot, the
slope is $d_f$.
""")

    depth = st.slider("Recursion depth", min_value=1, max_value=4, value=3, key="ch1_depth")

    if st.button("▶ Run", key="ch1_run"):
        _run_experiment(depth)
    elif "ch1_result" in st.session_state:
        _show_result(st.session_state["ch1_result"])

    st.markdown("""
---

### Reflection

Notice that the carpet is *denser* than the Vicsek fractal: more cells survive at each
iteration, so $d_f$ is higher. Yet neither fills the plane ($d_f < 2$), and neither is
just a curve ($d_f > 1$). The dimension captures this in a single number.

You might ask: why should a physicist care about this distinction? The answer becomes clear
in later chapters. The walk dimension $d_w$ and the spectral dimension $d_s = 2d_f/d_w$ both
depend on $d_f$. A walker's long-time behaviour — how fast it spreads, how often it returns
to the origin — is controlled by these three numbers and their relationships.
""")

    key_result(
        "Sierpiński carpet: $d_f = \\log 8 / \\log 3 \\approx 1.893$, "
        "$d_w \\approx 2.88$. &nbsp;"
        "Vicsek fractal: $d_f = \\log 5 / \\log 3 \\approx 1.465$, "
        "$d_w = \\log 15 / \\log 3 \\approx 2.46$."
    )

    st.markdown("---")
    prev_next("chapter_1")


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def _run_experiment(depth: int) -> None:
    depths = list(range(1, depth + 1))
    carpet_counts = []
    vicsek_counts = []

    with st.spinner("Building fractal graphs…"):
        for d in depths:
            Gc, _, _ = build_carpet(d)
            Gv, _, _ = build_vicsek(d)
            carpet_counts.append(Gc.number_of_nodes())
            vicsek_counts.append(Gv.number_of_nodes())

    result = {
        "depths": depths,
        "carpet_counts": carpet_counts,
        "vicsek_counts": vicsek_counts,
        "depth": depth,
    }
    st.session_state["ch1_result"] = result
    _show_result(result)


def _show_result(result: dict) -> None:
    depths = result["depths"]
    carpet_counts = result["carpet_counts"]
    vicsek_counts = result["vicsek_counts"]
    max_depth = result["depth"]

    # Fit slopes
    log_L = np.log(np.array([3**d for d in depths], dtype=float))
    log_Nc = np.log(np.array(carpet_counts, dtype=float))
    log_Nv = np.log(np.array(vicsek_counts, dtype=float))
    slope_c = float(np.polyfit(log_L, log_Nc, 1)[0]) if len(depths) >= 2 else np.nan
    slope_v = float(np.polyfit(log_L, log_Nv, 1)[0]) if len(depths) >= 2 else np.nan

    theory_c = np.log(8) / np.log(3)
    theory_v = np.log(5) / np.log(3)

    # --- Figure 1: side-by-side fractal pictures at max depth ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div style='text-align:center; color:{THEO_C}; font-size:0.9rem;"
            f" font-weight:600;'>Sierpiński Carpet — depth {max_depth} "
            f"({carpet_counts[-1]:,} nodes)</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Drawing carpet…"):
            Gc, pos_c, _ = build_carpet(max_depth)
        fig_c, ax_c = plt.subplots(figsize=(5, 5), facecolor=BG)
        draw_graph_2d(Gc, pos_c, ax=ax_c, node_size=max(1, 18 - 3 * max_depth), edge_alpha=0.3)
        ax_c.set_title(
            f"$d_f \\approx {theory_c:.3f}$  (measured {slope_c:.3f})",
            color=TEXT_C,
            fontsize=10,
            pad=6,
        )
        st.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)

    with col2:
        st.markdown(
            f"<div style='text-align:center; color:{THEO_C}; font-size:0.9rem;"
            f" font-weight:600;'>Vicsek Fractal — depth {max_depth} "
            f"({vicsek_counts[-1]:,} nodes)</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Drawing Vicsek…"):
            Gv, pos_v, _ = build_vicsek(max_depth)
        fig_v, ax_v = plt.subplots(figsize=(5, 5), facecolor=BG)
        draw_graph_2d(Gv, pos_v, ax=ax_v, node_size=max(2, 20 - 3 * max_depth), edge_alpha=0.4)
        ax_v.set_title(
            f"$d_f \\approx {theory_v:.3f}$  (measured {slope_v:.3f})",
            color=TEXT_C,
            fontsize=10,
            pad=6,
        )
        st.pyplot(fig_v, use_container_width=True)
        plt.close(fig_v)

    # --- Figure 2: node-count scaling log-log ---
    if len(depths) >= 2:
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor=BG)
        ax2.set_facecolor("#161b22")
        Ls = np.array([3**d for d in depths], dtype=float)

        ax2.plot(Ls, carpet_counts, "o-", color="#ff6b6b", label="Carpet (measured)", lw=2, ms=7)
        ax2.plot(Ls, vicsek_counts, "s-", color="#a78bfa", label="Vicsek (measured)", lw=2, ms=7)

        # Theory lines
        Ls_fine = np.linspace(Ls[0] * 0.9, Ls[-1] * 1.1, 60)
        for slope, color, lbl in [
            (theory_c, THEO_C, f"Theory $d_f={theory_c:.3f}$"),
            (theory_v, "#4a9fd4", f"Theory $d_f={theory_v:.3f}$"),
        ]:
            ref_idx = 0
            A = carpet_counts[ref_idx] / (Ls[ref_idx] ** slope) if lbl.startswith("Theory $d_f=1.8") else \
                vicsek_counts[ref_idx] / (Ls[ref_idx] ** slope)
            ax2.plot(Ls_fine, A * Ls_fine**slope, "--", color=color, alpha=0.7, label=lbl, lw=1.5)

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Box side $L = 3^n$", color=TEXT_C)
        ax2.set_ylabel("Node count $N$", color=TEXT_C)
        ax2.set_title("Node-count scaling: slope = $d_f$", color=TEXT_C, fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        col_a, col_b = st.columns(2)
        with col_a:
            delta_c = abs(slope_c - theory_c) / theory_c * 100
            st.metric(
                "Carpet $d_f$ (measured / theory)",
                f"{slope_c:.3f} / {theory_c:.3f}",
                delta=f"{delta_c:.1f}% error",
                delta_color="off",
            )
        with col_b:
            delta_v = abs(slope_v - theory_v) / theory_v * 100
            st.metric(
                "Vicsek $d_f$ (measured / theory)",
                f"{slope_v:.3f} / {theory_v:.3f}",
                delta=f"{delta_v:.1f}% error",
                delta_color="off",
            )
