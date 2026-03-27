"""Chapter II: Walking on a Fractal."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..graphs.carpet import build_carpet
from ..sim.observables import estimate_msd_rms_series
from ..theme import BG, MSD_C, RMS_C, THEO_C, TEXT_C
from ..ui.plotting import pos_walk_square
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited


def render() -> None:
    mark_visited("chapter_2")
    chapter_header(
        "II",
        "Walking on a Fractal",
        "Anomalous diffusion and the walk dimension",
    )

    st.markdown("""
On a regular two-dimensional lattice, a random walker's mean-squared displacement grows
linearly with time:

$$\\langle r^2(t) \\rangle \\sim t.$$

This is **normal diffusion**. The key word is "linear". The exponent is exactly 1.

On a fractal, the walker is perpetually running into dead ends, bottlenecks, and
branching structures that force it to backtrack. Progress is slower. The MSD still grows
as a power law, but the exponent is less than 1:

$$\\langle r^2(t) \\rangle \\sim t^{2/d_w}, \\quad d_w > 2.$$

This is **anomalous diffusion** (specifically, **subdiffusion**). The number $d_w$ is called
the **walk dimension** and it encodes everything about the geometry's resistance to diffusion.
""")

    definition_box(
        "Walk dimension $d_w$",
        "The exponent in $\\langle r^2 \\rangle \\sim t^{2/d_w}$. "
        "For a regular lattice in $d$ dimensions, $d_w = 2$. "
        "On a fractal, $d_w > 2$, meaning the walker spreads more slowly. "
        "For the Sierpiński carpet, $d_w \\approx 2.88$."
    )

    st.markdown("""
Why is $d_w$ larger than 2 on a fractal? Intuitively, the holes in the carpet force the
walker to take longer detours to travel the same Euclidean distance. More quantitatively,
Alexander and Orbach showed in 1982 that the three fractal dimensions are related by

$$d_s = \\frac{2 d_f}{d_w},$$

where $d_s$ is the **spectral dimension** (which controls the density of states and
return-to-origin probability). For the carpet, $d_s \\approx 2 \\times 1.893 / 2.88 \\approx 1.31$.

---

### Experiment: MSD on the Sierpiński carpet

We release 80 walkers from the same central node on a depth-3 carpet and track how the
mean-squared displacement grows with time. On a log-log plot, the slope is $2/d_w$. We
compare with the theoretical prediction.
""")

    n_walkers = st.slider("Number of walkers", 20, 200, 80, step=10, key="ch2_walkers")
    n_steps = st.slider("Steps", 200, 2000, 1200, step=100, key="ch2_steps")

    if st.button("▶ Run", key="ch2_run"):
        _run_experiment(n_walkers, n_steps)
    elif "ch2_msd" in st.session_state:
        _show_result(st.session_state["ch2_msd"])

    st.markdown("""
---

### Reflection

The measured slope should come out near $2/d_w \\approx 2/2.88 \\approx 0.694$.
If you use more walkers or more steps, the fit stabilises.

Notice that the log-log plot is not perfectly straight — especially at early times
(when the walker hasn't explored enough of the fractal structure) and very late times
(when it's wrapped around the finite graph several times). The clean power-law regime
sits in the middle: this is always where you look for scaling behaviour.

One of the deepest results in this field is that $d_w$ is not determined by $d_f$ alone.
Two fractals with the same Hausdorff dimension can have different walk dimensions. You need
to know something about the *connectivity* of the fractal — how the branches are joined —
to determine $d_w$. That's why the spectral dimension $d_s$ carries independent information.
""")

    key_result(
        "On the Sierpiński carpet: $d_w \\approx 2.88$, so "
        "$\\langle r^2 \\rangle \\sim t^{0.694}$. "
        "Compare to a regular 2D lattice where $\\langle r^2 \\rangle \\sim t^{1.0}$."
    )

    st.markdown("---")
    prev_next("chapter_2")


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def _run_experiment(n_walkers: int, n_steps: int) -> None:
    with st.spinner(f"Running {n_walkers} walkers for {n_steps} steps on carpet depth 3…"):
        G, pos, _ = build_carpet(3)
        nodes = list(G.nodes())
        # Start from the node closest to the geometric centre
        coords = np.array([pos[n] for n in nodes])
        centre = coords.mean(axis=0)
        dists = np.linalg.norm(coords - centre, axis=1)
        start = nodes[int(np.argmin(dists))]

        pos_walk = pos_walk_square(nodes)
        rng = np.random.default_rng(42)
        steps, rms, msd = estimate_msd_rms_series(
            G, pos_walk, start, n_walkers, n_steps, rng
        )

    result = {"steps": steps, "rms": rms, "msd": msd}
    st.session_state["ch2_msd"] = result
    _show_result(result)


def _show_result(result: dict) -> None:
    steps = result["steps"]
    msd = result["msd"]
    rms = result["rms"]

    # Fit in middle regime
    mask = (steps > 20) & (steps < steps[-1] * 0.7) & (msd > 1e-12)
    if mask.sum() >= 6:
        slope, intercept = np.polyfit(np.log(steps[mask]), np.log(msd[mask]), 1)
        d_w_meas = 2.0 / slope if abs(slope) > 1e-9 else float("nan")
    else:
        slope, intercept = float("nan"), float("nan")
        d_w_meas = float("nan")

    d_w_theory = 2.88
    theory_exp = 2.0 / d_w_theory
    theory_A = float(np.exp(intercept)) if not np.isnan(intercept) else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)

    # --- Left: linear scale ---
    ax = axes[0]
    ax.set_facecolor("#161b22")
    ax.plot(steps, msd, color=MSD_C, lw=1.5, alpha=0.9, label="MSD (measured)")
    ax.set_xlabel("Steps $t$", color=TEXT_C)
    ax.set_ylabel(r"$\langle r^2 \rangle$", color=TEXT_C)
    ax.set_title("MSD vs time (linear)", color=TEXT_C, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Right: log-log scale ---
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")

    ax2.plot(steps, msd, color=MSD_C, lw=1.5, alpha=0.9, label="MSD (measured)", zorder=3)
    ax2.plot(steps, rms**2, color=RMS_C, lw=1, alpha=0.5, label="RMS² check", zorder=2)

    # Fitted line
    if not np.isnan(slope):
        fit_line = theory_A * steps**slope
        ax2.plot(steps, fit_line, "--", color="#4caf50", lw=2,
                 label=f"Fit slope = {slope:.3f}  ($d_w$ = {d_w_meas:.2f})", zorder=4)

    # Theory line
    theory_line = theory_A * steps**theory_exp
    ax2.plot(steps, theory_line, ":", color=THEO_C, lw=2,
             label=f"Theory $2/d_w = {theory_exp:.3f}$  ($d_w = {d_w_theory}$)", zorder=4)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Steps $t$", color=TEXT_C)
    ax2.set_ylabel(r"$\langle r^2 \rangle$", color=TEXT_C)
    ax2.set_title("Log-log: slope = $2/d_w$", color=TEXT_C, fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Measured slope $2/d_w$", f"{slope:.3f}" if not np.isnan(slope) else "N/A")
    with col2:
        st.metric("Measured $d_w$", f"{d_w_meas:.3f}" if not np.isnan(d_w_meas) else "N/A")
    with col3:
        st.metric("Theory $d_w$", f"{d_w_theory:.2f}")
