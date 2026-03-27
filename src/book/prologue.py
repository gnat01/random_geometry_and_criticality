"""Prologue: the unifying thread."""

from __future__ import annotations

import streamlit as st

from .nav import chapter_header, key_result, prev_next, mark_visited


def render() -> None:
    mark_visited("prologue")
    chapter_header(
        None,
        "Prologue",
        "Geometry, randomness, and the edge of order",
    )

    st.markdown("""
This is a short book about two ideas that turn out to be the same idea.

The first is **fractal geometry** — the geometry of objects that look the same at every scale.
A coastline, a snowflake, the branching of a lung: zoom in, and the pattern repeats.
Benoit Mandelbrot gave this phenomenon a name and a number, the **Hausdorff dimension**,
which can be non-integer. A fractal curve filling more than a line but less than a plane
has a dimension strictly between 1 and 2.

The second idea is the **critical point** of a phase transition. Heat a ferromagnet and at
the Curie temperature something strange happens: fluctuations appear at every length scale
simultaneously. The system is neither ordered nor disordered — it sits exactly at the
boundary between the two phases. The correlation length diverges. Power laws appear everywhere.

The punch line is this: **a system at its critical point is a fractal**. The geometry of the
critical state — its clusters, its connected components, its support — has a non-integer
dimension. Anomalous diffusion, in which a random walker spreads as $r \\sim t^{1/d_w}$ with
$d_w > 2$, is the dynamical signature of walking on that fractal geometry.

This book builds the connection from the ground up. You will:

1. See two canonical fractal graphs (Sierpiński carpet and Vicsek fractal) and measure
   their dimensions directly from node-count scaling.
2. Watch a random walker spread anomalously on the carpet and fit the walk dimension $d_w$.
3. Build percolation networks — random graphs that become fractal precisely at the critical
   bond probability $p_c = \\tfrac{1}{2}$.
4. Measure the exact critical exponents $\\nu$, $\\beta$, and $\\tau$ numerically and verify
   the finite-size scaling collapse.
5. Grow DLA clusters whose fractal dimension emerges purely from a local sticking rule.
6. Ask whether a machine-learning classifier can identify the geometry type from walk
   statistics alone — and discover what features carry the most information.

Each chapter has a short experiment you can run in your browser. The experiments are not
toys: they reproduce the same physics you would see in a research-grade simulation, scaled
down to run in seconds. The numbers you measure should match the theory to within a few
percent.

Start reading. Then press **▶ Run** and watch the fractals speak.
""")

    key_result(
        "Central claim: critical phenomena and fractal geometry are two faces of the same "
        "mathematical structure. Anomalous diffusion ($d_w > 2$) is the walker's way of "
        "reporting that the substrate it lives on is a fractal."
    )

    st.markdown("---")
    prev_next("prologue")
