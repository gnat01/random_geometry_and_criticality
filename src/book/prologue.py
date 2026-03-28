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

    st.markdown(r"""
This is a book about two ideas that turn out to be the same idea.

The first is **fractal geometry** — the geometry of objects that look the same at every scale.
A coastline, a snowflake, the branching of a lung: zoom in, and the pattern repeats.
Benoit Mandelbrot gave this phenomenon a name and a number, the **Hausdorff dimension** $d_f$,
which can be non-integer. A fractal curve filling more than a line but less than a plane
has dimension strictly between 1 and 2.

The second idea is the **critical point** of a phase transition. Heat a ferromagnet to the
Curie temperature $T_c$ and something strange happens: fluctuations appear at every length
scale simultaneously. The system is neither ordered nor disordered — it sits exactly at the
boundary between the two phases. The correlation length diverges. Power laws appear everywhere.

The punch line: **a system at its critical point is a fractal**. The geometry of the critical
state has a non-integer dimension. Anomalous diffusion, $r \sim t^{1/d_w}$ with $d_w > 2$,
is the walker's report that the substrate it lives on is a fractal.

This book builds the connection from first principles across fifteen chapters:

**Part 1 — Geometry (I–III).** Start from the definition of fractal dimension. Construct
two canonical deterministic fractals (Sierpiński carpet, Vicsek cross) and one random
fractal (bond percolation). Measure $d_f$ directly from node-count scaling.

**Part 2 — Dynamics on fractals (II, VII, VIII).** Release a random walker on the carpet.
Watch the MSD grow as $t^{2/d_w}$ instead of $t$. Compute the Laplacian spectrum and
extract the spectral dimension $d_s$. Measure first-passage times and see how the
survival function $S(t)$ lies above the Euclidean baseline.

**Part 3 — Critical phenomena (IV, IX, X, XI, XII).** At the percolation threshold,
exact exponents emerge: $\nu = 4/3$, $\beta = 5/36$, $\tau = 187/91$ (all rational in 2-D).
Moving to 3-D removes exact solutions but not the structure. The Ising model adds a
magnetic analogue with its own exact exponents. Real-space renormalisation group explains
*why* universality exists. The transfer matrix provides noise-free $T_c$ extraction
from finite strips.

**Part 4 — Three mechanisms of anomalous diffusion (XIII, XIV, XV, XVI).** Fractal
geometry is only one route to anomalous diffusion. Continuous-time random walks (heavy-tailed
waiting times) and fractional Brownian motion (correlated increments) give the same MSD
exponent from completely different physics. A five-feature classifier — ergodicity ratio,
non-Gaussian parameter, VACF — can tell them apart. Directed percolation adds a new
ingredient: time has an arrow. The resulting universality class governs epidemic spreading,
interface depinning, and reaction-diffusion systems at the extinction threshold.

**Part 5 — The inverse problem (V, VI).** DLA clusters grow fractals from a local
sticking rule. A random forest trained on walk statistics can recover the geometry family —
but with caveats about identifiability and finite-sample noise.

---

Each chapter has an interactive experiment. The results are not toys: they reproduce
the physics you would see in a research-grade simulation, scaled to run in seconds.
The numbers you measure should match theory to within a few percent.

Press **▶ Run** and watch the physics speak.
""")

    key_result(
        "Central claim: critical phenomena and fractal geometry are two faces of the same "
        "mathematical structure. Anomalous diffusion ($d_w > 2$) is the walker's way of "
        "reporting that the substrate it lives on is a fractal."
    )

    st.markdown("---")
    prev_next("prologue")
