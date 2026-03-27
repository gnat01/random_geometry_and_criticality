"""Epilogue: the full arc."""

from __future__ import annotations

import streamlit as st

from ..theme import THEO_C, TEXT_C
from .nav import chapter_header, key_result, prev_next, mark_visited


def render() -> None:
    mark_visited("epilogue")
    chapter_header(
        None,
        "Epilogue",
        "The full arc: from geometry to criticality and back",
    )

    st.markdown("""
You started with two fractals — the Sierpiński carpet and the Vicsek fractal — and a
single dimensionless number, the Hausdorff dimension $d_f$, that measures how much space
they fill. That number, you saw, is encoded in the scaling of node counts: $N \\sim L^{d_f}$.

Then you walked on them. The random walker found a new number, the walk dimension $d_w$,
which is *not* determined by $d_f$ alone. It depends on the connectivity structure of
the fractal — how the branches are joined, how bottlenecks slow the walk. The subdiffusive
scaling $\\langle r^2 \\rangle \\sim t^{2/d_w}$ with $d_w > 2$ is the dynamical signature
of walking on a fractal.

Then disorder entered. Bond percolation introduced randomness into the geometry.
Below $p_c$, the network is fragmented. Above $p_c$, it is well-connected. At $p_c = 1/2$
exactly, the network is a random fractal — self-similar in a statistical sense, with
the same anomalous walk dimension as the carpet. The transition between the two phases
is sharp, and the geometry of the critical cluster carries exact power-law exponents:
$\\beta = 5/36$, $\\nu = 4/3$, $\\tau = 187/91$.

These exponents are **universal**: they don't depend on the microscopic details of the
model. Two completely different models — diluted magnets and porous media, say — will
have the same exponents if they belong to the same universality class. This is one of the
deepest facts in statistical physics.

DLA showed you that fractals can grow, not just be constructed. The Walker-Sander
cluster that emerges from purely local diffusion has a fractal dimension that emerges
from the global geometry of the harmonic measure. The fractal structure is not put in by
hand — it is *selected* by the dynamics.

Finally, the inverse problem: a machine learning classifier showed that the walk dimension
is the most informative feature for identifying the geometry family. The walker doesn't
see the graph — it discovers the geometry through its own anomalous spreading.

---

### The thread

Here is the full arc, stated plainly:

**A critical point is a fractal.** At the transition, the correlation length $\\xi$
diverges, meaning the system is scale-invariant. Scale invariance is the defining
property of a fractal. So the critical state has fractal geometry — clusters of all
sizes, power-law distributions, and a non-integer effective dimension.

**A random walker on a fractal is anomalously slow.** The walk dimension $d_w > 2$
because the fractal geometry forces detours. This slowness is measurable — in a
simulation, a lab experiment, or a single-molecule tracking dataset.

**Dynamics encodes geometry.** The MSD, the return probability, the distribution of
first-passage times — all of these carry information about the substrate's fractal
structure. A sufficiently careful observer can reconstruct the geometry from the dynamics.
This is the inverse problem, and it matters whenever the geometry is hidden but the
dynamics is observable.

The three ideas — fractal geometry, critical phenomena, anomalous diffusion — form a
single unified framework. You started with pictures. You end with equations. Between
them, you ran the experiments and saw the numbers come out right.
""")

    key_result(
        "The three pillars: $d_f$ (Hausdorff dimension of the geometry), "
        "$d_w$ (walk dimension of the diffusion), "
        "$d_s = 2d_f/d_w$ (spectral dimension of the vibrational modes). "
        "These three numbers classify the large-scale behaviour of any random walk "
        "on any fractal substrate."
    )

    st.markdown("""
---

### Further reading

- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature.* Freeman.
- Havlin, S. & Ben-Avraham, D. (1987). Diffusion in disordered media. *Advances in Physics*, 36(6), 695–798.
- Stauffer, D. & Aharony, A. (1992). *Introduction to Percolation Theory.* Taylor & Francis.
- Witten, T. A. & Sander, L. M. (1981). Diffusion-limited aggregation. *PRL* 47, 1400.
- Alexander, S. & Orbach, R. (1982). Density of states on fractals: fractons. *J. Phys. Lett.* 43, L625.
""")

    st.markdown("---")
    prev_next("epilogue")
