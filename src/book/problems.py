"""Problems: coding challenges and experiments, with hints, spanning all chapters."""

from __future__ import annotations

import streamlit as st

from .nav import chapter_header, prev_next, mark_visited


# ---------------------------------------------------------------------------
# Helper: render a single problem block with optional expandable hints
# ---------------------------------------------------------------------------

def _problem(
    label: str,
    statement: str,
    physics_hint: str | None = None,
    code_hint: str | None = None,
) -> None:
    st.markdown(f"**{label}**")
    st.markdown(statement)
    if physics_hint or code_hint:
        cols = st.columns(2 if (physics_hint and code_hint) else 1)
        if physics_hint:
            with cols[0]:
                with st.expander("Physics hint"):
                    st.markdown(physics_hint)
        if code_hint:
            with cols[-1]:
                with st.expander("Programming hint"):
                    st.markdown(code_hint)
    st.markdown("")  # spacing


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    mark_visited("problems")
    chapter_header(
        None,
        "Problems",
        "Exercises spanning all chapters — expand the hints only if you need them",
    )

    st.markdown("""
Each problem is self-contained but references the chapter where the relevant
ideas are developed. Hints are hidden behind expanders — try the problem first.
Some problems are quick experiments (an hour); others are open-ended investigations.
""")

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter I — The Geometry of Fractals")
    # -----------------------------------------------------------------------

    _problem(
        "P1.1 — Box-counting dimension",
        """Write a pure-NumPy box-counting function that works on any binary 2D image:
divide the image into $\\varepsilon \\times \\varepsilon$ boxes and count how many contain
at least one "on" pixel. Plot $\\log N(\\varepsilon)$ vs $\\log(1/\\varepsilon)$ for the
Sierpiński carpet, the Vicsek fractal, and a filled circle. Compare slopes to
analytical values ($\\log 8/\\log 3$, $\\log 5/\\log 3$, and $2$ respectively).""",
        physics_hint="""The slope should be independent of $\\varepsilon$ over a
well-defined scaling range. Very small $\\varepsilon$ hits pixel noise; very large
$\\varepsilon$ hits the finite image boundary. The scaling range is the plateau in the
local slope $d\\log N / d\\log(1/\\varepsilon)$.""",
        code_hint="""Render the fractal as a boolean NumPy array at a fixed depth.
For each $\\varepsilon$, reshape the array into blocks of size $\\varepsilon$ and use
`np.any(block)` to count non-empty boxes. `np.lib.stride_tricks.sliding_window_view`
or simple reshaping both work if the image size is a power of $\\varepsilon$.""",
    )

    _problem(
        "P1.2 — Dimension from node counts",
        """For a self-similar fractal with $N$ copies rescaled by $r$, the Hausdorff
dimension is $d_f = \\log N / \\log(1/r)$. Build the carpet graph at levels $k = 1, 2, 3, 4$
and verify that the node count $n(k) = 8^k$ exactly. Show that
$\\log n(k) / \\log(3^k) \\to d_f$ as $k$ increases. Do the same for the Vicsek fractal.""",
        physics_hint="""At finite $k$ the ratio $\\log n(k)/\\log L(k)$ approaches $d_f$
from below. The convergence is slow — this is a finite-size correction, the same
phenomenon you see in FSS but for the fractal construction rather than for a
statistical model.""",
        code_hint="""Use `len(G.nodes())` on the graph returned by `build_carpet(k)`
and `build_vicsek(k)`. Plot $k$ on the x-axis and the ratio on the y-axis; it should
flatline by $k = 3$.""",
    )

    _problem(
        "P1.3 — The T-square fractal",
        """The T-square is built by attaching four half-size squares to the corners
of each existing square at every step. Its Hausdorff dimension is $d_f = 2$ (it fills the
plane in the limit), yet its area fraction is strictly less than 1 at any finite level.
Implement the T-square as a graph of tile centres with edges between adjacent tiles.
Measure its box-counting dimension numerically and reconcile the result with $d_f = 2$.""",
        physics_hint="""A set can have $d_f = 2$ yet zero area measure — this is why
Hausdorff dimension and Lebesgue measure are different concepts. The T-square's area
fraction grows to 1 only in the limit $k \\to \\infty$, but at every finite level the
"holes" between branches keep it strictly below 1.""",
        code_hint="""Represent the fractal as a set of $(x, y, \\text{level})$ tuples.
At each step, for every existing tile at level $k$ add four new tiles of half the
side length at the corners. Use a set to avoid duplicates. Edges connect tiles whose
centres are within $1.5 \\times \\text{tile\\_size}$ of each other.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter II — Walking on a Fractal")
    # -----------------------------------------------------------------------

    _problem(
        "P2.1 — Walk dimension from the MSD",
        """Run $n = 2000$ walkers for $T = 5000$ steps each on the Sierpiński carpet
at level $k = 4$. Compute the ensemble-averaged MSD $\\langle r^2(t) \\rangle$ and fit the
log-log slope over the range $t \\in [50, 2000]$. The slope is $2/d_w$; compare your
estimate of $d_w$ to the theoretical value $\\approx 2.88$.
Repeat for the Vicsek fractal and check $d_w = \\log 15/\\log 3 \\approx 2.46$.""",
        physics_hint="""The MSD has a transient regime at short times (the walker
hasn't yet "felt" the fractal geometry) and a saturation regime at long times
(the walker has explored the whole finite graph). The scaling regime between these
two is where you measure $d_w$. Make the graph larger to push saturation to later times.""",
        code_hint="""Use `batch_random_steps` from `src.sim.walks` with `n_walkers`
and `n_steps`. Store node positions from the graph's `pos` attribute. The MSD at time
$t$ is `np.mean(np.sum((positions[t] - positions[0])**2, axis=1))`. Fit with
`np.polyfit(np.log(t_arr), np.log(msd_arr), 1)`.""",
    )

    _problem(
        "P2.2 — Return probability and spectral dimension",
        """The probability that a walker is at its starting node at time $t$ scales as
$P(t) \\sim t^{-d_s/2}$ where $d_s = 2d_f/d_w$ is the spectral dimension.
Run 5000 walkers on the carpet and measure the fraction at their origin at each $t$.
Fit the log-log slope, extract $d_s$, and verify the Alexander-Orbach relation
$d_s = 2 d_f / d_w$ using independently measured $d_f$ and $d_w$.""",
        physics_hint="""For $d_s < 2$ the walk is *recurrent*: the walker returns to
its origin with probability 1 (just slowly). For $d_s > 2$ it is *transient*.
The carpet has $d_s \\approx 1.31 < 2$, so it is recurrent — this is why the random
walk explores the fractal ergodically despite the anomalous slowness.""",
        code_hint="""Track the starting node index for each walker. At each step,
`P(t) = np.mean(current_node == start_node)`. This is a single boolean comparison
per walker per step — very cheap. Use at least 5000 walkers and 3000 steps for
a clean power-law tail.""",
    )

    _problem(
        "P2.3 — Biased walk and drift velocity",
        """Apply a gradient bias (toward the right edge) of strength $F$ to walkers
on the carpet using `make_gradient_bias`. Measure the drift velocity
$v = \\langle x(t) \\rangle / t$ for $F \\in \\{0.5, 1.0, 2.0, 4.0\\}$.
On a regular 2D grid, $v \\propto F$ (linear response / Einstein relation).
Does the same hold on the carpet? Plot $v$ vs $F$ for both substrates.""",
        physics_hint="""At small $F$ the response is always linear — this is just
the fluctuation-dissipation theorem. At large $F$ the walker gets "funnelled"
into channels aligned with the field, and the response saturates. The carpet may
show a different crossover scale than the regular grid due to its bottleneck geometry.""",
        code_hint="""Use `biased_batch_random_steps` with `make_gradient_bias(pos, target, strength=F)`.
Extract $\\langle x(t) \\rangle$ by averaging the x-coordinate of all walker positions
at each time step, then fit the slope over the linear regime $t \\in [10, 500]$.""",
    )

    _problem(
        "P2.4 — Lévy crossover",
        """Add Lévy long-range edges to the Sierpiński carpet with exponent
$\\alpha \\in \\{0.5, 1.0, 1.5, 2.0\\}$ and density $p_{\\text{long}} = 0.05$.
For each $\\alpha$, measure $d_w$ from the MSD slope. Plot $d_w$ vs $\\alpha$.
At small $\\alpha$ (many long jumps) $d_w$ should approach 2; at large $\\alpha$
it should recover the carpet value $\\approx 2.88$. Locate the crossover $\\alpha^*$.""",
        physics_hint="""The Lévy exponent $\\alpha$ controls the tail of the jump
length distribution: $p(\\ell) \\sim \\ell^{-(\\alpha+2)}$. For $\\alpha < d_w - d_f$
the long jumps dominate transport and the walk becomes superdiffusive ($d_w < 2$).
Above the crossover the fractal geometry dominates.""",
        code_hint="""Use `add_levy_edges(G, pos, alpha=alpha, p_long=0.05)` from
`src.graphs.modifiers`. Run the walk on the augmented graph with `batch_random_steps`.
Note that the Lévy edges add shortcuts, so the graph is no longer sparse — time
per step increases with $p_{\\text{long}}$.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter III — Disorder and the Random Substrate")
    # -----------------------------------------------------------------------

    _problem(
        "P3.1 — Cluster-size distribution at $p_c$",
        """At $p_c = 1/2$, the cluster-size distribution follows $n_s \\sim s^{-\\tau}$
with $\\tau = 187/91 \\approx 2.05$. Generate 500 realisations on a $100 \\times 100$
grid at $p = 0.5$. For each realisation exclude the largest cluster and histogram all
remaining cluster sizes on a log-log scale. Fit the slope. How close do you get to 2.05?
Repeat at $p = 0.45$ and $p = 0.55$ and show that the power law breaks away from $p_c$.""",
        physics_hint="""$\\tau \\approx 2.05$ is just above 2, which means the
second moment $\\langle s^2 \\rangle$ (proportional to the susceptibility $\\chi$)
diverges logarithmically at $p_c$ — a very weak divergence. This is why finite-size
effects are severe and why you need many realisations for a clean fit.""",
        code_hint="""Use `networkx.connected_components` to get all component sizes.
Sort and exclude the maximum. For the histogram, use `np.histogram` with logarithmically
spaced bins (`np.logspace`) and plot bin centres vs counts/bin-width on a log-log scale.
Fit only the range $s \\in [5, 200]$ to avoid small-cluster noise and finite-size truncation.""",
    )

    _problem(
        "P3.2 — Finite-size scaling collapse",
        """The order parameter satisfies $P_\\infty(L,p) = L^{-\\beta/\\nu} f[(p-p_c)L^{1/\\nu}]$
with $\\beta=5/36$, $\\nu=4/3$, $p_c=1/2$. Compute $P_\\infty$ for
$L \\in \\{20, 30, 50, 80\\}$ and $p \\in [0.3, 0.7]$ (100 samples per point).
First plot the raw $P_\\infty(p)$ curves — they should all cross near $p_c$.
Then plot $P_\\infty L^{\\beta/\\nu}$ vs $(p-p_c)L^{1/\\nu}$: the four curves should
collapse onto a single universal function.""",
        physics_hint="""The crossing point of raw $P_\\infty$ curves for different
$L$ is a finite-size estimator of $p_c$. It converges to $p_c = 0.5$ as $L \\to \\infty$
with corrections $\\sim L^{-\\omega}$, $\\omega \\approx 3/4$. The quality of the data
collapse after rescaling is a direct visual test of the scaling hypothesis.""",
        code_hint="""Use `order_parameter_sweep` from `src.sim.criticality`.
For the collapse plot, compute `x = (p - 0.5) * L**(3/4)` and `y = P_inf * L**(5/48)`
for each $(L, p)$ point. Plot all points on a single axes — if they lie on a single
curve, the collapse works. Try perturbing $\\nu$ by $\\pm 0.1$ and watch the collapse degrade.""",
    )

    _problem(
        "P3.3 — Crossing-point estimator of $p_c$",
        """The pairwise crossing point $p^*(L_1, L_2)$ of $P_\\infty(L_1,p)$ and
$P_\\infty(L_2,p)$ converges to $p_c$ as $\\min(L_1, L_2) \\to \\infty$.
Extract all pairwise crossings from your data in P3.2. Plot $p^*$ vs $1/L_\\text{min}$
and extrapolate to $1/L \\to 0$. How close to $0.5$ do you land?""",
        physics_hint="""The correction-to-scaling exponent $\\omega \\approx 3/4$
means $p^*(L) - p_c \\sim L^{-(1/\\nu + \\omega)}$. Fitting this power law to your
crossing points is a finite-size scaling analysis — the same technique used
experimentally to locate phase transitions in real materials.""",
        code_hint="""Interpolate each $P_\\infty(p)$ curve with `np.interp` or
`scipy.interpolate.interp1d`. Find the crossing between pairs $(L_i, L_j)$ by
finding the zero of their difference using `scipy.optimize.brentq`.""",
    )

    _problem(
        "P3.4 — Critical slowing down of diffusion",
        """Measure the MSD exponent $2/d_w$ for random walks on the largest component
of percolation graphs at $p \\in \\{0.5, 0.55, 0.6, 0.7, 0.8, 1.0\\}$ on a
$60 \\times 60$ grid. Plot the exponent vs $p$. At $p = 1$ (no disorder) the
grid is regular so $d_w = 2$ and the exponent is $1$. At $p = p_c = 0.5$ the
exponent dips to its minimum. Estimate the minimum value and compare to
$2/d_w \\approx 2/2.87 \\approx 0.70$.""",
        physics_hint="""Below $p_c$ the largest component is finite and the walk
saturates — $d_w$ is ill-defined. Just above $p_c$ the incipient infinite cluster
is fractal and $d_w > 2$. Well above $p_c$ the cluster becomes increasingly
Euclidean and $d_w \\to 2$. The crossover is not sharp — $d_w$ relaxes slowly
to 2 as $p \\to 1$.""",
        code_hint="""Restrict walkers to the largest connected component only
(use `max(networkx.connected_components(G), key=len)`). Start walkers from random
nodes inside that component. Fit MSD slope over $t \\in [20, 300]$.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter IV — The Critical Point")
    # -----------------------------------------------------------------------

    _problem(
        "P4.1 — Susceptibility divergence",
        """The susceptibility $\\chi = \\langle s^2 \\rangle / \\langle s \\rangle$
(mean cluster size, excluding the infinite cluster) diverges at $p_c$ as
$\\chi \\sim |p - p_c|^{-\\gamma}$ with $\\gamma = 43/18 \\approx 2.39$.
Measure $\\chi$ for $p \\in [0.35, 0.65]$ on a $L = 80$ grid (200 samples per $p$).
Plot $\\log \\chi$ vs $\\log |p - 0.5|$ on both sides of $p_c$ and verify the exponent.""",
        physics_hint="""$\\chi$ is the second moment of the cluster-size distribution.
Because $\\tau \\approx 2.05 < 3$, both $\\langle s \\rangle$ and $\\langle s^2 \\rangle$
are dominated by the largest finite clusters — a few very large clusters contribute
almost all of the susceptibility. This is why $\\chi$ is so noisy near $p_c$.""",
        code_hint="""For each realisation, compute `sizes = sorted(len(c) for c in nx.connected_components(G))`.
Exclude `sizes[-1]` (the spanning cluster if it exists). Then
`chi = np.mean([s**2 for s in sizes[:-1]]) / np.mean(sizes[:-1])`.
Average over realisations. Use `np.abs(p - 0.5)` for the x-axis and separate
the $p < p_c$ and $p > p_c$ branches.""",
    )

    _problem(
        "P4.2 — Interactive FSS collapse",
        """Reproduce the FSS collapse from P3.2 but add interactive sliders for
$p_c$, $\\nu$, and $\\beta$ in Streamlit. The user can drag each slider and watch
the collapse improve or degrade in real time. Start from the exact values and
perturb them — the collapse is visually very sensitive to $\\nu$.""",
        physics_hint="""This is exactly how exponents are measured in numerical
studies: vary the exponents until the collapse is optimal, typically quantified by
the variance of the rescaled data around a smoothed master curve. The sensitivity
of the collapse to $\\nu$ vs $\\beta$ reflects which exponent is harder to measure.""",
        code_hint="""Pre-compute the raw $P_\\infty(L,p)$ data once and store in
`st.session_state`. Then the collapse plot rerenders instantly from sliders without
re-running any simulation. Use `st.slider` with `min_value`, `max_value`, `step=0.01`.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter V — Growth as Fractal (DLA)")
    # -----------------------------------------------------------------------

    _problem(
        "P5.1 — DLA on Euclidean vs fractal substrate",
        """Grow 10 DLA clusters on a $50 \\times 50$ Euclidean grid ($p = 1$, all bonds present)
and 10 on the same grid at $p = 0.6$ (sparse percolation substrate).
Measure $d_f$ for each ensemble from the $R_g(M)$ slope. Plot the two mean $R_g(M)$
curves side by side and annotate both fitted slopes.""",
        physics_hint="""On the Euclidean substrate, diffusion is isotropic and
$d_f \\approx 1.71$ is the universal 2D DLA result. On the fractal substrate the
walker is forced to follow the available bonds, screening is geometry-dependent,
and the effective $d_f$ increases — the cluster fills more of the available fractal space.""",
        code_hint="""For the Euclidean case, pass $p = 1.0$ to `build_percolation` —
this gives a fully connected grid. Use `dla_ensemble` with the same parameters for
both runs. The fit window will differ between the two cases.""",
    )

    _problem(
        "P5.2 — Radial density profile",
        """For a single large DLA cluster (grow until $M > 300$ particles), compute
the radial density $\\rho(r)$ — the fraction of nodes at distance $r$ from the seed
that belong to the cluster. Plot $\\rho$ vs $r$ on a log-log scale. On a Euclidean
substrate, $\\rho(r) \\sim r^{d_f - 2}$ for $d_f \\approx 1.71$, giving a density
that decreases away from the centre.""",
        physics_hint="""A compact cluster would have $\\rho(r) \\approx \\text{const}$
up to its radius — all space is filled. DLA's $\\rho(r) \\to 0$ at large $r$ reflects
its tenuous, branching structure: most of the cluster mass is at small $r$ (the dense
centre), while the tips at large $r$ are sparse. This is the hallmark of fractality.""",
        code_hint="""Use the `pos` dict to compute Euclidean distances from the seed
for all cluster nodes. Bin nodes into annular rings of width $\\Delta r = 1$. For each
ring, `rho = n_cluster_in_ring / n_total_nodes_in_ring`. The denominator normalises
out the substrate geometry.""",
    )

    _problem(
        "P5.3 — Harmonic measure and screening",
        """Approximate the harmonic measure on a DLA cluster by releasing 2000 walkers
from a circle of radius $2R_g$ around the cluster and counting how often each boundary
node is hit first. Plot the distribution of hitting counts on a log-log scale.
Is the distribution a power law? Compare the dimension of the "active zone" (high hitting
probability) to the full cluster dimension $d_f$.""",
        physics_hint="""The harmonic measure is *multifractal*: tips have hitting
probability orders of magnitude larger than interior sites. The set of sites that
capture most of the probability has a dimension $d_1 < d_f$ (the information dimension).
This concentration on tips is precisely why DLA branches — growth is amplified where
the harmonic measure is largest.""",
        code_hint="""A walker released from a random point on a circle of radius
$2R_g$ performs a random walk until it hits a node adjacent to the cluster. Record which
boundary node is hit. After all walkers, you have a histogram over boundary nodes.
The hitting-count distribution is what you plot.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter VI — The Inverse Problem")
    # -----------------------------------------------------------------------

    _problem(
        "P6.1 — Minimal feature set",
        """Train the Chapter VI random forest with all 5 features, then with each single
feature alone, then with all pairs. Find the single feature that achieves the highest
accuracy on its own. Then find the pair closest in accuracy to the full 5-feature model.
Plot the 2D decision boundary for the best pair, overlaid with test-set points coloured
by true class.""",
        physics_hint="""The MSD log-log slope encodes $d_w$ — the direct dynamical
fingerprint of the geometry. The $\\pi$-entropy encodes degree heterogeneity.
Together these two features should nearly match the full model, because one captures
the walk dynamics and the other captures the static graph structure.""",
        code_hint="""Use `sklearn.inspection.DecisionBoundaryDisplay` for the 2D boundary
plot. For the single-feature sweep, iterate over `feature_names` and retrain with
`X[:, [i]]` each time. Store all accuracies in a dict and sort.""",
    )

    _problem(
        "P6.2 — Noise robustness",
        """Vary the measurement noise level $\\sigma \\in \\{0, 0.05, 0.1, 0.2, 0.4\\}$
added to the feature vector (see `SyntheticConfig`). For each $\\sigma$, train and test
the classifier 10 times (different seeds) and record mean accuracy ± std.
Plot accuracy vs $\\sigma$. Which class pair becomes confused first?""",
        physics_hint="""The carpet and percolation cluster at $p \\approx p_c$ are
the hardest to separate — both are fractal with similar $d_f$ and $d_w$. As noise
grows, the classifier loses the fine distinction between their walk statistics and
resorts to classifying them as the same. The Vicsek fractal, with its very different
$d_w \\approx 2.46$, is the last to be confused.""",
        code_hint="""Add a `noise_sigma` field to `SyntheticConfig` and apply
`X += rng.normal(0, noise_sigma, X.shape)` before training and *also* before testing
(test features are noisy too — don't add noise only at training time).""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter VII — The Sound of a Fractal")
    # -----------------------------------------------------------------------

    _problem(
        "P7.1 — Laplacian eigenspectrum",
        """Build the graph Laplacian $L = D - A$ for the Sierpiński carpet at level $k=4$
as a sparse matrix. Compute the 300 smallest eigenvalues using `scipy.sparse.linalg.eigsh`.
Plot the integrated density of states $N(\\omega)$ on a log-log scale — the slope is $d_s$.
Compare your measured $d_s$ to the prediction $2 d_f / d_w \\approx 1.314$.""",
        physics_hint="""The zero eigenvalue is always present (corresponding to the
constant eigenvector — the "zero mode"). Skip it and start from the smallest *nonzero*
eigenvalue. The density of states $g(\\omega) \\sim \\omega^{d_s - 1}$ means
$N(\\omega) \\sim \\omega^{d_s}$, so log-log slope of $N$ gives $d_s$ directly.""",
        code_hint="""Use `networkx.laplacian_matrix(G).astype(float)` to get a sparse matrix.
Then `vals, _ = scipy.sparse.linalg.eigsh(L, k=300, which='SM', tol=1e-6)`.
Sort `vals`, skip the near-zero one, and plot cumulative count vs eigenvalue on log-log axes.""",
    )

    _problem(
        "P7.2 — Three ways to measure $d_s$",
        """Measure the spectral dimension $d_s$ three independent ways on the same carpet graph:
(a) from the Laplacian eigenspectrum (P7.1),
(b) from the return probability $P(t) \\sim t^{-d_s/2}$ (dynamical measurement),
(c) by computing $d_f$ and $d_w$ separately and using $d_s = 2d_f/d_w$.
Report all three estimates and their uncertainties. They should agree to within $\\sim 5\\%$.""",
        physics_hint="""Method (c) accumulates errors from two separate fits, so it is
typically the least precise. Method (b) requires many walkers for a clean power-law tail.
Method (a) is the most direct but requires careful handling of the sparse eigenvalue problem.
Agreement between all three is a non-trivial consistency check of the Alexander-Orbach relation.""",
        code_hint="""For method (b), run 10,000 walkers for 5000 steps on the carpet.
At each $t$, compute the fraction at their origin. Fit log-log slope over $t \\in [100, 2000]$;
the slope is $-d_s/2$. For method (a), use the approach from P7.1.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter VIII — First-Passage Processes")
    # -----------------------------------------------------------------------

    _problem(
        "P8.1 — First-passage time distribution on a fractal",
        """Fix a target node $v^*$ and release 20,000 walkers from random starting
nodes on the Sierpiński carpet. Record the first time each walker hits $v^*$.
Plot the survival function $\\Pr(T_{\\rm fp} > t)$ on a log-log scale and fit
the power-law tail. The exponent should be $-d_s/2 \\approx -0.657$.
Repeat for a regular 2D grid and verify the exponent $-1/2$ (from $d_s = 2$, $P \\sim t^{-1/2}$).""",
        physics_hint="""The mean first-passage time $\\langle T_{\\rm fp}\\rangle$
is finite if the tail exponent is less than $-1$ (i.e., $d_s > 2$). For the carpet,
$d_s \\approx 1.31 < 2$, so the mean is formally infinite in the thermodynamic limit —
the walker takes arbitrarily long to find the target with non-negligible probability.
On a finite graph the mean is finite but grows strongly with graph size.""",
        code_hint="""Run each walker until it hits $v^*$ or exceeds a cutoff time $T_{\\rm max}$.
Walkers that never hit contribute to the right tail of the survival function — include them
as censored observations at $T_{\\rm max}$. Plot the empirical survival function
`1 - np.arange(1, n+1)/n` sorted by $T_{\\rm fp}$ values.""",
    )

    _problem(
        "P8.2 — Mean first-passage time vs graph size",
        """On a family of Sierpiński carpet graphs at levels $k = 2, 3, 4, 5$,
measure the mean first-passage time $\\langle T_{\\rm fp}\\rangle$ from a corner
node to the centre node (averaged over 5000 walkers). Plot $\\langle T_{\\rm fp}\\rangle$
vs $L = 3^k$ on a log-log scale. The slope should be $d_w \\approx 2.88$,
since the time to diffuse across a fractal of size $L$ scales as $L^{d_w}$.""",
        physics_hint="""This is the dynamical analogue of the static scaling $N \\sim L^{d_f}$.
The first-passage time to traverse a fractal of linear size $L$ scales as
$T \\sim L^{d_w}$, which is how $d_w$ enters transport phenomena: it is the
exponent of the "diffusion time" across a substrate of a given size.""",
        code_hint="""For each level $k$, run 5000 walkers from a fixed corner node
and record the first hit time at the central node. Take the mean over walkers that
do hit within $10 \\times L^{d_w}$ steps (use the theoretical $d_w$ as a guide for
the cutoff).""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter IX — 3D Percolation")
    # -----------------------------------------------------------------------

    _problem(
        "P9.1 — Locate $p_c$ on the cubic lattice",
        """Build bond percolation on a $20 \\times 20 \\times 20$ cubic grid using
`networkx.grid_graph(dim=[20,20,20])`. Sweep $p \\in [0.20, 0.30]$ and measure
the order parameter $P_\\infty$ (size of the largest component / total nodes).
Locate the crossing point of $P_\\infty$ curves for $L = 10, 15, 20$ and compare
to the exact value $p_c \\approx 0.2488$.""",
        physics_hint="""In 3D, the transition is sharper than in 2D at the same $L$
because the upper critical dimension is $d_c = 6$, and 3D is farther below $d_c$
than 2D — meaning critical fluctuations are larger and finite-size corrections are
stronger. You will need larger systems for the same precision as in 2D.""",
        code_hint="""Use `nx.grid_graph(dim=[L,L,L])` and randomly remove edges.
The graph has $L^3$ nodes. Keep $L \\leq 25$ for memory. Parallelise over
realisations using a list comprehension. The 3D Laplacian is already sparse
if you use `networkx.laplacian_matrix`.""",
    )

    _problem(
        "P9.2 — FSS collapse in 3D",
        """Perform the FSS collapse for 3D percolation using the 3D exponents
$\\nu \\approx 0.876$, $\\beta \\approx 0.418$, $p_c \\approx 0.2488$.
Compare the quality of the collapse to the 2D case: is it cleaner or noisier
at the same $L$? Why?""",
        physics_hint="""The 3D exponents are not exact — they are known only from
numerical simulations. When you perform the collapse you are effectively reproducing
the calculation that first determined these exponents. Any residual scatter in the
collapsed curve reflects both statistical noise and the uncertainty in the exponent values.""",
        code_hint="""Reuse the same `finite_size_collapse` function from `src.sim.criticality`
with updated exponent values. The collapse plot code is identical to Chapter IV — only
the constants change.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter X — The Ising Model")
    # -----------------------------------------------------------------------

    _problem(
        "P10.1 — Implement the Wolff algorithm",
        """Implement the Wolff single-cluster algorithm for the 2D Ising model on an
$L \\times L$ square lattice. Run sweeps at $T/T_c \\in \\{0.8, 0.9, 1.0, 1.1, 1.2\\}$
and plot the magnetisation $|M|$ and susceptibility $\\chi = L^2(\\langle M^2\\rangle - \\langle M\\rangle^2)$
vs $T/T_c$. The exact $T_c = 2J/\\ln(1+\\sqrt{2})$.""",
        physics_hint="""The Wolff algorithm builds a cluster by starting from a random spin,
then adding each same-spin neighbour with probability $p_{\\rm add} = 1 - e^{-2J/k_BT}$.
The entire cluster is then flipped. At $T_c$, the average cluster size is $\\sim \\xi^{d_f}$
where $d_f = 187/96$ is the fractal dimension of Ising clusters — so one Wolff step
flips a macroscopic fraction of the lattice, eliminating critical slowing down.""",
        code_hint="""Use a `deque` for the BFS cluster growth. Store spins as a 2D NumPy array
of $\\pm 1$. The acceptance probability for adding a neighbour is
`1 - np.exp(-2*J/T)` if the neighbour has the same spin, 0 otherwise.
Flip all spins in the cluster at the end of each step.""",
    )

    _problem(
        "P10.2 — Critical slowing down: Metropolis vs Wolff",
        """Measure the integrated autocorrelation time $\\tau_{\\rm int}$ of the magnetisation
for both Metropolis (single spin flip) and Wolff algorithms at $T = T_c$ for
$L \\in \\{16, 32, 64\\}$. Fit $\\tau_{\\rm int} \\sim L^z$ for each algorithm.
The Metropolis dynamic exponent is $z \\approx 2.17$; Wolff achieves $z < 0.5$.""",
        physics_hint="""Critical slowing down ($z > 0$) is unavoidable for local dynamics
because the correlation length $\\xi \\sim L$ at $T_c$, and information must propagate
across $\\xi$ lattice sites to decorrelate. Wolff sidesteps this by flipping entire
correlated clusters — each step changes the spin configuration on a length scale $\\sim \\xi$.""",
        code_hint="""Compute $\\tau_{\\rm int}$ using the Madras-Sokal windowed estimator:
sum the normalised autocorrelation function $\\hat{C}(t)/\\hat{C}(0)$ until the window
size equals $6\\tau_{\\rm int}$ (iterated to convergence). Run at least $10^4$ sweeps
after $10^3$ thermalisation sweeps.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter XI — Why Universality Exists (RG)")
    # -----------------------------------------------------------------------

    _problem(
        "P11.1 — Cobweb diagram and fixed points",
        """Implement the real-space RG map for bond percolation on the triangular lattice:
$p' = p^3 + 3p^2(1-p)$. Plot the cobweb diagram: starting from $p_0 = 0.4$, iterate
the map and draw the zigzag path between $y = x$ and $y = f(p)$.
Show that the orbit converges to $p^* = 0$ (disordered fixed point).
Repeat from $p_0 = 0.6$ — it should converge to $p^* = 1$ (ordered fixed point).
From $p_0 = 0.5 + \\varepsilon$ for tiny $\\varepsilon$, show it diverges from $p^* = 0.5$.""",
        physics_hint="""$p^* = 0.5$ is an *unstable* fixed point — any perturbation
drives the system away from it under RG. This is why the phase transition is sharp:
the RG flow separates the basin of attraction of $p^* = 0$ (ordered phase) from
$p^* = 1$ (disordered phase). The critical manifold is the single point $p_0 = 0.5$.""",
        code_hint="""The cobweb is drawn by alternating between vertical lines
(from $(p_n, p_n)$ to $(p_n, f(p_n))$) and horizontal lines
(from $(p_n, f(p_n))$ to $(f(p_n), f(p_n))$). Draw $y=x$ and $y=f(p)$ first,
then overlay the zigzag path.""",
    )

    _problem(
        "P11.2 — Correlation length exponent from linearisation",
        """Linearise the RG map $f(p) = p^3 + 3p^2(1-p)$ around $p^* = 0.5$.
The derivative $f'(p^*) = $ ?, and the correlation length exponent is
$\\nu = \\log b / \\log f'(p^*)$ where $b = 2$ is the spatial rescaling factor.
Compute $\\nu$ analytically and compare to the exact value $\\nu = 4/3$.""",
        physics_hint="""$f'(p) = 3p^2 + 6p(1-p) - 3p^2 = 6p(1-p)$.
At $p^* = 0.5$: $f'(0.5) = 6 \\times 0.25 = 1.5$. Therefore $\\nu = \\log 2 / \\log 1.5 \\approx 1.71$.
This is close to but not exactly $4/3 \\approx 1.33$ — the discrepancy is a known approximation
error of this particular RG scheme. The exact RG on the hierarchical lattice gives $\\nu = 4/3$ exactly.""",
        code_hint="""This problem is mostly analytical — implement it as a short calculation
with `np.log`. The lesson is that even an approximate RG gives the right *qualitative* picture
(unstable fixed point, exponential flow) with quantitative errors of order 20-30% in $\\nu$.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter XII — The Transfer Matrix")
    # -----------------------------------------------------------------------

    _problem(
        "P12.1 — Percolation on a strip via transfer matrix",
        """Build the transfer matrix $T$ for bond percolation on a strip of width $W = 6$
and compute the two largest eigenvalues $\\lambda_0 > \\lambda_1$ using
`scipy.sparse.linalg.eigs`. The finite-size correlation length is
$\\xi(W) = -W / \\log(\\lambda_1/\\lambda_0)$. Measure $\\xi(W)$ at $p = 0.5$ for
$W = 4, 5, 6, 7$ and verify $\\xi(W)/W \\to $ const as $W \\to \\infty$.""",
        physics_hint="""At $p_c$, the ratio $\\xi(W)/W$ converges to a universal constant
(Cardy's formula gives it in terms of the central charge $c = 0$ for percolation).
Away from $p_c$, $\\xi(W)$ saturates to the bulk correlation length. Plotting $\\xi(W)/W$
vs $p$ for multiple $W$ and looking for their crossing point gives another precise
estimator of $p_c$.""",
        code_hint="""States are subsets of $W$ bonds. Represent each state as an integer
bitmask $0 \\ldots 2^W - 1$. The transfer matrix entry $T[s', s]$ is the probability of
transitioning from column configuration $s$ to $s'$ when adding one new column.
Use `scipy.sparse.lil_matrix` to build $T$, then convert to CSR for the eigensolver.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter XIII — Continuous-Time Random Walks")
    # -----------------------------------------------------------------------

    _problem(
        "P13.1 — CTRW MSD matches fractal MSD",
        """Implement a CTRW on a regular 2D grid: at each site the walker waits a random
time drawn from $\\psi(t) \\sim t^{-(1+\\alpha)}$ with $\\alpha = 0.7$, then jumps to a
uniform random neighbour. Measure the MSD as a function of *physical* (clock) time and
show it scales as $\\langle r^2(t)\\rangle \\sim t^{0.7}$, matching a fractal walk with
$d_w = 2/0.7 \\approx 2.86$. This is the "same MSD, different mechanism" degeneracy.""",
        physics_hint="""The CTRW MSD exponent is $\\alpha$ (the waiting-time exponent),
while for fractal diffusion it is $2/d_w$. They can be equal for unrelated reasons.
The key distinction is in the displacement distribution: CTRW on a Euclidean lattice
has a non-Gaussian distribution at intermediate times (it is a Lévy stable distribution
in the diffusion limit), while fractal diffusion is Gaussian but on a different geometry.""",
        code_hint="""Generate power-law waiting times using `(np.random.pareto(alpha, n) + 1)`.
Maintain a priority queue of (next_jump_time, walker_id). At each event, advance
the walker and schedule the next jump. Plot MSD vs cumulative physical time
(sum of waiting times), not vs number of jumps.""",
    )

    _problem(
        "P13.2 — Non-Gaussian parameter distinguishes CTRW from fractal",
        """Compute the non-Gaussian parameter $\\alpha_2(t) = \\langle r^4\\rangle / (2\\langle r^2\\rangle^2) - 1$
for three walkers: (a) Brownian motion on a 2D grid, (b) CTRW with $\\alpha = 0.7$,
(c) random walk on the Sierpiński carpet. All three can be tuned to give approximately
the same MSD exponent. Plot $\\alpha_2(t)$ vs $t$ for all three.
Brownian motion gives $\\alpha_2 = 0$ identically; the others should differ.""",
        physics_hint="""$\\alpha_2 = 0$ is the hallmark of a Gaussian displacement distribution.
CTRW has $\\alpha_2 > 0$ at intermediate times, growing then decaying (the Lévy stable
limit is reached only at very long times). Fractal diffusion is subtler: the distribution
is determined by the fractal geometry, not a Lévy process, and $\\alpha_2$ converges
to a geometry-specific constant.""",
        code_hint="""Compute $\\langle r^4 \\rangle$ as `np.mean(np.sum(disp**2, axis=1)**2)`
where `disp` is the displacement array of shape `(n_walkers, 2)`. In 2D, the prefactor
in the denominator is 2 (not $d+2 = 4$) for the standard non-Gaussian parameter.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter XIV — Fractional Brownian Motion")
    # -----------------------------------------------------------------------

    _problem(
        "P14.1 — Generate fBm via the Davies-Harte method",
        """Implement fractional Brownian motion with Hurst exponent $H$ using the
circulant embedding (Davies-Harte) method:
1. Build the autocovariance vector $\\gamma_k = \\tfrac{1}{2}(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})$.
2. Embed in a circulant matrix and take the FFT.
3. Generate the fBm increment sequence.
Generate trajectories for $H \\in \\{0.3, 0.5, 0.7\\}$ and verify that the MSD
$\\langle r^2(t)\\rangle \\sim t^{2H}$.""",
        physics_hint="""$H = 0.5$ is standard Brownian motion (independent increments).
$H < 0.5$ gives anti-persistent increments — after a step right, the next step is
more likely to go left. $H > 0.5$ gives persistent increments — trending motion.
Both sub- and super-diffusion arise from the *temporal correlations* in the increments,
not from spatial geometry — this is what distinguishes fBm from fractal diffusion.""",
        code_hint="""Use `np.fft.rfft` for the forward transform and `np.fft.irfft` for the
inverse. The circulant eigenvalues are the FFT of the first row of the circulant — take
their square root and multiply by complex Gaussians. The real part of the inverse FFT
gives one fBm sample path. This is $O(N \\log N)$, not $O(N^2)$.""",
    )

    _problem(
        "P14.2 — Ergodicity breaking in CTRW but not fBm",
        """For CTRW and fBm both tuned to give $\\langle r^2(t)\\rangle \\sim t^{0.7}$,
compare the *ensemble-averaged* MSD to the *time-averaged* MSD
$\\overline{\\delta^2}(\\tau) = \\frac{1}{T-\\tau}\\int_0^{T-\\tau}[x(t+\\tau)-x(t)]^2 dt$.
For fBm they should agree (ergodic). For CTRW they should not — the time average
depends on the total trajectory length $T$ even as $T \\to \\infty$.""",
        physics_hint="""Ergodicity breaking is the signature of a non-stationary
process. CTRW is non-ergodic because the waiting times are heavy-tailed — a single
very long wait dominates the time average for any finite $T$. fBm is ergodic because
its increments, while correlated, are stationary. This distinction is experimentally
measurable from single-particle trajectories.""",
        code_hint="""Compute the time-averaged MSD for each trajectory individually,
then average over trajectories. If CTRW is ergodic, all trajectories should give the same
time-averaged MSD. Instead, you will see large trajectory-to-trajectory fluctuations —
the hallmark of weak ergodicity breaking.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter XV — Distinguishing Anomalous Diffusion")
    # -----------------------------------------------------------------------

    _problem(
        "P15.1 — Three mechanisms, same MSD, different fingerprints",
        """Generate 300 trajectories from each of: (a) fractal diffusion on the carpet,
(b) CTRW with matched $\\alpha$, (c) fBm with matched $H$. All tuned to give the same
MSD exponent $\\approx 0.70$. Extract four features per trajectory: MSD slope, non-Gaussian
parameter at $t = T/2$, ergodicity ratio $\\overline{\\delta^2}/\\langle r^2\\rangle$,
and kurtosis of the displacement distribution at $t = T/4$.
Train a random forest classifier and report accuracy. Which features matter most?""",
        physics_hint="""MSD slope alone gives $\\sim 33\\%$ accuracy (chance).
Adding the non-Gaussian parameter should push accuracy to $\\sim 70\\%$ (separates fBm from the other two).
Adding the ergodicity ratio separates CTRW from fractal diffusion.
The full feature set should achieve $> 90\\%$. This mirrors the real-world ANDI challenge.""",
        code_hint="""Compute the non-Gaussian parameter at a single time point
(not as a function of $t$) to give a scalar feature. The ergodicity ratio is
`np.mean(time_avg_msd) / ensemble_avg_msd` at a fixed lag $\\tau = T/4`.
Use `sklearn.ensemble.RandomForestClassifier` with `n_estimators=200`.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter XVI — Regression: Learning $d_f$ and $d_w$")
    # -----------------------------------------------------------------------

    _problem(
        "P16.1 — Predict $d_w$ from walk statistics",
        """Generate 300 percolation graphs with $p$ drawn uniformly from $[0.52, 0.95]$.
For each, measure $d_w$ from the MSD slope (ground truth), and extract the same five
features as Chapter VI. Train a `RandomForestRegressor` and a `GradientBoostingRegressor`.
Report $R^2$ and mean absolute error for $d_w$. Plot predicted vs true $d_w$ — the
diagonal is a perfect predictor.""",
        physics_hint="""$d_w$ decreases monotonically from $\\approx 2.87$ (near $p_c$,
fractal substrate) toward $2$ (at $p = 1$, regular grid). The MSD log-log slope feature
directly encodes $2/d_w$, so the regressor has essentially the right feature handed to
it. The interesting question is whether the other features add predictive power.""",
        code_hint="""Use `sklearn.model_selection.cross_val_score` with 5-fold CV to
avoid overfitting. For the scatter plot, colour points by $p$ to show that errors are
not randomly distributed — they will be largest near $p_c$ where the fractal behaviour
is most variable across realisations.""",
    )

    _problem(
        "P16.2 — Predict $d_f$ from walk statistics",
        """Repeat P16.1 but predict $d_f$ (fractal dimension of the largest component).
Measure $d_f$ by fitting node count vs strip width for the largest component.
Compare $R^2$ for $d_f$ and $d_w$: which is easier to predict from walk statistics,
and why does this make physical sense?""",
        physics_hint="""$d_w$ should be easier to predict because it is more directly
encoded in the walk dynamics (the MSD slope). $d_f$ is a static geometric property and
the walker only sees it indirectly — through the number of distinct nodes visited,
which is related to $d_f/d_w$ (the spectral dimension). Predicting $d_f$ from dynamics
is a harder inverse problem.""",
        code_hint="""Measure $d_f$ from the scaling $N_{\\rm LCC} \\sim L^{d_f}$ by varying
$L$ for the same $p$ value. Alternatively, use the node count of the largest connected
component vs $L$ across your grid sweep.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter XVII — Compact Growth: The Eden Model")
    # -----------------------------------------------------------------------

    _problem(
        "P17.1 — Eden vs DLA: fractal dimension comparison",
        """Implement the Eden growth model: at each step, pick a random perimeter node
(adjacent to the cluster but not in it) and add it to the cluster.
Grow 10 Eden clusters and 10 DLA clusters on the same percolation substrate ($p = 0.6$,
$L = 50$). Measure $d_f$ from $R_g(M)$ for each ensemble. Plot both mean $R_g(M)$ curves
on the same log-log axes and annotate the fitted slopes.""",
        physics_hint="""Eden growth is compact: every perimeter site is equally likely
to grow, so the cluster fills in convexly. DLA growth is fractal: only tips receive
incoming walkers (harmonic measure concentration), leading to branching.
The Eden cluster has $d_f = 2$ (fills space); the DLA cluster has $d_f \\approx 1.71$.""",
        code_hint="""Maintain a set `perimeter` of nodes adjacent to the cluster.
At each step: pick a random node from `perimeter`, add it to the cluster,
update `perimeter` by adding its neighbours (if not already in the cluster).
This is $O(1)$ per step using a set — much faster than the walker-based DLA.""",
    )

    _problem(
        "P17.2 — Interface roughness and KPZ",
        """For a large Eden cluster grown on a 2D Euclidean grid, measure the surface
roughness $W(L, t) = \\sqrt{\\langle h^2 \\rangle - \\langle h \\rangle^2}$, where $h(x, t)$
is the cluster height profile along one edge. The KPZ scaling predicts
$W \\sim t^\\beta$ with $\\beta = 1/3$ for short times, and $W \\sim L^\\alpha$ with
$\\alpha = 1/2$ at long times (Family-Vicsek scaling).""",
        physics_hint="""The KPZ universality class describes interface growth with noise
and a nonlinear "slope" term. The Eden model belongs to it because the growth rate is
proportional to the local curvature (sites on convex tips grow faster). The exponents
$\\alpha = 1/2$, $\\beta = 1/3$ are exact in 1D (the interface is a 1D object bounding a 2D cluster).""",
        code_hint="""Use a rectangular grid, seed a flat line along the bottom,
and grow upward. At each time, record $h(x)$ = height of the topmost cluster node
in each column $x$. Compute $W$ over the full width and plot vs $t$.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter XVIII — Directed Percolation")
    # -----------------------------------------------------------------------

    _problem(
        "P18.1 — Order parameter and $p_c$ for directed bond percolation",
        """Build directed bond percolation on a $100 \\times 100$ square lattice with
bonds pointing right and down only. Seed a cluster from the entire top row and measure
the survival probability $P_{\\rm surv}(p)$ — the fraction of realisations where
the cluster reaches the bottom row. Locate $p_c$ numerically. The exact value is
$p_c^{\\rm DP} \\approx 0.6447$ for the square lattice directed diagonally.""",
        physics_hint="""Directed percolation has *two* correlation lengths:
$\\xi_\\perp \\sim |p-p_c|^{-\\nu_\\perp}$ (across the preferred direction) and
$\\xi_\\parallel \\sim |p-p_c|^{-\\nu_\\parallel}$ (along the preferred direction).
Because $\\nu_\\parallel / \\nu_\\parallel \\neq 1$, the FSS collapse requires a 2D
scaling function $f(x_\\perp, x_\\parallel)$ rather than the 1D $f(x)$ of isotropic percolation.""",
        code_hint="""Represent each row as a set of "active" sites. For each site in
row $i$, propagate to site $(i+1, j)$ with probability $p$ for each of the two
downward bonds. This is purely row-by-row — no need for the full NetworkX graph.
Store only two rows at a time for memory efficiency.""",
    )

    _problem(
        "P18.2 — DP universality and epidemic spreading",
        """Simulate a simple SIR epidemic on a 2D lattice: each infected site infects
each susceptible neighbour independently with probability $\\lambda$, and recovers
(becomes permanently immune) after one step. This is exactly directed bond percolation
in disguise. Find $\\lambda_c$ numerically and compare to the directed percolation threshold
$p_c^{\\rm DP}$. They should agree.""",
        physics_hint="""The mapping is: infected site = active DP site, susceptible = empty,
recovered = permanently blocked. The epidemic threshold $\\lambda_c$ is the DP threshold
$p_c$. This is why DP is the universality class of epidemics, forest fires, and
interface depinning — they are all equivalent to directed percolation at the transition.""",
        code_hint="""Use a NumPy array with values 0 (susceptible), 1 (infected), 2 (recovered).
At each time step, for each infected site, independently infect each susceptible neighbour
with probability $\\lambda$, then set all infected sites to recovered.
Count total infected at each step; plot vs time for several $\\lambda$ values.""",
    )

    # -----------------------------------------------------------------------
    st.markdown("---\n### Chapter XIX — When One Dimension Is Not Enough")
    # -----------------------------------------------------------------------

    _problem(
        "P19.1 — Harmonic measure hitting statistics",
        """Grow a large DLA cluster ($M > 500$ particles) on a percolation substrate.
Release 5000 walkers from a circle of radius $2R_g$ and record which boundary node
each walker hits first. Rank boundary nodes by their hitting count. Plot the cumulative
hitting fraction vs rank on a log-log scale. A simple fractal would give a flat
distribution; the multifractal DLA shows a strongly non-uniform distribution.""",
        physics_hint="""The harmonic measure is so concentrated on tips that the top
$5\\%$ of boundary sites capture $> 80\\%$ of the probability. This is the Zipf-like
behaviour of the harmonic measure — a power-law rank-frequency distribution.
The exponent of this distribution is related to the multifractal spectrum $f(\\alpha)$.""",
        code_hint="""Use the existing DLA growth code to get the cluster. For the
boundary, find all cluster-adjacent nodes that are not in the cluster.
Walkers are released from uniformly random points on a circle; use the `pos` dict
to find the nearest node on the circle as the starting position.""",
    )

    _problem(
        "P19.2 — Generalised dimensions $D_q$",
        """From the hitting statistics in P19.1, compute the generalised dimensions
$D_q$ for $q \\in \\{-2, -1, 0, 1, 2, 3\\}$ using the box-counting moment method:
partition the boundary into boxes of size $\\varepsilon$, compute $\\sum_i p_i^q$ for
each $\\varepsilon$, and fit the slope vs $\\log \\varepsilon$.
$D_0$ is the box dimension of the boundary; $D_1$ is the information dimension;
$D_2$ is the correlation dimension. Show that $D_0 > D_1 > D_2$ (the hallmark of multifractality).""",
        physics_hint="""For a simple (mono)fractal, $D_q = d_f$ for all $q$.
For a multifractal, $D_q$ decreases strictly with $q$ — moments with large $q$
are dominated by the highest-probability sites (the tips), while $q < 0$ amplifies
the lowest-probability sites (the screened interior). The width of the $D_q$ spectrum
quantifies the degree of multifractality.""",
        code_hint="""With only 5000 walkers, statistics for $q < -1$ will be noisy
because they weight rare events. Use 20,000 walkers if possible, or restrict to
$q \\geq 0$. For each $q$, fit `np.polyfit(np.log(eps_arr), np.log(moment_arr), 1)`
where `moment_arr[i] = sum(p**q for p in box_probabilities[i])`.""",
    )

    st.markdown("---")
    prev_next("problems")
