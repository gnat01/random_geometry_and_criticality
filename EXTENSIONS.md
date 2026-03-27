# Extensions — ideas for the codebase and the book

Roughly ordered from "natural next step" to "ambitious research project".
Each entry notes what new code it needs and whether it warrants a new book chapter.

---

## Part I — Going deeper into existing physics

### 1. Spectral dimension from the Laplacian eigenvalue spectrum

**The physics.** The Laplacian matrix $L = D - A$ of a graph encodes its vibrational modes.
The density of states $g(\omega) \sim \omega^{d_s - 1}$ for small $\omega$, where
$d_s = 2 d_f / d_w$ is the spectral dimension. Alexander and Orbach (1982) conjectured
that $d_s = 4/3$ for percolation clusters in all dimensions — a conjecture that is
approximately (but not exactly) true.

**What to build.** Compute the full eigenspectrum of $L$ using `scipy.linalg.eigh` for
small graphs, or `scipy.sparse.linalg.eigsh` for the lowest 200 eigenvalues of large ones.
Plot the integrated density of states $N(\omega) = \int_0^\omega g(\omega') d\omega'$ on
a log-log scale; the slope is $d_s$. Compare across carpet, Vicsek, and critical percolation.

**Why it's interesting.** This connects the random walk (which gave $d_w$ dynamically) to the
full vibrational spectrum of the network — the physics of phonons in disordered solids.
The carpet walk dimension $d_w \approx 2.88$ predicts $d_s \approx 1.314$; you can verify
this directly from the eigenvalues without running any walkers at all.

**Difficulty.** Medium. The main cost is memory: the Laplacian of an $L = 3^5$ carpet has
$8^5 = 32768$ nodes. Use sparse matrices throughout.

**Book chapter.** Yes — "Chapter VII: The Sound of a Fractal". Could open with the physics
of amorphous solids and fractons, then derive $d_s$, then show the eigenspectrum.

---

### 2. Return probability $P(t)$ and spectral dimension from dynamics

**The physics.** The probability that a walker is at its starting node at time $t$ scales as
$P(t) \sim t^{-d_s/2}$. This is a purely dynamical measurement of $d_s$ and requires no
eigenvalue decomposition.

**What to build.** Measure the fraction of walkers that are at their origin at each time $t$.
This is a one-liner on top of the existing walk infrastructure. Fit the log-log slope.

**Why it's interesting.** It gives a second independent measurement of $d_s$ (alongside the
eigenspectrum approach above) and lets you directly verify the Alexander-Orbach relation
$d_s = 2 d_f / d_w$ using three separately measured quantities. The three numbers should
all be consistent — seeing this close the loop is deeply satisfying.

**Difficulty.** Easy — two hours of work on top of existing walker code.

**Book.** Fold into an extended Chapter II or into the Spectral chapter above.

---

### 3. First-passage time distributions

**The physics.** On a Euclidean lattice, the first-passage time to a target decays as
$P(T_{\rm fp} = t) \sim t^{-3/2}$ for large $t$. On a fractal, the exponent becomes
$-(1 + d_s/2)$, reflecting the slower exploration. On a critical percolation cluster the
tail is even heavier, leading to infinite-mean first-passage times in the thermodynamic limit.

**What to build.** Fix a target node, release walkers from random starting points, record
$T_{\rm fp}$, and plot the survival function $\Pr(T_{\rm fp} > t)$. Fit the power-law tail.

**Why it's interesting.** First-passage times matter in chemistry (reaction rates), biology
(gene search on DNA), and finance (barrier crossing). Showing that the substrate geometry
changes the tail exponent gives these abstract numbers physical meaning.

**Difficulty.** Easy to medium (the tail is noisy; needs $> 10^4$ walkers for a clean fit).

**Book chapter.** Definitely needs a new chapter called "Basic first-passage processes"
---

### 4. Multifractal analysis of the DLA harmonic measure

**The physics.** The harmonic measure on a DLA cluster — the probability distribution that
the next particle sticks at each boundary site — is not a simple fractal. Different regions
of the boundary have wildly different local densities. The correct description is a
*multifractal spectrum* $f(\alpha)$: a curve that encodes how the set of points with local
scaling exponent $\alpha$ has Hausdorff dimension $f(\alpha)$.

**What to build.** After growing a DLA cluster, approximate the harmonic measure by releasing
many random walkers and counting first-hit frequencies per boundary site. Compute the
generalised dimensions $D_q$ using the box-counting moment method:
$\sum_i p_i^q \sim \varepsilon^{(q-1)D_q}$. Plot $D_q$ vs $q$ and the Legendre-transform
spectrum $f(\alpha)$ via $\alpha = d D_q/dq$, $f = q\alpha - (q-1)D_q$.

**Why it's interesting.** Multifractals are the correct description of turbulence, financial
volatility, and the spatial distribution of rainfall — not just DLA. This chapter would
introduce the idea that a single $d_f$ is not always enough.

**Difficulty.** Hard. Clean multifractal spectra require very large DLA clusters
($> 10^5$ particles) and careful moment estimation. Consider as a long-term project.

**Book chapter.** Yes — "Chapter VIII: When One Dimension Is Not Enough".

---

### 5. Eden model — compact growth, contrast with DLA

**The physics.** In the Eden model, new particles attach uniformly at random to any
perimeter site (no diffusion bias). The result is a compact, roughly circular cluster
with a rough fractal *interface* — but the cluster itself has $d_f = 2$ (it fills space).
The interface roughness belongs to the KPZ universality class.

**What to build.** Implement Eden growth on the percolation substrate alongside the
existing DLA code. Compare the $R_g(M)$ scaling curves side by side. DLA: $d_f \approx 1.71$,
branching morphology. Eden: $d_f = 2$, compact blob with jagged edge.

**Why it's interesting.** It answers the question "is fractal growth inevitable?" — No.
DLA's fractality comes specifically from the diffusion field and harmonic measure concentration
on tips. Eden shows that without that bias you get compact growth.

**Difficulty.** Easy — Eden is simpler to implement than DLA.

**Book.** Add as a contrast section inside Chapter V ("Growth as Fractal"), or a short Chapter V½.

---

## Part II — New physics models

### 6. Directed percolation — a different universality class

**The physics.** In directed percolation, bonds are open only in one preferred direction
(think: fluid flowing downhill). This breaks the up/down symmetry and puts the model in
a completely different universality class from isotropic percolation. The exponents are
$\nu_\perp \approx 0.733$, $\nu_\parallel \approx 1.295$, $\beta \approx 0.277$ in 2D.
DP is the universality class of epidemic spreading, forest fires, and interface depinning.

**What to build.** Build a directed bond percolation graph on a square lattice (bonds
pointing right and down only). Measure the order parameter (survival probability of a
cluster seeded at the top row), the cluster-size distribution, and the FSS collapse.
Compare exponents to isotropic percolation.

**Why it's interesting.** It shows that "universality class" is not just an abstraction —
measurably different symmetries produce measurably different exponents. The FSS collapse
now has *two* correlation lengths ($\xi_\perp$ and $\xi_\parallel$), making it richer.

**Difficulty.** Medium. Main complication: the order parameter definition and geometry differ
from isotropic percolation.

**Book chapter.** Yes — "Chapter IX: Another Universality". Could contrast the two FSS
collapses side by side.

---

### 7. Ising model at criticality

**The physics.** The 2D Ising model has an exact critical point at
$T_c = 2J / \ln(1 + \sqrt{2}) \approx 2.269 J/k_B$ (Onsager, 1944). At $T_c$, spin
clusters have a fractal structure with $d_f = 187/96 \approx 1.948$ (the same exponents
appear as in percolation, but different — they belong to a different universality class).
The Wolff cluster algorithm makes this tractable: it flips whole correlated clusters rather
than single spins, eliminating critical slowing down.

**What to build.** Implement the Wolff algorithm on a square lattice. Measure the magnetisation
$M$, susceptibility $\chi$, and specific heat $C$ as functions of $T$. Perform FSS collapse
using $\nu = 1$, $\beta = 1/8$, $\gamma = 7/4$. Visualise the spin clusters at $T_c$ —
they look qualitatively similar to percolation clusters but with different exponents.

**Why it's interesting.** The Ising model is the paradigmatic critical system. Being able to
show it alongside percolation, with its own FSS collapse and its own universality class, makes
the abstract idea of universality concrete.

**Difficulty.** Medium-hard (Wolff algorithm is not trivial; critical slowing-down without it
makes single-spin-flip Metropolis unusable near $T_c$ on large grids).

**Book chapter.** Yes — this could be the opening of a "Chapter IX: Other Critical Points"
that surveys Ising, directed percolation, and KPZ.

---

### 8. Continuous-time random walks (CTRW) and trap models

**The physics.** Anomalous diffusion on a fractal is not the only mechanism for subdiffusion.
In a CTRW, the walker waits a random time at each site before jumping, with the waiting time
drawn from a power-law distribution $\psi(t) \sim t^{-(1+\alpha)}$, $0 < \alpha < 1$. The
MSD still scales as $\langle r^2 \rangle \sim t^\alpha$, *identical* to fractal diffusion.
Yet the two mechanisms are physically distinct and distinguishable by higher-order statistics.

**What to build.** Implement a CTRW on a regular 2D grid with power-law waiting times.
Add it to the ML inverse problem in Chapter VI as a fourth class. The classifier should
struggle to separate CTRW from fractal diffusion using only the MSD slope — but should
succeed when given the non-Gaussian parameter $\alpha_2(t) = \langle r^4 \rangle / (d+2)\langle r^2 \rangle^2 - 1$
as an additional feature.

**Why it's interesting.** This is precisely the question that matters in single-particle
tracking experiments in cells: is the anomalous diffusion geometric (crowded fractal-like
environment) or temporal (binding/unbinding traps)? The classifier that fails to distinguish
them from MSD alone — but succeeds with the non-Gaussian parameter — makes the point
viscerally.

**Difficulty.** Medium. Power-law random variates are easy (`scipy.stats.pareto`); the
non-Gaussian parameter is a standard formula.

**Book.** Extend Chapter VI ("The Inverse Problem") with this as a second experiment, or add
a new "Chapter VI½: Mechanisms of Anomalous Diffusion".

Comment : This deserves its own chapter "Advanced topics : Continuous time RWs"
---

### 9. Fractional Brownian motion (fBm)

**The physics.** fBm is a Gaussian process with stationary increments and Hurst exponent $H$:
$\langle r^2(t) \rangle \sim t^{2H}$. For $H < 1/2$, increments are anti-correlated
(subdiffusion); for $H > 1/2$, they are correlated (superdiffusion). Unlike fractal
diffusion and CTRW, fBm is time-reversible and has a Gaussian displacement distribution —
so it is distinguishable by higher-order statistics even when the MSD exponents match.

**What to build.** Implement the Hosking (or Davies-Harte FFT) method for generating fBm
trajectories. Add to the CTRW/fractal classifier in Chapter VI. This makes three mechanisms
with the same MSD exponent but different distributions.

**Difficulty.** Medium (FFT-based fBm generation is elegant but needs care).

**Book chapter.** : Needs its own chapter called "Fractional Brownian motion"

---

### 10. 3D percolation

**The physics.** Bond percolation on the cubic lattice has $p_c \approx 0.2488$ and
exponents $\nu \approx 0.876$, $\beta \approx 0.418$ — measurably different from 2D.
The 3D case is harder to visualise but easier to study statistically because finite-size
corrections are smaller.

**What to build.** Extend `build_percolation` to 3D using a cubic graph (`networkx.grid_graph`
with $\dim = [L, L, L]$). Visualise with a 3D scatter plot coloured by cluster membership
(Plotly or Matplotlib 3D). Perform the same FSS collapse as Chapter IV.

**Why it's interesting.** The 3D exponents do not have exact analytical values (unlike 2D);
they are known only numerically. So you are doing the same measurement physicists actually
did to find them.

**Difficulty.** Medium. Memory cost is $O(L^3)$; keep $L \leq 30$ for interactive use.

**Book.** Add as an extended section in Chapter III or IV: "What changes in 3D?"
Comment : NEW chapter please; this is a different beast altogether and deserves its own highlighting

---

### 11. Quantum walks

**The physics.** Replace the classical random walk with a quantum mechanical one: a unitary
evolution $|\psi(t+1)\rangle = U |\psi(t)\rangle$ where $U$ encodes the coin + shift
operators. On a 1D line, quantum walkers spread *ballistically* ($\langle r^2 \rangle \sim t^2$,
$d_w = 1$) instead of diffusively. On a fractal, the spreading interpolates between
ballistic and subdiffusive depending on the coin and substrate.

**What to build.** Implement discrete quantum walk on the carpet graph using sparse matrix
exponentiation (`scipy.sparse.linalg.expm_multiply`). Measure $\langle r^2 \rangle (t)$ by
taking the expectation value of position squared under the probability distribution
$p(v, t) = |\psi_v(t)|^2$.

**Why it's interesting.** Quantum walks are the basis of quantum search algorithms (Grover's
algorithm can be framed as a quantum walk). The contrast with classical walks on the same
substrate makes the quantum speedup concrete.

**Difficulty.** Hard. Complex amplitudes, coin choice, graph encoding all add complexity.
Good as a final "stretch" chapter.

Comment : SKIP THIS FOR NOW ENTIRELY

---

## Part III — Methods and algorithms

### 12. Real-space renormalisation group (RG)

**The physics.** The RG explains *why* universality exists. For percolation on a triangular
lattice, Kadanoff's real-space RG gives an exact recursion $p' = p^3 + 3p^2(1-p)$ for the
renormalised bond probability after coarse-graining by a factor of 2. The fixed points of
this map are at $p^* = 0$ (all open), $p^* = 1$ (all closed), and $p^* = 1/2$ (critical).
Linearising around $p^* = 1/2$ gives $\nu$ exactly.

**What to build.** Implement the RG map as a function. Plot the flow diagram (cobweb plot
of iterated applications). Show that trajectories starting near $p^* = 1/2$ diverge away
from it (unstable fixed point = phase transition). Compute $\nu = \log 2 / \log |f'(p^*)|$.

**Why it's interesting.** This is perhaps the most important idea in all of theoretical
physics since 1970 — and it can be understood with a single Python function and a cobweb
plot. It also directly explains the FSS collapse: the data collapse works because all $L$
curves are related by successive RG transformations.

**Difficulty.** Easy to medium for the triangular lattice case; harder for the square lattice
(which requires an approximation scheme).

**Book chapter.** Yes — "Chapter X: Why Universality Exists". This could be the capstone.

---

### 13. Transfer matrix method

**The physics.** For a strip of width $W$ and length $L \gg W$, the partition function
of a statistical mechanics model can be computed exactly as $Z = \text{tr}(T^L)$ where
$T$ is a $2^W \times 2^W$ transfer matrix. The ratio of the two largest eigenvalues of $T$
gives the finite-size correlation length $\xi(W)$, which scales as $\xi(W) \sim W$ at $p_c$.

**What to build.** Build the transfer matrix for bond percolation on an $L \times W$ strip
(bonds along the strip direction). Compute the top two eigenvalues using
`scipy.sparse.linalg.eigs`. Extract $\xi(W)$ and verify FSS.

**Why it's interesting.** It's an exact calculation with no Monte Carlo noise, reaching
system sizes inaccessible to simulation. It also shows that the eigenvalue structure of
the transfer matrix encodes the critical exponents.

**Difficulty.** Hard. The transfer matrix grows exponentially in $W$; needs sparse storage
and careful state enumeration. Worth it for the exactness.

**Book chapter.** : Absolutely. This is a superb technique that deserves highlighting

---

### 14. Wolff cluster algorithm

**The physics.** Standard Metropolis single-spin-flip dynamics is exponentially slow near
$T_c$ (critical slowing down, $\tau \sim \xi^z$ with $z \approx 2$). The Wolff algorithm
flips entire correlated clusters in one step, reducing the dynamic exponent to $z < 0.5$.
This makes equilibrium sampling near $T_c$ tractable.

**What to build.** Implement Wolff for the Ising model. Compare autocorrelation time
$\tau$ for Metropolis vs Wolff as a function of $L$ near $T_c$. Show the scaling
$\tau \sim L^z$ with dramatically different $z$ values.

**Difficulty.** Medium (needed for the Ising chapter above).

**Book chapter.** Yes ; profound technique that deserves highlighting

---

## Part IV — ML / data science extensions

### 15. Regression for $d_f$ and $d_w$

**The physics.** Chapter VI classifies geometry *family* (carpet vs Vicsek vs percolation).
A more ambitious target: given only walk statistics from a percolation cluster at unknown $p$,
predict the numerical values of $d_f$ and $d_w$ directly. These vary continuously with $p$
(above $p_c$), so it is a regression problem.

**What to build.** Generate 300 percolation graphs with $p \in [0.52, 0.95]$ and measure
$d_f$ (from cluster node count scaling) and $d_w$ (from MSD slope) as ground truth targets.
Train `RandomForestRegressor` and `GradientBoostingRegressor` on the five walk features.
Report $R^2$ for each target and plot predicted vs true.

**Difficulty.** Easy — all infrastructure exists.

**Book chapter.** : Yes. Easy but fun and insightful. 

---

### 16. Distinguishing anomalous diffusion mechanisms

**The physics.** CTRW, fBm, and fractal diffusion all produce $\langle r^2 \rangle \sim t^\alpha$
with matching $\alpha$, but differ in:
- displacement distribution (Gaussian for fBm; non-Gaussian for CTRW and fractal)
- time-averaged MSD vs ensemble-averaged MSD (different for CTRW, identical for ergodic processes)
- the non-Gaussian parameter $\alpha_2(t)$
- the velocity autocorrelation function

**What to build.** Add these features to the Chapter VI feature vector. Train on synthetic
data from all three mechanisms at matched MSD exponent. Show that MSD slope alone gives
chance accuracy; non-Gaussian parameter + ergodicity measure gets to > 90%.

**Difficulty.** Medium-hard (requires implementing CTRW and fBm first).

**Book.** Major extension to Chapter VI, or its own chapter.

COMMENT : New chapter please, this is a big extension to Chapter 6 after all

---

### 17. Graph neural network for geometry classification

**The physics.** The Chapter VI classifier uses hand-crafted physics features. A GNN
reads the raw graph structure and learns its own features. Does it learn something
equivalent to $d_w$, or something genuinely different?

**What to build.** Use PyTorch Geometric or DGL to build a simple message-passing GNN
that takes a graph as input and outputs a class label. Compare accuracy and feature
attribution (via GNNExplainer) to the random forest. Does the GNN discover the same
physics features?

**Difficulty.** Hard — requires PyTorch / DGL dependency.

COMMENT : SKIP FOR NOW

---

### 18. The ANDI challenge — real single-particle tracking data

**The context.** The Anomalous Diffusion (ANDI) challenge (2020, 2022) released labelled
single-particle trajectories from simulations of five anomalous diffusion models. The task
is to identify the model and estimate the exponent $\alpha$ from short, noisy trajectories.
State-of-the-art methods use recurrent neural networks; a physics-feature random forest
gets competitive performance.

**What to build.** Add ANDI data loading to the ML module. Extract the same features used
in Chapter VI from individual trajectories (not ensemble averages). Benchmark the random
forest against the ANDI leaderboard.

**Why it's interesting.** This is a real open benchmark. You can compare directly to
published work. It grounds the entire book in an active experimental challenge.

**Difficulty.** Hard (short trajectories make feature estimation noisy; need 1D walk features
rather than graph features)

COMMENT : SKIP FOR NOW
.

---

## Part V — Visualisation and interaction

### 19. Plotly / interactive 3D visualisation

Replace the static Matplotlib 3D plots (used sparingly) with interactive Plotly figures.
Specifically: 3D percolation cluster coloured by connected component, rotatable DLA cluster,
interactive FSS collapse where the user drags $\nu$ and $p_c$ sliders and watches the
collapse improve or degrade in real time.

The FSS interactive collapse would be the killer demo: seeing the curves snap into alignment
as you tune the exponent toward its exact value is genuinely striking.

**Difficulty.** Easy to medium (Plotly integrates with Streamlit via `st.plotly_chart`).

---

### 20. Animated random walk

Show a single walker taking steps in real time on the fractal graph using Streamlit's
`st.empty()` + `time.sleep()` animation loop. Colour visited nodes by recency (heat trail).
This is pedagogically powerful for Chapter II — watching the walker get trapped in
dead ends and backtrack makes subdiffusion visceral.

**Difficulty.** Easy.

---

### 21. Fractal zoom

For the Sierpiński carpet and Vicsek fractal, implement an interactive zoom that lets the
user scroll into the fractal at increasing iteration depth, rendered as a bitmap (not a
graph). Use NumPy array operations to build the bitmap at arbitrary depth without storing
all iterations. This demonstrates self-similarity more directly than any plot.

**Difficulty.** Easy (recursive NumPy Kronecker products).

---

## Part VI — Agreed build plan

### What we are skipping (for now)

| Item | Reason |
|---|---|
| 11 — Quantum walks | Heavy infrastructure, different physics domain |
| 17 — Graph neural network | PyTorch/DGL dependency, diminishing returns vs RF |
| 18 — ANDI challenge | Requires 1D trajectory features, separate data pipeline |

---

### Confirmed new chapters

| Chapter | Title | Key physics | Items |
|---|---|---|---|
| **VII** | The Sound of a Fractal | Laplacian eigenspectrum, density of states, fractons, Alexander-Orbach $d_s = 2d_f/d_w$ verified three ways | 1, 2 |
| **VIII** | First-Passage Processes | $P(T_{\rm fp}>t) \sim t^{-(d_s/2)}$, infinite-mean FPT at $p_c$, chemistry/biology applications | 3 |
| **IX** | 3D Percolation | Cubic lattice, $p_c \approx 0.2488$, numerical exponents, FSS collapse in 3D | 10 |
| **X** | The Ising Model | Wolff cluster algorithm, $M$/$\chi$/$C$ vs $T$, FSS collapse, fractal spin clusters at $T_c$ | 7, 14 |
| **XI** | Why Universality Exists | Real-space RG, cobweb flow diagram, fixed points, $\nu$ from linearisation | 12 |
| **XII** | The Transfer Matrix | Exact strip calculation, correlation length from eigenvalue ratio, no MC noise | 13 |
| **XIII** | Continuous-Time Random Walks | Power-law waiting times, $\langle r^2\rangle \sim t^\alpha$ same as fractal but different mechanism | 8 |
| **XIV** | Fractional Brownian Motion | Hurst exponent, Gaussian but correlated, Davies-Harte FFT generation | 9 |
| **XV** | Distinguishing Anomalous Diffusion | CTRW vs fBm vs fractal — same MSD exponent, different non-Gaussian parameter, ergodicity measure, classifier | 16 |
| **XVI** | Regression: Learning $d_f$ and $d_w$ | RF/GBT regressor on walk features, $R^2$ per target, predicted vs true scatter | 15 |
| **XVII** | Compact Growth — The Eden Model | Eden vs DLA side-by-side, $d_f=2$ vs $d_f\approx1.71$, KPZ interface roughness | 5 |
| **XVIII** | Directed Percolation | Bonds in one direction only, DP universality class, two correlation lengths, epidemic/forest-fire analogy | 6 |
| **XIX** | When One Dimension Is Not Enough | Multifractal spectrum $f(\alpha)$ of DLA harmonic measure, generalised dimensions $D_q$, Legendre transform | 4 |

*RG cobweb visualisation is part of Chapter XI — no separate item needed.*

---

### Coding clusters and order

| Cluster | Items | New chapters | Effort | Dependencies |
|---|---|---|---|---|
| **A — Spectral** | 1, 2 | VII | ~2 days | `scipy.sparse.linalg` (already available) |
| **B — First passage** | 3 | VIII | ~1 day | Existing walker code |
| **C — Anomalous diffusion taxonomy** | 8, 9, 16 | XIII, XIV, XV | ~4 days | `scipy.stats` (already available) |
| **D — Critical phenomena depth** | 6, 7, 12, 13, 14 | X, XI, XII, XVIII | ~1 week | No new dependencies |
| **E — 3D** | 10 | IX | ~2 days | No new dependencies |
| **F — Easy wins** | 5, 15, 19, 20, 21 | XVI, XVII + book polish | ~2 days | Plotly optional |
| **G — Multifractal** | 4 | XIX | ~3 days | Needs large DLA clusters (~10⁵ particles) |

Cluster A starts first. Each cluster is self-contained — we finish and merge one before starting the next.

---

### Standalone CLI for all chapters

Every book chapter can be launched independently — no need to open the full book:

```bash
python -m src.cli book           # full book (port 8500)
python -m src.cli chapter-7      # VII  — The Sound of a Fractal (port 8507)
python -m src.cli chapter-8      # VIII — First-Passage Processes (port 8508)
python -m src.cli chapter-13     # XIII — CTRW (port 8513)
python -m src.cli chapter-14     # XIV  — Fractional Brownian Motion (port 8514)
python -m src.cli chapter-15     # XV   — Distinguishing Anomalous Diffusion (port 8515)
```

New chapters added in Clusters D–G will each get their own `chapter-N` CLI entry at a matching port (chapter-9 → 8509, chapter-10 → 8510, etc.).

---

### Revised book structure (post-build)

```
Prologue
Chapter I     — The Geometry of Fractals
Chapter II    — Walking on a Fractal
Chapter III   — Disorder and the Random Substrate
Chapter IV    — The Critical Point
Chapter V     — Growth as Fractal (DLA)
Chapter VI    — The Inverse Problem (ML classification)
Chapter VII   — The Sound of a Fractal              [NEW — Cluster A]
Chapter VIII  — First-Passage Processes             [NEW — Cluster B]
Chapter IX    — 3D Percolation                      [NEW — Cluster E]
Chapter X     — The Ising Model                     [NEW — Cluster D]
Chapter XI    — Why Universality Exists (RG)        [NEW — Cluster D]
Chapter XII   — The Transfer Matrix                 [NEW — Cluster D]
Chapter XIII  — Continuous-Time Random Walks        [NEW — Cluster C]
Chapter XIV   — Fractional Brownian Motion          [NEW — Cluster C]
Chapter XV    — Distinguishing Anomalous Diffusion  [NEW — Cluster C]
Chapter XVI   — Regression: Learning d_f and d_w   [NEW — Cluster F]
Chapter XVII  — Compact Growth — The Eden Model     [NEW — Cluster F]
Chapter XVIII — Directed Percolation                [NEW — Cluster D]
Chapter XIX   — When One Dimension Is Not Enough    [NEW — Cluster G]
Problems
Epilogue
```
