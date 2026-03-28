# Theory companion — fractal graphs, random walks, and the ML toy

This note supports the code under `src/`: **Chapter A** (geometry), **Chapter B** (observables & first passage), and **Chapter C** (supervised learning on synthetic summaries).

---

## 1. Graph models

### 1.1 Sierpiński carpet (Chapter A)

We place sites on a square subset of ℤ² and connect **nearest neighbors in the four cardinal directions**. A site `(i, j)` belongs to the carpet if, when you repeatedly divide coordinates by 3, you **never** land in the **middle third of both** coordinates at the same scale (the standard “remove the central subsquare” rule).

- Taking the **largest connected component** removes rare tiny islands when the depth is small.
- The **Hausdorff dimension** of the infinite carpet is \(d_f = \log 8 / \log 3 \approx 1.893\).

### 1.2 Vicsek fractal (Chapter A)

We use the same square lattice, with survival determined by the **5 allowed** \(3\times 3\) blocks at each ternary digit: **center + four edge-mid blocks** (a connected Vicsek-cross variant). Edges are 4-neighbor links between surviving sites; we keep the largest component.

- Hausdorff dimension \(d_f = \log 5 / \log 3 \approx 1.465\).

### 1.3 Bond percolation (Chapter A)

Start from a full **grid graph** on an \(L\times L\) torus-like region with 4-neighbors. Each edge is **independently open** with probability \(p\). The walk lives on the **largest connected component** of open edges.

- Near the 2D percolation threshold \(p_c \approx 0.5\) (bond problem on the square lattice), clusters are **fractal** at large scales; far above threshold the giant component is more Euclidean.

---

## 2. Simple random walk (SRW)

On a finite, connected, undirected graph \(G=(V,E)\), the simple random walk picks a neighbor uniformly at each step. The **stationary distribution** (for irreducible aperiodic walk) is

\[
\pi(v) = \frac{\deg(v)}{2|E|}.
\]

Chapter B **occupation** compares empirical visit frequencies to \(\pi\). With a long trajectory, time averages converge to \(\pi\) (ergodic theorem for finite Markov chains).

---

## 3. Diffusion scaling and walk dimension

Let \(X_N\) be the walk position after \(N\) steps, measured in some embedding coordinates, with displacement from the start \(Y_N = X_N - X_0\).

- **Mean squared displacement** \(\mathrm{MSD}(N) = \mathbb{E}\|Y_N\|^2\) (averaged over walkers in the app).
- **Root mean square** \(\mathrm{RMS}(N) = \sqrt{\mathbb{E}\|Y_N\|^2}\).

On many fractals, one expects **anomalous scaling** in the large-\(N\) regime before finite-size saturation:

\[
\mathrm{RMS}(N) \sim N^{1/d_w}, \qquad \mathrm{MSD}(N) \sim N^{2/d_w},
\]

where \(d_w\) is the **walk dimension** (not necessarily an integer). On \(\mathbb{Z}^d\) for simple random walk, \(d_w = 2\) and RMS \(\sim N^{1/2}\).

**Caveat (Chapter A UI):** The optional “literature \(d_w\)” curves are **illustrative** only. Exact \(d_w\) values depend on the precise graph family and are not always known in closed form; empirical slopes from log–log fits are affected by **transients**, **finite graph size**, and **embedding choice** (we use planar coordinates from integer node IDs).

---

## 4. Chapter B observables

### 4.1 Occupation vs. \(\pi\)

For a long single trajectory, the fraction of time spent at \(v\) approximates \(\pi(v)\). Deviations shrink as runtime grows, but on small graphs or short runs you will see visible differences—especially near **bottlenecks** and **boundary effects**.

### 4.2 First passage

Let \(T_A = \inf\{n \ge 0 : X_n \in A\}\). The app estimates the distribution of \(T_A\) by **repeated trials** from random starts (excluding immediate starts inside \(A\)). On finite graphs, \(T_A\) is almost surely finite, but the tail can be heavy when targets are hard to reach through bottlenecks.

The visualization picks a **geometric target** (roughly “upper-right” nodes in layout space) to mimic a non-local crossing problem.

### 4.3 Traps and survival

If some vertices are **absorbing traps**, walkers are removed upon entry. The **survival fraction** vs. time is a basic reaction–diffusion quantity on a graph. In large, trap-sparse regimes it often looks exponential after a transient; on small graphs the curve is dominated by discreteness and geometry.

---

## 5. Chapter C — ML on synthetic physics

Each dataset row is produced by:

1. Sampling a **geometry label** (carpet / Vicsek / percolation).
2. Building an instance graph and computing **degree-based** stationary entropy of \(\pi\) (exact on the finite graph).
3. Running a **short parallel SRW** and estimating **log–log slopes** of RMS and MSD vs. step count over an intermediate time window.

**Features** are compact physics-style summaries: **graph statistics** (size, mean degree) and **dynamics summaries** (RMS/MSD slopes, stationary-entropy proxy).

A **RandomForestClassifier** (with **standardized** features) is a flexible baseline that highlights **feature importances**. This is a **toy inverse problem**: recover the discrete “generating family” from noisy, finite-time summaries.

### Identifiability caveats

- Slopes are **noisy** when graphs are small or walk lengths are short.
- Different families can produce **similar** summary statistics by coincidence.
- The model may lean on **leakage-style shortcuts** (e.g. percolation often has distinct mean degree or entropy at fixed sizes) — inspect importances and confusion matrices before over-interpreting “physics discovery.”

---

## 6. Relation to `fractal_walk.py`

The legacy script `fractal_walk.py` studies a **triangular gasket** with a tunable subdivision rule (`n_kept`). The `src/` chapters focus on **square-lattice** constructions and additional observables, but the **SRW** definition and the **RMS/MSD** interpretation are the same in spirit.
