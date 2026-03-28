# Theory companion — random geometry, criticality, and anomalous diffusion

This note supports the code under `src/`: **Chapter A** (geometry + walk modifiers), **Chapter B** (observables), **Chapter C** (supervised learning), **Chapter D** (DLA), and **Chapter E** (critical phenomena at the percolation transition).

The unifying theme: **fractals are the geometry of criticality, and anomalous diffusion is the dynamics of fractals.** Deterministic fractals (carpet, Vicsek) are constructed by hand; random fractals emerge from the percolation phase transition (Chapter E) and from growth processes (DLA, Chapter D). In all cases, the absence of a characteristic length scale produces the same dynamical signature — a power-law MSD with exponent 2/d_w < 1.

---

## 1. Graph models

### 1.1 Sierpiński carpet (Chapter A)

We place sites on a square subset of ℤ² and connect **nearest neighbors in the four cardinal directions**. A site `(i, j)` belongs to the carpet if, when you repeatedly divide coordinates by 3, you **never** land in the **middle third of both** coordinates at the same scale (the standard "remove the central subsquare" rule).

- We keep the **largest connected component** to remove rare isolated islands at small depths.
- **Hausdorff dimension:** d_f = log 8 / log 3 ≈ 1.893
- **Walk dimension (literature estimate):** d_w ≈ 2.88
- Anomalous diffusion exponent: 2/d_w ≈ 0.69 < 1 (subdiffusion)

### 1.2 Vicsek fractal (Chapter A)

Same square lattice, survival determined by the **5 allowed** 3×3 blocks at each ternary digit: **center + four edge-mid blocks** (a Vicsek-cross variant). Edges are 4-neighbor links between surviving sites; we keep the largest component.

- **Hausdorff dimension:** d_f = log 5 / log 3 ≈ 1.465
- **Walk dimension:** d_w = log 15 / log 3 ≈ 2.46

### 1.3 Bond percolation (Chapter A, D)

Start from a full **grid graph** on an L×L region with 4-neighbors. Each edge is **independently open** with probability p. The walk lives on the **largest connected component**.

- Near the 2D bond percolation threshold p_c ≈ 0.5, the giant cluster is **fractal** at large scales; far above threshold it is more Euclidean.
- The fractal dimension d_f and walk dimension d_w of the incipient cluster at p_c are known from renormalization-group theory but vary continuously away from p_c — see §5 for how we handle this in the ML chapter.

---

## 2. Simple random walk (SRW)

On a finite, connected, undirected graph G = (V, E), the SRW picks a neighbor **uniformly** at each step. The unique **stationary distribution** is

    π(v) = deg(v) / (2|E|)

This follows from detailed balance and holds for any connected undirected graph. Chapter B **occupation** compares empirical visit frequencies to π; with a long enough trajectory they converge by the ergodic theorem for finite Markov chains.

---

## 3. Diffusion scaling and walk dimension

Let X_N be the walker's position after N steps, embedded in the plane using integer lattice coordinates. Define the displacement Y_N = X_N − X_0.

- **MSD(N)** = E[‖Y_N‖²] (ensemble average over walkers)
- **RMS(N)** = √(E[‖Y_N‖²])

On many fractals, in the regime ℓ_lattice ≪ N^{1/d_w} ≪ ℓ_graph (lattice spacing well below the rms displacement which is well below the graph diameter), one expects **anomalous diffusion**:

    RMS(N) ~ N^{1/d_w},    MSD(N) ~ N^{2/d_w}

where d_w ≥ 2 is the **walk dimension**. Normal diffusion on ℤ^d has d_w = 2. Subdiffusion (d_w > 2) is generic on fractals because walkers get trapped in dead ends and bottlenecks.

**Alexander–Orbach relation (heuristic):** d_w ≈ d_f + d_f / (d_f − 1) for certain fractal classes; it is not exact in general but gives useful estimates.

**Finite-size caveat:** The Chapter A UI fits slopes in a window [50 steps, 25 % of saturation estimate]. The theory lines ("literature d_w") are illustrative; empirical slopes are affected by transients, finite graph size, and embedding choice.

---

## 4. Walk modifiers (Chapter A extensions)

### 4.1 Lévy long-range edges

To convert any substrate graph into a **Lévy-flight medium**, `add_levy_edges` augments the graph with non-local connections drawn from a power-law kernel. For each pair of nodes (u, v) not already connected:

    p(edge u–v) = p_long × (d_nn / d(u,v))^(α+2)

where:
- d(u, v) is the Euclidean distance in the plane embedding
- d_nn is the median existing edge length (the lattice unit of the substrate)
- α ∈ (0, 2) is the **Lévy exponent**
- p_long ∈ (0, 1) is the probability of adding an edge at exactly distance d_nn

**Why this exponent?** In 2D with uniform node density, the expected number of edges from node u to nodes at distance r scales as ρ(r) dr ∝ r × p(r) dr. For the resulting SRW step-length distribution to have a power-law tail P(|step| > r) ~ r^{−α}, we need p(r) ∝ r^{−(α+2)}, which is exactly the kernel above.

The resulting SRW on the augmented graph is a **lattice Lévy walk**:
- α < 1: superdiffusive, very long jumps dominate
- α ∈ (1, 2): superdiffusive but with a finite mean jump length
- α → 2: approaches Gaussian diffusion; the MSD exponent 2/d_w is modified toward 1

The fractal structure of the substrate and the Lévy perturbation compete: at early times the Lévy jumps dominate (ballistic or superdiffusive); at late times the substrate geometry re-asserts itself. The crossover length scale is approximately d_nn × p_long^{−1/α}.

**Implementation note:** The augmentation is O(N²) in node count and is applied to a copy of the base graph. The modified graph is not cached — it is rebuilt from the cached base graph whenever walk parameters change.

### 4.2 Biased random walk

The SRW step probability is **uniform** over all neighbors. A biased walk replaces this with a **weighted** draw:

    P(X_{n+1} = v | X_n = u) ∝ w(u, v)

where w(u, v) ≥ 0 is a user-supplied weight function. Four named presets are provided:

**toward-center / away-center**

    w(u, v) = exp(±γ · Δ / d(u, centroid))

where Δ = d(u, centroid) − d(v, centroid) is the reduction in distance to the graph centroid when stepping from u to v, normalised by d(u, centroid). Positive Δ means the step moves closer.

- `toward-center`: sign = +1, so steps that reduce distance are exponentially favoured
- `away-center`: sign = −1, so steps that increase distance are favoured (drift toward boundary)
- Strength γ controls how sharply the bias grows with directional progress; γ = 0 recovers SRW

**hub-seeking / hub-avoiding**

    w(u, v) = deg(v)^{±γ}

- `hub-seeking` (positive exponent): walker preferentially moves to high-degree nodes — an analogue of degree-biased PageRank walks
- `hub-avoiding` (negative exponent): walker preferentially moves to low-degree dead ends — useful for exploring periphery or trapping behaviour

**Stationarity:** For a biased walk that is not the SRW, the stationary distribution π is no longer deg(v) / 2|E|. For gradient-toward walks on a general graph there is no simple closed form; the stationary distribution is the left eigenvector of the transition matrix. Chapter B's occupation comparison will therefore deviate systematically from the SRW baseline when a bias is active — this is intentional and informative.

---

## 5. Chapter B observables

### 5.1 Occupation vs. π

For a long single trajectory, the fraction of time spent at v approximates π(v). Deviations shrink as runtime grows, but on small graphs or short runs you will see visible differences — especially near **bottlenecks** and **boundary effects**. The heatmap shows the absolute error |empirical − π(v)| mapped to node colour.

### 5.2 First passage

Let T_A = inf{n ≥ 0 : X_n ∈ A}. The app estimates the distribution of T_A by **repeated trials** from random starts (excluding immediate starts inside A). On finite graphs T_A is almost surely finite, but the tail can be heavy when the target is separated by bottlenecks. The target set is the "upper-right" nodes in layout space.

### 5.3 Traps and survival

If some vertices are **absorbing traps**, walkers are removed upon entry. The survival fraction S(t) = (surviving walkers at step t) / (initial walkers) measures how quickly the ensemble is absorbed.

On graphs where the target traps are few and spatially separated, S(t) often decays roughly exponentially after a transient. On small or highly connected graphs discreteness dominates.

---

## 6. Chapter C — ML on synthetic physics

### Dataset construction

Each row is produced by:

1. Sampling a **geometry label** (carpet / Vicsek / percolation) uniformly, with balancing to ensure equal class representation.
2. Building an instance graph (random depth for recursive families; random size and p_open for percolation).
3. Computing the **degree-based stationary entropy** H = −∑_v π(v) log π(v).
4. Running a **short parallel SRW** and estimating the **log–log slopes** of RMS and MSD vs. step count over an intermediate time window (steps > 15, avoiding early transients).
5. Adding **heteroscedastic Gaussian noise** to each feature to simulate finite-sample measurement uncertainty.

**Features:** log(n_nodes), mean_degree, rms_loglog_slope, msd_loglog_slope, pi_entropy.

### Classifier

A **RandomForestClassifier** (200 trees, max_depth = 12, balanced subsampling) trained on standardised features. The class imbalance correction via `class_weight="balanced_subsample"` is important because percolation instances can fail (disconnected) and get discarded, leading to subtle imbalances if not controlled.

### Identifiability caveats

- Slopes are **noisy** when graphs are small or walk lengths are short.
- Different families can produce **similar** summary statistics by coincidence — percolation near p_c and Vicsek have overlapping d_w ranges.
- The model may lean on **leakage-style shortcuts**: percolation often has distinct mean degree or entropy at fixed sizes because its node count varies continuously with p_open, while recursive fractals have discrete node counts. Inspect feature importances and confusion matrices before over-interpreting results.
- The **inverse problem is genuinely hard**: two different geometries can produce identical walk statistics in finite time. The classifier is learning a noisy, finite-time approximation.

### Regression extension (replacing classification)

The classification target (family label) can be replaced by continuous regression targets d_f and d_w:

- For carpet and Vicsek, both are known analytically.
- For percolation, d_f and d_w vary with p_open. We estimate them per instance: d_f by box-counting on the LCC node set, d_w from the MSD log–log slope on a long reference walk (thousands of steps). These estimated labels are noisy but unbiased given sufficient steps, making the regression problem realistic rather than artificially clean.

---

## 7. Chapter D — Diffusion-Limited Aggregation

### The DLA process

Starting from a seed node placed at the graph centroid, the cluster grows by repeatedly releasing random walkers from the surrounding non-cluster nodes. A walker **sticks** when it lands on a node adjacent to the current cluster boundary. Growth stops when the cluster reaches a user-specified fraction of the total substrate nodes.

**On-graph DLA rule:**
1. Identify the cluster boundary: non-cluster nodes with at least one cluster neighbour.
2. Launch a walker from a random non-cluster, non-boundary node.
3. Advance the walker by SRW steps until it arrives at a boundary node (it sticks there) or exceeds the timeout (trial is discarded, fresh walker launched).
4. Update the boundary set and repeat.

This is the **on-lattice** version of the classical DLA model of Witten & Sander (1981), restricted to the graph topology of the substrate rather than the full 2D plane.

### Fractal dimension from R_g scaling

The DLA cluster is self-similar with fractal dimension d_f ≈ 1.71 in 2D (empirically well-established). The natural estimator uses the **radius of gyration**:

    R_g(M) = √(mean of |x_i − x_cm|² over cluster nodes i)

where M is the cluster mass (number of nodes) and x_cm is the cluster centroid. By dimensional argument, R_g ~ M^{1/d_f}, so a log–log regression of R_g on M gives slope 1/d_f and hence d_f.

### Why the ensemble matters

DLA is a **stochastic** process: two runs on the same substrate produce different cluster shapes. The radius of gyration at mass M fluctuates across runs. An ensemble of independent realisations lets us:

- Separate **DLA stochasticity** (captured by the σ band) from **substrate geometry** (visible in the mean curve shape)
- Obtain a more reliable slope estimate by fitting the ensemble mean R_g(M)
- Visualise the run-to-run variability, which is part of the physics (DLA clusters are not self-averaging at finite sizes)

### Finite-size effects — honest window analysis

Three length scales determine the quality of the d_f estimate:

1. **Lattice scale** (below which the discrete grid dominates): ~ 1 node. Clusters of mass M < ~5 are in this regime.
2. **Scaling regime** (where DLA self-similarity holds): this is the honest fitting window.
3. **Finite-size saturation** (when R_g approaches the substrate diameter): the cluster wraps around the graph, R_g flattens, and the slope drops.

The code automatically detects the saturation onset by tracking the first derivative of the smoothed R_g(M) curve and finding where it falls to 20 % of its early-phase peak. The fit is performed only within [M_lo, M_hi], and both boundaries are shown as vertical dashed lines on the plot with explicit labels ("small-cluster noise ←" and "→ finite-size saturation").

**Practical guidance:**
- Grid size L = 40–50 (LCC ~900–1200 nodes near p_c) gives roughly one decade of scaling range with max_fraction = 0.15.
- Grid size L = 60–80 gives closer to 1.5 decades but increases run time significantly.
- Keeping max_fraction ≤ 0.15 ensures the cluster remains well within the substrate before saturation.

**Why percolation only?** The recursive fractal substrates (carpet, Vicsek) at tractable depth (2–4) contain 100–500 nodes. A 15 % cap would limit clusters to 15–75 nodes — too small for any meaningful scaling range. Percolation grids are cheap to generate at any size and provide the room needed for DLA to develop.

**DLA on fractal substrates (theory note):** On a fractal substrate the DLA cluster geometry is influenced by *both* the DLA process *and* the substrate topology. The effective d_f can differ from the Euclidean 2D value of 1.71 because the available space is itself fractal. Near p_c, the percolation substrate has d_f^{substrate} ≈ 1.90; the DLA cluster on this substrate is expected to have d_f^{DLA} < d_f^{substrate}, but the precise value depends on the walker dynamics and has been studied in the physics literature (e.g. DLA on percolation clusters, Meakin & Stanley 1983-era results).

---

## 8. Chapter E — Critical phenomena at the percolation transition

### The phase transition

Bond percolation on the square lattice undergoes a **second-order (continuous) phase transition** at p_c = 1/2. The exact value follows from a self-duality argument due to Hammersley (1957): the dual graph of a square lattice is also a square lattice, so if a critical threshold exists it must satisfy p_c = 1 − p_c, giving p_c = 1/2.

Near p_c, the system is characterised by a diverging **correlation length**:

    ξ ~ |p − p_c|^{−ν},    ν = 4/3  (exact)

On a finite L × L lattice, the transition is smeared over a window Δp ~ L^{−1/ν} around p_c.

### Order parameter and susceptibility

The **order parameter** is the fraction of sites belonging to the giant (infinite) component:

    P_∞(p) ~ (p − p_c)^β   for p > p_c,    β = 5/36  (exact)

P_∞ = 0 for p ≤ p_c (no infinite cluster) and rises continuously above p_c. This is the direct analogy of magnetisation in the Ising model.

The **susceptibility** analog is the mean size of finite clusters:

    S(p) = Σ_{s ≠ s_max} s² n_s / N  ~  |p − p_c|^{−γ},    γ = 43/18  (exact)

S(p) peaks sharply at p_c and diverges as L → ∞ — the percolation analog of the susceptibility divergence at a magnetic critical point.

### Cluster-size distribution

At p_c, clusters exist at **every size** with no characteristic scale:

    n_s ~ s^{−τ},    τ = 187/91 ≈ 2.05  (exact, Fisher exponent)

where n_s is the number of s-clusters per lattice site. Off-critical, the power law acquires an **exponential cutoff** at a characteristic size s* ~ |p − p_c|^{−1/σ} (σ = 36/91). The evolution of this cutoff as p moves away from p_c is directly visible in Chapter E's cluster-geometry mode.

### Finite-size scaling collapse

Near p_c the only relevant length scale is ξ. By scaling hypothesis, all thermodynamic quantities depend on L and p only through the dimensionless combination L / ξ ~ L · |p − p_c|^ν. This means:

    P_∞(p, L) = L^{−β/ν} · f[(p − p_c) · L^{1/ν}]

for some universal scaling function f. Plotting y = P_∞ · L^{β/ν} against x = (p − p_c) · L^{1/ν} collapses all L values onto a single curve — a stringent test of both the theory and the numerical measurements.

With exact 2D exponents β = 5/36, ν = 4/3:

    β/ν = (5/36) / (4/3) = 5/48 ≈ 0.104
    1/ν = 3/4 = 0.75

Residual spread in the collapsed curves comes from **corrections to scaling** (subleading terms in the RG flow), which are more visible for small L.

### Critical slowing down

The absence of a characteristic length scale at p_c translates directly into the absence of a characteristic **time** scale for diffusive processes. Two signatures:

**Survival with traps.** With absorbing traps at density ρ on the cluster, the survival fraction decays as:

    S(t) ~ t^{−d_s/2}   at p_c

where d_s is the **spectral dimension** of the incipient infinite cluster. For 2D percolation, d_s ≈ 1.33 (Alexander–Orbach conjecture; numerically well-supported). This power-law decay contrasts sharply with the exponential S(t) ~ exp(−ρ t^{d_s/2}) expected far above p_c on a well-connected graph. On the log-log plot the transition between these regimes is visually striking.

**MSD anomalous exponent.** The MSD exponent β = 2/d_w varies continuously with p:

    β → 1        for p ≫ p_c  (normal diffusion on connected grid)
    β ≈ 2/2.87 ≈ 0.70   at p_c   (subdiffusion on fractal cluster)

The crossover occurs near p_c and sharpens with system size. Plotting β vs p gives a curve that dips to its minimum at p_c — a dynamical signature of the transition that requires no knowledge of the cluster geometry.

### Connection to the rest of the project

The percolation transition is the common thread connecting all chapters:

- **Chapters A/B/C** use percolation as one of three substrate families, treating it as a given random graph. The RMS/MSD anomaly is visible but not explained.
- **Chapter D** shows DLA *on* a percolation substrate. The cluster d_f reflects both the DLA dynamics and the substrate geometry inherited from the transition.
- **Chapter E** opens the hood: the substrate geometry is itself a critical phenomenon, and its fractal nature at p_c is the direct cause of the anomalous diffusion seen throughout the project.

---

## 9. Relation to `fractal_walk.py`

The legacy script `fractal_walk.py` studies a **triangular gasket** with a tunable subdivision rule (`n_kept`). The `src/` chapters focus on **square-lattice** constructions and additional observables. The **SRW** definition and the **RMS/MSD** interpretation are the same in spirit; the triangular gasket is a different fractal family and is not covered in the theoretical framework above.
