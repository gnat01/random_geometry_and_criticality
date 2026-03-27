# Theory companion — random geometry, criticality, and anomalous diffusion

This note supports the full interactive book (`book.py`) and the legacy lab chapters (`src/app.py`).
The unifying theme: **fractals are the geometry of criticality, and anomalous diffusion is the dynamics of fractals.** Deterministic fractals (carpet, Vicsek) are constructed by hand; random fractals emerge from the percolation phase transition and from growth processes (DLA). In all cases, the absence of a characteristic length scale produces the same dynamical signature — a power-law MSD with exponent 2/d_w < 1.

## Running the code

### Interactive book

```bash
python -m src.cli book          # full book, opens at Prologue (port 8500)
python -m src.cli chapter-1     # Chapter I   — Geometry of Fractals (port 8501)
python -m src.cli chapter-2     # Chapter II  — Walking on a Fractal (port 8502)
python -m src.cli chapter-3     # Chapter III — Disorder and Random Substrate (port 8503)
python -m src.cli chapter-4     # Chapter IV  — The Critical Point (port 8504)
python -m src.cli chapter-5     # Chapter V   — Growth as Fractal (port 8505)
python -m src.cli chapter-6     # Chapter VI  — The Inverse Problem (port 8506)
python -m src.cli chapter-7     # Chapter VII — The Sound of a Fractal (port 8507)
python -m src.cli chapter-8     # Chapter VIII— First-Passage Processes (port 8508)
python -m src.cli chapter-9     # Chapter IX  — 3-D Percolation (port 8509)
python -m src.cli chapter-10    # Chapter X   — The Ising Model (port 8510)
python -m src.cli chapter-11    # Chapter XI  — Why Universality Exists (port 8511)
python -m src.cli chapter-12    # Chapter XII — The Transfer Matrix (port 8512)
python -m src.cli chapter-13    # Chapter XIII— CTRW (port 8513)
python -m src.cli chapter-14    # Chapter XIV — Fractional Brownian Motion (port 8514)
python -m src.cli chapter-15    # Chapter XV  — Distinguishing Anomalous Diffusion (port 8515)
python -m src.cli chapter-16    # Chapter XVI — Directed Percolation (port 8516)
```

### Legacy app chapters (A–E)

```bash
python -m src.cli chapter-a    # geometry + SRW
python -m src.cli chapter-b    # occupation / first passage / traps
python -m src.cli chapter-c    # ML classifier
python -m src.cli chapter-d    # DLA ensemble
python -m src.cli chapter-e    # critical phenomena
```

All subcommands accept `--port N` and `--help`.

---

## 1. Graph models

### 1.1 Sierpiński carpet (Chapters I, II)

Sites on a square subset of ℤ² are connected by nearest-neighbour 4-cardinal edges. A site `(i, j)` belongs to the carpet if, at every ternary scale, it does **not** land in the central third of both coordinates simultaneously (the "remove the central subsquare" rule).

- Largest connected component kept to remove rare isolated islands.
- **Hausdorff dimension:** d_f = log 8 / log 3 ≈ 1.893
- **Walk dimension:** d_w ≈ 2.88;  anomalous exponent 2/d_w ≈ 0.69

### 1.2 Vicsek fractal (Chapters I, II)

Same square lattice; survival determined by the 5 allowed 3×3 blocks (centre + four edge-mid blocks). Edges are 4-neighbour links; largest component kept.

- **Hausdorff dimension:** d_f = log 5 / log 3 ≈ 1.465
- **Walk dimension:** d_w = log 15 / log 3 ≈ 2.46

### 1.3 2-D bond percolation (Chapters III, IV, V, VI, VIII)

Full L×L grid graph; each edge open independently with probability p. Walk lives on the largest connected component.

- Threshold p_c = 1/2 (exact, from self-duality). Below p_c: no giant component. Above p_c: giant component fills O(L²) sites.
- At p_c the giant component is fractal: d_f ≈ 91/48 ≈ 1.896.

### 1.4 3-D cubic-lattice bond percolation (Chapter IX)

Sites on an L×L×L cubic lattice; bonds along all three axes open with probability p. Implemented via vectorised adjacency construction and scipy sparse connected_components.

- **Threshold:** p_c ≈ 0.2488 (numerical, cubic lattice)
- **Exponents (all numerical):** ν ≈ 0.876, β ≈ 0.418, γ ≈ 1.793
- Lower p_c than 2-D because coordination number z = 6 > 4.
- Exponents are irrational — no exact solution is known for d = 3.
- **Upper critical dimension** d_u = 6; for d ≥ 6, mean-field exponents apply (ν = 1/2, β = 1).

---

## 2. Simple random walk (SRW)

On a finite, connected, undirected graph G = (V, E), the SRW picks a neighbour uniformly at each step. The unique stationary distribution is:

    π(v) = deg(v) / (2|E|)

This follows from detailed balance and holds for any connected undirected graph.

---

## 3. Diffusion scaling and walk dimension

Let X_N be the walker's position after N steps embedded in the plane. Define displacement Y_N = X_N − X_0.

- **MSD(N)** = E[‖Y_N‖²]
- **RMS(N)** = √(E[‖Y_N‖²])

On fractals in the regime ℓ_lattice ≪ N^{1/d_w} ≪ ℓ_graph:

    RMS(N) ~ N^{1/d_w},    MSD(N) ~ N^{2/d_w}

where d_w ≥ 2 is the **walk dimension**. Normal diffusion on ℤ^d has d_w = 2. Subdiffusion (d_w > 2) is generic on fractals because of geometric trapping.

**Alexander–Orbach relation (heuristic):** d_w ≈ d_f + d_f / (d_f − 1) for certain fractal classes; not exact in general but gives useful estimates.

**Spectral dimension:** d_s = 2d_f / d_w. The return-to-origin probability decays as P(t) ~ t^{−d_s/2}. For 2-D percolation at p_c, d_s ≈ 1.33 (Alexander–Orbach conjecture; numerically well-supported).

---

## 4. Spectral dimension and the Laplacian (Chapter VII)

The Laplacian L = D − A (D diagonal degree matrix, A adjacency matrix) has eigenvalues 0 = λ_0 ≤ λ_1 ≤ … ≤ λ_{N−1}. The integrated density of states (spectral staircase) obeys:

    N(λ) ~ λ^{d_s/2}

near λ = 0. The return-to-origin probability of a random walk is:

    P(t) = (1/N) Σ_k exp(−λ_k t) ~ t^{−d_s/2}

Three independent measurements of d_s — from the eigenvalue density, the log-slope of P(t), and the Alexander–Orbach formula 2d_f/d_w — should agree to within finite-size errors.

---

## 5. First-passage processes (Chapter VIII)

The **first-passage time** T_fp to target node v* is the random variable:

    T_fp = inf{t ≥ 1 : X_t = v*}

The **survival function** S(t) = Pr(T_fp > t) gives the fraction of walkers yet to reach v*. In the infinite-system limit on a fractal at p_c:

    S(t) ~ t^{−d_s/2}    (power-law decay)

On a finite graph with N nodes, the mean FPT scales as:

    〈T_fp〉 ~ L^{d_w}    (where L ~ N^{1/d_f})

**Finite-size caveat.** The power-law window requires t ≪ 〈T_fp〉 ≪ ∞. On finite graphs of tractable depth, the mean FPT and the finite-size crossover time are of the same order, leaving no clean scaling window. Chapter VIII instead compares S(t) on the fractal against a same-sized 2-D grid — the fractal S(t) lies systematically above the grid, reflecting anomalously slow kinetics from dead ends and bottlenecks.

---

## 6. Walk modifiers (Chapter A extensions)

### 6.1 Lévy long-range edges

For each unconnected pair (u, v), an edge is added with probability:

    p(edge u–v) = p_long × (d_nn / d(u,v))^(α+2)

where d_nn is the median existing edge length and α ∈ (0, 2) is the Lévy exponent. This makes the SRW step-length distribution have a power-law tail P(|step| > r) ~ r^{−α}. The augmentation is O(N²) in node count.

### 6.2 Biased random walk

The SRW step probability is replaced by a weighted draw P(v | u) ∝ w(u, v). Four presets:

| Preset | Weight |
|--------|--------|
| `toward-center` | exp(+γ · Δ / d(u, centroid)) |
| `away-center` | exp(−γ · Δ / d(u, centroid)) |
| `hub-seeking` | deg(v)^{+γ} |
| `hub-avoiding` | deg(v)^{−γ} |

For biased walks, the stationary distribution is no longer deg(v) / 2|E|.

---

## 7. Chapter B observables

### 7.1 Occupation vs. π

For a long trajectory, the empirical visit frequency at v approximates π(v). The heatmap shows the absolute error |empirical − π(v)|.

### 7.2 First passage

The app estimates the distribution of T_A by repeated trials from random starts (excluding starts inside A). The target set is the upper-right nodes in layout space.

### 7.3 Traps and survival

Absorbing traps remove walkers on entry. S(t) = (survivors at t) / (initial walkers). On fractal substrates with sparse traps, S(t) decays more slowly than on Euclidean graphs of the same size.

---

## 8. Chapter C / VI — ML on synthetic physics

### Dataset construction

Each row is produced by:
1. Sampling a geometry label (carpet / Vicsek / percolation) uniformly.
2. Building an instance graph.
3. Computing degree-based stationary entropy H = −Σ_v π(v) log π(v).
4. Running a short parallel SRW; estimating log–log slopes of RMS and MSD.
5. Adding heteroscedastic Gaussian noise to simulate finite-sample measurement uncertainty.

**Features:** log(n_nodes), mean_degree, rms_loglog_slope, msd_loglog_slope, pi_entropy.

### Identifiability caveats

Different families can produce similar walk statistics by coincidence. The model may lean on leakage-style shortcuts (percolation has variable node count; recursive fractals have discrete node counts). Inspect feature importances and confusion matrices before over-interpreting results.

---

## 9. Chapter D / V — Diffusion-Limited Aggregation

### The DLA process

From a seed at the graph centroid, the cluster grows by releasing walkers from non-cluster nodes; a walker sticks when it reaches a node adjacent to the cluster boundary. This is the on-lattice version of the Witten–Sander (1981) model.

### Fractal dimension from R_g scaling

    R_g(M) = √(mean of |x_i − x_cm|²)    R_g ~ M^{1/d_f}

Log–log regression of R_g on M gives slope 1/d_f; d_f ≈ 1.71 in 2-D free space.

### Finite-size analysis

Three regimes: lattice-scale noise (M < ~5), scaling regime (honest fit window), finite-size saturation (R_g approaches substrate diameter). The code detects saturation automatically from the inflection of the smoothed R_g curve.

---

## 10. Chapter E / IV — 2-D percolation critical phenomena

### Critical exponents (all exact for 2-D square lattice)

| Observable | Power law | Exponent |
|---|---|---|
| Correlation length | ξ ~ \|p − p_c\|^{−ν} | ν = 4/3 |
| Order parameter | P∞ ~ (p − p_c)^β | β = 5/36 |
| Susceptibility | S ~ \|p − p_c\|^{−γ} | γ = 43/18 |
| Cluster-size distribution | n_s ~ s^{−τ} | τ = 187/91 ≈ 2.05 |

### Finite-size scaling collapse

    P∞(p, L) = L^{−β/ν} · f[(p − p_c) · L^{1/ν}]

With β/ν = 5/48 ≈ 0.104 and 1/ν = 3/4.

---

## 11. Three-Dimensional Percolation (Chapter IX)

The cubic lattice has coordination number z = 6. The threshold p_c ≈ 0.2488 is determined numerically; no exact self-duality argument exists for d = 3. Critical exponents are also numerical:

    ν ≈ 0.876,    β ≈ 0.418,    γ ≈ 1.793

The FSS collapse uses the same form as 2-D but with the 3-D exponents:

    P∞(p, L) = L^{−β/ν} · f[(p − p_c) · L^{1/ν}]

The larger β (0.418 vs 0.139) means P∞ turns on more steeply above p_c in 3-D. The smaller ν (0.876 vs 1.333) means correlations diverge less slowly. For d ≥ 6 the transition is governed by mean-field exponents (ν = 1/2, β = 1, γ = 1).

---

## 12. The Ising Model (Chapter X)

Spins σ_i ∈ {−1, +1} on a square lattice with Hamiltonian H = −J Σ_{〈ij〉} σ_i σ_j.

### Exact 2-D results (Onsager 1944)

    T_c = 2J / (k_B ln(1 + √2)) ≈ 2.2692    (J = k_B = 1)

| Observable | Exponent | Value |
|---|---|---|
| Magnetisation M ~ (T_c − T)^β | β | 1/8 |
| Susceptibility χ ~ \|T − T_c\|^{−γ} | γ | 7/4 |
| Correlation length ξ ~ \|T − T_c\|^{−ν} | ν | 1 |

### Wolff cluster algorithm

At criticality the Metropolis algorithm requires O(ξ^z) steps to decorrelate (z ≈ 2.17 — critical slowing-down). The Wolff algorithm flips entire correlated clusters:

1. Choose a random seed spin.
2. With probability p_add = 1 − exp(−2J/k_BT), add each same-spin neighbour to the cluster (BFS).
3. Flip all spins in the cluster.

The effective dynamical exponent is z ≈ 0.25 — virtually no slowing-down.

### Binder cumulant

    U_4 = 1 − 〈m^4〉 / (3〈m^2〉^2)

U_4 curves for different system sizes cross at T_c, providing an L-independent estimate of the critical temperature. U_4 → 0 in the disordered phase (Gaussian distribution), U_4 → 2/3 in the ordered phase (two-peak distribution).

---

## 13. Why Universality Exists — Real-Space RG (Chapter XI)

The renormalisation group (RG) coarse-grains the system by replacing blocks of sites with single effective sites and reads off a map on coupling constants p → f(p).

### Triangular-lattice majority rule (bond percolation)

Replace each triangle of three bonds with one effective bond that is open if ≥ 2 of the three are open:

    f(p) = 3p² − 2p³

**Fixed points:** p* = 0 (subcritical, stable), p* = 1/2 (critical, unstable), p* = 1 (supercritical, stable).

**Slope at the critical fixed point:** f'(1/2) = 3/2.

**Correlation-length exponent** from linearised RG (rescaling factor b = 2):

    ν_RG = log b / log f'(p*) = log 2 / log(3/2) ≈ 1.709

Compare to the exact value ν = 4/3 ≈ 1.333. The RG estimate is approximate because the one-step decimation is not exact, but it correctly identifies the fixed point and captures the divergence of ξ.

**Why universality?** Two systems are in the same universality class if their RG maps have the same slope at the critical fixed point. Microscopic details (lattice type, bond vs site percolation) only shift the fixed-point location — not the slope, and hence not the exponents.

---

## 14. The Transfer Matrix (Chapter XII)

### Ising strip

For an Ising strip of width W (periodic boundary in the width direction) and infinite length, the partition function factorises as Z = Tr T^N, where T is the 2^W × 2^W transfer matrix:

    T_{αβ} = exp(K Σ_i σ_i^α σ_i^β  +  (K/2)(h_α + h_β))

with K = J/k_BT and h_α = Σ_i σ_i^α σ_{i+1}^α (horizontal bonds within row α, periodic).

### Correlation length

    ξ(W, T) = 1 / ln(λ_1 / λ_2)

where λ_1 ≥ λ_2 are the two largest eigenvalues of T. At criticality, ξ grows proportionally to W — correlations span the full width. Away from T_c, ξ saturates.

### Noise-free T_c extraction

Plotting ξ/W vs T for several widths: all curves cross near T_c. The crossing point converges to the exact T_c as W → ∞, giving a precision estimate with no Monte Carlo noise. The pseudo-critical temperature T_c(W) (location of the ξ/W maximum) converges from above.

**Computational cost:** O(2^{3W}) to diagonalise the 2^W × 2^W matrix. W ≤ 8 is feasible on a laptop.

---

## 15. Continuous-Time Random Walks (Chapter XIII)

A CTRW replaces the fixed time step with a random waiting time τ drawn from a heavy-tailed distribution. For Pareto-distributed waiting times:

    ψ(τ) ~ τ^{−(1+α)},    0 < α < 1

the mean waiting time is **infinite**, and the diffusion is anomalous:

    MSD(t) ~ t^α    (subdiffusion, α < 1)

### Ergodicity breaking

The CTRW is **non-ergodic**: the time-averaged MSD

    δ²(τ; T) = (1/(T−τ)) ∫_0^{T−τ} |x(t+τ) − x(t)|² dt  ~  τ^α / T^{1−α}

depends on the total observation time T even as T → ∞. The ratio (time-averaged MSD) / (ensemble MSD) fluctuates from walker to walker and does not converge to 1.

### Non-Gaussian parameter

    α_2(t) = 〈r^4(t)〉 / (2〈r^2(t)〉²) − 1

Zero for a Gaussian displacement distribution (normal diffusion or fBm). Elevated for CTRW and fractal diffusion because the displacement distribution has heavy tails (some walkers make large jumps; others sit still).

---

## 16. Fractional Brownian Motion (Chapter XIV)

fBm is the unique Gaussian process with stationary, correlated increments and MSD ~ t^{2H}:

    MSD(t) = 〈|B^H(t)|²〉 = t^{2H},    0 < H < 1

### Hurst exponent H

| Range | Regime | Increment VACF |
|---|---|---|
| H < 1/2 | Subdiffusion (anti-persistent) | Negative at lag 1 |
| H = 1/2 | Normal diffusion (Brownian motion) | Zero |
| H > 1/2 | Superdiffusion (persistent) | Positive at lag 1 |

### Velocity autocorrelation function (VACF)

For fBm, the increment VACF at lag τ is:

    C(τ) = H(2H−1) τ^{2H−2}    (τ ≫ 1)

For H < 1/2, C(1) < 0 — each step is negatively correlated with the previous one (viscoelastic memory). This is the key feature that distinguishes fBm from CTRW and fractal diffusion.

### Davies-Harte algorithm

Generates exact 2-D fBm paths in O(N log N) via the circulant embedding of the fractional Gaussian noise covariance matrix.

---

## 17. Distinguishing Anomalous Diffusion (Chapter XV)

Three mechanisms all produce MSD ~ t^α with α < 1, but from different origins:

| Mechanism | Origin | Non-Gaussian? | Ergodic? | VACF lag 1 |
|---|---|---|---|---|
| Fractal substrate | Geometric disorder | Elevated | Yes | ≈ 0 |
| CTRW | Temporal disorder (heavy-tailed waits) | Elevated | **No** | ≈ 0 |
| fBm (H < 1/2) | Increment anti-correlations | Near 0 | Yes | **Negative** |

### Five-feature classifier

With α_ctrw = H_fbm = 0.70 (identical MSD exponent across all three classes), a random forest trained on:

1. **MSD slope** — same for all three (~0.70) → low discriminative power alone
2. **Non-Gaussian parameter α_2** — separates fBm (≈ 0) from fractal/CTRW (elevated)
3. **Ergodicity ratio** — separates CTRW (< 1) from fractal and fBm (≈ 1)
4. **VACF at lag 1** — separates fBm (negative) from the other two
5. **Kurtosis of |displacement|** — further separates fractal/CTRW from fBm

achieves > 85% accuracy even when the MSD exponent is identical. Feature noise (proportional to each feature's standard deviation) makes the problem realistic and prevents trivially perfect classifiers.

---

## 18. Directed Percolation (Chapter XVI)

### The DP process (1+1 dimensions)

Each active site at (t, x) can activate:
- (t+1, x) via a straight bond (open with probability p)
- (t+1, x+1) via a diagonal bond (open with probability p)

The all-inactive state is an **absorbing state** — once the system dies, it cannot recover. This breaks time-reversal symmetry and places DP outside the equilibrium universality class.

### Critical point and exponents (1+1D DP)

    p_c ≈ 0.6447    (bond, 1+1D)

| Exponent | Symbol | Value |
|---|---|---|
| Order parameter | β | ≈ 0.2765 |
| Spatial correlation length | ν_⊥ | ≈ 1.097 |
| Temporal correlation length | ν_∥ | ≈ 1.734 |
| Density decay exponent | δ = β/ν_∥ | ≈ 0.160 |

At criticality, the active density decays as ρ(t) ~ t^{−δ}. Below p_c, activity dies exponentially. Above p_c, it saturates to a finite density.

### DP universality class

By the Janssen–Grassberger conjecture, any active-absorbing phase transition without additional conservation law or symmetry belongs to the DP class. Physical realisations include epidemic spreading (SIS models at the extinction threshold), interface depinning, and certain reaction-diffusion systems.

### Why p_c > 1/2?

The diagonal bond shifts connectivity rightward; maintaining a percolating cluster in the forward time direction requires a higher bond density than undirected percolation.

---

## 19. Relation to `fractal_walk.py`

The legacy script `fractal_walk.py` studies a **triangular gasket** with a tunable subdivision rule (`n_kept`). The `src/` chapters and the book focus on **square-lattice** constructions and additional observables. The SRW definition and RMS/MSD interpretation are the same in spirit; the triangular gasket is a different fractal family and is not covered in the framework above.
