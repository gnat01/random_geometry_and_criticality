# Random geometry and criticality

This repository contains:

1. **`fractal_walk.py`** — Streamlit demo of many walkers on a **generalized Sierpiński gasket** (triangle lattice), with RMS / MSD scaling plots.
2. **`src/`** — A five-chapter lab (**geometry → observables → ML → DLA → critical phenomena**) with a **CLI** and Streamlit UI. See **`THEORY.md`** for the mathematical background.

<p align="center"><em>"How does diffusion look when space is fractal, disordered, or poised at a phase transition?"</em></p>

---

## Setup

```bash
cd /path/to/random_geometry_and_criticality
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: **Streamlit**, **Matplotlib**, **NetworkX**, **NumPy**, **scikit-learn** (Chapter C).

---

## Part 1 — Original gasket app

```bash
streamlit run fractal_walk.py
```

Adjust recursion depth, `n_kept`, walkers, and steps in the sidebar; click **Run Walk**.

---

## Part 2 — Chapters A → B → C → D (`src/`)

All chapters share **`src/app.py`**. The CLI sets `FRACTAL_*` environment variables and starts Streamlit on a chosen port.

### Overview

| Chapter | Focus | CLI subcommand | Default port |
|---------|-------|----------------|--------------|
| **A** | **Geometry** — carpet, Vicsek, or bond percolation + SRW / RMS / MSD; optional Lévy long-range edges and biased walk presets | `chapter-a` | 8501 |
| **B** | **Observables** — occupation vs π, first-passage times, traps & survival | `chapter-b` | 8502 |
| **C** | **ML** — synthetic dataset from random graphs + short walks; RandomForest classifier | `chapter-c` | 8503 |
| **D** | **DLA** — diffusion-limited aggregation on bond-percolation substrates; ensemble R_g scaling & honest finite-size analysis | `chapter-d` | 8504 |
| **E** | **Critical phenomena** — order parameter, susceptibility, cluster-size distribution, finite-size collapse, survival power law, anomalous diffusion exponent vs p | `chapter-e` | 8505 |

### Launch via CLI (recommended)

```bash
python -m src.cli chapter-a --graph carpet --depth 4
python -m src.cli chapter-b --mode occupation
python -m src.cli chapter-c
python -m src.cli chapter-d --perc-size 45 --dla-runs 20
python -m src.cli chapter-e --mode order_parameter
```

All subcommands accept `--help`.

---

### `chapter-a` flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--graph {carpet,vicsek,percolation}` | Graph family | `carpet` |
| `--depth N` | Ternary recursion depth (carpet / Vicsek) | `4` |
| `--perc-size N` | Grid side L for percolation | `24` |
| `--p-open F` | Bond open probability p for percolation | `0.55` |
| `--seed N` | RNG seed for the percolation instance | `42` |
| `--levy-alpha F` | Lévy exponent α ∈ (0, 2). Set > 0 to add long-range edges. 0 = disabled | `0.0` |
| `--levy-p F` | Edge probability at the nearest-neighbour length scale | `0.05` |
| `--bias {none,toward-center,away-center,hub-seeking,hub-avoiding}` | Walk bias preset | `none` |
| `--bias-strength F` | Strength parameter for the bias | `3.0` |
| `--port N` | Streamlit port | `8501` |

**Walk modifier details**

*Lévy long-range edges* — for each pair of nodes (u, v) not already connected, an edge is added with probability `p_long * (d_nn / d(u,v))^(α+2)`, where `d_nn` is the median existing edge length. This makes the augmented graph support Lévy-flight statistics: the probability that a SRW step covers distance > r decays as `r^{−α}`. The graph preview updates the title to show how many Lévy edges were added.

*Bias presets* — replace uniform neighbour selection with a weighted draw:

| Preset | Weight function |
|--------|----------------|
| `toward-center` | exp(+γ · normalised progress toward graph centroid) |
| `away-center` | exp(−γ · normalised progress toward graph centroid) |
| `hub-seeking` | deg(neighbour)^γ |
| `hub-avoiding` | deg(neighbour)^{−γ} |

Both Lévy edges and a bias preset can be active simultaneously.

---

### `chapter-b` flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--mode {occupation,first_passage,traps}` | Which observable suite to open on | `occupation` |
| `--port N` | Streamlit port | `8502` |

---

### `chapter-c` flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--port N` | Streamlit port | `8503` |

In the app choose dataset size, seeds, walk length, noise level, and test fraction, then **Generate & train**.

---

### `chapter-d` flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--perc-size N` | Percolation grid side (larger = more finite-size room) | `45` |
| `--p-open F` | Bond open probability | `0.55` |
| `--seed N` | Substrate RNG seed | `42` |
| `--dla-runs N` | Number of independent DLA realisations in the ensemble | `20` |
| `--dla-max-frac F` | Stop each cluster when it reaches this fraction of substrate nodes | `0.15` |
| `--port N` | Streamlit port | `8504` |

**Why percolation only?** DLA on the recursive fractals (carpet, Vicsek) would terminate quickly because the substrates are small at any tractable depth. Percolation grids can be made arbitrarily large, giving a genuine decade of scaling range before finite-size saturation sets in.

**What the ensemble plot shows**

- Faint lines: individual R_g(M) realisations
- Solid line + band: ensemble mean ± 1σ
- Dashed fit line: slope = 1/d_f, annotated with d_f and R²
- Vertical dotted lines: fit window boundaries, with explicit labels marking the small-cluster-noise region (left) and finite-size-saturation region (right)

The fit window is chosen automatically: the lower boundary is 5 % of max cluster mass; the upper boundary is detected from the inflection of the smoothed R_g curve where growth slows to < 20 % of its early-phase rate, capped at 60 % of max mass.

### `chapter-e` flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--mode {order_parameter,cluster_geometry,finite_size_collapse,critical_dynamics}` | Observable mode to open on | `order_parameter` |
| `--perc-size N` | L for dynamics modes (sweep modes use L = 15, 25, 35 internally) | `28` |
| `--n-samples N` | Percolation samples per (L, p) point | `30` |
| `--seed N` | RNG seed | `42` |
| `--port N` | Streamlit port | `8505` |

**Modes**

| Mode | What it shows |
|------|--------------|
| `order_parameter` | P_∞(p) and susceptibility S(p) for L = 15, 25, 35; transition sharpens with L |
| `cluster_geometry` | Log-binned n_s at user-chosen p values; power law n_s ~ s^{−τ} visible at p_c |
| `finite_size_collapse` | Rescaled P_∞ · L^{β/ν} vs (p − p_c) · L^{1/ν}; all L collapse onto one curve |
| `critical_dynamics` | Survival S(t) on log-linear and log-log axes (exponential vs power-law); MSD exponent β vs p |

**Exact 2D bond-percolation exponents used**

| Symbol | Value | Meaning |
|--------|-------|---------|
| p_c | 1/2 (exact) | Critical threshold (Hammersley self-duality) |
| ν | 4/3 (exact) | Correlation-length exponent |
| β | 5/36 (exact) | Order-parameter exponent |
| τ | 187/91 ≈ 2.05 (exact) | Fisher exponent for n_s ~ s^{−τ} |
| d_w | ≈ 2.87 (numerical) | Walk dimension at p_c |

---

## Project layout

```
random_geometry_and_criticality/
├── fractal_walk.py       # Legacy triangular gasket Streamlit app
├── requirements.txt
├── README.md
├── THEORY.md             # Mathematical background for all chapters
└── src/
    ├── app.py            # Streamlit entry (reads FRACTAL_CHAPTER env var)
    ├── cli.py            # argparse launcher for all four chapters
    ├── theme.py          # Shared dark-theme plot styling
    ├── graphs/
    │   ├── carpet.py     # Sierpiński carpet builder
    │   ├── vicsek.py     # Vicsek cross fractal builder
    │   ├── percolation.py# Bond percolation builder
    │   ├── modifiers.py  # add_levy_edges — post-build graph modifier
    │   └── common.py     # Shared helpers (LCC, positions, mean degree)
    ├── sim/
    │   ├── walks.py      # SRW + biased walk primitives + preset factories
    │   ├── observables.py# Occupation, first passage, traps, MSD/RMS series
    │   ├── dla.py        # DLA growth, ensemble, fit-in-window analysis
    │   └── criticality.py# P_∞, n_s, FSS collapse, survival, MSD exponent vs p
    ├── ml/
    │   └── synthetic.py  # Synthetic dataset generation + RandomForest (Ch. C)
    └── ui/
        ├── chapter_a.py  # Geometry + SRW + Lévy + bias controls
        ├── chapter_b.py  # Occupation / first passage / traps
        ├── chapter_c.py  # ML classifier UI
        ├── chapter_d.py  # DLA ensemble UI
        ├── chapter_e.py  # Critical phenomena UI (4 modes)
        └── plotting.py   # draw_graph_2d, pos_walk_square
```

---

## Notes

- **Performance:** Large depths (carpet/Vicsek ≥ 4) or dense percolation grids slow down the graph draw. For DLA, grid sizes ≥ 60 are recommended for more than one decade of scaling range but will take longer per ensemble run.
- **Lévy edges on large grids:** `add_levy_edges` is O(N²) in node count. It is fine for all graph sizes used in the app (< ~3 000 nodes) but will be noticeably slow on percolation grids larger than ~50×50.
- **Chapter C** is intentionally a **pedagogical** pipeline — see `THEORY.md` §5 for identifiability and shortcut-feature caveats.
- **Chapter D** displays honest finite-size warnings on the plot and in the summary line. A low R² (< 0.90) triggers an additional warning suggesting a larger substrate.

---

## License

Follow your repository's license; documentation describes the bundled code only.
