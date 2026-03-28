# Random Geometry and Criticality

> *"How does diffusion look when space is fractal, disordered, or poised at a phase transition?"*

An interactive Streamlit physics book covering fractal geometry, random walks, critical phenomena, and anomalous diffusion — from first principles to machine learning. See **`THEORY.md`** for the full mathematical background.

---

## Setup

```bash
cd /path/to/random_geometry_and_criticality
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: **Streamlit**, **Matplotlib**, **NetworkX**, **NumPy**, **SciPy**, **scikit-learn**.

---

## The Interactive Book

An integrated Streamlit book (`book.py`) covering all chapters. Each chapter is a self-contained interactive experiment — run sliders, click buttons, see physics.

### Launch the full book

```bash
python -m src.cli book          # opens at Prologue, port 8500
```

### Jump directly to any chapter

```bash
python -m src.cli chapter-1     # port 8501 (overrides available via --port N)
python -m src.cli chapter-2     # port 8502
# ... etc.
```

### Full chapter CLI reference

| Subcommand | Chapter | Topic | Port |
|------------|---------|-------|------|
| `book` | — | Full book (opens at Prologue) | 8500 |
| `chapter-1` | I | The Geometry of Fractals | 8501 |
| `chapter-2` | II | Walking on a Fractal | 8502 |
| `chapter-3` | III | Disorder and the Random Substrate | 8503 |
| `chapter-4` | IV | The Critical Point | 8504 |
| `chapter-5` | V | Growth as Fractal (DLA) | 8505 |
| `chapter-6` | VI | The Inverse Problem (ML) | 8506 |
| `chapter-7` | VII | The Sound of a Fractal | 8507 |
| `chapter-8` | VIII | First-Passage Processes | 8508 |
| `chapter-9` | IX | Three-Dimensional Percolation | 8509 |
| `chapter-10` | X | The Ising Model | 8510 |
| `chapter-11` | XI | Why Universality Exists (RG) | 8511 |
| `chapter-12` | XII | The Transfer Matrix | 8512 |
| `chapter-13` | XIII | Continuous-Time Random Walks | 8513 |
| `chapter-14` | XIV | Fractional Brownian Motion | 8514 |
| `chapter-15` | XV | Distinguishing Anomalous Diffusion | 8515 |
| `chapter-18` | XVIII | Directed Percolation | 8518 |

All subcommands accept `--port N` and `--help`.

### Book contents

| # | Chapter | What you explore |
|---|---------|-----------------|
| Prologue | — | The central question: geometry → dynamics |
| I | The Geometry of Fractals | Sierpiński carpet, Vicsek fractal, Hausdorff dimension, self-similarity |
| II | Walking on a Fractal | Anomalous diffusion, MSD ~ t^α, measuring d_w empirically |
| III | Disorder and the Random Substrate | Bond percolation, giant component, fractal structure near p_c |
| IV | The Critical Point | Exact 2-D exponents (ν=4/3, β=5/36), finite-size scaling collapse |
| V | Growth as Fractal | DLA on percolation substrates, R_g scaling, fractal dimension from growth |
| VI | The Inverse Problem | Random forest classifies geometry family from walk statistics |
| VII | The Sound of a Fractal | Laplacian spectrum, density of states, spectral dimension d_s |
| VIII | First-Passage Processes | Survival function S(t), mean FPT, fractal vs 2-D grid comparison |
| IX | Three-Dimensional Percolation | p_c≈0.2488, numerical exponents, FSS with 3-D ν, LCC visualisation |
| X | The Ising Model | Wolff cluster algorithm, M/χ/Binder cumulant, exact exponents, FSS |
| XI | Why Universality Exists | RG cobweb f(p)=3p²−2p³, fixed points, ν from f'(p*) |
| XII | The Transfer Matrix | Exact strip ξ(T), ξ/W crossing → T_c, noise-free critical point extraction |
| XIII | Continuous-Time Random Walks | Pareto waiting times, temporal subdiffusion, non-Gaussian parameter |
| XIV | Fractional Brownian Motion | Davies-Harte fGn, correlated increments, VACF signature |
| XV | Distinguishing Anomalous Diffusion | 5-feature classifier: fractal vs CTRW vs fBm, ergodicity ratio |
| XVIII | Directed Percolation | Time-directed bonds, DP universality class, ρ(t)~t^{-δ} at p_c |
| Problems | — | Coding challenges spanning all chapters |
| Epilogue | — | The full arc: geometry → dynamics → criticality → universality |

---

## Legacy App (Chapters A–E)

The original five-chapter lab (`src/app.py`) is still available for direct access to the underlying simulations with more CLI control:

```bash
python -m src.cli chapter-a    # geometry + SRW + Lévy edges + biased walks
python -m src.cli chapter-b    # occupation, first passage, traps
python -m src.cli chapter-c    # ML classifier on synthetic data
python -m src.cli chapter-d    # DLA ensemble + R_g scaling
python -m src.cli chapter-e    # critical phenomena: P∞, n_s, FSS collapse
```

See `THEORY.md` §1–8 for the mathematical background of all legacy chapters.

---

## Legacy Gasket App

```bash
streamlit run fractal_walk.py
```

Generalized Sierpiński gasket (triangle lattice) with adjustable recursion depth and walk parameters.

---

## Project layout

```
random_geometry_and_criticality/
├── book.py               # Interactive book entry point
├── fractal_walk.py       # Legacy triangular gasket app
├── requirements.txt
├── README.md
├── THEORY.md             # Mathematical background (all chapters)
├── EXTENSIONS.md         # Extension ideas and architecture notes
└── src/
    ├── cli.py            # argparse launcher (book + all chapter-N subcommands)
    ├── theme.py          # Shared dark-theme constants
    ├── app.py            # Legacy app entry (reads FRACTAL_CHAPTER env var)
    ├── book/
    │   ├── nav.py        # Chapter registry, sidebar, header/callout helpers
    │   ├── prologue.py
    │   ├── chapter_1.py  # I   — Geometry of Fractals
    │   ├── chapter_2.py  # II  — Walking on a Fractal
    │   ├── chapter_3.py  # III — Disorder and Random Substrate
    │   ├── chapter_4.py  # IV  — The Critical Point
    │   ├── chapter_5.py  # V   — Growth as Fractal
    │   ├── chapter_6.py  # VI  — The Inverse Problem
    │   ├── chapter_7.py  # VII — The Sound of a Fractal
    │   ├── chapter_8.py  # VIII— First-Passage Processes
    │   ├── chapter_9.py  # IX  — 3-D Percolation
    │   ├── chapter_10.py # X   — The Ising Model
    │   ├── chapter_11.py # XI  — Why Universality Exists
    │   ├── chapter_12.py # XII — The Transfer Matrix
    │   ├── chapter_13.py # XIII— Continuous-Time Random Walks
    │   ├── chapter_14.py # XIV — Fractional Brownian Motion
    │   ├── chapter_15.py # XV  — Distinguishing Anomalous Diffusion
    │   ├── chapter_18.py # XVIII—Directed Percolation
    │   ├── problems.py
    │   └── epilogue.py
    ├── sim/
    │   ├── walks.py          # SRW + biased walk primitives
    │   ├── observables.py    # Occupation, first passage, traps, MSD/RMS
    │   ├── dla.py            # DLA growth, ensemble, fit-in-window analysis
    │   ├── criticality.py    # P∞, n_s, FSS collapse, survival, MSD exponent vs p
    │   ├── spectral.py       # Laplacian eigenvalues, return probability
    │   ├── firstpassage.py   # First-passage times, survival function
    │   ├── ctrw.py           # CTRW positions at clock times
    │   ├── fbm.py            # 2-D fBm via Davies-Harte
    │   ├── anomalous.py      # Feature extraction + classifier (Ch. XV)
    │   ├── percolation3d.py  # Cubic-lattice bond percolation, FSS, LCC
    │   ├── ising.py          # Wolff cluster algorithm, temperature scan
    │   ├── transfermatrix.py # Ising strip transfer matrix, ξ from eigenvalues
    │   └── directed_perc.py  # 1+1-D directed bond percolation
    ├── graphs/
    │   ├── carpet.py         # Sierpiński carpet builder
    │   ├── vicsek.py         # Vicsek cross fractal builder
    │   ├── percolation.py    # 2-D bond percolation builder
    │   ├── modifiers.py      # add_levy_edges
    │   └── common.py         # LCC, positions, mean degree helpers
    ├── ml/
    │   └── synthetic.py      # Synthetic dataset + RandomForest (legacy Ch. C)
    └── ui/
        ├── chapter_a.py  through chapter_e.py   # Legacy UI panels
        └── plotting.py
```

---

## Notes

- **Performance:** 3-D percolation and Ising temperature scans can be slow for large L — the defaults are tuned for interactive speed. Increase sizes for better statistics.
- **Transfer matrix:** grows as 2^W × 2^W; W ≤ 6 is fast, W = 7–8 takes a few seconds.
- **Chapter XV noise slider:** set to 0 for clean features (100% accuracy), ~0.5 for realistic measurement noise, 1.0+ for stress-testing the classifier.
- **Lévy edges** (legacy Chapter A) are O(N²) — fine for all graph sizes in the app but slow on grids > 50×50.

---

## License

Follow your repository's license; documentation describes the bundled code only.
