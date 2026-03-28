# Fractal random walks

This repository contains:

1. **`fractal_walk.py`** — Streamlit demo of many walkers on a **generalized Sierpiński gasket** (triangle lattice), with RMS / MSD scaling plots.
2. **`src/`** — A three-chapter lab (**geometry → observables → ML**) with a **CLI** and Streamlit UI. See **`THEORY.md`** for the mathematical background.

<p align="center"><em>“How does diffusion look when space is fractal or disordered?”</em></p>

---

## Setup

```bash
cd /path/to/fractals
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

## Part 2 — Chapters A → B → C (`src/`)

All chapters share **`src/app.py`**. The CLI sets `FRACTAL_*` environment variables and starts Streamlit on a chosen port.

### Global idea

| Chapter | Focus | CLI subcommand |
|--------|--------|----------------|
| **A** | **Geometry** — carpet, Vicsek, or bond percolation + SRW & RMS/MSD | `chapter-a` |
| **B** | **Observables** — occupation vs π, first-passage times, traps & survival | `chapter-b` |
| **C** | **ML** — synthetic dataset from random graphs + short walks; RandomForest classifier | `chapter-c` |

### Launch via CLI (recommended)

From the repo root:

```bash
python -m src.cli chapter-a --graph carpet --depth 4 --port 8501
python -m src.cli chapter-b --mode occupation --port 8502
python -m src.cli chapter-c --port 8503
```

Equivalent:

```bash
python3 -m src.cli chapter-a --help
python3 -m src.cli chapter-b --help
python3 -m src.cli chapter-c --help
```

### `chapter-a` flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--graph {carpet,vicsek,percolation}` | Graph family | `carpet` |
| `--depth` | Ternary recursion depth (carpet / Vicsek) | `4` |
| `--perc-size` | Grid side \(L\) for percolation (also seeds sidebar default via env) | `24` |
| `--p-open` | Bond open probability \(p\) for percolation | `0.55` |
| `--seed` | RNG seed for the percolation instance | `42` |
| `--port` | Streamlit port | `8501` |

Environment variables set for the session: `FRACTAL_CHAPTER=A`, `FRACTAL_GRAPH`, `FRACTAL_DEPTH`, `FRACTAL_SEED`, `FRACTAL_PERC_SIZE`, `FRACTAL_P_OPEN`. The UI still lets you change graph and parameters interactively.

### `chapter-b` flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--mode {occupation,first_passage,traps}` | Which observable suite to open on | `occupation` |
| `--port` | Streamlit port | `8502` |

Sets `FRACTAL_CHAPTER=B`, `FRACTAL_MODE=...`. Use the sidebar to switch graph family and mode-specific parameters (walk length, trials, traps, etc.).

### `chapter-c` flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--port` | Streamlit port | `8503` |

Sets `FRACTAL_CHAPTER=C`. In the app, choose dataset size, seeds, walk length, then **Generate & train**.

### Launch without CLI

```bash
export FRACTAL_CHAPTER=A   # or B, C
streamlit run src/app.py --server.port 8501
```

Optional: set `FRACTAL_GRAPH`, `FRACTAL_DEPTH`, `FRACTAL_MODE`, etc., as in the CLI sections above.

---

## Project layout

```
fractals/
├── fractal_walk.py      # Legacy triangular gasket Streamlit app
├── requirements.txt
├── README.md
├── THEORY.md            # Theory walkthrough (fractals, SRW, Ch. C caveats)
└── src/
    ├── app.py           # Streamlit entry (reads FRACTAL_CHAPTER)
    ├── cli.py           # argparse launcher
    ├── theme.py         # Shared plot styling
    ├── graphs/          # Carpet, Vicsek, percolation builders
    ├── sim/             # SRW + Chapter B observables
    ├── ml/              # Synthetic dataset + RandomForest (Chapter C)
    └── ui/              # chapter_a / chapter_b / chapter_c Streamlit pages
```

---

## Notes

- **Performance:** Large depths (carpet/Vicsek) or dense percolation graphs can be slow to draw.
- **Chapter C** is intentionally a **pedagogical** pipeline: check **`THEORY.md`** for identifiability and shortcut features.

---

## License

Follow your repository’s license; documentation describes the bundled code only.
