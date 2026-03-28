"""
Anomalous Diffusion on a Fractal Gasket
─────────────────────────────────────────
Multiple random walkers on a Sierpiński-like gasket whose Hausdorff dimension
is tunable via two knobs:
  • depth   – recursion depth (resolution)
  • n_kept  – sub-triangles kept per level → d_f = log(n_kept)/log(2)

Run with:  streamlit run fractal_walk.py
"""

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st

# ─────────────────────────── page config ────────────────────────────────────
st.set_page_config(
    page_title="Fractal Random Walk",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── colour palette ──────────────────────────────────
BG      = "#0e1117"
PANEL   = "#161b22"
EDGE_C  = "#2a5f8f"
NODE_C  = "#4a9fd4"
WALK_C  = "#ff6b6b"
START_C = "#00e676"
RMS_C   = "#ff6b6b"
VAR_C   = "#a78bfa"
THEO_C  = "#ffd700"
TEXT_C  = "#dce1ec"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor":   PANEL,
        "axes.edgecolor":   "#2c3040",
        "text.color":       TEXT_C,
        "axes.labelcolor":  TEXT_C,
        "xtick.color":      TEXT_C,
        "ytick.color":      TEXT_C,
        "grid.color":       "#2c3040",
        "grid.alpha":       0.6,
        "legend.facecolor": "#1a1f2e",
        "legend.edgecolor": "#2c3040",
        "legend.labelcolor": TEXT_C,
    }
)

# ─────────────────────────── sidebar ────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Fractal Parameters")

    depth = st.slider(
        "Recursion depth", 2, 6, 5,
        help="Higher = finer structure and more graph nodes (slower to render). Depth ≥5 recommended for clean power-law scaling.",
    )
    n_kept = st.slider(
        "Sub-triangles kept  (n_kept)", 2, 4, 3,
        help=(
            "At each subdivision, 4 triangles are created.\n"
            "**2** → two branches  (d_f = 1.000)\n"
            "**3** → Sierpiński    (d_f = 1.585)\n"
            "**4** → solid triangle (d_f = 2.000)"
        ),
    )

    st.markdown("## 🚶 Walk Parameters")
    n_walkers    = st.slider("Walkers",         10, 300, 150)
    n_steps      = st.slider("Total steps",    500, 8000, 4000)
    update_every = st.slider("Update every N steps", 5, 100, 40)
    loglog       = st.checkbox("Log-log axes for stat plots", value=False)

    st.divider()

    # ── derived quantities ──────────────────────────────────────────────────
    d_f = np.log(n_kept) / np.log(2)
    if n_kept == 3:
        d_w     = np.log(5) / np.log(2)        # exact for Sierpiński gasket
        d_w_str = f"{d_w:.3f}  (exact)"
    elif n_kept == 4:
        d_w     = 2.0                           # regular 2-D diffusion
        d_w_str = "2.000  (normal diffusion)"
    else:
        d_w     = None
        d_w_str = "unknown — read from plot"

    st.markdown("### 📐 Fractal Dimensions")
    ca, cb = st.columns(2)
    ca.metric("d_f", f"{d_f:.3f}", help="Hausdorff dimension")
    cb.metric("d_w", f"{d_w:.3f}" if d_w else "?",
              help="Walk (anomalous diffusion) dimension")

    if d_w:
        st.caption(
            f"**RMS exponent** 1/d_w = **{1/d_w:.3f}**  \n"
            f"RMS ∝ N^(1/d_w)  ·  MSD ∝ N^(2/d_w)"
        )
    else:
        st.caption("Walk dimension for n_kept=2 is not known analytically — observe empirically!")

    st.divider()
    st.caption(
        "**Reference values**  \n"
        "Sierpiński (n=3): d_f≈1.585, d_w≈2.322  \n"
        "Normal walk: d_f=2, d_w=2, RMS ∝ √N"
    )

# ─────────────────────────── graph construction ──────────────────────────────
@st.cache_data
def build_gasket(depth: int, n_kept: int):
    """
    Build a generalised Sierpiński gasket as a NetworkX graph.

    Integer coordinates guarantee exact midpoints.
    Node IDs = (i, j) integer tuples.
    Returns (G, pos_dict) where pos_dict maps node → np.array([x, y]).
    """
    G = nx.Graph()
    N = 2 ** depth          # side length in integer units

    def recurse(A, B, C, level):
        if level == 0:
            # Base case: add the triangle
            G.add_nodes_from([A, B, C])
            G.add_edges_from([(A, B), (B, C), (A, C)])
            return

        # Exact integer midpoints
        AB = ((A[0] + B[0]) // 2, (A[1] + B[1]) // 2)
        BC = ((B[0] + C[0]) // 2, (B[1] + C[1]) // 2)
        AC = ((A[0] + C[0]) // 2, (A[1] + C[1]) // 2)

        # Four potential sub-triangles (corner 0,1,2 + inverted centre)
        candidates = [
            (A,  AB, AC),   # bottom-left  corner
            (AB,  B, BC),   # bottom-right corner
            (AC, BC,  C),   # top          corner
            (AB, BC, AC),   # centre (inverted)
        ]
        for tri in candidates[:n_kept]:
            recurse(*tri, level - 1)

    recurse((0, 0), (N, 0), (0, N), depth)

    # Two position dicts:
    #  pos_draw  – normalised [0,1] for rendering
    #  pos_walk  – un-normalised (lattice spacing = 1) for distance stats
    pos_draw, pos_walk = {}, {}
    for (i, j) in G.nodes():
        pos_draw[(i, j)] = np.array([(i + j * 0.5) / N,
                                      (j * 0.86603) / N])
        pos_walk[(i, j)] = np.array([ i + j * 0.5,
                                       j * 0.86603])
    return G, pos_draw, pos_walk, N


with st.spinner("Building fractal graph …"):
    G, pos, pos_w, N_scale = build_gasket(depth, n_kept)

nodes     = list(G.nodes())
adj       = {n: list(G.neighbors(n)) for n in nodes}
n_nodes   = G.number_of_nodes()
n_edges   = G.number_of_edges()
pos_mat   = np.array([pos[n] for n in nodes])   # (n_nodes, 2) normalised

with st.sidebar:
    st.caption(f"Graph: **{n_nodes}** nodes · **{n_edges}** edges")
    if n_kept == 4 and depth >= 5:
        st.warning("n_kept=4 at depth≥5 produces a dense graph — rendering may be slow.")

# ─────────────────────────── drawing helpers ─────────────────────────────────
def _draw_gasket_axes(ax, walker_nodes=None, start_node=None, step=0):
    ax.set_facecolor(BG)

    # ── edges (batched as a single line with None-breaks) ──────────────────
    segs_x, segs_y = [], []
    for u, v in G.edges():
        segs_x += [pos[u][0], pos[v][0], None]
        segs_y += [pos[u][1], pos[v][1], None]
    ax.plot(segs_x, segs_y, color=EDGE_C, alpha=0.55, lw=0.7, zorder=1)

    # ── all nodes ───────────────────────────────────────────────────────────
    ax.scatter(pos_mat[:, 0], pos_mat[:, 1],
               s=4, c=NODE_C, alpha=0.55, linewidths=0, zorder=2)

    # ── start marker ────────────────────────────────────────────────────────
    if start_node is not None:
        sx, sy = pos[start_node]
        ax.scatter([sx], [sy], s=90, c=START_C, marker="*",
                   zorder=7, linewidths=0, label="Start")

    # ── walkers ─────────────────────────────────────────────────────────────
    if walker_nodes:
        wx = [pos[w][0] for w in walker_nodes]
        wy = [pos[w][1] for w in walker_nodes]
        ax.scatter(wx, wy, s=20, c=WALK_C, alpha=0.75,
                   linewidths=0, zorder=5)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Step {step}   ·   d_f = {d_f:.3f}   ·   {n_nodes} nodes",
        color=TEXT_C, fontsize=10, pad=6,
    )


def make_gasket_fig(walker_nodes=None, start_node=None, step=0):
    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=BG)
    _draw_gasket_axes(ax, walker_nodes, start_node, step)
    fig.tight_layout()
    return fig


def make_stats_fig(step_arr, rms_arr, msd_arr, sat_step=None):
    steps = np.asarray(step_arr, dtype=float)
    rms   = np.asarray(rms_arr,  dtype=float)
    msd   = np.asarray(msd_arr,  dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), facecolor=BG)

    plot_fn = "loglog" if loglog and len(steps) > 5 else "plot"

    # ── RMS plot ─────────────────────────────────────────────────────────────
    getattr(ax1, plot_fn)(steps, rms, color=RMS_C, lw=1.8, label="Empirical RMS", zorder=3)
    ax1.set_xlabel("Steps  N")
    ax1.set_ylabel("RMS distance")
    ax1.set_title("RMS Distance  ⟨r²⟩^½", color=TEXT_C)
    ax1.grid(True)

    # ── MSD plot ──────────────────────────────────────────────────────────────
    getattr(ax2, plot_fn)(steps, msd, color=VAR_C, lw=1.8, label="MSD  ⟨r²⟩", zorder=3)
    ax2.set_xlabel("Steps  N")
    ax2.set_ylabel("MSD  ⟨r²⟩")
    ax2.set_title("Mean Squared Displacement  ⟨r²⟩", color=TEXT_C)
    ax2.grid(True)

    if len(steps) < 20:
        ax1.legend(fontsize=9); ax2.legend(fontsize=9)
        fig.tight_layout(); return fig

    # ── Saturation vertical line ──────────────────────────────────────────────
    if sat_step and sat_step < steps[-1]:
        for ax in (ax1, ax2):
            ax.axvline(sat_step, color="#ff9900", lw=1.2, ls="-.",
                       alpha=0.8, label=f"Saturation ≈{int(sat_step)}")

    # Fit range: steps 50 to ~25% of sat_step (clean power-law window).
    # Starting below ~50 catches the transient; going past 25% of sat risks
    # including flattening. We do a log-log regression across this window.
    fit_start = 50
    fit_end   = int(sat_step * 0.25) if sat_step else int(steps[-1] * 0.50)
    fit_end   = max(fit_end, fit_start + 20)   # at least some window
    mask = (steps >= fit_start) & (steps <= fit_end) & (rms > 0) & (msd > 0)

    if mask.sum() > 5:
        log_s = np.log(steps[mask])

        # ── RMS slope via log-log fit ──────────────────────────────────────
        slope_r, intercept_r = np.polyfit(log_s, np.log(rms[mask]), 1)
        fitted_rms = np.exp(intercept_r) * steps ** slope_r
        # Only draw fitted line up to sat_step (or end of data)
        draw_end_idx = np.searchsorted(steps, fit_end * 2.0)
        draw_steps   = steps[:draw_end_idx]
        draw_rms     = np.exp(intercept_r) * draw_steps ** slope_r
        lbl_r_fit = f"Fit slope = {slope_r:.3f}"
        getattr(ax1, plot_fn)(draw_steps, draw_rms, "-.", color="#00cfff",
                               lw=1.4, alpha=0.9, label=lbl_r_fit, zorder=4)

        # ── MSD slope via log-log fit ─────────────────────────────────────
        slope_v, intercept_v = np.polyfit(log_s, np.log(msd[mask]), 1)
        draw_msd = np.exp(intercept_v) * draw_steps ** slope_v
        lbl_v_fit = f"Fit slope = {slope_v:.3f}"
        getattr(ax2, plot_fn)(draw_steps, draw_msd, "-.", color="#00cfff",
                               lw=1.4, alpha=0.9, label=lbl_v_fit, zorder=4)

    # ── Theory reference lines (only drawn up to 2× saturation or data end) ──
    if d_w:
        # Anchor at the midpoint of the fit window so theory doesn't diverge
        anchor_i = np.searchsorted(steps, max(fit_start, fit_end // 2))
        anchor_i = min(anchor_i, len(steps) - 1)
        end_i    = np.searchsorted(steps, fit_end * 2.0)
        end_i    = min(end_i, len(steps))
        theo_steps = steps[:end_i]

        c_r    = rms[anchor_i] / (steps[anchor_i] ** (1.0 / d_w) + 1e-12)
        theo_r = c_r * theo_steps ** (1.0 / d_w)
        getattr(ax1, plot_fn)(theo_steps, theo_r, "--", color=THEO_C,
                               lw=1.5, alpha=0.85,
                               label=f"Theory  N^(1/{d_w:.2f})={1/d_w:.3f}", zorder=2)

        c_v    = msd[anchor_i] / (steps[anchor_i] ** (2.0 / d_w) + 1e-12)
        theo_v = c_v * theo_steps ** (2.0 / d_w)
        getattr(ax2, plot_fn)(theo_steps, theo_v, "--", color=THEO_C,
                               lw=1.5, alpha=0.85,
                               label=f"Theory  N^(2/{d_w:.2f})={2/d_w:.3f}", zorder=2)

    # ── Normal-diffusion reference (dashed grey, same window) ──────────────
    anchor_i = max(1, len(steps) // 10)
    end_i    = np.searchsorted(steps, fit_end * 2.0)
    end_i    = min(end_i, len(steps))
    ref_steps = steps[:end_i]

    c_n   = rms[anchor_i] / (np.sqrt(steps[anchor_i]) + 1e-12)
    getattr(ax1, plot_fn)(ref_steps, c_n * np.sqrt(ref_steps), ":",
                           color="#888", lw=1.2, label="Normal  ∝ √N", zorder=1)
    c_nv  = msd[anchor_i] / (steps[anchor_i] + 1e-12)
    getattr(ax2, plot_fn)(ref_steps, c_nv * ref_steps, ":",
                           color="#888", lw=1.2, label="Normal  ∝ N", zorder=1)

    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ─────────────────────────── main layout ─────────────────────────────────────
st.markdown(
    "## 🌀 Anomalous Diffusion on a Fractal Gasket\n"
    "Walkers diffuse on a Sierpiński-like graph. "
    "Because the fractal is tortuous, the RMS distance grows as "
    r"$N^{1/d_w}$ — **slower** than $\sqrt{N}$ for a regular lattice. "
    "Adjust the sliders then hit **Run Walk**."
)

col_left, col_right = st.columns([1.1, 1.0])

with col_left:
    gasket_ph = st.empty()

with col_right:
    stats_ph = st.empty()

# Static gasket preview before walk
gasket_ph.pyplot(make_gasket_fig())

st.divider()
run_btn = st.button("▶  Run Walk", type="primary")

# ─────────────────────────── walk loop ──────────────────────────────────────
if run_btn:
    # Pick start node closest to the centroid of the equilateral triangle
    # Start at corner (0,0) — maximises graph diameter to distance,
    # giving the longest power-law scaling window before saturation.
    start_node = (0, 0) if (0, 0) in adj else nodes[0]
    start_xy_w = pos_w[start_node]   # un-normalised, for distance stats

    walkers                       = [start_node] * n_walkers
    rms_hist: list[float]         = []
    msd_hist: list[float]         = []
    step_hist: list[int]          = []

    # Estimated saturation step: walkers equilibrate after ~L^d_w steps
    # where L = N_scale (linear size of the graph)
    sat_step_est = float(N_scale ** d_w) if d_w else None

    t0       = time.time()
    progress = st.progress(0, text="Walking …")

    for step in range(1, n_steps + 1):

        # ── advance each walker one step ──────────────────────────────────
        walkers = [
            adj[w][np.random.randint(len(adj[w]))]
            for w in walkers
        ]

        # ── MSD and RMS in un-normalised lattice coords ───────────────────
        sq = np.array([np.sum((pos_w[w] - start_xy_w) ** 2) for w in walkers])
        rms_hist.append(float(np.sqrt(np.mean(sq))))
        msd_hist.append(float(np.mean(sq)))
        step_hist.append(step)

        # ── refresh plots ─────────────────────────────────────────────────
        if step % update_every == 0 or step == n_steps:
            gasket_ph.pyplot(make_gasket_fig(walkers, start_node, step))
            stats_ph.pyplot(
                make_stats_fig(step_hist,
                               np.array(rms_hist),
                               np.array(msd_hist),
                               sat_step=sat_step_est)
            )
            plt.close("all")
            progress.progress(step / n_steps, text=f"Step {step} / {n_steps}")

    elapsed = time.time() - t0
    progress.empty()
    st.success(
        f"✅  Walk complete — {n_steps} steps · {n_walkers} walkers · {elapsed:.1f} s"
    )
