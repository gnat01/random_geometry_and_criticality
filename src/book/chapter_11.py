"""Chapter XI: Why Universality Exists — Real-Space Renormalisation Group."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..theme import BG, THEO_C, TEXT_C, PANEL, MSD_C, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited


# ---------------------------------------------------------------------------
# Triangular-lattice majority-rule RG for bond percolation
# ---------------------------------------------------------------------------
P_C_EXACT: float = 0.5          # 2-D bond percolation p_c
NU_EXACT: float = 4.0 / 3.0    # exact ν
NU_RG: float = np.log(2) / np.log(1.5)  # ≈ 1.709  (RG estimate)


def _rg_map(p: float) -> float:
    """Majority-rule RG recursion on the triangular lattice: f(p) = 3p²−2p³."""
    return 3.0 * p ** 2 - 2.0 * p ** 3


def _iterate_rg(p0: float, n_steps: int) -> np.ndarray:
    """Return RG trajectory starting at p0 for n_steps iterations."""
    trajectory = np.zeros(n_steps + 1)
    trajectory[0] = p0
    for i in range(n_steps):
        trajectory[i + 1] = _rg_map(trajectory[i])
    return trajectory


def render() -> None:
    mark_visited("chapter_11")
    chapter_header(
        "XI",
        "Why Universality Exists",
        "Real-space renormalisation group and the geometry of the critical fixed point",
    )

    st.markdown(r"""
Chapters IV and X observed that completely different systems — bond percolation and the Ising
model — share similar power-law structure near their critical points. **Why?**

The answer is the **renormalisation group** (RG). The idea: repeatedly coarse-grain the system
by replacing a block of sites with a single effective site. Under this transformation, the
coupling constants (e.g., bond probability $p$, temperature $T$) flow under a map $p \to f(p)$.

**Fixed points of $f$ are the critical points.** If $p^*$ is a fixed point ($f(p^*) = p^*$):
- Stable fixed points ($|f'(p^*)| < 1$): trivial phases (all-connected or all-disconnected)
- Unstable fixed point ($|f'(p^*)| > 1$): the critical point. Small deviations grow under iteration.

The correlation-length exponent comes directly from the slope:

$$\nu = \frac{\log b}{\log |f'(p^*)|},$$

where $b$ is the rescaling factor. The magic: **any system whose RG map has the same slope
at the fixed point belongs to the same universality class** — regardless of microscopic details.

**Example: triangular lattice majority rule.** Replace each triangle of three bonds with one
effective bond that is open if $\geq 2$ of the three bonds are open:

$$f(p) = 3p^2 - 2p^3.$$

Fixed points:
- $p^* = 0$ (all disconnected) — stable ($f'(0) = 0$)
- $p^* = 1/2$ (critical) — unstable ($f'(1/2) = 3/2$)
- $p^* = 1$ (all connected) — stable ($f'(1) = 0$)

The rescaling factor for this triangular decimation is $b = 2$.  Therefore:

$$\nu_{\mathrm{RG}} = \frac{\log 2}{\log(3/2)} \approx 1.709,$$

compared to the exact $\nu = 4/3 \approx 1.333$.  Not perfect, but in the right ballpark —
and, crucially, obtained **without solving the model**.
""")

    definition_box(
        "Renormalisation group",
        r"A map on the space of coupling constants obtained by integrating out short-distance "
        r"degrees of freedom and rescaling. Fixed points correspond to scale-invariant states "
        r"(phases or phase transitions). The linearised map at the critical fixed point "
        r"determines all critical exponents.",
    )

    st.markdown("""
---

### Experiment A — Cobweb diagram: visualising RG flow
""")
    _section_a()

    st.markdown(r"""
---

### Experiment B — Basin of attraction and the critical manifold

Starting from a grid of initial $p_0$ values, iterate the RG map and classify each point as
flowing to 0 (subcritical) or 1 (supercritical). The boundary is the critical manifold — a
single point $p^* = 1/2$ in this 1-D example, but in higher-dimensional coupling spaces it
becomes a full surface.
""")
    _section_b()

    st.markdown(r"""
---

### Experiment C — Linearised RG and ν

Near $p^* = 1/2$, write $p = 1/2 + \epsilon$. The linearised map is:

$$\epsilon \to f'(p^*)\, \epsilon = \frac{3}{2}\, \epsilon.$$

The deviation from criticality grows by a factor of $3/2$ at each RG step. This is why the
correlation length diverges: $\xi$ scales as the inverse of how fast we approach the trivial
fixed point, giving $\xi \sim |\epsilon|^{-\nu}$ with $\nu = \log b / \log(3/2)$.
""")
    _section_c()

    key_result(
        r"Real-space RG on the triangular lattice: $f(p) = 3p^2 - 2p^3$. "
        r"Fixed point at $p^* = 1/2$, slope $f'(p^*) = 3/2$. "
        r"$\nu_\mathrm{RG} = \log 2 / \log(3/2) \approx 1.709$ vs exact $\nu = 4/3 \approx 1.333$. "
        r"Universality arises because the slope at the fixed point is the same for all systems "
        r"in the class — microscopic details only shift the fixed-point location, not the exponents."
    )

    st.markdown("---")
    prev_next("chapter_11")


# ---------------------------------------------------------------------------
# Section A — cobweb
# ---------------------------------------------------------------------------

def _section_a() -> None:
    st.markdown("""
A cobweb diagram alternates between the map $f(p)$ and the diagonal $y = p$.
Starting above $p^*$: flows to 1.  Starting below: flows to 0.  Starting exactly at $p^*$: fixed.
""")

    col1, col2 = st.columns(2)
    with col1:
        p0 = st.slider("Initial $p_0$", 0.01, 0.99, 0.65, step=0.01, key="ch11a_p0")
    with col2:
        n_iter = st.slider("Iterations", 3, 15, 8, step=1, key="ch11a_n")

    _show_cobweb(p0, n_iter)


def _show_cobweb(p0: float, n_iter: int) -> None:
    p_arr = np.linspace(0, 1, 400)
    fp = _rg_map(p_arr)

    traj = _iterate_rg(p0, n_iter)

    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=BG)
    ax.set_facecolor(PANEL)

    ax.plot(p_arr, fp, color=MSD_C, lw=2, label="$f(p) = 3p^2 - 2p^3$")
    ax.plot(p_arr, p_arr, color=TEXT_C, lw=1, ls="--", alpha=0.5, label="$y = p$")

    # Draw cobweb
    p_cur = p0
    for i in range(n_iter):
        f_cur = _rg_map(p_cur)
        # Vertical: (p_cur, p_cur) -> (p_cur, f_cur)
        ax.plot([p_cur, p_cur], [p_cur, f_cur], color=THEO_C, lw=1.2, alpha=0.8)
        # Horizontal: (p_cur, f_cur) -> (f_cur, f_cur)
        ax.plot([p_cur, f_cur], [f_cur, f_cur], color=THEO_C, lw=1.2, alpha=0.8)
        p_cur = f_cur

    # Mark fixed points
    ax.axvline(0.5, color=RMS_C, lw=1.2, ls=":", alpha=0.7, label="$p^* = 1/2$")
    ax.scatter([p0], [p0], color=THEO_C, s=60, zorder=5, label=f"$p_0 = {p0:.2f}$")

    direction = "→ 1 (supercritical)" if p0 > 0.5 else ("→ 0 (subcritical)" if p0 < 0.5 else "→ p* (fixed)")
    ax.set_title(f"RG cobweb: $p_0 = {p0:.2f}$, {direction}", color=TEXT_C, fontsize=10)
    ax.set_xlabel("$p$", color=TEXT_C, fontsize=12)
    ax.set_ylabel("$f(p)$", color=TEXT_C, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    cols = st.columns(3)
    with cols[0]:
        st.metric("$p^*$ (fixed point)", "0.5000")
    with cols[1]:
        st.metric("$f'(p^*)$", "1.5000")
    with cols[2]:
        final = traj[-1]
        st.metric(f"$p$ after {n_iter} steps", f"{final:.4f}")


# ---------------------------------------------------------------------------
# Section B — basin of attraction
# ---------------------------------------------------------------------------

def _section_b() -> None:
    n_grid = st.slider("Grid resolution", 50, 300, 150, step=50, key="ch11b_res")
    n_rg = st.slider("RG iterations", 5, 30, 15, step=5, key="ch11b_rg")

    if st.button("▶ Compute basin", key="ch11b_run"):
        p0_grid = np.linspace(0, 1, n_grid)
        final = np.array([_iterate_rg(p0, n_rg)[-1] for p0 in p0_grid])
        st.session_state["ch11b_data"] = (p0_grid, final)
        _show_b(p0_grid, final, n_rg)
    elif "ch11b_data" in st.session_state:
        p0_grid, final = st.session_state["ch11b_data"]
        _show_b(p0_grid, final, n_rg)


def _show_b(p0_grid: np.ndarray, final: np.ndarray, n_rg: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=BG)
    ax.set_facecolor(PANEL)

    sup = final > 0.5
    ax.scatter(p0_grid[sup], np.ones(sup.sum()), c=THEO_C, s=15, alpha=0.8, label="→ 1 (supercritical)")
    ax.scatter(p0_grid[~sup], np.zeros((~sup).sum()), c=MSD_C, s=15, alpha=0.8, label="→ 0 (subcritical)")
    ax.axvline(0.5, color=RMS_C, lw=2, ls="--", label="$p^* = 1/2$")

    ax.set_xlabel("Initial $p_0$", color=TEXT_C, fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0 (sub)", "1 (super)"], color=TEXT_C, fontsize=9)
    ax.set_title(f"Basin of attraction after {n_rg} RG steps", color=TEXT_C, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    boundary_idx = np.where(np.diff(sup.astype(int)))[0]
    if len(boundary_idx) > 0:
        p_boundary = 0.5 * (p0_grid[boundary_idx[0]] + p0_grid[boundary_idx[0] + 1])
        st.metric("Numerical critical point", f"{p_boundary:.4f}", delta=f"{(p_boundary - 0.5)*1000:.1f}×10⁻³ from 0.5", delta_color="off")


# ---------------------------------------------------------------------------
# Section C — linearised RG
# ---------------------------------------------------------------------------

def _section_c() -> None:
    st.markdown(r"""
The plot below shows how $|p - p^*|$ grows (or shrinks) at each RG step.
For $p_0 \neq p^*$: distance from the fixed point grows by factor $3/2$ per step.
The slope on a log-linear plot equals $\log(3/2)$.
""")
    eps0 = st.slider("Initial deviation $\\epsilon_0 = p_0 - 1/2$", 0.001, 0.2, 0.05, step=0.005, key="ch11c_eps")

    p_traj = _iterate_rg(0.5 + eps0, 12)
    eps_traj = np.abs(p_traj - 0.5)
    # Clip to [0, 0.5] before taking log
    valid = eps_traj > 1e-10
    steps = np.arange(len(eps_traj))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    axes[0].plot(steps, eps_traj, "o-", color=THEO_C, lw=1.8, ms=5)
    axes[0].set_xlabel("RG step", color=TEXT_C, fontsize=11)
    axes[0].set_ylabel("$|p - p^*|$", color=TEXT_C, fontsize=11)
    axes[0].set_title("Distance from fixed point", color=TEXT_C, fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(steps[valid], eps_traj[valid], "o-", color=MSD_C, lw=1.8, ms=5, label="Numerical")
    # Theoretical: ε_n = ε_0 * (3/2)^n
    theory_eps = eps0 * (1.5 ** steps)
    theory_eps = np.minimum(theory_eps, 0.5)
    axes[1].semilogy(steps, theory_eps, "--", color=THEO_C, lw=1.5, alpha=0.7, label="$(3/2)^n \\epsilon_0$")
    axes[1].set_xlabel("RG step", color=TEXT_C, fontsize=11)
    axes[1].set_ylabel("$|p - p^*|$  (log)", color=TEXT_C, fontsize=11)
    axes[1].set_title("Log-linear: slope = $\\log(3/2)$", color=TEXT_C, fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("$f'(p^*) = 3/2$", "1.5000")
    with col2:
        st.metric("$\\nu_\\mathrm{RG} = \\log 2 / \\log(3/2)$", f"{NU_RG:.4f}")
    with col3:
        st.metric("Exact $\\nu = 4/3$", f"{NU_EXACT:.4f}")
