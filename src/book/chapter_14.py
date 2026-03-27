"""Chapter XIV: Fractional Brownian Motion."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..sim.fbm import fbm_positions
from ..theme import BG, THEO_C, TEXT_C, PANEL, MSD_C, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited


def render() -> None:
    mark_visited("chapter_14")
    chapter_header(
        "XIV",
        "Fractional Brownian Motion",
        "Anomalous diffusion from correlated increments",
    )

    st.markdown(r"""
Chapters II–III introduced **geometric** disorder as a cause of subdiffusion (fractal
substrate, walk dimension $d_w > 2$). Chapter XIII introduced **temporal** disorder
(heavy-tailed waiting times in CTRW). Both mechanisms produce MSD $\sim t^\alpha$ with
$\alpha < 1$.

There is a third mechanism that requires neither spatial disorder nor temporal trapping:
**correlated increments**. If successive displacements are statistically dependent, the
net displacement can grow faster or slower than $\sqrt{t}$ even on a completely homogeneous
substrate.

Fractional Brownian motion (fBm) is the canonical model. It is the unique Gaussian
process $B_H(t)$ with stationary increments and MSD $\langle r^2(t) \rangle \sim t^{2H}$,
where $H \in (0, 1)$ is the **Hurst exponent**.

- **$H = 1/2$**: standard Brownian motion — increments are independent.
- **$H < 1/2$**: **anti-persistent** — each step tends to reverse the previous one.
  The walker is sub-diffusive ($2H < 1$). Physical realisation: a polymer monomer in a
  viscoelastic fluid, where the memory of the surrounding medium creates a restoring
  drift after each displacement. Measured Hurst exponents in the cytoplasm of living cells
  range from 0.3 to 0.45.
- **$H > 1/2$**: **persistent** — steps tend to continue in the same direction.
  The walker is super-diffusive ($2H > 1$). Physical realisations include active matter,
  driven colloidal particles, and financial time series (where $H \approx 0.6\text{–}0.7$
  has been claimed for some asset classes, though this remains controversial).

Unlike CTRW, fBm is **ergodic**: time-averaged and ensemble-averaged MSDs coincide as
$T \to \infty$. Unlike fractal walks, it is **Gaussian**: the displacement distribution
is always normal. These two properties make fBm the simplest and most analytically
tractable of the three anomalous diffusion models, and the only one for which an
exact analytical solution exists for essentially any observable.

The fingerprint of fBm is the **velocity autocorrelation function** (VACF). For $H < 1/2$
the VACF at lag 1 is negative — successive velocity increments anti-correlate. For
$H > 1/2$ it is positive. For $H = 1/2$ the VACF is identically zero at all non-zero
lags. This is a sharp diagnostic that distinguishes fBm from CTRW (VACF $\approx 0$)
and from fractal diffusion.
""")

    definition_box(
        "Fractional Brownian motion",
        r"A Gaussian process $B_H(t)$ with $B_H(0) = 0$ and covariance "
        r"$\mathbb{E}[B_H(s) B_H(t)] = \tfrac{1}{2}(s^{2H} + t^{2H} - |t-s|^{2H})$. "
        r"The Hurst exponent $H \in (0, 1)$ controls the persistence: "
        r"$H < \tfrac{1}{2}$ (subdiffusion, anti-persistent increments), "
        r"$H = \tfrac{1}{2}$ (standard Brownian motion), "
        r"$H > \tfrac{1}{2}$ (superdiffusion, persistent increments).",
    )

    st.markdown(r"""
---

### Experiment 1 — Sample paths at three Hurst exponents

We generate five 2-D fBm paths for $H = 0.3$, $0.5$, and $0.7$ simultaneously.
The qualitative difference is striking: anti-persistent paths ($H = 0.3$) look jagged and
confined; persistent paths ($H = 0.7$) have long, smooth, directional streaks.
""")

    if st.button("▶ Generate sample paths", key="ch14_run1"):
        _run_paths()
    elif "ch14_paths" in st.session_state:
        _show_paths(st.session_state["ch14_paths"])

    st.markdown(r"""
---

### Experiment 2 — MSD and velocity autocorrelation

The left panel shows the MSD on a log-log scale; the fitted slope should match the theory
value $2H$. The right panel shows the normalised VACF $C(\ell) / C(0)$ for the current
$H$ and for the $H = 0.5$ reference. The sign of $C(1)$ is the sharpest diagnostic
for identifying fBm in experimental data.
""")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        H_val = st.slider(
            "Hurst exponent $H$", 0.20, 0.80, 0.35, step=0.05, key="ch14_H"
        )
    with col_b:
        n_walkers_2 = st.slider(
            "Walkers", 200, 2000, 500, step=100, key="ch14_nw"
        )
    with col_c:
        n_steps_2 = st.slider(
            "Steps", 200, 1000, 400, step=100, key="ch14_ns"
        )

    if st.button("▶ Run MSD + VACF experiment", key="ch14_run2"):
        _run_msd(H_val, n_walkers_2, n_steps_2)
    elif "ch14_result" in st.session_state:
        _show_msd(st.session_state["ch14_result"])

    key_result(
        r"fBm produces anomalous diffusion through increment correlations, not through "
        r"spatial disorder or temporal trapping. It is the only one of the three "
        r"subdiffusion mechanisms that is (a) Gaussian and (b) ergodic. The velocity "
        r"autocorrelation function (VACF) at lag 1 is negative for $H < \tfrac{1}{2}$ "
        r"and positive for $H > \tfrac{1}{2}$ — a direct fingerprint of the mechanism "
        r"distinguishing it from fractal walks and CTRW."
    )

    st.markdown("---")
    prev_next("chapter_14")


# ---------------------------------------------------------------------------
# Experiment 1: sample paths
# ---------------------------------------------------------------------------

def _run_paths() -> None:
    with st.spinner("Generating fBm sample paths…"):
        rng = np.random.default_rng(99)
        n_steps = 300
        n_show = 5
        H_values = [0.3, 0.5, 0.7]
        paths = {}
        for H in H_values:
            paths[H] = fbm_positions(n_show, n_steps, H, rng)

    st.session_state["ch14_paths"] = {"paths": paths, "n_steps": n_steps}
    _show_paths(st.session_state["ch14_paths"])


def _show_paths(res: dict) -> None:
    paths = res["paths"]
    H_values = list(paths.keys())
    palette = {0.3: "#4a9fd4", 0.5: "#aaaaaa", 0.7: "#ff6b6b"}
    titles = {
        0.3: "H = 0.3 (anti-persistent, subdiffusion)",
        0.5: "H = 0.5 (Brownian motion)",
        0.7: "H = 0.7 (persistent, superdiffusion)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor=BG)

    for ax, H in zip(axes, H_values):
        ax.set_facecolor(PANEL)
        pos = paths[H]  # (n_show, n_steps+1, 2)
        col = palette[H]
        for i in range(pos.shape[0]):
            ax.plot(pos[i, :, 0], pos[i, :, 1], color=col, lw=0.9, alpha=0.75)
            ax.scatter([pos[i, 0, 0]], [pos[i, 0, 1]], s=20, color=col, zorder=3)

        ax.set_title(titles[H], color=TEXT_C, fontsize=10)
        ax.set_xlabel("$x$", color=TEXT_C, fontsize=10)
        ax.set_ylabel("$y$", color=TEXT_C, fontsize=10)
        ax.tick_params(colors=TEXT_C, labelsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: MSD and VACF
# ---------------------------------------------------------------------------

def _run_msd(H: float, n_walkers: int, n_steps: int) -> None:
    with st.spinner(f"Simulating fBm (H={H}, {n_walkers} walkers, {n_steps} steps)…"):
        rng = np.random.default_rng(42)
        pos_H = fbm_positions(n_walkers, n_steps, H, rng)
        # BM reference
        pos_bm = fbm_positions(n_walkers, n_steps, 0.5, rng)

    t_arr = np.arange(n_steps + 1, dtype=np.float64)

    def _msd(pos):
        disp = pos - pos[:, :1, :]
        return np.mean(np.sum(disp ** 2, axis=-1), axis=0)

    def _vacf(pos, max_lag=21):
        v = pos[:, 1:, :] - pos[:, :-1, :]  # (n_walkers, n_steps, 2)
        vv0 = float(np.mean(np.sum(v ** 2, axis=-1)))
        if vv0 < 1e-20:
            return np.zeros(max_lag)
        c = np.empty(max_lag)
        c[0] = 1.0
        for lag in range(1, max_lag):
            if lag >= v.shape[1]:
                c[lag] = 0.0
            else:
                c[lag] = float(np.mean(np.sum(v[:, :-lag, :] * v[:, lag:, :], axis=-1))) / vv0
        return c

    msd_H = _msd(pos_H)
    msd_bm = _msd(pos_bm)

    vacf_H = _vacf(pos_H)
    vacf_bm = _vacf(pos_bm)

    # Fit MSD slope
    def _fit_slope(t, y):
        mask = (t > 5) & (y > 1e-12)
        if mask.sum() < 4:
            return float("nan")
        return float(np.polyfit(np.log(t[mask]), np.log(y[mask]), 1)[0])

    slope_H = _fit_slope(t_arr[1:], msd_H[1:])

    result = {
        "H": H,
        "t_arr": t_arr,
        "msd_H": msd_H,
        "msd_bm": msd_bm,
        "vacf_H": vacf_H,
        "vacf_bm": vacf_bm,
        "slope_H": slope_H,
    }
    st.session_state["ch14_result"] = result
    _show_msd(result)


def _show_msd(res: dict) -> None:
    t_arr = res["t_arr"]
    H = res["H"]
    msd_H = res["msd_H"]
    msd_bm = res["msd_bm"]
    vacf_H = res["vacf_H"]
    vacf_bm = res["vacf_bm"]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=BG)
        ax.set_facecolor(PANEL)

        mask = t_arr > 0
        ax.plot(t_arr[mask], msd_H[mask], color=THEO_C, lw=2,
                label=f"fBm ($H = {H:.2f}$)")
        ax.plot(t_arr[mask], msd_bm[mask], color=MSD_C, lw=2, linestyle="--",
                label="BM ($H = 0.50$)")

        # Theory reference slope
        idx_ref = len(t_arr) // 4
        if idx_ref > 0 and msd_H[idx_ref] > 0:
            t_ref = t_arr[idx_ref:]
            scale = msd_H[idx_ref] / (t_arr[idx_ref] ** (2 * H))
            ax.plot(t_ref, scale * t_ref ** (2 * H), ":", color=RMS_C, lw=1.5,
                    label=f"Theory $t^{{2H}} = t^{{{2*H:.2f}}}$")

        if not np.isnan(res["slope_H"]):
            ax.annotate(
                f"Fit slope = {res['slope_H']:.3f}\nTheory 2H = {2*H:.3f}",
                xy=(0.05, 0.95), xycoords="axes fraction",
                va="top", fontsize=9, color=THEO_C,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time $t$", color=TEXT_C, fontsize=11)
        ax.set_ylabel(r"$\langle r^2(t) \rangle$", color=TEXT_C, fontsize=11)
        ax.set_title("MSD: fBm", color=TEXT_C, fontsize=11)
        ax.tick_params(colors=TEXT_C)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5), facecolor=BG)
        ax2.set_facecolor(PANEL)

        lags = np.arange(len(vacf_H))
        ax2.bar(lags, vacf_H, color=THEO_C, alpha=0.8, label=f"fBm $H={H:.2f}$",
                width=0.4, align="center")
        ax2.bar(lags + 0.4, vacf_bm, color=MSD_C, alpha=0.6, label="BM $H=0.50$",
                width=0.4, align="center")
        ax2.axhline(0, color=TEXT_C, lw=0.8, alpha=0.5)

        if H < 0.5:
            ax2.annotate("Anti-persistent\n(H<0.5): negative VACF",
                         xy=(1, vacf_H[1]), xytext=(5, -0.15),
                         color=THEO_C, fontsize=8,
                         arrowprops=dict(arrowstyle="->", color=THEO_C, lw=0.8))
        elif H > 0.5:
            ax2.annotate("Persistent\n(H>0.5): positive VACF",
                         xy=(1, vacf_H[1]), xytext=(5, 0.15),
                         color=THEO_C, fontsize=8,
                         arrowprops=dict(arrowstyle="->", color=THEO_C, lw=0.8))

        ax2.set_xlabel("Lag $\\ell$", color=TEXT_C, fontsize=11)
        ax2.set_ylabel("$C(\\ell) / C(0)$", color=TEXT_C, fontsize=11)
        ax2.set_title("Velocity autocorrelation function (VACF)", color=TEXT_C, fontsize=11)
        ax2.tick_params(colors=TEXT_C)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2, axis="y")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.metric(
            "Fitted MSD slope",
            f"{res['slope_H']:.3f}" if not np.isnan(res["slope_H"]) else "N/A",
        )
    with mc2:
        st.metric("Theory $2H$", f"{2*H:.3f}")
