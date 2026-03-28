"""Chapter XIII: Continuous-Time Random Walks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..sim.ctrw import ctrw_positions, brownian_positions
from ..theme import BG, THEO_C, TEXT_C, PANEL, MSD_C, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited


def render() -> None:
    mark_visited("chapter_13")
    chapter_header(
        "XIII",
        "Continuous-Time Random Walks",
        "Subdiffusion from temporal disorder",
    )

    st.markdown(r"""
Chapters II and III showed two routes to anomalous diffusion rooted in **geometry**: the
fractal substrate forces the walker to take long detours, raising the walk dimension above
2. But there is a second, entirely distinct route — one involving no spatial disorder at all.

In a **continuous-time random walk** (CTRW), the walker moves on a perfectly regular
Euclidean lattice. After each jump it *waits* at the current site for a random time $W$
before jumping again. If the waiting-time distribution is light-tailed (e.g. exponential),
the mean waiting time is finite and normal diffusion is recovered. The anomaly arises when
$W$ has a **heavy tail**:

$$P(W > t) \sim t^{-\alpha}, \quad 0 < \alpha < 1.$$

When $\alpha < 1$ the mean waiting time diverges. Rare, extremely long pauses dominate
the dynamics. The ensemble-averaged mean-squared displacement grows as

$$\langle r^2(t) \rangle \sim t^{\alpha} \quad (\alpha < 1),$$

which is subdiffusion — slower than linear — even though the underlying spatial substrate
is a flat integer lattice.

The CTRW model was introduced by Montroll and Weiss in 1965 and mathematically formalised
by the fractional Fokker-Planck equation. It has become the standard model for:

- **Protein diffusion** in the crowded cell interior, where the protein transiently binds
  to immobile obstacles and waits before diffusing on.
- **Charge transport** in amorphous semiconductors (the original application), where
  carriers are repeatedly trapped at disorder-induced energy wells.
- **Diffusion in living cells** measured by single-particle tracking, where $\alpha$ values
  of 0.5–0.9 are routinely observed.

Two fingerprints distinguish CTRW from fractal subdiffusion. First, the displacement
distribution is **non-Gaussian**: the non-Gaussian parameter $\alpha_2 > 0$ and grows
with time. Second, the process is **weakly non-ergodic**: even as the observation time
$T \to \infty$, the time-averaged MSD from a single trajectory does not converge to the
ensemble MSD. These experimental signatures are measurable and allow CTRW to be identified
in biophysical data without requiring any structural information about the medium.
""")

    definition_box(
        "Continuous-time random walk",
        r"A random walk in which the walker waits at each site for a random time $W$ "
        r"before jumping. If $W$ has a power-law tail $P(W > t) \sim t^{-\alpha}$ with "
        r"$0 < \alpha < 1$, the mean waiting time diverges and the MSD grows as "
        r"$\langle r^2(t) \rangle \sim t^{\alpha}$ (subdiffusion).",
    )

    st.markdown("""
---

### Experiment 1 — Trajectories and waiting times

We plot the $x$-coordinate vs time for three CTRW walkers (solid) and three regular
Brownian walkers (dashed). The CTRW trajectories show long flat segments — the walker is
stuck waiting — followed by sudden jumps. The right panel shows the empirical
distribution of 10 000 waiting times on a log-log scale with its power-law fit.
""")

    alpha_1 = st.slider(
        r"$\alpha$ (tail exponent)", 0.30, 0.90, 0.70, step=0.05, key="ch13_alpha1"
    )

    if st.button("▶ Run trajectory experiment", key="ch13_run1"):
        _run_trajectories(alpha_1)
    elif "ch13_traj" in st.session_state:
        _show_trajectories(st.session_state["ch13_traj"])

    st.markdown(r"""
---

### Experiment 2 — MSD and non-Gaussian parameter

We run a large ensemble of CTRW walkers and compare the MSD and the non-Gaussian parameter
$\alpha_2(t) = \langle r^4 \rangle / (2 \langle r^2 \rangle^2) - 1$
to a regular Brownian walk. For a Gaussian process $\alpha_2 = 0$ at all times; for CTRW
$\alpha_2 > 0$ and grows, reflecting the increasingly non-Gaussian displacement
distribution caused by the rare long trapping events.
""")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        alpha_2 = st.slider(
            r"$\alpha$", 0.30, 0.90, 0.70, step=0.05, key="ch13_alpha2"
        )
    with col_b:
        n_walkers_2 = st.slider(
            "Walkers", 500, 3000, 1000, step=500, key="ch13_nw2"
        )
    with col_c:
        n_jumps_2 = st.slider(
            "Jumps per walker", 2000, 20000, 8000, step=2000, key="ch13_nj2"
        )

    if st.button("▶ Run MSD experiment", key="ch13_run2"):
        _run_msd(alpha_2, n_walkers_2, n_jumps_2)
    elif "ch13_result" in st.session_state:
        _show_msd(st.session_state["ch13_result"])

    st.markdown(r"""
---

### Ergodicity breaking

For a CTRW with $\alpha < 1$, the **time-averaged MSD** measured from a single trajectory
of total length $T$,

$$\bar{\delta}^2(\tau; T) = \frac{1}{T - \tau} \int_0^{T-\tau} |x(t+\tau) - x(t)|^2 \, dt,$$

scales as $\bar{\delta}^2(\tau) \propto \tau^\alpha / T^{1-\alpha}$. This depends on
$T$ even as $T \to \infty$ — the process never "self-averages". In practice this means
that two observers who measure the same physical system over different time windows will
extract different apparent diffusion coefficients. This is a measurable, experimentally
accessible signature of CTRW that cannot be faked by a Gaussian ergodic process.

""")

    key_result(
        r"CTRW produces MSD $\sim t^\alpha$ with $\alpha < 1$ (subdiffusion) on a regular "
        r"Euclidean lattice — no fractal geometry required. The mechanism is purely temporal: "
        r"rare long waiting times dominate transport. The displacement distribution is "
        r"non-Gaussian ($\alpha_2 > 0$) and the process is weakly non-ergodic: the "
        r"time-averaged MSD depends on the total observation time $T$ even as $T \to \infty$."
    )

    st.markdown("---")
    prev_next("chapter_13")


# ---------------------------------------------------------------------------
# Experiment 1: trajectories
# ---------------------------------------------------------------------------

def _run_trajectories(alpha: float) -> None:
    with st.spinner("Simulating CTRW and BM trajectories…"):
        rng = np.random.default_rng(17)
        n_show = 3
        t_eval = np.arange(0, 201, dtype=np.float64)

        pos_ctrw = ctrw_positions(
            n_walkers=n_show,
            n_jumps=5000,
            alpha=alpha,
            t_eval=t_eval,
            rng=rng,
        )  # (n_show, T, 2)

        pos_bm = brownian_positions(
            n_walkers=n_show,
            t_eval=t_eval,
            rng=rng,
        )

        # Generate waiting-time samples for histogram
        u = rng.random(10_000)
        u = np.clip(u, 1e-14, 1.0)
        waiting_samples = u ** (-1.0 / alpha)

    result = {
        "alpha": alpha,
        "t_eval": t_eval,
        "pos_ctrw": pos_ctrw,
        "pos_bm": pos_bm,
        "waiting_samples": waiting_samples,
    }
    st.session_state["ch13_traj"] = result
    _show_trajectories(result)


def _show_trajectories(res: dict) -> None:
    t_eval = res["t_eval"]
    pos_ctrw = res["pos_ctrw"]
    pos_bm = res["pos_bm"]
    waiting_samples = res["waiting_samples"]
    alpha = res["alpha"]

    ctrw_colors = [THEO_C, MSD_C, RMS_C]
    bm_colors = ["#4a9fd4", "#a78bfa", "#ff6b6b"]

    col1, col2 = st.columns([3, 2])

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG)
        ax.set_facecolor(PANEL)

        for i in range(pos_ctrw.shape[0]):
            ax.plot(
                t_eval, pos_ctrw[i, :, 0],
                color=ctrw_colors[i], lw=1.2, alpha=0.9,
                label=f"CTRW #{i+1}" if i == 0 else None,
            )

        for i in range(pos_bm.shape[0]):
            ax.plot(
                t_eval, pos_bm[i, :, 0],
                "--", color=bm_colors[i], lw=1.0, alpha=0.7,
                label="BM" if i == 0 else None,
            )

        ax.set_xlabel("Time $t$", color=TEXT_C, fontsize=11)
        ax.set_ylabel("$x(t)$", color=TEXT_C, fontsize=11)
        ax.set_title(
            f"Trajectories: CTRW ($\\alpha={alpha:.2f}$) vs BM",
            color=TEXT_C, fontsize=11,
        )
        ax.tick_params(colors=TEXT_C)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.5), facecolor=BG)
        ax2.set_facecolor(PANEL)

        w = waiting_samples[waiting_samples >= 1.0]
        if len(w) > 10:
            bins = np.logspace(np.log10(max(1.0, w.min())), np.log10(w.max()), 40)
            counts, edges = np.histogram(w, bins=bins)
            centres = 0.5 * (edges[:-1] + edges[1:])
            mask = counts > 0
            ax2.scatter(centres[mask], counts[mask], s=12, color=THEO_C, alpha=0.8,
                        label="Empirical")

            # Power-law fit reference
            log_c = np.log(centres[mask])
            log_n = np.log(counts[mask])
            fit_mask = log_c > np.log(10)
            if fit_mask.sum() >= 3:
                coeff = np.polyfit(log_c[fit_mask], log_n[fit_mask], 1)
                slope = coeff[0]
                xfit = np.logspace(np.log10(10), np.log10(w.max()), 50)
                ax2.plot(xfit, np.exp(np.polyval(coeff, np.log(xfit))),
                         "--", color=MSD_C, lw=2,
                         label=f"Fit slope = {slope:.2f}\n(theory: $-\\alpha - 1 = {-alpha-1:.2f}$)")

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Waiting time $W$", color=TEXT_C, fontsize=10)
        ax2.set_ylabel("Count", color=TEXT_C, fontsize=10)
        ax2.set_title("Waiting-time distribution", color=TEXT_C, fontsize=10)
        ax2.tick_params(colors=TEXT_C)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# ---------------------------------------------------------------------------
# Experiment 2: MSD and non-Gaussian parameter
# ---------------------------------------------------------------------------

def _run_msd(alpha: float, n_walkers: int, n_jumps: int) -> None:
    with st.spinner(f"Running CTRW ({n_walkers:,} walkers, {n_jumps:,} jumps)…"):
        rng = np.random.default_rng(42)
        t_eval = np.unique(
            np.round(np.logspace(0, np.log10(500), 60))
        ).astype(np.float64)

        pos_ctrw = ctrw_positions(n_walkers, n_jumps, alpha, t_eval, rng)
        pos_bm = brownian_positions(n_walkers, t_eval, rng)

    # --- MSD ---
    def _msd(pos):
        disp = pos - pos[:, :1, :]
        return np.mean(np.sum(disp ** 2, axis=-1), axis=0)  # (T,)

    def _ng(pos):
        disp = pos - pos[:, :1, :]
        r2 = np.sum(disp ** 2, axis=-1)  # (n_walkers, T)
        r4 = r2 ** 2
        msd_t = r2.mean(axis=0)
        mr4_t = r4.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ng = np.where(msd_t > 1e-12, mr4_t / (2.0 * msd_t ** 2) - 1.0, 0.0)
        return ng

    msd_ctrw = _msd(pos_ctrw)
    msd_bm = _msd(pos_bm)
    ng_ctrw = _ng(pos_ctrw)
    ng_bm = _ng(pos_bm)

    # Fit slopes
    def _fit_slope(t, y):
        mask = (t > 5) & (y > 1e-12)
        if mask.sum() < 4:
            return float("nan")
        return float(np.polyfit(np.log(t[mask]), np.log(y[mask]), 1)[0])

    slope_ctrw = _fit_slope(t_eval, msd_ctrw)
    slope_bm = _fit_slope(t_eval, msd_bm)

    result = {
        "alpha": alpha,
        "t_eval": t_eval,
        "msd_ctrw": msd_ctrw,
        "msd_bm": msd_bm,
        "ng_ctrw": ng_ctrw,
        "ng_bm": ng_bm,
        "slope_ctrw": slope_ctrw,
        "slope_bm": slope_bm,
    }
    st.session_state["ch13_result"] = result
    _show_msd(result)


def _show_msd(res: dict) -> None:
    t_eval = res["t_eval"]
    alpha = res["alpha"]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=BG)
        ax.set_facecolor(PANEL)

        ax.plot(t_eval, res["msd_ctrw"], color=THEO_C, lw=2,
                label=f"CTRW ($\\alpha={alpha:.2f}$)")
        ax.plot(t_eval, res["msd_bm"], color=MSD_C, lw=2, linestyle="--",
                label="BM (normal)")

        # Theory reference lines
        if not np.isnan(res["slope_ctrw"]):
            idx_ref = len(t_eval) // 4
            t_ref = t_eval[idx_ref:]
            scale = res["msd_ctrw"][idx_ref] / (t_eval[idx_ref] ** alpha)
            ax.plot(t_ref, scale * t_ref ** alpha, ":", color=RMS_C, lw=1.5,
                    label=f"Theory $t^{{{alpha:.2f}}}$")
            ax.annotate(
                f"slope = {res['slope_ctrw']:.2f}",
                xy=(t_eval[len(t_eval) * 3 // 4], res["msd_ctrw"][len(t_eval) * 3 // 4]),
                xytext=(10, 10), textcoords="offset points",
                color=THEO_C, fontsize=9,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time $t$", color=TEXT_C, fontsize=11)
        ax.set_ylabel(r"$\langle r^2(t) \rangle$", color=TEXT_C, fontsize=11)
        ax.set_title("MSD: CTRW vs Brownian motion", color=TEXT_C, fontsize=11)
        ax.tick_params(colors=TEXT_C)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5), facecolor=BG)
        ax2.set_facecolor(PANEL)

        ax2.plot(t_eval, res["ng_ctrw"], color=THEO_C, lw=2,
                 label=f"CTRW ($\\alpha={alpha:.2f}$)")
        ax2.plot(t_eval, res["ng_bm"], color=MSD_C, lw=2, linestyle="--",
                 label="BM")
        ax2.axhline(0, color=TEXT_C, lw=0.8, alpha=0.5)

        ax2.set_xscale("log")
        ax2.set_xlabel("Time $t$", color=TEXT_C, fontsize=11)
        ax2.set_ylabel(
            r"$\alpha_2(t) = \langle r^4 \rangle / (2\langle r^2 \rangle^2) - 1$",
            color=TEXT_C, fontsize=10,
        )
        ax2.set_title("Non-Gaussian parameter", color=TEXT_C, fontsize=11)
        ax2.tick_params(colors=TEXT_C)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.25)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.metric(
            "CTRW MSD slope",
            f"{res['slope_ctrw']:.3f}" if not np.isnan(res["slope_ctrw"]) else "N/A",
            help=f"Theory: α = {alpha:.2f}",
        )
    with mc2:
        st.metric(
            "BM MSD slope",
            f"{res['slope_bm']:.3f}" if not np.isnan(res["slope_bm"]) else "N/A",
            help="Theory: 1.0 (normal diffusion)",
        )
