"""Chapter X: The Ising Model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..sim.ising import (
    T_C, NU_ISING, BETA_ISING, GAMMA_ISING,
    temperature_scan,
    finite_size_collapse_ising,
    wolff_sweep,
    _random_spins,
)
from ..theme import BG, THEO_C, TEXT_C, PANEL, MSD_C, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited

_COLORS = ["#ff6b6b", "#a78bfa", "#4a9fd4", "#4caf50"]


def render() -> None:
    mark_visited("chapter_10")
    chapter_header(
        "X",
        "The Ising Model",
        "Magnetic phase transition, Wolff clusters, and exact 2-D exponents",
    )

    st.markdown(r"""
The Ising model is the hydrogen atom of statistical mechanics. Spins $\sigma_i \in \{-1, +1\}$
sit on a lattice and interact ferromagnetically with their neighbours:

$$\mathcal{H} = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j.$$

At low temperature the system is ordered (all spins aligned). At high temperature, thermal
fluctuations destroy the order. The transition occurs at the Curie temperature $T_c$.

**2-D exact solution (Onsager, 1944).** On the square lattice with periodic boundaries:

$$T_c = \frac{2J}{k_B \ln(1 + \sqrt{2})} \approx 2.2692 \quad (J = k_B = 1).$$

The exponents are exact rational fractions:

| Observable | Power law | Exponent |
|---|---|---|
| Magnetisation | $M \sim (T_c - T)^\beta$ | $\beta = 1/8$ |
| Susceptibility | $\chi \sim \|T - T_c\|^{-\gamma}$ | $\gamma = 7/4$ |
| Correlation length | $\xi \sim \|T - T_c\|^{-\nu}$ | $\nu = 1$ |

**Critical slowing-down.** Near $T_c$, the Metropolis single-spin-flip algorithm requires
$O(\xi^z)$ steps to decorrelate, with $z \approx 2.17$. The **Wolff algorithm** bypasses
this by flipping entire correlated clusters at once, achieving $z \approx 0.25$ — almost
no slowing-down.
""")

    definition_box(
        "Binder cumulant",
        r"$U_4 = 1 - \langle m^4 \rangle / (3 \langle m^2 \rangle^2)$. "
        r"For a Gaussian distribution $U_4 = 0$. For a two-peak distribution (ordered phase) "
        r"$U_4 \to 2/3$. Crucially, $U_4$ curves at different $L$ all cross at $T_c$ — "
        r"providing a clean, $L$-independent estimate of the critical temperature.",
    )

    st.markdown("""
---

### Experiment A — Temperature scan: M, χ, and Binder cumulant
""")
    _section_a()

    st.markdown(r"""
---

### Experiment B — FSS collapse of magnetisation

The magnetisation obeys the scaling form:

$$M(T, L) = L^{-\beta/\nu}\, f\!\left((T - T_c)\, L^{1/\nu}\right).$$

Plotting $M \cdot L^{\beta/\nu}$ against $(T - T_c) \cdot L^{1/\nu}$ collapses all sizes.
With exact exponents $\beta = 1/8$, $\nu = 1$, the collapse is exact in the thermodynamic limit.
""")
    _section_b()

    st.markdown("""
---

### Experiment C — Spin configuration viewer
""")
    _section_c()

    key_result(
        r"2-D Ising: $T_c = 2/\ln(1+\sqrt{2}) \approx 2.2692$ (exact). "
        r"Exponents $\beta=1/8$, $\nu=1$, $\gamma=7/4$ are exact. "
        r"Binder cumulant curves cross at $T_c$ regardless of $L$. "
        r"Wolff cluster flips eliminate critical slowing-down ($z \approx 0.25$)."
    )

    st.markdown("---")
    prev_next("chapter_10")


# ---------------------------------------------------------------------------
# Section A
# ---------------------------------------------------------------------------

def _section_a() -> None:
    col_a, col_b = st.columns(2)
    with col_a:
        L_str = st.selectbox("System sizes L", ["10, 16, 24", "8, 12, 20"], index=0, key="ch10_sizes")
    with col_b:
        n_meas = st.slider("Measurements per T", 20, 100, 40, step=20, key="ch10_nm")

    n_T = 18

    if st.button("▶ Run temperature scan", key="ch10a_run"):
        sizes = [int(x.strip()) for x in L_str.split(",")]
        T_values = np.linspace(1.5, 3.0, n_T)
        rng = np.random.default_rng(42)
        n_therm = 20
        with st.spinner(f"Running Wolff for L = {sizes} over {n_T} temperatures…"):
            sweep = temperature_scan(sizes, T_values, n_therm, n_meas, rng)
        st.session_state["ch10a_sweep"] = sweep
        _show_a(sweep)
    elif "ch10a_sweep" in st.session_state:
        _show_a(st.session_state["ch10a_sweep"])


def _show_a(sweep: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    for i, (L, data) in enumerate(sorted(sweep.items())):
        c = _COLORS[i % len(_COLORS)]
        axes[0].plot(data["T"], data["M"], "o-", color=c, lw=1.8, ms=4, label=f"$L={L}$")
        axes[1].plot(data["T"], data["chi"], "o-", color=c, lw=1.8, ms=4, label=f"$L={L}$")
        axes[2].plot(data["T"], data["U4"], "o-", color=c, lw=1.8, ms=4, label=f"$L={L}$")

    for ax in axes:
        ax.axvline(T_C, color=THEO_C, lw=1.5, ls="--", alpha=0.7, label=f"$T_c \\approx {T_C:.4f}$")
        ax.set_xlabel("$T$", color=TEXT_C, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("$\\langle |m| \\rangle$", color=TEXT_C, fontsize=12)
    axes[0].set_title("Magnetisation", color=TEXT_C, fontsize=11)
    axes[1].set_ylabel("$\\chi$", color=TEXT_C, fontsize=12)
    axes[1].set_title("Susceptibility", color=TEXT_C, fontsize=11)
    axes[2].set_ylabel("$U_4$", color=TEXT_C, fontsize=12)
    axes[2].set_title("Binder cumulant", color=TEXT_C, fontsize=11)
    axes[2].axhline(2.0 / 3.0, color=MSD_C, lw=1, ls=":", alpha=0.6, label="$2/3$ (ordered)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Susceptibility peak locations
    cols = st.columns(len(sweep))
    for col, (L, data) in zip(cols, sorted(sweep.items())):
        with col:
            peak_idx = int(np.argmax(data["chi"]))
            T_peak = float(data["T"][peak_idx])
            st.metric(f"$\\chi$ peak, $L={L}$", f"$T = {T_peak:.3f}$",
                      delta=f"{(T_peak - T_C)*1000:.1f}×10⁻³ from $T_c$",
                      delta_color="off")


# ---------------------------------------------------------------------------
# Section B
# ---------------------------------------------------------------------------

def _section_b() -> None:
    if st.button("▶ Run FSS collapse", key="ch10b_run"):
        sizes = [10, 16, 24]
        T_values = np.linspace(1.8, 2.8, 22)
        rng = np.random.default_rng(77)
        with st.spinner("Running Wolff for FSS collapse (L = 10, 16, 24)…"):
            sweep = temperature_scan(sizes, T_values, n_therm=20, n_meas=50, rng=rng)
            collapse = finite_size_collapse_ising(sweep)
        st.session_state["ch10b_sweep"] = sweep
        st.session_state["ch10b_collapse"] = collapse
        _show_b(sweep, collapse)
    elif "ch10b_sweep" in st.session_state:
        _show_b(st.session_state["ch10b_sweep"], st.session_state["ch10b_collapse"])


def _show_b(sweep: dict, collapse: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(PANEL)

    for i, (L, data) in enumerate(sorted(sweep.items())):
        c = _COLORS[i % len(_COLORS)]
        axes[0].plot(data["T"], data["M"], "o-", color=c, lw=1.8, ms=5, label=f"$L={L}$")

    axes[0].axvline(T_C, color=THEO_C, lw=1.5, ls="--", alpha=0.7, label=f"$T_c \\approx {T_C:.4f}$")
    axes[0].set_xlabel("$T$", color=TEXT_C, fontsize=12)
    axes[0].set_ylabel("$\\langle|m|\\rangle$", color=TEXT_C, fontsize=12)
    axes[0].set_title("Before collapse (raw)", color=TEXT_C, fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for i, (L, cdata) in enumerate(sorted(collapse.items())):
        c = _COLORS[i % len(_COLORS)]
        axes[1].plot(cdata["x"], cdata["y"], "o-", color=c, lw=1.8, ms=5, label=f"$L={L}$")

    axes[1].set_xlabel(
        f"$(T - T_c) \\cdot L^{{1/\\nu}}$  ($\\nu = {NU_ISING}$)",
        color=TEXT_C, fontsize=10,
    )
    axes[1].set_ylabel(
        f"$M \\cdot L^{{\\beta/\\nu}}$  ($\\beta = {BETA_ISING}$)",
        color=TEXT_C, fontsize=10,
    )
    axes[1].set_title(
        f"FSS collapse ($\\nu={NU_ISING}$, $\\beta={BETA_ISING}$)",
        color=TEXT_C, fontsize=11,
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section C
# ---------------------------------------------------------------------------

def _section_c() -> None:
    st.markdown("""
Watch the spin configuration evolve through the phase transition.
Below $T_c$: large ordered domains. At $T_c$: fractal domain walls.
Above $T_c$: disordered.
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        L_sp = st.slider("L", 32, 96, 48, step=16, key="ch10c_L")
    with col2:
        T_sp = st.slider("Temperature T", 1.5, 3.2, float(round(T_C, 2)), step=0.05, key="ch10c_T")
    with col3:
        n_sweeps = st.slider("Wolff sweeps", 10, 100, 30, step=10, key="ch10c_sw")

    if st.button("▶ Generate spin configuration", key="ch10c_run"):
        rng = np.random.default_rng(7)
        spins = _random_spins(L_sp, rng)
        with st.spinner(f"Running {n_sweeps} Wolff sweeps at T = {T_sp:.2f}…"):
            wolff_sweep(spins, T_sp, n_sweeps, rng)
        st.session_state["ch10c_spins"] = spins.copy()
        st.session_state["ch10c_T"] = T_sp
        _show_c(spins, T_sp)
    elif "ch10c_spins" in st.session_state:
        _show_c(st.session_state["ch10c_spins"], st.session_state["ch10c_T"])


def _show_c(spins: np.ndarray, T: float) -> None:
    L = spins.shape[0]
    m = float(spins.mean())

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(spins, cmap="RdYlBu", vmin=-1, vmax=1, interpolation="nearest")
    label = "ordered" if T < T_C - 0.05 else ("critical" if abs(T - T_C) < 0.15 else "disordered")
    ax.set_title(
        f"$T = {T:.2f}$  ({label})  $|m| = {abs(m):.3f}$",
        color=TEXT_C, fontsize=11,
    )
    ax.axis("off")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Magnetisation |m|", f"{abs(m):.3f}")
    with col2:
        st.metric("Temperature T", f"{T:.3f}")
    with col3:
        st.metric("T_c (exact)", f"{T_C:.4f}")
