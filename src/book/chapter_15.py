"""Chapter XV: Distinguishing Anomalous Diffusion."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..sim.anomalous import (
    generate_anomalous_dataset,
    train_anomalous_classifier,
    LABELS,
)
from ..theme import BG, THEO_C, TEXT_C, PANEL, MSD_C, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited


_LABEL_COLORS = ["#ff6b6b", "#a78bfa", "#4a9fd4"]


def render() -> None:
    mark_visited("chapter_15")
    chapter_header(
        "XV",
        "Distinguishing Anomalous Diffusion",
        "Three mechanisms, one MSD exponent — can a classifier tell them apart?",
    )

    st.markdown(r"""
Chapters II, XIII, and XIV introduced three distinct mechanisms that all produce
MSD $\sim t^\alpha$ with $\alpha < 1$:

| Mechanism | Origin of anomaly |
|---|---|
| Fractal substrate (Ch. II) | Geometric disorder — bottlenecks and dead ends |
| CTRW (Ch. XIII) | Temporal disorder — heavy-tailed waiting times |
| fBm (Ch. XIV) | Increment correlations — viscoelastic memory |

In a biophysics experiment you observe a trajectory and want to determine the **mechanism**,
not just the exponent. This is the central challenge of **single-particle tracking** (SPT):
a fluorescently labelled protein inside a cell gives you a noisy 2-D or 3-D trajectory.
You fit an MSD and find $\alpha \approx 0.7$ — but is the anomaly from a fractal cytoskeletal
network? From transient binding (CTRW)? From viscoelastic drag (fBm)?

The exponent alone cannot answer this. All three mechanisms can be tuned to give exactly
the same $\alpha$. You need additional statistical diagnostics.

Chapter VI showed that machine learning can distinguish geometry families from walk statistics.
Here we apply the same idea at the mechanistic level. We define five features — each
inspired by a theoretical property of the mechanisms — and train a random forest to classify
trajectories by mechanism, even when the MSD exponent is identical across all three classes.

**Feature summary:**

| Feature | Fractal | CTRW | fBm |
|---|---|---|---|
| MSD slope | $\approx 0.70$ | $\approx 0.70$ | $\approx 0.70$ |
| Displacement Gaussian? | No | No | **YES** |
| Ergodic? | Yes | **No** | Yes |
| VACF lag 1 | $\approx 0$ | $\approx 0$ | **Negative** ($H < 0.5$) |
| Kurtosis | Elevated | Elevated | Near 0 |

The non-Gaussian parameter separates fBm from the others. The ergodicity ratio separates
CTRW from fractal diffusion. The VACF at lag 1 provides a second handle on fBm. With all
five features together, the classifier can achieve $> 85\%$ accuracy — demonstrating that
mechanism identification is feasible even when the MSD exponent is degenerate.
""")

    definition_box(
        "Ergodicity",
        r"A stochastic process is ergodic if time averages equal ensemble averages. "
        r"For a random walk: the time-averaged MSD "
        r"$\bar{\delta}^2(\tau) = \frac{1}{T-\tau}\int_0^{T-\tau}|x(t+\tau)-x(t)|^2\,dt$ "
        r"should equal the ensemble MSD $\langle r^2(\tau)\rangle$. "
        r"CTRW violates this: $\bar{\delta}^2(\tau) \propto \tau^\alpha / T^{1-\alpha}$, "
        r"which depends on the total observation time $T$ even as $T \to \infty$.",
    )

    st.markdown("""
---

### Experiment: generate, classify, evaluate
""")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        n_per_class = st.slider(
            "Samples per class", 20, 80, 40, step=10, key="ch15_npc"
        )
    with col_b:
        n_walkers = st.slider(
            "Walkers per sample", 100, 400, 200, step=50, key="ch15_nw"
        )
    with col_c:
        n_steps = st.slider(
            "Steps per walker", 200, 500, 300, step=50, key="ch15_ns"
        )

    noise_scale = st.slider(
        "Feature noise level (0 = clean, 0.5 = realistic, 1.0 = heavy)",
        min_value=0.0, max_value=1.5, value=0.5, step=0.05, key="ch15_noise",
        help="Gaussian noise added to each feature at this multiple of its dataset std. "
             "Mimics finite-sample measurement error.",
    )

    if st.button("▶ Run experiment", key="ch15_run"):
        _run_experiment(n_per_class, n_walkers, n_steps, noise_scale)
    elif "ch15_result" in st.session_state:
        _show_result(st.session_state["ch15_result"])

    key_result(
        "MSD slope alone gives ~33% accuracy (random chance — all three mechanisms are "
        "tuned to the same exponent). The non-Gaussian parameter separates fBm from the "
        "other two. The ergodicity ratio separates CTRW from fractal diffusion. With all "
        "five features the classifier reaches >85% accuracy — demonstrating that mechanism "
        "identification is possible even when the MSD exponent is identical."
    )

    st.markdown("---")
    prev_next("chapter_15")


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def _run_experiment(n_per_class: int, n_walkers: int, n_steps: int,
                    noise_scale: float = 0.5) -> None:
    with st.spinner(
        f"Generating {n_per_class * 3} labeled trajectory batches "
        f"({n_walkers} walkers × {n_steps} steps each)…"
    ):
        X, y, feature_names = generate_anomalous_dataset(
            n_per_class=n_per_class,
            n_walkers=n_walkers,
            n_steps=n_steps,
            alpha_ctrw=0.70,
            H_fbm=0.35,
            noise_scale=noise_scale,
            seed=42,
        )

    with st.spinner("Training classifier…"):
        ml = train_anomalous_classifier(X, y, feature_names)
        ml["feature_names"] = feature_names
        ml["X"] = X
        ml["y"] = y

    st.session_state["ch15_result"] = ml
    _show_result(ml)


def _show_result(result: dict) -> None:
    cm = result["confusion_matrix"]
    importances = result["feature_importances"]
    feature_names = result["feature_names"]
    X = result["X"]
    y = result["y"]

    st.markdown("#### Feature scatter plots")
    _show_scatter(X, y, feature_names)

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    # --- Confusion matrix ---
    with col1:
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5), facecolor=BG)
        ax_cm.set_facecolor(PANEL)

        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_pct = cm_norm / row_sums

        im = ax_cm.imshow(cm_pct, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

        for i in range(3):
            for j in range(3):
                ax_cm.text(
                    j, i,
                    f"{cm[i, j]}\n({cm_pct[i, j]*100:.0f}%)",
                    ha="center", va="center", fontsize=11,
                    color="black" if cm_pct[i, j] > 0.5 else TEXT_C,
                )

        ax_cm.set_xticks(range(3))
        ax_cm.set_yticks(range(3))
        ax_cm.set_xticklabels(list(LABELS), color=TEXT_C, fontsize=10)
        ax_cm.set_yticklabels(list(LABELS), color=TEXT_C, fontsize=10)
        ax_cm.set_xlabel("Predicted", color=TEXT_C, fontsize=11)
        ax_cm.set_ylabel("True", color=TEXT_C, fontsize=11)
        ax_cm.set_title("Confusion matrix", color=TEXT_C, fontsize=11)

        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    # --- Feature importances ---
    with col2:
        imp_vals = np.array([importances[f] for f in feature_names])
        idx_sort = np.argsort(imp_vals)

        _display_names = {
            "msd_slope": "MSD slope",
            "nongaussian": "Non-Gaussian α₂",
            "ergodicity": "Ergodicity ratio",
            "vacf_lag1": "VACF lag 1",
            "kurtosis": "Kurtosis",
        }

        fig_fi, ax_fi = plt.subplots(figsize=(5.5, 4.5), facecolor=BG)
        ax_fi.set_facecolor(PANEL)
        ax_fi.barh(
            range(len(feature_names)),
            imp_vals[idx_sort],
            color=THEO_C, alpha=0.85,
        )
        ax_fi.set_yticks(range(len(feature_names)))
        ax_fi.set_yticklabels(
            [_display_names.get(feature_names[i], feature_names[i]) for i in idx_sort],
            color=TEXT_C, fontsize=10,
        )
        ax_fi.set_xlabel("Importance", color=TEXT_C, fontsize=11)
        ax_fi.set_title("Feature importances", color=TEXT_C, fontsize=11)
        ax_fi.grid(True, axis="x", alpha=0.3)

        st.pyplot(fig_fi, use_container_width=True)
        plt.close(fig_fi)

    # --- Classification report ---
    st.markdown("**Classification report**")
    st.code(result["classification_report"], language="")

    # --- Single-feature accuracy table ---
    st.markdown("**Single-feature accuracy vs all 5 features**")
    _display_names = {
        "msd_slope": "MSD slope",
        "nongaussian": "Non-Gaussian α₂",
        "ergodicity": "Ergodicity ratio",
        "vacf_lag1": "VACF lag 1",
        "kurtosis": "Kurtosis",
    }
    sf_acc = result["accuracy_1feat"]
    all_acc = result["accuracy_all"]

    acc_cols = st.columns(len(feature_names) + 1)
    for col, fname in zip(acc_cols[:-1], feature_names):
        with col:
            st.metric(
                _display_names.get(fname, fname),
                f"{sf_acc[fname]*100:.1f}%",
            )
    with acc_cols[-1]:
        st.metric("All 5 features", f"{all_acc*100:.1f}%")

    # --- Per-class metrics row ---
    st.markdown("---")
    if cm.sum() > 0:
        per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(1)
        overall_acc = cm.diagonal().sum() / cm.sum()
        m_cols = st.columns(4)
        for col, (label, acc) in zip(m_cols[:3], zip(LABELS, per_class_acc)):
            with col:
                st.metric(f"{label} accuracy", f"{acc*100:.1f}%")
        with m_cols[3]:
            st.metric("Overall accuracy", f"{overall_acc*100:.1f}%")


def _show_scatter(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
    fi = {name: i for i, name in enumerate(feature_names)}

    pairs = [
        ("nongaussian", "ergodicity"),
        ("vacf_lag1", "nongaussian"),
    ]
    xlabels = {
        "nongaussian": "Non-Gaussian α₂",
        "vacf_lag1": "VACF lag 1",
        "ergodicity": "Ergodicity ratio",
    }
    ylabels = {
        "ergodicity": "Ergodicity ratio",
        "nongaussian": "Non-Gaussian α₂",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=BG)

    for ax, (xfeat, yfeat) in zip(axes, pairs):
        ax.set_facecolor(PANEL)
        xi = fi[xfeat]
        yi = fi[yfeat]
        for label_idx, (label, color) in enumerate(zip(LABELS, _LABEL_COLORS)):
            mask = y == label_idx
            ax.scatter(
                X[mask, xi], X[mask, yi],
                c=color, s=18, alpha=0.7, label=label, edgecolors="none",
            )
        ax.set_xlabel(xlabels.get(xfeat, xfeat), color=TEXT_C, fontsize=10)
        ax.set_ylabel(ylabels.get(yfeat, yfeat), color=TEXT_C, fontsize=10)
        ax.set_title(
            f"{xlabels.get(xfeat, xfeat)} vs {ylabels.get(yfeat, yfeat)}",
            color=TEXT_C, fontsize=10,
        )
        ax.tick_params(colors=TEXT_C, labelsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
