"""Chapter VI: The Inverse Problem (ML)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..ml.synthetic import SyntheticConfig, generate_labeled_batch, train_classifier, LABELS
from ..theme import BG, THEO_C, TEXT_C, PANEL, MSD_C, RMS_C
from .nav import chapter_header, key_result, definition_box, prev_next, mark_visited


_LABEL_COLORS = ["#ff6b6b", "#a78bfa", "#4a9fd4"]


def render() -> None:
    mark_visited("chapter_6")
    chapter_header(
        "VI",
        "The Inverse Problem",
        "Can a machine learn the geometry from dynamics?",
    )

    st.markdown("""
Everything so far has been the **forward problem**: given a geometry, compute the dynamics.
Carpet → $d_w \\approx 2.88$. Percolation at $p_c$ → anomalous diffusion. DLA → branching.

The **inverse problem** asks the opposite: given only the walker's statistics, can you
reconstruct what kind of geometry it was walking on?

This is a real question in biophysics and materials science. You can track a single
molecule inside a cell and measure its MSD, return probability, and degree-distribution
entropy. But you cannot directly image the network it's diffusing on. Can the walk statistics
alone tell you whether the substrate is a compact gel, a fractal aggregate, or something
random and disordered?

Here we set up a synthetic experiment. We generate labeled graphs from three families —
Sierpiński carpet, Vicsek fractal, bond percolation — run short random walks on each, and
extract five physics-inspired features:

| Feature | Meaning |
|---|---|
| `log_n_nodes` | log of graph size |
| `mean_degree` | average connectivity |
| `rms_loglog_slope` | walk dimension proxy ($1/d_w$) |
| `msd_loglog_slope` | walk dimension proxy (MSD) |
| `pi_entropy` | entropy of stationary distribution |

A random forest classifier then tries to identify the geometry family from these five
numbers. Crucially, we add realistic measurement noise to the features, so the problem is
genuinely hard.
""")

    definition_box(
        "Stationary distribution entropy",
        "For a random walk on an undirected graph, the stationary distribution is "
        "$\\pi(v) = \\deg(v) / (2|E|)$. Its Shannon entropy "
        "$H = -\\sum_v \\pi(v) \\log \\pi(v)$ is high for regular graphs and lower for "
        "graphs with hubs — it encodes the degree heterogeneity."
    )

    st.markdown("""
---

### Experiment: generate, train, and evaluate

We generate 120 labeled samples (40 per class), train the classifier on 75% of them, and
test on the rest. The confusion matrix shows where the classifier makes mistakes — and
the feature importances reveal which walk statistics carry the most discriminating power.
""")

    n_samples = st.slider("Total samples", 60, 200, 120, step=20, key="ch6_samples")
    n_walkers = st.slider("Walkers per graph", 20, 100, 50, step=10, key="ch6_walkers")
    n_steps = st.slider("Steps per walker", 100, 500, 250, step=50, key="ch6_steps")

    if st.button("▶ Run ML experiment", key="ch6_run"):
        _run_ml(n_samples, n_walkers, n_steps)
    elif "ch6_ml" in st.session_state:
        _show_ml(st.session_state["ch6_ml"])

    st.markdown("""
---

### Reflection

Look at the confusion matrix carefully. The classifier almost never confuses carpet with
Vicsek — their MSD slopes differ enough. But carpet and percolation at $p \\approx p_c$
are similar: both are fractal with nearly the same $d_f$ and $d_w$. The classifier must
rely on structural features (mean degree, entropy) to tell them apart.

The feature importances typically show that `msd_loglog_slope` and `rms_loglog_slope`
are the most informative. This is exactly what you would expect from the physics: the
walk dimension is the dynamical fingerprint of the geometry. The entropy and degree
features are secondary but help break degeneracies.

This is the final chapter's message: dynamics encodes geometry. A random walker is not
just a particle diffusing on a substrate — it is a *probe* of that substrate's fractal
structure. And a sufficiently clever observer can read the geometry from the walk.
""")

    key_result(
        "The MSD and RMS log-log slopes are the most informative features for "
        "geometry classification. This is because $d_w$ is a direct dynamical "
        "fingerprint of the Hausdorff dimension and connectivity of the substrate."
    )

    st.markdown("---")
    prev_next("chapter_6")


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def _run_ml(n_samples: int, n_walkers: int, n_steps: int) -> None:
    with st.spinner(f"Generating {n_samples} labeled graphs and training classifier…"):
        cfg = SyntheticConfig(
            n_samples=n_samples,
            seed=42,
            n_walkers=n_walkers,
            n_steps=n_steps,
        )
        X, y, feature_names = generate_labeled_batch(cfg)
        ml_result = train_classifier(X, y, feature_names)
        ml_result["feature_names"] = feature_names
        ml_result["n_train"] = int(len(y) * 0.75)
        ml_result["n_test"] = len(y) - int(len(y) * 0.75)

    st.session_state["ch6_ml"] = ml_result
    _show_ml(ml_result)


def _show_ml(result: dict) -> None:
    cm = result["confusion_matrix"]
    importances = result["feature_importances"]
    feature_names = result.get("feature_names", list(importances.keys()))
    report = result["classification_report"]

    col1, col2 = st.columns([1, 1])

    # --- Left: confusion matrix ---
    with col1:
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5), facecolor=BG)
        ax_cm.set_facecolor(PANEL)

        # Normalise by row for display
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

    # --- Right: feature importances ---
    with col2:
        imp_vals = np.array([importances[f] for f in feature_names])
        idx_sort = np.argsort(imp_vals)[::-1]

        fig_fi, ax_fi = plt.subplots(figsize=(5.5, 4.5), facecolor=BG)
        ax_fi.set_facecolor(PANEL)

        bars = ax_fi.barh(
            range(len(feature_names)),
            imp_vals[idx_sort],
            color=THEO_C,
            alpha=0.85,
        )
        ax_fi.set_yticks(range(len(feature_names)))
        _display_names = {
            "log_n_nodes": "log(N nodes)",
            "mean_degree": "mean degree",
            "rms_loglog_slope": "RMS log-log slope",
            "msd_loglog_slope": "MSD log-log slope",
            "pi_entropy": "π entropy",
        }
        ax_fi.set_yticklabels(
            [_display_names.get(feature_names[i], feature_names[i]) for i in idx_sort],
            color=TEXT_C,
            fontsize=10,
        )
        ax_fi.set_xlabel("Importance", color=TEXT_C, fontsize=11)
        ax_fi.set_title("Feature importances", color=TEXT_C, fontsize=11)
        ax_fi.grid(True, axis="x", alpha=0.3)

        st.pyplot(fig_fi, use_container_width=True)
        plt.close(fig_fi)

    # Classification report
    st.markdown("**Classification report**")
    st.code(report, language="")

    # Per-class accuracy from diagonal
    if cm.sum() > 0:
        per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(1)
        overall_acc = cm.diagonal().sum() / cm.sum()
        cols = st.columns(4)
        for col, (label, acc) in zip(cols[:3], zip(LABELS, per_class_acc)):
            with col:
                st.metric(f"{label} accuracy", f"{acc*100:.1f}%")
        with cols[3]:
            st.metric("Overall accuracy", f"{overall_acc*100:.1f}%")
