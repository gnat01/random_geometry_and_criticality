"""Chapter C — ML on synthetic random-walk features."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..ml.synthetic import LABELS, SyntheticConfig, generate_labeled_batch, train_classifier
from ..theme import TEXT_C, apply_mpl_style, fig_bg


def render_chapter_c():
    apply_mpl_style()
    st.markdown(
        "## Chapter C — Machine learning on synthetic physics\n"
        "Each **row** is one random graph (carpet, Vicsek, or percolation) plus a **short SRW** simulation. "
        "Features summarize size, degree, diffusion slopes, and entropy of π. "
        "The **label** is the geometry family — a toy inverse problem: can a classifier recover the source?"
    )

    with st.sidebar:
        n_samples = st.slider("Dataset size", 60, 800, 200, 20)
        seed = st.number_input("Global seed", 0, 999999, 42)
        depth_min = st.slider("Depth min (recursive)", 2, 4, 2)
        depth_max = st.slider("Depth max (recursive)", 2, 5, 4)
        n_walkers = st.slider("Walkers per sample", 40, 200, 80)
        n_steps = st.slider("Steps per sample", 200, 1200, 400)
        noise = st.slider(
            "Measurement noise (feature uncertainty)",
            0.0,
            1.5,
            0.8,
            0.05,
        )
        test_frac = st.slider("Test fraction", 0.15, 0.4, 0.25)

    cfg = SyntheticConfig(
        n_samples=n_samples,
        seed=int(seed),
        depth_min=int(depth_min),
        depth_max=max(int(depth_min), int(depth_max)),
        n_walkers=int(n_walkers),
        n_steps=int(n_steps),
        feature_noise=float(noise),
    )

    if st.button("Generate & train", type="primary", key="train_c"):
        with st.spinner("Building synthetic dataset…"):
            X, y, names = generate_labeled_batch(cfg)
        if len(y) < 30:
            st.error("Too few valid samples — relax depth range or increase attempts.")
            return
        class_counts = np.bincount(y, minlength=3)
        st.info(f"Using **{len(y)}** labeled samples · {X.shape[1]} features")
        st.caption(
            "Class counts: "
            + ", ".join(f"{LABELS[i]}={int(class_counts[i])}" for i in range(3))
        )

        with st.spinner("Training RandomForest…"):
            out = train_classifier(X, y, names, test_size=float(test_frac), seed=int(seed))

        st.text(out["classification_report"])

        cm = out["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor=fig_bg())
        ax.set_facecolor(fig_bg())
        im = ax.imshow(cm, cmap="magma", aspect="auto")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(LABELS, rotation=35, ha="right")
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white", fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Confusion matrix (test)", color=TEXT_C)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        feats = sorted(out["feature_importances"].items(), key=lambda x: -x[1])
        labels = [f[0] for f in feats]
        vals = [f[1] for f in feats]
        fig2, ax2 = plt.subplots(figsize=(5, 3.2), facecolor=fig_bg())
        ax2.set_facecolor(fig_bg())
        ax2.barh(labels[::-1], vals[::-1], color="#4a9fd4")
        ax2.set_title("Feature importances", color=TEXT_C)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.caption(
            "Slopes are noisy on small graphs; increasing **steps** and **samples** usually helps. "
            "See THEORY.md for identifiability caveats."
        )
