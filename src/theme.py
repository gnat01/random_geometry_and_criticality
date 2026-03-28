"""Shared dark theme for Matplotlib + Streamlit page config."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

BG = "#0e1117"
PANEL = "#161b22"
EDGE_C = "#2a5f8f"
NODE_C = "#4a9fd4"
WALK_C = "#ff6b6b"
START_C = "#00e676"
TARGET_C = "#ffd700"
TRAP_C = "#ff4444"
RMS_C = "#ff6b6b"
MSD_C = "#a78bfa"
THEO_C = "#ffd700"
TEXT_C = "#dce1ec"


def page_config():
    st.set_page_config(
        page_title="Fractal walks · Chapters A–C",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_mpl_style():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#2c3040",
            "text.color": TEXT_C,
            "axes.labelcolor": TEXT_C,
            "xtick.color": TEXT_C,
            "ytick.color": TEXT_C,
            "grid.color": "#2c3040",
            "grid.alpha": 0.6,
            "legend.facecolor": "#1a1f2e",
            "legend.edgecolor": "#2c3040",
            "legend.labelcolor": TEXT_C,
        }
    )


def fig_bg():
    return BG
