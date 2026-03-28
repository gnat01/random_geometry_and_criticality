"""
Root entry point for the interactive book.

Launch via:
    streamlit run book.py
or:
    python -m src.cli book --port 8500
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# Must be the very first Streamlit call
st.set_page_config(
    page_title="Random Geometry & Criticality — An Interactive Book",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.theme import apply_mpl_style
from src.book.nav import init_session_state, render_sidebar, mark_visited
from src.book import prologue, chapter_1, chapter_2, chapter_3, chapter_4, chapter_5, chapter_6, chapter_7, chapter_8, chapter_9, chapter_10, chapter_11, chapter_12, chapter_13, chapter_14, chapter_15, chapter_18, problems, epilogue

apply_mpl_style()
_start_chapter = os.environ.get("BOOK_CHAPTER", "prologue")
init_session_state(default_chapter=_start_chapter)
render_sidebar()

_ROUTES = {
    "prologue": prologue.render,
    "chapter_1": chapter_1.render,
    "chapter_2": chapter_2.render,
    "chapter_3": chapter_3.render,
    "chapter_4": chapter_4.render,
    "chapter_5": chapter_5.render,
    "chapter_6": chapter_6.render,
    "chapter_7": chapter_7.render,
    "chapter_8": chapter_8.render,
    "chapter_13": chapter_13.render,
    "chapter_14": chapter_14.render,
    "chapter_15": chapter_15.render,
    "chapter_9": chapter_9.render,
    "chapter_10": chapter_10.render,
    "chapter_11": chapter_11.render,
    "chapter_12": chapter_12.render,
    "chapter_18": chapter_18.render,
    "problems": problems.render,
    "epilogue": epilogue.render,
}

chapter_id = st.session_state.get("chapter", "prologue")
render_fn = _ROUTES.get(chapter_id, prologue.render)
render_fn()
