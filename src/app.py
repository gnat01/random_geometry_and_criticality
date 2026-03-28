"""
Streamlit entry for Chapters A–C.

Prefer launching via `python -m src.cli chapter-a` (sets FRACTAL_* env vars),
or:  streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.theme import page_config, apply_mpl_style
from src.ui.chapter_a import render_chapter_a
from src.ui.chapter_b import render_chapter_b
from src.ui.chapter_c import render_chapter_c

page_config()
apply_mpl_style()

ch = os.environ.get("FRACTAL_CHAPTER", "A").strip().upper()
if ch == "B":
    render_chapter_b()
elif ch == "C":
    render_chapter_c()
else:
    render_chapter_a()
