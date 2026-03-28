"""Navigation shell, CSS, session state, and chapter header helpers."""

from __future__ import annotations

import streamlit as st

from ..theme import BG, TEXT_C, THEO_C

# ---------------------------------------------------------------------------
# Chapter registry
# ---------------------------------------------------------------------------

CHAPTERS = [
    {
        "id": "prologue",
        "number": None,
        "title": "Prologue",
        "subtitle": "Geometry, randomness, and the edge of order",
    },
    {
        "id": "chapter_1",
        "number": "I",
        "title": "The Geometry of Fractals",
        "subtitle": "Self-similarity, Hausdorff dimension, and two canonical examples",
    },
    {
        "id": "chapter_2",
        "number": "II",
        "title": "Walking on a Fractal",
        "subtitle": "Anomalous diffusion and the walk dimension",
    },
    {
        "id": "chapter_3",
        "number": "III",
        "title": "Disorder and the Random Substrate",
        "subtitle": "Percolation as a random fractal",
    },
    {
        "id": "chapter_4",
        "number": "IV",
        "title": "The Critical Point",
        "subtitle": "Phase transitions, exact exponents, and scaling collapse",
    },
    {
        "id": "chapter_5",
        "number": "V",
        "title": "Growth as Fractal",
        "subtitle": "Diffusion-limited aggregation and fractal dimension from growth",
    },
    {
        "id": "chapter_6",
        "number": "VI",
        "title": "The Inverse Problem",
        "subtitle": "Can a machine learn the geometry from dynamics?",
    },
    {
        "id": "chapter_7",
        "number": "VII",
        "title": "The Sound of a Fractal",
        "subtitle": "Laplacian spectrum, density of states, and the spectral dimension",
    },
    {
        "id": "chapter_8",
        "number": "VIII",
        "title": "First-Passage Processes",
        "subtitle": "How long does it take a random walker to find its target?",
    },
    {
        "id": "chapter_13",
        "number": "XIII",
        "title": "Continuous-Time Random Walks",
        "subtitle": "Subdiffusion from temporal disorder",
    },
    {
        "id": "chapter_14",
        "number": "XIV",
        "title": "Fractional Brownian Motion",
        "subtitle": "Anomalous diffusion from correlated increments",
    },
    {
        "id": "chapter_15",
        "number": "XV",
        "title": "Distinguishing Anomalous Diffusion",
        "subtitle": "Three mechanisms, one MSD exponent",
    },
    {
        "id": "chapter_9",
        "number": "IX",
        "title": "Three-Dimensional Percolation",
        "subtitle": "Beyond the exact solution — numerical exponents and FSS in 3-D",
    },
    {
        "id": "chapter_10",
        "number": "X",
        "title": "The Ising Model",
        "subtitle": "Wolff clusters, exact 2-D exponents, and the Binder cumulant",
    },
    {
        "id": "chapter_11",
        "number": "XI",
        "title": "Why Universality Exists",
        "subtitle": "Real-space renormalisation group and the critical fixed point",
    },
    {
        "id": "chapter_12",
        "number": "XII",
        "title": "The Transfer Matrix",
        "subtitle": "Exact finite-strip solution and noise-free T_c extraction",
    },
    {
        "id": "chapter_18",
        "number": "XVIII",
        "title": "Directed Percolation",
        "subtitle": "Time-directed connectivity and the DP universality class",
    },
    {
        "id": "problems",
        "number": None,
        "title": "Problems",
        "subtitle": "Coding challenges and experiments",
    },
    {
        "id": "epilogue",
        "number": None,
        "title": "Epilogue",
        "subtitle": "The full arc: from geometry to criticality and back",
    },
]

_CHAPTER_IDS = [c["id"] for c in CHAPTERS]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state(default_chapter: str = "prologue") -> None:
    """Initialise navigation state on first load."""
    if "chapter" not in st.session_state:
        st.session_state.chapter = default_chapter
    if "visited" not in st.session_state:
        st.session_state.visited = set()


def mark_visited(chapter_id: str) -> None:
    st.session_state.visited.add(chapter_id)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_SIDEBAR_CSS = f"""
<style>
[data-testid="stSidebar"] {{
    background-color: {BG};
}}
.book-nav-item {{
    display: block;
    padding: 6px 10px 6px 14px;
    margin: 2px 0;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.88rem;
    color: {TEXT_C};
    border-left: 3px solid transparent;
    text-decoration: none;
    line-height: 1.35;
}}
.book-nav-item:hover {{
    background: rgba(255,215,0,0.07);
    border-left-color: {THEO_C};
}}
.book-nav-active {{
    background: rgba(255,215,0,0.12);
    border-left-color: {THEO_C};
    color: {THEO_C};
    font-weight: 600;
}}
.book-nav-num {{
    font-size: 0.72rem;
    opacity: 0.55;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1px;
}}
.book-nav-title {{
    font-size: 0.88rem;
}}
.book-nav-check {{
    float: right;
    color: #4caf50;
    font-size: 0.75rem;
    margin-top: 2px;
}}
</style>
"""


def render_sidebar() -> None:
    """Render the table of contents in the sidebar."""
    with st.sidebar:
        st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{THEO_C}; font-size:0.78rem; letter-spacing:0.12em;"
            f" text-transform:uppercase; font-weight:700; padding:12px 0 8px 14px;'>"
            f"Random Geometry<br>&amp; Criticality</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        current = st.session_state.get("chapter", "prologue")
        visited = st.session_state.get("visited", set())

        for ch in CHAPTERS:
            cid = ch["id"]
            is_active = cid == current
            is_visited = cid in visited

            num_label = f"Chapter {ch['number']}" if ch["number"] else ""
            check = "✓" if is_visited else ""
            active_cls = "book-nav-active" if is_active else ""

            label = f"""
<div class='book-nav-item {active_cls}'>
  <div class='book-nav-num'>{num_label}<span class='book-nav-check'>{check}</span></div>
  <div class='book-nav-title'>{ch['title']}</div>
</div>
"""
            st.markdown(label, unsafe_allow_html=True)
            if not is_active:
                if st.button(
                    f"Go to {ch['title']}",
                    key=f"nav_{cid}",
                    use_container_width=True,
                    type="secondary",
                ):
                    st.session_state.chapter = cid
                    st.rerun()
            else:
                # Placeholder to keep spacing consistent
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chapter header helpers
# ---------------------------------------------------------------------------

def chapter_header(number: str | None, title: str, subtitle: str) -> None:
    """Render a styled chapter header with a gold left border."""
    num_html = (
        f"<div style='font-size:0.78rem; letter-spacing:0.12em; text-transform:uppercase;"
        f" color:{THEO_C}; opacity:0.75; margin-bottom:6px;'>Chapter {number}</div>"
        if number
        else f"<div style='font-size:0.78rem; letter-spacing:0.12em; text-transform:uppercase;"
             f" color:{THEO_C}; opacity:0.75; margin-bottom:6px;'>&nbsp;</div>"
    )
    st.markdown(
        f"""
<div style='border-left: 4px solid {THEO_C}; padding-left: 18px; margin-bottom: 28px;'>
  {num_html}
  <div style='font-size:2.1rem; font-weight:700; color:{TEXT_C}; line-height:1.15;
              margin-bottom:8px;'>{title}</div>
  <div style='font-size:1.05rem; font-style:italic; color:{TEXT_C}; opacity:0.65;'>{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def key_result(text: str) -> None:
    """Render a gold-bordered callout for key results."""
    st.markdown(
        f"<p style='font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; "
        f"color:{THEO_C}; font-weight:700; border-left: 4px solid {THEO_C}; "
        f"padding: 8px 14px; margin: 24px 0 2px 0;'>Key result</p>",
        unsafe_allow_html=True,
    )
    st.markdown(text)


def definition_box(label: str, text: str) -> None:
    """Render a blue-bordered callout for definitions."""
    st.markdown(
        f"<p style='font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; "
        f"color:#4a9fd4; font-weight:700; border-left: 4px solid #4a9fd4; "
        f"padding: 8px 14px; margin: 24px 0 2px 0;'>Definition</p>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{label}.** {text}")


# ---------------------------------------------------------------------------
# Prev / next navigation
# ---------------------------------------------------------------------------

def prev_next(current_id: str) -> None:
    """Render prev / next chapter buttons at the bottom of the page."""
    idx = _CHAPTER_IDS.index(current_id) if current_id in _CHAPTER_IDS else 0
    col1, col2 = st.columns(2)

    if idx > 0:
        prev = CHAPTERS[idx - 1]
        with col1:
            label = f"← {prev['title']}"
            if st.button(label, key=f"prev_{current_id}", use_container_width=True):
                st.session_state.chapter = prev["id"]
                st.rerun()

    if idx < len(CHAPTERS) - 1:
        nxt = CHAPTERS[idx + 1]
        with col2:
            label = f"{nxt['title']} →"
            if st.button(label, key=f"next_{current_id}", use_container_width=True):
                st.session_state.chapter = nxt["id"]
                st.rerun()
