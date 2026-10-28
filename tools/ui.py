"""Bagian tampilan bersama untuk semua halaman Coconut (memakai Streamlit).

Tujuan: setiap halaman cukup
    from tools import ui
    ui.setup_page(); ui.render_menu()
    uploaded = ui.uploader(); papers = ui.load(uploaded)
Tema/branding diatur di .streamlit/config.toml, bukan CSS bersandi
data-testid yang mudah rusak saat versi Streamlit berganti.
"""
from __future__ import annotations

import hashlib
import traceback

import streamlit as st

from . import loaders

TOOLS = [
    {"n": 1, "title": "Scattertext", "button": "Go to Scattertext",
     "path": "pages/1 Scattertext.py", "image": "images/scattertext.png", "howto": "Scattertext"},
    {"n": 2, "title": "Topic Modeling", "button": "Go to Topic Modeling",
     "path": "pages/2 Topic Modeling.py", "image": "images/topicmodeling.png", "howto": "Topic Modeling"},
    {"n": 3, "title": "Bidirected Network", "button": "Go to Bidirected Network",
     "path": "pages/3 Bidirected Network.py", "image": "images/bidirected.png", "howto": "Bidirected Network"},
    {"n": 4, "title": "Sunburst", "button": "Go to Sunburst Visualization",
     "path": "pages/4 Sunburst.py", "image": "images/sunburst.png", "howto": "Sunburst"},
    {"n": 5, "title": "Burst Detection", "button": "Go to Burst Detection",
     "path": "pages/5 Burst Detection.py", "image": "images/burst.png", "howto": "Burst Detection"},
    {"n": 6, "title": "Keywords Stem", "button": "Go to Keywords Stem",
     "path": "pages/6 Keywords Stem.py", "image": "images/lemma.png", "howto": "Keyword Stem"},
    {"n": 7, "title": "Sentiment Analysis", "button": "Go to Sentiment Analysis",
     "path": "pages/7 Sentiment Analysis.py", "image": "images/sentiment.png", "howto": "Sentiment Analysis"},
    {"n": 8, "title": "Shifterator", "button": "Go to Shifterator",
     "path": "pages/8 Shifterator.py", "image": "images/shifterator.png", "howto": "Shifterator"},
    {"n": 9, "title": "WordCloud", "button": "Go to WordCloud",
     "path": "pages/9 WordCloud.py", "image": "images/wordcloud.png", "howto": "WordCloud"},
]
_NUM_ICONS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]
HOME_URL = "https://www.coconut-libtool.com/"


def setup_page() -> None:
    """Harus dipanggil sebagai perintah Streamlit pertama di halaman."""
    st.set_page_config(page_title="Coconut", page_icon="🥥", layout="wide",
                       initial_sidebar_state="collapsed")


def render_menu() -> None:
    with st.popover("🔗 Menu"):
        st.page_link(HOME_URL, label="Home", icon="🏠")
        for tool in TOOLS:
            st.page_link(tool["path"], label=tool["title"], icon=_NUM_ICONS[tool["n"] - 1])


# --------------------------------------------------------------------------
# Berkas unggahan: dibaca sekali per unggahan, disimpan di session_state
# (per pengguna, hilang saat sesi berakhir) -- tidak ada cache global.
# --------------------------------------------------------------------------
_STORE = "_coconut_loaded"


def uploader(types=None, key: str | None = None):
    """Uploader standar. Bila file dihapus pengguna, data di memori ikut dibuang."""
    uploaded = st.file_uploader("Upload your file", type=types or loaders.SUPPORTED_TYPES,
                                key=key, label_visibility="collapsed")
    if uploaded is None:
        st.session_state.pop(_STORE, None)
    return uploaded


def load(uploaded_file):
    """DataFrame hasil parse (salinan baru tiap pemanggilan), atau None bila gagal."""
    data = uploaded_file.getvalue()
    file_id = getattr(uploaded_file, "file_id", None) or hashlib.md5(data).hexdigest()
    entry = st.session_state.get(_STORE)
    if entry is None or entry["id"] != file_id:
        try:
            with st.spinner("Reading your file..."):
                df, notes = loaders.parse_file(uploaded_file.name, data)
        except Exception as exc:  # pesan jelas, detail teknis tetap bisa dilihat
            fail("Your file could not be read. Please check the format and the "
                 "requirements in 'Before you start'.", exc)
            return None
        entry = {"id": file_id, "df": df, "notes": notes}
        st.session_state[_STORE] = entry
    for note in entry["notes"]:
        st.info(note, icon="ℹ️")
    return entry["df"].copy()


def fail(message: str, exc: BaseException | None = None) -> None:
    """Tampilkan pesan ramah + detail teknis (tidak lagi menelan error)."""
    st.error(message, icon="🚨")
    if exc is not None:
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(exc)), language="text")


def text_columns(df) -> list[str]:
    """Kolom bertipe teks (object/string), terurut."""
    return sorted(c for c in df.columns
                  if str(df[c].dtype) in ("object", "str", "string") or df[c].dtype == object)


def keyword_columns(df) -> list[str]:
    return [c for c in df.columns if "keyword" in c.lower()]


def show_html(html: str, height: int = 800) -> None:
    """Tampilkan HTML hasil visualisasi (Scattertext, pyLDAvis, dll.).

    st.components.v1.html sudah ditandai usang di Streamlit baru; pakai
    st.iframe bila tersedia. HTML di sini dihasilkan dari data yang diunggah
    pengguna itu sendiri, bukan dari sumber lain.
    """
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # pragma: no cover - Streamlit lama
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=True)
