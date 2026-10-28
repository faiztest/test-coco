import pandas as pd
import streamlit as st

from tools import ui

ui.setup_page()
st.page_link("https://www.coconut-libtool.com/the-app", label="Go to app", icon="🥥")

st.header('File Checker', anchor=False)
st.subheader('Put your file here...', anchor=False)


#===checks: setiap fungsi mengembalikan None (lolos) atau alasan gagal===
def _text_cols(df):
    """Kolom teks yang benar-benar berisi teks (bukan kosong semua)."""
    return [c for c in ui.text_columns(df) if df[c].dropna().astype(str).str.strip().ne("").any()]


def _year_ok(df):
    return "Year" in df.columns and pd.to_numeric(df["Year"], errors="coerce").notna().any()


def check_keywords(df):
    cols = [c for c in ui.keyword_columns(df) if df[c].dropna().astype(str).str.strip().ne("").any()]
    if not cols:
        return ("Unfortunately, you don't have a column containing keywords in your data. "
                "Please check again. If your keywords are in another column, please rename it to 'Keywords'.")


def check_sunburst(df):
    missing = [c for c in ['Document Type', 'Source title', 'Cited by', 'Year'] if c not in df.columns]
    if missing:
        return f"Unfortunately, you don't have: {', '.join(missing)}. Please check again."
    if not _year_ok(df):
        return "Your 'Year' column does not contain valid years."


def check_topic(df):
    if df.shape[0] < 2:
        return "Your data needs at least 2 documents."
    if not [c for c in ('Abstract', 'Title') if c in df.columns and c in _text_cols(df)]:
        return "Unfortunately, you don't have an 'Abstract' or 'Title' column containing text. Please check again."


def check_burst(df):
    if not _text_cols(df):
        return "Unfortunately, you don't have a column containing text in your data. Please check again."
    if not _year_ok(df):
        return "Unfortunately, you don't have a valid 'Year' column. Please check again."
    if pd.to_numeric(df["Year"], errors="coerce").nunique() < 2:
        return "Your documents all come from the same year, so a time-based analysis is not possible."


def check_text(df):
    if df.shape[0] < 2 or not _text_cols(df):
        return "Unfortunately, you don't have a column containing text in your data (or fewer than 2 documents). Please check again."


LAYOUT = [
    [("Keyword Stem", check_keywords), ("Sunburst", check_sunburst), ("Topic Modeling", check_topic)],
    [("Burst Detection", check_burst), ("Bidirected Network", check_keywords), ("Scattertext", check_text)],
    [("Shifterator", check_text), ("Sentiment Analysis", check_text), ("Wordcloud", check_text)],
]

#===read data===
uploaded_file = ui.uploader()

if uploaded_file is not None:
    data = ui.load(uploaded_file)
    if data is not None:
        for column, checks in zip(st.columns(3), LAYOUT):
            with column:
                for name, check in checks:
                    box = st.container(border=True)
                    reason = check(data)
                    if reason is None:
                        box.subheader(f'✔️ {name}', divider='blue', anchor=False)
                        box.write(f'Congratulations! You can use {name}')
                    else:
                        box.subheader(f'❌ {name}', divider='red', anchor=False)
                        box.write(reason)
