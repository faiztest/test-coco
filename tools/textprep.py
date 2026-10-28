"""Pembersihan teks bersama (Python murni, tanpa Streamlit).

Menggantikan fungsi clean_csv yang disalin di banyak halaman. Perbedaan utama:
- stopword memakai set (bukan list) dan lemmatisasi dilakukan per KATA UNIK,
  bukan per dokumen dengan cache di tiap baris -> jauh lebih cepat;
- tanda baca: hanya set yang dimaksud (, : ; . ! ? • =), angka dan tanda
  hubung TIDAK ikut terhapus (regex lama '[...!-?...]' menghapus 0-9);
- kata yang dibuang pengguna dicocokkan dengan bentuk asli maupun hasil
  lemmatisasi, dan tidak peka huruf besar/kecil.
"""
from __future__ import annotations

import itertools
import re
import string
import threading

import pandas as pd

_NLTK_LOCK = threading.Lock()
_NLTK_READY: set[str] = set()

PUNCT_RE = re.compile(r"[,:;.!?•=]")
ALL_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
QUOTES_RE = re.compile(r"[“”‘’]")
COPYRIGHT_TAIL_RE = re.compile(r"©.*", flags=re.DOTALL)
COPYRIGHT_LEAD_RE = re.compile(r"©[^.]{0,200}\.")


def ensure_nltk(*resources: str) -> None:
    """Pastikan korpus NLTK ada. Hanya mengunduh bila belum tersedia
    (dulu nltk.download dipanggil di setiap rerun di setiap halaman)."""
    import nltk

    locations = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "vader_lexicon": "sentiment/vader_lexicon",
    }
    with _NLTK_LOCK:
        for name in resources:
            if name in _NLTK_READY:
                continue
            try:
                nltk.data.find(locations.get(name, f"corpora/{name}"))
            except LookupError:
                nltk.download(name, quiet=True)
            _NLTK_READY.add(name)


def strip_copyright(text: str) -> str:
    """Hapus pernyataan hak cipta. Jika © berada di bagian awal teks, hanya
    kalimat pernyataannya yang dibuang (bukan seluruh abstrak)."""
    idx = text.find("©")
    if idx == -1:
        return text
    if idx > len(text) / 3:
        return COPYRIGHT_TAIL_RE.sub("", text)
    return COPYRIGHT_LEAD_RE.sub("", text, count=1)


def parse_word_list(raw: str, sep: str = ";") -> list[str]:
    """'a; B ;;c' -> ['a', 'b', 'c'] (rapikan spasi, huruf kecil, buang kosong)."""
    return [w.strip().lower() for w in (raw or "").split(sep) if w.strip()]


def normalize_text(df: pd.DataFrame, col: str, *, lower: bool = True,
                   rem_punc: bool = False, rem_copyright: bool = False,
                   punct: str = "basic") -> pd.DataFrame:
    """Salinan df tanpa baris kosong di `col`, dengan teks dinormalkan.

    punct="basic": hanya , : ; . ! ? • =  (angka dan '-' tetap)
    punct="all":   seluruh tanda baca ASCII (untuk topic modeling), spasi dirapikan.
    """
    out = df.dropna(subset=[col]).copy()
    s = out[col].astype(str)
    if rem_copyright:
        s = s.map(strip_copyright)
    if lower:
        s = s.str.lower()
    if rem_punc:
        s = s.str.replace(QUOTES_RE, "", regex=True)
        if punct == "all":
            s = s.str.replace(ALL_PUNCT_RE, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        else:
            s = s.str.replace(PUNCT_RE, " ", regex=True)
    out[col] = s
    return out


def clean_dataframe(df: pd.DataFrame, col: str, *, rem_punc: bool = False,
                    rem_copyright: bool = False, remove_words: list[str] | None = None,
                    lemmatize: bool = True, stopwords: bool = True,
                    punct: str = "basic") -> pd.DataFrame:
    """Normalisasi + buang stopword + lemmatisasi + buang kata pilihan pengguna."""
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer

    ensure_nltk("stopwords", "wordnet")
    out = normalize_text(df, col, lower=True, rem_punc=rem_punc, rem_copyright=rem_copyright, punct=punct)
    tokens = out[col].str.split()

    lemmatizer = WordNetLemmatizer()
    vocab = set(itertools.chain.from_iterable(tokens))
    lemma = {w: (lemmatizer.lemmatize(w) if lemmatize else w) for w in vocab}

    blocked = set(nltk_stopwords.words("english")) if stopwords else set()
    user_words = set(remove_words or [])
    user_words |= {lemmatizer.lemmatize(w) for w in user_words}

    def keep(w: str) -> bool:
        return w not in blocked and w not in user_words and lemma[w] not in user_words

    out[col] = tokens.map(lambda ws: " ".join(lemma[w] for w in ws if keep(w)))
    return out


def lemmatize_phrase(phrase: str) -> str:
    """Bentuk lemma sebuah frasa pencarian agar cocok dengan teks yang sudah dilemmatisasi."""
    from nltk.stem import WordNetLemmatizer

    ensure_nltk("wordnet")
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(w) for w in phrase.lower().split())


# --------------------------------------------------------------------------
# Normalisasi keyword (dipakai Keywords Stem dan Bidirected Network)
# --------------------------------------------------------------------------
def split_keywords(cell) -> list[str]:
    """'a; b ;c' -> ['a', 'b', 'c'] (tahan terhadap spasi dan ';' tanpa spasi)."""
    return [p.strip() for p in str(cell).split(";") if p.strip()]


def make_keyword_normalizer(method: str):
    """Fungsi frasa -> frasa ternormalisasi: huruf kecil, '-' jadi spasi, lemma/stem per kata."""
    if method == "Lemmatization":
        from nltk.stem import WordNetLemmatizer

        ensure_nltk("wordnet")
        word_fn = WordNetLemmatizer().lemmatize
    else:
        from nltk.stem.snowball import SnowballStemmer

        word_fn = SnowballStemmer("english").stem
    return lambda phrase: " ".join(word_fn(w) for w in re.sub("-", " ", phrase).lower().split())


def keyword_lists(series: pd.Series, normalize) -> list[list[str]]:
    """Daftar keyword ternormalisasi per dokumen (tanpa duplikat dalam satu dokumen).
    Tiap keyword unik dinormalkan SEKALI."""
    cells = series.dropna().astype(str)
    mapping = {p: normalize(p) for p in {p for cell in cells for p in split_keywords(cell)}}
    return [list(dict.fromkeys(mapping[p] for p in split_keywords(cell))) for cell in cells]


def match_terms(paper: pd.DataFrame, col: str, terms: list[str]) -> pd.DataFrame:
    """Baris yang memuat salah satu istilah sebagai kata utuh.

    Istilah di-lemma-kan (supaya cocok dengan teks yang sudah dilemmatisasi) dan
    regex di-escape (istilah seperti 'c++' atau '(' tidak lagi merusak pencarian).
    """
    mask = pd.Series(False, index=paper.index)
    for term in terms:
        pattern = r"\b" + re.escape(lemmatize_phrase(term)) + r"\b"
        mask |= paper[col].str.contains(pattern, case=False, na=False, regex=True)
    return paper[mask]
