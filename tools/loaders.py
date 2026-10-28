"""Pemuatan dan standardisasi file unggahan (Python murni, tanpa Streamlit).

Satu pintu masuk untuk semua halaman: ``parse_file(nama, bytes)``.
Format dikenali dari ISI file (bukan hanya nama), lalu kolom dinormalkan ke
nama standar gaya Scopus: Title, Abstract, Year, Cited by, Source title,
Document Type, Keywords, ...
"""
from __future__ import annotations

import io
import json

import pandas as pd

from . import sourceformat as sf

SUPPORTED_TYPES = ["csv", "txt", "json", "tar.gz", "xml", "xls", "xlsx"]

# --- peta nama kolom per sumber -------------------------------------------
WOS_COLUMNS = {
    "TI": "Title", "SO": "Source title", "DE": "Author Keywords",
    "DT": "Document Type", "AB": "Abstract", "TC": "Cited by",
    "PY": "Year", "ID": "Keywords Plus",
}
DIMENSIONS_COLUMNS = {
    "MeSH terms": "Keywords", "PubYear": "Year", "Times cited": "Cited by",
    "Publication Type": "Document Type",
}
OPENALEX_COLUMNS = {
    "keywords.display_name": "Keywords", "publication_year": "Year",
    "cited_by_count": "Cited by", "type": "Document Type",
    "primary_location.source.display_name": "Source title",
    "title": "Title",
}
LENS_COLUMNS = {
    "Publication Year": "Year", "Citing Works Count": "Cited by",
    "Publication Type": "Document Type", "Source Title": "Source title",
}
HATHI_COLUMNS = {
    "title": "Title", "rights_date_used": "Year",
    "content_provider_code": "Source title",
}


def _rename(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Ganti nama kolom hanya bila kolom tujuan belum ada (tidak menimpa)."""
    usable = {old: new for old, new in mapping.items()
              if old in df.columns and new not in df.columns}
    return df.rename(columns=usable)


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _read_csv(data: bytes, sep: str = ",") -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data), sep=sep, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), sep=sep, encoding="latin-1")


def _standardize_hathi(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    df = sf.htrc(df)
    failed = df.attrs.get("hathi_failed", 0)
    if failed:
        notes.append(f"{failed} volume HathiTrust gagal diambil keyword-nya "
                     "(jaringan/katalog); barisnya dibiarkan kosong.")
    df = _rename(df, HATHI_COLUMNS)
    if "Document Type" not in df.columns:
        df["Document Type"] = "HathiTrust volume"
    if "Cited by" not in df.columns:
        df["Cited by"] = 0
    notes.append("Data HathiTrust tidak memuat jumlah sitasi; kolom 'Cited by' diisi 0.")
    return df


def _standardize_table(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Kenali sumber dari isi tabel lalu samakan nama kolom."""
    first = str(df.columns[0]) if len(df.columns) else ""
    if first.startswith("About the data"):               # Dimensions
        df = _rename(sf.dim(df), DIMENSIONS_COLUMNS)
    elif "ids.openalex" in df.columns:                   # OpenAlex
        df = _rename(df, OPENALEX_COLUMNS)
    elif "Publication Year" in df.columns:               # Lens.org
        df = _rename(df, LENS_COLUMNS)
    elif "htid" in df.columns:                           # HathiTrust (tabel)
        df = _standardize_hathi(df, notes)
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in ("Year", "Cited by"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _parse_text(data: bytes, notes: list[str]) -> pd.DataFrame:
    text = _decode(data)
    if sf.looks_like_medline(text):
        notes.append("Format MEDLINE terdeteksi; data tidak memuat jumlah sitasi ('Cited by' diisi 0).")
        return sf.medline(text)
    df = pd.read_csv(io.StringIO(text), sep="\t")
    if "htid" in df.columns:
        return _standardize_hathi(df, notes)
    return _rename(df, WOS_COLUMNS)


def _parse_json(data: bytes, notes: list[str]) -> pd.DataFrame:
    payload = json.loads(_decode(data))
    if isinstance(payload, dict) and "gathers" in payload:
        records = payload["gathers"]
    else:
        records = payload
    df = pd.DataFrame.from_records(records) if isinstance(records, list) \
        else pd.json_normalize(records)
    if "htid" in df.columns:
        return _standardize_hathi(df, notes)
    return _standardize_table(df, notes)


def _parse_excel(name: str, data: bytes, notes: list[str]) -> pd.DataFrame:
    engine = "openpyxl" if name.endswith("xlsx") else "xlrd"   # openpyxl tidak bisa baca .xls
    df = pd.read_excel(io.BytesIO(data), sheet_name=0, engine=engine)
    return _standardize_table(df, notes)


def parse_file(name: str, data: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Parse satu file unggahan. Mengembalikan (DataFrame, catatan_untuk_pengguna)."""
    lname = name.lower()
    notes: list[str] = []

    if lname.endswith(".csv"):
        df = _standardize_table(_read_csv(data), notes)
    elif lname.endswith(".txt"):
        df = _parse_text(data, notes)
    elif lname.endswith(".json"):
        df = _parse_json(data, notes)
    elif lname.endswith((".tar.gz", ".tgz")):
        df = sf.readPub(data)
        notes.append("Data PubMed Central tidak memuat jumlah sitasi ('Cited by' diisi 0).")
    elif lname.endswith(".xml"):
        df = sf.readxml(data)
        notes.append("Data PubMed Central tidak memuat jumlah sitasi ('Cited by' diisi 0).")
    elif lname.endswith((".xls", ".xlsx")):
        df = _parse_excel(lname, data, notes)
    else:
        raise ValueError(f"Format file tidak didukung: {name}")

    if df.empty:
        raise ValueError("File terbaca tetapi tidak berisi data.")
    return _finalize(df), notes
