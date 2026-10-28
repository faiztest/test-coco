"""Parser format sumber data (Python murni, tanpa Streamlit).

Semua fungsi mengembalikan DataFrame dengan nama kolom yang sama dengan hasil
ekspor Scopus/WoS supaya halaman-halaman analisis bisa memakainya tanpa
pengecualian per sumber.

Perubahan penting dibanding versi sebelumnya:
- Setiap rekaman dibangun sebagai satu baris (dict), bukan list paralel,
  sehingga kolom tidak bisa bergeser antar baris.
- htrc(): keyword dikosongkan untuk tiap volume (dulu menumpuk), panggilan
  jaringan memakai timeout, paralel, dan satu kegagalan tidak membatalkan semua.
- htrcxtra(): indentasi loop diperbaiki dan memakai MARC 300$a (jumlah halaman).
- readPub/readxml: tahun diambil dari pub-date (bukan semua tag <year>),
  Source title = jurnal, Document Type = article-type, dan XML dibaca dengan
  defusedxml bila tersedia (aman terhadap entity expansion).
- medline(): pembaca rekaman memakai tag di awal baris + baris lanjutan,
  mendukung Abstract/Source title dan CRLF.
"""
from __future__ import annotations

import io
import re
import string
import tarfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pandas as pd

try:  # lebih aman untuk file unggahan publik
    from defusedxml import ElementTree as ET
    from defusedxml.common import DefusedXmlException as _XmlSecurityError
except ImportError:  # pragma: no cover
    from xml.etree import ElementTree as ET

    class _XmlSecurityError(Exception):
        pass

from xml.etree.ElementTree import ParseError

_XML_ERRORS = (ParseError, ValueError, _XmlSecurityError)

JATS_COLUMNS = [
    "Title", "Abstract", "Authors", "Year", "Source title",
    "Document Type", "Publisher", "Keywords", "DOI", "Cited by",
]


# --------------------------------------------------------------------------
# HathiTrust
# --------------------------------------------------------------------------
HATHI_API = "https://catalog.hathitrust.org/api/volumes/full/htid/{}.json"


@lru_cache(maxsize=4096)
def _hathi_marc(htid: str):
    """Ambil record MARC satu volume (di-cache per proses; gagal = exception)."""
    import pymarc
    import requests

    resp = requests.get(HATHI_API.format(htid), timeout=20)
    resp.raise_for_status()
    data = resp.json()
    recid = data["items"][0]["fromRecord"]
    xml = data["records"][recid]["marc-xml"]
    return pymarc.parse_xml_to_array(io.StringIO(xml))[0]


def _hathi_keywords(marc) -> str:
    terms: list[str] = []
    for field in marc.get_fields("650"):
        value = field.value()
        low = value.lower()
        if "http" in low or "ocolc" in low:
            continue
        cleaned = value.translate(str.maketrans("", "", string.punctuation)).strip()
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
    return "; ".join(terms)


def _hathi_pages(marc) -> str:
    parts = []
    for field in marc.get_fields("300"):
        parts.extend(field.get_subfields("a"))
    return " ".join(p.strip() for p in parts if p.strip())


def _hathi_collect(htids, extractor, max_workers: int = 8):
    """Jalankan extractor(marc) untuk tiap htid secara paralel.

    Mengembalikan (hasil_per_htid, jumlah_gagal). Kegagalan satu volume tidak
    menghentikan volume lain.
    """
    failed = 0

    def work(htid):
        nonlocal failed
        try:
            return extractor(_hathi_marc(htid))
        except Exception:
            failed += 1
            return ""

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(work, htids))
    return results, failed


def htrc(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """Tambahkan kolom Keywords dari katalog HathiTrust (subject headings 650)."""
    out = df.copy()
    htids = out["htid"].astype(str).tolist()
    out["Keywords"], failed = _hathi_collect(htids, _hathi_keywords, max_workers)
    out.attrs["hathi_failed"] = failed
    return out


def htrcxtra(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """Tambahkan kolom pages (MARC 300$a) dari katalog HathiTrust."""
    out = df.copy()
    htids = out["htid"].astype(str).tolist()
    out["pages"], failed = _hathi_collect(htids, _hathi_pages, max_workers)
    out.attrs["hathi_failed"] = failed
    return out


# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------
def dim(file: pd.DataFrame) -> pd.DataFrame:
    """Ekspor Dimensions: baris pertama 'About the data', kolom pertama Rank.

    Kolom pertama dibuang dan baris header asli dipromosikan lewat round-trip
    CSV (sekaligus membuat pandas menebak ulang tipe kolom).
    """
    formatted = file.drop(file.columns[[0]], axis=1)
    return pd.read_csv(io.StringIO(formatted.to_csv(header=False, index=False)))


# --------------------------------------------------------------------------
# JATS / PubMed Central XML
# --------------------------------------------------------------------------
def _txt(el) -> str | None:
    """Teks lengkap sebuah elemen (termasuk markup inline seperti <i>)."""
    if el is None:
        return None
    text = " ".join("".join(el.itertext()).split())
    return text or None


def _first(root, *paths):
    for path in paths:
        el = root.find(path)
        if el is not None and _txt(el):
            return el
    return None


def _pub_year(meta) -> float:
    dates = meta.findall("pub-date") if meta is not None else []
    preferred = ("ppub", "epub", "collection", "epub-ppub", "pmc-release")

    def year_of(node):
        y = node.find("year")
        match = re.search(r"\d{4}", _txt(y) or "")
        return int(match.group()) if match else None

    for kind in preferred:
        for d in dates:
            if kind in (d.get("pub-type"), d.get("date-type")):
                year = year_of(d)
                if year:
                    return year
    for d in dates:
        year = year_of(d)
        if year:
            return year
    return float("nan")


def _jats_row(article) -> dict:
    front = article.find("front") if article.tag == "article" else article
    if front is None:
        front = article
    meta = front.find(".//article-meta")
    journal = front.find(".//journal-meta")
    scope = meta if meta is not None else front

    authors = []
    for contrib in scope.iter("contrib"):
        if contrib.get("contrib-type", "author") != "author":
            continue
        name = contrib.find("name")
        if name is not None:
            given, sur = _txt(name.find("given-names")), _txt(name.find("surname"))
            full = " ".join(p for p in (given, sur) if p)
        else:
            full = _txt(contrib.find("string-name")) or _txt(contrib.find("collab"))
        if full:
            authors.append(full)

    keywords = []
    for kwd in scope.iter("kwd"):
        value = _txt(kwd)
        if value and value not in keywords:
            keywords.append(value)

    doi = None
    for aid in scope.iter("article-id"):
        if aid.get("pub-id-type") == "doi":
            doi = _txt(aid)
            break

    return {
        "Title": _txt(_first(scope, ".//title-group/article-title", ".//article-title")),
        "Abstract": _txt(scope.find("abstract")),
        "Authors": ", ".join(authors) or None,
        "Year": _pub_year(meta),
        "Source title": _txt(journal.find(".//journal-title")) if journal is not None else None,
        "Document Type": article.get("article-type") if article.tag == "article" else None,
        "Publisher": _txt(journal.find(".//publisher-name")) if journal is not None else None,
        "Keywords": "; ".join(keywords) or None,
        "DOI": doi,
        "Cited by": 0,
    }


def _rows_from_xml(data: bytes | str) -> list[dict]:
    try:
        root = ET.fromstring(data)
    except _XML_ERRORS:
        return []
    articles = list(root.iter("article"))
    if not articles:
        articles = list(root.iter("front"))
    return [_jats_row(a) for a in articles]


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=JATS_COLUMNS)
    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    return frame


def readPub(tar: bytes, max_members: int = 20000, max_member_bytes: int = 25_000_000,
            max_total_bytes: int = 400_000_000) -> pd.DataFrame:
    """Baca arsip .tar.gz berisi banyak artikel JATS (PubMed Central)."""
    rows: list[dict] = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(tar), mode="r:gz") as archive:
        for index, member in enumerate(archive):
            if index >= max_members:
                break
            if not member.isfile() or member.size > max_member_bytes:
                continue
            total += member.size
            if total > max_total_bytes:
                break
            handle = archive.extractfile(member)
            if handle is None:
                continue
            rows.extend(_rows_from_xml(handle.read()))
    return _frame(rows)


def readxml(file: bytes | str) -> pd.DataFrame:
    """Baca satu file XML yang berisi satu atau banyak artikel JATS."""
    return _frame(_rows_from_xml(file))


# --------------------------------------------------------------------------
# MEDLINE / PubMed (format .txt dengan tag "TI  - ")
# --------------------------------------------------------------------------
_MEDLINE_TAG = re.compile(r"^([A-Z][A-Z0-9]{1,3})\s*-\s(.*)$")


def looks_like_medline(text: str) -> bool:
    return re.search(r"^PMID-\s", text, flags=re.MULTILINE) is not None


def medline(src) -> pd.DataFrame:
    """Parse file MEDLINE. `src` boleh bytes, str, atau objek file."""
    if hasattr(src, "read"):
        src = src.read()
    text = src.decode("utf-8-sig", errors="replace") if isinstance(src, bytes) else str(src)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    rows = []
    for record in re.split(r"\n\s*\n", text.strip()):
        fields: dict[str, list[str]] = {}
        tag = None
        for line in record.split("\n"):
            match = _MEDLINE_TAG.match(line)
            if match:
                tag = match.group(1)
                fields.setdefault(tag, []).append(match.group(2).strip())
            elif tag and line.startswith("      "):
                fields[tag][-1] += " " + line.strip()
        if "TI" not in fields and "PMID" not in fields:
            continue

        def joined(tag_name, sep="; "):
            values = fields.get(tag_name)
            return sep.join(values) if values else None

        year = re.search(r"\d{4}", (fields.get("DP") or [""])[0])
        mesh, other = joined("MH"), joined("OT")
        both = "; ".join(v for v in (mesh, other) if v) or None
        rows.append({
            "Title": joined("TI", " "),
            "Abstract": joined("AB", " "),
            "Authors": joined("FAU") or joined("AU"),
            "Year": int(year.group()) if year else float("nan"),
            "Source title": joined("JT", " "),
            "Document Type": joined("PT"),
            "MeSH Keywords": mesh,
            "Other Keywords": other,
            "Keywords": both,
            "Cited by": 0,
        })
    return pd.DataFrame(rows)
