"""Uji unit untuk paket tools/ (tanpa Streamlit). Jalankan: pytest tests/test_tools.py"""
import io
import os
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import loaders, textprep  # noqa: E402

JATS = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
<article article-type="research-article"><front>
<journal-meta><journal-title-group><journal-title>Journal X</journal-title></journal-title-group><publisher><publisher-name>Pub Y</publisher-name></publisher></journal-meta>
<article-meta><article-id pub-id-type="doi">10.1/abc</article-id>
<title-group><article-title>A <italic>great</italic> title</article-title></title-group>
<contrib-group><contrib contrib-type="author"><name><surname>Doe</surname><given-names>Jane</given-names></name></contrib></contrib-group>
<pub-date pub-type="epub"><year>2020</year></pub-date><pub-date pub-type="ppub"><year>2021</year></pub-date>
<history><date date-type="received"><year>2019</year></date></history>
<kwd-group><kwd>k1</kwd><kwd>k2</kwd></kwd-group></article-meta></front></article>'''


def test_pmc_year_title_and_columns_are_aligned():
    df, _ = loaders.parse_file("a.xml", JATS.encode())
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Year"] == 2021                      # ppub dipilih, bukan semua tag <year>
    assert row["Title"] == "A great title"          # teks di dalam <italic> ikut terambil
    assert row["Source title"] == "Journal X" and row["Document Type"] == "research-article"
    assert row["Cited by"] == 0


def test_pmc_tar_gz_many_articles():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for i in range(3):
            info = tarfile.TarInfo(f"PMC{i}/a{i}.nxml")
            data = JATS.encode()
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    df, _ = loaders.parse_file("x.tar.gz", buf.getvalue())
    assert len(df) == 3 and df["Year"].tolist() == [2021] * 3


def test_xml_entity_bomb_is_rejected():
    bomb = (b'<?xml version="1.0"?><!DOCTYPE l [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]>'
            b'<article><front><article-meta><title-group><article-title>&b;</article-title></title-group></article-meta></front></article>')
    with pytest.raises(ValueError):
        loaders.parse_file("bomb.xml", bomb)


def test_medline_multiline_and_crlf():
    text = ("PMID- 1\nTI  - A long title that\n      wraps over lines\nAB  - Abstract here\n      continued.\n"
            "FAU - Doe, Jane\nDP  - 2021 Mar\nJT  - Journal of Tests\nPT  - Journal Article\nMH  - Humans\nOT  - alpha\n\n"
            "PMID- 2\nTI  - Second\nDP  - 2022\nMH  - Cats\n\n").replace("\n", "\r\n")
    df, notes = loaders.parse_file("pubmed.txt", text.encode())
    assert len(df) == 2                              # tidak ada baris kosong tambahan
    assert df.loc[0, "Title"] == "A long title that wraps over lines"
    assert df.loc[0, "Year"] == 2021 and df.loc[1, "Year"] == 2022
    assert df.loc[0, "Keywords"] == "Humans; alpha"
    assert notes


def test_wos_tab_file_columns_renamed_and_bom_ok():
    wos = "\ufeffPT\tTI\tSO\tDE\tPY\tTC\tDT\nJ\tT\tJ\tkey1; key2\t2019\t5\tArticle\n"
    df, _ = loaders.parse_file("savedrecs.txt", wos.encode("utf-8"))
    assert {"Title", "Source title", "Author Keywords", "Year", "Cited by", "Document Type"} <= set(df.columns)
    assert df.loc[0, "Year"] == 2019


def test_dimensions_detected_by_content_not_filename():
    csv = ('"About the data: Exported on Jan 1, 2024. Criteria: x",,,,\n'
           'Rank,Publication ID,Title,PubYear,Times cited\n1,pub.1,Dim title,2020,7\n')
    df, _ = loaders.parse_file("whatever.csv", csv.encode())
    assert df.loc[0, "Year"] == 2020 and df.loc[0, "Cited by"] == 7 and "Rank" not in df.columns


def test_clean_dataframe_keeps_digits_and_removes_user_words():
    df = pd.DataFrame({"A": ["The Libraries of 2020 study COVID-19; e-learning! © 2020 Elsevier", None]})
    out = textprep.clean_dataframe(df, "A", rem_punc=True, rem_copyright=True,
                                   remove_words=textprep.parse_word_list("Library"))
    assert out["A"].tolist() == ["2020 study covid-19 e-learning"]   # angka & '-' tetap, "libraries" ikut terbuang


def test_copyright_at_start_does_not_delete_abstract():
    assert "studies" in textprep.strip_copyright("© 2021 The Authors. Studies on library services.").lower()


def test_keyword_normalizer_dedupes_inside_document():
    norm = textprep.make_keyword_normalizer("Lemmatization")
    lists = textprep.keyword_lists(pd.Series(["Libraries; library;E-Learning", None]), norm)
    assert lists == [["library", "e learning"]]


def test_burst_proportional_term_is_not_a_burst_even_when_volume_grows():
    """Kleinberg: istilah dengan proporsi tetap tidak boleh 'burst' hanya karena volume terbit naik."""
    from tools import burst
    years = np.arange(2010, 2025)
    totals = (20 + (years - 2010) * 10).astype(float)
    assert burst.detect_bursts(totals * 0.25, totals, "x").empty           # tidak melempar error pada seri/tie


def test_burst_detects_real_burst_with_inclusive_end():
    from tools import burst
    totals = np.full(12, 100.0)
    counts = np.array([2, 3, 2, 2, 30, 35, 32, 2, 3, 2, 2, 3], float)
    out = burst.detect_bursts(counts, totals, "blockchain")
    assert out[["begin", "end"]].values.tolist() == [[4, 6]]               # begin & end inklusif
    assert out.loc[0, "weight"] > 0


def test_burst_large_counts_do_not_underflow():
    from tools import burst
    d = np.full(15, 1e6)
    r = np.array([1e4] * 6 + [5e4] * 3 + [1e4] * 6, float)
    out = burst.detect_bursts(r, d, "z")
    assert out[["begin", "end"]].values.tolist() == [[6, 8]] and np.isfinite(out.loc[0, "weight"])


def test_exact_viterbi_never_costs_more_than_greedy():
    from tools import burst
    rng = np.random.default_rng(1)
    for _ in range(30):
        n = int(rng.integers(6, 20))
        d = rng.integers(20, 400, n).astype(float)
        r = rng.binomial(d.astype(int), rng.uniform(.01, .3)).astype(float)
        p0, p1 = burst._proportions(r, d, 2.0)

        def cost(states):
            c = sum(burst._fit_cost(d[t], r[t], (p0, p1)[states[t]]) for t in range(n))
            return c + sum(np.log(n) for t in range(1, n) if states[t] > states[t - 1])
        assert cost(burst.kleinberg_states(r, d, exact=True)) <= cost(burst.kleinberg_states(r, d)) + 1e-9
