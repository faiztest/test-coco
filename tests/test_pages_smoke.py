"""Smoke test: setiap halaman harus terbuka tanpa exception dengan data contoh.
Jalankan: pytest tests/test_pages_smoke.py  (butuh dependensi lengkap kecuali BERTopic)."""
import pytest

from tests.helpers import label, run_page, scopus_csv

DATA = scopus_csv(120, (2008, 2024))


def ok(at):
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", ["0 FileChecker.py", "1 Scattertext.py", "2 Topic Modeling.py", "3 Bidirected Network.py",
                                  "4 Sunburst.py", "5 Burst Detection.py", "6 Keywords Stem.py", "7 Sentiment Analysis.py",
                                  "8 Shifterator.py", "9 WordCloud.py"])
def test_page_opens(page):
    ok(run_page(page, "scopus.csv", DATA))


def test_sunburst_submit_renders_chart():
    def go(at):
        [b for b in at.button if b.label == "Submit"][0].click().run()
    at = run_page("4 Sunburst.py", "scopus.csv", DATA, interact=go)
    ok(at)
    assert len(at.get("plotly_chart")) == 1


def test_scattertext_years_mode():
    def go(at):
        label(at.selectbox, "Type of comparison").select("Years").run()
    at = run_page("1 Scattertext.py", "scopus.csv", DATA, interact=go)
    ok(at)
    assert len(at.get("iframe")) == 1
