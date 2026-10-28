"""Bantuan uji: jalankan halaman Streamlit dengan AppTest + file unggahan tiruan."""
import io
import os
import random
import sys
from unittest import mock

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class FakeUpload(io.BytesIO):
    def __init__(self, name, data):
        super().__init__(data)
        self.name, self.file_id, self.size = name, "fid-" + name, len(data)


def run_page(page, name, data, interact=None, timeout=300):
    """Masuk lewat Home.py (seperti aplikasi asli) lalu pindah ke halaman."""
    fake = FakeUpload(name, data)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        with mock.patch("streamlit.file_uploader", lambda *a, **k: fake):
            at = AppTest.from_file(os.path.join(ROOT, "Home.py"), default_timeout=timeout)
            at.run()
            at.switch_page(f"pages/{page}")
            at.run()
            if interact:
                interact(at)
    finally:
        os.chdir(cwd)
    return at


def label(widgets, text):
    return [w for w in widgets if w.label == text][0]


def scopus_csv(n=60, years=(2010, 2024)):
    random.seed(1)
    rows = ['Authors,Title,Year,Source title,Cited by,Document Type,Abstract,Author Keywords,Index Keywords']
    topics = ['library', 'machine learning', 'citation analysis', 'open access', 'data mining', 'digital archive']
    for i in range(n):
        t, t2, y = random.choice(topics), random.choice(topics), random.randint(*years)
        cited = random.randint(0, 50) if i % 7 else ""
        rows.append(f'"Doe, J.","Study of {t} number {i}",{y},"Journal {i % 4}",{cited},{"Article" if i % 3 else "Review"},'
                    f'"This paper studies {t} and {t2} in libraries with results {i}. The findings of {t} show growth in {y}. (c) 2020 Elsevier",'
                    f'"{t}; {t2}; libraries","{t}; indexing"')
    return '\n'.join(rows).encode()
