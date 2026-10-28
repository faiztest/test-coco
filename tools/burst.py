"""Deteksi burst Kleinberg (2002), model batch dua-state. Python murni.

Menggantikan paket ``burst_detection`` (fork) agar aplikasi tidak bergantung pada
git dependency dan agar tiga masalah berikut hilang:
- seri biaya (tie) membuat paket lama melempar ValueError;
- ``binom.pmf`` bisa underflow ke 0 untuk hitungan besar sehingga biaya menjadi inf
  (dipakai ``logpmf``: hasil sama bila tidak underflow);
- akhir burst: paket lama menyimpan ``end`` yang kadang eksklusif, kadang inklusif.
  Di sini ``begin`` dan ``end`` SELALU inklusif.

Argumen penting: ``r`` = jumlah kejadian relevan per periode (hitungan kata),
``d`` = TOTAL kejadian per periode (jumlah dokumen atau jumlah term per tahun).
Jangan mengisi ``d`` dengan angka tahun.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom


def _fit_cost(d, r, p):
    """-log P(r | d, p) dengan distribusi binomial (stabil untuk hitungan besar)."""
    return -binom.logpmf(k=r, n=d, p=p)


def _proportions(r, d, s):
    p0 = float(np.nansum(r)) / float(np.nansum(d))
    return p0, min(p0 * s, 0.99999)


def kleinberg_states(r, d, s: float = 2.0, gamma: float = 1.0, exact: bool = False) -> np.ndarray:
    """Deret state 0/1 per periode.

    exact=False meniru paket asli: state dipilih serakah per periode berdasarkan
    state periode sebelumnya. exact=True memakai Viterbi penuh (biaya total minimum).
    Seri (tie) dimenangkan oleh state 0.
    """
    r = np.asarray(r, dtype=float)
    d = np.asarray(d, dtype=float)
    n = len(r)
    p = _proportions(r, d, s)
    fit = np.vstack([_fit_cost(d, r, p[0]), _fit_cost(d, r, p[1])]).T          # n x 2
    up = gamma * np.log(n)                                                     # biaya naik dari 0 ke 1
    trans = np.array([[0.0, up], [0.0, 0.0]])                                  # trans[dari, ke]

    if not exact:
        states = np.zeros(n, dtype=int)
        states[0] = int(np.argmin(fit[0]))
        for t in range(1, n):
            states[t] = int(np.argmin(trans[states[t - 1]] + fit[t]))
        return states

    cost = np.zeros((n, 2))
    back = np.zeros((n, 2), dtype=int)
    cost[0] = fit[0]
    for t in range(1, n):
        for j in range(2):
            candidates = cost[t - 1] + trans[:, j]
            back[t, j] = int(np.argmin(candidates))
            cost[t, j] = candidates[back[t, j]] + fit[t, j]
    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmin(cost[-1]))
    for t in range(n - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


def burst_spans(states) -> list[tuple[int, int]]:
    """Rentang burst (indeks awal, indeks akhir INKLUSIF): naik 0->1 sampai turun 1->0."""
    spans, start = [], None
    for t in range(1, len(states)):
        if start is None and states[t] > states[t - 1]:
            start = t
        elif start is not None and states[t] < states[t - 1]:
            spans.append((start, t - 1))
            start = None
    if start is not None:
        spans.append((start, len(states) - 1))
    return spans


def detect_bursts(r, d, label: str, s: float = 2.0, gamma: float = 1.0, exact: bool = False) -> pd.DataFrame:
    """DataFrame burst untuk satu kata: label, begin, end (posisi, inklusif), weight."""
    r = np.asarray(r, dtype=float)
    d = np.asarray(d, dtype=float)
    spans = burst_spans(kleinberg_states(r, d, s, gamma, exact))
    p0, p1 = _proportions(r, d, s)
    rows = []
    for begin, last in spans:
        sl = slice(begin, last + 1)
        weight = float(np.sum(_fit_cost(d[sl], r[sl], p0) - _fit_cost(d[sl], r[sl], p1)))
        rows.append((label, begin, last, weight))
    return pd.DataFrame(rows, columns=["label", "begin", "end", "weight"])
