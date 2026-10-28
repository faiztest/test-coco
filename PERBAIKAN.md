# Catatan perbaikan Coconut Libtool

Hasil review kode (bug, kode berulang, proses terlalu panjang, cara kerja keliru) beserta perbaikannya.
Diuji dengan Python 3.12, Streamlit 1.66. BERTopic **tidak** bisa dijalankan di lingkungan uji (lihat bagian "Belum teruji").

## Struktur baru

```
Home.py                  menu dibuat dari daftar tool tunggal (tools/ui.py)
pages/0..9 *.py          nama file tidak diubah; fungsi pemuat/pembersih ada di tools/
tools/
  loaders.py             satu pintu masuk: parse_file(nama, bytes) -> (DataFrame, catatan)
  sourceformat.py        parser HathiTrust, Dimensions, PMC/JATS, MEDLINE (diperbaiki)
  textprep.py            pembersihan teks, normalisasi keyword, pencarian istilah
  burst.py               deteksi burst Kleinberg (menggantikan paket burst_detection)
  ui.py                  config halaman, menu, uploader, pemuatan per sesi, penanganan error
.streamlit/config.toml   tema/branding (hijau pelepah + krem sabut), mode toolbar minimal
tests/                   tes unit (tools/) dan smoke test tiap halaman
```

Berkas unggahan sekarang dibaca sekali per unggahan dan disimpan di `st.session_state` (per pengguna,
hilang saat sesi berakhir). Tidak ada lagi `st.cache_data.clear()` global dan tidak ada file sementara
bernama tetap (`result.png`, `output.html`, ...).

## Bug yang diperbaiki

| Area | Masalah | Perbaikan |
|---|---|---|
| Burst Detection | Penyebut model Kleinberg diisi **angka tahun**, bukan total dokumen/term per tahun | Diisi total per tahun (`tools/burst.py`); istilah dengan proporsi tetap tidak lagi "burst" saat volume terbit naik |
| Burst Detection | Seri biaya / hitungan besar membuat error; akhir burst kadang eksklusif kadang inklusif | `begin`/`end` selalu inklusif; `logpmf` tanpa underflow; seri dimenangkan state 0 |
| Topic Modeling pyLDA | "Lambda" dikirim ke `gamma_threshold`; "N-grams" dikirim sebagai `min_count`; `passes`=1; hanya 10 topik diekspor | λ dikirim ke panel pyLDAvis; N-gram menentukan bigram/trigram; ada `Passes` dan `Minimum phrase count`; semua topik diekspor |
| Topic Modeling Biterm | Kosakata dari teks ber-n-gram tetapi dokumen dari teks tanpa n-gram; `alpha=50/8` ditulis mati; API bitermplus 1.0 berubah | Teks sama untuk keduanya; `alpha=50/T`; kompatibel dengan bitermplus lama dan 1.0 |
| Topic Modeling | Opsi "Abstract + Title" tidak pernah muncul; hasil basi bila setelan berubah | Opsi berfungsi; hasil disimpan per sesi dan ada pemberitahuan bila setelan berubah |
| Sunburst | `NameError` bila semua dokumen satu tahun; filter kosong menghapus semua baris; sitasi kosong dibuang diam-diam; judul sumber dipakai sebagai regex | Semua diperbaiki; hasil tidak hilang saat widget lain berubah |
| FileChecker | `.xls/.xlsx` → `NameError`; deteksi Dimensions lewat nama file | Deteksi lewat isi file; pemeriksaan berbasis tabel |
| Loader | `.xls` dibaca dengan `openpyxl` (tidak bisa) → pakai `xlrd`; BOM pada file WoS | Diperbaiki di `tools/loaders.py` |
| HathiTrust | Keyword menumpuk antar volume; `htrcxtra` salah indentasi dan memakai MARC 350 | Diperbaiki (MARC 300$a), paralel + timeout, kegagalan satu volume tidak membatalkan semua |
| PMC/MEDLINE | Daftar paralel → tahun/judul bergeser; Document Type & Source title tertukar; string `"empty"` ikut dianalisis | Tiap rekaman = satu baris; tahun dari `pub-date`; XML dibaca dengan `defusedxml` |
| "Cited by" | Diisi jumlah dokumen dengan string keyword sama (bukan sitasi) | Diisi 0 dan pengguna diberi tahu; Sunburst mewarnai berdasarkan jumlah dokumen bila tidak ada sitasi |
| Pembersihan teks | Regex `[...!-?...]` menghapus **angka 0–9**, tanda hubung, apostrof | Hanya tanda baca yang dimaksud; `©` di awal teks tidak lagi menghapus seluruh abstrak |
| Scattertext/Shifterator | Istilah pencarian dipakai sebagai regex dan tidak di-lemma-kan; sumber dicocokkan substring; rentang tahun default tumpang tindih | Istilah di-escape + di-lemma-kan; pencocokan sumber persis; default tidak tumpang tindih |
| Sentiment | Tabel VADER urut alfabet; frasa kosong tampil `e m p t y` | Urut berdasarkan frekuensi; dokumen tanpa frasa dihitung terpisah |
| Bidirected | `method is 'Lemmatization'`; `TransactionEncoder` global dipakai bersama; graf Altair kosong saat cache hit | `==`; encoder per panggilan + matriks sparse; graf dibangun ulang tiap tampil |
| Home | `st.button(on_click="ignore")` (tidak valid); tombol dummy memicu rerun; gambar dari URL eksternal | Tombol dinonaktifkan; gambar lokal dari `images/` |
| Umum | `nltk.download` tiap rerun; model spaCy dimuat tiap rerun; `stopwords.words()` dipanggil per kata; `components.html` usang | Unduh/muat sekali; set stopword; lemmatisasi per kata unik; `st.iframe` bila ada |

## Perilaku yang berubah (periksa apakah sesuai keinginan Anda)

- Panel **Before you start** tertutup secara default.
- **Sunburst**: warna = rata-rata sitasi per dokumen (sebelumnya jumlah sitasi grup); tombol Submit tetap ada.
- **Burst Detection**: grafik interaktif Plotly (sebelumnya PNG statis); PNG resolusi tinggi dibuat saat diminta
  (butuh Chrome + kaleido), dan ada unduhan HTML interaktif; parameter `s` dan `γ` kini bisa diatur (default sama: 2 dan 1).
- **Topic Modeling**: format `results.csv` pyLDA menjadi kolom `Topic, Rank, Term, Weight`.
  BERTopic: opsi OpenAI diberi label `gpt-4o-mini` (sesuai model yang sebenarnya dipakai), kolom API key berupa password.
- **Sentiment**: default tabel VADER menampilkan dokumen yang paling sering muncul, bukan urutan alfabet.
- **Keywords Stem**: keyword duplikat dalam satu dokumen digabung; hasil juga tersedia untuk xlsx/json/xml.
- **WordCloud**: ada batas ukuran gambar; tab "Reference" dan "Download Help" yang sebelumnya kosong diisi.
- File `.txt` MEDLINE dan data PMC/HathiTrust: kolom `Cited by` = 0 (tidak ada data sitasi) dengan catatan di layar.

## Belum teruji / perlu Anda cek

- **BERTopic**: kode ditulis ulang tetapi tidak bisa dijalankan di lingkungan uji (tanpa torch). Jalankan sekali dengan data kecil.
- **Cuplikan λ pyLDAvis** (`location.hash`) dan **ekspor gambar html2image / kaleido** butuh browser/Chrome; tidak teruji.
- Lingkungan uji memakai spaCy 3.8 dan bitermplus 1.0, bukan versi pin produksi (spaCy 3.7.2).
  Di lingkungan uji, `spacy.load` gagal dengan `preshed`/`cymem` terbaru (OverflowError); karena itu keduanya di-pin.
- Pin versi di `requirements.txt` belum dicoba sebagai instalasi bersih. Setelah aplikasi berjalan di server Anda,
  jalankan `pip freeze > requirements.lock.txt`.
- `tools/burst.py` meniru metode paket lama (state dipilih serakah per periode, bukan Viterbi penuh):
  hasilnya identik pada 300 dari 300 data acak. Opsi `exact=True` (Viterbi penuh) tersedia tetapi tidak dipakai di UI;
  silakan putuskan apakah dipakai untuk publikasi.
- `st.iframe`/`width="stretch"` butuh Streamlit terbaru; `requirements.txt` meminta `>=1.50`.

## Menjalankan tes

```bash
pip install pytest
pytest tests/test_tools.py          # cepat, tanpa Streamlit
pytest tests/test_pages_smoke.py    # membuka tiap halaman dengan data contoh
```
