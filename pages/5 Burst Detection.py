import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from tools import ui
from tools.burst import detect_bursts

ui.setup_page()
ui.render_menu()

with st.expander("Before you start", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download Visualization"])
    with tab1:
        st.write("Burst detection identifies periods when a specific event occurs with unusually high frequency, referred to as 'bursty'. This method can be applied to identify bursts in a continuous stream of events or in discrete groups of events (such as poster title submissions to an annual conference).") 
        st.divider()
        st.write('💡 The idea came from this:') 
        st.write('Kleinberg, J. (2002). Bursty and hierarchical structure in streams. Knowledge Discovery and Data Mining. https://doi.org/10.1145/775047.775061')
                    
    with tab2:
        st.text("1. Put your file. Choose your preferred column to analyze.")
        st.text("2. Choose your preferred method to compare.")
        st.text("3. Finally, you can visualize your data.")
        st.error("This app includes lemmatization and stopwords. Currently, we only offer English words.", icon="💬")

    with tab3:
        st.code("""
        +----------------+------------------------+----------------------------------+
        |     Source     |       File Type        |              Column              |
        +----------------+------------------------+----------------------------------+
        | Scopus         | Comma-separated values | Choose your preferred column     |
        |                | (.csv)                 | that you have to analyze and     |
        +----------------+------------------------| and need a column called "Year"  |
        | Web of Science | Tab delimited file     |                                  |
        |                | (.txt)                 |                                  |
        +----------------+------------------------|                                  |
        | Lens.org       | Comma-separated values |                                  |
        |                | (.csv)                 |                                  |
        +----------------+------------------------|                                  |
        | Dimensions     | Comma-separated values |                                  |
        |                | (.csv)                 |                                  |
        +----------------+------------------------|                                  |
        | OpenAlex       | Comma-separated values |                                  |
        |                | (.csv)                 |                                  |
        +----------------+------------------------|                                  |
        | Other          | .csv .xls .xlsx        |                                  |
        +----------------+------------------------|                                  |
        | Hathitrust     | .json                  |                                  |
        +----------------+------------------------+----------------------------------+
        """, language=None)
                
    with tab4:
        st.subheader(':blue[Burst Detection]', anchor=False)
        st.button('📊 Download high resolution image.', disabled=True)
        st.text("Click download button.") 

        st.divider()
        st.subheader(':blue[Top words]', anchor=False)
        st.button('👉 Click to download list of top words.', disabled=True)
        st.text("Click download button.")  

        st.divider()
        st.subheader(':blue[Burst]', anchor=False)
        st.button('👉 Click to download the list of detected bursts.', disabled=True)
        st.text("Click download button.")

st.header("Burst Detection", anchor=False)
st.subheader('Put your file here...', anchor=False)


#===language model (dimuat sekali per proses, bukan tiap rerun)===
@st.cache_resource(show_spinner="Loading language model...")
def get_nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


@st.cache_data(max_entries=4, ttl=1800, show_spinner="Preprocessing text...")
def preprocess_texts(texts: tuple) -> list:
    """Lemmatisasi + buang stopword, dalam batch (nlp.pipe). Hanya bergantung pada teks."""
    nlp = get_nlp()
    return [' '.join(t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop)
            for doc in nlp.pipe(texts, batch_size=64)]


@st.cache_data(max_entries=4, ttl=1800, show_spinner="Counting terms...")
def yearly_counts(processed: tuple, years: tuple, count_method: str, xgram: int):
    """Matriks tahun x term (sparse) + total kejadian per tahun.

    totals adalah penyebut model Kleinberg: jumlah DOKUMEN per tahun (Document
    Frequency) atau jumlah TERM per tahun (Term Frequency). Versi lama salah
    mengisinya dengan angka tahun.
    """
    years_arr = np.asarray(years, dtype=int)
    vec = CountVectorizer(lowercase=False, tokenizer=str.split, token_pattern=None,
                          binary=(count_method == "Document Frequency"), ngram_range=(1, xgram))
    X = vec.fit_transform(processed)
    all_years = np.arange(years_arr.min(), years_arr.max() + 1)
    idx = years_arr - years_arr.min()
    group = sparse.csr_matrix((np.ones(len(idx)), (idx, np.arange(len(idx)))),
                              shape=(len(all_years), len(idx)))
    yearly = (group @ X).tocsc()                                # tahun x term, tetap sparse
    if count_method == "Document Frequency":
        totals = np.bincount(idx, minlength=len(all_years)).astype(float)
    else:
        totals = np.asarray(yearly.sum(axis=1)).ravel().astype(float)
    return yearly, vec.get_feature_names_out(), all_years, totals


def select_top_terms(yearly, terms, top_n, mode, words):
    totals = np.asarray(yearly.sum(axis=0)).ravel()
    if words:
        hit = np.isin(terms, list(words))
        mask = ~hit if mode == "Words to exclude" else hit
    else:
        mask = np.ones(len(terms), dtype=bool)
    candidates = np.where(mask)[0]
    return candidates[np.argsort(-totals[candidates], kind='stable')][:top_n]


def run_bursts(yearly, terms, columns, years, totals, s, gamma):
    """Jalankan deteksi burst Kleinberg untuk kolom-kolom terpilih.

    Mengembalikan (bursts, freq): bursts berisi posisi begin/end inklusif
    (indeks pada `years`); freq adalah frekuensi per tahun.
    """
    frames, freq = [], pd.DataFrame(index=years)
    for col in columns:
        word = terms[col]
        r = yearly[:, col].toarray().ravel().astype(float)
        freq[word] = r
        frame = detect_bursts(r, totals, word, s=s, gamma=gamma)
        if not frame.empty:
            frames.append(frame)
    bursts = (pd.concat(frames, ignore_index=True).sort_values('weight', ascending=False, ignore_index=True)
              if frames else pd.DataFrame(columns=['label', 'begin', 'end', 'weight']))
    return bursts, freq


def bursts_with_years(bursts, years):
    out = bursts.copy()
    out['begin'] = [int(years[i]) for i in out['begin']]
    out['end'] = [int(years[i]) for i in out['end']]
    return out


#===visualisasi (mengembalikan figur; tidak menulis file)===
def heatmap_figure(bursts_years, freq):
    melted = freq.reset_index(names="Year").melt(id_vars=["Year"], var_name="Category", value_name="Value")
    melted = melted[melted["Value"] > 0]
    wordlist = melted["Category"].unique()
    years = freq.index.tolist()

    points = [(y, row["label"], row["weight"])
              for _, row in bursts_years.iterrows() for y in range(row["begin"], row["end"] + 1)]
    burst_df = pd.DataFrame(points, columns=["Year", "Category", "Weight"])

    n_years = max(years) - min(years) + 1
    plot_height = len(wordlist) * 50 + 100
    plot_width = n_years * 60 + 150

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=burst_df["Year"], y=burst_df["Category"], mode='markers',
                             marker=dict(symbol='square', size=40, color='red', opacity=0.5),
                             hoverinfo='text', text=burst_df["Weight"], showlegend=False))
    vmax = melted["Value"].max()
    fig.add_trace(go.Scatter(
        x=melted["Year"], y=melted["Category"], mode='markers+text',
        marker=dict(symbol='square', size=30, color=melted["Value"], colorscale='Blues', showscale=False),
        text=melted["Value"], textposition="middle center",
        textfont=dict(size=16, color=['white' if v > vmax / 2 else 'black' for v in melted["Value"]])))
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1, range=[min(years) - 1, max(years) + 1],
                   tickfont=dict(size=16), automargin=True, showgrid=False, zeroline=False),
        yaxis=dict(tickvals=wordlist, ticktext=wordlist, tickmode='array',
                   tickfont=dict(size=16), automargin=True, showgrid=False, zeroline=False),
        plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20), height=plot_height)
    return fig, plot_width, plot_height


def line_figure(bursts, freq, running_total):
    words = list(freq.columns)
    num_rows = max(1, math.ceil(len(words) / 2))
    years = list(freq.index)
    spacing = 100
    x_positions = np.arange(len(years)) * spacing

    fig = make_subplots(rows=num_rows, cols=2, subplot_titles=words)
    for i, column in enumerate(words):
        row, col = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scatter(
            x=x_positions, y=freq[column].to_numpy(), mode='lines+markers+text', name=column,
            line_shape='linear', hoverinfo='text',
            hovertext=[f"Year: {yr}<br>Frequency: {f}" for yr, f in zip(years, freq[column])],
            text=freq[column], textposition='top center'), row=row, col=col)

        for _, b in bursts[bursts['label'] == column].iterrows():
            lo, hi = int(b['begin']), int(b['end'])            # posisi, inklusif
            x_vals = x_positions[lo:hi + 1]
            y_vals = freq[column].iloc[lo:hi + 1].to_numpy()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', mode='lines',
                                     fillcolor='rgba(0,100,80,0.2)', line=dict(width=0)), row=row, col=col)
            fig.add_annotation(
                x=x_vals[0] + 0.5, y=float(np.nanmin(freq[column])) * 0.95,
                text=f"Weight: {b['weight']:.2f}", showarrow=False, font=dict(color="black", size=12),
                align="left" if running_total == "Running total" else "center",
                valign="bottom" if running_total == "Running total" else "middle",
                textangle=270, row=row, col=col)

    fig.update_xaxes(range=[-spacing / 2, x_positions[-1] + spacing / 2], tickmode='array',
                     tickvals=x_positions, ticktext=[str(y) for y in years])
    plot_height = num_rows * 500
    plot_width = len(years) * spacing + 200
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=100, b=20), height=plot_height)
    return fig, plot_width, plot_height


def png_download(fig, width, height, key):
    """Ekspor PNG butuh kaleido + Chrome; dibuat hanya bila diminta dan tidak boleh menjatuhkan halaman."""
    ready = st.session_state.get(key)
    if ready is None:
        if st.button("📊 Prepare high resolution image", key=key + "_prepare"):
            try:
                st.session_state[key] = fig.to_image(format="png", width=width, height=height, scale=4)
                st.rerun()
            except Exception:
                st.warning("The image export tool is not available on this server. "
                           "Use the interactive HTML download instead.", icon="⚠️")
    else:
        st.download_button("📊 Download high resolution image", ready, file_name="burst.png", mime="image/png")


uploaded_file = ui.uploader()

if uploaded_file is not None:
    try:
        df = ui.load(uploaded_file)
        if df is None:
            st.stop()
        if 'Year' not in df.columns:
            st.error("We need a 'Year' column.", icon="🚨")
            st.stop()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year']).reset_index(drop=True)
        df['Year'] = df['Year'].astype(int)

        if 'Title' in df.columns and 'Abstract' in df.columns:
            coldf = ['Abstract', 'Title']
        elif 'Title' in df.columns:
            coldf = ['Title']
        elif 'Abstract' in df.columns:
            coldf = ['Abstract']
        else:
            coldf = ui.text_columns(df)
        MIN, MAX = int(df['Year'].min()), int(df['Year'].max())

        c1, c2, c3 = st.columns([3, 3, 4])
        top_n = c1.number_input("Number of top words to analyze", min_value=5, value=10, step=1)
        viz_selected = c2.selectbox("Option for visualization", ("Line graph", "Heatmap"))
        running_total = c3.selectbox("Calculation method", ("Running total", "By occurrences each year"))
        count_method = c1.selectbox("Count by", ("Term Frequency", "Document Frequency"))
        col_name = c2.selectbox("Select column to analyze", coldf)
        xgram = int(c3.selectbox("N-grams", ("1", "2", "3")))

        st.divider()
        d1, d2 = st.columns([3, 7])
        exc_inc = d1.radio("Select to exclude or focus on specific words",
                           ["Words to exclude", "Focus on these words"], horizontal=True)
        words_input = d2.text_input("Words to exclude or focus on (comma-separated)")

        if MIN == MAX:
            st.info(f'You only have data in {MAX}, so burst detection needs at least two different years.', icon="ℹ️")
            st.stop()
        YEAR = st.slider('Year', min_value=MIN, max_value=MAX, value=(MIN, MAX))
        with st.expander("🧮 Show advance settings"):
            a1, a2 = st.columns(2)
            s_param = a1.number_input("Burst intensity (s)", min_value=1.1, max_value=10.0, value=2.0, step=0.1,
                                      help="How many times more frequent a word must be than its baseline to count as bursting.")
            gamma_param = a2.number_input("Difficulty to enter a burst (γ)", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                          help="Higher values make bursts rarer and longer.")

        # Teks diproses SEKALI untuk seluruh dokumen; mengubah rentang tahun tidak mengulang spaCy.
        texts = tuple(df[col_name].fillna('').astype(str).tolist())
        processed_all = pd.Series(preprocess_texts(texts), index=df.index)
        sel = df['Year'].between(*YEAR)
        years_sel = tuple(df.loc[sel, 'Year'].tolist())
        processed_sel = tuple(processed_all[sel].tolist())

        user_words = {w.strip().lower() for w in words_input.split(',') if w.strip()}
        if user_words:   # cocokkan juga dengan bentuk lemma yang dipakai di kosakata
            user_words |= {w for w in preprocess_texts(tuple(sorted(user_words))) if w}

        yearly, terms, years, totals = yearly_counts(processed_sel, years_sel, count_method, xgram)
        columns = select_top_terms(yearly, terms, int(top_n), exc_inc, user_words)
        if len(columns) == 0:
            st.warning('No words are left after your filter.', icon='⚠️')
            st.stop()
        bursts, freq_data = run_bursts(yearly, terms, columns, years, totals, s_param, gamma_param)
        num_unique_labels = bursts['label'].nunique()
        if running_total == "Running total":
            freq_data = freq_data.cumsum()
        bursts_y = bursts_with_years(bursts, years)

        tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading"])

        with tab1:
            if bursts.empty:
                st.warning('We cannot detect any bursts', icon='⚠️')
            else:
                n_words = len(columns)
                if num_unique_labels == n_words:
                    st.info(f'We detect a burst on {num_unique_labels} word(s)', icon="ℹ️")
                else:
                    st.info(f'We only detect a burst on {num_unique_labels} word(s), which is '
                            f'{n_words - num_unique_labels} fewer than the top word(s)', icon="ℹ️")

                if viz_selected == "Line graph":
                    fig, w, h = line_figure(bursts, freq_data, running_total)
                else:
                    fig, w, h = heatmap_figure(bursts_y, freq_data)
                st.plotly_chart(fig)

                e1, e2, e3 = st.columns(3)
                with e1:
                    png_download(fig, w, h, key=f"burst_png_{viz_selected}_{count_method}_{col_name}_{xgram}_{top_n}_{running_total}_{YEAR}_{words_input}_{exc_inc}_{s_param}_{gamma_param}".replace(" ", ""))
                    st.download_button("🌐 Download interactive chart (HTML)", fig.to_html(include_plotlyjs="cdn"),
                                       file_name="burst.html", mime="text/html")
                e2.download_button("👉 Click to download list of top words",
                                   freq_data.to_csv().encode("utf-8"), "top-keywords.csv", "text/csv")
                e3.download_button("👉 Click to download the list of detected bursts",
                                   bursts_y.to_csv(index=False).encode("utf-8"), "burst.csv", "text/csv")

        with tab2:
            st.markdown('**Kleinberg, J. (2002). Bursty and hierarchical structure in streams. Knowledge Discovery and Data Mining.** https://doi.org/10.1145/775047.775061')

        with tab3:
            st.markdown('**Li, M., Zheng, Z., & Yi, Q. (2024). The landscape of hot topics and research frontiers in Kawasaki disease: scientometric analysis. Heliyon, 10(8), e29680–e29680.** https://doi.org/10.1016/j.heliyon.2024.e29680')
            st.markdown('**Domicián Máté, Ni Made Estiyanti and Novotny, A. (2024) ‘How to support innovative small firms? Bibliometric analysis and visualization of start-up incubation’, Journal of Innovation and Entrepreneurship, 13(1).** https://doi.org/10.1186/s13731-024-00361-z')
            st.markdown('**Lamba, M., Madhusudhan, M. (2022). Burst Detection. In: Text Mining for Information Professionals. Springer, Cham.** https://doi.org/10.1007/978-3-030-85085-2_6')
            st.markdown('**Santosa, F. A. (2025). Artificial Intelligence in Library Studies: A Textual Analysis. JLIS.It, 16(1).** https://doi.org/10.36253/jlis.it-626')

    except Exception as exc:
        ui.fail("Please ensure that your file or settings are correct. If you think there is a mistake, feel free to reach out to us!", exc)
