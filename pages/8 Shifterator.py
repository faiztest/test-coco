import io
import re
from collections import Counter

import matplotlib
import pandas as pd
import shifterator as sh
import streamlit as st
import matplotlib.pyplot as plt

from tools import textprep, ui

ui.setup_page()
ui.render_menu()

with st.expander("Before you start", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download Visualization"])
    with tab1:
        st.write("Shifterator is a tool that helps compare two pieces of text by showing which words make them different, and in what way. It uses clear bar chart visuals, called word shift graphs, to explain these differences. You can use it to compare texts directly, analyze sentiment, or even as a more reliable alternative to word clouds.")
        st.divider()
        st.write('💡 The idea came from this:')
        st.write('Gallagher, R.J., Frank, M.R., Mitchell, L. et al. (2021). Generalized Word Shift Graphs: A Method for Visualizing and Explaining Pairwise Comparisons Between Texts. EPJ Data Science, 10(4). https://doi.org/10.1140/epjds/s13688-021-00260-3')
        
    with tab2:
        st.text("1. Put your file. Choose your preferred column to analyze.")
        st.text("2. Choose your preferred method to count the words and decide how many top words you want to include or remove.")
        st.text("3. Finally, you can visualize your data.")
        st.error("This app includes lemmatization and stopwords. Currently, we only offer English words.", icon="💬")
        
    with tab3:
        st.code("""
        +----------------+------------------------+----------------------------------+
        |     Source     |       File Type        |              Column              |
        +----------------+------------------------+----------------------------------+
        | Scopus         | Comma-separated values | Choose your preferred column     |
        |                | (.csv)                 | that you have to analyze.        |
        +----------------+------------------------|                                  |
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
        st.subheader(':blue[Shifterator]', anchor=False)
        st.button('📥 Download Graph.', disabled=True)
        st.text("Click Download Graph button.")  

        st.divider()
        st.subheader(':blue[Shifterator Dataframe]', anchor=False)
        st.button('📥 Press to download result.', disabled=True)
        st.text("Click the Download button to get the CSV result.")

matplotlib.use("Agg")

st.header("Shifterator", anchor=False)
st.subheader('Put your file here...', anchor=False)

SHIFT_METHODS = ('Proportion Shifts', 'Shannon Entropy Shifts', 'Tsallis Entropy Shifts',
                 'Kullback-Leibler Divergence Shifts', 'Jensen-Shannon Divergence Shifts')


def pick_text_columns(papers):
    df_col = ui.text_columns(papers)
    titles = [c for c in df_col if c.lower() == "title"]
    abstracts = [c for c in df_col if re.search('abstract', c, re.IGNORECASE)]
    selected = (abstracts + titles) if (titles and abstracts) else (titles or abstracts or df_col)
    return selected or df_col


def comparison_options(papers):
    options = ['Word-to-word', 'Manual label']
    if any('year' in c.lower() for c in papers.columns):
        options.append('Years')
    if any('source title' in c.lower() for c in papers.columns):
        options.append('Sources')
    return sorted(options)


def df2dict(series_1, series_2):
    """Frekuensi kata untuk dua kelompok dokumen (angka dibuang)."""
    def count(series):
        text = re.sub(r'\d+', '', ' '.join(series.dropna().astype(str)))
        return {k: int(v) for k, v in Counter(re.findall(r'\b\w+\b', text.lower())).items()}
    return count(series_1), count(series_2)


@st.cache_data(max_entries=4, ttl=1800, show_spinner=False)
def run_shifterator(dict1, dict2, method_shifts):
    """Gambar shift graph -> bytes PNG (bukan objek figure; figure ditutup agar tidak bocor memori)."""
    names = dict(system_names=['Topic 1', 'Topic 2'], title=method_shifts)
    try:
        if method_shifts == 'Proportion Shifts':
            shift = sh.ProportionShift(type2freq_1=dict1, type2freq_2=dict2)
        elif method_shifts == 'Shannon Entropy Shifts':
            shift = sh.EntropyShift(type2freq_1=dict1, type2freq_2=dict2, base=2)
        elif method_shifts == 'Tsallis Entropy Shifts':
            shift = sh.EntropyShift(type2freq_1=dict1, type2freq_2=dict2, base=2, alpha=0.8)
        elif method_shifts == 'Kullback-Leibler Divergence Shifts':
            shift = sh.KLDivergenceShift(type2freq_1=dict1, type2freq_2=dict2, base=2)
        else:
            shift = sh.JSDivergenceShift(type2freq_1=dict1, type2freq_2=dict2, weight_1=0.5, weight_2=0.5, base=2, alpha=1)
        ax = shift.get_shift_graph(**names)
        fig = ax.get_figure()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches='tight', dpi=150)
        plt.close(fig)
        return buffer.getvalue()
    except ValueError:
        plt.close('all')
        return None


@st.cache_data(max_entries=4, ttl=1800, show_spinner=False)
def shifts_csv(type2freq_1, type2freq_2):
    shift = sh.ProportionShift(type2freq_1=type2freq_1, type2freq_2=type2freq_2)
    rows = [{'word': word, 'freq_text1': shift.type2freq_1.get(word, 0),
             'freq_text2': shift.type2freq_2.get(word, 0), 'shift_score': score}
            for word, score in shift.get_shift_scores().items()]
    return pd.DataFrame(rows).sort_values('shift_score').to_csv(index=False).encode('utf-8')


#===Read data===
uploaded_file = ui.uploader()

if uploaded_file is not None:
    try:
        papers = ui.load(uploaded_file)
        if papers is None:
            st.stop()
        selected_cols = pick_text_columns(papers)
        comparison = comparison_options(papers)

        #Menu
        c1, c2, c3 = st.columns([4, 0.1, 4])
        ColCho = c1.selectbox('Choose column to analyze', selected_cols)
        c2.write('')
        compare = c3.selectbox('Type of comparison', comparison)

        with st.expander("🧮 Show advance settings"):
            y1, y2, y3 = st.columns([4, 0.1, 4])
            t1, t2 = st.columns([3, 3])
            words_to_remove = y1.text_input('Input your text', placeholder='Remove specific words. Separate words by semicolons (;)')
            method_shifts = y3.selectbox("Choose preferred method", SHIFT_METHODS)
            rem_copyright = t1.toggle('Remove copyright statement', value=True)
            rem_punc = t2.toggle('Remove punctuation', value=False)

        if method_shifts == 'Kullback-Leibler Divergence Shifts':
            st.info('The Kullback-Leibler Divergence is only well-defined if every single word in the comparison text is also in the reference text.', icon="ℹ️")

        paper = textprep.clean_dataframe(papers, ColCho, rem_punc=rem_punc, rem_copyright=rem_copyright,
                                         remove_words=textprep.parse_word_list(words_to_remove))

        tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading"])

        with tab1:
            #===visualization===
            type2freq_1 = type2freq_2 = None

            if compare == 'Word-to-word':
                col1, col2, col3 = st.columns([4, 0.1, 4])
                text1 = col1.text_input('First Term', placeholder='put comma if you have more than one')
                search_terms1 = [t.strip() for t in text1.split(",") if t.strip()]
                col2.write('')
                text2 = col3.text_input('Second Term', placeholder='put comma if you have more than one')
                search_terms2 = [t.strip() for t in text2.split(",") if t.strip()]

                if search_terms1 and search_terms2:
                    dfs1 = textprep.match_terms(paper, ColCho, search_terms1).drop_duplicates()
                    dfs2 = textprep.match_terms(paper, ColCho, search_terms2).drop_duplicates()
                    if dfs1.empty and dfs2.empty:
                        st.warning('We cannot find anything in your document.', icon="⚠️")
                    elif dfs1.empty:
                        st.warning(f'We cannot find {text1} in your document.', icon="⚠️")
                    elif dfs2.empty:
                        st.warning(f'We cannot find {text2} in your document.', icon="⚠️")
                    else:
                        type2freq_1, type2freq_2 = df2dict(dfs1[ColCho], dfs2[ColCho])

            elif compare == 'Manual label':
                col1, col2, col3 = st.columns(3)
                column_selected = col1.selectbox('Choose column', sorted(paper.columns.tolist()))
                values = [w for w in paper[column_selected].dropna() if str(w).strip() != ""]
                list_unique = sorted(set(values))
                label1 = col2.selectbox('Choose first label', list_unique)
                label2 = col3.selectbox('Choose second label', list_unique, index=0 if len(list_unique) == 1 else 1)
                if label1 == label2:
                    st.warning('Please choose two different labels.', icon="⚠️")
                else:
                    type2freq_1, type2freq_2 = df2dict(paper[paper[column_selected] == label1][ColCho],
                                                       paper[paper[column_selected] == label2][ColCho])

            elif compare == 'Sources':
                col1, col2, col3 = st.columns([4, 0.1, 4])
                list_stitle = sorted(paper['Source title'].dropna().unique())
                stitle1 = col1.selectbox('Choose first label', list_stitle)
                col2.write('')
                stitle2 = col3.selectbox('Choose second label', list_stitle, index=0 if len(list_stitle) == 1 else 1)
                if stitle1 == stitle2:
                    st.warning('Please choose two different sources.', icon="⚠️")
                else:
                    type2freq_1, type2freq_2 = df2dict(paper[paper['Source title'] == stitle1][ColCho],
                                                       paper[paper['Source title'] == stitle2][ColCho])

            elif compare == 'Years':
                col1, col2, col3 = st.columns([4, 0.1, 4])
                MIN, MAX = int(paper['Year'].min()), int(paper['Year'].max())
                if MAX != MIN:
                    MID = round((MIN + MAX) / 2)
                    first_range = col1.slider('First Range', min_value=MIN, max_value=MAX, value=(MIN, MID))
                    col2.write('')
                    second_range = col3.slider('Second Range', min_value=MIN, max_value=MAX, value=(min(MID + 1, MAX), MAX))
                    type2freq_1, type2freq_2 = df2dict(paper[paper['Year'].between(*first_range)][ColCho],
                                                       paper[paper['Year'].between(*second_range)][ColCho])
                else:
                    st.write('You only have data in ', MAX)

            if type2freq_1 and type2freq_2:
                with st.spinner('Processing. Please wait until the visualization comes up'):
                    png = run_shifterator(type2freq_1, type2freq_2, method_shifts)
                if png is None:
                    st.warning('Please check your data.', icon="⚠️")
                else:
                    st.image(png, width="stretch")
                    d1, d2 = st.columns(2)
                    d1.download_button(label="📥 Download Graph", data=png, file_name="shifterator.png", mime="image/png")
                    d2.download_button(label="📥 Click to download result",
                                       data=shifts_csv(type2freq_1, type2freq_2),
                                       file_name="shifterator_scores.csv", mime="text/csv")

        with tab2:
            st.markdown('**Gallagher, R.J., Frank, M.R., Mitchell, L. et al. (2021). Generalized Word Shift Graphs: A Method for Visualizing and Explaining Pairwise Comparisons Between Texts. EPJ Data Science, 10(4).** https://doi.org/10.1140/epjds/s13688-021-00260-3')
    
        with tab3:
            st.markdown('**Sánchez-Franco, M. J., & Rey-Tienda, S. (2023). The role of user-generated content in tourism decision-making: an exemplary study of Andalusia, Spain. Management Decision, 62(7).** https://doi.org/10.1108/md-06-2023-0966')
            st.markdown('**Ipek Baris Schlicht, Fernandez, E., Chulvi, B., & Rosso, P. (2023). Automatic detection of health misinformation: a systematic review. Journal of Ambient Intelligence and Humanized Computing, 15.** https://doi.org/10.1007/s12652-023-04619-4')
            st.markdown('**Torricelli, M., Falkenberg, M., Galeazzi, A., Zollo, F., Quattrociocchi, W., & Baronchelli, A. (2023). Hurricanes Increase Climate Change Conversations on Twitter. PLOS Climate, 2(11)** https://doi.org/10.1371/journal.pclm.0000277')
    except Exception as exc:
        ui.fail("Please ensure that your file is correct. Please contact us if you find that this is an error.", exc)
