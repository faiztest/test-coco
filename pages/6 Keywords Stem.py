import re

import pandas as pd
import streamlit as st

from tools import loaders, textprep, ui

ui.setup_page()
ui.render_menu()

with st.expander("Before you start", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download Result"])
    with tab1:
        st.write("This approach is effective for locating basic words and aids in catching the true meaning of the word, which can lead to improved semantic analysis and comprehension of the text. Some people find it difficult to check keywords before performing bibliometrics (using software such as VOSviewer and Bibliometrix). This strategy makes it easy to combine and search for fundamental words from keywords, especially if you have a large number of keywords. To do stemming or lemmatization on other text, change the column name to 'Keyword' in your file.")
        st.divider()
        st.write('💡 The idea came from this:')
        st.write('Santosa, F. A. (2022). Prior steps into knowledge mapping: Text mining application and comparison. Issues in Science and Technology Librarianship, 102. https://doi.org/10.29173/istl2736')
        
    with tab2:
        st.text("1. Put your file.")
        st.text("2. Choose your preferable method. Picture below may help you to choose wisely.")
        st.markdown("![Source: https://studymachinelearning.com/stemming-and-lemmatization/](https://studymachinelearning.com/wp-content/uploads/2019/09/stemmin_lemm_ex-1.png)")
        st.text('Source: https://studymachinelearning.com/stemming-and-lemmatization/')
        st.text("3. Now you need to select what kind of keywords you need.")
        st.text("4. Finally, you can download and use the file on VOSviewer, Bibliometrix, or put it on OpenRefine to get better result!")
        st.error("Please check what has changed. It's possible some keywords failed to find their roots.", icon="🚨")
        
    with tab3:
        st.code("""
        +----------------+------------------------+---------------------------------+
        |     Source     |       File Type        |             Column              |
        +----------------+------------------------+---------------------------------+
        | Scopus         | Comma-separated values | Author Keywords                 |
        |                | (.csv)                 | Index Keywords                  |
        +----------------+------------------------+---------------------------------+
        | Web of Science | Tab delimited file     | Author Keywords                 |
        |                | (.txt)                 | Keywords Plus                   |
        +----------------+------------------------+---------------------------------+
        | Lens.org       | Comma-separated values | Keywords (Scholarly Works)      |
        |                | (.csv)                 |                                 |
        +----------------+------------------------+---------------------------------+
        | Dimensions     | Comma-separated values | MeSH terms                      |
        |                | (.csv)                 |                                 |
        +----------------+------------------------+---------------------------------+
        | OpenAlex       | Comma-separated values | Keywords                        |
        |                | (.csv)                 |                                 |
        +----------------+------------------------+---------------------------------+
        | Other          | .csv .xls .xlsx        | Change your column to 'Keyword' |
        +----------------+------------------------+---------------------------------+
        | Hathitrust     | .json                  | htid (Hathitrust ID)            |
        +----------------+------------------------+---------------------------------+
        """, language=None)

    with tab4:
        st.subheader(':blue[Result]', anchor=False)
        st.button('Click to download result 👈.', disabled=True)
        st.text("Go to Result and click Download button.")  

        st.divider()
        st.subheader(':blue[List of Keywords]', anchor=False)
        st.button('Click to download keywords 👈.', disabled=True)
        st.text("Go to List of Keywords and click Download button.")

st.header("Keywords Stem", anchor=False)
st.subheader('Put your file here...', anchor=False)


def normalize_column(series, normalize):
    """Ganti isi kolom keyword dengan bentuk ternormalisasi (keyword unik dinormalkan sekali)."""
    cells = series.fillna('').astype(str)
    mapping = {p: normalize(p) for p in {p for cell in cells for p in textprep.split_keywords(cell)}}
    return cells.map(lambda cell: '; '.join(dict.fromkeys(mapping[p] for p in textprep.split_keywords(cell))))


def keyword_table(series, normalize):
    """Daftar keyword unik: label asli (tanda '-' jadi spasi) dan bentuk barunya."""
    raw = sorted({re.sub('-', ' ', p) for cell in series.dropna().astype(str) for p in textprep.split_keywords(cell)})
    return pd.DataFrame({'label': raw, 'new': [normalize(p) for p in raw]})


uploaded_file = ui.uploader()

if uploaded_file is not None:
    try:
        keywords = ui.load(uploaded_file)
        if keywords is None:
            st.stop()
        list_of_column_key = ui.keyword_columns(keywords)

        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox('Choose method', ('Lemmatization', 'Stemming'))
        with col2:
            keyword = st.selectbox('Choose column', list_of_column_key)

        if keyword is None:
            st.error('Error: Please check your Author/Index Keywords column.')
            st.stop()

        normalize = textprep.make_keyword_normalizer(method)
        key = keyword_table(keywords[keyword], normalize)
        result = keywords.copy()
        result[keyword] = normalize_column(keywords[keyword], normalize)

        st.write('Congratulations! 🤩 You choose', keyword, 'with', method, 'method. Now, you can easily download the result by clicking the button below')
        st.divider()

        #===show & download csv===
        tab1, tab2, tab3, tab4 = st.tabs(["📥 Result", "📥 List of Keywords", "📃 Reference", "📃 Recommended Reading"])

        with tab1:
            st.dataframe(result, hide_index=True)
            if uploaded_file.name.lower().endswith('.txt'):
                # kembalikan nama kolom ke kode Web of Science agar file bisa dipakai ulang di VOSviewer dll.
                back = {new: old for old, new in loaders.WOS_COLUMNS.items() if new in result.columns}
                st.download_button("Click to download result 👈",
                                   result.rename(columns=back).to_csv(index=False, sep='\t', lineterminator='\r').encode('utf-8'),
                                   "result.txt", "text/csv")
            else:
                st.download_button("Click to download result 👈",
                                   result.to_csv(index=False).encode('utf-8'), "result.csv", "text/csv")

        with tab2:
            merged = set(key.loc[key['new'].duplicated(keep=False), 'new'])   # keyword yang digabung
            st.dataframe(key.style.map(lambda v: 'background-color: yellow' if v in merged else '', subset=['new']),
                         hide_index=True)
            st.download_button("Click to download keywords 👈",
                               key.to_csv(index=False).encode('utf-8'), "keywords.csv", "text/csv")

        with tab3:
            st.markdown('**Santosa, F. A. (2023). Prior steps into knowledge mapping: Text mining application and comparison. Issues in Science and Technology Librarianship, 102.** https://doi.org/10.29173/istl2736')
         
        with tab4:
            st.markdown('**Beri, A. (2021, January 27). Stemming vs Lemmatization. Medium.** https://towardsdatascience.com/stemming-vs-lemmatization-2daddabcb221')
            st.markdown('**Khyani, D., Siddhartha B S, Niveditha N M, &amp; Divya B M. (2020). An Interpretation of Lemmatization and Stemming in Natural Language Processing. Journal of University of Shanghai for Science and Technology , 22(10), 350–357.**  https://jusst.org/an-interpretation-of-lemmatization-and-stemming-in-natural-language-processing/')
            st.markdown('**Lamba, M., & Madhusudhan, M. (2021, July 31). Text Pre-Processing. Text Mining for Information Professionals, 79–103.** https://doi.org/10.1007/978-3-030-85085-2_3')
    except Exception as exc:
        ui.fail("Please ensure that your file is correct. Please contact us if you find that this is an error.", exc)
