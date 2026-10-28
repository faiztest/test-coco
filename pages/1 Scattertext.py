import re

import pandas as pd
import scattertext as stx
import streamlit as st

from tools import textprep, ui

ui.setup_page()
ui.render_menu()

with st.expander("Before you start", expanded=False):
            
        tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download"])
        with tab1:
            st.write("Scattertext is an open-source tool designed to visualize linguistic variations between document categories in a language-independent way. It presents a scatterplot, with each axis representing the rank-frequency of a term's occurrence within a category of documents.") 
            st.divider()
            st.write('💡 The idea came from this:') 
            st.write('Kessler, J. S. (2017). Scattertext: a Browser-Based Tool for Visualizing how Corpora Differ. https://doi.org/10.48550/arXiv.1703.00565')
                
        with tab2:
            st.text("1. Put your file. Choose your preferred column to analyze.")
            st.text("2. Choose your preferred method to compare and decide words you want to remove.")
            st.text("3. Finally, you can visualize your data.")
            st.error("This app includes lemmatization and stopwords. Currently, we only offer English words.", icon="💬")
            
        with tab3:
            st.code("""
            +----------------+------------------------+----------------------------------+
            |     Source     |       File Type        |              Column              |
            +----------------+------------------------+----------------------------------+
            | Scopus         | Comma-separated values | Choose your preferred column     |
            |                | (.csv)                 | that you have                    |
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
            st.subheader(':blue[Image]', anchor=False)
            st.write("Click the :blue[Download SVG] on the right side.")  
            st.divider()
            st.subheader(':blue[Scattertext Dataframe]', anchor=False)
            st.button('📥 Click to download result.', disabled=True)
            st.text("Click the Download button to get the CSV result.")

st.header("Scattertext", anchor=False)
st.subheader('Put your file here...', anchor=False)


def pick_text_columns(papers):
    """Kolom teks yang ditawarkan: Abstract+Title bila ada, selain itu semua kolom teks."""
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


@st.cache_data(max_entries=3, ttl=1800, show_spinner=False)
def run_scattertext(filtered_df, cat_col, catname, noncatname, text_col, min_term):
    """Semua masukan eksplisit -> kunci cache lengkap. Mengembalikan (csv, html) atau (None, None)."""
    try:
        corpus = (stx.CorpusFromPandas(filtered_df, category_col=cat_col, text_col=text_col,
                                       nlp=stx.whitespace_nlp_with_sentences)
                  .build().get_unigram_corpus().remove_infrequent_words(minimum_term_count=min_term))
        disp_csv = stx.Dispersion(corpus).get_df().to_csv(index=False).encode('utf-8')
        options = dict(category=catname, category_name=str(catname), not_category_name=str(noncatname),
                       width_in_pixels=900, minimum_term_frequency=0, save_svg_button=True)
        if 'Title' in filtered_df.columns:
            options['metadata'] = filtered_df['Title']
        return disp_csv, stx.produce_scattertext_explorer(corpus, **options)
    except ValueError:
        return None, None


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
            y1, y2 = st.columns([8, 2])
            t1, t2 = st.columns([3, 3])
            words_to_remove = y1.text_input('Input your text', placeholder='Remove specific words. Separate words by semicolons (;)')
            min_term = y2.number_input("Minimum term count", min_value=0, max_value=10, value=3, step=1)
            rem_copyright = t1.toggle('Remove copyright statement', value=True)
            rem_punc = t2.toggle('Remove punctuation', value=False)

        st.info('Scattertext is an expensive process when dealing with a large volume of text with our existing resources. Please kindly wait until the visualization appears.', icon="ℹ️")

        paper = textprep.clean_dataframe(papers, ColCho, rem_punc=rem_punc, rem_copyright=rem_copyright,
                                         remove_words=textprep.parse_word_list(words_to_remove))

        tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading"])

        with tab1:
            #===visualization===
            filtered_df, cat_col, cat1, cat2 = None, None, None, None

            if compare == 'Word-to-word':
                col1, col2, col3 = st.columns([4, 0.1, 4])
                text1 = col1.text_input('First Term', placeholder='put comma if you have more than one')
                search_terms1 = [t.strip() for t in text1.split(",") if t.strip()]
                col2.write('')
                text2 = col3.text_input('Second Term', placeholder='put comma if you have more than one')
                search_terms2 = [t.strip() for t in text2.split(",") if t.strip()]

                if search_terms1 and search_terms2:
                    dfs1 = textprep.match_terms(paper, ColCho, search_terms1).drop_duplicates().assign(Topic='First Term')
                    dfs2 = textprep.match_terms(paper, ColCho, search_terms2).drop_duplicates().assign(Topic='Second Term')
                    if dfs1.empty and dfs2.empty:
                        st.warning('We cannot find anything in your document.', icon="⚠️")
                    elif dfs1.empty:
                        st.warning(f'We cannot find {text1} in your document.', icon="⚠️")
                    elif dfs2.empty:
                        st.warning(f'We cannot find {text2} in your document.', icon="⚠️")
                    else:
                        both = len(set(dfs1.index) & set(dfs2.index))
                        if both:
                            st.caption(f'{both} document(s) contain both terms and appear in both groups.')
                        filtered_df = pd.concat([dfs1, dfs2], ignore_index=True)
                        cat_col, cat1, cat2 = 'Topic', 'First Term', 'Second Term'

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
                    filtered_df = paper[paper[column_selected].isin([label1, label2])].reset_index(drop=True)
                    cat_col, cat1, cat2 = column_selected, label1, label2

            elif compare == 'Sources':
                col1, col2, col3 = st.columns([4, 0.1, 4])
                list_stitle = sorted(paper['Source title'].dropna().unique())
                stitle1 = col1.selectbox('Choose first label', list_stitle)
                col2.write('')
                stitle2 = col3.selectbox('Choose second label', list_stitle, index=0 if len(list_stitle) == 1 else 1)
                if stitle1 == stitle2:
                    st.warning('Please choose two different sources.', icon="⚠️")
                else:
                    filtered_df = pd.concat([paper[paper['Source title'] == stitle1].assign(Topic=stitle1),
                                             paper[paper['Source title'] == stitle2].assign(Topic=stitle2)],
                                            ignore_index=True)
                    cat_col, cat1, cat2 = 'Source title', stitle1, stitle2

            elif compare == 'Years':
                col1, col2, col3 = st.columns([4, 0.1, 4])
                MIN, MAX = int(paper['Year'].min()), int(paper['Year'].max())
                if MAX != MIN:
                    MID = round((MIN + MAX) / 2)
                    first_range = col1.slider('First Range', min_value=MIN, max_value=MAX, value=(MIN, MID))
                    col2.write('')
                    second_range = col3.slider('Second Range', min_value=MIN, max_value=MAX, value=(min(MID + 1, MAX), MAX))
                    first = paper[paper['Year'].between(*first_range)].assign(**{'Topic Range': 'First range'})
                    second = paper[paper['Year'].between(*second_range)].assign(**{'Topic Range': 'Second range'})
                    filtered_df = pd.concat([first, second], ignore_index=True)
                    cat_col, cat1, cat2 = 'Topic Range', 'First range', 'Second range'
                else:
                    st.write('You only have data in ', MAX)

            if filtered_df is not None:
                with st.spinner('Processing. Please wait until the visualization comes up'):
                    disp_df, html = run_scattertext(filtered_df, cat_col, cat1, cat2, ColCho, int(min_term))
                if html is None:
                    st.warning('Please decrease the Minimum term count in the advanced settings.', icon="⚠️")
                else:
                    ui.show_html(html, height=1200)
                    st.download_button("📥 Click to download result", disp_df, "scattertext_dataframe.csv",
                                       "text/csv", on_click="ignore")

        with tab2:
            st.markdown('**Jason Kessler. 2017. Scattertext: a Browser-Based Tool for Visualizing how Corpora Differ. In Proceedings of ACL 2017, System Demonstrations, pages 85–90, Vancouver, Canada. Association for Computational Linguistics.** https://doi.org/10.48550/arXiv.1703.00565')
    
        with tab3:
            st.markdown('**Sánchez-Franco, M. J., & Rey-Tienda, S. (2023). The role of user-generated content in tourism decision-making: an exemplary study of Andalusia, Spain. Management Decision, 62(7).** https://doi.org/10.1108/md-06-2023-0966')
            st.markdown('**Marrone, M., & Linnenluecke, M.K. (2020). Interdisciplinary Research Maps: A new technique for visualizing research topics. PLoS ONE, 15.** https://doi.org/10.1371/journal.pone.0242283')
            st.markdown('**Moreno, A., & Iglesias, C.A. (2021). Understanding Customers’ Transport Services with Topic Clustering and Sentiment Analysis. Applied Sciences.** https://doi.org/10.3390/app112110169')
            st.markdown('**Santosa, F. A. (2025). Artificial Intelligence in Library Studies: A Textual Analysis. JLIS.It, 16(1).** https://doi.org/10.36253/jlis.it-626')
    except Exception as exc:
        ui.fail("Please ensure that your file is correct. Please contact us if you find that this is an error.", exc)
