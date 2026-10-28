import re

import plotly.express as px
import streamlit as st

from tools import ui

ui.setup_page()
ui.render_menu()

with st.expander("Before you start", expanded=False):
     
        tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download"])
        with tab1:
            st.write("Sunburst's ability to present a thorough and intuitive picture of complex hierarchical data is an essential benefit. Librarians can easily browse and grasp the relationships between different levels of the hierarchy by employing sunburst visualizations. Sunburst visualizations can also be interactive, letting librarians and users drill down into certain categories or subcategories for further information. This interactive and visually appealing depiction improves the librarian's understanding of the collection and provides users with an engaging and user-friendly experience, resulting in improved information retrieval and decision-making.")
            
        with tab2:
            st.text("1. Put your CSV file.")
            st.text("2. You can set the range of years to see how it changed.")
            st.text("3. The sunburst has 3 levels. The inner circle is the type of data, meanwhile, the middle is the source title and the outer is the year the article was published.")
            st.text("4. The size of the slice depends on total documents. The average of inner and middle levels is calculated by formula below:")
            st.code('avg = sum(a * weights) / sum(weights)', language='python')
            
        with tab3:
            st.code("""
            +----------------+------------------------+--------------------------------------+
            |     Source     |       File Type        |     Column                           |
            +----------------+------------------------+--------------------------------------+
            | Scopus         | Comma-separated values | Source title,                        |
            |                | (.csv)                 | Document Type,                       |
            +----------------+------------------------| Cited by, Year                       |
            | Web of Science | Tab delimited file     |                                      |
            |                | (.txt)                 |                                      |
            +----------------+------------------------+--------------------------------------+
            | Lens.org       | Comma-separated values | Publication Year,                    |
            |                | (.csv)                 | Publication Type,                    | 
            |                |                        | Source Title,                        |
            |                |                        | Citing Works Count                   |
            +----------------+------------------------+--------------------------------------+
            | OpenAlex       | Comma-separated values | publication_year,                    |
            |                | (.csv)                 | cited_by_count,                      | 
            |                |                        | type,                                |
            |                |                        | primary_location.source.display_name |
            +----------------+------------------------+--------------------------------------+
            | Hathitrust     | .json                  | htid(Hathitrust ID)                  |
            +----------------+------------------------+--------------------------------------+
            """, language=None)          

        with tab4:  
            st.subheader(':blue[Sunburst]', anchor=False)
            st.text("Click the camera icon on the top right menu (you may need to hover your cursor within the visualization)")
            st.image("images/download_bertopic.jpg")
            st.subheader(":blue[Download table as CSV]", anchor=False)
            st.text("Hover cursor over table, and click download arrow")
            st.image("images/tablenetwork.png")

st.header("Sunburst Visualization", anchor=False)
st.subheader('Put your file here...', anchor=False)

REQUIRED = ['Document Type', 'Source title', 'Cited by', 'Year']


def build_sunburst(papers, year_range, cite_range, source, filter_col, filter_terms, invert):
    """Hitung tabel dan figur sunburst. Fungsi murni: semua masukan eksplisit."""
    data = papers.copy()
    data['Cited by'] = data['Cited by'].fillna(0)      # sitasi kosong dihitung 0, bukan dibuang
    data = data[data['Year'].between(*year_range) & data['Cited by'].between(*cite_range)]
    if source:
        data = data[data['Source title'] == source]    # pencocokan persis, bukan regex/substring

    terms = [t.strip() for t in filter_terms.split(';') if t.strip()]
    if terms:
        pattern = '|'.join(re.escape(t) for t in terms)
        hit = data[filter_col].astype(str).str.contains(pattern, case=False, na=False, regex=True)
        data = data[hit] if invert else data[~hit]

    before = len(data)
    data = data.dropna(subset=['Document Type', 'Source title', 'Year'])
    dropped = before - len(data)
    if data.empty:
        return None, None, dropped

    viz = (data.assign(Year=data['Year'].astype(int))
               .groupby(['Document Type', 'Source title', 'Year'])['Cited by']
               .agg(['sum', 'count']).reset_index()
               .rename(columns={'Document Type': 'doctype', 'Source title': 'source', 'Year': 'year',
                                'sum': 'cited by', 'count': 'total docs'}))
    viz['avg cited by'] = viz['cited by'] / viz['total docs']

    if viz['cited by'].sum() > 0:
        # warna = rata-rata sitasi per dokumen; simpul induk otomatis menjadi rata-rata
        # berbobot jumlah dokumen anak = total sitasi / total dokumen.
        fig = px.sunburst(viz, path=['doctype', 'source', 'year'], values='total docs',
                          color='avg cited by', color_continuous_scale='RdBu',
                          color_continuous_midpoint=viz['cited by'].sum() / viz['total docs'].sum())
    else:
        fig = px.sunburst(viz, path=['doctype', 'source', 'year'], values='total docs',
                          color='total docs', color_continuous_scale='Blues')
    fig.update_layout(height=800)
    return fig, viz, dropped


uploaded_file = ui.uploader()

if uploaded_file is not None:
    papers = ui.load(uploaded_file)
    if papers is not None:
        tab1, tab2 = st.tabs(["📈 Generate visualization", "📓 Recommended Reading"])

        with tab1:
            missing = [c for c in REQUIRED if c not in papers.columns]
            if missing or not papers['Year'].notna().any():
                st.error('We require these columns: Document Type, Source title, Cited by, Year', icon="🚨")
            else:
                try:
                    cites = papers['Cited by'].fillna(0)
                    MIN, MAX = int(papers['Year'].min()), int(papers['Year'].max())
                    MIN1, MAX1 = int(cites.min()), int(cites.max())
                    list_stitle = sorted(papers['Source title'].dropna().unique())

                    stitle = st.selectbox('Focus on', list_stitle, index=None)
                    col1, col2 = st.columns(2)
                    if MIN != MAX:
                        YEAR = col1.slider('Year', min_value=MIN, max_value=MAX, value=(MIN, MAX))
                    else:
                        col1.write(f'You only have data in {MAX}')
                        YEAR = (MIN, MAX)
                    if MIN1 != MAX1:
                        KEYLIM = col2.slider('Cited By Count', min_value=MIN1, max_value=MAX1, value=(MIN1, MAX1))
                    else:
                        col2.write('Your data has no varying citation counts.')
                        KEYLIM = (MIN1, MAX1)

                    with st.expander("Filtering settings"):
                        invert_keys = st.toggle("Invert keys")
                        filtered_keys = st.text_input("Filter words in source, seperate with semicolon (;)")
                        text_cols = ui.text_columns(papers)
                        select_col = st.selectbox("Column to filter from", text_cols,
                                                  index=text_cols.index('Source title') if 'Source title' in text_cols else 0)

                    # Hasil disimpan per sesi supaya tidak hilang saat widget lain berubah.
                    if st.button("Submit"):
                        st.session_state['sunburst_result'] = build_sunburst(
                            papers, YEAR, KEYLIM, stitle, select_col, filtered_keys, invert_keys)

                    result = st.session_state.get('sunburst_result')
                    if result:
                        fig, viz, dropped = result
                        if fig is None:
                            st.warning('No documents match your filters.', icon="⚠️")
                        else:
                            if dropped:
                                st.caption(f'{dropped} documents without Document Type, Source title or Year were left out.')
                            st.plotly_chart(fig)
                            st.dataframe(viz)
                except Exception as exc:
                    ui.fail("Please ensure that your file is correct. Please contact us if you find that this is an error.", exc)

        with tab2:
            st.markdown('**numpy.average — NumPy v1.24 Manual. (n.d.). Numpy.Average — NumPy v1.24 Manual.** https://numpy.org/doc/stable/reference/generated/numpy.average.html')
            st.markdown('**Sunburst. (n.d.). Sunburst Charts in Python.** https://plotly.com/python/sunburst-charts/')
