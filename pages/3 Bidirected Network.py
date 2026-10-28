import inspect

import networkx as nx
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from streamlit_agraph import Config, Edge, Node, agraph

from tools import textprep, ui

ui.setup_page()
ui.render_menu()

with st.expander("Before you start", expanded=False):

        tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download"])
        with tab1:
            st.write("The use of network text analysis by librarians can be quite beneficial. Finding hidden correlations and connections in textual material is a significant advantage. Using network text analysis tools, librarians can improve knowledge discovery, obtain deeper insights, and support scholars meaningfully, ultimately enhancing the library's services and resources. This menu provides a two-way relationship instead of the general network of relationships to enhance the co-word analysis. Since it is based on ARM, you may obtain transactional data information using this menu. Please name the column in your file 'Keyword' instead.")
            st.divider()
            st.write('💡 The idea came from this:') 
            st.write('Santosa, F. A. (2023). Adding Perspective to the Bibliometric Mapping Using Bidirected Graph. Open Information Science, 7(1), 20220152. https://doi.org/10.1515/opis-2022-0152')

        with tab2:
            st.text("1. Put your file.")
            st.text("2. Choose your preferable method. Picture below may help you to choose wisely.")
            st.markdown("![Source: https://studymachinelearning.com/stemming-and-lemmatization/](https://studymachinelearning.com/wp-content/uploads/2019/09/stemmin_lemm_ex-1.png)")
            st.text('Source: https://studymachinelearning.com/stemming-and-lemmatization/')
            st.text("3. Choose the value of Support and Confidence. If you're not sure how to use it please read the article above or just try it!")
            st.text("4. You can see the table and a simple visualization before making a network visualization.")
            st.text('5. Click "Generate network visualization" to see the network')
            st.error("The more data on your table, the more you'll see on network.", icon="🚨")
            st.error("If the table contains many rows, the network will take more time to process. Please use it efficiently.", icon="⌛")
            
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
            |                |                        | and separate the words with ';' |
            +----------------+------------------------+---------------------------------+
            | Hathitrust     | .json                  | htid (Hathitrust ID)            |
            +----------------+------------------------+---------------------------------+
            """, language=None)    

        with tab4:
            st.subheader(":blue[Download visualization]", anchor=False)
            st.text("Zoom in, zoom out, or shift the nodes as desired, then right-click and select Save image as ...")
            st.image("images/download_bidirected.jpg")     
            st.subheader(":blue[Download table as CSV]", anchor=False)
            st.text("Hover cursor over table, and click download arrow")
            st.image("images/tablenetwork.png")

import altair_nx as anx

st.header("Bidirected Network", anchor=False)
st.subheader('Put your file here...', anchor=False)

RULE_COLUMNS = ['antecedents', 'consequents', 'antecedent support', 'consequent support',
                'support', 'confidence', 'lift', 'conviction']


@st.cache_data(max_entries=3, ttl=1800, show_spinner="Mining association rules...")
def mine_rules(transactions: tuple, support: float, confidence: float, max_len: int):
    """Frequent itemsets (FP-Growth) + aturan asosiasi. Semua masukan eksplisit."""
    encoder = TransactionEncoder()                      # satu objek per panggilan (dulu global & dipakai bersama)
    encoder.fit(transactions)
    try:    # matriks sparse hemat memori untuk kosakata besar; butuh scipy yang cukup baru untuk pandas
        onehot = pd.DataFrame.sparse.from_spmatrix(encoder.transform(transactions, sparse=True), columns=encoder.columns_)
        freq_item = fpgrowth(onehot, min_support=support, use_colnames=True, max_len=max_len)
    except ImportError:
        onehot = pd.DataFrame(encoder.transform(transactions), columns=encoder.columns_)
        freq_item = fpgrowth(onehot, min_support=support, use_colnames=True, max_len=max_len)
    if freq_item.empty:
        return freq_item, None
    kwargs = {}
    if 'num_itemsets' in inspect.signature(association_rules).parameters:   # mlxtend >= 0.23.2
        kwargs['num_itemsets'] = len(onehot)
    rules = association_rules(freq_item, metric='confidence', min_threshold=confidence, **kwargs)
    rules = rules[RULE_COLUMNS].copy()
    for col in ('antecedents', 'consequents'):
        rules[col] = rules[col].apply(lambda x: ', '.join(sorted(x)))
    return freq_item, rules


def build_edges_and_nodes(res):
    res = res.copy()
    res['to'] = (res['antecedents'] + ' → ' + res['consequents'] + '\n Support = ' + res['support'].astype(str)
                 + '\n Confidence = ' + res['confidence'].astype(str) + '\n Conviction = ' + res['conviction'].astype(str))
    res_ant = res[['antecedents', 'antecedent support']].rename(columns={'antecedents': 'node', 'antecedent support': 'size'})
    res_con = res[['consequents', 'consequent support']].rename(columns={'consequents': 'node', 'consequent support': 'size'})
    res_node = pd.concat([res_ant, res_con]).drop_duplicates(keep='first')
    return res_node, res


def draw_agraph(res_node, res):
    nodes = [Node(id=x, label=x, size=50 * w + 10, shape="dot", labelHighlightBold=True, group=x, opacity=10, mass=1)
             for w, x in zip(res_node['size'], res_node['node'])]
    edges = [Edge(source=y, target=z, title=b, width=a * 2, physics=True, smooth=True)
             for y, z, a, b in zip(res['antecedents'], res['consequents'], res['confidence'], res['to'])]
    config = Config(width=1200, height=800, directed=True, physics=True, hierarchical=False, maxVelocity=5)
    agraph(nodes=nodes, edges=edges, config=config)


def draw_altair(res_node, res, layout):
    graph = nx.DiGraph()                                  # graf baru tiap gambar (bukan hasil cache)
    for w, x in zip(res_node['size'], res_node['node']):
        graph.add_node(x, size=(400 + 2000 * w))
    for y, z, a in zip(res['antecedents'], res['consequents'], res['confidence']):
        graph.add_edge(y, z, weight=int(a * 10))
    layouts = {"Spring": nx.spring_layout, "Kamada Kawai": nx.kamada_kawai_layout,
               "Circular": nx.circular_layout, "Random": nx.random_layout, "Shell": nx.shell_layout}
    pos = layouts[layout](graph)
    chart = anx.draw_networkx(graph, pos, node_label='node', edge_width='weight', node_size='size',
                              curved_edges=True, node_font_size=12, edge_alpha=0.25, edge_colour="grey",
                              node_colour="royalblue", chart_width=800, chart_height=600).interactive()
    with st.container(border=True):
        st.altair_chart(chart)


uploaded_file = ui.uploader()

if uploaded_file is not None:
    try:
        papers = ui.load(uploaded_file)
        if papers is None:
            st.stop()
        list_of_column_key = ui.keyword_columns(papers)

        col1, col2 = st.columns(2)
        with col1:
            dispmethod = st.selectbox('Choose display method', ("Agraph", "Altair-nx"))
            method = st.selectbox('Choose method', ('Lemmatization', 'Stemming'))
        with col2:
            keyword = st.selectbox('Choose column', list_of_column_key)
            if dispmethod == "Altair-nx":
                layout = st.selectbox('Choose graph layout', ['Circular', 'Kamada Kawai', 'Random', 'Spring', 'Shell'])

        if keyword is None:
            st.error('Error: Please check your Author/Index Keywords column.')
            st.stop()

        normalize = textprep.make_keyword_normalizer(method)
        transactions = tuple(tuple(t) for t in textprep.keyword_lists(papers[keyword], normalize) if t)
        if len(transactions) < 2:
            st.error('We need at least 2 documents that contain keywords.', icon="🚨")
            st.stop()

        col1, col2, col3 = st.columns(3)
        with col1:
            supp = st.slider('Support', 0.001, 1.000, (0.010), help='Frequency of occurrence of keywords in a set of documents')
        with col2:
            conf = st.slider('Confidence', 0.001, 1.000, (0.050), help='Presence of keywords in documents that included the antecedents')
        with col3:
            maxlen = st.slider('Maximum length of the itemsets generated', 2, 8, (2), help='')

        tab1, tab2, tab3 = st.tabs(["📈 Result & Generate visualization", "📃 Reference", "📓 Recommended Reading"])

        with tab1:
            #===Association rules===
            col1, col2 = st.columns(2)
            with col1:
                st.write('🚨 The more data you have, the longer you will have to wait.')
            with col2:
                showall = st.checkbox('Show all nodes', value=True)

            freq_item, rules = mine_rules(transactions, float(supp), float(conf), int(maxlen))
            if freq_item.empty or rules is None or rules.empty:
                st.error('Please lower your value.', icon="🚨")
            else:
                restab = rules.assign(Show=bool(showall))
                restab = st.data_editor(restab, hide_index=True)
                res = restab[restab['Show'] == True]

                #===visualize===
                if st.button('📈 Generate network visualization'):
                    st.session_state['arul_selection'] = res     # grafik bertahan walau widget lain berubah

                chosen = st.session_state.get('arul_selection')
                if chosen is not None and not chosen.empty:
                    with st.spinner('Visualizing, please wait ....'):
                        res_node, res_edges = build_edges_and_nodes(chosen)
                        if dispmethod == "Agraph":
                            draw_agraph(res_node, res_edges)
                        else:
                            draw_altair(res_node, res_edges, layout)

        with tab2:
             st.markdown('**Santosa, F. A. (2023). Adding Perspective to the Bibliometric Mapping Using Bidirected Graph. Open Information Science, 7(1), 20220152.** https://doi.org/10.1515/opis-2022-0152')
             
        with tab3:
            st.markdown('**Agrawal, R., Imieliński, T., & Swami, A. (1993). Mining association rules between sets of items in large databases. In ACM SIGMOD Record (Vol. 22, Issue 2, pp. 207–216). Association for Computing Machinery (ACM).** https://doi.org/10.1145/170036.170072')
            st.markdown('**Brin, S., Motwani, R., Ullman, J. D., & Tsur, S. (1997). Dynamic itemset counting and implication rules for market basket data. ACM SIGMOD Record, 26(2), 255–264.** https://doi.org/10.1145/253262.253325')
            st.markdown('**Edmonds, J., & Johnson, E. L. (2003). Matching: A Well-Solved Class of Integer Linear Programs. Combinatorial Optimization — Eureka, You Shrink!, 27–30.** https://doi.org/10.1007/3-540-36478-1_3') 
            st.markdown('**Li, M. (2016, August 23). An exploration to visualise the emerging trends of technology foresight based on an improved technique of co-word analysis and relevant literature data of WOS. Technology Analysis & Strategic Management, 29(6), 655–671.** https://doi.org/10.1080/09537325.2016.1220518')
    except Exception as exc:
        ui.fail("Please ensure that your file is correct. Please contact us if you find that this is an error.", exc)
