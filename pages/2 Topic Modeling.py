import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from tools import textprep, ui

ui.setup_page()
ui.render_menu()

with st.expander("Before you start", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download"])
        with tab1:
            st.write("Topic modeling has numerous advantages for librarians in different aspects of their work. A crucial benefit is an ability to quickly organize and categorize a huge volume of textual content found in websites, institutional archives, databases, emails, and reference desk questions. Librarians can use topic modeling approaches to automatically identify the primary themes or topics within these documents, making navigating and retrieving relevant information easier. Librarians can identify and understand the prevailing topics of discussion by analyzing text data with topic modeling tools, allowing them to assess user feedback, tailor their services to meet specific needs and make informed decisions about collection development and resource allocation. Making ontologies, automatic subject classification, recommendation services, bibliometrics, altmetrics, and better resource searching and retrieval are a few examples of topic modeling. To do topic modeling on other text like chats and surveys, change the column name to 'Abstract' in your file.")
            st.divider()
            st.write('💡 The idea came from this:')
            st.write('Lamba, M., & Madhusudhan, M. (2021, July 31). Topic Modeling. Text Mining for Information Professionals, 105–137. https://doi.org/10.1007/978-3-030-85085-2_4')
        with tab2:
            st.text("1. Put your file. Choose your preferred column.")
            st.text("2. Choose your preferred method. LDA is the most widely used, whereas Biterm is appropriate for short text, and BERTopic works well for large text data as well as supports more than 50+ languages.")
            st.text("3. Finally, you can visualize your data.")
            st.error("This app includes lemmatization and stopwords for the abstract text. Currently, we only offer English words.", icon="💬")
            
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
            st.subheader(':blue[pyLDA]', anchor=False)
            st.button('Download image.', disabled=True)
            st.text("Click Download Image button.")
            
            st.divider()
            st.subheader(':blue[Biterm]', anchor=False)
            st.text("Click the three dots at the top right then select the desired format.")
            st.image("images/download_biterm.jpg")
            
            st.divider()
            st.subheader(':blue[BERTopic]', anchor=False)
            st.text("Click the camera icon on the top right menu")
            st.image("images/download_bertopic.jpg")
            st.divider()
            st.subheader(':blue[CSV Result]', anchor=False)
            st.text("Click Download button")
            st.button('Download Results.', disabled=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"      # menghindari deadlock tokenizer

st.header("Topic Modeling", anchor=False)
st.subheader('Put your file here...', anchor=False)

LLM_OPTIONS = ["OpenAI/gpt-4o-mini", "Google/flan-t5", "LiquidAI/LFM2-350M"]


#===persiapan korpus===
def corpus_column(papers):
    """Pilihan kolom: Abstract/Title/Abstract + Title bila ada, selain itu semua kolom teks."""
    has_abs, has_title = 'Abstract' in papers.columns, 'Title' in papers.columns
    if has_abs and has_title:
        return ["Abstract", "Title", "Abstract + Title"]
    if has_abs or has_title:
        return ["Abstract"] if has_abs else ["Title"]
    return ui.text_columns(papers)


def build_corpus(papers, col, rem_punc, rem_copyright, words_to_remove):
    frame = papers.copy()
    if col == "Abstract + Title":
        frame[col] = (frame['Title'].fillna('').astype(str) + '. ' + frame['Abstract'].fillna('').astype(str)).str.strip('. ')
    cleaned = textprep.clean_dataframe(frame, col, rem_punc=rem_punc, rem_copyright=rem_copyright,
                                       remove_words=textprep.parse_word_list(words_to_remove), punct="all")
    texts = [t for t in cleaned[col].tolist() if t.strip()]       # dokumen kosong dibuang
    return texts


def build_phrases(tokenized, xgram, threshold, min_count):
    """N-gram lewat gensim Phrases: 1 = tidak ada frasa, 2 = bigram, 3 = bigram + trigram."""
    from gensim.models import Phrases
    from gensim.models.phrases import Phraser
    if xgram <= 1:
        return tokenized
    bigram = Phraser(Phrases(tokenized, min_count=min_count, threshold=threshold))
    docs = [bigram[d] for d in tokenized]
    if xgram >= 3:
        trigram = Phraser(Phrases(docs, min_count=min_count, threshold=threshold))
        docs = [trigram[d] for d in docs]
    return docs


#===pyLDA===
def screenshot_html(html):
    """Screenshot HTML lewat Chrome headless (opsional). Memakai folder sementara unik, bukan nama file tetap."""
    try:
        from html2image import Html2Image
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'output.html')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(html)
            hti = Html2Image(output_path=tmp)
            hti.browser.flags = ['--default-background-color=ffffff', '--hide-scrollbars']
            hti.browser.use_new_headless = None
            hti.screenshot(other_file=path, css_str="body {background: white;}", size=(1500, 800), save_as='ldavis_img.png')
            with open(os.path.join(tmp, 'ldavis_img.png'), 'rb') as handle:
                return handle.read()
    except Exception:
        return None


def run_pylda(topic_abs, num_topic, random_state, chunksize, passes, xgram, threshold, min_count, relevance):
    import pyLDAvis
    import pyLDAvis.gensim_models
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    from gensim.models.coherencemodel import CoherenceModel

    docs = build_phrases([t.split(' ') for t in topic_abs], xgram, threshold, min_count)
    id2word = Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topic, random_state=random_state,
                         chunksize=chunksize, passes=passes, alpha='auto', per_word_topics=False)
    topics = lda_model.show_topics(num_topics=num_topic, num_words=30, formatted=False)   # semua topik (default gensim: 10)
    coherence = CoherenceModel(model=lda_model, texts=docs, dictionary=id2word, coherence='c_v', processes=1).get_coherence()

    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    html = pyLDAvis.prepared_data_to_html(vis)
    # λ awal untuk panel pyLDAvis (dibaca dari location.hash oleh LDAvis; slider λ di panel tetap bisa diubah)
    html = f'<script>try{{location.hash="#topic=0&lambda={relevance}&term=";}}catch(e){{}}</script>' + html

    rows = [(topic_id + 1, rank + 1, word, prob) for topic_id, words in topics for rank, (word, prob) in enumerate(words)]
    result = pd.DataFrame(rows, columns=['Topic', 'Rank', 'Term', 'Weight'])
    return {'html': html, 'coherence': coherence, 'png': screenshot_html(html),
            'csv': result.to_csv(index=False).encode('utf-8')}


#===Biterm===
def run_biterm(topic_abs, num_topic, seed, iterations, xgram, threshold, min_count):
    import bitermplus as btm
    import tmplot as tmp

    docs = build_phrases([t.split(' ') for t in topic_abs], xgram, threshold, min_count)
    texts = [' '.join(doc) for doc in docs]
    X, vocabulary, _ = btm.get_words_freqs(texts)
    docs_vec = btm.get_vectorized_docs(texts, vocabulary)           # memakai teks ber-n-gram yang SAMA dengan kosakata
    biterms = btm.get_biterms(docs_vec)
    model = btm.BTM(X, vocabulary, seed=seed, T=num_topic, M=20, alpha=50 / num_topic, beta=0.01)   # alpha = 50/T
    model.fit(biterms, iterations=iterations)
    try:                                                             # bitermplus/tmplot baru butuh korpus untuk p(z|d)
        coords = tmp.prepare_coords(model, corpus=docs_vec)
    except TypeError:                                                # tmplot lama tidak punya argumen corpus
        coords = tmp.prepare_coords(model)
    try:
        perplexity = model.perplexity_                               # bitermplus < 1.0
    except Exception:                                                # bitermplus >= 1.0: hitung dari p(z|d)
        p_zd = model.transform(docs_vec)
        perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, model.topics_num_)
    return {'coords': coords, 'phi': tmp.get_phi(model), 'totaltop': coords.label.values.tolist(),
            'perplexity': float(perplexity), 'csv': model.df_words_topics_.to_csv().encode('utf-8')}


#===BERTopic===
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder(name):
    if name == 'en_core_web_sm':
        import spacy
        return spacy.load('en_core_web_sm', exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def run_bertopic(topic_abs, num_topic, xgram, embed_name, top_n_words, random_state, n_components, n_neighbors,
                 fine_tuning, topic_labelling, llm_provider, api_key):
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, TextGeneration
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0, metric='cosine',
                      random_state=random_state)
    cluster_model = KMeans(n_clusters=num_topic, random_state=random_state, n_init=10)
    model = get_embedder(embed_name)
    if embed_name == 'en_core_web_sm':
        embeddings = np.array([doc.vector for doc in model.pipe(topic_abs)])
    else:
        embeddings = model.encode(topic_abs, show_progress_bar=False)
    lang = 'multilingual' if embed_name.startswith('paraphrase-multilingual') else 'en'

    representation_model = ""
    if fine_tuning:
        keybert, mmr = KeyBERTInspired(), MaximalMarginalRelevance(diversity=0.3)
        representation_model = {"KeyBERT": keybert, "MMR": mmr}
        if topic_labelling:
            if llm_provider == "OpenAI/gpt-4o-mini":
                import openai
                representation_model["labels"] = OpenAI(openai.OpenAI(api_key=api_key), model="gpt-4o-mini",
                                                        delay_in_seconds=1)
            else:
                from transformers import pipeline
                if llm_provider == "Google/flan-t5":
                    pipe = pipeline("text2text-generation", model="google/flan-t5-base")
                else:
                    pipe = pipeline("text-generation", model="LiquidAI/LFM2-350M")
                representation_model["labels"] = TextGeneration(pipe)

    vectorizer_model = CountVectorizer(ngram_range=(1, xgram), stop_words='english')
    topic_model = BERTopic(representation_model=representation_model, embedding_model=model, hdbscan_model=cluster_model,
                           language=lang, umap_model=umap_model, vectorizer_model=vectorizer_model,
                           top_n_words=top_n_words)
    topic_model.fit_transform(topic_abs, embeddings=embeddings)

    if fine_tuning and topic_labelling:
        labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["labels"].values()]
        topic_model.set_topic_labels(labels)

    figures = {
        'topics': topic_model.visualize_topics(custom_labels=True),
        'terms': topic_model.visualize_barchart(top_n_topics=num_topic, custom_labels=True),
        'documents': topic_model.visualize_document_datamap(topic_abs, embeddings=embeddings, custom_labels=True),
        'hierarchy': topic_model.visualize_hierarchy(top_n_topics=num_topic, custom_labels=True),
        'heatmap': topic_model.visualize_heatmap(n_clusters=max(num_topic - 1, 1), width=1000, height=1000, custom_labels=True),
    }
    return {'figures': figures, 'csv': topic_model.get_topic_info().to_csv().encode('utf-8')}


def show_figure(fig):
    """Plotly -> st.plotly_chart; Matplotlib (datamap BERTopic) -> st.pyplot."""
    if hasattr(fig, 'to_plotly_json'):
        st.plotly_chart(fig)
    else:
        st.pyplot(fig)


uploaded_file = ui.uploader()

if uploaded_file is not None:
    try:
        papers = ui.load(uploaded_file)
        if papers is None:
            st.stop()
        file_id = getattr(uploaded_file, 'file_id', None) or uploaded_file.name

        c1, c2, c3 = st.columns([3, 3, 4])
        method = c1.selectbox('Choose method', ('Choose...', 'pyLDA', 'Biterm', 'BERTopic'))
        ColCho = c2.selectbox('Choose column', corpus_column(papers))
        num_cho = c3.number_input('Choose number of topics', min_value=2, max_value=30, value=5)

        d1, d2 = st.columns([3, 7])
        xgram = int(d1.selectbox("N-grams", ("1", "2", "3")))
        words_to_remove = d2.text_input("Remove specific words. Separate words by semicolons (;)")
        rem_copyright = d1.toggle('Remove copyright statement', value=True)
        rem_punc = d2.toggle('Remove punctuation', value=True)

        #===advance settings===
        settings = {}
        with st.expander("🧮 Show advance settings"):
            t1, t2, t3, t4 = st.columns(4)
            if method == 'pyLDA':
                settings['random_state'] = t1.number_input('Random state', min_value=0, max_value=None, step=1, help='Ensuring the reproducibility.')
                settings['chunksize'] = t2.number_input('Chunk size', value=100, min_value=10, max_value=None, step=1, help='Number of documents to be used in each training chunk.')
                settings['threshold'] = t3.number_input('Threshold (Gensim)', value=100, min_value=1, max_value=None, step=1, help='Lower = More phrases. Higher = Fewer phrases.')
                settings['relevance'] = t4.number_input('Lambda (λ)', value=0.6, min_value=0.0, max_value=1.0, step=0.01, help='Initial relevance weight in the pyLDAvis panel. Lower = More unique. Higher = More frequent. You can still move the λ slider inside the panel.')
                u1, u2 = st.columns(2)
                settings['passes'] = u1.number_input('Passes', value=10, min_value=1, max_value=200, step=1, help='How many times the model goes through the whole corpus. More passes = more stable topics, but slower.')
                settings['min_count'] = u2.number_input('Minimum phrase count', value=3, min_value=1, step=1, help='Only used when N-grams > 1.')
            elif method == 'Biterm':
                settings['seed'] = t1.number_input('Random state seed', value=100, min_value=1, max_value=None, step=1, help='Ensuring the reproducibility.')
                settings['iterations'] = t2.number_input('Iterations number', value=20, min_value=2, max_value=None, step=1, help='Number of Gibbs sampling iterations. 20 is fast but rough; a few hundred gives more stable topics.')
                settings['threshold'] = t3.number_input('Threshold (Gensim)', value=100, min_value=1, max_value=None, step=1, help='Lower = More phrases. Higher = Fewer phrases.')
                settings['min_count'] = t4.number_input('Minimum phrase count', value=3, min_value=1, step=1, help='Only used when N-grams > 1.')
            elif method == 'BERTopic':
                u1, u2, u3 = st.columns([5, 2.5, 2.5])
                settings['top_n_words'] = t1.number_input('top_n_words', value=5, min_value=5, max_value=25, step=1, help='Number of words per topic.')
                settings['random_state'] = t2.number_input('random_state', value=42, min_value=1, max_value=None, step=1, help="Seed for UMAP and KMeans. With the same library versions the result should be reproducible.")
                settings['n_components'] = t3.number_input('n_components', value=5, min_value=1, max_value=None, step=1, help='The dimensionality of the embeddings after reducing them.')
                settings['n_neighbors'] = t4.number_input('n_neighbors', value=15, min_value=1, max_value=None, step=1, help='The number of neighboring sample points used when making the manifold approximation.')
                settings['embed_name'] = u1.radio("embedding_model", ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2", "en_core_web_sm"], index=0, horizontal=True, help='Select paraphrase-multilingual if your documents are in a language other than English.')
                settings['fine_tuning'] = u2.toggle("Use Fine-tuning")
                settings['topic_labelling'] = False
                settings['llm_provider'] = None
                settings['api_key'] = ""
                if settings['fine_tuning']:
                    settings['topic_labelling'] = u3.toggle("Automatic topic labelling")
                    if settings['topic_labelling']:
                        v1, v2 = st.columns(2)
                        settings['llm_provider'] = v1.selectbox("Model", LLM_OPTIONS)
                        v2.warning("AI may make mistakes, so please do not rely on it alone. Always apply human judgment when interpreting the analysis.", icon="⚠️")
                        if settings['llm_provider'] == "OpenAI/gpt-4o-mini":
                            settings['api_key'] = v1.text_input("API Key", type="password")
                            v2.error("AI consumes a significant amount of energy. Consider the environmental impact when using it.", icon="⚠️")
            else:
                st.write('Please choose your preferred method')

        if method == 'BERTopic':
            st.info("BERTopic is an expensive process when dealing with a large volume of text with our existing resources. Please kindly wait until the visualization appears.")

        #===Submit: hitung sekali, simpan hasil per sesi===
        signature = repr((file_id, method, ColCho, int(num_cho), xgram, words_to_remove, rem_copyright, rem_punc,
                          sorted((k, v) for k, v in settings.items() if k != 'api_key')))
        if st.button("Submit") and method != 'Choose...':
            topic_abs = build_corpus(papers, ColCho, rem_punc, rem_copyright, words_to_remove)
            num_topic = int(num_cho)
            if len(topic_abs) < max(num_topic, 5):
                st.error('🙇‍♂️ There are too few usable documents for this number of topics. Please lower the number of topics.')
            else:
                with st.spinner('Performing computations. Please wait ...'):
                    try:
                        if method == 'pyLDA':
                            data = run_pylda(topic_abs, num_topic, int(settings['random_state']), int(settings['chunksize']),
                                             int(settings['passes']), xgram, settings['threshold'], int(settings['min_count']),
                                             float(settings['relevance']))
                        elif method == 'Biterm':
                            data = run_biterm(topic_abs, num_topic, int(settings['seed']), int(settings['iterations']),
                                              xgram, settings['threshold'], int(settings['min_count']))
                        else:
                            data = run_bertopic(topic_abs, num_topic, xgram, settings['embed_name'], int(settings['top_n_words']),
                                                int(settings['random_state']), int(settings['n_components']),
                                                int(settings['n_neighbors']), settings['fine_tuning'],
                                                settings['topic_labelling'], settings['llm_provider'], settings['api_key'])
                        st.session_state['topic_result'] = {'method': method, 'file_id': file_id, 'signature': signature, 'data': data}
                    except ValueError as exc:
                        st.session_state.pop('topic_result', None)
                        ui.fail('🙇‍♂️ Please raise the number of topics or check your data, then click Submit.', exc)

        stored = st.session_state.get('topic_result')
        has_result = bool(stored) and stored['method'] == method and stored['file_id'] == file_id
        if has_result and stored['signature'] != signature:
            st.info('Your settings have changed. Click Submit to update the result.', icon="ℹ️")
        data = stored['data'] if has_result else None

        #===topic===
        if method == 'Choose...':
            st.write('')

        elif method == 'pyLDA':
            tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading"])

            with tab1:
                if data is None:
                    st.warning('🖱️ Please click Submit')
                else:
                    st.write('Coherence score: ', data['coherence'])
                    ui.show_html(data['html'], height=800)
                    st.markdown('Copyright (c) 2015, Ben Mabey. https://github.com/bmabey/pyLDAvis')
                    e1, e2, e3 = st.columns(3)
                    if data['png']:
                        e1.download_button(label="Download image", data=data['png'], file_name="ldavis_img.png", mime="image/png")
                    else:
                        e1.download_button(label="Download interactive HTML", data=data['html'], file_name="ldavis.html", mime="text/html")
                    e2.download_button(label="Download Results", data=data['csv'], file_name="results.csv", mime="text/csv", on_click="ignore")

            with tab2:
                st.markdown('**Sievert, C., & Shirley, K. (2014). LDAvis: A method for visualizing and interpreting topics. Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces.** https://doi.org/10.3115/v1/w14-3110')
            with tab3:
                st.markdown('**Chen, X., & Wang, H. (2019, January). Automated chat transcript analysis using topic modeling for library reference services. Proceedings of the Association for Information Science and Technology, 56(1), 368–371.** https://doi.org/10.1002/pra2.31')
                st.markdown('**Joo, S., Ingram, E., & Cahill, M. (2021, December 15). Exploring Topics and Genres in Storytime Books: A Text Mining Approach. Evidence Based Library and Information Practice, 16(4), 41–62.** https://doi.org/10.18438/eblip29963')
                st.markdown('**Lamba, M., & Madhusudhan, M. (2021, July 31). Topic Modeling. Text Mining for Information Professionals, 105–137.** https://doi.org/10.1007/978-3-030-85085-2_4')
                st.markdown('**Lamba, M., & Madhusudhan, M. (2019, June 7). Mapping of topics in DESIDOC Journal of Library and Information Technology, India: a study. Scientometrics, 120(2), 477–505.** https://doi.org/10.1007/s11192-019-03137-5')
        #===Biterm===
        elif method == 'Biterm':
            tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading"])
            with tab1:
                if data is None:
                    st.warning('🖱️ Please click Submit')
                else:
                    import tmplot as tmp
                    col1, col2 = st.columns([4, 6])
                    with col1:
                        st.write('Perplexity score: ', data['perplexity'])
                        st.write('')
                        numvis = st.selectbox('Choose topic', data['totaltop'])
                        st.altair_chart(tmp.plot_scatter_topics(data['coords'], size_col='size', label_col='label', topic=numvis))
                    with col2:
                        terms_probs = tmp.calc_terms_probs_ratio(data['phi'], topic=numvis, lambda_=1)
                        st.altair_chart(tmp.plot_terms(terms_probs, font_size=12))
                    st.download_button(label="Download Results", data=data['csv'], file_name="results.csv", mime="text/csv", on_click="ignore")

            with tab2: 
                st.markdown('**Yan, X., Guo, J., Lan, Y., & Cheng, X. (2013, May 13). A biterm topic model for short texts. Proceedings of the 22nd International Conference on World Wide Web.** https://doi.org/10.1145/2488388.2488514')
            with tab3:
                st.markdown('**Cai, M., Shah, N., Li, J., Chen, W. H., Cuomo, R. E., Obradovich, N., & Mackey, T. K. (2020, August 26). Identification and characterization of tweets related to the 2015 Indiana HIV outbreak: A retrospective infoveillance study. PLOS ONE, 15(8), e0235150.** https://doi.org/10.1371/journal.pone.0235150')
                st.markdown('**Chen, Y., Dong, T., Ban, Q., & Li, Y. (2021). What Concerns Consumers about Hypertension? A Comparison between the Online Health Community and the Q&A Forum. International Journal of Computational Intelligence Systems, 14(1), 734.** https://doi.org/10.2991/ijcis.d.210203.002')
                st.markdown('**George, Crissandra J., "AMBIGUOUS APPALACHIANNESS: A LINGUISTIC AND PERCEPTUAL INVESTIGATION INTO ARC-LABELED PENNSYLVANIA COUNTIES" (2022). Theses and Dissertations-- Linguistics. 48.** https://doi.org/10.13023/etd.2022.217')
                st.markdown('**Li, J., Chen, W. H., Xu, Q., Shah, N., Kohler, J. C., & Mackey, T. K. (2020). Detection of self-reported experiences with corruption on twitter using unsupervised machine learning. Social Sciences & Humanities Open, 2(1), 100060.** https://doi.org/10.1016/j.ssaho.2020.100060')
        #===BERTopic===
        elif method == 'BERTopic':
            tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading"])
            with tab1:
                if data is None:
                    st.warning('🖱️ Please click Submit')
                else:
                    figs = data['figures']
                    bertab1, bertab2, bertab3, bertab4, bertab5 = st.tabs(["Visualize Topics", "Visualize Terms", "Visualize Documents",
                                                                           "Visualize Document Hierarchy", "Visualize Topic Similarity"])
                    with bertab1:
                        show_figure(figs['topics'])
                    with bertab2:
                        show_figure(figs['terms'])
                    with bertab3:
                        show_figure(figs['documents'])
                    with bertab4:
                        show_figure(figs['hierarchy'])
                    with bertab5:
                        show_figure(figs['heatmap'])
                    st.download_button(label="Download Results", data=data['csv'], file_name="results.csv", mime="text/csv", on_click="ignore")

            with tab2:
                st.markdown('**Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.** https://doi.org/10.48550/arXiv.2203.05794')
            with tab3:
                st.markdown('**Jeet Rawat, A., Ghildiyal, S., & Dixit, A. K. (2022, December 1). Topic modelling of legal documents using NLP and bidirectional encoder representations from transformers. Indonesian Journal of Electrical Engineering and Computer Science, 28(3), 1749.** https://doi.org/10.11591/ijeecs.v28.i3.pp1749-1755')
                st.markdown('**Yao, L. F., Ferawati, K., Liew, K., Wakamiya, S., & Aramaki, E. (2023, April 20). Disruptions in the Cystic Fibrosis Community’s Experiences and Concerns During the COVID-19 Pandemic: Topic Modeling and Time Series Analysis of Reddit Comments. Journal of Medical Internet Research, 25, e45249.** https://doi.org/10.2196/45249')
    except Exception as exc:
        ui.fail("Please ensure that your file is correct. Don't forget to put your API key if you use OpenAI/gpt-4o-mini. Please contact us if you find that this is an error.", exc)
