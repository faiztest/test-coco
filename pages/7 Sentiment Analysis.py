import re

import pandas as pd
import plotly.express as px
import streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tools import textprep, ui

ui.setup_page()
ui.render_menu()

with st.expander("Before you start", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download"])
    with tab1:
        st.write('Sentiment analysis uses natural language processing to identify patterns in large text datasets, revealing the writer’s opinions, emotions, and attitudes. It assesses subjectivity (objective vs. subjective), polarity (positive, negative, neutral), and emotions (e.g., anger, joy, sadness, surprise, jealousy).') 
        st.divider()
        st.write('💡 The idea came from this:')
        st.write('Lamba, M., & Madhusudhan, M. (2021, July 31). Sentiment Analysis. Text Mining for Information Professionals, 191–211. https://doi.org/10.1007/978-3-030-85085-2_7')
        
    with tab2:
        st.write("1. Put your file. Choose your prefered column to analyze")
        st.write("2. Choose your preferred method and decide which words you want to remove")
        st.write("3. Finally, you can visualize your data.")
        
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
        st.subheader(':blue[Sentiment Analysis]', anchor=False)
        st.write("Click the three dots at the top right then select the desired format")
        st.image("images/download_sentiment.png")
        st.divider()
        st.subheader(':blue[CSV Results]', anchor=False)
        st.text("Click Download button")
        st.image("images/sentitable.png")

st.header("Sentiment Analysis", anchor=False)
st.subheader('Put your file here...', anchor=False)

VADER_COLUMNS = {'neg': 'Negativity', 'neu': 'Neutrality', 'pos': 'Positivity', 'compound': 'Compound'}


def prepare_sentences(papers, col, rem_punc, rem_copyright, words_to_remove):
    """Teks yang dinilai: huruf kecil, (opsional) tanda baca & copyright dibuang, lalu kata pilihan pengguna dibuang."""
    paper = textprep.normalize_text(papers, col, lower=True, rem_punc=rem_punc, rem_copyright=rem_copyright)
    words = textprep.parse_word_list(words_to_remove)
    sentences = paper[col]
    if words:
        pattern = r'\b(?:' + "|".join(map(re.escape, words)) + r')\b'
        sentences = sentences.str.replace(pattern, '', regex=True)
    paper['Sentences__'] = sentences.str.split().str.join(' ')      # rapikan spasi ganda
    return paper


@st.cache_data(max_entries=3, ttl=1800, show_spinner="Scoring texts...")
def vader_scores(texts: tuple) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    return pd.DataFrame([analyzer.polarity_scores(t) for t in texts]).rename(columns=VADER_COLUMNS)[list(VADER_COLUMNS.values())]


@st.cache_data(max_entries=3, ttl=1800, show_spinner="Scoring texts...")
def textblob_scores(texts: tuple) -> pd.DataFrame:
    """Per dokumen: polaritas/subjektivitas dokumen + frasa bermuatan sentimen pertama."""
    rows = []
    for text in texts:
        polarity, subjectivity, assessments = TextBlob(text).sentiment_assessments
        if assessments:
            words, phrase_pol, phrase_subj = assessments[0][0], assessments[0][1], assessments[0][2]
            rows.append((polarity, subjectivity, ' '.join(words), phrase_pol, phrase_subj))
        else:
            rows.append((polarity, subjectivity, None, 0.0, 0.0))
    return pd.DataFrame(rows, columns=['Polarity', 'Subjectivity', 'Phrase', 'Phrase polarity', 'Phrase subjectivity'])


def label_polarity(value):
    return "Positive" if value > 0 else "Negative" if value < 0 else "Neutral"


uploaded_file = ui.uploader()

if uploaded_file is not None:
    try:
        papers = ui.load(uploaded_file)
        if papers is None:
            st.stop()
        coldf = ui.text_columns(papers)

        c1, c2 = st.columns(2)
        ColCho = c1.selectbox('Choose column', coldf)
        method = c2.selectbox('Choose method', ['TextBlob', 'NLTKvader'])
        words_to_remove = c1.text_input("Remove specific words. Separate words by semicolons (;)")
        wordcount = int(c2.number_input(label="Words displayed", min_value=1, step=1, value=5))
        rem_copyright = c1.toggle('Remove copyright statement', value=True)
        rem_punc = c2.toggle('Remove punctuation', value=True)

        paper = prepare_sentences(papers, ColCho, rem_punc, rem_copyright, words_to_remove)
        texts = tuple(paper['Sentences__'].tolist())

        if method == 'NLTKvader':
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Result", "📃 Reference", "📓 Recommended Reading", "⬇️ Download Help"])
            with tab1:
                scoreframe = vader_scores(texts).assign(Phrase=texts)
                grouped = (scoreframe.groupby(['Phrase'] + list(VADER_COLUMNS.values()), as_index=False)
                           .size().rename(columns={'size': 'Count'}))
                # tampilkan yang paling sering muncul (seri: Compound tertinggi), bukan urutan alfabet
                shown = grouped.sort_values(['Count', 'Compound'], ascending=False).head(wordcount)
                with st.expander("Sentence and Results"):
                    st.dataframe(shown.rename(columns={'Phrase': 'Sentence'}), hide_index=True)

            with tab2:
                st.markdown('**Hutto, C. and Gilbert, E. (2014) ‘VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text’, Proceedings of the International AAAI Conference on Web and Social Media, 8(1), pp. 216–225.** https://doi.org/10.1609/icwsm.v8i1.14550')

            with tab3:
                st.markdown('**Lamba, M., & Madhusudhan, M. (2021, July 31). Sentiment Analysis. Text Mining for Information Professionals, 191–211.** https://doi.org/10.1007/978-3-030-85085-2_7')
            with tab4:
                st.subheader(':blue[CSV Results]', anchor=False)
                st.text("Click Download button")
                st.image("images/sentitable.png")

        else:
            scores = textblob_scores(texts)
            with_phrase = scores.dropna(subset=['Phrase'])
            phraseframe = (with_phrase.groupby(['Phrase', 'Phrase polarity', 'Phrase subjectivity'], as_index=False)
                           .size().rename(columns={'size': 'Count', 'Phrase polarity': 'Polarity',
                                                   'Phrase subjectivity': 'Subjectivity'}))
            phraseframe['Score'] = phraseframe['Polarity'].apply(label_polarity)

            def top(label):
                part = phraseframe[phraseframe['Score'] == label]
                return part.sort_values('Count', ascending=False).head(wordcount).reset_index(drop=True)

            tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "📓 Recommended Reading"])
            with tab1:
                missing = int(scores['Phrase'].isna().sum())
                if missing:
                    st.caption(f'{missing} document(s) contain no sentiment-bearing phrase and are left out of the tables below.')
                for label, color in (("Positive", None), ("Negative", "#e57d7d"), ("Neutral", "#737a72")):
                    with st.expander(f"{label} Sentiment"):
                        part = top(label)
                        st.dataframe(part)
                        options = {'color_discrete_sequence': [color]} if color else {}
                        st.plotly_chart(px.bar(part, x="Phrase", y="Count", labels={"Phrase": "Word"}, **options))

                with st.expander("Sentence and Results"):
                    st.dataframe(pd.DataFrame({'Sentence': paper['Sentences__'].to_numpy(),
                                               'Polarity': scores['Polarity'].to_numpy(),
                                               'Subjectivity': scores['Subjectivity'].to_numpy()}))

            with tab2:
                st.markdown('**Steven, L. et al. (2018) TextBlob: Simplified Text Processing — TextBlob 0.15.2 documentation, Readthedocs.io.** https://textblob.readthedocs.io/en/dev/')

            with tab3:
                st.markdown('**Lamba, M., & Madhusudhan, M. (2021, July 31). Sentiment Analysis. Text Mining for Information Professionals, 191–211.** https://doi.org/10.1007/978-3-030-85085-2_7')
    except Exception as exc:
        ui.fail("Please ensure that your file is correct. Please contact us if you find that this is an error.", exc)
