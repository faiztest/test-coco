import re

import streamlit as st
from nltk.corpus import stopwords
from wordcloud import WordCloud

from tools import sourceformat as sf
from tools import textprep, ui

ui.setup_page()
ui.render_menu()

with st.expander("Before you start", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download Visualization"])
    with tab1:
        st.write("A word cloud is a simple yet powerful way to see which words appear most often in a collection of text. Words that occur more frequently are shown larger, giving you an at-a-glance view of the key themes and topics. While it doesn’t provide deep analysis, a word cloud is a quick and intuitive tool to spot trends & highlight important terms")
        st.divider()
        st.write('💡 The idea came from this:')
        st.write('Mueller, A. (2012). A Wordcloud in Python. Peekaboo. Available at: https://peekaboo-vision.blogspot.com/2012/11/a-wordcloud-in-python.html.')
        
    with tab2:
        st.text("1. Put your file. Choose your preferred column to analyze (if CSV).")
        st.text("2. Choose your preferred method to count the words and decide how many top words you want to include or remove.")
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
        st.subheader(':blue[WordCloud Download]', anchor=False)
        st.write("Right-click image and click \"Save-as\"")

st.header("Wordcloud", anchor=False)
st.subheader('Put your file here...', anchor=False)

MAX_PIXELS = 40_000_000     # batas ukuran kanvas agar server tidak kehabisan memori
TOKEN_RE = re.compile(r"\w[\w']+")


@st.cache_data(max_entries=3, ttl=1800, show_spinner=False)
def remove_stopwords(text: str) -> str:
    """Buang stopword bahasa Inggris. Daftar stopword dibuat SEKALI (bukan per kata)."""
    textprep.ensure_nltk("stopwords")
    blocked = set(stopwords.words('english'))
    return ' '.join(w for w in TOKEN_RE.findall(text) if w.lower() not in blocked)


def read_text_source(uploaded_file, c1):
    """Kembalikan teks yang akan dibuat wordcloud: kolom tabel pilihan pengguna, atau teks polos .txt."""
    if uploaded_file.name.lower().endswith('.txt'):
        raw = uploaded_file.getvalue().decode('utf-8-sig', errors='replace')
        first_line = raw.split('\n', 1)[0]
        if '\t' not in first_line and not sf.looks_like_medline(raw):
            return raw                                         # teks polos
    table = ui.load(uploaded_file)
    if table is None:
        st.stop()
    colcho = c1.selectbox("Choose Column", list(table))
    return " ".join(table[colcho].dropna().astype(str))


#===Read data===
uploaded_file = ui.uploader(types=['txt', 'csv', 'xls', 'xlsx'])

if uploaded_file is not None:
    tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "⬇️ Download Help"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            max_font = st.number_input("Max Font Size", min_value=1, value=100)
            image_height = st.number_input("Image height", min_value=50, max_value=4000, value=400)
            background = st.selectbox("Background color", ["white", "black"])
        with c2:
            max_words = st.number_input("Max Word Count", min_value=1, max_value=2000, value=250)
            image_width = st.number_input("Image width", min_value=50, max_value=4000, value=500)
            scale = st.number_input("Scale", min_value=1, max_value=4, value=2)
            words_to_remove = st.text_input("Remove specific words. Separate words by semicolons (;)")
        filterwords = set(textprep.parse_word_list(words_to_remove))

        try:
            text = read_text_source(uploaded_file, c1)
            text = remove_stopwords(text)

            if st.button("Submit"):
                if image_width * image_height * scale * scale > MAX_PIXELS:
                    st.warning('The image is too large. Please reduce the width, height or scale.', icon="⚠️")
                elif not text.strip():
                    st.warning('There is no text left to visualize.', icon="⚠️")
                else:
                    cloud = WordCloud(max_font_size=max_font, max_words=max_words,
                                      background_color=background, stopwords=filterwords,
                                      height=int(image_height), width=int(image_width),
                                      scale=int(scale)).generate(text)
                    st.session_state['wordcloud_image'] = cloud.to_image()

            img = st.session_state.get('wordcloud_image')
            if img is not None:
                with st.container(border=True):
                    st.image(img, width="stretch")
                import io
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                st.download_button("📥 Download image", buffer.getvalue(), "wordcloud.png", "image/png")
        except Exception as exc:
            ui.fail("Please ensure that your file is correct. Please contact us if you find that this is an error.", exc)

    with tab2:
        st.markdown('**Mueller, A. (n.d.). word_cloud: A little word cloud generator in Python.** https://github.com/amueller/word_cloud')

    with tab3:
        st.text("Click the Download image button below the picture, or right-click the picture and choose 'Save image as...'.")
