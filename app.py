import streamlit as st
import joblib
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pyvis.network import Network
import streamlit.components.v1 as components
import spacy

from utils.loader import load_text
from utils.network import load_character_graph
from utils.generator import load_generator, generate_text


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Game of Thrones NLP Dashboard",
    page_icon="⚔️",
    layout="wide"
)

st.title("⚔️ Game of Thrones NLP Dashboard")

st.markdown("""
Analyze **A Song of Ice and Fire** using:

• Word Frequency Analysis  
• Word Cloud Visualization  
• Character Relationship Networks  
• Topic Modeling  
• GPT-2 Story Generation
""")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

text_data, tokens = load_text()

freq = joblib.load("models/word_freq.joblib")


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    nlp = spacy.load("en_core_web_sm")

    tokenizer, model = load_generator()

    G = load_character_graph()

    lda = joblib.load("models/lda_model.joblib")

    dictionary = joblib.load("models/lda_dictionary.joblib")

    return nlp, tokenizer, model, G, lda, dictionary


nlp, tokenizer, model, G, lda, dictionary = load_models()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Settings")

num_topics = st.sidebar.slider(
    "Number of Topics",
    2,
    10,
    5
)


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Statistics", "👥 Characters", "🧠 Topics", "✍️ GPT-2 Generator"]
)


# --------------------------------------------------
# STATISTICS TAB
# --------------------------------------------------

with tab1:

    st.subheader("Text Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Characters", len(text_data))
    col2.metric("Words", len(tokens))
    col3.metric("Vocabulary Size", len(set(tokens)))

    df = pd.DataFrame(
        freq.most_common(20),
        columns=["Word", "Count"]
    )

    st.subheader("Top 20 Words")

    st.bar_chart(df.set_index("Word"))

    st.subheader("Word Cloud")

    wc = WordCloud(
        width=800,
        height=400,
        background_color="black"
    ).generate(" ".join(tokens))

    fig, ax = plt.subplots()

    ax.imshow(wc)
    ax.axis("off")

    st.pyplot(fig)


# --------------------------------------------------
# CHARACTER NETWORK
# --------------------------------------------------

with tab2:

    st.subheader("Character Relationship Network")

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#111111",
        font_color="white"
    )

    for node in G.nodes():
        net.add_node(node, label=node)

    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], value=edge[2]["weight"])

    net.repulsion(node_distance=200)

    net.save_graph("graph.html")

    with open("graph.html", "r", encoding="utf-8") as f:
        components.html(f.read(), height=600)

    st.subheader("Most Important Characters")

    centrality = nx.degree_centrality(G)

    df = pd.DataFrame(
        sorted(
            centrality.items(),
            key=lambda x: x[1],
            reverse=True
        ),
        columns=["Character", "Importance"]
    )

    st.dataframe(df.head(10))


# --------------------------------------------------
# TOPIC MODELING
# --------------------------------------------------

with tab3:

    st.subheader("Topic Modeling")

    topics = lda.show_topics(
        num_topics=num_topics,
        num_words=10,
        formatted=False
    )

    for topic_id, words in topics:

        word_list = [word for word, prob in words]

        st.markdown(
            f"### Topic {topic_id + 1}"
        )

        st.write(", ".join(word_list))


# --------------------------------------------------
# GPT-2 GENERATOR
# --------------------------------------------------

with tab4:

    st.subheader("GPT-2 Story Generator")

    prompt = st.text_area(
        "Start the story",
        "The night was cold and the wall stood silent..."
    )

    temperature = st.slider(
        "Temperature",
        0.5,
        1.5,
        1.0
    )

    max_len = st.slider(
        "Story Length",
        50,
        500,
        150
    )

    top_k = st.slider(
        "Top K",
        10,
        100,
        80
    )

    top_p = st.slider(
        "Top P",
        0.5,
        1.0,
        0.95
    )

    if st.button("Generate Story"):

        with st.spinner("Generating story..."):

            result = generate_text(
                prompt,
                tokenizer,
                model,
                max_len,
                temperature,
                top_k,
                top_p
            )

        st.success("Story Generated")

        st.write(result)
