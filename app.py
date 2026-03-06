import streamlit as st
import joblib
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import spacy

from utils.loader import load_text
from utils.network import load_character_graph
from utils.topic_model import run_lda
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
• Character Importance Ranking  
• Topic Modeling (LDA)  
• GPT-2 Story Generation
""")

# --------------------------------------------------
# PATH SETUP (important for deployment)
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

text_data, tokens = load_text()

freq = joblib.load(os.path.join(MODELS_DIR, "word_freq.joblib"))

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    nlp = spacy.load("en_core_web_sm")

    tokenizer, model = load_generator()

    G = load_character_graph()

    return nlp, tokenizer, model, G


nlp, tokenizer, model, G = load_models()

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

    # Word Cloud
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
# CHARACTER TAB
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

    # Character Importance
    st.subheader("Most Important Characters")

    centrality = nx.degree_centrality(G)

    df = pd.DataFrame(
        sorted(centrality.items(), key=lambda x: x[1], reverse=True),
        columns=["Character", "Importance"]
    )

    st.dataframe(df.head(10))

# --------------------------------------------------
# TOPIC MODELING
# --------------------------------------------------

with tab3:

    st.subheader("Topic Modeling (LDA)")

    if len(tokens) == 0:
        st.warning("No tokens available")
    else:
        topics = run_lda(tokens, num_topics)

        for t in topics:
            st.info(t)

# --------------------------------------------------
# GPT-2 GENERATOR
# --------------------------------------------------

with tab4:

    st.subheader("GPT-2 Story Generator")

    prompt = st.text_area(
        "Start the story",
        "The night was cold and the wall stood silent..."
    )

    st.markdown("### Generation Controls")

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.slider(
            "Temperature",
            0.5,
            1.5,
            1.0,
            0.05
        )

        max_len = st.slider(
            "Story Length",
            50,
            500,
            150
        )

    with col2:

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
            0.95,
            0.05
        )

    if st.button("Generate Story"):

        if prompt.strip() == "":
            st.warning("Please enter a prompt")

        else:

            with st.spinner("Generating story..."):

                result = generate_text(
                    prompt,
                    tokenizer,
                    model,
                    max_len,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

            st.success("Story Generated")

            st.write(result)
