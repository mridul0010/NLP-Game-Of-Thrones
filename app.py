import streamlit as st
import spacy
import networkx as nx
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

from utils.loader import load_text
from utils.network import build_character_graph
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

st.markdown(
"""
Analyze **A Song of Ice and Fire** using:

• Word Frequency Analysis  
• Character Relationship Networks  
• Topic Modeling (LDA)  
• GPT-2 Story Generation
"""
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

text_data, tokens = load_text()

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    tokenizer, model = load_generator()
    return nlp, tokenizer, model

nlp, tokenizer, model = load_models()

# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.header("⚙️ Settings")

sample_size = st.sidebar.slider(
    "Text Sample Size",
    20000,
    200000,
    100000
)

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
["📊 Statistics","👥 Characters","🧠 Topics","✍️ GPT-2 Generator"]
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

    if len(tokens) > 0:

        freq = Counter(tokens)

        df = pd.DataFrame(
            freq.most_common(20),
            columns=["Word","Count"]
        )

        st.subheader("Top 20 Words")

        st.bar_chart(df.set_index("Word"))

# --------------------------------------------------
# CHARACTER GRAPH
# --------------------------------------------------

with tab2:

    st.subheader("Character Relationship Graph")

    doc = nlp(text_data[:sample_size])

    G = build_character_graph(doc)

    st.write("Characters detected:", len(G.nodes()))

    if len(G.nodes()) > 20:

        top_nodes = sorted(
            G.degree,
            key=lambda x: x[1],
            reverse=True
        )[:20]

        names = [n[0] for n in top_nodes]

        G = G.subgraph(names)

    fig, ax = plt.subplots(figsize=(8,6))

    pos = nx.spring_layout(G)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="gold",
        font_size=9,
        ax=ax
    )

    st.pyplot(fig)

# --------------------------------------------------
# TOPIC MODELING
# --------------------------------------------------

with tab3:

    st.subheader("Topic Modeling (LDA)")

    if len(tokens) == 0:

        st.warning("No tokens available for topic modeling")

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

    # UI Controls for dynamic parameters
    col1, col2 = st.columns(2)
    with col1:
        # Match argument name 'temperature' from generator.py
        temp_val = st.slider("Temperature (Creativity)", 0.1, 1.5, 0.9, 0.1)
        max_len = st.slider("Story Length", 50, 500, 120)
    
    with col2:
        top_k_val = st.slider("Top K", 1, 100, 50)
        top_p_val = st.slider("Top P (Nucleus Sampling)", 0.0, 1.0, 0.95, 0.05)

    if st.button("Generate Story"):
        with st.spinner("The maesters are forging the text..."):
            # CORRECTION: Passing temperature, top_k, and top_p to match your generator.py
            result = generate_text(
                prompt,
                tokenizer,
                model,
                max_len,
                temperature=temp_val, 
                top_k=top_k_val,
                top_p=top_p_val
            )
        st.success(result)
