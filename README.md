<div align="center">

# ⚔️ NLP — Game of Thrones

**An interactive NLP dashboard for analyzing *A Song of Ice and Fire* and generating new stories with GPT-2**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](./LICENSE)

</div>

---

## 📖 Table of Contents

- [Why This Project?](#-why-this-project)
- [Features](#-features)
- [Understanding the Generation Controls](#-understanding-the-generation-controls)
  - [Temperature](#temperature)
  - [Top-K Sampling](#top-k-sampling)
  - [Top-P (Nucleus) Sampling](#top-p-nucleus-sampling)
  - [How They Work Together](#how-they-work-together)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 💡 Why This Project?

Natural Language Processing can feel abstract until you see it in action on a dataset you care about. *A Song of Ice and Fire* — the book series behind *Game of Thrones* — offers a massive, rich corpus that is perfect for exploring NLP techniques:

- **Linguistic analysis** — Discover word patterns, vocabulary richness, and thematic elements across five books.
- **Character networks** — Map relationships between hundreds of characters using Named Entity Recognition (NER) and graph theory.
- **Topic discovery** — Uncover hidden themes with Latent Dirichlet Allocation (LDA).
- **Creative text generation** — Fine-tune GPT-2 on George R. R. Martin's writing style and generate new passages that sound like they belong in Westeros.

This project bundles all of these capabilities into a single, interactive Streamlit dashboard that anyone can explore — no NLP background required.

---

## ✨ Features

| Tab | What It Does |
|-----|--------------|
| **📊 Statistics** | Character count, word count, vocabulary size, top-20 word frequency bar chart, and a word cloud |
| **👥 Characters** | Interactive force-directed character relationship network graph and a top-10 character importance ranking (degree centrality) |
| **🧠 Topics** | Adjustable LDA topic modeling (2–10 topics) to surface recurring themes across the books |
| **✍️ GPT-2 Generator** | GPT-2-powered story generator with real-time controls for Temperature, Top-K, Top-P, and story length |

---

## 🎛️ Understanding the Generation Controls

When the GPT-2 model generates text, it predicts a probability distribution over its entire vocabulary for each next token. The three parameters below control **how** a token is sampled from that distribution, giving you fine-grained control over the creativity and coherence of the output.

### Temperature

| | |
|---|---|
| **Range** | 0.5 – 1.5 |
| **Default** | 1.0 |

Temperature **scales** the raw logits (scores) before they are converted into probabilities.

- **Low temperature (< 1.0)** — Sharpens the distribution, making the model more **confident** and **deterministic**. The most probable tokens become even more likely, producing safer, more predictable text.
- **Temperature = 1.0** — No scaling; the original probability distribution is used as-is.
- **High temperature (> 1.0)** — Flattens the distribution, giving less-probable tokens a better chance of being picked. This makes the output more **creative** and **surprising**, but can also introduce incoherence.

> **Analogy:** Think of temperature like a creativity dial. Turn it down for a careful narrator; turn it up for a wildcard storyteller.

### Top-K Sampling

| | |
|---|---|
| **Range** | 10 – 100 |
| **Default** | 80 |

Top-K sampling restricts the model to only consider the **K most probable** tokens at each step. All other tokens are discarded before sampling.

- **Low K (e.g., 10)** — Only a handful of tokens are candidates, leading to highly focused and repetitive text.
- **High K (e.g., 100)** — A larger pool of tokens is available, allowing more diversity.

> **Example:** If the vocabulary has 50,000 tokens and K = 80, the model ignores 49,920 tokens and samples only from the top 80.

### Top-P (Nucleus) Sampling

| | |
|---|---|
| **Range** | 0.5 – 1.0 |
| **Default** | 0.95 |

Top-P sampling (also called *nucleus sampling*) keeps the **smallest set of tokens whose cumulative probability** meets or exceeds the threshold **P**. Unlike Top-K, the number of candidate tokens varies dynamically at each step.

- **Low P (e.g., 0.5)** — The nucleus is small; only tokens that together cover 50% of the probability mass are kept, producing conservative output.
- **High P (e.g., 0.95)** — The nucleus is large; nearly all likely tokens are included, allowing more variety.
- **P = 1.0** — No filtering; all tokens are considered (equivalent to disabling nucleus sampling).

> **Why is it useful?** Top-P naturally adapts. When the model is very confident about one token, the nucleus is small; when probability is spread evenly, the nucleus grows — giving you the best of both worlds.

### How They Work Together

In this project, **all three controls are active simultaneously**. The generation pipeline applies them in the following order:

1. **Temperature** scales the logits.
2. **Top-K** trims the distribution to the K highest-scoring tokens.
3. **Top-P** further trims the remaining tokens to the smallest set whose cumulative probability ≥ P.
4. A token is **randomly sampled** from this final set.

Additionally, a **repetition penalty (1.2)** discourages the model from repeating phrases, and a **no-repeat-ngram** constraint prevents any 3-gram from appearing twice — keeping stories fresh and readable.

| Setting | Effect |
|---------|--------|
| Low temperature + Low K + Low P | Very safe, predictable, and repetitive |
| High temperature + High K + High P | Highly creative but potentially incoherent |
| **Default (1.0 / 80 / 0.95)** | **Balanced creativity and coherence** |

---

## 📁 Project Structure

```
NLP-Game-Of-Thrones/
├── app.py                  # Streamlit dashboard (main entry point)
├── utils/
│   ├── generator.py        # GPT-2 model loading & text generation
│   ├── loader.py           # Text & token loading from pre-processed data
│   ├── topic_model.py      # LDA topic modeling
│   └── network.py          # Character relationship graph (spaCy NER + NetworkX)
├── data/
│   ├── 001ssb.txt          # Book 1 — A Game of Thrones
│   ├── 002ssb.txt          # Book 2 — A Clash of Kings
│   ├── 003ssb.txt          # Book 3 — A Storm of Swords
│   ├── 004ssb.txt          # Book 4 — A Feast for Crows
│   └── 005ssb.txt          # Book 5 — A Dance with Dragons
├── models/                 # Pre-trained models (joblib / GPT-2 weights)
├── got.ipynb               # Jupyter notebook for experimentation
├── requirements.txt        # Python dependencies
└── LICENSE                 # GPL-3.0
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/mridul0010/NLP-Game-Of-Thrones.git
cd NLP-Game-Of-Thrones

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The GPT-2 model weights are downloaded automatically from Google Drive on first run.

### Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 🖥️ Usage

1. **Statistics tab** — Browse text metrics and visualize the most frequent words.
2. **Characters tab** — Explore the interactive character network; hover over nodes to see names and drag to rearrange.
3. **Topics tab** — Adjust the number of topics in the sidebar and observe how themes shift.
4. **GPT-2 Generator tab** — Enter a story prompt, tweak Temperature / Top-K / Top-P, and click **Generate Story** to produce new text in the style of *A Song of Ice and Fire*.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| [Streamlit](https://streamlit.io/) | Interactive web dashboard |
| [Transformers](https://huggingface.co/docs/transformers) | GPT-2 model & tokenizer |
| [PyTorch](https://pytorch.org/) | Deep learning backend |
| [spaCy](https://spacy.io/) | Named Entity Recognition |
| [Gensim](https://radimrehurek.com/gensim/) | LDA topic modeling |
| [NetworkX](https://networkx.org/) | Graph construction & analysis |
| [PyVis](https://pyvis.readthedocs.io/) | Interactive graph visualization |
| [WordCloud](https://amueller.github.io/word_cloud/) | Word cloud generation |
| [Matplotlib](https://matplotlib.org/) | Plotting |
| [Pandas](https://pandas.pydata.org/) | Data manipulation |

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](./LICENSE) file for details.