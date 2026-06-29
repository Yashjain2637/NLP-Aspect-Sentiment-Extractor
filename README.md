<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&pause=1000&color=6C63FF&center=true&vCenter=true&width=700&lines=NLP+Aspect+Sentiment+Extractor;Fine-Grained+Opinion+Mining;Zero+ML+Training+Required" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.9.4-green?style=for-the-badge)](https://www.nltk.org/)
[![VADER](https://img.shields.io/badge/VADER-3.3.2-orange?style=for-the-badge)](https://github.com/cjhutto/vaderSentiment)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)
[![uv](https://img.shields.io/badge/Managed%20by-uv-blue?style=for-the-badge)](https://github.com/astral-sh/uv)

<br/>

> **Rule-based Aspect-Based Sentiment Analysis (ABSA)** pipeline.  
> Extracts `(feature, opinion, sentiment)` triples from raw text reviews — no ML training, no GPU, no cloud.

</div>

---

## ⚡ What It Does

Given raw product reviews in `reviews.txt`, the pipeline outputs structured JSON:

```json
{
  "id": 1,
  "review": "The battery drains fast but the screen is stunning.",
  "aspects": [
    { "feature": "battery", "opinion": "drains fast", "sentiment": "negative" },
    { "feature": "screen",  "opinion": "stunning",    "sentiment": "positive" }
  ],
  "summary": {
    "total_aspects": 2,
    "positive": 1,
    "negative": 1,
    "neutral": 0
  }
}
```

---

## 🏗️ Architecture

```
reviews.txt
    │
    ▼
┌─────────────────────────────────────────┐
│            analyze.py                   │
│                                         │
│  split_clauses()  ──► clause chunks     │
│       │                                 │
│       ▼                                 │
│  NLTK POS Tagging                       │
│       │                                 │
│       ▼                                 │
│  collect_np()     ──► Noun Phrase       │
│       │                                 │
│       ▼                                 │
│  get_opinions()   ──► Opinion Words     │
│       │                                 │
│       ▼                                 │
│  get_sentiment()  ──► VADER + Negation  │
│       │                                 │
│       ▼                                 │
│  {feature, opinion, sentiment}          │
└─────────────────────────────────────────┘
    │
    ▼
output/output.json
```

---

## 🔬 Pipeline Deep Dive

### 1 · Clause Splitting (`split_clauses`)

Splits adversative conjunctions (`but`, `however`, `though`, `although`, `yet`, `whereas`, `while`), then commas, then `and` (only when both sides are self-contained NP+opinion).

### 2 · POS Tagging

Uses `nltk.pos_tag` with the `averaged_perceptron_tagger_eng` model.

| Tag Set | Purpose |
|---|---|
| `NN NNS NNP NNPS` | Feature (aspect) candidates |
| `JJ JJR JJS` | Adjective opinions |
| `RB RBR RBS` | Adverb opinions |
| `VB VBD VBG VBN VBP VBZ` | Verb opinions |

### 3 · Noun-Phrase Extraction (`collect_np`)

Walks **back** up to 4 tokens over adj/noun tokens (stops at copula/conjunction), then walks **forward** over all contiguous nouns. Strips trailing verb-nouns (e.g. `drains`, `overheat`) from NP boundaries.

### 4 · Opinion Extraction (`get_opinions`)

- Copula pattern (`X is/was ADJ`) → grabs adj/adv after copula
- Default → grabs up to 4 OPN-tagged tokens immediately after NP
- Deduplicates opinions; strips tokens already inside NP

### 5 · Sentiment Scoring (`get_sentiment`)

- VADER `SentimentIntensityAnalyzer` with **custom lexicon extensions** (50+ domain words added with calibrated scores: e.g. `'laggy': -2.5`, `'stunning': 3.0`)
- Opinion text double-weighted: `opinion_text + " " + opinion_text + " " + clause_text`
- Negation flip: any `NEGATIONS` token (`not`, `no`, `never`, `neither`, `barely`, `hardly`, `without`) in clause → score negated
- Threshold: `compound >= 0.05` → `positive`; `<= -0.05` → `negative`; else `neutral`

---

## 📁 Repo Structure

```
NLP-Aspect-Sentiment-Extractor/
├── analyze.py          # Core ABSA engine (279 lines)
├── main.py             # Entry point
├── reviews.txt         # Input: one review per line
├── requirements.txt    # nltk==3.9.4, vaderSentiment==3.3.2
├── pyproject.toml      # uv project config
├── uv.lock             # Lockfile
├── .python-version     # Pinned Python version
└── output/
    └── output.json     # Generated results
```

---

## 🚀 Quick Start

### pip

```bash
git clone https://github.com/Yashjain2637/NLP-Aspect-Sentiment-Extractor.git
cd NLP-Aspect-Sentiment-Extractor
pip install -r requirements.txt
python analyze.py
```

### uv (recommended — uses lockfile)

```bash
git clone https://github.com/Yashjain2637/NLP-Aspect-Sentiment-Extractor.git
cd NLP-Aspect-Sentiment-Extractor
uv sync
uv run analyze.py
```

> **Note:** NLTK corpora (`punkt`, `punkt_tab`, `averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng`, `vader_lexicon`) auto-download on first run.

---

## 📥 Input Format

`reviews.txt` — plain text, **one review per line**, UTF-8:

```
The camera quality is outstanding but the battery life is poor.
Delivery was delayed but the product quality is premium and the packaging is attractive.
Customer service was unhelpful and unresponsive to my queries.
```

---

## 📤 Output Schema

`output/output.json` — JSON array:

```jsonc
[
  {
    "id": 1,                          // 1-indexed review number
    "review": "<original text>",
    "aspects": [
      {
        "feature": "<noun phrase>",   // e.g. "battery life"
        "opinion": "<opinion words>", // e.g. "poor"
        "sentiment": "positive|negative|neutral"
      }
    ],
    "summary": {
      "total_aspects": 2,
      "positive": 1,
      "negative": 1,
      "neutral": 0
    }
  }
]
```

---

## 🧠 VADER Lexicon Extensions

50+ domain-specific words added with calibrated compound scores:

| Category | Words (sample) | Score Range |
|---|---|---|
| Strong negative | `horrible`, `terrible`, `poor`, `unhelpful` | `-3.5` → `-3.0` |
| Moderate negative | `laggy`, `buggy`, `unstable`, `overheats`, `slow` | `-2.5` → `-2.0` |
| Mild negative | `limited`, `average`, `heavy`, `quickly` | `-1.5` → `-0.5` |
| Mild positive | `clear`, `loud`, `simple`, `useful` | `+1.5` → `+2.0` |
| Strong positive | `stunning`, `outstanding`, `vivid`, `vibrant` | `+2.5` → `+3.0` |

---

## ⚙️ Dependencies

| Package | Version | Role |
|---|---|---|
| `nltk` | `3.9.4` | Tokenization, POS tagging |
| `vaderSentiment` | `3.3.2` | Sentiment scoring |

Zero deep-learning dependencies. Runs on CPU, works offline after first NLTK data download.

---

## 🗺️ Roadmap

- [ ] CLI flags: custom `--input` / `--output` paths
- [ ] Batch mode: directory of `.txt` files
- [ ] Aspect aggregation across reviews (per-feature sentiment histogram)
- [ ] `spaCy` dependency parser for higher-precision opinion linking
- [ ] REST API wrapper (`FastAPI`)
- [ ] Docker image

---

## 🤝 Contributing

```bash
# Fork → clone → branch
git checkout -b feat/your-feature

# Make changes, then
git commit -m "feat: describe change"
git push origin feat/your-feature
# Open PR against main
```

---

## 📄 License

MIT © [Yashjain2637](https://github.com/Yashjain2637)

---

<div align="center">

**Built with 🧠 NLTK · VADER · Pure Python**

⭐ Star the repo if it helps you!

</div>
