# IMDB Sentiment Analysis

Sentiment classification of IMDB movie reviews using **Multinomial Naïve Bayes** and **Linear SVM** with TF-IDF features.

---

## Problem Statement

Given a free-form English movie review from IMDB, predict whether the expressed sentiment is **positive** or **negative**.

This is a **supervised binary classification** problem in the **Text Mining** domain.

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | IMDB Dataset of 50K Movie Reviews |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| **Instances** | 50,000 |
| **Attributes** | `review` (text), `sentiment` (positive / negative) |
| **Class Distribution** | Balanced — 25,000 positive, 25,000 negative |

---

## Preprocessing Pipeline

1. Duplicate removal (418 duplicates → 49,582 reviews)
2. HTML tag stripping via regex
3. Special-character and digit removal
4. Lowercasing
5. Tokenisation
6. Stop-word removal (NLTK English)
7. Lemmatisation (WordNet)
8. **TF-IDF vectorisation** — 10,000 features, unigrams + bigrams, sublinear TF

---

## Models & Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Multinomial Naïve Bayes | 0.8695 | 0.8609 | 0.8827 | 0.8716 | 0.9415 |
| **Linear SVM** | **0.8946** | **0.8893** | **0.9024** | **0.8958** | **0.9619** |

- **Linear SVM** outperforms Naïve Bayes across all metrics
- Both models substantially beat the 50% majority-class baseline
- Train/Test split: 80/20, stratified

---

## Repository Structure

```
.
├── IMDB_Sentiment_Analysis.ipynb   # Main notebook (code + results)
├── IMDB Dataset.csv                # Dataset (50K reviews)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/abhishukla0204/imdb-sentiment-text-mining.git
cd imdb-sentiment-text-mining
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data (first time only)
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Run the notebook
```bash
jupyter notebook IMDB_Sentiment_Analysis.ipynb
```

Run all cells sequentially. The notebook will:
- Load and explore the dataset
- Preprocess text and extract TF-IDF features
- Train both classifiers
- Evaluate and generate all visualisations

---

## Visualisations

The notebook generates 4 visualisation sets:
- **EDA plots** — class distribution, review length histogram, word count box-plots
- **Confusion matrices** — heatmaps for both classifiers
- **ROC-AUC curves** — comparison of both models
- **Word clouds** — most frequent terms per sentiment class

---

## Tech Stack

- **Python 3.8+**
- pandas, numpy — data handling
- scikit-learn — ML models (MultinomialNB, LinearSVC, TfidfVectorizer)
- NLTK — text preprocessing (stopwords, WordNet lemmatiser)
- matplotlib, seaborn — visualisation
- wordcloud — word cloud generation

---

## References

1. L. Maas, R. Daly, P. Pham, D. Huang, A. Ng, and C. Potts, "Learning Word Vectors for Sentiment Analysis," in *Proc. 49th Annual Meeting of the ACL*, 2011, pp. 142–150.
2. F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.

---


