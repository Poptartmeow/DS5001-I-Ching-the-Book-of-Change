# DS5001—I Ching: Digital Text Analysis

A comprehensive computational analysis of James Legge's 1899 English translation of the *I Ching* (Book of Changes), applying NLP and unsupervised machine learning methods to recover thematic, semantic, and philosophical structure in this ancient Chinese divination text.

## Overview

This project treats the *I Ching* as a corpus suitable for large-scale text analytics, using the full Standard Text Analytic Data Model (F0 → F5) pipeline:

- **F0 → F1**: Web scraping from [sacred-texts.com](https://sacred-texts.com/ich/) to extract raw hexagram and appendix texts
- **F2**: Parsing into a three-table schema (LIB, CORPUS, VOCAB)
- **F3**: Linguistic annotation (tokenization, POS tagging, stemming)
- **F4**: Vectorization (TFIDF, L2 normalization)
- **F5**: Unsupervised models (PCA, LDA, Word2Vec)

## Key Findings

- The traditional **Upper Canon / Lower Canon division** does not cleanly separate in linguistic space; instead, the major divide is between hexagram texts and Ten Wings appendices.
- **PC0** isolates the appendix commentary register from the hexagrams' oracular language.
- **PC2** reveals a domestic-relational pole (marriage, family) opposite natural-elemental vocabulary (water, change).
- **LDA** identifies five latent topics, with T04 (formulaic/structural) dominating the appendix and T02 (relational) peaking in the Lower Canon.
- **Word2Vec** clusters moral evaluation terms, structural boilerplate, and cosmological vocabulary into distinct semantic regions.

## Repository Structure

```
.
├── README.md                          # This file
├── Final_Project_Report.ipynb         # Main analytical notebook with all models & visualizations
├── Project_Notebook_I_CHING.ipynb     # Detailed computation pipeline
├── scraping.py                        # Original scraper (web scraping from sacred-texts.com)
├── scraping_v2.py                     # Improved scraper with page-number filtering
├── output/                            # Generated tables & visualizations
│   ├── i_ching_data.csv               # Raw scraped paragraphs
│   ├── i_ching_paragraphs.csv         # Cleaned paragraph rows (doc_id, para_num, para_text)
│   ├── LIB.csv                        # Document metadata (75 docs: 64 hexagrams + appendices)
│   ├── CORPUS.csv                     # Token-level OHCO table (123,763 tokens)
│   ├── VOCAB.csv                      # Vocabulary with features (n, p, i, dfidf, pos, stop, porter_stem)
│   ├── BOW.csv                        # Bag-of-words (doc_id × term)
│   ├── DTM.csv                        # Document-term matrix
│   ├── TFIDF.csv & TFIDF_L2.csv       # Vectorized representations
│   ├── DCM.csv                        # PCA document-component matrix (75 × 10)
│   ├── LOADINGS.csv                   # PCA component-term matrix (500 × 10)
│   ├── THETA.csv                      # LDA document-topic matrix (75 × 5)
│   ├── PHI.csv                        # LDA topic-term matrix (5 × vocabulary)
│   ├── BOW_SENT.csv & DOC_SENT.csv    # Sentiment annotations
│   ├── VOCAB_W2V.csv                  # Word2Vec embeddings (100-dim)
│   ├── pca_viz_1.png                  # PC0 vs PC1 scatter (documents + loadings)
│   ├── pca_viz2.png                   # PC2 vs PC3 scatter
│   ├── lda_pca_viz.png                # LDA topics in PCA space
│   ├── Sentiment Score by Hexagram.png  # Bar chart of mean doc sentiment
│   ├── w2v_tsne.png                   # Word2Vec embeddings in t-SNE space
│   ├── Hierarchical Clustering of Hexagrams.png  # Ward dendrogram
│   ├── Topic Distribution Heatmap by Section.png # LDA topics by canon section
│   └── KDE Plot of Sentiment Distribution by Book.png
└── data/                              # Reference data (external if any)
```

## Installation & Setup

### Prerequisites
- Python 3.13+
- Conda or pip

### Environment

A pre-configured environment file is provided:

```bash
conda env create -f environment.yml
conda activate ds5001-iching
pip install -r requirements-pip.txt
```

Or manually:

```bash
pip install pandas numpy nltk scikit-learn gensim matplotlib seaborn requests beautifulsoup4
```

### Data

Raw data is scraped on-the-fly from [sacred-texts.com](https://sacred-texts.com/ich/). To regenerate:

```bash
python scraping_v2.py
# Output: output/i_ching_paragraphs.csv
```

## Usage

### 1. Run Full Pipeline

Open and run `Final_Project_Report.ipynb` or `Project_Notebook_I_CHING.ipynb`:

```bash
jupyter notebook Final_Project_Report.ipynb
```

### 2. Load Pre-Computed Results

```python
import pandas as pd

DCM = pd.read_csv('output/DCM.csv', sep='|')          # PCA scores
THETA = pd.read_csv('output/THETA.csv', sep='|')      # LDA topics
VOCAB_W2V = pd.read_csv('output/VOCAB_W2V.csv', sep='|')  # Word embeddings
```

### 3. Regenerate a Specific Model

Example: refit PCA with different components:

```python
from sklearn.decomposition import PCA
TFIDF_L2 = pd.read_csv('output/TFIDF_L2.csv', sep='|')
pca = PCA(n_components=15)
DCM = pd.DataFrame(pca.fit_transform(TFIDF_L2), index=TFIDF_L2.index)
```

## Authors

- **Shawn Ding** (exf7sx@virginia.edu)
- **Tianyin Mao** (qhh3bv@virginia.edu)

University of Virginia, DS 5001 Text as Data (Spring 2026)

## Source Material

- **Text**: *I Ching* (Book of Changes), translated by James Legge (1899)
- **URL**: https://sacred-texts.com/ich/
- **License**: Public Domain

## References

- Legge, J. (1899). *The I Ching: Book of Changes*. Sacred Texts Archive.
- Scikit-learn: Machine Learning in Python (Pedregosa et al., 2011)
- Gensim: Topic Modelling for Humans (Řehůřek & Sojka, 2010)
- NLTK: Natural Language Toolkit (Bird, Klein, & Loper, 2009)

## License

This analysis is provided as-is for educational purposes under the University of Virginia DS 5001 course.
