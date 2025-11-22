import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords

# Download stopwords only once
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean and normalize text for NLP analysis.

    Steps:
    - Lowercase
    - Remove non-alphanumeric characters
    - Remove stopwords
    """
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def get_top_keywords(corpus, top_n=20):
    """
    Extract most frequent words/phrases in the corpus.
    """
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(corpus)
    word_counts = X.sum(axis=0)

    keywords = [
        (word, word_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()
    ]

    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    return keywords[:top_n]


def run_lda(corpus, n_topics=6, n_words=15):
    """
    Run Latent Dirichlet Allocation (LDA) topic modeling.

    Returns:
        list of (topic_id, top_words[])
    """
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2),
        max_features=5000,
    )
    X = vectorizer.fit_transform(corpus)

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, learning_method="batch"
    )
    lda.fit(X)

    topics = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        top_word_indices = topic_weights.argsort()[-n_words:]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_word_indices]
        topics.append((topic_idx, top_words))

    return topics
