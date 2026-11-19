"""Resume classifiers and the train/test split shared by the notebooks and the app."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from .embed import encode


def split(resumes, test_size=0.2, seed=42):
    """Same stratified 80/20 split everywhere, so that all models are compared on the same test resumes."""
    return train_test_split(resumes["text"], resumes["category"], test_size=test_size,
                            stratify=resumes["category"], random_state=seed)


def tfidf_classifier():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10),
    )


def chunks(text, size=150, max_chunks=8):
    """Split a long text into pieces of `size` words: the embedding model only reads the first 256 tokens."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), size)][:max_chunks]


def embed_documents(texts, name=None, chunked=True):
    """One vector per document: mean of its chunk embeddings (or the first 256 tokens only if chunked=False)."""
    texts = list(texts)
    if not chunked:
        return encode(texts, name=name)
    pieces, owner = [], []
    for i, t in enumerate(texts):
        for c in chunks(t):
            pieces.append(c)
            owner.append(i)
    emb = encode(pieces, name=name)
    owner = np.array(owner)
    doc = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    np.add.at(doc, owner, emb)
    doc /= np.linalg.norm(doc, axis=1, keepdims=True)          # back to unit length: cosine = dot product
    return doc


def embedding_classifier():
    return LogisticRegression(max_iter=3000, C=10)
