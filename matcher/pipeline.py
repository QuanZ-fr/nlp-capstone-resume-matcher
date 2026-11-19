"""End-to-end pipeline used by the app: resume text -> category, top offers, skill gap.

Choices come from the notebooks:
- classifier: TF-IDF + logistic regression (notebook 03: macro F1 0.806 vs 0.726 for embeddings);
- matching: hybrid of TF-IDF cosine and embedding cosine (notebook 04: best precision@5 under both labellings).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import DATA_DIR
from .data import load_resumes, load_jobs, load_role_skills
from .models import tfidf_classifier, embed_documents
from .skills import extract_skills, display, role_gap
from .text import clean_text

MIN_WORDS = 30          # below this, the text is not a resume
# Embedding cosine of the best offer under which the match is weak. Test resumes: median 0.69, min 0.31;
# short resumes typed by hand score lower (0.53-0.55 for a data analyst), an airline pilot 0.44 (no such offer).
WEAK_MATCH = 0.45


class ResumeMatcher:
    def __init__(self):
        self.jobs = load_jobs()
        self.job_emb = embed_documents(self.jobs["text"], name="jobs_chunks")
        self.job_skills = self.jobs["text"].map(extract_skills)
        self.role_skills = load_role_skills()
        self.roles = sorted(self.role_skills["role"].unique())
        self.classifier, self.vectorizer = self._load_models()
        self.job_tfidf = self.vectorizer.transform(self.jobs["text"])

    def _load_models(self):
        """Trained once on all resumes (the notebooks keep 20% for the test), then cached in data/."""
        path = DATA_DIR / "models.joblib"
        if path.exists():
            return joblib.load(path)
        resumes = load_resumes()
        print(f"Training on {len(resumes)} resumes (first launch only, about a minute)...")
        classifier = tfidf_classifier().fit(resumes["text"], resumes["category"])
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True,
                                     stop_words="english").fit(pd.concat([resumes["text"], self.jobs["text"]]))
        joblib.dump((classifier, vectorizer), path)
        return classifier, vectorizer

    @staticmethod
    def check(text):
        """Error message for an input that is not a usable resume, or None."""
        n = len(clean_text(text).split())
        if n == 0:
            return "No text found. Paste a resume or upload a PDF that contains text (a scanned image cannot be read)."
        if n < MIN_WORDS:
            return f"Only {n} words: paste a full resume (at least {MIN_WORDS} words)."
        return None

    def analyse(self, text, k=5):
        cleaned = clean_text(text)
        probs = self.classifier.predict_proba([cleaned])[0]
        top_cat = np.argsort(probs)[::-1][:5]
        categories = {self.classifier.classes_[i]: float(probs[i]) for i in top_cat}

        s_emb = self.job_emb @ embed_documents([cleaned])[0]
        s_tfidf = (self.job_tfidf @ self.vectorizer.transform([cleaned]).T).toarray().ravel()
        hybrid = s_emb / s_emb.max() + s_tfidf / max(s_tfidf.max(), 1e-9)
        top = np.argsort(-hybrid)[:k]
        offers = self.jobs.iloc[top][["title", "company"]].copy()
        offers.insert(0, "similarity", s_emb[top].astype(float).round(2))   # shown to the user: embedding cosine (0-1)
        offers = offers.reset_index().rename(columns={"index": "job_id"})

        return {"categories": categories, "offers": offers, "skills": extract_skills(cleaned),
                "weak_match": bool(s_emb.max() < WEAK_MATCH)}

    def gap(self, resume_skills, job_id):
        job = self.jobs.loc[job_id]
        needed = self.job_skills.loc[job_id]
        matched = sorted(map(display, needed & resume_skills))
        missing = sorted(map(display, needed - resume_skills))
        return job, matched, missing

    def market_gap(self, text, role):
        return role_gap(text, self.role_skills, role)
