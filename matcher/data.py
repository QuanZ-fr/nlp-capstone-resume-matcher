"""Loading the three datasets (downloaded from Hugging Face, cached locally by `datasets`)."""
import ast

import pandas as pd
from datasets import load_dataset

from . import DATA_DIR
from .text import clean_text


def _plain(df):
    """Recent pandas stores text in Arrow arrays, which numpy 2-D indexing (labels[top_k]) does not support."""
    return df.astype({c: object for c in df.columns if df[c].dtype != object and df[c].dtype.kind not in "biuf"})


def load_resumes():
    """13,389 resumes, 43 categories. Exact duplicates removed (they would leak between train and test)."""
    df = load_dataset("ahmedheakl/resume-atlas")["train"].to_pandas()
    df = df.rename(columns={"Category": "category", "Text": "text"})
    df = df.drop_duplicates("text").reset_index(drop=True)
    return _plain(df)


def load_jobs():
    """853 job descriptions (all kinds of roles). Columns: title, company, description, text (cleaned)."""
    df = load_dataset("jacob-hugging-face/job-descriptions")["train"].to_pandas()
    df = df.rename(columns={"position_title": "title", "company_name": "company", "job_description": "description"})
    df = df[df["description"].str.split().str.len() >= 30]          # a few descriptions are almost empty
    df = df.drop_duplicates("description").reset_index(drop=True)
    df["text"] = (df["title"] + " " + df["description"]).map(clean_text)
    return _plain(df[["title", "company", "description", "text"]])


def load_role_skills(min_postings=1000):
    """Share of postings asking for each skill, per data role (from 785k postings of `data_jobs`).

    Returns a DataFrame with columns role, skill, postings and share (0-1).
    Computed once and cached in data/role_skills.parquet (the full dataset takes ~1 GB of memory).
    """
    cache = DATA_DIR / "role_skills.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = load_dataset("lukebarousse/data_jobs")["train"].to_pandas()[["job_title_short", "job_skills"]]
    df = df.dropna(subset=["job_skills"])
    df["job_skills"] = df["job_skills"].map(lambda s: ast.literal_eval(s) if isinstance(s, str) else list(s))
    n_per_role = df["job_title_short"].value_counts()
    long = df.explode("job_skills").dropna()
    counts = long.groupby(["job_title_short", "job_skills"]).size().rename("postings").reset_index()
    counts["share"] = counts["postings"] / counts["job_title_short"].map(n_per_role)
    counts = counts[counts["job_title_short"].map(n_per_role) >= min_postings]
    counts = counts.rename(columns={"job_title_short": "role", "job_skills": "skill"})
    counts.to_parquet(cache)
    return counts
