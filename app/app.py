"""Resume matcher – first version of the interface.

Paste a resume, get its predicted category (TF-IDF + logistic regression baseline).
Matching with job offers and the skill gap will be added next.

    uv run python app/app.py
"""
import re

import gradio as gr
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def clean(text):
    """Same format as the training resumes: lowercase, letters and digits only."""
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def train_classifier():
    resumes = load_dataset("ahmedheakl/resume-atlas")["train"].to_pandas().drop_duplicates("Text")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10),
    )
    print(f"Training on {len(resumes)} resumes...")
    model.fit(resumes["Text"], resumes["Category"])
    return model


def build_interface(model):
    def classify(resume):
        if not resume or len(resume.split()) < 20:
            return {"Please paste a full resume (at least 20 words)": 1.0}
        probs = model.predict_proba([clean(resume)])[0]
        top = probs.argsort()[::-1][:5]
        return {model.classes_[i]: float(probs[i]) for i in top}

    with gr.Blocks(title="Resume matcher") as demo:
        gr.Markdown("# Resume matcher\nPaste a resume to get its job category. Job matching and missing skills: coming next.")
        resume = gr.Textbox(lines=15, label="Resume")
        button = gr.Button("Analyse", variant="primary")
        category = gr.Label(num_top_classes=5, label="Predicted category")
        button.click(classify, inputs=resume, outputs=category)
    return demo


if __name__ == "__main__":
    build_interface(train_classifier()).launch()
