"""Resume matcher – Gradio app.

Paste a resume or upload a PDF, then get:
  1. the predicted job category,
  2. the 5 closest job offers (hybrid of TF-IDF and sentence-embedding cosine similarity),
  3. the skills of a selected offer missing from the resume,
  4. the skills most asked for a data role in 785k postings, present or missing in the resume.

    uv run python app/app.py              # first launch: ~1 minute to train the classifier, then cached in data/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import pandas as pd
from pypdf import PdfReader

from matcher.llm import get_llm, llm_skills
from matcher.pipeline import ResumeMatcher
from matcher.skills import display, extract_skills

matcher = ResumeMatcher()
HAS_LLM = get_llm() is not None


def read_pdf(file):
    if file is None:
        return ""
    try:
        if str(file).lower().endswith(".txt"):
            return Path(file).read_text(errors="ignore")
        return "\n".join(page.extract_text() or "" for page in PdfReader(file).pages)
    except Exception as e:                       # corrupted or protected file
        return f"__error__ {e}"


def analyse(text, file, use_llm):
    if file is not None:
        text = read_pdf(file)
        if text.startswith("__error__"):
            raise gr.Error("This file could not be read. Try another PDF or paste the text.")
    error = matcher.check(text or "")
    if error:
        raise gr.Error(error)

    result = matcher.analyse(text)
    skills = set(result["skills"])
    note = ""
    if use_llm and HAS_LLM:
        try:
            skills |= set().union(*[extract_skills(s) for s in llm_skills(text)])
            note = " (vocabulary + LLM)"
        except Exception as e:
            note = f" (LLM unavailable: {type(e).__name__}; vocabulary only)"

    skills_md = f"**Skills found in the resume{note}:** " + (", ".join(sorted(map(display, skills))) or "none")
    if result["weak_match"]:
        skills_md += ("\n\n⚠️ Weak match: no offer of the dataset is close to this resume "
                      "(the 853 offers do not cover every field). The offers below are only loosely related.")
    offers = result["offers"]
    choices = [(f"{r.title} – {r.company} ({r.similarity:.2f})", int(r.job_id)) for r in offers.itertuples()]
    state = {"text": text, "skills": sorted(skills)}
    gap_md = offer_gap(choices[0][1], state)
    return (result["categories"], skills_md, offers.drop(columns="job_id"),
            gr.update(choices=choices, value=choices[0][1]), gap_md, state)


def offer_gap(job_id, state):
    if not state or job_id is None:
        return ""
    job, matched, missing = matcher.gap(set(state["skills"]), int(job_id))
    return (f"### {job['title']} – {job['company']}\n"
            f"✅ **Skills of the offer in your resume:** {', '.join(matched) or 'none'}\n\n"
            f"❌ **Missing skills:** {', '.join(missing) or 'none found by the vocabulary'}\n\n"
            f"<details><summary>Offer description</summary>\n\n{job['description'][:2500]}\n</details>")


def market_gap(role, state):
    if not state:
        return pd.DataFrame()
    df = matcher.market_gap(state["text"], role)
    df["share of postings"] = (df["share"] * 100).round(0).astype(int).astype(str) + " %"
    df["in your resume"] = df["in_resume"].map({True: "✅", False: "❌"})
    return df[["skill", "share of postings", "in your resume"]]


with gr.Blocks(title="Resume matcher") as demo:
    state = gr.State(None)
    gr.Markdown("# Resume matcher\nPaste a resume (English) or upload a PDF: job category, closest offers and missing skills.")
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(lines=14, label="Resume text")
        with gr.Column(scale=1):
            file = gr.File(label="…or upload a PDF / TXT", file_types=[".pdf", ".txt"], type="filepath")
            use_llm = gr.Checkbox(label="Also extract skills with an LLM" + ("" if HAS_LLM else " (no API key in .env)"),
                                  value=False, interactive=HAS_LLM)
            button = gr.Button("Analyse", variant="primary")

    with gr.Row():
        category = gr.Label(num_top_classes=5, label="Predicted category")
        skills = gr.Markdown()
    offers = gr.Dataframe(label="Closest job offers (ranked by TF-IDF + embedding similarity; similarity shown: "
                                "embedding cosine, 0-1)", interactive=False)
    with gr.Row():
        with gr.Column():
            offer = gr.Dropdown(label="Skill gap for the offer", choices=[])
            gap = gr.Markdown()
        with gr.Column():
            role = gr.Dropdown(label="Skills asked for a data role (785k postings, 2023)", choices=matcher.roles,
                               value="Data Analyst")
            market = gr.Dataframe(interactive=False)

    button.click(analyse, [text, file, use_llm], [category, skills, offers, offer, gap, state]).then(
        market_gap, [role, state], market)
    offer.change(offer_gap, [offer, state], gap)
    role.change(market_gap, [role, state], market)

if __name__ == "__main__":
    demo.launch()
