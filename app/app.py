"""Resume matcher – Gradio app.

Paste a resume or upload a PDF, then get:
  1. the predicted job category,
  2. the 5 closest job offers (hybrid of TF-IDF and sentence-embedding cosine similarity),
  3. the skills of a selected offer missing from the resume,
  4. the skills most asked for a data role in 785k postings, present or missing in the resume.

Visual idea: the resume is a sheet of paper reviewed by a recruiter. Skills you have are marked with a
highlighter, skills you lack with a red pen; the same two marks are used everywhere in the page.

    uv run python app/app.py              # first launch: ~1 minute to train the classifier, then cached in data/
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from pypdf import PdfReader

from matcher.llm import get_llm, llm_skills
from matcher.pipeline import ResumeMatcher
from matcher.skills import display, extract_skills

matcher = ResumeMatcher()
HAS_LLM = get_llm() is not None

# ----------------------------------------------------------------------------- look

LIGHT = dict(
    body_background_fill="#E6EBEF",
    body_text_color="#1E2A3B",
    body_text_color_subdued="#5F6B7A",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F3F5F7",
    block_background_fill="#FFFFFF",
    block_border_color="#FFFFFF",
    block_label_text_color="#5F6B7A",
    block_label_background_fill="#FFFFFF",
    block_title_text_color="#1E2A3B",
    border_color_primary="#DDE3E9",
    input_background_fill="#FFFFFF",
    input_border_color="#CDD5DD",
    input_border_color_focus="#2B59C3",
    input_placeholder_color="#8A95A3",
    button_primary_background_fill="#1E2A3B",
    button_primary_background_fill_hover="#2B59C3",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#F3F5F7",
    button_secondary_text_color="#1E2A3B",
    checkbox_background_color="#FFFFFF",
    checkbox_background_color_selected="#2B59C3",
    checkbox_border_color="#9AA5B1",
    checkbox_border_color_selected="#2B59C3",
    checkbox_label_background_fill="#FFFFFF",
    checkbox_label_text_color="#1E2A3B",
    color_accent="#2B59C3",
    color_accent_soft="#E3EAF8",
)
# The page is a sheet of paper: the same light colours are used when the browser is in dark mode.
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Instrument Sans"), "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "monospace"],
).set(
    **LIGHT, **{f"{k}_dark": v for k, v in LIGHT.items() if k != "color_accent"},
    block_border_width="0px",
    block_shadow="0 1px 2px rgba(30, 42, 59, 0.08)",
    block_shadow_dark="0 1px 2px rgba(30, 42, 59, 0.08)",
    block_radius="6px",
    button_large_radius="6px",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,500;7..72,650&display=swap');
:root { --ink:#1E2A3B; --pencil:#5F6B7A; --hi:#E8F06A; --red:#B93A32; --blue:#2B59C3; --rule:#DDE3E9; }

.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
.masthead { padding: 28px 4px 8px; }
.masthead h1 { font-family: 'Literata', Georgia, serif; font-weight: 650; font-size: 2.6rem;
               line-height: 1.1; letter-spacing: -0.01em; margin: 0 0 8px; color: var(--ink); }
.masthead p { font-size: 1.05rem; color: var(--pencil); max-width: 62ch; margin: 0; }

/* the resume is a sheet of paper */
#sheet textarea { font-size: 0.98rem; line-height: 1.6; padding: 28px 32px !important;
                  background-image: linear-gradient(#FFFFFF 0 0); min-height: 340px; }
#sheet { box-shadow: 0 1px 2px rgba(30,42,59,.08), 0 10px 28px -12px rgba(30,42,59,.25); }

.card { background: #FFFFFF !important; border-radius: 6px !important; padding: 22px 26px !important;
        box-shadow: 0 1px 2px rgba(30,42,59,.08) !important; }
.card .prose, .card > div { background: transparent !important; border: 0 !important; padding: 0 !important; }
.panel { padding: 0; }
.panel h2 { font-family: 'Literata', Georgia, serif; font-weight: 500; font-size: 1.3rem;
            margin: 0 0 4px; color: var(--ink); }
.panel h2.sub { font-size: 1.05rem; margin: 18px 0 8px; }
.panel .hint { color: var(--pencil); font-size: 0.9rem; margin: 0 0 14px; }
.panel .empty { color: var(--pencil); font-size: 0.95rem; padding: 10px 0 6px; }

/* the two marks */
.hl { background: linear-gradient(104deg, transparent 0.4em, var(--hi) 0.5em, var(--hi) calc(100% - 0.4em), transparent 100%);
      padding: 0.08em 0.45em; margin: 0 0.15em 0.4em 0; display: inline-block; color: var(--ink);
      border-radius: 0.2em 0.5em 0.25em 0.6em; }
.pen { display: inline-block; margin: 0 0.6em 0.4em 0; color: var(--red);
       text-decoration: underline wavy var(--red); text-decoration-thickness: 1.5px; text-underline-offset: 4px; }
.marks { line-height: 2; }

/* category bars */
.bars { display: grid; grid-template-columns: minmax(9rem, max-content) 1fr 3rem; gap: 6px 12px; align-items: center; }
.bars .name { font-size: 0.95rem; }
.bars .name.top { font-weight: 600; }
.bars .track { height: 8px; background: #EEF1F4; border-radius: 4px; overflow: hidden; }
.bars .fill { height: 100%; background: var(--blue); border-radius: 4px; }
.bars .fill.dim { background: #9FB3DE; }
.bars .pct { font-variant-numeric: tabular-nums; color: var(--pencil); font-size: 0.9rem; text-align: right; }

/* ranked offers */
ol.offers { list-style: none; counter-reset: rank; margin: 0; padding: 0; }
ol.offers li { counter-increment: rank; display: grid; grid-template-columns: 2rem 1fr auto; gap: 2px 10px;
               padding: 10px 0; border-top: 1px solid var(--rule); }
ol.offers li:first-child { border-top: 0; }
ol.offers li::before { content: counter(rank); font-family: 'Literata', Georgia, serif; font-size: 1.25rem;
                       color: var(--pencil); grid-row: span 2; }
ol.offers .title { font-weight: 600; }
ol.offers .company { color: var(--pencil); font-size: 0.9rem; grid-column: 2; }
ol.offers .score { grid-row: span 2; align-self: center; font-variant-numeric: tabular-nums; font-size: 0.9rem;
                   color: var(--blue); }

.warning { border-left: 3px solid var(--red); padding: 8px 12px; margin: 12px 0 0; background: #FBF1F0;
           color: var(--ink); font-size: 0.92rem; }

/* market table */
table.market { width: 100%; border-collapse: collapse; font-size: 0.95rem; border: 0 !important; margin: 0; }
table.market tr, table.market td { border-left: 0 !important; border-right: 0 !important; border-bottom: 0 !important; background: transparent !important; }
table.market td { padding: 6px 0 !important; border-top: 1px solid var(--rule) !important; vertical-align: middle; }
table.market tr:first-child td { border-top: 0 !important; }
table.market td.bar { width: 45%; padding: 0 12px; }
table.market td.share { width: 3.5rem; text-align: right; font-variant-numeric: tabular-nums; color: var(--pencil); }

details.desc { margin-top: 14px; color: var(--pencil); font-size: 0.92rem; }
details.desc summary { cursor: pointer; color: var(--blue); }
details.desc div { white-space: pre-wrap; max-height: 320px; overflow: auto; margin-top: 8px; color: var(--ink); }

#analyse { font-size: 1.05rem; min-height: 52px; }
:focus-visible { outline: 2px solid var(--blue) !important; outline-offset: 2px; }
footer { display: none !important; }
@media (max-width: 720px) { .masthead h1 { font-size: 2rem; } #sheet textarea { padding: 18px !important; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""


def panel(title, hint, body):
    hint_html = f'<p class="hint">{hint}</p>' if hint else ""
    return f'<div class="panel"><h2>{title}</h2>{hint_html}{body}</div>'


def marks(skills, kind):
    return "".join(f'<span class="{kind}">{escape(s)}</span>' for s in skills)


EMPTY = {
    "category": panel("Likely job family", "Based on 12,085 labelled resumes.",
                      '<p class="empty">Paste a resume and click Analyse.</p>'),
    "offers": panel("Closest offers", "Out of 842 job descriptions.", '<p class="empty">Your 5 closest offers will appear here.</p>'),
    "gap": panel("What this offer asks for", "Highlighted: in your resume. Underlined in red: missing.",
                 '<p class="empty">Pick an offer after the analysis.</p>'),
    "market": panel("What the market asks for", "Share of 2023 job postings asking for each skill.",
                    '<p class="empty">Pick a data role after the analysis.</p>'),
}

# ----------------------------------------------------------------------------- logic


def read_file(file):
    if file is None:
        return ""
    try:
        if str(file).lower().endswith(".txt"):
            return Path(file).read_text(errors="ignore")
        return "\n".join(page.extract_text() or "" for page in PdfReader(file).pages)
    except Exception as e:                       # corrupted or protected file
        return f"__error__ {e}"


def category_html(categories, skills, note, weak):
    rows = []
    best = max(categories.values())
    for i, (name, p) in enumerate(categories.items()):
        rows.append(f'<span class="name{" top" if i == 0 else ""}">{escape(name)}</span>'
                    f'<span class="track"><span class="fill{"" if i == 0 else " dim"}" style="width:{p / best * 100:.0f}%; display:block"></span></span>'
                    f'<span class="pct">{p:.0%}</span>')
    found = marks(sorted(map(display, skills)), "hl") or '<span class="empty">No known skill found.</span>'
    body = (f'<div class="bars">{"".join(rows)}</div>'
            f'<h2 style="margin-top:22px">Skills in your resume</h2><p class="hint">Found by the skill vocabulary{note}.</p>'
            f'<div class="marks">{found}</div>')
    if weak:
        body += ('<p class="warning">No offer in the dataset is close to this resume: the 842 offers do not cover '
                 'every field, so the offers shown are only loosely related.</p>')
    return panel("Likely job family", "Based on 12,085 labelled resumes.", body)


def offers_html(offers):
    items = "".join(f'<li><span class="title">{escape(r.title)}</span><span class="score">{r.similarity:.2f}</span>'
                    f'<span class="company">{escape(r.company)}</span></li>' for r in offers.itertuples())
    return panel("Closest offers", "Out of 842 job descriptions. Score: similarity from 0 to 1.",
                 f'<ol class="offers">{items}</ol>')


def analyse(text, file, use_llm):
    if file is not None:
        text = read_file(file)
        if text.startswith("__error__"):
            raise gr.Error("This file can't be read. Upload another PDF or paste the text.")
    error = matcher.check(text or "")
    if error:
        raise gr.Error(error)

    result = matcher.analyse(text)
    skills = set(result["skills"])
    note = ""
    if use_llm and HAS_LLM:
        try:
            skills |= set().union(*[extract_skills(s) for s in llm_skills(text)])
            note = " and the LLM"
        except Exception as e:
            note = f" (the LLM is unavailable: {type(e).__name__})"

    offers = result["offers"]
    choices = [(f"{i}. {r.title} – {r.company}", int(r.job_id)) for i, r in enumerate(offers.itertuples(), 1)]
    state = {"text": text, "skills": sorted(skills)}
    return (category_html(result["categories"], skills, note, result["weak_match"]), offers_html(offers),
            gr.update(choices=choices, value=choices[0][1]), offer_gap(choices[0][1], state), state)


def offer_gap(job_id, state):
    if not state or job_id is None:
        return EMPTY["gap"]
    job, matched, missing = matcher.gap(set(state["skills"]), int(job_id))
    if not matched and not missing:
        body = '<p class="empty">The skill vocabulary finds no skill in this offer. Read the description below.</p>'
    else:
        have = marks(matched, "hl") or '<span class="empty">None of them yet.</span>'
        lack = marks(missing, "pen") or '<span class="empty">Nothing: you cover every skill found in the offer.</span>'
        body = (f'<p class="hint">{len(matched)} of the {len(matched) + len(missing)} skills of the offer '
                f'are in your resume.</p><div class="marks">{have}</div>'
                f'<h2 class="sub">Missing from your resume</h2><div class="marks">{lack}</div>')
    body += (f'<details class="desc"><summary>Read the offer</summary>'
             f'<div>{escape(job["description"][:4000])}</div></details>')
    return panel(f"{escape(job['title'])} – {escape(job['company'])}", "", body)


def market_gap(role, state):
    if not state:
        return EMPTY["market"]
    df = matcher.market_gap(state["text"], role)
    rows = "".join(
        f'<tr><td><span class="{"hl" if r.in_resume else "pen"}">{escape(r.skill)}</span></td>'
        f'<td class="bar"><span class="track" style="display:block;height:8px;background:#EEF1F4;border-radius:4px">'
        f'<span style="display:block;height:100%;width:{r.share / df["share"].max() * 100:.0f}%;'
        f'background:#2B59C3;border-radius:4px"></span></span></td>'
        f'<td class="share">{r.share:.0%}</td></tr>' for r in df.itertuples())
    have = int(df["in_resume"].sum())
    return panel(f"What {escape(role)} postings ask for",
                 f"Share of 2023 postings asking for each skill. You have {have} of these {len(df)}.",
                 f'<table class="market">{rows}</table>')


# ----------------------------------------------------------------------------- page

with gr.Blocks(title="Resume matcher") as demo:
    state = gr.State(None)
    gr.HTML('<header class="masthead"><h1>Resume matcher</h1>'
            '<p>Paste your resume in English or upload it as a PDF. You get the job family it fits, '
            'the closest offers, and the skills you would need for them.</p></header>')
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            text = gr.Textbox(lines=16, label="Your resume", show_label=False, elem_id="sheet",
                              placeholder="Paste the text of your resume here: title, experience, education, skills…")
        with gr.Column(scale=1, min_width=260):
            file = gr.File(label="Or upload a PDF or TXT file", file_types=[".pdf", ".txt"], type="filepath")
            use_llm = gr.Checkbox(label="Also find skills with an LLM (slower, sends the text to Gemini)" if HAS_LLM
                                  else "LLM skill search: add an API key to .env to use it",
                                  value=False, interactive=HAS_LLM)
            button = gr.Button("Analyse", variant="primary", elem_id="analyse")

    with gr.Row(equal_height=False):
        category = gr.HTML(EMPTY["category"], elem_classes="card")
        offers = gr.HTML(EMPTY["offers"], elem_classes="card")
    with gr.Row(equal_height=False):
        with gr.Column():
            offer = gr.Dropdown(label="Compare with an offer", choices=[], interactive=True)
            gap = gr.HTML(EMPTY["gap"], elem_classes="card")
        with gr.Column():
            role = gr.Dropdown(label="Compare with a data role", choices=matcher.roles, value="Data Analyst")
            market = gr.HTML(EMPTY["market"], elem_classes="card")

    button.click(analyse, [text, file, use_llm], [category, offers, offer, gap, state]).then(
        market_gap, [role, state], market)
    offer.change(offer_gap, [offer, state], gap)
    role.change(market_gap, [role, state], market)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
