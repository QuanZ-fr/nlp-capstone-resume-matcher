# Resume matcher

NLP capstone project. A job seeker pastes a resume or uploads a PDF and gets:

1. **the job category** of the resume (43 categories, for example Data Science, Java Developer, Human Resources);
2. **the 5 closest job offers**, with a similarity score;
3. **the missing skills**: the skills asked by a selected offer that the resume does not mention;
4. **the skills in demand** for a data role (share of 785k postings asking for each skill), present or missing in the resume.

**Users:** students and junior candidates who want to know which offers fit their resume and what to learn next.

**Team:** Quan Zhou, Zhongjie Wu, Habib Moutaly

## Data

| Dataset | Content | Use |
|---------|---------|-----|
| [ahmedheakl/resume-atlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas) | 13,389 resumes, 43 categories (12,085 after removing duplicates) | resume classification |
| [jacob-hugging-face/job-descriptions](https://huggingface.co/datasets/jacob-hugging-face/job-descriptions) | 853 job descriptions from all fields (842 kept) | resume ↔ offer matching, skill gap |
| [lukebarousse/data_jobs](https://huggingface.co/datasets/lukebarousse/data_jobs) | 785,741 data job postings (2023), list of skills per posting, no description | skill vocabulary, skills in demand per role |

The notebooks download the data from Hugging Face; no data file is stored in the repository.

**Data quality issues found** (notebook 01): 1,304 duplicate resumes, 169 of them with conflicting categories; resumes already lowercased without punctuation; template resumes with placeholder text; job offers that cover only 39 of the 43 resume categories.

## Pipeline

| Step | Notebook | Method |
|------|----------|--------|
| Exploration | `01_explore_data` | size, balance, length, duplicates, quality |
| Classification baseline | `02_classifier_baseline` | TF-IDF (words + bigrams) + logistic regression |
| Classification comparison | `03_classifier_embeddings` | sentence embeddings (`all-MiniLM-L6-v2`) + logistic regression, two ways of handling long resumes |
| Matching | `04_matching` | TF-IDF cosine vs embedding cosine vs hybrid, precision@5 with a bias check |
| Skill gap | `05_skill_gap` | vocabulary of 218 skills (+56 alternative spellings) with rules for ambiguous words, compared with LLM extraction (Gemini) |
| Application | `app/app.py` | Gradio interface using the best choice of each step |

The shared code is in `matcher/` (used by the notebooks and the app): `text.py` (cleaning), `data.py` (loading), `models.py` (classifiers, embeddings of long texts), `skills.py` (skill extraction), `llm.py` (optional LLM), `pipeline.py` (what the app calls).

## Results

**Classification** (2,417 test resumes, 43 classes, random = 0.023)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| **TF-IDF + logistic regression** | **0.806** | **0.806** |
| Sentence embeddings (first 256 tokens) + LR | 0.732 | 0.726 |
| Sentence embeddings (mean of chunks) + LR | 0.707 | 0.708 |

TF-IDF keeps the job-specific keywords that decide the category; embeddings blur close categories (Consultant, Management, PMO). Errors are mostly between categories that overlap in reality (Management / Operations Manager, SQL Developer / Database).

**Matching** (precision@5: share of the 5 recommended offers in the resume's category; 1,841 resumes)

| Method | Jobs labelled by TF-IDF classifier | Jobs labelled by embedding classifier |
|--------|-----|-----|
| TF-IDF cosine | 0.362 | 0.308 |
| Embedding cosine | 0.283 | 0.284 |
| **Hybrid** | **0.369** | **0.330** |
| Random | 0.033 | 0.032 |

The jobs have no category, so they are labelled by a classifier; labelling them with two different classifiers checks that the ranking of the methods does not come from that choice.

**Skill extraction** (15 offers): the vocabulary finds 5.5 skills per offer, the LLM 26.8. Only 22% of the LLM skills are in the vocabulary; for those, the vocabulary finds 77%. The vocabulary is kept by default (free, instant, explainable); the LLM is an option in the app.

## Limits

- The resumes are clean template resumes in English; real resumes (PDF layout, other languages) are harder.
- 853 offers are few and cover some fields poorly (data science, engineering): the app warns when the best match is weak.
- The skill vocabulary is centred on technical and business skills; domain skills (healthcare, law) are missed unless the LLM option is used.
- The category of a resume is a single label, while many profiles fit several categories (the app shows the 5 most likely).

## How to run

```bash
uv sync                                   # create the environment (Python 3.11)
cp .env.example .env                      # optional: paste an API key in .env for the LLM option
uv run python app/app.py                  # application on http://localhost:7860
uv run jupyter lab                        # notebooks (run them in order: 01 to 05)
```

The first launch of the app trains the classifier (about a minute) and caches it in `data/`. API keys are read from `.env` (python-dotenv), which is listed in `.gitignore` and never committed.

## Who did what

| Member | Work |
|--------|------|
| Zhongjie Wu | |
| Quan Zhou | |
| Habib Moutaly | |
