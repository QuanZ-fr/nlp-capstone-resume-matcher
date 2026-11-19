# Resume matcher

NLP capstone project. A user uploads or pastes a resume and gets:

1. **the job category** of the resume (for example Data Science, Java Developer, Human Resources);
2. **the closest job offers**, ranked by semantic similarity;
3. **the missing skills**: the skills asked by a selected offer that the resume does not mention.

**Team:** Zhongjie Wu, Zhou Quan

## Data

| Dataset | Content | Use |
|---------|---------|-----|
| [ahmedheakl/resume-atlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas) | 13,389 resumes, 43 categories (text already lowercased, punctuation removed) | resume classification |
| [jacob-hugging-face/job-descriptions](https://huggingface.co/datasets/jacob-hugging-face/job-descriptions) | 853 full job descriptions, all kinds of roles | resume ↔ offer matching |
| [lukebarousse/data_jobs](https://huggingface.co/datasets/lukebarousse/data_jobs) | 785,741 data-related job postings (2023), with a list of skills per posting but no description | skill vocabulary, skills in demand per role |

The datasets are downloaded from Hugging Face by the notebooks; no data file is stored in the repository.

## Pipeline

1. **Exploration** (`notebooks/01_explore_data.ipynb`): size, categories, text length, duplicates, quality issues.
2. **Classification** (`notebooks/02_classifier_baseline.ipynb`): TF-IDF + logistic regression baseline, then sentence embeddings + logistic regression; accuracy, macro F1, confusion between close categories, error analysis.
3. **Matching**: sentence embeddings (`all-MiniLM-L6-v2`) of resumes and offers, cosine similarity, top 5 offers.
4. **Skill gap**: skills extracted from the resume and the offer (skill vocabulary built from `data_jobs`, optionally an LLM returning JSON), difference between the two sets.
5. **Application** (`app/app.py`): Gradio interface.

## Results

To be completed.

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| TF-IDF + logistic regression (baseline) | 0.806 | 0.806 |
| Sentence embeddings + logistic regression | | |

## How to run

```bash
uv sync                                   # create the environment
cp .env.example .env                      # then paste an API key in .env (only for the LLM part)
uv run jupyter lab                        # notebooks
uv run python app/app.py                  # application
```

API keys are read from `.env`, which is listed in `.gitignore` and never committed.

## Who did what

| Member | Work |
|--------|------|
| Zhongjie Wu | |
| Zhou Quan | |
