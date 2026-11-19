"""Skill extraction with a vocabulary (no model, fully explainable).

Vocabulary (218 skills + 56 alternative spellings) = 132 technical skills kept from the 252 listed in the 785k
postings of `data_jobs` (rare tools and spelling variants dropped or merged) + 86 general and business skills
written by hand (the job descriptions also cover sales, HR, finance...).

Text and skills are cleaned the same way (`clean_text`), so "Node.js" in a resume and "node.js" in the
vocabulary both become "nodejs". Matching is done on whole words: "excel" does not match "excellent".

Some skill names are also common English words ("r", "go", "c", "word", "spring", "shell"...).
They only count when another technical skill appears within a few words, as in "python r sql".
"""
from collections import defaultdict

from .text import clean_text

# Not counted as skills: "training", "reporting" and "compensation" are mostly section titles
# ("training and certifications") or benefits of the offer ("competitive compensation").

# Technical skills of data_jobs (the long tail of rare tools and spelling variants is merged or dropped below)
TECH_SKILLS = """sql python aws azure r tableau excel spark power-bi java sas hadoop scala snowflake databricks gcp git kafka
airflow oracle nosql docker sql-server go kubernetes mongodb redshift tensorflow pyspark pandas linux word sap powerpoint
pytorch mysql javascript bigquery jira c++ postgresql numpy terraform scikit-learn looker github jenkins ssis shell qlik
c# c windows vba cassandra keras alteryx matlab unix confluence spss elasticsearch bash gitlab jupyter ssrs dax react
outlook matplotlib ruby sharepoint t-sql ansible html express spring powershell gdpr dynamodb css redis sheets cognos
angular microstrategy visio typescript splunk bitbucket flask perl db2 php plotly django seaborn neo4j aurora node.js
vue ms-access vmware graphql nltk julia kotlin selenium fastapi firebase visual-basic opencv rust swift ggplot2 jquery
asp.net mariadb hugging-face tidyverse datarobot rshiny dplyr smartsheet trello asana flutter sqlite cobol fortran
heroku looker-studio""".split()
TECH_SKILLS = [s.replace("-", " ") if s not in ("scikit-learn", "t-sql") else s for s in TECH_SKILLS]

# Skills that are not tools: methods, domains, business and soft skills
GENERAL_SKILLS = """machine learning|deep learning|natural language processing|computer vision|statistics|data analysis|
data visualization|data mining|data modeling|data warehousing|etl|big data|a/b testing|forecasting|predictive modeling|
regression|time series|business intelligence|dashboards|kpi|agile|scrum|devops|ci/cd|microservices|rest api|
cloud computing|cybersecurity|networking|project management|product management|stakeholder management|risk management|
change management|budgeting|financial analysis|financial modeling|accounting|auditing|bookkeeping|payroll|tax|
quickbooks|crm|salesforce|erp|supply chain|logistics|inventory management|procurement|recruitment|onboarding|
employee relations|compensation and benefits|customer service|sales|negotiation|account management|business development|
lead generation|marketing|digital marketing|seo|social media|content creation|copywriting|public relations|
communication|leadership|teamwork|problem solving|time management|presentation|autocad|solidworks|revit|photoshop|
illustrator|figma|ux design|ui design|six sigma|lean|quality assurance|testing|troubleshooting""".replace("\n", "").split("|")

# Other spellings -> canonical skill
ALIASES = {
    "powerbi": "power bi", "power bi desktop": "power bi", "mongo": "mongodb", "golang": "go", "no sql": "nosql",
    "sqlserver": "sql server", "ms sql": "sql server", "mssql": "sql server", "msaccess": "ms access",
    "microsoft access": "ms access", "reactjs": "react", "react js": "react", "vuejs": "vue", "angularjs": "angular",
    "huggingface": "hugging face", "rstudio": "r", "r programming": "r", "r studio": "r", "sklearn": "scikit-learn",
    "postgres": "postgresql", "k8s": "kubernetes", "amazon web services": "aws", "google cloud": "gcp",
    "google cloud platform": "gcp", "microsoft azure": "azure", "microsoft excel": "excel", "ms excel": "excel",
    "advanced excel": "excel", "microsoft word": "word", "ms word": "word", "microsoft powerpoint": "powerpoint",
    "ms powerpoint": "powerpoint", "google sheets": "sheets", "c plus plus": "c++", "nodejs": "node.js",
    "nlp": "natural language processing", "ml": "machine learning", "bi": "business intelligence",
    "data visualisation": "data visualization", "ab testing": "a/b testing", "ci cd": "ci/cd", "restful api": "rest api",
    "rest apis": "rest api", "customer support": "customer service", "recruiting": "recruitment",
    "public speaking": "presentation", "power query": "power bi", "google analytics": "digital marketing",
    "pivot tables": "excel", "dashboard": "dashboards", "data analytics": "data analysis",
    "data visualizations": "data visualization", "visualization": "data visualization", "statistical modeling": "statistics", "vlookup": "excel", "statistical analysis": "statistics",
}

# Names that are also ordinary words: kept only next to another technical skill
AMBIGUOUS = {"r", "go", "c", "c++", "c#", "word", "express", "spring", "shell", "windows", "sheets", "outlook", "react",
             "swift", "julia", "rust", "flutter", "ruby", "ml", "bi", "aurora", "flask", "spark", "lean"}

# Display names (everything else is shown in lowercase)
DISPLAY = {"sql": "SQL", "python": "Python", "aws": "AWS", "azure": "Azure", "r": "R", "tableau": "Tableau",
           "excel": "Excel", "spark": "Spark", "power bi": "Power BI", "java": "Java", "sas": "SAS", "gcp": "GCP",
           "c": "C/C++/C#", "c++": "C/C++/C#", "c#": "C/C++/C#", "javascript": "JavaScript", "typescript": "TypeScript",
           "html": "HTML", "css": "CSS", "vba": "VBA", "dax": "DAX", "etl": "ETL", "crm": "CRM", "erp": "ERP",
           "seo": "SEO", "kpi": "KPI", "sap": "SAP", "spss": "SPSS", "gdpr": "GDPR", "ssis": "SSIS", "ssrs": "SSRS",
           "nosql": "NoSQL", "mysql": "MySQL", "postgresql": "PostgreSQL", "bigquery": "BigQuery", "pyspark": "PySpark",
           "scikit-learn": "scikit-learn", "pytorch": "PyTorch", "tensorflow": "TensorFlow", "github": "GitHub",
           "gitlab": "GitLab", "mongodb": "MongoDB", "node.js": "Node.js", "t-sql": "T-SQL", "a/b testing": "A/B testing",
           "ci/cd": "CI/CD", "rest api": "REST API", "ux design": "UX design", "ui design": "UI design"}


def display(skill):
    return DISPLAY.get(skill, skill)


def _build_patterns():
    """Cleaned token sequence -> canonical skill. Several canonical spellings collapse after cleaning (c, c++, c#)."""
    patterns = {}
    for skill in TECH_SKILLS + GENERAL_SKILLS:
        patterns[tuple(clean_text(skill).split())] = skill
    for alias, skill in ALIASES.items():
        patterns[tuple(clean_text(alias).split())] = skill
    by_first = defaultdict(list)
    for tokens, skill in patterns.items():
        if tokens:
            by_first[tokens[0]].append((tokens, skill))
    for first in by_first:
        by_first[first].sort(key=lambda p: -len(p[0]))     # longest match first: "sql server" before "sql"
    return by_first


_PATTERNS = _build_patterns()
_TECH = set(TECH_SKILLS)


def extract_skills(text, window=4):
    """Set of canonical skills found in a text (raw or already cleaned)."""
    tokens = clean_text(text).split()
    hits = []                                   # (position, skill, matched alias)
    i = 0
    while i < len(tokens):
        for pattern, skill in _PATTERNS.get(tokens[i], []):
            if tuple(tokens[i:i + len(pattern)]) == pattern:
                hits.append((i, skill, " ".join(pattern)))
                i += len(pattern) - 1
                break
        i += 1
    anchors = [pos for pos, skill, alias in hits if skill in _TECH and alias not in AMBIGUOUS and skill not in AMBIGUOUS]
    found = set()
    for pos, skill, alias in hits:
        if alias in AMBIGUOUS or skill in AMBIGUOUS:
            if not any(0 < abs(pos - a) <= window for a in anchors):
                continue                        # "go", "word", "spring"... used as ordinary words
        found.add("c" if skill in ("c++", "c#") else skill)
    return found


def skill_gap(resume_text, job_text):
    """Skills of the job found / missing in the resume. Returns (matched, missing) as sorted display names."""
    resume, job = extract_skills(resume_text), extract_skills(job_text)
    return sorted(map(display, job & resume)), sorted(map(display, job - resume))


def role_gap(resume_text, role_skills, role, min_share=0.10, top=15):
    """Skills most asked for a data role (share of postings), marked as present or missing in the resume."""
    resume = extract_skills(resume_text)
    rows = role_skills[(role_skills["role"] == role) & (role_skills["share"] >= min_share)]
    rows = rows.sort_values("share", ascending=False).head(top).copy()
    canon = rows["skill"].map(lambda s: "c" if s in ("c++", "c#") else ALIASES.get(s, s))
    rows["in_resume"] = canon.isin(resume)
    rows["skill"] = canon.map(display)
    return rows[["skill", "share", "in_resume"]].drop_duplicates("skill").reset_index(drop=True)
