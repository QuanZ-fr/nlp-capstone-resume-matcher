"""Optional LLM call (skill extraction as JSON), through the chat-completions format shared by
Gemini, OpenAI and Anthropic. The key is read from the `.env` file with python-dotenv; without a key
the project still works (vocabulary-based extraction only).
"""
import json
import os
import re
import time

import requests
from dotenv import load_dotenv

from . import ROOT

PROVIDERS = {   # env variable -> (url, default model)
    "GEMINI_API_KEY": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "gemini-flash-lite-latest"),
    "OPENAI_API_KEY": ("https://api.openai.com/v1/chat/completions", "gpt-4o-mini"),
    "ANTHROPIC_API_KEY": ("https://api.anthropic.com/v1/chat/completions", "claude-haiku-5-5"),
}


def get_llm():
    """First provider with a key in .env or in the environment. Returns (url, model, key) or None."""
    load_dotenv(ROOT / ".env")
    for var, (url, model) in PROVIDERS.items():
        key = os.environ.get(var) or next((v for k, v in os.environ.items() if k.startswith(var) and v), "")
        if key:
            return url, os.environ.get("LLM_MODEL", model), key
    return None


def chat(prompt, max_tokens=1024, retries=4):
    llm = get_llm()
    if llm is None:
        raise RuntimeError("No API key: add GEMINI_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY) to .env")
    url, model, key = llm
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    for attempt in range(retries):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=90)
        except requests.RequestException:          # timeout or network error: wait and retry
            time.sleep(2 ** attempt + 1)
            continue
        if r.status_code in (429, 500, 503, 529):
            time.sleep(2 ** attempt + 1)
            continue
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content") or "").strip()
    raise RuntimeError("LLM rate limit: try again in a minute")


def llm_skills(text):
    """Skills of a resume or job description extracted by the LLM, as a list of short lowercase names."""
    prompt = f"""Extract the professional skills (tools, technologies, methods, domain and soft skills) mentioned in the text below.
Return only a JSON list of short lowercase skill names, for example ["python", "sql", "project management"].

Text:
{text[:6000]}

JSON:"""
    answer = chat(prompt)
    match = re.search(r"\[.*\]", answer, re.S)
    return [s.strip().lower() for s in json.loads(match.group(0))] if match else []
