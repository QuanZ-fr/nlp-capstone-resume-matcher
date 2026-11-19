"""Shared code of the resume matcher: used by the notebooks and by the Gradio app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # cached files (embeddings, models); ignored by git
DATA_DIR.mkdir(exist_ok=True)
