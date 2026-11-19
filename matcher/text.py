"""Text cleaning.

The resumes of `resume-atlas` are already preprocessed: lowercase, punctuation deleted
("node.js" -> "nodejs", "c++" -> "c", "info@x.com" -> "infoxcom"), no line breaks.
Any text given to the models (a resume pasted in the app, a job description) goes through
the same cleaning, otherwise the vocabulary seen at prediction time differs from training.
"""
import re

_SEPARATORS = re.compile(r"[\s,;:/|()\[\]{}<>\"'•·–—!?]+")   # characters that separate words
_OTHER = re.compile(r"[^a-z0-9 ]")                            # what is left: . + # - _ & ... deleted


def clean_text(text):
    """Lowercase, separators -> space, other punctuation deleted, spaces collapsed."""
    if not isinstance(text, str):
        return ""
    text = _SEPARATORS.sub(" ", text.lower())
    text = text.replace(". ", " ")                 # end of sentence (before deleting the dots inside words)
    text = _OTHER.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
