"""
doc_corrector.py
Generates a corrected version of the uploaded DOCX file.
"""

import io
import docx
import re
from .ai_analyzer import COMMON_MISSPELLINGS

try:
    import language_tool_python
    tool = language_tool_python.LanguageTool('en-US')
except Exception:
    tool = None


def correct_docx(file_bytes: bytes) -> bytes:
    doc = docx.Document(io.BytesIO(file_bytes))

    for para in doc.paragraphs:
        for run in para.runs:
            if not run.text.strip():
                continue

            original_text = run.text
            new_text = original_text

            words_in_text = set(re.findall(r'\b[a-zA-Z]+\b', new_text))
            for w in words_in_text:
                lower_w = w.lower()
                if lower_w in COMMON_MISSPELLINGS:
                    correction = COMMON_MISSPELLINGS[lower_w]
                    if w.istitle():
                        correction = correction.title()
                    elif w.isupper():
                        correction = correction.upper()
                    new_text = re.sub(r'\b' + re.escape(w) + r'\b', correction, new_text)

            if tool and new_text.strip():
                try:
                    new_text = tool.correct(new_text)
                except Exception:
                    pass

            if new_text != original_text:
                run.text = new_text

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
