import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)