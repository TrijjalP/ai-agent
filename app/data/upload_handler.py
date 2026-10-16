"""
upload_handler.py
Extracts text from a user-uploaded PDF, DOCX, or TXT file and wraps it
as a Paper object, so it can flow through the exact same process_paper()
-> chunk -> embed pipeline as any fetched paper. Uploads are always
treated as full-text (we have the whole document, not just an abstract).

Also makes a best-effort guess at the paper's real title (instead of
just using the filename) and its publication year (instead of leaving
it as "unknown"). Academic papers don't carry structured metadata the
way an API response does, so this is pattern-matching over the document
itself, not guaranteed-accurate extraction — it won't be perfect on
every PDF, but is far better than showing the raw filename.
"""

import os
import re
import pdfplumber
from docx import Document as DocxDocument

from app.data.europepmc_fetcher import Paper

MIN_TITLE_LEN = 10
MAX_TITLE_LEN = 300

TITLE_REJECT_PATTERNS = [
    r"^\s*(abstract|introduction|keywords?|doi|issn|isbn)\b",
    r"^\s*\d+\s*$",
    r"^\s*(page|vol\.?|volume|issue)\b",
    r"^https?://",
    r"^\s*©",
]


def _looks_like_title(line: str) -> bool:
    stripped = line.strip()
    if not (MIN_TITLE_LEN <= len(stripped) <= MAX_TITLE_LEN):
        return False
    for pattern in TITLE_REJECT_PATTERNS:
        if re.match(pattern, stripped, re.IGNORECASE):
            return False
    letters = sum(c.isalpha() for c in stripped)
    if letters < len(stripped) * 0.5:
        return False
    return True


def _guess_year(text: str) -> str:
    """
    Looks for a plausible publication year (1990-2026) within the first
    ~3000 characters of the document, where a title/journal/date block
    usually lives — searching the whole document risks picking up a
    reference-list year instead of the paper's own.
    """
    window = text[:3000]
    matches = re.findall(r"\b(19[9]\d|20[0-2]\d)\b", window)
    if matches:
        return matches[0]
    return "unknown"


def _extract_pdf_title_by_font_size(page, fallback: str) -> str:
    """
    A paper's title is reliably the largest text near the top of page 1 —
    a much stronger signal than "first non-empty line" (which usually
    grabs a running header or journal name instead). Groups characters
    into visual lines by vertical position, finds the largest font size
    among lines in the upper half of the page, and joins consecutive
    lines at that size (titles often wrap across 1-3 lines).
    """
    try:
        chars = page.chars
        if not chars:
            return fallback

        lines_by_top = {}
        for ch in chars:
            top_key = round(ch["top"])
            lines_by_top.setdefault(top_key, []).append(ch)

        line_records = []
        for top_key, chs in lines_by_top.items():
            chs_sorted = sorted(chs, key=lambda c: c["x0"])
            text = "".join(c["text"] for c in chs_sorted).strip()
            if not text:
                continue
            size = max(c["size"] for c in chs)
            line_records.append({"top": top_key, "text": text, "size": size})

        if not line_records:
            return fallback

        page_height = page.height
        candidates = [l for l in line_records if l["top"] < page_height * 0.5] or line_records

        max_size = max(l["size"] for l in candidates)
        title_lines = [
            l for l in sorted(candidates, key=lambda l: l["top"])
            if l["size"] >= max_size - 0.5
        ]

        good_text = [l["text"] for l in title_lines if _looks_like_title(l["text"])]
        if good_text:
            return re.sub(r"\s+", " ", " ".join(good_text)).strip()
    except Exception as e:
        print(f"[Upload Handler] Font-size title extraction failed: {e}")

    return fallback


def extract_pdf(path: str, filename_fallback: str):
    with pdfplumber.open(path) as pdf:
        if not pdf.pages:
            return "", filename_fallback
        title = _extract_pdf_title_by_font_size(pdf.pages[0], filename_fallback)
        pages_text = [page.extract_text() or "" for page in pdf.pages]

    full_text = re.sub(r"\s+", " ", " ".join(pages_text)).strip()
    return full_text, title


def extract_docx(path: str, filename_fallback: str):
    doc = DocxDocument(path)

    # Prefer an actual Title/Heading-styled paragraph if the document has one
    title = None
    for p in doc.paragraphs[:20]:
        if not p.text.strip():
            continue
        style_name = (p.style.name or "").lower() if p.style else ""
        if "title" in style_name or "heading 1" in style_name:
            if _looks_like_title(p.text):
                title = p.text.strip()
                break

    if not title:
        for p in doc.paragraphs[:10]:
            if _looks_like_title(p.text):
                title = p.text.strip()
                break

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = re.sub(r"\s+", " ", " ".join(paragraphs)).strip()
    return full_text, (title or filename_fallback)


def extract_txt(path: str, filename_fallback: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    title = None
    for line in raw.split("\n")[:10]:
        if _looks_like_title(line):
            title = line.strip()
            break

    full_text = re.sub(r"\s+", " ", raw).strip()
    return full_text, (title or filename_fallback)


def build_paper_from_upload(path: str) -> Paper:
    """
    Builds a Paper object from a single uploaded file, with a best-effort
    real title (extracted from the document itself) and publication year
    (from a year pattern near the start of the document), falling back
    to the filename / "unknown" only if those heuristics find nothing.
    """
    filename_fallback = os.path.splitext(os.path.basename(path))[0].replace("_", " ").replace("-", " ").strip() or "Uploaded paper"

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        full_text, title = extract_pdf(path, filename_fallback)
    elif ext == ".docx":
        full_text, title = extract_docx(path, filename_fallback)
    elif ext == ".txt":
        full_text, title = extract_txt(path, filename_fallback)
    else:
        raise ValueError(f"Unsupported upload file type: {ext}")

    if not full_text:
        raise ValueError(f"Could not extract any text from {path}")

    published = _guess_year(full_text)

    return Paper(
        id=f"upload_{os.path.basename(path)}",
        title=title,
        summary=full_text[:500],  # short excerpt; process_paper() will prefer full_text anyway
        authors=[],
        published=published,
        pdf_url=None,
        source="user_upload",
        is_india_affiliated=False,
        is_open_access=True,
        full_text=full_text
    )


def build_papers_from_uploads(paths: list) -> list:
    papers = []
    for path in paths:
        try:
            papers.append(build_paper_from_upload(path))
        except Exception as e:
            print(f"[Upload Handler] Could not process {path}: {e}")
    return papers