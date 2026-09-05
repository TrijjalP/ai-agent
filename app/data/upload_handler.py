"""
upload_handler.py
Handles user-uploaded papers (PDF or plain text), up to a max limit.
Extracts text and wraps each into the same Paper shape used everywhere
else in the system, tagged with source="user_upload" so it flows through
the exact same clean/chunk/embed/store/retrieve pipeline as fetched papers.
"""

from pathlib import Path
from pypdf import PdfReader
from app.data.europepmc_fetcher import Paper

MAX_UPLOADS = 4


def extract_text_from_pdf(file_path: str) -> str:
    """Extracts all text from a PDF file."""
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def extract_text_from_txt(file_path: str) -> str:
    """Reads a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def process_uploaded_files(file_paths: list) -> list:
    """
    Takes a list of file paths (PDF or .txt), extracts their text, and
    wraps each into a Paper object tagged as a user upload.

    Args:
        file_paths: list of local file paths (max 4 enforced)

    Returns:
        A list of Paper objects, one per uploaded file.
    """
    if len(file_paths) > MAX_UPLOADS:
        print(f"[Upload Handler] Warning: {len(file_paths)} files given, only first {MAX_UPLOADS} will be processed.")
        file_paths = file_paths[:MAX_UPLOADS]

    papers = []
    for i, path in enumerate(file_paths):
        file = Path(path)
        if not file.exists():
            print(f"[Upload Handler] Warning: file not found: {path}. Skipping.")
            continue

        try:
            if file.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(str(file))
            elif file.suffix.lower() == ".txt":
                text = extract_text_from_txt(str(file))
            else:
                print(f"[Upload Handler] Warning: unsupported file type: {file.suffix}. Skipping.")
                continue

            if not text.strip():
                print(f"[Upload Handler] Warning: no extractable text in {file.name}. Skipping.")
                continue

            papers.append(
                Paper(
                    id=f"upload_{i}_{file.stem}",
                    title=file.stem,  # filename without extension, used as a fallback title
                    summary=text,     # full extracted text goes here (processor.py will chunk it)
                    authors=["User Uploaded"],
                    published="unknown",
                    pdf_url=None,
                    source="user_upload",
                    is_india_affiliated=False
                )
            )
            print(f"[Upload Handler] Processed: {file.name} ({len(text)} characters extracted)")

        except Exception as e:
            print(f"[Upload Handler] Error processing {file.name}: {e}. Skipping.")

    return papers


if __name__ == "__main__":
    # Manual test — put a real PDF path here to test
    test_files = ["sample_paper.pdf"]  # replace with a real file path on your machine
    papers = process_uploaded_files(test_files)

    for p in papers:
        print(f"\nTitle: {p.title}")
        print(f"Source: {p.source}")
        print(f"Text length: {len(p.summary)} characters")
        print(f"Preview: {p.summary[:300]}...")