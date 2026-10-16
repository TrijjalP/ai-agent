"""
europepmc_fetcher.py
Fetches biomedical papers from Europe PMC (covers PubMed + PMC full text + preprints).
This is our PRIMARY biomedical data source.
No API key required.

Supports an optional filter for India-affiliated research, to help ground
our hypotheses in research relevant to India. Also supports year filtering.

For full text: Europe PMC's structured XML ("fullTextXML" endpoint) is
only available for the narrow set of papers formally deposited in PMC's
XML archive. "isOpenAccess"/"inEPMC" being Y does NOT guarantee that —
it only guarantees Europe PMC can show you the article as HTML or let
you download the PDF (see each result's "fullTextUrlList"). So for most
open-access papers, we fetch the PDF Europe PMC links to and extract
text from that instead. Closed-access papers only ever have an abstract
available through any free API — that's a real constraint, not something
more code can fetch around.
"""

import re
import io
import requests
import pdfplumber
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import List, Optional, Tuple

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class Paper(BaseModel):
    """Same shape regardless of source, so downstream code doesn't care where it came from."""
    id: str
    title: str
    summary: str
    authors: List[str]
    published: str
    pdf_url: Optional[str] = None
    source: str = "europepmc"
    is_india_affiliated: bool = False
    is_open_access: bool = False
    full_text: Optional[str] = None  # only populated when we could actually get full text


def _find_fulltext_source(result: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Inspects a result's fullTextUrlList to find the best available
    Europe-PMC-hosted full text. Prefers structured XML (rare, best
    quality) and falls back to the PDF (common for OA papers).
    Returns (url, style) where style is "xml" or "pdf", or (None, None).
    """
    urls = (result.get("fullTextUrlList") or {}).get("fullTextUrl", [])
    xml_entry = next(
        (u for u in urls if u.get("site") == "Europe_PMC" and u.get("documentStyle") == "xml"),
        None
    )
    if xml_entry:
        return xml_entry["url"], "xml"

    pdf_entry = next(
        (u for u in urls if u.get("site") == "Europe_PMC" and u.get("documentStyle") == "pdf"),
        None
    )
    if pdf_entry:
        return pdf_entry["url"], "pdf"

    return None, None


def _extract_text_from_fulltext_xml(xml_text: str) -> str:
    soup = BeautifulSoup(xml_text, "lxml-xml")
    body = soup.find("body")
    target = body if body else soup
    text = target.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
    text = " ".join(pages_text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://europepmc.org/",
}


def fetch_fulltext(url: str, style: str, label: str = "") -> Optional[str]:
    """
    Fetches and extracts full text from a Europe-PMC-hosted URL.
    style is "xml" or "pdf" (as returned by _find_fulltext_source).
    Returns None if unavailable or the fetch/parse fails, printing a
    clear reason each time so systematic failures stay visible.
    """
    try:
        response = requests.get(url, timeout=30, headers=FETCH_HEADERS)
        if response.status_code != 200:
            print(f"[Europe PMC Fetcher] No full text for {label} (HTTP {response.status_code})")
            return None

        if style == "xml":
            text = _extract_text_from_fulltext_xml(response.text)
        else:  # "pdf"
            text = _extract_text_from_pdf(response.content)

        if not text:
            print(f"[Europe PMC Fetcher] Full text for {label} parsed to empty text")
            return None
        return text
    except Exception as e:
        print(f"[Europe PMC Fetcher] Could not fetch/parse full text for {label}: {type(e).__name__}: {e}")
        return None


def fetch_europepmc_papers(
    query: str,
    max_results: int = 5,
    india_only: bool = False,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    fetch_full_text: bool = False
) -> List[Paper]:
    """
    Search Europe PMC for biomedical papers matching the query.

    Args:
        query: search term, e.g. "CRISPR gene editing cancer therapy"
        max_results: how many papers to fetch
        india_only: if True, restrict results to India-affiliated authors
        year_from: if set, only include papers published on/after this year
        year_to: if set, only include papers published on/before this year
        fetch_full_text: if True, attempt to fetch full text (XML if
            available, otherwise the OA PDF) for each result that has one

    Returns:
        A list of Paper objects.
    """
    full_query = query

    if india_only:
        full_query = f'({full_query}) AND AFF:"India"'

    if year_from or year_to:
        start = year_from if year_from else 1900
        end = year_to if year_to else 2100
        full_query = f'({full_query}) AND PUB_YEAR:[{start} TO {end}]'

    params = {
        "query": full_query,
        "format": "json",
        "pageSize": max_results,
        "resultType": "core"
    }

    response = requests.get(BASE_URL, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()
    results = data.get("resultList", {}).get("result", [])

    papers = []
    for r in results:
        summary = r.get("abstractText")
        if not summary:
            continue

        authors_str = r.get("authorString", "")
        authors = [a.strip() for a in authors_str.split(",")] if authors_str else []

        pmcid = r.get("pmcid")
        pdf_url = None
        if pmcid:
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"

        is_open_access = r.get("isOpenAccess") == "Y"
        title_short = r.get("title", "")[:60]

        full_text = None
        if fetch_full_text:
            source_url, style = _find_fulltext_source(r)
            if source_url:
                print(f"[Europe PMC Fetcher] Fetching full text ({style}): {title_short}...")
                full_text = fetch_fulltext(source_url, style, label=title_short)

        papers.append(
            Paper(
                id=r.get("id", r.get("pmid", "unknown")),
                title=r.get("title", "No title"),
                summary=summary,
                authors=authors,
                published=r.get("firstPublicationDate", "unknown"),
                pdf_url=pdf_url,
                source="europepmc",
                is_india_affiliated=india_only,
                is_open_access=is_open_access,
                full_text=full_text
            )
        )

    return papers


if __name__ == "__main__":
    print("=== General search (no India filter) ===")
    results = fetch_europepmc_papers("tuberculosis drug resistance", max_results=3)
    for i, paper in enumerate(results, start=1):
        print(f"\n--- Paper {i} ---")
        print("Title:", paper.title)
        print("Open access:", paper.is_open_access)
        print("Full text fetched:", bool(paper.full_text))
        if paper.full_text:
            print("Full text length:", len(paper.full_text), "characters")