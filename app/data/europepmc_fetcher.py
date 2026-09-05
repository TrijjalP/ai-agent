"""
europepmc_fetcher.py
Fetches biomedical papers from Europe PMC (covers PubMed + PMC full text + preprints).
This is our PRIMARY biomedical data source.
No API key required.

Supports an optional filter for India-affiliated research, to help ground
our hypotheses in research relevant to India.
"""

import requests
from pydantic import BaseModel
from typing import List, Optional

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class Paper(BaseModel):
    """Same shape as the arXiv Paper model, so downstream code doesn't care about the source."""
    id: str
    title: str
    summary: str
    authors: List[str]
    published: str
    pdf_url: Optional[str] = None
    source: str = "europepmc"


def fetch_europepmc_papers(
    query: str,
    max_results: int = 5,
    india_only: bool = False
) -> List[Paper]:
    """
    Search Europe PMC for biomedical papers matching the query.

    Args:
        query: search term, e.g. "CRISPR gene editing cancer therapy"
        max_results: how many papers to fetch
        india_only: if True, restrict results to papers with at least one
                    author affiliated with an Indian institution

    Returns:
        A list of Paper objects.
    """
    full_query = query
    if india_only:
        # AFF:"India" filters on author affiliation text containing "India"
        full_query = f'({query}) AND AFF:"India"'

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

        pdf_url = None
        if r.get("pmcid"):
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{r['pmcid']}/"

        papers.append(
            Paper(
                id=r.get("id", r.get("pmid", "unknown")),
                title=r.get("title", "No title"),
                summary=summary,
                authors=authors,
                published=r.get("firstPublicationDate", "unknown"),
                pdf_url=pdf_url,
                source="europepmc"
            )
        )

    return papers


if __name__ == "__main__":
    print("=== General search (no India filter) ===")
    results = fetch_europepmc_papers("tuberculosis drug resistance", max_results=3)
    for i, paper in enumerate(results, start=1):
        print(f"\n--- Paper {i} ---")
        print("Title:", paper.title)
        print("Authors:", ", ".join(paper.authors))
        print("Published:", paper.published)

    print("\n\n=== India-affiliated search ===")
    india_results = fetch_europepmc_papers(
        "tuberculosis drug resistance", max_results=3, india_only=True
    )
    for i, paper in enumerate(india_results, start=1):
        print(f"\n--- Paper {i} ---")
        print("Title:", paper.title)
        print("Authors:", ", ".join(paper.authors))
        print("Published:", paper.published)