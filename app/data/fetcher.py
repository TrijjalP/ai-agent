"""
fetcher.py
Unified interface over all literature sources (Europe PMC, arXiv).
This is the ONLY file the rest of the app (agents, orchestrator) should
import from when it needs papers.
"""

from typing import List, Optional
from app.data.europepmc_fetcher import fetch_europepmc_papers, Paper
from app.data.arxiv_fetcher import fetch_arxiv_papers


def fetch_papers(
    query: str,
    max_results: int = 5,
    india_only: bool = False,
    include_arxiv: bool = False,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None
) -> List[Paper]:
    """
    Fetch biomedical literature for a given query.

    Args:
        query: search term/topic
        max_results: max papers PER SOURCE (not total)
        india_only: if True, restrict Europe PMC results to India-affiliated authors
        include_arxiv: if True, also pull q-bio preprints from arXiv
        year_from: only include papers published on/after this year
        year_to: only include papers published on/before this year

    Returns:
        A combined list of Paper objects from all requested sources.
    """
    all_papers: List[Paper] = []

    europepmc_results = fetch_europepmc_papers(
        query=query,
        max_results=max_results,
        india_only=india_only,
        year_from=year_from,
        year_to=year_to
    )
    all_papers.extend(europepmc_results)

    if include_arxiv:
        arxiv_results = fetch_arxiv_papers(query=query, max_results=max_results)
        for p in arxiv_results:
            all_papers.append(
                Paper(
                    id=p.id,
                    title=p.title,
                    summary=p.summary,
                    authors=p.authors,
                    published=p.published,
                    pdf_url=p.pdf_url,
                    source="arxiv",
                    is_india_affiliated=False
                )
            )

    return all_papers


if __name__ == "__main__":
    print("=== Default (Europe PMC only) ===")
    papers = fetch_papers("diabetes type 2 South Asian population", max_results=2)
    for p in papers:
        print(f"[{p.source}] {p.title}")

    print("\n=== India-affiliated only ===")
    papers = fetch_papers("diabetes type 2 South Asian population", max_results=2, india_only=True)
    for p in papers:
        print(f"[{p.source}] {p.title}")

    print("\n=== Europe PMC + arXiv ===")
    papers = fetch_papers("gene expression machine learning", max_results=2, include_arxiv=True)
    for p in papers:
        print(f"[{p.source}] {p.title}")