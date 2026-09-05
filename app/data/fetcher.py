"""
fetcher.py
Unified interface over all literature sources (Europe PMC, arXiv).
This is the ONLY file the rest of the app (agents, orchestrator) should
import from when it needs papers — it hides the fact that there are
multiple sources underneath.
"""

from typing import List, Optional
from app.data.europepmc_fetcher import fetch_europepmc_papers, Paper
from app.data.arxiv_fetcher import fetch_arxiv_papers


def fetch_papers(
    query: str,
    max_results: int = 5,
    india_only: bool = False,
    include_arxiv: bool = False
) -> List[Paper]:
    """
    Fetch biomedical literature for a given query.

    Args:
        query: search term/topic
        max_results: max papers PER SOURCE (not total)
        india_only: if True, restrict Europe PMC results to India-affiliated authors
        include_arxiv: if True, also pull q-bio preprints from arXiv
                        (off by default since Europe PMC is our primary source)

    Returns:
        A combined list of Paper objects from all requested sources.
    """
    all_papers: List[Paper] = []

    # Primary source: Europe PMC (PubMed + PMC + preprints)
    europepmc_results = fetch_europepmc_papers(
        query=query,
        max_results=max_results,
        india_only=india_only
    )
    all_papers.extend(europepmc_results)

    # Optional secondary source: arXiv (mainly q-bio preprints)
    if include_arxiv:
        arxiv_results = fetch_arxiv_papers(query=query, max_results=max_results)
        # Convert arXiv's Paper objects into the same shape, tagging source
        for p in arxiv_results:
            all_papers.append(
                Paper(
                    id=p.id,
                    title=p.title,
                    summary=p.summary,
                    authors=p.authors,
                    published=p.published,
                    pdf_url=p.pdf_url,
                    source="arxiv"
                )
            )

    return all_papers


if __name__ == "__main__":
    # Test: general biomedical query, Europe PMC only
    print("=== Default (Europe PMC only) ===")
    papers = fetch_papers("diabetes type 2 South Asian population", max_results=2)
    for p in papers:
        print(f"[{p.source}] {p.title}")

    # Test: India-only filter
    print("\n=== India-affiliated only ===")
    papers = fetch_papers("diabetes type 2 South Asian population", max_results=2, india_only=True)
    for p in papers:
        print(f"[{p.source}] {p.title}")

    # Test: including arXiv too
    print("\n=== Europe PMC + arXiv ===")
    papers = fetch_papers("gene expression machine learning", max_results=2, include_arxiv=True)
    for p in papers:
        print(f"[{p.source}] {p.title}")