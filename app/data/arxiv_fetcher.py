"""
arxiv_fetcher.py
Fetches papers from arXiv based on a search query.
This is the first piece of our Data / Knowledge Layer.
"""

import arxiv
from pydantic import BaseModel
from typing import List


class Paper(BaseModel):
    """A clean, structured representation of a single paper."""
    id: str
    title: str
    summary: str
    authors: List[str]
    published: str
    pdf_url: str


def fetch_arxiv_papers(query: str, max_results: int = 5) -> List[Paper]:
    """
    Search arXiv for papers matching the query.

    Args:
        query: search term, e.g. "graph neural networks for drug discovery"
        max_results: how many papers to fetch

    Returns:
        A list of Paper objects.
    """
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in client.results(search):
        papers.append(
            Paper(
                id=result.entry_id,
                title=result.title,
                summary=result.summary,
                authors=[author.name for author in result.authors],
                published=str(result.published),
                pdf_url=result.pdf_url
            )
        )

    return papers


if __name__ == "__main__":
    # Quick manual test — run this file directly to check it works
    results = fetch_arxiv_papers("agentic AI multi-agent systems", max_results=3)
    for i, paper in enumerate(results, start=1):
        print(f"\n--- Paper {i} ---")
        print("Title:", paper.title)
        print("Authors:", ", ".join(paper.authors))
        print("Published:", paper.published)
        print("PDF:", paper.pdf_url)
        print("Summary:", paper.summary[:200], "...")