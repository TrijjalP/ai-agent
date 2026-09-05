"""
retrieval_agent.py
The Retrieval Agent: given a research topic, this agent ensures relevant
literature is fetched (if not already in the knowledge base) and returns
the most semantically relevant chunks for downstream agents to use.
"""

from app.data.fetcher import fetch_papers
from app.data.processor import process_paper
from app.embeddings.vector_store import store_chunks, search_chunks
from app.orchestrator.state import ResearchState


def run_retrieval_agent(state: ResearchState, n_results: int = 8, fetch_fresh: bool = True) -> ResearchState:
    topic = state["topic"]
    india_only = state.get("india_only", False)
    year_from = state.get("year_from")
    year_to = state.get("year_to")

    print(f"[Retrieval Agent] Working on topic: '{topic}' (india_only={india_only})")

    if fetch_fresh:
        print("[Retrieval Agent] Fetching fresh papers from Europe PMC...")
        papers = fetch_papers(
            query=topic,
            max_results=10,
            india_only=india_only,
            year_from=year_from,
            year_to=year_to
        )
        print(f"[Retrieval Agent] Fetched {len(papers)} fresh papers")

        all_chunks = []
        for paper in papers:
            all_chunks.extend(process_paper(paper))

        store_chunks(all_chunks)

    print(f"[Retrieval Agent] Searching knowledge base for top {n_results} relevant chunks...")
    results = search_chunks(query=topic, n_results=n_results, india_only=india_only)

    retrieved_chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved_chunks.append({
                "text": doc,
                "paper_title": meta["paper_title"],
                "paper_id": meta["paper_id"],
                "source": meta["source"],
                "is_india_affiliated": meta["is_india_affiliated"],
                "published": meta.get("published", "unknown"),
                "relevance_distance": dist
            })

    print(f"[Retrieval Agent] Retrieved {len(retrieved_chunks)} relevant chunks")

    state["retrieved_chunks"] = retrieved_chunks
    return state


if __name__ == "__main__":
    initial_state: ResearchState = {
        "topic": "tuberculosis drug resistance in India",
        "india_only": True,
        "year_from": 2020,
        "year_to": 2026
    }

    updated_state = run_retrieval_agent(initial_state, n_results=5)

    print("\n=== Retrieved Chunks ===")
    for i, chunk in enumerate(updated_state["retrieved_chunks"], start=1):
        print(f"\n--- Chunk {i} ---")
        print("Paper:", chunk["paper_title"])
        print("Published:", chunk["published"])
        print("India-affiliated:", chunk["is_india_affiliated"])
        print("Distance:", round(chunk["relevance_distance"], 4))
        print("Text:", chunk["text"][:150], "...")