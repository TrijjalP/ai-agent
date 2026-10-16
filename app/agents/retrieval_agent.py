"""
retrieval_agent.py
The Retrieval Agent: given a research topic, fetches/searches relevant
literature, groups the results by PAPER (not just by chunk), and selects
a small working set (default 4) of the most relevant papers for deep
comparison — while exposing the remaining matching papers as a
swappable "more papers like this" list, so the user can override the
agent's picks before Comparison/Hypothesis Generation runs.

Special case: if the user uploaded their own papers, those are compared
directly — exactly the papers uploaded (2, 3, 4, whatever count), no
topic search, no padding from the knowledge base. A topic is optional
in that mode (used only as extra framing context if provided).

Every paper's chunks are capped (see _cap_chunks_per_paper) before being
handed to the LLM agents downstream — Groq's free tier on this model has
an 8,000-token-per-request limit, and a fully-parsed uploaded PDF can
have 50+ chunks on its own, which blows way past that limit if left
uncapped.
"""

from app.data.fetcher import fetch_papers
from app.data.processor import process_paper
from app.data.upload_handler import build_papers_from_uploads
from app.embeddings.vector_store import store_chunks, search_chunks
from app.orchestrator.state import ResearchState
from app.utils.topic_guard import is_biomedical_topic


def _group_chunks_by_paper(chunks: list) -> list:
    """
    Groups raw retrieved chunks by paper_id, computing each paper's best
    (lowest) relevance distance and collecting all its chunk texts.
    Returns a list of paper dicts, sorted by relevance (most relevant first).
    """
    papers_by_id = {}

    for chunk in chunks:
        pid = chunk["paper_id"]
        if pid not in papers_by_id:
            papers_by_id[pid] = {
                "paper_id": pid,
                "paper_title": chunk["paper_title"],
                "source": chunk["source"],
                "is_india_affiliated": chunk["is_india_affiliated"],
                "published": chunk.get("published", "unknown"),
                "is_full_text": chunk.get("is_full_text", False),
                "chunks": [],
                "best_relevance_distance": chunk["relevance_distance"],
            }
        papers_by_id[pid]["chunks"].append(chunk["text"])
        papers_by_id[pid]["best_relevance_distance"] = min(
            papers_by_id[pid]["best_relevance_distance"], chunk["relevance_distance"]
        )
        if chunk.get("is_full_text", False):
            papers_by_id[pid]["is_full_text"] = True

    papers = list(papers_by_id.values())
    papers.sort(key=lambda p: p["best_relevance_distance"])
    return papers


def _cap_chunks_per_paper(papers: list, max_chunks: int = 3) -> list:
    """
    Caps how many chunks each paper carries forward into the LLM agents.
    Groq's free tier on this model has an 8,000-token-per-request limit,
    and a fully-parsed uploaded PDF can have 50+ chunks on its own — so
    this cap is what actually keeps every downstream prompt (Comparison,
    Hypothesis Generation, Critique) within that budget, for both
    uploads-only and topic-driven mode.
    """
    for p in papers:
        if len(p["chunks"]) > max_chunks:
            p["chunks"] = p["chunks"][:max_chunks]
    return papers


def _flatten_chunks(selected_papers: list) -> list:
    retrieved_chunks = []
    for p in selected_papers:
        for chunk_text in p["chunks"]:
            retrieved_chunks.append({
                "text": chunk_text,
                "paper_title": p["paper_title"],
                "paper_id": p["paper_id"],
                "source": p["source"],
                "is_india_affiliated": p["is_india_affiliated"],
                "published": p["published"],
                "is_full_text": p["is_full_text"],
            })
    return retrieved_chunks


def run_retrieval_agent(
    state: ResearchState,
    n_working_papers: int = 4,
    n_candidate_pool: int = 15,
    fetch_fresh: bool = True,
) -> ResearchState:
    """
    Retrieval Agent's job — two modes:

    UPLOADS-ONLY MODE (triggered whenever uploaded_file_paths is non-empty):
      Compare exactly the uploaded papers. No topic required, no
      knowledge-base search, no padding up to n_working_papers — if the
      user uploaded 2 papers, exactly 2 are compared; if 4, exactly 4.

    TOPIC-DRIVEN MODE (the original behavior, used when nothing is
    uploaded):
      1. Fetch/search a candidate pool of papers for the topic (+ filters)
      2. Filter out anything outside the requested year range
      3. Group results by paper, sorted by relevance
      4. Select the top `n_working_papers` (default 4) as the working set
      5. Expose the rest as 'candidate_papers' for the UI's swap feature
    """
    uploaded_file_paths = state.get("uploaded_file_paths", [])

    # ---------------- Uploads-only mode ----------------
    if uploaded_file_paths:
        print(f"[Retrieval Agent] Uploads-only mode: processing {len(uploaded_file_paths)} uploaded file(s)...")
        uploaded_papers = build_papers_from_uploads(uploaded_file_paths)
        print(f"[Retrieval Agent] Successfully processed {len(uploaded_papers)} uploaded paper(s)")

        if not uploaded_papers:
            state["is_valid_topic"] = False
            state["validation_message"] = (
                "Could not read any of the uploaded files. "
                "Please check they're valid PDF, DOCX, or TXT files."
            )
            state["selected_papers"] = []
            state["candidate_papers"] = []
            state["retrieved_chunks"] = []
            return state

        all_chunks = []
        for paper in uploaded_papers:
            all_chunks.extend(process_paper(paper))

        store_chunks(all_chunks)

        raw_chunks = [
            {
                "text": c.text,
                "paper_title": c.paper_title,
                "paper_id": c.paper_id,
                "source": c.source,
                "is_india_affiliated": c.is_india_affiliated,
                "published": c.published,
                "is_full_text": c.is_full_text,
                "relevance_distance": 0.0,  # not relevance-ranked — every uploaded paper is used
            }
            for c in all_chunks
        ]

        selected = _group_chunks_by_paper(raw_chunks)  # ALL uploaded papers, no cap on paper COUNT
        selected = _cap_chunks_per_paper(selected)      # but cap chunks per paper, for Groq's token limit

        if not state.get("topic"):
            state["topic"] = "Comparison of uploaded papers"

        state["is_valid_topic"] = True
        state["validation_message"] = ""
        state["selected_papers"] = selected
        state["candidate_papers"] = []  # nothing else to swap in — uploads are the whole set
        state["retrieved_chunks"] = _flatten_chunks(selected)

        print(f"[Retrieval Agent] Uploads-only working set: {len(selected)} paper(s), "
              f"{len(state['retrieved_chunks'])} chunk(s) total.")
        return state

    # ---------------- Topic-driven mode (unchanged) ----------------
    topic = state["topic"]

    is_valid, reason = is_biomedical_topic(topic)
    state["is_valid_topic"] = is_valid
    state["validation_message"] = reason

    if not is_valid:
        print(f"[Retrieval Agent] Topic REJECTED: {reason}")
        state["selected_papers"] = []
        state["candidate_papers"] = []
        state["retrieved_chunks"] = []
        return state

    india_only = state.get("india_only", False)
    year_from = state.get("year_from")
    year_to = state.get("year_to")
    confirmed_paper_ids = state.get("confirmed_paper_ids")

    print(f"[Retrieval Agent] Working on topic: '{topic}' (india_only={india_only})")

    all_new_chunks = []

    if fetch_fresh:
        print("[Retrieval Agent] Fetching fresh papers from Europe PMC...")
        papers = fetch_papers(
            query=topic,
            max_results=n_candidate_pool,
            india_only=india_only,
            year_from=year_from,
            year_to=year_to
        )
        print(f"[Retrieval Agent] Fetched {len(papers)} fresh papers")

        for paper in papers:
            all_new_chunks.extend(process_paper(paper))

    if all_new_chunks:
        store_chunks(all_new_chunks)

    search_n = max(n_candidate_pool * 4, 40)
    print(f"[Retrieval Agent] Searching knowledge base (pulling up to {search_n} chunks)...")
    results = search_chunks(query=topic, n_results=search_n, india_only=india_only)

    raw_chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            raw_chunks.append({
                "text": doc,
                "paper_title": meta["paper_title"],
                "paper_id": meta["paper_id"],
                "source": meta["source"],
                "is_india_affiliated": meta["is_india_affiliated"],
                "published": meta.get("published", "unknown"),
                "is_full_text": meta.get("is_full_text", False),
                "relevance_distance": dist
            })

    if year_from or year_to:
        filtered_chunks = []
        for chunk in raw_chunks:
            pub_year = None
            published_str = chunk.get("published", "")
            if published_str and published_str != "unknown":
                try:
                    pub_year = int(published_str[:4])
                except ValueError:
                    pub_year = None

            if pub_year is None:
                continue
            if year_from and pub_year < year_from:
                continue
            if year_to and pub_year > year_to:
                continue
            filtered_chunks.append(chunk)

        print(f"[Retrieval Agent] Year filter ({year_from}-{year_to}): "
              f"{len(raw_chunks)} chunks -> {len(filtered_chunks)} chunks")
        raw_chunks = filtered_chunks

    all_papers = _group_chunks_by_paper(raw_chunks)
    print(f"[Retrieval Agent] Found {len(all_papers)} distinct paper(s) in candidate pool")

    if confirmed_paper_ids:
        id_set = set(confirmed_paper_ids)
        selected = [p for p in all_papers if p["paper_id"] in id_set]
        remaining = [p for p in all_papers if p["paper_id"] not in id_set]
        print(f"[Retrieval Agent] Using user-confirmed paper selection ({len(selected)} papers)")
    else:
        selected = all_papers[:n_working_papers]
        remaining = all_papers[n_working_papers:n_candidate_pool]
        print(f"[Retrieval Agent] Auto-selected top {len(selected)} most relevant paper(s)")

    selected = _cap_chunks_per_paper(selected)

    candidate_papers = [
        {
            "paper_id": p["paper_id"],
            "paper_title": p["paper_title"],
            "source": p["source"],
            "is_india_affiliated": p["is_india_affiliated"],
            "published": p["published"],
            "is_full_text": p["is_full_text"],
        }
        for p in remaining
    ]

    state["selected_papers"] = selected
    state["candidate_papers"] = candidate_papers
    state["retrieved_chunks"] = _flatten_chunks(selected)

    print(f"[Retrieval Agent] Working set: {len(selected)} paper(s), "
          f"{len(state['retrieved_chunks'])} chunk(s) total. "
          f"{len(candidate_papers)} alternate paper(s) available to swap in.")

    return state


if __name__ == "__main__":
    initial_state: ResearchState = {
        "topic": "human cloning research",
        "india_only": False,
        "year_from": 2020,
        "year_to": 2020,
        "uploaded_file_paths": []
    }

    updated_state = run_retrieval_agent(initial_state)

    print("\n=== Selected Working Set ===")
    for p in updated_state["selected_papers"]:
        print(f"\n- {p['paper_title']} ({p['published']}, {p['source']})")
        print(f"  Full text: {p['is_full_text']}")
        print(f"  Chunks available: {len(p['chunks'])}")

    print("\n=== More Papers Like This (swap options) ===")
    for p in updated_state["candidate_papers"]:
        print(f"- {p['paper_title']} ({p['published']}, {p['source']}) [Full text: {p['is_full_text']}]")