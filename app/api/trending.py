"""
trending.py
Works out the "trending topics" shown in the welcome popup — purely from the
papers already collected in the knowledge base (no external calls).

How: for every curated ingest topic, look up its closest chunks in the vector
store, keep only papers published in TREND_YEAR, and count the distinct papers
per topic. The topics with the most papers this year are "trending"; each is
shown with its most recent paper (title + publication date).

The result is cached after the first computation (the knowledge base doesn't
change while the server runs), and warm_trending() can be called at startup so
the first visitor doesn't wait.
"""

import ast
import threading
from pathlib import Path

from app.embeddings.vector_store import search_chunks

TREND_YEAR = "2026"
MAX_TOPICS = 3
CHUNKS_PER_TOPIC = 80  # how many nearest chunks to inspect per topic

# Used only if the curated TOPICS list in app/data/ingest.py can't be read.
FALLBACK_TOPICS = [
    "cancer immunotherapy",
    "CRISPR gene editing",
    "mRNA vaccines",
    "Alzheimer's disease",
    "antimicrobial resistance",
    "diabetes and obesity",
    "gut microbiome",
    "stem cell therapy",
]

_cache = {"topics": None}
_lock = threading.Lock()


def _load_topics() -> list:
    """
    Reads the TOPICS list out of app/data/ingest.py WITHOUT importing it
    (importing could run ingestion code). Falls back to a built-in list.
    """
    try:
        ingest_path = Path(__file__).resolve().parents[1] / "data" / "ingest.py"
        tree = ast.parse(ingest_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "TOPICS" for t in node.targets
            ):
                raw = ast.literal_eval(node.value)
                topics = []
                for t in raw:
                    if isinstance(t, str):
                        topics.append(t)
                    elif isinstance(t, dict):
                        name = t.get("topic") or t.get("query") or t.get("name")
                        if name:
                            topics.append(str(name))
                    elif isinstance(t, (tuple, list)) and t:
                        topics.append(str(t[0]))
                if topics:
                    return topics
    except Exception as e:
        print(f"[Trending] Could not read TOPICS from ingest.py ({e}); using fallback list")
    return FALLBACK_TOPICS


def _papers_for_topic(topic: str) -> dict:
    """paper_id -> {paper_title, published} for TREND_YEAR papers close to the topic."""
    try:
        results = search_chunks(query=topic, n_results=CHUNKS_PER_TOPIC, india_only=False)
    except Exception as e:
        print(f"[Trending] Search failed for '{topic}': {e}")
        return {}

    metas = results.get("metadatas") or []
    metas = metas[0] if metas else []

    papers = {}
    for m in metas:
        published = str(m.get("published", "") or "")
        paper_id = m.get("paper_id")
        if not paper_id or not published.startswith(TREND_YEAR):
            continue
        if m.get("source") == "user_upload":  # uploaded files aren't "collected data"
            continue
        papers[paper_id] = {
            "paper_id": paper_id,
            "paper_title": m.get("paper_title") or "Untitled",
            "published": published,
        }
    return papers


def compute_trending_topics() -> list:
    candidates = []
    for topic in _load_topics():
        papers = _papers_for_topic(topic)
        if papers:
            ordered = sorted(papers.values(), key=lambda p: p["published"], reverse=True)
            candidates.append({"topic": topic, "count": len(papers), "papers": ordered})

    # Most papers this year first; ties go to the topic with the newest paper.
    candidates.sort(key=lambda c: (c["count"], c["papers"][0]["published"]), reverse=True)

    chosen, used_ids = [], set()
    for c in candidates:
        paper = next((p for p in c["papers"] if p["paper_id"] not in used_ids), None)
        if paper is None:
            continue
        used_ids.add(paper["paper_id"])
        chosen.append({
            "topic": c["topic"],
            "paper_count": c["count"],
            "paper_title": paper["paper_title"],
            "published": paper["published"],
            "paper_id": paper["paper_id"],
        })
        if len(chosen) == MAX_TOPICS:
            break

    print(f"[Trending] {len(chosen)} trending topic(s) for {TREND_YEAR}: {[c['topic'] for c in chosen]}")
    return chosen


def get_trending_topics() -> list:
    with _lock:
        if _cache["topics"] is None:
            _cache["topics"] = compute_trending_topics()
        return _cache["topics"]


def warm_trending():
    """Call from a background thread at startup to pre-fill the cache."""
    try:
        get_trending_topics()
    except Exception as e:
        print(f"[Trending] Warm-up failed: {e}")