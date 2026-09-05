"""
vector_store.py
Stores and retrieves embedded text chunks using ChromaDB (local, free, no server needed).
"""

import chromadb
from typing import List
from app.data.processor import Chunk
from app.embeddings.embedder import embed_texts

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "biomedical_chunks"

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def store_chunks(chunks: List[Chunk]):
    if not chunks:
        return

    collection = get_collection()

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    ids = [f"{c.paper_id}_{c.chunk_index}" for c in chunks]

    metadatas = [
        {
            "paper_id": c.paper_id,
            "paper_title": c.paper_title,
            "source": c.source,
            "chunk_index": c.chunk_index,
            "is_india_affiliated": c.is_india_affiliated
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Stored {len(chunks)} chunk(s) in ChromaDB.")


def search_chunks(query: str, n_results: int = 3, india_only: bool = False):
    """
    Semantic search: find the most relevant stored chunks for a query.

    Args:
        query: search text
        n_results: how many results to return
        india_only: if True, only search within India-affiliated chunks
    """
    collection = get_collection()
    query_embedding = embed_texts([query])[0]

    where_filter = {"is_india_affiliated": True} if india_only else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter
    )

    return results


def count_chunks() -> int:
    """Returns how many chunks are currently stored in the database."""
    collection = get_collection()
    return collection.count()


if __name__ == "__main__":
    from app.data.fetcher import fetch_papers
    from app.data.processor import process_paper

    papers = fetch_papers("tuberculosis drug resistance India", max_results=3, india_only=True)

    all_chunks = []
    for paper in papers:
        all_chunks.extend(process_paper(paper))

    store_chunks(all_chunks)

    print("\n=== Semantic search test ===")
    results = search_chunks("new treatments for resistant TB strains", n_results=2)

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n--- Result {i+1} (distance: {dist:.4f}) ---")
        print("From paper:", meta["paper_title"])
        print("Text:", doc[:200], "...")