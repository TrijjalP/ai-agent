"""
ingest.py
Bulk ingestion script — builds up our actual knowledge base by fetching
a meaningful volume of papers across multiple biomedical topics and years,
then chunking + embedding + storing them all in ChromaDB.
"""

from app.data.fetcher import fetch_papers
from app.data.processor import process_paper
from app.embeddings.vector_store import store_chunks, count_chunks

TOPICS = [
    # --- Original core topics (India/global disease burden focus) ---
    "tuberculosis drug resistance",
    "diabetes type 2 South Asian population",
    "CRISPR gene editing cancer therapy",
    "antimicrobial resistance India",
    "cardiovascular disease risk factors India",
    "dengue fever epidemiology",
    "cancer immunotherapy",
    "malaria treatment resistance",

    # --- Cutting-edge / trending topics (2024-2026 relevance) ---
    "large language models clinical medical question answering",
    "medical AI safety evaluation clinical benchmarks",
    "GLP-1 receptor agonists metabolic disease",
    "semaglutide tirzepatide cardiovascular risk",
    "GLP-1 receptor agonist neuroprotection addiction",
    "CAR T-cell therapy autoimmune disease",
    "CAR T-cell therapy lupus rheumatoid arthritis",
    "synthetic DNA expanded genetic alphabet",
    "hachimoji synthetic biology RNA polymerase",
    "microplastics nanoplastics human brain accumulation",
    "microplastics blood-brain barrier health effects",
]

PAPERS_PER_TOPIC = 15
YEAR_FROM = 2020
YEAR_TO = 2026


def ingest_topic(topic: str, india_only: bool = False):
    print(f"\nFetching papers for: '{topic}' (india_only={india_only})")

    papers = fetch_papers(
        query=topic,
        max_results=PAPERS_PER_TOPIC,
        india_only=india_only,
        year_from=YEAR_FROM,
        year_to=YEAR_TO
    )

    print(f"  Retrieved {len(papers)} papers with abstracts")

    all_chunks = []
    for paper in papers:
        all_chunks.extend(process_paper(paper))

    store_chunks(all_chunks)
    return len(papers), len(all_chunks)


def run_ingestion():
    print(f"Starting bulk ingestion across {len(TOPICS)} topics...")
    print(f"Year range: {YEAR_FROM}-{YEAR_TO}\n")

    total_papers = 0
    total_chunks = 0

    for topic in TOPICS:
        p1, c1 = ingest_topic(topic, india_only=False)
        p2, c2 = ingest_topic(topic, india_only=True)
        total_papers += p1 + p2
        total_chunks += c1 + c2

    print("\n=== Ingestion complete ===")
    print(f"Total papers processed this run: {total_papers}")
    print(f"Total chunks added this run: {total_chunks}")
    print(f"Total chunks now in database: {count_chunks()}")


if __name__ == "__main__":
    run_ingestion()