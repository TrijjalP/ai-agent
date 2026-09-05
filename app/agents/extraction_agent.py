"""
extraction_agent.py
The Extraction Agent: takes retrieved literature chunks and uses the local
LLM to pull out structured facts — key findings, entities, and statistics —
so downstream agents (Gap Detection, Hypothesis Generation) can reason over
clean structured data instead of raw paragraphs.
"""

import json
from app.utils.llm_client import call_llm
from app.orchestrator.state import ResearchState


EXTRACTION_PROMPT_TEMPLATE = """You are a biomedical research assistant. Read the excerpt below from a scientific paper and extract structured information.

Excerpt:
\"\"\"
{chunk_text}
\"\"\"

Respond ONLY with a valid JSON object in this exact format, with no extra text before or after:
{{
  "key_finding": "one sentence summarizing the main finding or claim in this excerpt",
  "entities": ["list", "of", "important", "drugs", "genes", "diseases", "or", "methods", "mentioned"],
  "statistics": ["list of any numeric findings/statistics mentioned, as short strings, empty list if none"]
}}
"""


def extract_facts_from_chunk(chunk_text: str) -> dict:
    """
    Calls the LLM on a single chunk and parses its JSON response.
    Falls back to a safe default if the LLM output isn't valid JSON
    (small local models occasionally produce malformed JSON).
    """
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
    raw_response = call_llm(prompt, temperature=0.2)

    try:
        # Small models sometimes wrap JSON in markdown code fences — strip those if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()

        parsed = json.loads(cleaned)
        return {
            "key_finding": parsed.get("key_finding", ""),
            "entities": parsed.get("entities", []),
            "statistics": parsed.get("statistics", [])
        }
    except (json.JSONDecodeError, AttributeError):
        # Graceful fallback — don't crash the whole pipeline over one bad extraction
        print("[Extraction Agent] Warning: could not parse LLM output as JSON. Using fallback.")
        return {
            "key_finding": raw_response[:200] if raw_response else "Could not extract",
            "entities": [],
            "statistics": []
        }


def run_extraction_agent(state: ResearchState) -> ResearchState:
    """
    Extraction Agent's job:
    1. Take each chunk from state['retrieved_chunks']
    2. Extract structured facts from it via the LLM
    3. Attach the extracted facts back, keeping traceability to the source paper

    Returns:
        Updated state with 'extracted_facts' populated.
    """
    chunks = state.get("retrieved_chunks", [])
    print(f"[Extraction Agent] Extracting facts from {len(chunks)} chunk(s)...")

    extracted_facts = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"[Extraction Agent] Processing chunk {i}/{len(chunks)}...")
        facts = extract_facts_from_chunk(chunk["text"])

        extracted_facts.append({
            "paper_title": chunk["paper_title"],
            "paper_id": chunk["paper_id"],
            "is_india_affiliated": chunk["is_india_affiliated"],
            "key_finding": facts["key_finding"],
            "entities": facts["entities"],
            "statistics": facts["statistics"]
        })

    print(f"[Extraction Agent] Done. Extracted facts from {len(extracted_facts)} chunk(s).")

    state["extracted_facts"] = extracted_facts
    return state


if __name__ == "__main__":
    # Test: chain Retrieval Agent -> Extraction Agent, exactly how LangGraph will do it later
    from app.agents.retrieval_agent import run_retrieval_agent

    initial_state: ResearchState = {
        "topic": "tuberculosis drug resistance in India",
        "india_only": True,
        "year_from": 2020,
        "year_to": 2026
    }

    state = run_retrieval_agent(initial_state, n_results=3)  # small number for a quick test
    state = run_extraction_agent(state)

    print("\n=== Extracted Facts ===")
    for i, fact in enumerate(state["extracted_facts"], start=1):
        print(f"\n--- Fact {i} ---")
        print("Paper:", fact["paper_title"])
        print("Key finding:", fact["key_finding"])
        print("Entities:", fact["entities"])
        print("Statistics:", fact["statistics"])