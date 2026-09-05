"""
topic_guard.py
Lightweight topic validation - checks if a query is biomedical/medical
related BEFORE running the expensive agent pipeline. Uses fast keyword
heuristics first (no LLM call, no latency, no failure risk), and only
falls back to a single LLM check for genuinely ambiguous cases.
"""

from app.utils.llm_json import call_llm_for_json

# Strong biomedical signal words - if any appear, we're confident it's in-scope
BIOMEDICAL_KEYWORDS = [
    "medicine", "medical", "biomedical", "disease", "cancer", "tumor", "drug",
    "clinical", "patient", "diagnosis", "diagnostic", "therapy", "therapeutic",
    "treatment", "vaccine", "virus", "bacteria", "infection", "gene", "genetic",
    "genomics", "protein", "cell", "biology", "immunology", "immune", "pathology",
    "epidemiology", "public health", "healthcare", "surgery", "pharma",
    "pharmaceutical", "tuberculosis", "diabetes", "cardiovascular", "oncology",
    "neurology", "psychiatry", "mental health", "nursing", "dna", "rna", "clinical trial",
    "biomarker", "hospital", "physician", "doctor", "nurse", "symptom", "syndrome",
    "chronic", "acute", "pathogen", "microbiome", "biotech", "crispr", "antibody",
    "antibiotic", "resistance", "mutation", "biopsy", "radiology", "anatomy",
    "physiology", "pandemic", "outbreak", "wellness", "nutrition", "obesity"
]

# Strong non-biomedical signal words - if any appear (and no biomedical terms), reject fast
CLEARLY_UNRELATED_KEYWORDS = [
    "smartphone", "laptop", "recipe", "cooking", "car", "vehicle", "movie",
    "football", "cricket", "video game", "gaming", "stock market", "cryptocurrency",
    "fashion", "travel destination", "hotel booking", "programming language",
    "web development", "furniture", "real estate", "music album", "celebrity"
]

AMBIGUOUS_FALLBACK_PROMPT = """Is the following topic related to biomedical, medical, life sciences, public health, or clinical research? Answer only yes or no concepts apply broadly (e.g. "AI in healthcare" counts as yes).

Topic: "{topic}"

Respond ONLY with valid JSON: {{"is_biomedical": true or false}}
"""


def is_biomedical_topic(topic: str) -> tuple:
    """
    Checks if a topic is biomedical-related.

    Returns:
        (is_valid: bool, reason: str)
    """
    topic_lower = topic.lower()

    # Layer 1: fast keyword check - biomedical signal found
    if any(keyword in topic_lower for keyword in BIOMEDICAL_KEYWORDS):
        return True, "Topic contains clear biomedical/medical terminology."

    # Layer 1: fast keyword check - clearly unrelated, no biomedical terms at all
    if any(keyword in topic_lower for keyword in CLEARLY_UNRELATED_KEYWORDS):
        return False, "Topic appears unrelated to biomedical or medical research."

    # Layer 2: ambiguous case, single LLM fallback check
    prompt = AMBIGUOUS_FALLBACK_PROMPT.format(topic=topic)
    result, _ = call_llm_for_json(prompt, temperature=0.1, max_retries=1, agent_name="Topic Guard")

    if result and isinstance(result, dict):
        is_bio = result.get("is_biomedical", True)  # default true if ambiguous, don't over-block
        return is_bio, "Determined via secondary topic check." if is_bio else "Topic does not appear biomedical-related."

    # If even the fallback fails to parse, don't block a legitimate user - allow through
    return True, "Validation inconclusive; proceeding by default."


if __name__ == "__main__":
    test_topics = [
        "tuberculosis drug resistance in India",
        "best smartphones to buy in 2026",
        "AI in biomedical research",
        "how to cook pasta",
        "gene therapy for rare diseases",
        "quantum computing applications"  # ambiguous - not biomedical, not in either keyword list
    ]

    for t in test_topics:
        valid, reason = is_biomedical_topic(t)
        print(f"'{t}' -> Valid: {valid} | {reason}")