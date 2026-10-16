"""
hypothesis_critique_agent.py
The Hypothesis Critique Agent: takes each generated hypothesis and
critically evaluates it using an explicit 4-criteria rubric (Feasibility,
Novelty, Evidence Grounding, Specificity). Each criterion is scored 0-10
by the LLM, and we COMPUTE the overall soundness score ourselves as a
weighted average.

The weighting is USER-CONFIGURABLE via state['score_weights'].
"""

from app.utils.llm_json import call_llm_for_json
from app.orchestrator.state import ResearchState


DEFAULT_SCORE_WEIGHTS = {
    "feasibility_score": 0.25,
    "novelty_score": 0.25,
    "evidence_grounding_score": 0.25,
    "specificity_score": 0.25
}


CRITIQUE_PROMPT_TEMPLATE = """You are a peer reviewer. Evaluate this specific hypothesis carefully and output ONLY valid JSON, nothing else - no explanations, no markdown, no extra words before or after.

Hypothesis: "{hypothesis}"
Addresses gap: "{addresses_gap}"
Supporting literature: {chunks_block}

Carefully assign DIFFERENT scores based on this hypothesis's actual strengths and weaknesses - do not use the same score for every criterion. Score each from 0 (very poor) to 10 (excellent):
- feasibility_score: can this be tested with realistic methods?
- novelty_score: how different is this from existing research?
- evidence_grounding_score: how well does the supporting literature above actually support it?
- specificity_score: how precise and falsifiable is the claim?

List exactly 2 short weaknesses (under 10 words each), specific to THIS hypothesis.

Output must be valid JSON matching this exact structure (keys and value types only - do NOT copy these example values, compute your own based on the hypothesis above):
{{"feasibility_score": <integer 0-10>, "novelty_score": <integer 0-10>, "evidence_grounding_score": <integer 0-10>, "specificity_score": <integer 0-10>, "weaknesses": ["<short weakness>", "<short weakness>"], "existing_evidence_relation": "<short sentence>"}}

Output only the JSON object with your computed scores. Nothing else.
"""


def format_chunks_for_prompt(chunks: list) -> str:
    lines = []
    for chunk in chunks:
        snippet = chunk["text"][:200] + ("..." if len(chunk["text"]) > 200 else "")
        lines.append(f"- \"{chunk['paper_title']}\": {snippet}")
    return "\n".join(lines) if lines else "No supporting literature available."


def validate_weights(weights: dict) -> dict:
    if not weights:
        return DEFAULT_SCORE_WEIGHTS

    required_keys = set(DEFAULT_SCORE_WEIGHTS.keys())
    if set(weights.keys()) != required_keys:
        print("[Hypothesis Critique Agent] Warning: score_weights missing required keys. Using defaults.")
        return DEFAULT_SCORE_WEIGHTS

    total = sum(weights.values())
    if not (0.99 <= total <= 1.01):
        print(f"[Hypothesis Critique Agent] Warning: score_weights sum to {total}, not 1.0. Using defaults.")
        return DEFAULT_SCORE_WEIGHTS

    return weights


def compute_overall_score(scores: dict, weights: dict) -> float:
    total = 0.0
    for key, weight in weights.items():
        total += scores.get(key, 0) * weight
    return round(total, 2)


def critique_single_hypothesis(hypothesis: dict, chunks: list, weights: dict) -> dict:
    chunks_block = format_chunks_for_prompt(chunks)

    prompt = CRITIQUE_PROMPT_TEMPLATE.format(
        hypothesis=hypothesis.get("hypothesis", ""),
        addresses_gap=hypothesis.get("addresses_gap", ""),
        chunks_block=chunks_block
    )

    parsed, raw_response = call_llm_for_json(prompt, temperature=0.3, max_retries=3, agent_name="Hypothesis Critique Agent")

    if parsed and isinstance(parsed, dict):
        sub_scores = {
            "feasibility_score": parsed.get("feasibility_score", 0),
            "novelty_score": parsed.get("novelty_score", 0),
            "evidence_grounding_score": parsed.get("evidence_grounding_score", 0),
            "specificity_score": parsed.get("specificity_score", 0)
        }
        overall_score = compute_overall_score(sub_scores, weights)

        return {
            **sub_scores,
            "overall_soundness_score": overall_score,
            "weaknesses": parsed.get("weaknesses", []),
            "existing_evidence_relation": parsed.get("existing_evidence_relation", "")
        }
    else:
        print("[Hypothesis Critique Agent] Warning: could not parse LLM output as JSON. Using fallback.")
        return {
            "feasibility_score": 0,
            "novelty_score": 0,
            "evidence_grounding_score": 0,
            "specificity_score": 0,
            "overall_soundness_score": 0,
            "weaknesses": ["Could not parse critique"],
            "existing_evidence_relation": raw_response[:200] if raw_response else "N/A"
        }


def run_hypothesis_critique_agent(state: ResearchState) -> ResearchState:
    hypotheses = state.get("hypotheses", [])
    chunks = state.get("retrieved_chunks", [])
    weights = validate_weights(state.get("score_weights"))

    print(f"[Hypothesis Critique Agent] Critiquing {len(hypotheses)} hypothesis/hypotheses...")
    print(f"[Hypothesis Critique Agent] Using score weights: {weights}")

    critiques = []
    for i, hyp in enumerate(hypotheses, start=1):
        print(f"[Hypothesis Critique Agent] Evaluating hypothesis {i}/{len(hypotheses)}...")
        critique = critique_single_hypothesis(hyp, chunks, weights)
        critiques.append({
            "hypothesis": hyp.get("hypothesis", ""),
            **critique
        })

    print(f"[Hypothesis Critique Agent] Done. Critiqued {len(critiques)} hypothesis/hypotheses.")

    state["critiques"] = critiques
    return state


if __name__ == "__main__":
    from app.agents.retrieval_agent import run_retrieval_agent
    from app.agents.gap_detection_agent import run_gap_detection_agent
    from app.agents.hypothesis_generation_agent import run_hypothesis_generation_agent

    custom_weights = {
        "feasibility_score": 0.35,
        "novelty_score": 0.15,
        "evidence_grounding_score": 0.35,
        "specificity_score": 0.15
    }

    initial_state: ResearchState = {
        "topic": "AI in biomedical research",
        "india_only": False,
        "year_from": 2021,
        "year_to": 2026,
        "gap_focus": "Compare AI in biomedical research in India versus other countries, focused on research from 2021 onwards",
        "score_weights": custom_weights
    }

    state = run_retrieval_agent(initial_state, n_results=6)
    state = run_gap_detection_agent(state)
    state = run_hypothesis_generation_agent(state)
    state = run_hypothesis_critique_agent(state)

    print("\n=== Hypothesis Critiques ===")
    for i, crit in enumerate(state["critiques"], start=1):
        print(f"\n--- Critique {i} ---")
        print("Hypothesis:", crit["hypothesis"][:100], "...")
        print(f"  OVERALL SOUNDNESS: {crit['overall_soundness_score']}/10")