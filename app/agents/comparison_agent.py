"""
comparison_agent.py
The Comparison Agent: takes the small working set of selected papers
(default 4) and, in a single LLM call:
1. Summarizes each paper's methodology and conclusion
2. Identifies concrete differences/under-explored angles ACROSS the set
   (reusing the same 'gap' shape the Hypothesis Generation Agent already
   expects, so that agent needs zero code changes)
3. Recommends which single paper is the strongest one to use as a
   reference, with a short rationale

This replaces the old Extraction + Gap Detection agents.
"""

from app.utils.llm_json import call_llm_for_json
from app.orchestrator.state import ResearchState


COMPARISON_PROMPT_TEMPLATE = """You are a senior biomedical research analyst comparing a small set of research papers for someone deciding which one to rely on and what remains unexplored.

The user's research topic is: "{topic}"
Comparison focus: "{gap_focus}"

Below are excerpts from {n_papers} papers, each numbered:
{papers_block}

Do THREE things:

1. For EACH paper, write a methodology_summary (1-2 sentences: what did they actually do/study) and a conclusion_summary (1-2 sentences: what did they find/conclude), based ONLY on the excerpts given.

2. Identify 3 to 5 concrete differences or under-explored angles ACROSS these papers (e.g. different methods used, conflicting conclusions, one covers something the others don't). For each:
   - gap_description: 20-35 words, clearly explaining the difference/gap.
   - gap_type: one short label (e.g. 'Methodological Difference', 'Conflicting Conclusion', 'Coverage Gap').
   - supporting_paper_numbers: list of paper NUMBERS (integers) this difference/gap relates to.

3. Recommend which ONE paper (by number) is the strongest to use as a reference overall, and why, in 1-2 sentences.

Respond ONLY with a valid JSON object in this exact format, with no extra text before or after:
{{
  "paper_summaries": [
    {{"paper_number": 1, "methodology_summary": "...", "conclusion_summary": "..."}}
  ],
  "gaps": [
    {{"gap_description": "...", "gap_type": "...", "supporting_paper_numbers": [1, 2]}}
  ],
  "recommended_paper_number": 1,
  "recommendation_rationale": "..."
}}
"""


def format_papers_for_prompt(papers: list, max_chunks_per_paper: int = 8) -> str:
    """
    Builds the per-paper excerpt block for the prompt. Capped per paper
    so a handful of small search-matched chunks (topic-driven mode) and
    a full uploaded document's worth of chunks (uploads-only mode) both
    stay within a safe prompt size — without this cap, a fully-parsed
    uploaded PDF (dozens of chunks) can blow past what the LLM will
    reliably return a complete, valid JSON response for.
    """
    lines = []
    for i, paper in enumerate(papers, start=1):
        region_tag = "India-affiliated" if paper.get("is_india_affiliated") else "General/Global"
        chunks = paper["chunks"][:max_chunks_per_paper]
        excerpt_text = "\n   ".join(chunks)
        truncated_note = ""
        if len(paper["chunks"]) > max_chunks_per_paper:
            truncated_note = f" [excerpt — {len(paper['chunks'])} total sections, showing first {max_chunks_per_paper}]"
        lines.append(
            f"{i}. Paper: \"{paper['paper_title']}\" "
            f"(Published: {paper.get('published', 'unknown')}, {region_tag}, Source: {paper['source']}){truncated_note}\n"
            f"   Excerpts: {excerpt_text}"
        )
    return "\n\n".join(lines)


def run_comparison_agent(state: ResearchState) -> ResearchState:
    """
    Comparison Agent's job:
    1. Take the selected working-set papers from state['selected_papers']
    2. Ask the LLM to summarize each, find differences/gaps across them,
       and recommend the strongest one
    3. Resolve paper NUMBERS back to real paper_id/paper_title ourselves,
       for accurate traceability

    Returns:
        Updated state with 'paper_summaries', 'identified_gaps', and
        'recommended_paper' populated.
    """
    papers = state.get("selected_papers", [])
    topic = state.get("topic", "the given research area")
    gap_focus = state.get("gap_focus", "Compare these papers' methodology, conclusions, and differences")

    print(f"[Comparison Agent] Comparing {len(papers)} paper(s) with focus: '{gap_focus}'")

    if not papers:
        print("[Comparison Agent] No papers available to compare.")
        state["paper_summaries"] = []
        state["identified_gaps"] = []
        state["recommended_paper"] = {}
        return state

    papers_block = format_papers_for_prompt(papers)
    prompt = COMPARISON_PROMPT_TEMPLATE.format(
        topic=topic,
        gap_focus=gap_focus,
        n_papers=len(papers),
        papers_block=papers_block
    )

    parsed, raw_response = call_llm_for_json(prompt, temperature=0.4, max_retries=3, max_tokens=3072, agent_name="Comparison Agent")

    if not parsed or not isinstance(parsed, dict):
        print("[Comparison Agent] Warning: could not parse LLM output as JSON. Using fallback.")
        state["paper_summaries"] = [
            {"paper_id": p["paper_id"], "paper_title": p["paper_title"],
             "methodology_summary": "Could not generate summary.", "conclusion_summary": "Could not generate summary."}
            for p in papers
        ]
        state["identified_gaps"] = [{
            "gap_description": raw_response[:300] if raw_response else "Could not identify differences",
            "supporting_papers": [],
            "gap_type": "Unparsed"
        }]
        state["recommended_paper"] = {
            "paper_id": papers[0]["paper_id"],
            "paper_title": papers[0]["paper_title"],
            "rationale": "Default to the most relevant retrieved paper (comparison could not be generated)."
        }
        return state

    # Resolve paper_summaries: attach real paper_id/paper_title by number
    paper_summaries = []
    for item in parsed.get("paper_summaries", []):
        num = item.get("paper_number")
        if isinstance(num, str) and num.strip().isdigit():
            num = int(num.strip())
        if isinstance(num, int) and 1 <= num <= len(papers):
            matched = papers[num - 1]
            paper_summaries.append({
                "paper_id": matched["paper_id"],
                "paper_title": matched["paper_title"],
                "methodology_summary": item.get("methodology_summary", ""),
                "conclusion_summary": item.get("conclusion_summary", "")
            })

    # Resolve gaps: supporting_paper_numbers -> real paper titles (same
    # number-based resolution pattern used everywhere else for accuracy)
    gaps = parsed.get("gaps", [])
    for gap in gaps:
        numbers = gap.get("supporting_paper_numbers", [])
        resolved_titles = []
        for num in numbers:
            if isinstance(num, str) and num.strip().isdigit():
                num = int(num.strip())
            if isinstance(num, int) and 1 <= num <= len(papers):
                resolved_titles.append(papers[num - 1]["paper_title"])
        gap["supporting_papers"] = resolved_titles

    # Resolve the recommended paper
    rec_num = parsed.get("recommended_paper_number")
    if isinstance(rec_num, str) and rec_num.strip().isdigit():
        rec_num = int(rec_num.strip())

    if isinstance(rec_num, int) and 1 <= rec_num <= len(papers):
        matched = papers[rec_num - 1]
        recommended_paper = {
            "paper_id": matched["paper_id"],
            "paper_title": matched["paper_title"],
            "rationale": parsed.get("recommendation_rationale", "")
        }
    else:
        # Fallback: the most relevant paper by retrieval score
        recommended_paper = {
            "paper_id": papers[0]["paper_id"],
            "paper_title": papers[0]["paper_title"],
            "rationale": "Default to the most relevant retrieved paper (recommendation could not be parsed)."
        }

    print(f"[Comparison Agent] Summarized {len(paper_summaries)} paper(s), "
          f"found {len(gaps)} difference(s)/gap(s), recommended: \"{recommended_paper['paper_title']}\"")

    state["paper_summaries"] = paper_summaries
    state["identified_gaps"] = gaps
    state["recommended_paper"] = recommended_paper
    return state


if __name__ == "__main__":
    from app.agents.retrieval_agent import run_retrieval_agent

    initial_state: ResearchState = {
        "topic": "human cloning research",
        "india_only": False,
        "year_from": 2020,
        "year_to": 2020,
        "gap_focus": "Compare these papers' methodology, conclusions, and key differences",
        "uploaded_file_paths": []
    }

    state = run_retrieval_agent(initial_state)
    state = run_comparison_agent(state)

    print("\n=== Paper Summaries ===")
    for s in state["paper_summaries"]:
        print(f"\n- {s['paper_title']}")
        print(f"  Methodology: {s['methodology_summary']}")
        print(f"  Conclusion: {s['conclusion_summary']}")

    print("\n=== Key Differences / Gaps ===")
    for i, gap in enumerate(state["identified_gaps"], start=1):
        print(f"\n--- {i} ({gap.get('gap_type', 'Unknown')}) ---")
        print("Description:", gap["gap_description"])
        print("Supporting papers:", gap.get("supporting_papers", []))

    print("\n=== Recommended Paper ===")
    print(state["recommended_paper"])