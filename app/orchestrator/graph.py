"""
graph.py
The LangGraph orchestrator: wires the 4 reasoning agents into a formal
graph. Ranking is deliberately NOT a graph node — it's pure sorting +
formatting over data the Critique Agent already produced, so it runs as
a plain Python function right after the graph finishes, not as an LLM
agent step.
"""

from langgraph.graph import StateGraph, END
from app.orchestrator.state import ResearchState

from app.agents.retrieval_agent import run_retrieval_agent
from app.agents.comparison_agent import run_comparison_agent
from app.agents.hypothesis_generation_agent import run_hypothesis_generation_agent
from app.agents.hypothesis_critique_agent import run_hypothesis_critique_agent
from app.agents.ranking_agent import run_ranking_agent


def build_research_graph():
    """
    Builds and compiles the LangGraph pipeline:
    Retrieval -> Comparison -> Hypothesis Generation -> Critique -> END

    Retrieval auto-selects the top 4 papers (by relevance) for now;
    candidate_papers is still returned in state so a "swap this paper"
    UI feature can be added later without touching this graph.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("retrieval", run_retrieval_agent)
    graph.add_node("comparison", run_comparison_agent)
    graph.add_node("hypothesis_generation", run_hypothesis_generation_agent)
    graph.add_node("critique", run_hypothesis_critique_agent)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "comparison")
    graph.add_edge("comparison", "hypothesis_generation")
    graph.add_edge("hypothesis_generation", "critique")
    graph.add_edge("critique", END)

    return graph.compile()


# Build once, reuse across calls (e.g. from FastAPI)
research_graph = build_research_graph()


def run_full_pipeline(initial_state: ResearchState) -> ResearchState:
    """
    Runs the full pipeline: the 4-agent LangGraph chain, then Ranking as a
    plain deterministic code step (not an LLM agent) immediately after.
    This is the single entry point the API/UI should call.
    """
    state = research_graph.invoke(initial_state)

    if not state.get("is_valid_topic", True):
        # Topic was rejected by the Retrieval Agent's built-in topic guard —
        # nothing to rank.
        return state

    state = run_ranking_agent(state)
    return state


if __name__ == "__main__":
    custom_weights = {
        "feasibility_score": 0.35,
        "novelty_score": 0.15,
        "evidence_grounding_score": 0.35,
        "specificity_score": 0.15
    }

    initial_state: ResearchState = {
        "topic": "compare human cloning research papers",
        "india_only": False,
        "year_from": 2020,
        "year_to": 2026,
        "gap_focus": "Compare these papers' methodology, conclusions, and differences",
        "score_weights": custom_weights
    }

    print("Running full research pipeline (4 agents + code-based ranking)...\n")
    final_state = run_full_pipeline(initial_state)

    if not final_state.get("is_valid_topic", True):
        print(f"\nTopic rejected: {final_state.get('validation_message')}")
        print("This system only processes biomedical/medical/life sciences research topics.")
    else:
        print("\n=== SELECTED PAPERS (auto-selected top 4) ===")
        for p in final_state.get("selected_papers", []):
            print(f"- {p.get('title', p.get('paper_title', 'Untitled'))}")

        print("\n=== CANDIDATE PAPERS (available to swap in, not used yet) ===")
        for p in final_state.get("candidate_papers", []):
            print(f"- {p.get('title', p.get('paper_title', 'Untitled'))}")

        print("\n=== PAPER SUMMARIES & COMPARISON ===")
        for s in final_state.get("paper_summaries", []):
           title = s.get("title") or s.get("paper_title") or "Untitled"
           methodology = s.get("methodology") or s.get("methodology_summary") or "N/A"
           conclusion = s.get("conclusion") or s.get("conclusion_summary") or "N/A"
           print(f"\n- {title}")
           print(f"  Methodology: {methodology}")
           print(f"  Conclusion: {conclusion}")

        print("\n=== RECOMMENDED PAPER ===")
        print(final_state.get("recommended_paper"))

        print("\n=== FINAL RANKED HYPOTHESES ===")
        for hyp in final_state["ranked_hypotheses"]:
            print(f"\n--- Rank #{hyp['rank']} (Score: {hyp['overall_soundness_score']}/10) ---")
            print("Hypothesis:", hyp["hypothesis"])
            print("Addresses gap:", hyp["addresses_gap"])
            print(f"  Sub-scores — Feasibility: {hyp['feasibility_score']}, Novelty: {hyp['novelty_score']}, "
                  f"Grounding: {hyp['evidence_grounding_score']}, Specificity: {hyp['specificity_score']}")
            print("  Weaknesses:", hyp["weaknesses"])
            print("  Sources used:")
            for src in hyp.get("source_breakdown", []):
                print(f"    - \"{src['title']}\" [{src['origin']}]")
            if hyp.get("ranking_rationale"):
                print("Why ranked #1:", hyp["ranking_rationale"])