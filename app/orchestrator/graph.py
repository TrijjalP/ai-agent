"""
graph.py
The LangGraph orchestrator: wires all 6 agents into a formal graph,
replacing the manual `state = run_x(state)` chains used in each agent's
test block. This becomes the single entry point the FastAPI backend
(and eventually the Streamlit UI) will call to run the entire pipeline.
"""

from langgraph.graph import StateGraph, END
from app.orchestrator.state import ResearchState

from app.agents.retrieval_agent import run_retrieval_agent
from app.agents.extraction_agent import run_extraction_agent
from app.agents.gap_detection_agent import run_gap_detection_agent
from app.agents.hypothesis_generation_agent import run_hypothesis_generation_agent
from app.agents.hypothesis_critique_agent import run_hypothesis_critique_agent
from app.agents.ranking_agent import run_ranking_agent


def build_research_graph():
    """
    Builds and compiles the LangGraph pipeline:
    Retrieval -> Extraction -> Gap Detection -> Hypothesis Generation
    -> Critique -> Ranking -> END

    Returns:
        A compiled LangGraph app that can be invoked with an initial state.
    """
    graph = StateGraph(ResearchState)

    # Register each agent as a node in the graph
    graph.add_node("retrieval", run_retrieval_agent)
    graph.add_node("extraction", run_extraction_agent)
    graph.add_node("gap_detection", run_gap_detection_agent)
    graph.add_node("hypothesis_generation", run_hypothesis_generation_agent)
    graph.add_node("critique", run_hypothesis_critique_agent)
    graph.add_node("ranking", run_ranking_agent)

    # Define the sequential flow (a straight line for now; this is where
    # future loops/branches, e.g. "if scores too low, regenerate", would go)
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "extraction")
    graph.add_edge("extraction", "gap_detection")
    graph.add_edge("gap_detection", "hypothesis_generation")
    graph.add_edge("hypothesis_generation", "critique")
    graph.add_edge("critique", "ranking")
    graph.add_edge("ranking", END)

    return graph.compile()


# Build once, reuse across calls (e.g. from FastAPI later)
research_graph = build_research_graph()


if __name__ == "__main__":
    custom_weights = {
        "feasibility_score": 0.35,
        "novelty_score": 0.15,
        "evidence_grounding_score": 0.35,
        "specificity_score": 0.15
    }

    initial_state: ResearchState = {
        "topic": "biomedical engineering current trends and future prospects",
        "india_only": False,
        "year_from": 2021,
        "year_to": 2026,
        "gap_focus": "Compare AI in biomedical research in India versus other countries, focused on research from 2021 onwards",
        "score_weights": custom_weights,
        "uploaded_file_paths": ["test_paper.pdf"]
    }

    print("Running full research pipeline via LangGraph...\n")
    final_state = research_graph.invoke(initial_state)

    if not final_state.get("is_valid_topic", True):
        print(f"\n❌ Topic rejected: {final_state.get('validation_message')}")
        print("This system only processes biomedical/medical/life sciences research topics.")
    else:
        print("\n=== FINAL RANKED HYPOTHESES (via LangGraph) ===")
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