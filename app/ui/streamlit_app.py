"""
streamlit_app.py
The Streamlit frontend for the Biomedical Agentic Research Assistant.
Collects user input, submits to the FastAPI backend, polls for results,
and renders the final research brief across clear sections.
"""

import streamlit as st
import requests
import time
import os
import tempfile

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Biomedical Research Assistant", layout="wide")

st.title("🧬 Biomedical Agentic Research Assistant")
st.caption("AI-powered literature review, gap analysis, and hypothesis generation — grounded in real, traceable research.")

# ---------- SIDEBAR: INPUT FORM ----------
with st.sidebar:
    st.header("Research Query")

    topic = st.text_input("Research topic", placeholder="e.g. tuberculosis drug resistance in India")

    india_only = st.checkbox("Prioritize Indian research", value=False)

    col1, col2 = st.columns(2)
    with col1:
        year_from = st.number_input("From year", min_value=1990, max_value=2026, value=2020)
    with col2:
        year_to = st.number_input("To year", min_value=1990, max_value=2026, value=2026)

    gap_focus = st.text_area(
        "What kind of gap analysis do you want?",
        placeholder="e.g. Compare India vs other countries, or leave general",
        value="General gaps in current research"
    )

    st.subheader("Score Weighting")
    st.caption("Adjust based on what matters most for your research context. Must total 100%.")
    feasibility_w = st.slider("Feasibility", 0, 100, 25)
    novelty_w = st.slider("Novelty", 0, 100, 25)
    grounding_w = st.slider("Evidence Grounding", 0, 100, 25)
    specificity_w = st.slider("Specificity", 0, 100, 25)

    total_weight = feasibility_w + novelty_w + grounding_w + specificity_w
    if total_weight != 100:
        st.warning(f"Weights sum to {total_weight}%, not 100%. Will default to equal weighting if not fixed.")

    st.subheader("Upload Your Own Papers (optional, max 4)")
    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type=["pdf"],
        accept_multiple_files=True
    )
    if uploaded_files and len(uploaded_files) > 4:
        st.warning("Only the first 4 files will be used.")
        uploaded_files = uploaded_files[:4]

    submit = st.button("🔬 Run Research Pipeline", type="primary", use_container_width=True)


def save_uploaded_files(files):
    """Saves Streamlit-uploaded files to a temp directory and returns their paths."""
    if not files:
        return []
    paths = []
    temp_dir = tempfile.mkdtemp()
    for f in files:
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        paths.append(path)
    return paths


def get_score_weights():
    if total_weight == 100:
        return {
            "feasibility_score": feasibility_w / 100,
            "novelty_score": novelty_w / 100,
            "evidence_grounding_score": grounding_w / 100,
            "specificity_score": specificity_w / 100
        }
    return None  # let backend default to equal weighting


# ---------- MAIN AREA: SUBMIT + POLL + RENDER ----------
if submit:
    if not topic.strip():
        st.error("Please enter a research topic.")
        st.stop()

    file_paths = save_uploaded_files(uploaded_files)

    payload = {
        "topic": topic,
        "india_only": india_only,
        "year_from": int(year_from),
        "year_to": int(year_to),
        "gap_focus": gap_focus,
        "score_weights": get_score_weights(),
        "uploaded_file_paths": file_paths
    }

    with st.spinner("Starting research pipeline..."):
        response = requests.post(f"{API_BASE_URL}/research", json=payload)
        job_id = response.json()["job_id"]

    st.session_state["job_id"] = job_id
    st.session_state["result"] = None

if "job_id" in st.session_state and st.session_state.get("result") is None:
    job_id = st.session_state["job_id"]

    progress_placeholder = st.empty()
    status = "queued"

    while status not in ("completed", "failed", "not_found"):
        poll_response = requests.get(f"{API_BASE_URL}/research/{job_id}")
        data = poll_response.json()
        status = data["status"]

        progress_placeholder.info(f"⏳ Status: **{status}**... this may take a few minutes.")
        time.sleep(5)

    progress_placeholder.empty()

    if status == "completed":
        st.session_state["result"] = data["result"]
    elif status == "failed":
        st.error(f"Pipeline failed: {data.get('error')}")
    else:
        st.error("Job not found.")

# ---------- RENDER RESULTS ----------
result = st.session_state.get("result")

if result:
    if not result.get("is_valid_topic", True):
        st.error(f"❌ Topic rejected: {result.get('validation_message')}")
        st.info("This system only processes biomedical/medical/life sciences research topics.")
    else:
        # SECTION 1: Query Summary
        st.header("📋 Query Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Topic", result["topic"])
        c2.metric("Year Range", f"{result.get('year_from', 'N/A')}–{result.get('year_to', 'N/A')}")
        c3.metric("Papers Analyzed", len(result.get("retrieved_chunks", [])))
        if result.get("india_only"):
            st.caption("🇮🇳 Prioritizing Indian-affiliated research")

        # SECTION 2: Literature Overview
        st.header("📚 Literature Overview")
        seen = set()
        for chunk in result.get("retrieved_chunks", []):
            title = chunk["paper_title"]
            if title in seen:
                continue
            seen.add(title)
            with st.expander(f"{title}  {'🇮🇳' if chunk['is_india_affiliated'] else ''}"):
                st.write(f"**Source:** {chunk['source']} | **Published:** {chunk['published']}")
                st.write(chunk["text"][:400] + "...")

        # SECTION 3: Extracted Key Findings
        st.header("🔍 Extracted Key Findings")
        for fact in result.get("extracted_facts", []):
            with st.expander(fact["paper_title"]):
                st.write(f"**Finding:** {fact['key_finding']}")
                st.write(f"**Entities:** {', '.join(fact.get('entities', []))}")
                if fact.get("statistics"):
                    st.write(f"**Statistics:** {', '.join(fact['statistics'])}")

        # SECTION 4: Identified Gaps
        st.header("🕳️ Identified Research Gaps")
        for i, gap in enumerate(result.get("identified_gaps", []), start=1):
            st.subheader(f"Gap {i}: {gap.get('gap_type', 'General')}")
            st.write(gap["gap_description"])
            if gap.get("supporting_papers"):
                st.caption("Supporting papers: " + ", ".join(gap["supporting_papers"]))

        # SECTION 5 & 6: Ranked Hypotheses + Rationale
        st.header("💡 Generated Hypotheses (Ranked)")
        for hyp in result.get("ranked_hypotheses", []):
            with st.container(border=True):
                st.subheader(f"Rank #{hyp['rank']} — Score: {hyp['overall_soundness_score']}/10")
                st.write(f"**{hyp['hypothesis']}**")
                st.write(f"*Addresses gap:* {hyp['addresses_gap']}")

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Feasibility", hyp["feasibility_score"])
                sc2.metric("Novelty", hyp["novelty_score"])
                sc3.metric("Grounding", hyp["evidence_grounding_score"])
                sc4.metric("Specificity", hyp["specificity_score"])

                st.write("**Weaknesses:**")
                for w in hyp.get("weaknesses", []):
                    st.write(f"- {w}")

                if hyp.get("ranking_rationale"):
                    st.info(f"**Why ranked #1:** {hyp['ranking_rationale']}")

                st.write("**Sources used:**")
                for src in hyp.get("source_breakdown", []):
                    tag = "🇮🇳" if "uploaded" in src["origin"].lower() else "🌐"
                    st.caption(f"{tag} {src['title']} — *{src['origin']}*")
else:
    st.info("👈 Enter a research topic in the sidebar and click **Run Research Pipeline** to get started.")