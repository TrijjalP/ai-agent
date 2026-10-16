# Agentic-AI_Project
An AI biomedical research agent that reads papers, compares them, and proposes new research hypotheses.
Give it a research question (or upload 2–4 papers) and it returns a side-by-side comparison, the gaps in current evidence, and ranked, testable hypotheses, each linked to its source papers.
How it works
User input → Retrieval → Comparison → Hypothesis Generation → Critique → Ranking → Results
Stage	What it does
Retrieval Agent	Checks the topic is biomedical, finds the most relevant papers (Europe PMC + vector knowledge base), applies India/Global and year filters, and keeps extra papers as swappable alternatives.
Comparison Agent	Summarizes each paper's methodology and conclusion and recommends the most relevant one.
Hypothesis Agent	Identifies gaps in the evidence and proposes novel hypotheses, citing papers by index so references are never invented.
Critique Agent	Scores each hypothesis on Feasibility, Novelty, Evidence Grounding and Specificity.
Ranking (code)	Combines the scores using the user's priority sliders and ranks hypotheses out of 10.
Two ways to start: a topic search, or an uploads-only comparison of the exact papers you provide. You can also swap a paper and re-run the agents.
Key features
Four cooperating agents orchestrated with LangGraph
Domain-specific biomedical embeddings with a ChromaDB knowledge base
India / Global paper scope and year-range filters
Adjustable evidence priorities (novelty, clinical relevance, rigor, specificity)
Upload your own PDF, DOCX or TXT papers
Trending topics from the collected 2026 papers
Responsive web interface, works on mobile
Tech stack
Python · FastAPI · LangGraph · Groq (LLM) · ChromaDB · Sentence-Transformers · Europe PMC API · HTML / CSS / JavaScript
