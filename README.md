# Agentic-Ai_Project
# BioMed-Agent

**An AI biomedical research agent that reads papers, compares them, and proposes new research hypotheses.**

Enter a research question, or upload 2–4 papers, and BioMed-Agent returns a side-by-side comparison of the papers, the gaps in current evidence, and ranked, testable hypotheses linked to their source papers.

## How it works

User input → Retrieval → Comparison → Hypothesis Generation → Critique → Ranking → Results

- **Retrieval Agent:** checks that the topic is biomedical, finds the most relevant papers, and applies India/Global and year filters
- **Comparison Agent:** summarizes each paper's methodology and conclusion and recommends the most relevant one
- **Hypothesis Agent:** finds gaps in the evidence and proposes hypotheses, citing papers by index so references are never invented
- **Critique Agent:** scores each hypothesis on feasibility, novelty, evidence grounding and specificity
- **Ranking:** combines the scores using your priority sliders and ranks hypotheses out of 10

## Features

- Topic search or uploads-only comparison (PDF, DOCX, TXT)
- Swap a paper and re-run the analysis
- India / Global scope and year-range filters
- Adjustable evidence priorities
- Trending topics from the collected 2026 papers
- Responsive web interface

## Tech stack

Python, FastAPI, LangGraph, Groq, ChromaDB, Sentence-Transformers, Europe PMC API, HTML/CSS/JavaScript

## Author

Designed and developed by **Trijjal Prajapati** (MTech project)
