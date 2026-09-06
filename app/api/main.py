"""
main.py
FastAPI backend exposing the LangGraph research pipeline as an async API.
Flow: POST /research starts a background job and returns a job_id.
      GET /research/{job_id} polls for status and result.
"""

import uuid
import threading
import traceback
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.orchestrator.graph import research_graph
from app.orchestrator.state import ResearchState

app = FastAPI(title="Biomedical Agentic Research Assistant API")

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; tighten for real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store: job_id -> job info dict
# NOTE: this resets if the server restarts. Fine for a local/demo project;
# a real production system would use a database or Redis instead.
JOBS = {}


class ResearchRequest(BaseModel):
    topic: str
    india_only: bool = False
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    gap_focus: str = "General gaps in current research"
    score_weights: Optional[dict] = None
    uploaded_file_paths: Optional[List[str]] = []


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "running", "completed", "failed"
    created_at: str
    result: Optional[dict] = None
    error: Optional[str] = None


def run_pipeline_in_background(job_id: str, initial_state: ResearchState):
    """
    Runs the full LangGraph pipeline in a background thread, updating
    the job store with progress and final result/error.
    """
    try:
        JOBS[job_id]["status"] = "running"
        final_state = research_graph.invoke(initial_state)

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = dict(final_state)

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        print(f"[API] Job {job_id} failed: {e}")
        traceback.print_exc()


@app.post("/research", response_model=dict)
def start_research(request: ResearchRequest):
    """
    Starts a new research job in the background and returns its job_id
    immediately, so the client can poll for progress instead of waiting
    on one long HTTP request.
    """
    job_id = str(uuid.uuid4())

    initial_state: ResearchState = {
        "topic": request.topic,
        "india_only": request.india_only,
        "year_from": request.year_from,
        "year_to": request.year_to,
        "gap_focus": request.gap_focus,
        "score_weights": request.score_weights or {},
        "uploaded_file_paths": request.uploaded_file_paths or []
    }

    JOBS[job_id] = {
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None
    }

    thread = threading.Thread(target=run_pipeline_in_background, args=(job_id, initial_state))
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/research/{job_id}", response_model=JobStatusResponse)
def get_research_status(job_id: str):
    """
    Returns the current status of a research job, and its full result
    once completed.
    """
    job = JOBS.get(job_id)

    if not job:
        return JobStatusResponse(
            job_id=job_id,
            status="not_found",
            created_at="",
            error="No job found with this ID."
        )

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        result=job["result"],
        error=job["error"]
    )


@app.get("/")
def root():
    return {"message": "Biomedical Agentic Research Assistant API is running."}