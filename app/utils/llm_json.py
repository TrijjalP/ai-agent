"""
llm_json.py
Combines calling the LLM with JSON extraction and automatic retry.
Small local LLMs occasionally produce malformed/incomplete JSON on a given
call - retrying once significantly reduces failure rate. Every agent that
needs JSON back from the LLM should use this instead of calling call_llm
and extract_json separately.
"""

from app.utils.llm_client import call_llm
from app.utils.json_parser import extract_json


def call_llm_for_json(prompt: str, temperature: float = 0.3, max_retries: int = 1, agent_name: str = "Agent"):
    """
    Calls the LLM and attempts to parse JSON from its response, retrying
    on failure up to max_retries additional times.

    Returns:
        (parsed_json_or_None, raw_response_from_last_attempt)
    """
    parsed = None
    raw_response = ""
    attempts = max_retries + 1

    for attempt in range(attempts):
        raw_response = call_llm(prompt, temperature=temperature)
        parsed = extract_json(raw_response)

        if parsed is not None:
            return parsed, raw_response

        if attempt < attempts - 1:
            print(f"[{agent_name}] JSON parse failed, retrying ({attempt + 1}/{max_retries})...")

    return None, raw_response