"""
llm_json.py
Combines calling the LLM with JSON extraction and automatic retry.
Small local LLMs occasionally produce malformed/incomplete JSON on a given
call - retrying with progressively lower temperature significantly
increases the chance of a clean, parseable response.
"""

from app.utils.llm_client import call_llm
from app.utils.json_parser import extract_json


def call_llm_for_json(prompt: str, temperature: float = 0.3, max_retries: int = 3, agent_name: str = "Agent"):
    """
    Calls the LLM and attempts to parse JSON from its response, retrying
    on failure. Each retry lowers the temperature slightly, since lower
    temperature tends to produce more strictly formatted, rule-following
    output — helpful specifically for fixing JSON formatting failures.

    Returns:
        (parsed_json_or_None, raw_response_from_last_attempt)
    """
    parsed = None
    raw_response = ""
    attempts = max_retries + 1

    for attempt in range(attempts):
        current_temp = max(0.0, temperature - (attempt * 0.1))  # step down each retry
        raw_response = call_llm(prompt, temperature=current_temp)
        parsed = extract_json(raw_response)

        if parsed is not None:
            return parsed, raw_response

        if attempt < attempts - 1:
            print(f"[{agent_name}] JSON parse failed (attempt {attempt + 1}/{attempts}), retrying at temperature {max(0.0, temperature - ((attempt+1) * 0.1)):.1f}...")

    return None, raw_response