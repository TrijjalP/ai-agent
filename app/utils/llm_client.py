"""
llm_client.py
A thin wrapper around Ollama's local API. Every agent that needs to call
the LLM should use this function, so if we ever switch models or add
retry logic, we only change it in one place.
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "phi3"


def call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.3) -> str:
    """
    Send a prompt to the local Ollama LLM and return its text response.

    Args:
        prompt: the full prompt text to send
        model: which local model to use (default: phi3)
        temperature: lower = more focused/deterministic, higher = more creative

    Returns:
        The model's response as a plain string.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


if __name__ == "__main__":
    # Quick manual test
    reply = call_llm("In one sentence, what is tuberculosis?")
    print("LLM response:", reply)