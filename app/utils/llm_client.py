"""
llm_client.py
A thin wrapper around Ollama's local API. Every agent that needs to call
the LLM should use this function, so if we ever switch models or add
retry logic, we only change it in one place.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "phi3"


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    context_window: int = 4096
) -> str:
    """
    Send a prompt to the local Ollama LLM and return its text response.

    Args:
        prompt: the full prompt text to send
        model: which local model to use (default: phi3)
        temperature: lower = more focused/deterministic, higher = more creative
        max_tokens: max length of the response (num_predict)
        context_window: total token budget shared between prompt + response
                         (num_ctx) — increased from Ollama's default (often
                         2048) since our prompts with multiple facts can be
                         long, leaving too little room for the response
                         otherwise, causing truncated/invalid JSON.

    Returns:
        The model's response as a plain string.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": context_window
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


if __name__ == "__main__":
    reply = call_llm("In one sentence, what is tuberculosis?")
    print("LLM response:", reply)