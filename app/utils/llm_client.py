"""
llm_client.py
Calls the Groq-hosted LLM (replacing the old local Ollama/Phi-3 call).
Keeps the same function signature as before (prompt, model, temperature,
max_tokens) so every agent that already calls call_llm() keeps working
without any changes on their end.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)  # reads GROQ_API_KEY from your .env file

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to your .env file in the project root."
            )
        print(f"[DEBUG] Using key starting with: {api_key[:7]}... (length {len(api_key)})")
        _client = Groq(api_key=api_key)
    return _client


def call_llm(
    prompt: str,
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    context_window: int = None,  # kept for backward compatibility, unused by Groq
) -> str:
    """
    Sends a prompt to the Groq-hosted LLM and returns the plain text response.
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[LLM Client] Groq API call failed: {e}")
        return ""


if __name__ == "__main__":
    test_response = call_llm("Say hello in one short sentence.")
    print("Response:", test_response)