"""
json_parser.py
Shared utility for safely parsing JSON out of LLM responses.
Small local LLMs often wrap JSON in markdown code fences (```json ... ```)
or add stray text before/after — this handles that robustly in one place.
"""

import json
import re
from typing import Union


def extract_json(raw_response: str) -> Union[dict, list, None]:
    """
    Attempts to extract and parse a JSON object or array from raw LLM text,
    even if it's wrapped in markdown code fences or has extra text around it.

    Returns:
        The parsed JSON (dict or list), or None if parsing fails.
    """
    if not raw_response:
        return None

    text = raw_response.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    object_match = re.search(r"\{.*\}", text, re.DOTALL)

    candidates = []
    if object_match:
        candidates.append(object_match.group(0))
    if array_match:
        candidates.append(array_match.group(0))
    candidates.append(text)

    # Prefer the LARGEST valid match. A nested array inside an object
    # (e.g. the "weaknesses" list inside a dict) is technically valid JSON
    # on its own, so without this we could incorrectly parse just the
    # inner piece instead of the full intended structure.
    candidates = sorted(set(candidates), key=len, reverse=True)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    return None


if __name__ == "__main__":
    test1 = '```json\n[{"a": 1}]\n```'
    test2 = 'Here is the result:\n```\n{"key": "value"}\n```\nHope that helps!'
    test3 = '[{"x": "y"}]'
    test4 = 'garbage text with no json'
    test5 = '{"feasibility": "Medium", "weaknesses": ["a", "b"], "score": 5}'

    print(extract_json(test1))
    print(extract_json(test2))
    print(extract_json(test3))
    print(extract_json(test4))
    print(extract_json(test5))