from __future__ import annotations

import json
import re
from typing import Any

from .exceptions import BedrockResponseParseError


def extract_response_text(response_envelope: dict[str, Any]) -> str:
    content = response_envelope.get("content")
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text", "")))
        return "".join(chunks)

    if isinstance(response_envelope.get("output_text"), str):
        return response_envelope["output_text"]

    if isinstance(response_envelope.get("completion"), str):
        return response_envelope["completion"]

    return json.dumps(response_envelope, ensure_ascii=False)


def cleanup_json_text(text: str) -> str:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.IGNORECASE | re.DOTALL)
    match = re.search(r"(\{.*\})", clean, flags=re.DOTALL)
    return match.group(1).strip() if match else clean


def parse_response_json(text: str) -> dict[str, Any]:
    clean = cleanup_json_text(text)
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise BedrockResponseParseError(f"Failed to parse JSON response: {exc}") from exc
    if not isinstance(data, dict):
        raise BedrockResponseParseError("Parsed JSON response is not an object.")
    return data
