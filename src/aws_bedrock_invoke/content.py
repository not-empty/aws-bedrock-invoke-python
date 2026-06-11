from __future__ import annotations

import json
from typing import Any

from .config import BedrockPromptCacheConfig


def prompt_cache_to_cache_control(
    config: BedrockPromptCacheConfig | None,
) -> dict[str, Any] | None:
    if config is None or not config.enabled:
        return None
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if config.ttl:
        cache_control["ttl"] = config.ttl
    return cache_control


def json_text_block(
    payload: dict[str, Any],
    *,
    bedrock_prompt_cache: BedrockPromptCacheConfig | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "text",
        "text": json.dumps(payload, ensure_ascii=False),
    }
    cache_control = prompt_cache_to_cache_control(bedrock_prompt_cache)
    if cache_control is not None:
        block["cache_control"] = cache_control
    return block


def text_block(
    text: str,
    *,
    bedrock_prompt_cache: BedrockPromptCacheConfig | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "text",
        "text": text,
    }
    cache_control = prompt_cache_to_cache_control(bedrock_prompt_cache)
    if cache_control is not None:
        block["cache_control"] = cache_control
    return block
