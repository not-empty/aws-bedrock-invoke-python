from __future__ import annotations

import os
from dataclasses import dataclass


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class BedrockPromptCacheConfig:
    enabled: bool = False
    ttl: str | None = None

    @classmethod
    def from_env(cls) -> "BedrockPromptCacheConfig":
        return cls(
            enabled=_env_truthy("BEDROCK_PROMPT_CACHE_ENABLED", default=False),
            ttl=str(os.getenv("BEDROCK_PROMPT_CACHE_TTL", "") or "").strip() or None,
        )


@dataclass(slots=True)
class BedrockClientConfig:
    region: str
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    connect_timeout: int = 10
    read_timeout: int = 300
    max_pool_connections: int | None = None

    @classmethod
    def from_env(cls) -> "BedrockClientConfig":
        region = str(os.getenv("AWS_REGION", "us-east-1") or "").strip()
        return cls(
            region=region,
            aws_access_key_id=str(os.getenv("AWS_ACCESS_KEY_ID", "") or "").strip() or None,
            aws_secret_access_key=str(os.getenv("AWS_SECRET_ACCESS_KEY", "") or "").strip() or None,
        )
