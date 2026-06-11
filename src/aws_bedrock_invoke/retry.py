from __future__ import annotations

import os
import random
from dataclasses import dataclass, field


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_codes(name: str, default: set[str]) -> set[str]:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return set(default)
    extra = {part.strip() for part in raw.split(",") if part.strip()}
    return set(default) | extra


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_seconds: float = 5.0
    max_delay_seconds: float = 45.0
    jitter: bool = True
    retryable_error_codes: set[str] = field(
        default_factory=lambda: {
            "ServiceUnavailableException",
            "InternalServerException",
        }
    )

    @classmethod
    def from_env(cls) -> "RetryPolicy":
        default = cls()
        return cls(
            max_attempts=_env_int("BEDROCK_RETRY_MAX_ATTEMPTS", default.max_attempts),
            base_delay_seconds=_env_float("BEDROCK_RETRY_BASE_SECONDS", default.base_delay_seconds),
            max_delay_seconds=_env_float("BEDROCK_RETRY_MAX_SECONDS", default.max_delay_seconds),
            jitter=_env_bool("BEDROCK_RETRY_JITTER", default.jitter),
            retryable_error_codes=_env_codes(
                "BEDROCK_RETRYABLE_ERROR_CODES",
                default.retryable_error_codes,
            ),
        )

    def compute_sleep_seconds(self, attempt: int) -> float:
        base_delay = self.base_delay_seconds * (2 ** (attempt - 1))
        capped_delay = min(base_delay, self.max_delay_seconds)
        if not self.jitter:
            return capped_delay
        return capped_delay + random.uniform(0.0, 1.0)
