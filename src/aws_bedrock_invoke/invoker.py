from __future__ import annotations

import json
import time
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import BedrockClientConfig
from .parsing import extract_response_text, parse_response_json
from .retry import RetryPolicy
from .utils import resolve_model_id

DEFAULT_ANTHROPIC_VERSION = "bedrock-2023-05-31"


@dataclass(slots=True)
class InvokeResult:
    model_id: str
    response_text: str
    parsed_json: dict[str, Any] | None
    usage: dict[str, Any]
    response_envelope: dict[str, Any]


@dataclass(slots=True)
class RetryEvent:
    attempt: int
    max_attempts: int
    error_code: str
    error: ClientError
    sleep_seconds: float
    model_id: str
    elapsed_seconds: float


class BedrockInvoker:
    def __init__(
        self,
        *,
        client_config: BedrockClientConfig,
        client: Any | None = None,
        retry_policy: RetryPolicy | None = None,
        on_retry: Callable[[RetryEvent], None] | None = None,
    ) -> None:
        self.client_config = client_config
        self.retry_policy = retry_policy or RetryPolicy.from_env()
        self.on_retry = on_retry
        self._client = client if client is not None else self._build_client()

    @classmethod
    def from_env(
        cls,
        *,
        client: Any | None = None,
        on_retry: Callable[[RetryEvent], None] | None = None,
    ) -> "BedrockInvoker":
        return cls(
            client_config=BedrockClientConfig.from_env(),
            client=client,
            retry_policy=RetryPolicy.from_env(),
            on_retry=on_retry,
        )

    def _build_client(self) -> Any:
        retries = {"total_max_attempts": 1, "mode": "standard"}
        config_kwargs: dict[str, Any] = {
            "connect_timeout": self.client_config.connect_timeout,
            "read_timeout": self.client_config.read_timeout,
            "retries": retries,
        }
        if self.client_config.max_pool_connections is not None:
            config_kwargs["max_pool_connections"] = self.client_config.max_pool_connections

        return boto3.client(
            "bedrock-runtime",
            region_name=self.client_config.region,
            aws_access_key_id=self.client_config.aws_access_key_id,
            aws_secret_access_key=self.client_config.aws_secret_access_key,
            config=Config(**config_kwargs),
        )

    def _build_body(
        self,
        *,
        system_prompt: str | Sequence[dict[str, Any]],
        user_content: Sequence[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        anthropic_version: str,
    ) -> dict[str, Any]:
        return {
            "anthropic_version": anthropic_version,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": list(user_content),
                }
            ],
        }

    def _get_error_code(self, error: ClientError) -> str:
        return str((error.response.get("Error") or {}).get("Code", "") or "")

    def _invoke_with_retry(
        self,
        *,
        model_id: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        response = None
        started_at = time.time()
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                response = self._client.invoke_model(
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                    modelId=model_id,
                )
                break
            except ClientError as error:
                error_code = self._get_error_code(error)
                if (
                    error_code not in self.retry_policy.retryable_error_codes
                    or attempt >= self.retry_policy.max_attempts
                ):
                    raise

                sleep_seconds = self.retry_policy.compute_sleep_seconds(attempt)
                if self.on_retry is not None:
                    self.on_retry(
                        RetryEvent(
                            attempt=attempt,
                            max_attempts=self.retry_policy.max_attempts,
                            error_code=error_code,
                            error=error,
                            sleep_seconds=sleep_seconds,
                            model_id=model_id,
                            elapsed_seconds=time.time() - started_at,
                        )
                    )
                time.sleep(sleep_seconds)

        if response is None:
            raise RuntimeError("Bedrock invoke_model returned no response.")

        with closing(response["body"]) as body_stream:
            return json.loads(body_stream.read().decode("utf-8"))

    def invoke_text(
        self,
        *,
        model: str,
        system_prompt: str | Sequence[dict[str, Any]],
        user_content: Sequence[dict[str, Any]],
        profile_arn_fmt: str | None = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
    ) -> InvokeResult:
        model_id = resolve_model_id(
            model,
            region=self.client_config.region,
            profile_arn_fmt=profile_arn_fmt,
        )
        body = self._build_body(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=temperature,
            anthropic_version=anthropic_version,
        )
        envelope = self._invoke_with_retry(model_id=model_id, body=body)
        text = extract_response_text(envelope)
        usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
        return InvokeResult(
            model_id=model_id,
            response_text=text,
            parsed_json=None,
            usage=usage,
            response_envelope=envelope,
        )

    def invoke_json(
        self,
        *,
        model: str,
        system_prompt: str | Sequence[dict[str, Any]],
        user_content: Sequence[dict[str, Any]],
        profile_arn_fmt: str | None = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
    ) -> InvokeResult:
        result = self.invoke_text(
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            profile_arn_fmt=profile_arn_fmt,
            max_tokens=max_tokens,
            temperature=temperature,
            anthropic_version=anthropic_version,
        )
        return InvokeResult(
            model_id=result.model_id,
            response_text=result.response_text,
            parsed_json=parse_response_json(result.response_text),
            usage=result.usage,
            response_envelope=result.response_envelope,
        )
