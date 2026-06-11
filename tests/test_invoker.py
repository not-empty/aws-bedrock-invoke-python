from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from botocore.exceptions import ClientError

from aws_bedrock_invoke.config import BedrockClientConfig
from aws_bedrock_invoke.invoker import BedrockInvoker
from aws_bedrock_invoke.retry import RetryPolicy


def _client_error(code: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": code}},
        operation_name="InvokeModel",
    )


class _FakeBody:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.closed = False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def close(self) -> None:
        self.closed = True


class _FlakyClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.bodies: list[_FakeBody] = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        current = self.responses.pop(0)
        if isinstance(current, Exception):
            raise current
        body = _FakeBody(current)
        self.bodies.append(body)
        return {"body": body}


class BedrockInvokerTests(unittest.TestCase):
    def test_invoke_json_retries_retryable_client_error_and_calls_hook(self) -> None:
        fake_client = _FlakyClient(
            [
                _client_error("ServiceUnavailableException"),
                {"content": [{"type": "text", "text": '{"ok":true}'}], "usage": {"input_tokens": 10}},
            ]
        )
        retry_events: list[dict] = []
        policy = RetryPolicy(max_attempts=4, base_delay_seconds=5.0, max_delay_seconds=45.0, jitter=False)

        with patch("aws_bedrock_invoke.invoker.boto3.client", return_value=fake_client), patch(
            "aws_bedrock_invoke.invoker.time.sleep"
        ) as sleep_mock:
            invoker = BedrockInvoker(
                client_config=BedrockClientConfig(
                    region="us-east-1",
                    aws_access_key_id="key",
                    aws_secret_access_key="secret",
                ),
                retry_policy=policy,
                on_retry=retry_events.append,
            )
            result = invoker.invoke_json(
                model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                profile_arn_fmt="arn:aws:bedrock:{region}:acct:inference-profile/{model}",
                system_prompt="Reply with JSON.",
                user_content=[{"type": "text", "text": '{"hello":"world"}'}],
            )

        self.assertEqual({"ok": True}, result.parsed_json)
        self.assertEqual(2, len(fake_client.calls))
        self.assertEqual(1, len(retry_events))
        self.assertEqual("ServiceUnavailableException", retry_events[0].error_code)
        self.assertTrue(fake_client.bodies[-1].closed)
        sleep_mock.assert_called_once_with(5.0)

    def test_invoke_json_raises_non_retryable_client_error(self) -> None:
        fake_client = _FlakyClient([_client_error("ValidationException")])
        policy = RetryPolicy(max_attempts=4, base_delay_seconds=5.0, max_delay_seconds=45.0, jitter=False)

        with patch("aws_bedrock_invoke.invoker.boto3.client", return_value=fake_client), patch(
            "aws_bedrock_invoke.invoker.time.sleep"
        ) as sleep_mock:
            invoker = BedrockInvoker(
                client_config=BedrockClientConfig(
                    region="us-east-1",
                    aws_access_key_id="key",
                    aws_secret_access_key="secret",
                ),
                retry_policy=policy,
            )
            with self.assertRaises(ClientError):
                invoker.invoke_json(
                    model="arn:aws:bedrock:us-east-1:acct:inference-profile/test",
                    system_prompt="Reply with JSON.",
                    user_content=[{"type": "text", "text": '{"hello":"world"}'}],
                )

        sleep_mock.assert_not_called()

    def test_invoke_json_requires_profile_arn_fmt_for_shorthand_model(self) -> None:
        fake_client = _FlakyClient([])

        with patch("aws_bedrock_invoke.invoker.boto3.client", return_value=fake_client):
            invoker = BedrockInvoker(
                client_config=BedrockClientConfig(
                    region="us-east-1",
                    aws_access_key_id="key",
                    aws_secret_access_key="secret",
                ),
                retry_policy=RetryPolicy(jitter=False),
            )
            with self.assertRaises(ValueError):
                invoker.invoke_json(
                    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    system_prompt="Reply with JSON.",
                    user_content=[{"type": "text", "text": '{"hello":"world"}'}],
                )

    def test_invoke_json_accepts_full_arn_without_profile_arn_fmt(self) -> None:
        fake_client = _FlakyClient(
            [{"content": [{"type": "text", "text": '{"ok":true}'}], "usage": {}}]
        )

        with patch("aws_bedrock_invoke.invoker.boto3.client", return_value=fake_client):
            invoker = BedrockInvoker(
                client_config=BedrockClientConfig(
                    region="us-east-1",
                    aws_access_key_id="key",
                    aws_secret_access_key="secret",
                ),
                retry_policy=RetryPolicy(jitter=False),
            )
            result = invoker.invoke_json(
                model="arn:aws:bedrock:us-east-1:acct:inference-profile/test",
                system_prompt="Reply with JSON.",
                user_content=[{"type": "text", "text": '{"hello":"world"}'}],
            )

        self.assertEqual({"ok": True}, result.parsed_json)

    def test_invoker_uses_injected_client_without_building_boto_client(self) -> None:
        fake_client = _FlakyClient(
            [{"content": [{"type": "text", "text": '{"ok":true}'}], "usage": {}}]
        )

        with patch("aws_bedrock_invoke.invoker.boto3.client") as boto_client_mock:
            invoker = BedrockInvoker(
                client_config=BedrockClientConfig(
                    region="us-east-1",
                    aws_access_key_id="key",
                    aws_secret_access_key="secret",
                ),
                client=fake_client,
                retry_policy=RetryPolicy(jitter=False),
            )
            result = invoker.invoke_json(
                model="arn:aws:bedrock:us-east-1:acct:inference-profile/test",
                system_prompt="Reply with JSON.",
                user_content=[{"type": "text", "text": '{"hello":"world"}'}],
            )

        boto_client_mock.assert_not_called()
        self.assertEqual({"ok": True}, result.parsed_json)

    def test_invoke_text_returns_text_without_json_parsing(self) -> None:
        fake_client = _FlakyClient(
            [{"content": [{"type": "text", "text": "plain text response"}], "usage": {"output_tokens": 3}}]
        )

        with patch("aws_bedrock_invoke.invoker.boto3.client", return_value=fake_client):
            invoker = BedrockInvoker(
                client_config=BedrockClientConfig(
                    region="us-east-1",
                    aws_access_key_id="key",
                    aws_secret_access_key="secret",
                ),
                retry_policy=RetryPolicy(jitter=False),
            )
            result = invoker.invoke_text(
                model="arn:aws:bedrock:us-east-1:acct:inference-profile/test",
                system_prompt="Reply in plain text.",
                user_content=[{"type": "text", "text": "hello"}],
            )

        self.assertEqual("plain text response", result.response_text)
        self.assertIsNone(result.parsed_json)
