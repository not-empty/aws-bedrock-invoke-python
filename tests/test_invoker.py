from __future__ import annotations

import json
import os
import unittest
from unittest.mock import call, patch

from botocore.exceptions import ClientError

from aws_bedrock_invoke.config import BedrockClientConfig
from aws_bedrock_invoke.exceptions import (
    BedrockMaxTokensError,
    BedrockRefusalError,
    BedrockResponseParseError,
)
from aws_bedrock_invoke.invoker import BedrockInvoker, InvokeResult, RetryEvent
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


class BedrockInvokerRequestBodyTests(unittest.TestCase):
    METHODS = ("invoke_text", "invoke_json")

    def _sent_body(self, method: str, **kwargs) -> dict:
        fake_client = _FlakyClient(
            [{"content": [{"type": "text", "text": '{"ok":true}'}], "usage": {}}]
        )
        invoker = BedrockInvoker(
            client_config=BedrockClientConfig(region="us-east-1"),
            client=fake_client,
            retry_policy=RetryPolicy(jitter=False),
        )
        getattr(invoker, method)(
            model="arn:aws:bedrock:us-east-1:acct:inference-profile/test",
            system_prompt="Reply with JSON.",
            user_content=[{"type": "text", "text": "hello"}],
            **kwargs,
        )
        return json.loads(fake_client.calls[0]["body"])

    def test_default_body_is_unchanged_from_1_0_0(self) -> None:
        expected = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "temperature": 0.0,
            "system": "Reply with JSON.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
        }
        for method in self.METHODS:
            with self.subTest(method=method):
                self.assertEqual(expected, self._sent_body(method))

    def test_explicit_temperature_is_sent(self) -> None:
        for method in self.METHODS:
            with self.subTest(method=method):
                self.assertEqual(0.5, self._sent_body(method, temperature=0.5)["temperature"])

    def test_temperature_none_is_omitted(self) -> None:
        for method in self.METHODS:
            with self.subTest(method=method):
                self.assertNotIn("temperature", self._sent_body(method, temperature=None))

    def test_thinking_and_output_config_are_sent_as_given(self) -> None:
        thinking = {"type": "adaptive"}
        output_config = {
            "effort": "medium",
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        }
        for method in self.METHODS:
            with self.subTest(method=method):
                body = self._sent_body(method, thinking=thinking, output_config=output_config)
                self.assertEqual(thinking, body["thinking"])
                self.assertEqual(output_config, body["output_config"])

    def test_thinking_blocks_are_not_part_of_response_text(self) -> None:
        fake_client = _FlakyClient(
            [
                {
                    "content": [
                        {"type": "thinking", "thinking": "{not json}", "signature": "sig"},
                        {"type": "text", "text": '{"ok":true}'},
                    ],
                    "usage": {},
                }
            ]
        )
        invoker = BedrockInvoker(
            client_config=BedrockClientConfig(region="us-east-1"),
            client=fake_client,
            retry_policy=RetryPolicy(jitter=False),
        )
        result = invoker.invoke_json(
            model="arn:aws:bedrock:us-east-1:acct:inference-profile/test",
            system_prompt="Reply with JSON.",
            user_content=[{"type": "text", "text": "hello"}],
        )
        self.assertEqual('{"ok":true}', result.response_text)
        self.assertEqual({"ok": True}, result.parsed_json)


class BedrockInvokerStopReasonTests(unittest.TestCase):
    def _invoke(self, method: str, envelope: dict):
        invoker = BedrockInvoker(
            client_config=BedrockClientConfig(region="us-east-1"),
            client=_FlakyClient([envelope]),
            retry_policy=RetryPolicy(jitter=False),
        )
        return getattr(invoker, method)(
            model="arn:aws:bedrock:us-east-1:acct:inference-profile/test",
            system_prompt="Reply with JSON.",
            user_content=[{"type": "text", "text": "hello"}],
        )

    def test_stop_reason_is_exposed_on_result(self) -> None:
        envelope = {
            "content": [{"type": "text", "text": '{"ok":true}'}],
            "stop_reason": "end_turn",
            "usage": {},
        }
        for method in ("invoke_text", "invoke_json"):
            with self.subTest(method=method):
                self.assertEqual("end_turn", self._invoke(method, envelope).stop_reason)

    def test_stop_reason_is_none_when_missing(self) -> None:
        envelope = {"content": [{"type": "text", "text": '{"ok":true}'}], "usage": {}}
        self.assertIsNone(self._invoke("invoke_json", envelope).stop_reason)

    def test_invoke_result_can_still_be_built_without_stop_reason(self) -> None:
        result = InvokeResult("model", "text", None, {}, {})
        self.assertIsNone(result.stop_reason)

    def test_invoke_text_does_not_raise_on_refusal_or_max_tokens(self) -> None:
        for stop_reason in ("refusal", "max_tokens"):
            with self.subTest(stop_reason=stop_reason):
                envelope = {
                    "content": [{"type": "text", "text": "partial"}],
                    "stop_reason": stop_reason,
                    "usage": {},
                }
                result = self._invoke("invoke_text", envelope)
                self.assertEqual("partial", result.response_text)
                self.assertEqual(stop_reason, result.stop_reason)

    def test_invoke_json_raises_refusal_error_when_json_is_invalid(self) -> None:
        envelope = {"content": [], "stop_reason": "refusal", "usage": {}}
        with self.assertRaises(BedrockRefusalError) as ctx:
            self._invoke("invoke_json", envelope)
        self.assertIsInstance(ctx.exception, BedrockResponseParseError)
        self.assertEqual("refusal", ctx.exception.result.stop_reason)
        self.assertEqual(envelope, ctx.exception.result.response_envelope)

    def test_invoke_json_raises_max_tokens_error_when_json_is_truncated(self) -> None:
        envelope = {
            "content": [{"type": "text", "text": '{"answer": "cut'}],
            "stop_reason": "max_tokens",
            "usage": {},
        }
        with self.assertRaises(BedrockMaxTokensError) as ctx:
            self._invoke("invoke_json", envelope)
        self.assertIsInstance(ctx.exception, BedrockResponseParseError)
        self.assertEqual('{"answer": "cut', ctx.exception.result.response_text)

    def test_invoke_json_generic_parse_error_reports_other_stop_reason(self) -> None:
        envelope = {
            "content": [{"type": "text", "text": "not json"}],
            "stop_reason": "end_turn",
            "usage": {},
        }
        with self.assertRaises(BedrockResponseParseError) as ctx:
            self._invoke("invoke_json", envelope)
        self.assertIs(BedrockResponseParseError, type(ctx.exception))
        self.assertIn("stop_reason='end_turn'", str(ctx.exception))
        self.assertEqual("end_turn", ctx.exception.result.stop_reason)

    def test_invoke_json_returns_valid_json_even_with_max_tokens(self) -> None:
        envelope = {
            "content": [{"type": "text", "text": '{"ok":true}'}],
            "stop_reason": "max_tokens",
            "usage": {},
        }
        result = self._invoke("invoke_json", envelope)
        self.assertEqual({"ok": True}, result.parsed_json)
        self.assertEqual("max_tokens", result.stop_reason)

    def test_invoke_json_parse_error_message_unchanged_without_stop_reason(self) -> None:
        envelope = {"content": [{"type": "text", "text": "not json"}], "usage": {}}
        with self.assertRaises(BedrockResponseParseError) as ctx:
            self._invoke("invoke_json", envelope)
        self.assertTrue(str(ctx.exception).startswith("Failed to parse JSON response:"))
        self.assertNotIn("stop_reason", str(ctx.exception))
        self.assertIsNone(ctx.exception.result.stop_reason)

    def test_parse_error_can_still_be_raised_with_message_only(self) -> None:
        error = BedrockResponseParseError("boom")
        self.assertEqual("boom", str(error))
        self.assertIsNone(error.result)

    def test_parse_error_accepts_same_arguments_as_1_0_0(self) -> None:
        for args in ((), ("message",), ("a", "b")):
            with self.subTest(args=args):
                error = BedrockResponseParseError(*args)
                self.assertEqual(args, error.args)
                self.assertIsNone(error.result)

    def test_exceptions_are_exported_from_package(self) -> None:
        import aws_bedrock_invoke

        for name in (
            "BedrockInvokeError",
            "BedrockResponseParseError",
            "BedrockRefusalError",
            "BedrockMaxTokensError",
        ):
            with self.subTest(name=name):
                self.assertIn(name, aws_bedrock_invoke.__all__)
                self.assertTrue(hasattr(aws_bedrock_invoke, name))


class BedrockInvokerRetryAcrossModelsTests(unittest.TestCase):
    """The retry loop must behave the same for Claude 4.x and Claude 5.x requests."""

    MODEL_PARAMS = {
        "claude-4x": {},
        "claude-5x": {
            "max_tokens": 16000,
            "temperature": None,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        },
    }
    METHODS = ("invoke_text", "invoke_json")
    OK_ENVELOPE = {
        "content": [{"type": "text", "text": '{"ok":true}'}],
        "stop_reason": "end_turn",
        "usage": {},
    }

    def _run(self, responses, method, params, policy=None):
        fake_client = _FlakyClient(responses)
        events: list[RetryEvent] = []
        invoker = BedrockInvoker(
            client_config=BedrockClientConfig(region="us-east-1"),
            client=fake_client,
            retry_policy=policy
            or RetryPolicy(max_attempts=3, base_delay_seconds=5.0, max_delay_seconds=45.0, jitter=False),
            on_retry=events.append,
        )
        with patch("aws_bedrock_invoke.invoker.time.sleep") as sleep_mock:
            try:
                result = getattr(invoker, method)(
                    model="arn:aws:bedrock:us-east-1:acct:inference-profile/test",
                    system_prompt="Reply with JSON.",
                    user_content=[{"type": "text", "text": "hello"}],
                    **params,
                )
                error = None
            except ClientError as exc:
                result, error = None, exc
        return fake_client, events, sleep_mock, result, error

    def _cases(self):
        for name, params in self.MODEL_PARAMS.items():
            for method in self.METHODS:
                yield name, params, method

    def test_retryable_error_is_retried_until_success(self) -> None:
        for name, params, method in self._cases():
            with self.subTest(model=name, method=method):
                client, events, sleep_mock, result, error = self._run(
                    [
                        _client_error("ServiceUnavailableException"),
                        _client_error("InternalServerException"),
                        _client_error("ServiceUnavailableException"),
                        self.OK_ENVELOPE,
                    ],
                    method,
                    params,
                    policy=RetryPolicy(
                        max_attempts=4, base_delay_seconds=5.0, max_delay_seconds=45.0, jitter=False
                    ),
                )
                self.assertIsNone(error)
                self.assertEqual('{"ok":true}', result.response_text)
                self.assertEqual(4, len(client.calls))
                self.assertEqual([1, 2, 3], [event.attempt for event in events])
                self.assertEqual(
                    [
                        "ServiceUnavailableException",
                        "InternalServerException",
                        "ServiceUnavailableException",
                    ],
                    [event.error_code for event in events],
                )
                self.assertEqual([call(5.0), call(10.0), call(20.0)], sleep_mock.call_args_list)
                bodies = [c["body"] for c in client.calls]
                self.assertEqual(1, len(set(bodies)), "every attempt must send the same body")

    def test_non_retryable_error_is_raised_immediately(self) -> None:
        for name, params, method in self._cases():
            with self.subTest(model=name, method=method):
                client, events, sleep_mock, _, error = self._run(
                    [_client_error("ValidationException")],
                    method,
                    params,
                )
                self.assertEqual("ValidationException", error.response["Error"]["Code"])
                self.assertEqual(1, len(client.calls))
                self.assertEqual([], events)
                sleep_mock.assert_not_called()

    def test_last_error_is_raised_when_attempts_are_exhausted(self) -> None:
        for name, params, method in self._cases():
            with self.subTest(model=name, method=method):
                client, events, sleep_mock, _, error = self._run(
                    [_client_error("ServiceUnavailableException")] * 3,
                    method,
                    params,
                )
                self.assertEqual("ServiceUnavailableException", error.response["Error"]["Code"])
                self.assertEqual(3, len(client.calls))
                self.assertEqual(2, len(events))
                self.assertEqual(2, sleep_mock.call_count)

    def test_error_codes_added_through_env_are_retried(self) -> None:
        with patch.dict(os.environ, {"BEDROCK_RETRYABLE_ERROR_CODES": "ThrottlingException"}, clear=True):
            policy = RetryPolicy.from_env()
        for name, params, method in self._cases():
            with self.subTest(model=name, method=method):
                client, events, _, result, error = self._run(
                    [
                        _client_error("ThrottlingException"),
                        _client_error("ServiceUnavailableException"),
                        self.OK_ENVELOPE,
                    ],
                    method,
                    params,
                    policy=policy,
                )
                self.assertIsNone(error)
                self.assertEqual(3, len(client.calls))
                self.assertEqual(
                    ["ThrottlingException", "ServiceUnavailableException"],
                    [event.error_code for event in events],
                )

    def test_parse_errors_are_not_retried(self) -> None:
        for name, params in self.MODEL_PARAMS.items():
            with self.subTest(model=name):
                client = _FlakyClient(
                    [{"content": [], "stop_reason": "refusal", "usage": {}}, self.OK_ENVELOPE]
                )
                invoker = BedrockInvoker(
                    client_config=BedrockClientConfig(region="us-east-1"),
                    client=client,
                    retry_policy=RetryPolicy(jitter=False),
                )
                with self.assertRaises(BedrockResponseParseError):
                    invoker.invoke_json(
                        model="arn:aws:bedrock:us-east-1:acct:inference-profile/test",
                        system_prompt="Reply with JSON.",
                        user_content=[{"type": "text", "text": "hello"}],
                        **params,
                    )
                self.assertEqual(1, len(client.calls))
