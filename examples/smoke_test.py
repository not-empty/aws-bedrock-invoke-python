"""Real Bedrock calls to check Claude Sonnet 4.5 and Claude Sonnet 5.5.

Each run makes 5 small, billed InvokeModel calls.

Usage (from the repository root):
    set -a; . ./.env; set +a
    PYTHONPATH=src python3 examples/smoke_test.py

Override the model IDs with SONNET_45_MODEL and SONNET_55_MODEL.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Callable

import boto3

from aws_bedrock_invoke import BedrockInvoker, InvokeResult, json_text_block

SONNET_45 = os.getenv("SONNET_45_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
SONNET_55 = os.getenv("SONNET_55_MODEL", "us.anthropic.claude-sonnet-5-5")

ANSWER_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "country": {"type": "string"},
        },
        "required": ["capital", "country"],
        "additionalProperties": False,
    },
}


def profile_arn_fmt() -> str:
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"arn:aws:bedrock:{{region}}:{account_id}:inference-profile/{{model}}"


def show(result: InvokeResult) -> None:
    envelope = result.response_envelope
    block_types = [block.get("type") for block in envelope.get("content", [])]
    print(f"  model_id:    {result.model_id}")
    print(f"  stop_reason: {envelope.get('stop_reason')}")
    print(f"  blocks:      {block_types}")
    print(f"  usage:       {result.usage}")
    print(f"  text:        {result.response_text[:300]!r}")
    if result.parsed_json is not None:
        print(f"  parsed_json: {result.parsed_json}")


def run(name: str, call: Callable[[], InvokeResult], *, expect_error: bool = False) -> bool:
    print(f"\n=== {name} ===")
    try:
        result = call()
    except Exception as error:  # noqa: BLE001 - report any failure
        if expect_error:
            print(f"  OK (expected error): {type(error).__name__}: {error}")
            return True
        print("  FAILED:")
        traceback.print_exc()
        return False
    if expect_error:
        print("  FAILED: expected an error, but the call succeeded")
        show(result)
        return False
    show(result)
    return True


def main() -> None:
    invoker = BedrockInvoker.from_env()
    arn_fmt = profile_arn_fmt()
    question: list[dict[str, Any]] = [
        json_text_block({"question": "What is the capital of Brazil?"}),
    ]

    cases: list[tuple[str, Callable[[], InvokeResult], bool]] = [
        (
            "Sonnet 4.5 | invoke_json called as in 1.0.0 (no new parameters)",
            lambda: invoker.invoke_json(
                model=SONNET_45,
                profile_arn_fmt=arn_fmt,
                system_prompt='Reply with JSON only: {"capital": ..., "country": ...}',
                user_content=question,
            ),
            False,
        ),
        (
            "Sonnet 4.5 | invoke_text called as in 1.0.0",
            lambda: invoker.invoke_text(
                model=SONNET_45,
                profile_arn_fmt=arn_fmt,
                system_prompt="Answer in one sentence.",
                user_content=question,
            ),
            False,
        ),
        (
            "Sonnet 5.5 | invoke_json with thinking + effort + format",
            lambda: invoker.invoke_json(
                model=SONNET_55,
                profile_arn_fmt=arn_fmt,
                system_prompt="Answer the question.",
                user_content=question,
                max_tokens=16000,
                temperature=None,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium", "format": ANSWER_SCHEMA},
            ),
            False,
        ),
        (
            "Sonnet 5.5 | invoke_text with low effort",
            lambda: invoker.invoke_text(
                model=SONNET_55,
                profile_arn_fmt=arn_fmt,
                system_prompt="Answer in one sentence.",
                user_content=question,
                max_tokens=16000,
                temperature=None,
                output_config={"effort": "low"},
            ),
            False,
        ),
        (
            "Sonnet 5.5 | default temperature (0.0) must return a 400 error",
            lambda: invoker.invoke_text(
                model=SONNET_55,
                profile_arn_fmt=arn_fmt,
                system_prompt="Answer in one sentence.",
                user_content=question,
            ),
            True,
        ),
    ]

    results = [run(name, call, expect_error=expect) for name, call, expect in cases]
    print(f"\n{sum(results)}/{len(results)} cases passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
