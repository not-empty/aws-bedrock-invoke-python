# AWS Bedrock Invoke Python

AWS Bedrock Invoke Python is a small library for invoking Claude 4.x models on
AWS Bedrock with one consistent request path.

It focuses only on the transport layer:

- boto3 client creation
- optional client injection
- Bedrock model ID / inference-profile resolution
- Bedrock prompt-cache helpers
- manual retry with env-driven configuration
- response text extraction
- fenced JSON cleanup and parsing

It intentionally does not own business rules such as batching strategies,
workflow orchestration, application result caching, or prompt authoring.

## Installation

```bash
pip install aws-bedrock-invoke
```

## Scope

This library is intentionally narrow:

- Claude 4.x on AWS Bedrock only
- manual retry managed by the library
- Bedrock prompt-cache support
- text and JSON invoke helpers

This library intentionally does not include:

- business-specific batching/windowing
- application result caching
- Slack notifications or persistence
- prompt engineering or workflow orchestration

## Quick Start

### `invoke_json()`

```python
from aws_bedrock_invoke import BedrockInvoker, json_text_block

invoker = BedrockInvoker.from_env()

result = invoker.invoke_json(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    profile_arn_fmt="arn:aws:bedrock:{region}:354173731728:inference-profile/{model}",
    system_prompt="Reply with strict JSON.",
    user_content=[
        json_text_block({"question": "hello"}),
    ],
)

print(result.parsed_json)
print(result.response_text)
```

### `invoke_text()`

```python
from aws_bedrock_invoke import BedrockInvoker, text_block

invoker = BedrockInvoker.from_env()

result = invoker.invoke_text(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    profile_arn_fmt="arn:aws:bedrock:{region}:354173731728:inference-profile/{model}",
    system_prompt="Answer briefly.",
    user_content=[
        text_block("What is a lien waiver?"),
    ],
)

print(result.response_text)
```

## Client Injection

Client injection is optional.

If you do not provide a client, `BedrockInvoker` builds one from
`BedrockClientConfig`.

If you already have a prepared boto Bedrock client, you can inject it directly.
This is especially useful for adapters and tests.

```python
import boto3

from aws_bedrock_invoke import BedrockClientConfig, BedrockInvoker, text_block

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id="...",
    aws_secret_access_key="...",
)

invoker = BedrockInvoker(
    client_config=BedrockClientConfig(
        region="us-east-1",
        aws_access_key_id="...",
        aws_secret_access_key="...",
    ),
    client=client,
)

result = invoker.invoke_text(
    model="arn:aws:bedrock:us-east-1:123456789012:inference-profile/example",
    system_prompt="Answer briefly.",
    user_content=[text_block("hello")],
)
```

## Environment Variables

### AWS / Bedrock

- `AWS_REGION`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

### Retry

- `BEDROCK_RETRY_MAX_ATTEMPTS`
- `BEDROCK_RETRY_BASE_SECONDS`
- `BEDROCK_RETRY_MAX_SECONDS`
- `BEDROCK_RETRY_JITTER`
- `BEDROCK_RETRYABLE_ERROR_CODES`

The Bedrock SDK retry behavior is effectively disabled in the client, and the
library uses its own manual progressive retry loop instead.

Default retryable error codes:

- `ServiceUnavailableException`
- `InternalServerException`

`BEDROCK_RETRYABLE_ERROR_CODES` is additive. If you set it, the provided codes
are added on top of those defaults rather than replacing them.

Default retry behavior:

- `BEDROCK_RETRY_MAX_ATTEMPTS=4`
- `BEDROCK_RETRY_BASE_SECONDS=5`
- `BEDROCK_RETRY_MAX_SECONDS=45`
- `BEDROCK_RETRY_JITTER=true`

`on_retry` receives a typed `RetryEvent` with:

- `attempt`
- `max_attempts`
- `error_code`
- `error`
- `sleep_seconds`
- `model_id`
- `elapsed_seconds`

### Prompt Cache

- `BEDROCK_PROMPT_CACHE_ENABLED`
- `BEDROCK_PROMPT_CACHE_TTL`

## Public API

- `BedrockInvoker`
- `BedrockClientConfig`
- `RetryPolicy`
- `BedrockPromptCacheConfig`
- `InvokeResult`
- `RetryEvent`
- `json_text_block`
- `text_block`
- `resolve_model_id`

## Model Resolution

If `model` is a full ARN, the library uses it directly.

If `model` is a shorthand like `us.anthropic...`, you must provide
`profile_arn_fmt` on the invoke call.

## Testing

Run the test suite locally with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Examples

Runnable examples live in [examples](/home/disarli/Projects/aws-bedrock-invoke-python/examples):

```bash
PYTHONPATH=src python3 examples/invoke_json_example.py
PYTHONPATH=src python3 examples/invoke_text_example.py
```

## Consumer Integration

This repo includes library-level tests and Bedrock smoke examples.

Consumer repositories should still add their own integration smoke that uses
their real prompt builders and payload-shaping logic through this library.
