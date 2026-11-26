# Changelog

## 1.1.0

- Added optional `thinking` and `output_config` parameters to `invoke_text()` and
  `invoke_json()`
- Added `stop_reason` to `InvokeResult` (defaults to `None`)
- `invoke_json()` now raises `BedrockRefusalError` or `BedrockMaxTokensError` when
  the JSON cannot be parsed because the model refused or hit `max_tokens`; both
  subclass `BedrockResponseParseError`
- `BedrockResponseParseError` raised by `invoke_json()` now carries the
  `InvokeResult` in `error.result`, and its message includes the `stop_reason`
  when present
- Exceptions are now exported from the package root
- `temperature` now accepts `None`, which omits it from the request (required by
  Claude Sonnet 5.5)
- No changes for existing callers: `temperature` still defaults to `0.0` and the
  default request body has the same content

## 1.0.0

- Added a focused Claude 4.x AWS Bedrock invocation library
- Added manual progressive retry with env-driven configuration
- Added additive default retryable Bedrock error-code handling
- Added optional Bedrock prompt-cache helpers
- Added JSON and text invocation helpers
- Added optional public client injection for adapters and tests
- Added unit tests and real Bedrock smoke coverage
