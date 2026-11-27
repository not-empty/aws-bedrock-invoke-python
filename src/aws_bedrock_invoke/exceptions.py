from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .invoker import InvokeResult


class BedrockInvokeError(Exception):
    pass


class BedrockResponseParseError(BedrockInvokeError):
    def __init__(self, *args: object, result: InvokeResult | None = None) -> None:
        super().__init__(*args)
        self.result = result


class BedrockRefusalError(BedrockResponseParseError):
    """The model refused the request (``stop_reason == "refusal"``), so no JSON was returned."""


class BedrockMaxTokensError(BedrockResponseParseError):
    """The response hit ``max_tokens`` (``stop_reason == "max_tokens"``) and the JSON is incomplete."""
