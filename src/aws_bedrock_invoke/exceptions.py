from __future__ import annotations


class BedrockInvokeError(Exception):
    pass


class BedrockResponseParseError(BedrockInvokeError):
    pass
