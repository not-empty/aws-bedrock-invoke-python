from __future__ import annotations

import unittest

from aws_bedrock_invoke.exceptions import BedrockResponseParseError
from aws_bedrock_invoke.parsing import cleanup_json_text, extract_response_text, parse_response_json


class ParsingTests(unittest.TestCase):
    def test_extract_response_text_from_content_blocks(self) -> None:
        envelope = {
            "content": [
                {"type": "text", "text": '{"hello":'},
                {"type": "text", "text": '"world"}'},
            ]
        }
        self.assertEqual('{"hello":"world"}', extract_response_text(envelope))

    def test_cleanup_json_text_removes_fences(self) -> None:
        text = '```json\n{"hello":"world"}\n```'
        self.assertEqual('{"hello":"world"}', cleanup_json_text(text))

    def test_parse_response_json_returns_object(self) -> None:
        parsed = parse_response_json('```json\n{"hello":"world"}\n```')
        self.assertEqual({"hello": "world"}, parsed)

    def test_parse_response_json_rejects_non_object(self) -> None:
        with self.assertRaises(BedrockResponseParseError):
            parse_response_json('["hello"]')
