from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from aws_bedrock_invoke.config import BedrockPromptCacheConfig
from aws_bedrock_invoke.content import json_text_block, prompt_cache_to_cache_control, text_block


class PromptCacheConfigTests(unittest.TestCase):
    def test_prompt_cache_config_from_env(self) -> None:
        env = {
            "BEDROCK_PROMPT_CACHE_ENABLED": "true",
            "BEDROCK_PROMPT_CACHE_TTL": "5m",
        }
        with patch.dict(os.environ, env, clear=True):
            config = BedrockPromptCacheConfig.from_env()

        self.assertTrue(config.enabled)
        self.assertEqual("5m", config.ttl)

    def test_prompt_cache_to_cache_control_returns_none_when_disabled(self) -> None:
        config = BedrockPromptCacheConfig(enabled=False, ttl="5m")
        self.assertIsNone(prompt_cache_to_cache_control(config))

    def test_json_text_block_includes_cache_control_when_enabled(self) -> None:
        config = BedrockPromptCacheConfig(enabled=True, ttl="5m")
        block = json_text_block({"hello": "world"}, bedrock_prompt_cache=config)

        self.assertEqual("text", block["type"])
        self.assertEqual({"hello": "world"}, json.loads(block["text"]))
        self.assertEqual({"type": "ephemeral", "ttl": "5m"}, block["cache_control"])

    def test_json_text_block_omits_cache_control_when_not_enabled(self) -> None:
        block = json_text_block({"hello": "world"})

        self.assertEqual("text", block["type"])
        self.assertEqual({"hello": "world"}, json.loads(block["text"]))
        self.assertNotIn("cache_control", block)

    def test_text_block_includes_cache_control_when_enabled(self) -> None:
        config = BedrockPromptCacheConfig(enabled=True, ttl="5m")
        block = text_block("hello", bedrock_prompt_cache=config)

        self.assertEqual("text", block["type"])
        self.assertEqual("hello", block["text"])
        self.assertEqual({"type": "ephemeral", "ttl": "5m"}, block["cache_control"])
