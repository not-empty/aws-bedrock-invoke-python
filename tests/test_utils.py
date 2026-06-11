from __future__ import annotations

import unittest

from aws_bedrock_invoke.utils import resolve_model_id


class ResolveModelIdTests(unittest.TestCase):
    def test_resolve_model_id_returns_full_arn_unchanged(self) -> None:
        model = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/test"
        resolved = resolve_model_id(model, region="us-east-1")
        self.assertEqual(model, resolved)

    def test_resolve_model_id_formats_shorthand_with_profile_arn_fmt(self) -> None:
        resolved = resolve_model_id(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region="us-east-1",
            profile_arn_fmt="arn:aws:bedrock:{region}:acct:inference-profile/{model}",
        )
        self.assertEqual(
            "arn:aws:bedrock:us-east-1:acct:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            resolved,
        )

    def test_resolve_model_id_requires_profile_arn_fmt_for_shorthand(self) -> None:
        with self.assertRaises(ValueError):
            resolve_model_id("us.anthropic.claude-sonnet-4-5-20250929-v1:0", region="us-east-1")
