from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from aws_bedrock_invoke.retry import RetryPolicy


class RetryPolicyTests(unittest.TestCase):
    def test_from_env_uses_library_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            policy = RetryPolicy.from_env()

        self.assertEqual(4, policy.max_attempts)
        self.assertEqual(5.0, policy.base_delay_seconds)
        self.assertEqual(45.0, policy.max_delay_seconds)
        self.assertTrue(policy.jitter)
        self.assertEqual(
            {"ServiceUnavailableException", "InternalServerException"},
            policy.retryable_error_codes,
        )

    def test_from_env_adds_retryable_codes_instead_of_replacing(self) -> None:
        env = {
            "BEDROCK_RETRYABLE_ERROR_CODES": "ThrottlingException,ModelNotReadyException",
        }
        with patch.dict(os.environ, env, clear=True):
            policy = RetryPolicy.from_env()

        self.assertEqual(
            {
                "ServiceUnavailableException",
                "InternalServerException",
                "ThrottlingException",
                "ModelNotReadyException",
            },
            policy.retryable_error_codes,
        )

    def test_compute_sleep_seconds_without_jitter_is_progressive_and_capped(self) -> None:
        policy = RetryPolicy(
            max_attempts=4,
            base_delay_seconds=5.0,
            max_delay_seconds=12.0,
            jitter=False,
        )

        self.assertEqual(5.0, policy.compute_sleep_seconds(1))
        self.assertEqual(10.0, policy.compute_sleep_seconds(2))
        self.assertEqual(12.0, policy.compute_sleep_seconds(3))
        self.assertEqual(12.0, policy.compute_sleep_seconds(4))
