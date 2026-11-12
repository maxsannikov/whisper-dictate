import importlib.util
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cloudflare_asr
import multi_provider
import provider_quota


class ProviderDailyQuotaTests(unittest.TestCase):
    def test_persisted_block_expires_at_utc_midnight(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "daily-limit.json"
            now = datetime(2026, 10, 23, 21, 30, tzinfo=timezone.utc)
            result = provider_quota.mark_daily_limit("openrouter", "daily_quota_exhausted", state, now)
            self.assertEqual(result["until_utc"], "2026-10-24T00:00:00+00:00")
            self.assertEqual(provider_quota.daily_limit_state(state, now)["provider"], "openrouter")
            self.assertEqual(provider_quota.daily_only_routes(["openrouter", "cloudflare", "local"], state, now), ["cloudflare", "local"])
            provider_quota.mark_daily_limit("cloudflare", "daily_quota_exhausted", state, now)
            self.assertEqual(provider_quota.daily_only_routes(["openrouter", "cloudflare", "local"], state, now), ["local"])
            after_reset = datetime(2026, 10, 24, 0, 0, tzinfo=timezone.utc)
            self.assertIsNone(provider_quota.daily_limit_state(state, after_reset))
            self.assertEqual(provider_quota.daily_only_routes(["cloudflare", "openrouter", "local"], state, after_reset),
                             ["cloudflare", "openrouter", "local"])
            self.assertEqual(state.stat().st_mode & 0o777, 0o600)

    def test_one_provider_block_preserves_the_other_provider_until_utc_midnight(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "daily-limit.json"
            now = datetime(2026, 10, 23, 21, 30, tzinfo=timezone.utc)
            provider_quota.mark_daily_limit("cloudflare", "daily_quota_exhausted", state, now)
            self.assertEqual(provider_quota.daily_only_routes(
                ["openrouter", "cloudflare", "local"], state, now), ["openrouter", "local"])
            self.assertIsNone(provider_quota.daily_limit_state(state, now, "openrouter"))
            self.assertEqual(provider_quota.daily_limit_state(state, now, "cloudflare")["until_utc"],
                             "2026-10-24T00:00:00+00:00")

    def test_openrouter_local_attempt_counter_never_reserves_attempt_51(self):
        with tempfile.TemporaryDirectory() as directory:
            counter = Path(directory) / "usage.json"
            now = datetime(2026, 10, 23, 21, 30, tzinfo=timezone.utc)
            for _ in range(50):
                self.assertTrue(multi_provider._increment_openrouter_free_attempts(counter, now))
            self.assertFalse(multi_provider._increment_openrouter_free_attempts(counter, now))
            self.assertEqual(json.loads(counter.read_text(encoding="utf-8"))["count"], 50)
            after_reset = datetime(2026, 10, 24, 0, 0, tzinfo=timezone.utc)
            self.assertTrue(multi_provider._increment_openrouter_free_attempts(counter, after_reset))
            self.assertEqual(json.loads(counter.read_text(encoding="utf-8"))["count"], 1)

    def test_openrouter_daily_cap_rejects_before_network_request(self):
        with tempfile.TemporaryDirectory() as directory:
            counter = Path(directory) / "usage.json"
            counter.write_text(json.dumps({"date": datetime.now(timezone.utc).date().isoformat(),
                                           "count": 50}), encoding="utf-8")
            with patch.object(multi_provider, "_openrouter_usage_path", return_value=counter), \
                    patch.object(multi_provider.urllib.request, "urlopen") as urlopen:
                with self.assertRaises(multi_provider.ProviderError) as raised:
                    multi_provider._chat("openrouter", "https://api.invalid", "key",
                                         "vendor/model:free", "synthetic")
            self.assertEqual(raised.exception.category, "daily_quota_exhausted")
            urlopen.assert_not_called()

    def test_only_allowlisted_daily_quota_errors_create_a_block(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "daily-limit.json"
            self.assertIsNone(provider_quota.mark_daily_limit("openrouter", "http_429", state))
            self.assertIsNone(provider_quota.mark_daily_limit("groq", "daily_quota_exhausted", state))
            self.assertFalse(state.exists())

    def test_openrouter_daily_limit_is_distinguished_from_minute_and_upstream_429(self):
        daily = urllib.error.HTTPError("https://invalid", 429, "failure", {}, io.BytesIO(
            b'{"error":{"message":"Rate limit exceeded: free-models-per-day"}}'))
        minute = urllib.error.HTTPError("https://invalid", 429, "failure", {
            "X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "1792800060000"}, io.BytesIO(b'{"error":{"message":"Rate limit exceeded"}}'))
        upstream = urllib.error.HTTPError("https://invalid", 429, "failure", {}, io.BytesIO(
            b'{"error":{"message":"provider temporarily rate-limited upstream"}}'))
        self.assertTrue(multi_provider._openrouter_daily_quota_error(daily))
        self.assertFalse(multi_provider._openrouter_daily_quota_error(minute))
        self.assertFalse(multi_provider._openrouter_daily_quota_error(upstream))

    def test_cloudflare_3036_is_daily_quota_but_capacity_3040_is_transient(self):
        daily = urllib.error.HTTPError("https://invalid", 429, "failure", {}, io.BytesIO(
            b'{"success":false,"errors":[{"code":3036,"message":"private"}]}'))
        capacity = urllib.error.HTTPError("https://invalid", 429, "failure", {}, io.BytesIO(
            b'{"success":false,"errors":[{"code":3040,"message":"private"}]}'))
        self.assertEqual(cloudflare_asr.cloudflare_http_error_category(daily), "daily_quota_exhausted")
        self.assertEqual(cloudflare_asr.cloudflare_http_error_category(capacity), "http_429")


if __name__ == "__main__":
    unittest.main()
