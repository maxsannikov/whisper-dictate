import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "attempt_statistics", ROOT / "scripts/attempt_statistics.py")
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


class AttemptStatisticsTests(unittest.TestCase):
    def write_metrics(self, path, records):
        path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")

    def test_selects_only_recent_successful_free_models_and_excludes_current(self):
        now = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            rows = [
                {"timestamp": "2026-10-23T10:00:00Z", "attempts": [
                    {"stage": "correction", "provider": "openrouter",
                     "model": "vendor/current:free", "success": True},
                    {"stage": "correction", "provider": "openrouter",
                     "model": "vendor/reliable:free", "success": True, "seconds": 2},
                    {"stage": "correction", "provider": "openrouter",
                     "model": "vendor/third:free", "success": True, "seconds": 3},
                    {"stage": "correction", "provider": "openrouter",
                     "model": "vendor/reliable:free", "success": True, "seconds": 2},
                    {"stage": "correction", "provider": "openrouter",
                     "model": "vendor/failed:free", "success": False},
                    {"stage": "correction", "provider": "openrouter",
                     "model": "vendor/unverified", "success": True},
                    {"stage": "asr", "provider": "openrouter",
                     "model": "vendor/asr:free", "success": True},
                ]},
                {"timestamp": "2026-09-01T10:00:00Z", "attempts": [
                    {"stage": "correction", "provider": "openrouter",
                     "model": "vendor/old:free", "success": True}]},
            ]
            self.write_metrics(path, rows)
            self.assertEqual(stats.choose_openrouter_fallback_model(
                path, "vendor/current:free", now), "vendor/reliable:free")
            self.assertEqual(stats.choose_openrouter_fallback_model(
                path, {"vendor/current:free", "vendor/reliable:free"}, now),
                "vendor/third:free")

    def test_empty_missing_and_malformed_metrics_fail_closed(self):
        now = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            self.assertIsNone(stats.choose_openrouter_fallback_model(path, now=now))
            path.write_text("{broken\n[]\n", encoding="utf-8")
            self.assertIsNone(stats.choose_openrouter_fallback_model(path, now=now))

    def test_daily_table_attributes_known_and_unknown_model_failures(self):
        now = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            self.write_metrics(path, [{"timestamp": "2026-10-23T11:00:00Z",
                "audio_duration_seconds": 60, "attempts": [
                    {"stage": "correction", "provider": "openrouter",
                     "model": "openrouter/free", "success": False},
                    {"stage": "correction", "provider": "openrouter",
                     "model": "openrouter/free", "success": False,
                     "error_category": "network_error"},
                    {"stage": "correction", "provider": "openrouter",
                     "model": "vendor/known:free", "success": False},
                    {"stage": "correction", "provider": "openrouter",
                     "model": "vendor/known:free", "success": True, "seconds": 4},
                ]}])
            rows = stats.aggregate_daily_attempts(path, now=now, reset_path=Path(directory) / "reset")
            by_model = {row["model"]: row for row in rows}
            self.assertEqual(by_model["openrouter/free (upstream model unknown)"]["failures"], 2)
            self.assertEqual(by_model["vendor/known:free"]["successes"], 1)
            self.assertEqual(by_model["vendor/known:free"]["failures"], 1)
            self.assertEqual(by_model["vendor/known:free"]["seconds_per_audio_minute"], 4)


if __name__ == "__main__":
    unittest.main()
