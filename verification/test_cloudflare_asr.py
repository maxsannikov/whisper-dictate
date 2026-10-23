import contextlib
import asyncio
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import urllib.error

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "cloudflare_asr.py"
spec = importlib.util.spec_from_file_location("cloudflare_asr", MODULE_PATH)
cloudflare_asr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloudflare_asr)


class CloudflareResponseTests(unittest.TestCase):
    def test_parses_text(self):
        self.assertEqual(cloudflare_asr.parse_response({"success": True, "result": {"text": " Привет "}}), "Привет")

    def test_request_uses_300_second_timeout(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"success":true,"result":{"text":"ok"}}'

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"synthetic audio")
            with patch.object(cloudflare_asr, "read_credentials", return_value=("account", "token")), \
                    patch.object(cloudflare_asr.urllib.request, "urlopen", return_value=Response()) as request:
                cloudflare_asr.transcribe(audio)
        self.assertEqual(request.call_args.kwargs["timeout"], 300)

    def test_rejects_missing_or_invalid_text(self):
        for payload in ({}, {"result": {}}, {"success": False, "result": {"text": "secret"}}):
            with self.subTest(payload=payload), self.assertRaises(cloudflare_asr.CloudflareError):
                cloudflare_asr.parse_response(payload)

    def test_daily_usage_counts_only_successful_cloudflare_audio_for_utc_day(self):
        now = cloudflare_asr.datetime(2026, 10, 23, 12, tzinfo=cloudflare_asr.timezone.utc)
        rows = [
            {"timestamp": "2026-10-23T00:10:00+00:00", "model": cloudflare_asr.MODEL,
             "audio_duration_seconds": 120, "error_code": None},
            {"timestamp": "2026-10-23T01:10:00+00:00", "model": "large-v3",
             "audio_duration_seconds": 600, "error_code": None},
            {"timestamp": "2026-10-22T23:59:00+00:00", "model": cloudflare_asr.MODEL,
             "audio_duration_seconds": 600, "error_code": None},
            {"timestamp": "2026-10-23T02:10:00+00:00", "model": cloudflare_asr.MODEL,
             "audio_duration_seconds": 60, "error_code": "cloudflare_asr_failed"},
            {"timestamp": "2026-10-23T02:20:00+00:00", "model": "large-v3",
             "audio_duration_seconds": 90, "error_code": None, "corrector_backend": "cloudflare",
             "corrector_usage": {"estimated_neurons": 1.75}},
            {"timestamp": "2026-10-23T03:20:00+00:00", "model": "large-v3",
             "audio_duration_seconds": None, "error_code": None, "corrector_backend": "cloudflare",
             "corrector_usage": {"estimated_neurons": 0.5, "estimated": True}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            metrics.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            usage = cloudflare_asr.daily_usage(metrics, now)
        self.assertEqual(usage["requests"], 1)
        self.assertEqual(usage["audio_seconds"], 120)
        self.assertAlmostEqual(usage["estimated_neurons"], 2 * 46.63)
        self.assertAlmostEqual(usage["estimated_remaining"], 10000 - 2 * 46.63 - 2.25)
        self.assertEqual(usage["corrector_requests"], 2)
        self.assertEqual(usage["corrector_estimated_token_requests"], 1)
        self.assertEqual(usage["corrector_estimated_neurons"], 2.25)

    def test_http_error_log_does_not_include_secrets_or_response_body(self):
        token = "test-api-token-must-never-appear"
        account = "test-account-id-must-never-appear"
        response_body = "Authorization: Bearer " + token + " private response details"
        http_error = urllib.error.HTTPError("https://invalid", 401, "Unauthorized", {}, io.BytesIO(response_body.encode()))
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"test audio")
            with patch.object(cloudflare_asr, "read_credentials", return_value=(account, token)), \
                    patch.object(cloudflare_asr.urllib.request, "urlopen", side_effect=http_error), \
                    contextlib.redirect_stderr(stderr):
                self.assertEqual(cloudflare_asr.main(["cloudflare_asr.py", str(audio), "ru"]), 1)
        logged = stderr.getvalue()
        self.assertIn("401/403", logged)
        self.assertNotIn(token, logged)
        self.assertNotIn(account, logged)
        self.assertNotIn(response_body, logged)

    def test_http_status_errors_are_safe_for_401_403_429_and_server_error(self):
        token = "test-token-that-must-not-appear"
        account = "test-account-that-must-not-appear"
        for status in (401, 403, 429, 500):
            body = f"Authorization: Bearer {token}; account={account}; private body"
            error = urllib.error.HTTPError("https://invalid", status, "failure", {}, io.BytesIO(body.encode()))
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                audio = Path(directory) / "audio.wav"
                audio.write_bytes(b"synthetic audio")
                with patch.object(cloudflare_asr, "read_credentials", return_value=(account, token)), \
                        patch.object(cloudflare_asr.urllib.request, "urlopen", side_effect=error):
                    with self.assertRaises(cloudflare_asr.CloudflareError) as raised:
                        cloudflare_asr.transcribe(audio)
                self.assertIn(str(status), str(raised.exception))
                self.assertNotIn(token, str(raised.exception))
                self.assertNotIn(account, str(raised.exception))
                self.assertNotIn(body, str(raised.exception))

    def test_network_and_timeout_errors_do_not_expose_exception_details(self):
        token = "network-detail-secret-marker"
        for failure in (urllib.error.URLError(token), TimeoutError(token)):
            with self.subTest(error=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                audio = Path(directory) / "audio.wav"
                audio.write_bytes(b"synthetic audio")
                with patch.object(cloudflare_asr, "read_credentials", return_value=("account", "token")), \
                        patch.object(cloudflare_asr.urllib.request, "urlopen", side_effect=failure):
                    with self.assertRaises(cloudflare_asr.CloudflareError) as raised:
                        cloudflare_asr.transcribe(audio)
                self.assertNotIn(token, str(raised.exception))
                self.assertIn("Сетевая ошибка", str(raised.exception))

    def test_usage_menu_refreshes_when_opened_and_marks_value_as_estimate(self):
        source = (Path(__file__).parents[1] / "scripts" / "whisper-tray.py").read_text(encoding="utf-8")
        self.assertIn("cloudflare_daily_usage(METRICS_FILE)", source)
        self.assertIn('cloudflare_submenu.connect("show", refresh_cloudflare_usage)', source)
        self.assertIn("Оценка расхода по локальным метрикам (UTC)", source)

    def test_cloudflare_failure_delegates_to_local_only_when_enabled(self):
        calls = []

        async def cloudflare_call():
            calls.append("cloudflare")
            raise cloudflare_asr.CloudflareError("Cloudflare Workers AI достиг лимита запросов (429)")

        async def local_call():
            calls.append("local")
            return "Локальная расшифровка."

        result, used_fallback, reason = asyncio.run(
            cloudflare_asr.cloudflare_first(cloudflare_call, local_call, fallback_local=True))
        self.assertEqual(result, "Локальная расшифровка.")
        self.assertTrue(used_fallback)
        self.assertIn("429", reason)
        self.assertEqual(calls, ["cloudflare", "local"])

    def test_cloudflare_failure_does_not_delegate_when_disabled(self):
        calls = []

        async def cloudflare_call():
            calls.append("cloudflare")
            raise cloudflare_asr.CloudflareError("Cloudflare Workers AI достиг лимита запросов (429)")

        async def local_call():
            calls.append("local")
            return "must not run"

        with self.assertRaises(cloudflare_asr.CloudflareError):
            asyncio.run(cloudflare_asr.cloudflare_first(cloudflare_call, local_call, fallback_local=False))
        self.assertEqual(calls, ["cloudflare"])

    def test_hotkey_profile_parser_accepts_config_without_new_cloudflare_key(self):
        script = Path(__file__).parents[1] / "scripts" / "whisper-dictate-v2"
        source = script.read_text(encoding="utf-8")
        marker = "mapfile -t _voice_values < <(/usr/bin/python3 - \"$VOICE_CONFIG\" <<'PY'\n"
        start = source.index(marker) + len(marker)
        end = source.index("\nPY\n)", start)
        parser = source[start:end]
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"model": "large-v3", "audio_ducking": {"enabled": True}}), encoding="utf-8")
            result = subprocess.run([sys.executable, "-", str(config)], input=parser,
                                    capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cloudflare_asr_enabled=false", result.stdout)
        self.assertIn("audio_duck_enabled=true", result.stdout)

    def test_hotkey_bootstrap_enables_default_local_fallback_with_cloudflare(self):
        script = Path(__file__).parents[1] / "scripts" / "whisper-dictate-v2"
        source = script.read_text(encoding="utf-8")
        bootstrap = source[:source.index("# Получить название модели из переменной окружения или конфига")]
        profile = {"model": "large-v3", "language": "ru", "device": "cuda",
                   "compute_type": "int8_float16", "beam_size": 1, "best_of": 1,
                   "temperature": 0.0, "vad": False, "result_mode": "clipboard",
                   "cloudflare_workers_ai_enabled": True}
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / ".config" / "whisper-dictate"
            config_dir.mkdir(parents=True)
            (config_dir / "config.json").write_text(json.dumps(profile), encoding="utf-8")
            env = os.environ.copy()
            env["HOME"] = directory
            result = subprocess.run(
                ["bash", "-c", bootstrap + "\nprintf 'provider=%s fallback=%s\\n' \"$WHISPER_ASR_PROVIDER\" \"$WHISPER_CLOUDFLARE_FALLBACK\""],
                env=env, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("provider=cloudflare fallback=true", result.stdout)


if __name__ == "__main__":
    unittest.main()
