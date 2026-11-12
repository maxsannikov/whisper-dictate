import importlib.util
import ast
import difflib
import logging
from pathlib import Path
import time
import unittest
import os
import io
import tempfile
import urllib.error
import urllib.request
import json
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("multi_provider", ROOT / "scripts/multi_provider.py")
multi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multi)


class FakeCloudflareError(Exception):
    def __init__(self, message, category):
        super().__init__(message)
        self.category = category


class MultiProviderPolicyTests(unittest.TestCase):
    def load_correct_text(self, overrides=None):
        source_path = ROOT / "scripts/whisper-daemon.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                    and item.name == "correct_text_sync")
        def legacy_route_chain(source, settings, cloudflare_enabled):
            raw = dict(settings or {})
            correction = dict(raw.get("correction", {})) if isinstance(raw.get("correction", {}), dict) else {}
            correction.setdefault("codex_enabled", False)
            correction.setdefault("claude_enabled", False)
            raw["correction"] = correction
            return multi.correction_chain(source, raw, cloudflare_enabled)
        namespace = {
            "time": time, "json": json, "urllib": __import__("urllib"),
            "logger": logging.getLogger("multi-provider-test"),
            "multi_correction_chain": legacy_route_chain,
            "normalize_asr_text": lambda text: text.strip(), "finalize_text": lambda text: text,
            "correction_chunks": lambda text: [text], "correction_response": lambda *_: ("unused", "stop"),
            "correction_safety_reason": lambda _original, candidate, _reason: None if isinstance(candidate, str) and bool(candidate) else "invalid",
            "correction_is_safe": lambda _original, candidate, _reason: isinstance(candidate, str) and bool(candidate),
            "ollama_loaded_device": lambda *_: "cpu", "correction_token_counts": lambda *_: (1, 1, True),
            "correction_neurons": lambda *_: 0,
            "estimate_cloudflare_correction_neurons": lambda *_: 1,
            "reserve_cloudflare_daily_neurons": lambda *_: True,
            "cloudflare_correct": lambda *_: ("unused", 1, 1),
            "groq_correct": lambda *_: "fixed text", "openrouter_correct": lambda *_args, **_kwargs: ("fixed text", "vendor/model"),
            "codex_correct": lambda *_args, **_kwargs: "fixed text",
            "claude_correct": lambda *_args, **_kwargs: "fixed text",
            "google_correct": lambda *_: "fixed text", "ProviderError": multi.ProviderError,
            "CloudflareError": FakeCloudflareError,
            "safe_provider_error_category": lambda error: error.category,
            "safe_provider_settings": multi.safe_settings,
            "CORRECTOR_INSTRUCTION": "", "RETRY_INSTRUCTION": "",
            "daily_limit_state": lambda provider=None: None, "mark_daily_limit": lambda *_args: None,
            "daily_only_routes": lambda routes: list(routes),
            "METRICS": Path("/tmp/whisper-dictate-test-metrics.jsonl"),
            "choose_openrouter_fallback_model": lambda *_args: None,
            "cloudflare_correction_model_order": lambda _path, models, _priority=None: list(models),
            "CLOUDFLARE_CORRECTOR_MODELS": ["cf"],
            "record_attempt": lambda *_args, **_kwargs: None,
        }
        namespace.update(overrides or {})
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
        return namespace["correct_text_sync"], namespace

    def test_global_correction_off_returns_unmodified_asr_without_provider_attempts(self):
        correct, namespace = self.load_correct_text()
        attempts, backends = [], []
        result = correct("  raw ASR text  ", {"enabled": True, "provider": "ollama"},
            provider_settings={"correction": {"enabled": False, "codex_enabled": True}},
            source="batch", backend_callback=backends.append, attempt_log=attempts)
        self.assertEqual(result, ("  raw ASR text  ", None, 0.0, "local_disabled", None))
        self.assertEqual(backends, ["local"])
        self.assertEqual(attempts, [])

    def test_local_and_remote_correction_execution_follow_saved_order(self):
        def run(order):
            calls = []
            correct, _namespace = self.load_correct_text({
                "correction_response": lambda *_args: (calls.append("local") or "LOCAL", "stop"),
                "codex_correct": lambda *_args, **_kwargs: calls.append("codex") or "REMOTE",
            })
            result = correct("source text", {"enabled": True, "provider": "ollama",
                "model": "qwen-local", "prompt": ""},
                provider_settings={"correction": {"enabled": True,
                    "codex_enabled": True, "order": order}}, source="live_dictation")
            return result[0], calls

        text, calls = run(["local", "codex"])
        self.assertEqual((text, calls), ("LOCAL", ["local"]))
        text, calls = run(["codex", "local"])
        self.assertEqual((text, calls), ("REMOTE", ["codex"]))

    def test_asr_chains_are_explicit_and_local_only_for_non_live_sources(self):
        settings = {"asr": {"groq_enabled": True, "google_enabled": True, "quality": "accuracy"}}
        self.assertEqual(multi.asr_chain("live_dictation", settings, True),
                         ["cloudflare", "groq_precise", "google", "local"])
        settings["asr"]["quality"] = "fast"
        self.assertEqual(multi.asr_chain("live_dictation", settings, False),
                         ["groq_fast", "google", "local"])
        for source in ("file", "batch", "youtube", "rutube", "vk", "unknown", None):
            self.assertEqual(multi.asr_chain(source, settings, True), ["local"])

    def test_local_stage_can_be_reordered_without_losing_safe_asr_fallback(self):
        settings = {"asr": {"order": ["local", "cloudflare", "groq", "google"],
                             "groq_enabled": True, "google_enabled": True}}
        self.assertEqual(multi.asr_chain("live_dictation", settings, True),
                         ["local", "cloudflare", "groq_precise", "google", "local"])
        self.assertEqual(multi.asr_chain("live_dictation", {"asr": {"order": ["google"]}}, False),
                         ["local"])

    def test_global_correction_switch_defaults_on_and_preserves_provider_toggles(self):
        disabled = multi.safe_settings({"correction": {
            "enabled": False, "codex_enabled": True, "claude_enabled": True,
            "openrouter_enabled": True, "groq_enabled": True, "google_enabled": True}})
        self.assertFalse(disabled["correction"]["enabled"])
        for key in ("codex_enabled", "claude_enabled", "openrouter_enabled",
                    "groq_enabled", "google_enabled"):
            self.assertTrue(disabled["correction"][key])
        self.assertEqual(multi.correction_chain("live_dictation", disabled, True), ["local"])
        self.assertEqual(multi.correction_chain("file", disabled, True), ["local"])
        reenabled = multi.safe_settings({"correction": {
            **disabled["correction"], "enabled": True}})
        self.assertEqual(multi.correction_chain("live_dictation", reenabled, False)[0], "codex")
        self.assertTrue(reenabled["correction"]["openrouter_enabled"])
        self.assertTrue(multi.safe_settings({})["correction"]["enabled"])

    def test_local_correction_stage_order_is_preserved(self):
        settings = {"correction": {"enabled": True, "codex_enabled": True,
            "order": ["local", "codex", "claude", "cloudflare", "openrouter",
                      "groq_primary", "groq_fallback", "google"]}}
        self.assertEqual(multi.correction_chain("live_dictation", settings, True)[:3],
                         ["local", "codex", "cloudflare"])

    def test_local_model_lists_are_filtered_deduplicated_and_keep_order(self):
        normalized = multi.safe_settings({"asr": {"local_models": [
            "gigaam-v3-e2e-rnnt", "not-a-model", "faster-whisper",
            "gigaam-v3-e2e-rnnt"]}, "correction": {"local_models": [
            "qwen3.5:2b-q4_K_M", "unknown", "qwen3.5:4b-q4_K_M",
            "qwen3.5:2b-q4_K_M"]}})
        self.assertEqual(normalized["asr"]["local_models"],
                         ["gigaam-v3-e2e-rnnt", "faster-whisper"])
        self.assertEqual(normalized["correction"]["local_models"],
                         ["qwen3.5:2b-q4_K_M", "qwen3.5:4b-q4_K_M"])

    def test_invalid_or_missing_local_model_lists_default_empty_and_legacy_local_only(self):
        for value in ({}, {"asr": {"local_models": "gigaam-v3-e2e-rnnt"},
                          "correction": {"local_models": {"model": "qwen3.5:4b-q4_K_M"}}}):
            normalized = multi.safe_settings(value)
            self.assertEqual(normalized["asr"]["local_models"], [])
            self.assertEqual(normalized["correction"]["local_models"], [])
        legacy = multi.safe_settings({})
        for source in ("file", "batch", "youtube", "rutube", "vk", "unknown"):
            self.assertEqual(multi.asr_chain(source, legacy, True), ["local"])
            self.assertEqual(multi.correction_chain(source, legacy, True), ["local"])

    def test_explicit_order_wins_and_legacy_cloudflare_position_migrates(self):
        settings = {"correction": {"groq_enabled": True, "openrouter_enabled": True,
            "google_enabled": True, "codex_enabled": False, "cloudflare_position": "after_openrouter"}}
        self.assertEqual(multi.correction_chain("live_dictation", settings, True),
            ["openrouter", "cloudflare", "groq_primary", "groq_fallback", "google", "local"])
        normalized = multi.safe_settings({"correction": {"cloudflare_position": "after_google"}})
        self.assertEqual(normalized["correction"]["order"][
            normalized["correction"]["order"].index("google") + 1], "cloudflare")
        self.assertNotIn("cloudflare_position", normalized["correction"])
        conflict = {"correction": {"order": ["codex", "claude", "openrouter", "cloudflare"],
                                   "cloudflare_position": "after_google", "claude_enabled": True,
                                   "openrouter_enabled": True}}
        self.assertEqual(multi.correction_chain("live_dictation", conflict, True),
                         ["codex", "claude", "openrouter", "cloudflare", "local"])
        self.assertEqual(multi.correction_chain("file", settings, True), ["local"])

    def test_openrouter_primary_with_cloudflare_then_local(self):
        settings = {"correction": {"openrouter_enabled": True, "codex_enabled": False,
                                   "cloudflare_position": "after_openrouter"}}
        self.assertEqual(multi.correction_chain("live_dictation", settings, True),
                         ["openrouter", "cloudflare", "local"])
        self.assertEqual(multi.correction_chain("live_dictation", settings, False),
                         ["openrouter", "local"])
        self.assertEqual(multi.correction_chain("live_dictation", {}, True),
                         ["codex", "cloudflare", "local"])

    def test_invalid_settings_fail_closed(self):
        self.assertEqual(multi.asr_chain("live_dictation", {"asr": "invalid"}, True),
                         ["cloudflare", "local"])

    def test_default_provider_order_and_free_openrouter_model(self):
        self.assertEqual(multi.MODEL_IDS["openrouter_correct"], "openrouter/free")
        self.assertEqual(multi.asr_chain("live_dictation", {}, True), ["cloudflare", "local"])
        self.assertEqual(multi.correction_chain("live_dictation", {
            "correction": {"openrouter_enabled": True}}, True),
            ["codex", "cloudflare", "openrouter", "local"])

    def test_requested_live_correction_priority_is_default_when_providers_enabled(self):
        settings = {"correction": {"claude_enabled": True, "openrouter_enabled": True}}
        self.assertEqual(multi.correction_chain("live_dictation", settings, True),
                         ["codex", "claude", "cloudflare", "openrouter", "local"])

    def test_codex_correction_is_live_only_first_by_default_and_independently_toggleable(self):
        self.assertEqual(multi.correction_chain("live_dictation", {}, False), ["codex", "local"])
        self.assertEqual(multi.correction_chain("file", {}, True), ["local"])
        settings = {"correction": {"codex_enabled": False, "claude_enabled": True,
            "openrouter_enabled": True, "order": ["claude", "codex", "openrouter"]}}
        self.assertEqual(multi.correction_chain("live_dictation", settings, False),
                         ["claude", "openrouter", "local"])

    def test_codex_retries_guard_rejection_before_advancing_to_next_provider(self):
        codex_calls = []
        fallback_calls = []
        safety_calls = []
        recorded = []
        def codex(*_args, **_kwargs):
            codex_calls.append(True)
            return "candidate"
        def safety(_source, _candidate, _reason):
            safety_calls.append(True)
            return "word_coverage_order" if len(safety_calls) == 1 else None
        def record(attempts, stage, provider, model, success, _started,
                   error_category=None, device=None):
            attempts.append({"stage": stage, "provider": provider, "model": model,
                "success": success, "error_category": error_category, "device": device})
        correction, _namespace = self.load_correct_text(overrides={
            "codex_correct": codex,
            "openrouter_correct": lambda *_args, **_kwargs:
                (fallback_calls.append(True) or "fallback", "vendor/model"),
            "correction_safety_reason": safety,
            "record_attempt": record,
        })
        settings = {"correction": {"codex_enabled": True, "codex_retry_attempts": 2,
            "openrouter_enabled": True, "openrouter_retry_attempts": 0}}
        result = correction("source", {"enabled": False}, provider_settings=settings,
                            source="live_dictation", attempt_log=recorded)
        self.assertEqual(result[3], "codex")
        self.assertEqual(len(codex_calls), 2)
        self.assertEqual(fallback_calls, [])
        self.assertEqual([item["success"] for item in recorded], [False, True])

    def test_codex_exhausts_three_attempts_then_uses_next_provider(self):
        codex_calls = []
        fallback_calls = []
        recorded = []
        def record(attempts, stage, provider, model, success, _started,
                   error_category=None, device=None):
            attempts.append({"stage": stage, "provider": provider, "model": model,
                "success": success, "error_category": error_category, "device": device})
        correction, _namespace = self.load_correct_text(overrides={
            "codex_correct": lambda *_args, **_kwargs:
                (codex_calls.append(True) or "rejected candidate"),
            "openrouter_correct": lambda *_args, **_kwargs:
                (fallback_calls.append(True) or "accepted fallback", "vendor/model"),
            "correction_safety_reason": lambda _source, candidate, _reason:
                "word_coverage_order" if candidate == "rejected candidate" else None,
            "record_attempt": record,
        })
        settings = {"correction": {"codex_enabled": True, "codex_retry_attempts": 2,
            "openrouter_enabled": True, "openrouter_retry_attempts": 0}}
        result = correction("source", {"enabled": False}, provider_settings=settings,
                            source="live_dictation", attempt_log=recorded)
        self.assertEqual(result[3], "openrouter")
        self.assertEqual(len(codex_calls), 3)
        self.assertEqual(len(fallback_calls), 1)
        self.assertEqual([item["success"] for item in recorded], [False, False, False, True])

    def test_codex_retries_cli_unavailable_twice_then_advances(self):
        codex_calls = []
        claude_calls = []
        def codex(*_args, **_kwargs):
            codex_calls.append(True)
            raise multi.ProviderError("codex", "cli_unavailable")
        correction, _namespace = self.load_correct_text(overrides={
            "codex_correct": codex,
            "claude_correct": lambda *_args, **_kwargs:
                (claude_calls.append(True) or "accepted"),
        })
        settings = {"correction": {"codex_enabled": True, "codex_retry_attempts": 2,
                                   "claude_enabled": True, "claude_retry_attempts": 2}}
        result = correction("source", {"enabled": False}, provider_settings=settings,
                            source="live_dictation")
        self.assertEqual(codex_calls, [True, True, True])
        self.assertEqual(claude_calls, [True])
        self.assertEqual(result[3], "claude")

    def test_claude_retries_twice_then_advances_to_cloudflare(self):
        claude_calls = []
        cloudflare_calls = []
        def claude(*_args, **_kwargs):
            claude_calls.append(True)
            raise multi.ProviderError("claude", "cli_unavailable")
        correction, _namespace = self.load_correct_text(overrides={
            "claude_correct": claude,
            "cloudflare_correct": lambda *_args: (cloudflare_calls.append(True) or "accepted", 1, 1),
        })
        settings = {"correction": {"codex_enabled": False, "claude_enabled": True,
                                   "claude_retry_attempts": 2,
                                   "order": ["codex", "claude", "cloudflare", "openrouter"]}}
        result = correction("source", {"enabled": False},
                            cloudflare_profile={"enabled": True, "model": "cf",
                                                "max_tokens": 100, "temperature": 0},
                            provider_settings=settings, source="live_dictation")
        self.assertEqual(claude_calls, [True, True, True])
        self.assertEqual(cloudflare_calls, [True])
        self.assertEqual(result[3], "cloudflare")

    def test_codex_retry_setting_defaults_to_two_and_is_bounded(self):
        self.assertEqual(multi.safe_settings({})["correction"]["codex_retry_attempts"], 2)
        self.assertEqual(multi.safe_settings({"correction": {
            "codex_retry_attempts": 0}})["correction"]["codex_retry_attempts"], 0)
        self.assertEqual(multi.safe_settings({"correction": {
            "codex_retry_attempts": 1}})["correction"]["codex_retry_attempts"], 1)
        self.assertEqual(multi.safe_settings({"correction": {
            "codex_retry_attempts": 3}})["correction"]["codex_retry_attempts"], 2)

    def test_claude_retries_default_to_two_and_remain_bounded(self):
        settings = multi.safe_settings({})["correction"]
        self.assertEqual(settings["claude_retry_attempts"], 2)
        self.assertEqual(multi.safe_settings({"correction": {
            "claude_retry_attempts": 0}})["correction"]["claude_retry_attempts"], 0)

    def test_other_provider_retry_settings_default_to_zero_and_bound_to_two(self):
        settings = multi.safe_settings({})["correction"]
        self.assertEqual(settings["groq_primary_retry_attempts"], 0)
        self.assertEqual(settings["groq_fallback_retry_attempts"], 0)
        self.assertEqual(settings["google_retry_attempts"], 0)
        for key in ("groq_primary_retry_attempts",
                    "groq_fallback_retry_attempts", "google_retry_attempts"):
            self.assertEqual(multi.safe_settings({"correction": {key: 2}})["correction"][key], 2)
            self.assertEqual(multi.safe_settings({"correction": {key: 4}})["correction"][key], 0)
        self.assertEqual(multi.safe_settings({"correction": {
            "claude_retry_attempts": 4}})["correction"]["claude_retry_attempts"], 2)

    def test_google_retries_transient_failure_and_records_each_attempt(self):
        calls = []
        recorded = []
        def google(*_args):
            calls.append(True)
            if len(calls) == 1:
                raise multi.ProviderError("google", "network_error")
            return "fixed text"
        def record(attempts, stage, provider, model, success, _started,
                   error_category=None, device=None):
            attempts.append({"stage": stage, "provider": provider, "model": model,
                "success": success, "error_category": error_category, "device": device})
        correction, _namespace = self.load_correct_text(overrides={
            "google_correct": google, "record_attempt": record,
        })
        settings = {"correction": {"codex_enabled": False, "google_enabled": True,
            "google_retry_attempts": 1}}
        result = correction("source", {"enabled": False}, provider_settings=settings,
                            source="live_dictation", attempt_log=recorded)
        self.assertEqual(result[3], "google")
        self.assertEqual(len(calls), 2)
        self.assertEqual([item["success"] for item in recorded], [False, True])

    def test_google_does_not_retry_auth_or_quota_failures(self):
        for category in ("http_401_403", "http_429", "daily_quota_exhausted"):
            with self.subTest(category=category):
                calls = []
                def google(*_args):
                    calls.append(True)
                    raise multi.ProviderError("google", category)
                correction, _namespace = self.load_correct_text(overrides={
                    "google_correct": google,
                })
                settings = {"correction": {"codex_enabled": False,
                    "google_enabled": True, "google_retry_attempts": 2}}
                corrector = {"enabled": True, "provider": "ollama", "model": "synthetic",
                             "prompt": "", "num_gpu": 0}
                correction("source", corrector, provider_settings=settings,
                           source="live_dictation")
                self.assertEqual(len(calls), 1)

    def test_cli_adapter_uses_ephemeral_no_tool_execution_and_stdin_only(self):
        class CompletedProcess:
            returncode = 0
            def __init__(self, command, **kwargs):
                self.command, self.kwargs = command, kwargs
                self.stdin_data = None
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text("synthetic corrected", encoding="utf-8")
            def communicate(self, input=None, timeout=None):
                self.stdin_data = input
                return (None, None)
            def poll(self):
                return 0
        with patch.object(multi.shutil, "which", return_value="/usr/bin/codex"), \
             patch.object(multi.subprocess, "Popen", side_effect=CompletedProcess) as popen:
            result = multi.codex_correct("synthetic source", "gpt-6.1-sol")
        self.assertEqual(result, "synthetic corrected")
        command = popen.call_args.args[0]
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--sandbox", command)
        self.assertIn("--disable", command)
        self.assertIn("shell_tool", command)
        self.assertIn("-", command)
        self.assertEqual(popen.call_args.kwargs["stdout"], multi.subprocess.DEVNULL)
        self.assertEqual(popen.call_args.kwargs["stdin"], multi.subprocess.PIPE)
        self.assertTrue(popen.call_args.kwargs["env"]["PATH"].startswith("/usr/bin"))

    def test_cli_lookup_finds_user_npm_bin_when_service_path_is_short(self):
        with tempfile.TemporaryDirectory() as directory:
            cli_dir = Path(directory) / ".local" / "npm" / "bin"
            cli_dir.mkdir(parents=True)
            executable = cli_dir / "codex"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o700)
            with patch.object(multi.shutil, "which", return_value=None), \
                 patch.object(multi.Path, "home", return_value=Path(directory)):
                self.assertEqual(multi.find_cli_executable("codex"), str(executable))
                self.assertIsNone(multi.find_cli_executable("claude"))
                self.assertIsNone(multi.find_cli_executable("unknown"))

    def test_openrouter_retry_setting_is_limited_to_two_additional_calls(self):
        self.assertEqual(multi.safe_settings({"correction": {
            "openrouter_retry_attempts": 0}})["correction"]["openrouter_retry_attempts"], 0)
        self.assertEqual(multi.safe_settings({"correction": {
            "openrouter_retry_attempts": 1}})["correction"]["openrouter_retry_attempts"], 1)
        self.assertEqual(multi.safe_settings({"correction": {
            "openrouter_retry_attempts": 3}})["correction"]["openrouter_retry_attempts"], 2)

    def test_daemon_uses_same_openrouter_model_id_as_adapter(self):
        daemon = (ROOT / "scripts/whisper-daemon.py").read_text(encoding="utf-8")
        self.assertIn('"openrouter": "openrouter/free"', daemon)
        self.assertNotIn('"openrouter": "qwen/qwen3.8-27b:free"', daemon)
        self.assertEqual(multi.correction_chain("live_dictation", {"correction": "bad"}, True),
                         ["codex", "cloudflare", "local"])

    def test_credentials_are_private_and_never_returned_in_status(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(multi, "CREDENTIAL_DIR", Path(directory)):
                empty = Path(directory) / "openrouter.key"
                empty.write_text("", encoding="utf-8")
                empty.chmod(0o600)
                self.assertFalse(multi.credentials_configured("openrouter"))
                multi.store_credential("groq", "test-key-value")
                path = Path(directory) / "groq.key"
                self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
                self.assertTrue(multi.credentials_configured("groq"))

    def test_tray_has_open_file_actions_and_help_explains_credential_fields(self):
        tray = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        guide = (ROOT / "docs/guide.md").read_text(encoding="utf-8")
        self.assertIn('icon_menu_item("Открыть файл Account ID и API token…"', tray)
        self.assertIn('icon_menu_item(f"Открыть файл API key {provider_label}…"', tray)
        self.assertIn('os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600', tray)
        for field in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "groq.key", "google.key", "openrouter.key"):
            self.assertIn(field, guide)

    def test_groq_audio_limit_and_http_categories(self):
        self.assertEqual(multi._category_for_http(401), "http_401_403")
        self.assertEqual(multi._category_for_http(403), "http_401_403")
        self.assertEqual(multi._category_for_http(413), "http_413")
        self.assertEqual(multi._category_for_http(429), "http_429")
        self.assertEqual(multi._category_for_http(404), "unsupported_model")
        self.assertEqual(multi._category_for_http(503), "provider_unavailable")

    def test_gemini_interaction_response_parser(self):
        self.assertEqual(multi._interaction_text({"steps": [{"type": "model_output",
            "content": [{"type": "text", "text": "transcript"}]}]}), "transcript")

    def test_chat_adapter_builds_text_only_request_and_parses_mock_response(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(multi, "_openrouter_usage_path", return_value=Path(directory) / "usage.json"), \
             patch.object(multi, "_credential", return_value="synthetic-key"), \
             patch.object(multi, "_request", return_value={"choices": [{"message": {"content": "fixed."}}]}) as request:
            result = multi.openrouter_correct("synthetic transcript")
            self.assertEqual(multi.openrouter_local_usage()[1], 1)
        self.assertEqual(result, "fixed.")
        req = request.call_args.args[1]
        self.assertIn("Authorization", req.headers)
        self.assertNotIn("synthetic transcript", str(req.headers))
        self.assertIn("synthetic transcript", req.data.decode("utf-8"))

    def test_cancellable_openrouter_child_is_terminated_without_api_request(self):
        class ChildProcess:
            def __init__(self):
                self.returncode = None
                self.terminated = False
            def poll(self): return self.returncode
            def terminate(self):
                self.terminated = True
                self.returncode = -15
            def communicate(self, **_kwargs): return b"", None
            def kill(self): self.returncode = -9
            def wait(self): return self.returncode

        child = ChildProcess()
        with patch.object(multi.subprocess, "Popen", return_value=child) as start:
            with self.assertRaises(multi.ProviderError) as raised:
                multi.openrouter_correct("synthetic transcript", cancel_callback=lambda: True)
        self.assertEqual(raised.exception.category, "cancelled_by_user")
        self.assertTrue(child.terminated)
        start.assert_called_once()

    def test_openrouter_returns_only_safe_selected_model_metadata(self):
        with patch.object(multi, "_increment_openrouter_free_attempts", return_value=True), \
             patch.object(multi, "_credential", return_value="synthetic-key"), \
             patch.object(multi, "_request", return_value={"model": "vendor/model:free", "choices": [{"message": {"content": "fixed"}}]}):
            self.assertEqual(multi.openrouter_correct("synthetic", include_model=True), ("fixed", "vendor/model:free"))
        for payload, category in (({}, "invalid_response_missing_choice"),
                                  ({"choices": [{"message": {}}]}, "invalid_response_missing_content"),
                                  ({"choices": [{"message": {"content": []}}]}, "invalid_response_non_string"),
                                  ({"choices": [{"message": {"content": " "}}]}, "invalid_response_empty")):
            with self.subTest(category=category), patch.object(multi, "_increment_openrouter_free_attempts", return_value=True), \
                    patch.object(multi, "_credential", return_value="synthetic-key"), \
                    patch.object(multi, "_request", return_value=payload):
                with self.assertRaises(multi.ProviderError) as raised:
                    multi.openrouter_correct("synthetic", include_model=True)
                self.assertEqual(raised.exception.category, category)

    def test_groq_multipart_request_success_is_mocked(self):
        with tempfile.NamedTemporaryFile() as audio:
            audio.write(b"synthetic audio")
            audio.flush()
            with patch.object(multi, "_credential", return_value="synthetic-key"), \
                 patch.object(multi, "_request", return_value={"text": "synthetic words"}) as request:
                result = multi._groq_transcribe_single(audio.name, "ru", "whisper-large-v3")
        self.assertEqual(result, "synthetic words")
        req = request.call_args.args[1]
        self.assertEqual(req.full_url, "https://api.groq.com/openai/v1/audio/transcriptions")
        self.assertIn(b"whisper-large-v3", req.data)
        self.assertIn(b"synthetic audio", req.data)

    def test_oversized_audio_uses_local_ffmpeg_chunks(self):
        with tempfile.NamedTemporaryFile() as audio:
            audio.truncate(multi.GROQ_MAX_AUDIO_BYTES + 1)
            calls = []
            def fake_ffmpeg(command, **_kwargs):
                calls.append(command)
                pattern = Path(command[-1])
                (pattern.parent / "chunk-000.wav").write_bytes(b"chunk-a")
                (pattern.parent / "chunk-001.wav").write_bytes(b"chunk-b")
            with patch.object(multi.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                 patch.object(multi.subprocess, "run", side_effect=fake_ffmpeg), \
                 patch.object(multi, "_groq_transcribe_single", side_effect=["first", "second"]):
                result = multi.groq_transcribe(audio.name, "ru", "whisper-large-v3")
        self.assertEqual(result, "first second")
        self.assertIn("-map_metadata", calls[0])
        self.assertIn("-segment_time", calls[0])

    def test_gemini_files_and_interaction_are_mocked_and_file_is_deleted(self):
        class UploadStart:
            headers = {"X-Goog-Upload-URL": "https://generativelanguage.googleapis.com/upload/token"}
            def __enter__(self): return self
            def __exit__(self, *_args): return False
        class EmptyResponse:
            def close(self): pass
        with tempfile.NamedTemporaryFile() as audio:
            audio.write(b"synthetic audio")
            audio.flush()
            with patch.object(multi, "_credential", return_value="synthetic-key"), \
                 patch.object(multi.urllib.request, "urlopen", side_effect=[UploadStart(), EmptyResponse()]) as urlopen, \
                 patch.object(multi, "_request", side_effect=[
                     {"file": {"name": "files/synthetic", "uri": "https://file.invalid/audio"}},
                     {"steps": [{"type": "model_output", "content": [{"type": "text", "text": "words"}]}]}]) as request:
                result = multi.transcribe_gemini(audio.name)
        self.assertEqual(result, "words")
        interaction = request.call_args_list[1].args[1]
        self.assertIn(b"gemini-3.5-transcribe", interaction.data)
        self.assertIn(b'"store": false', interaction.data)
        self.assertEqual(urlopen.call_args.args[0].method, "DELETE")

    def test_http_errors_are_sanitized_and_timeout_is_categorized(self):
        secret_body = b"private-response-token-marker"
        for status, expected in ((401, "http_401_403"), (403, "http_401_403"),
                                 (413, "http_413"), (429, "http_429")):
            error = urllib.error.HTTPError("https://api.invalid", status, "failure", {}, io.BytesIO(secret_body))
            with patch.object(multi.urllib.request, "urlopen", side_effect=error):
                with self.assertRaises(multi.ProviderError) as raised:
                    multi._request("groq", urllib.request.Request("https://api.invalid"))
            self.assertEqual(raised.exception.category, expected)
            self.assertNotIn("private-response-token-marker", str(raised.exception))
        with patch.object(multi.urllib.request, "urlopen", side_effect=TimeoutError()):
            with self.assertRaises(multi.ProviderError) as raised:
                multi._request("google", urllib.request.Request("https://api.invalid"))
        self.assertEqual(raised.exception.category, "timeout")
        with patch.object(multi.urllib.request, "urlopen", side_effect=urllib.error.URLError("network")):
            with self.assertRaises(multi.ProviderError) as raised:
                multi._request("google", urllib.request.Request("https://api.invalid"))
        self.assertEqual(raised.exception.category, "network_error")

    def test_missing_key_and_malformed_json_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(multi, "CREDENTIAL_DIR", Path(directory)):
                with self.assertRaises(multi.ProviderError) as raised:
                    multi._credential("openrouter")
        self.assertEqual(raised.exception.category, "missing_credentials")
        class BadJson:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self, _limit): return b"not-json"
        with patch.object(multi.urllib.request, "urlopen", return_value=BadJson()):
            with self.assertRaises(multi.ProviderError) as raised:
                multi._request("openrouter", urllib.request.Request("https://api.invalid"))
        self.assertEqual(raised.exception.category, "invalid_response")

    def test_oversized_groq_payload_is_rejected_before_file_read(self):
        with tempfile.NamedTemporaryFile() as audio:
            audio.truncate(multi.GROQ_MAX_AUDIO_BYTES + 1)
            with self.assertRaises(multi.ProviderError) as raised:
                multi._groq_transcribe_single(audio.name, "ru", "whisper-large-v3")
        self.assertEqual(raised.exception.category, "http_413")

    def test_correction_stops_on_first_safe_success_and_falls_through_on_failure(self):
        settings = {"correction": {"groq_enabled": True, "openrouter_enabled": True,
                                   "google_enabled": True}}
        calls = []
        def openrouter(*_args, **_kwargs):
            calls.append("openrouter")
            raise multi.ProviderError("openrouter", "provider_unavailable")
        def groq(_text, model):
            calls.append(model)
            if model.endswith("qwen3.8-27b"):
                raise multi.ProviderError("groq", "http_429")
            return "corrected"
        correct, _ = self.load_correct_text({"groq_correct": groq,
            "openrouter_correct": openrouter})
        result = correct("source text", {"enabled": False, "provider": "ollama", "model": "local", "prompt": ""},
                         None, None, None, settings, "live_dictation")
        self.assertEqual(result[0], "corrected")
        self.assertEqual(result[3], "groq")
        self.assertEqual(calls, ["openrouter", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"])

    def test_openrouter_daily_quota_skips_only_openrouter_and_falls_through_to_cloudflare(self):
        calls = []
        blocked = []
        def openrouter(*_args, **_kwargs):
            calls.append("openrouter")
            raise multi.ProviderError("openrouter", "daily_quota_exhausted")
        def state(provider=None):
            return {"provider": "openrouter", "until_utc": "2026-10-24T00:00:00+00:00"} \
                if blocked and provider == "openrouter" else None
        def mark(provider, category):
            blocked.append((provider, category))
        correct, _ = self.load_correct_text({
            "openrouter_correct": openrouter,
            "cloudflare_correct": lambda *_: calls.append("cloudflare") or ("fixed", 1, 1),
            "daily_limit_state": state, "mark_daily_limit": mark,
            "correction_response": lambda text, *_args: (text, "stop"),
        })
        result = correct("source text", {"enabled": True, "provider": "ollama", "model": "local", "prompt": ""},
            {"enabled": True, "model": "cf-model", "temperature": 0, "max_tokens": 128}, None, None,
            {"correction": {"openrouter_enabled": True,
                            "order": ["codex", "claude", "openrouter", "cloudflare"]}}, "live_dictation")
        self.assertEqual(blocked, [("openrouter", "daily_quota_exhausted")])
        self.assertEqual(calls, ["openrouter", "cloudflare"])
        self.assertEqual(result[0], "fixed")
        self.assertEqual(result[3], "cloudflare")

    def test_cloudflare_estimate_exhaustion_prevents_request_and_preserves_local_fallback(self):
        calls, blocked = [], []
        def mark(provider, category):
            blocked.append((provider, category))
        correct, _ = self.load_correct_text({
            "reserve_cloudflare_daily_neurons": lambda *_: False,
            "cloudflare_correct": lambda *_: calls.append("cloudflare") or ("unexpected", 1, 1),
            "mark_daily_limit": mark,
            "correction_response": lambda *_: ("local safe", "stop"),
        })
        result = correct("source text", {"enabled": True, "provider": "ollama",
            "model": "local", "prompt": ""}, {"enabled": True, "model": "cf",
            "temperature": 0, "max_tokens": 128}, None, None, {}, "live_dictation")
        self.assertEqual(calls, [])
        self.assertEqual(blocked, [("cloudflare", "daily_quota_exhausted")])
        self.assertEqual(result[0], "local safe")
        self.assertEqual(result[3], "local_fallback")

    def test_openrouter_guard_failure_retries_one_different_verified_model(self):
        calls, attempts = [], []
        responses = iter([("bad answer", "vendor/first:free"),
                          ("safe answer", "vendor/alternate:free")])
        def openrouter(_text, model, include_model=False, cancel_callback=None):
            calls.append(model)
            return next(responses)
        def choose(_path, excluded=None):
            if excluded is None:
                return "vendor/first:free"
            self.assertEqual(excluded, {"vendor/first:free"})
            return "vendor/alternate:free"
        def safety(_original, candidate, _reason):
            return "length" if candidate == "bad answer" else None
        correct, _ = self.load_correct_text({
            "openrouter_correct": openrouter,
            "choose_openrouter_fallback_model": choose,
            "correction_safety_reason": safety,
            "record_attempt": lambda _log, stage, provider, model, success, *_args, **_kwargs:
                attempts.append((stage, provider, model, success)),
            "cloudflare_profile": None,
        })
        result = correct("source text", {"enabled": True, "provider": "ollama",
            "model": "local", "prompt": ""}, None, None, None,
            {"correction": {"openrouter_enabled": True}}, "live_dictation",
            attempt_log=attempts)
        self.assertEqual(result[0], "safe answer")
        self.assertEqual(calls, ["vendor/first:free", "vendor/alternate:free"])
        self.assertEqual(attempts[:2], [
            ("correction", "openrouter", "vendor/first:free", False),
            ("correction", "openrouter", "vendor/alternate:free", True)])

    def test_openrouter_error_with_upstream_model_is_attributed_and_all_three_attempts_are_bounded(self):
        calls, attempts = [], []
        def openrouter(_text, model, include_model=False, cancel_callback=None):
            calls.append(model)
            if len(calls) == 1:
                raise multi.ProviderError("openrouter", "invalid_response_missing_content",
                                          "vendor/parser-failed:free")
            raise multi.ProviderError("openrouter", "network_error")
        def choose(_path, excluded=None):
            if excluded is None:
                return "vendor/first:free"
            for candidate in ("vendor/alternate:free", "vendor/third:free"):
                if candidate not in excluded:
                    return candidate
            return None
        correct, _ = self.load_correct_text({
            "openrouter_correct": openrouter,
            "choose_openrouter_fallback_model": choose,
            "cloudflare_correct": lambda *_args: ("cloudflare safe", 1, 1),
            "correction_response": lambda *_args: ("local safe", "stop"),
            "record_attempt": lambda _log, stage, provider, model, success, *_args, **_kwargs:
                attempts.append((provider, model, success)),
            "daily_only_routes": lambda routes: routes,
        })
        result = correct("source text", {"enabled": True, "provider": "ollama",
            "model": "local", "prompt": ""}, {"enabled": True, "model": "cf",
            "temperature": 0, "max_tokens": 100},
            None, None, {"correction": {"openrouter_enabled": True,
                                        "order": ["codex", "claude", "openrouter", "cloudflare"]}}, "live_dictation",
            attempt_log=attempts)
        self.assertEqual(calls, ["vendor/first:free", "vendor/alternate:free", "vendor/third:free"])
        self.assertEqual(attempts[:3], [
            ("openrouter", "vendor/parser-failed:free", False),
            ("openrouter", "vendor/alternate:free", False),
            ("openrouter", "vendor/third:free", False)])
        self.assertEqual(result[0], "cloudflare safe")
        self.assertEqual(result[3], "cloudflare")

    def test_openrouter_without_alternate_and_daily_quota_do_not_retry(self):
        calls = []
        def openrouter(*_args, **_kwargs):
            calls.append("openrouter")
            raise multi.ProviderError("openrouter", "daily_quota_exhausted")
        correct, _ = self.load_correct_text({
            "openrouter_correct": openrouter,
            "choose_openrouter_fallback_model": lambda *_args: calls.append("chooser") or None,
            "daily_only_routes": lambda routes: routes,
            "daily_limit_state": lambda provider=None: None,
            "mark_daily_limit": lambda *_args: None,
            "correction_response": lambda *_args: ("local safe", "stop"),
        })
        result = correct("source text", {"enabled": True, "provider": "ollama",
            "model": "local", "prompt": ""}, None, None, None,
            {"correction": {"openrouter_enabled": True,
                            "order": ["codex", "claude", "openrouter", "cloudflare"]}}, "live_dictation")
        self.assertEqual(calls, ["chooser", "openrouter"])
        self.assertEqual(result[0], "local safe")

    def test_openrouter_without_eligible_models_falls_through_without_retry(self):
        calls = []
        def openrouter(*_args, **_kwargs):
            calls.append("openrouter")
            raise multi.ProviderError("openrouter", "provider_unavailable")
        correct, _ = self.load_correct_text({
            "openrouter_correct": openrouter,
            "choose_openrouter_fallback_model": lambda *_args: None,
            "cloudflare_correct": lambda *_args: ("cloudflare safe", 1, 1),
        })
        result = correct("source text", {"enabled": True, "provider": "ollama",
            "model": "local", "prompt": ""}, {"enabled": True, "model": "cf",
            "temperature": 0, "max_tokens": 100}, None, None,
            {"correction": {"openrouter_enabled": True,
                            "order": ["codex", "claude", "openrouter", "cloudflare"]}}, "live_dictation")
        self.assertEqual(calls, ["openrouter"])
        self.assertEqual(result[0], "cloudflare safe")

    def test_openrouter_other_error_tries_next_unique_historically_successful_model(self):
        calls, downstream_calls = [], []
        def openrouter(_text, model, include_model=False, cancel_callback=None):
            calls.append(model)
            if len(calls) == 1:
                raise multi.ProviderError("openrouter", "other_error", model)
            return "safe answer", model
        def choose(_path, excluded=None):
            if excluded is None:
                return "vendor/first:free"
            self.assertIn("vendor/first:free", excluded)
            return "vendor/next:free" if "vendor/next:free" not in excluded else None
        correct, _ = self.load_correct_text({
            "openrouter_correct": openrouter,
            "choose_openrouter_fallback_model": choose,
            "cloudflare_correct": lambda *_args: downstream_calls.append("cloudflare") or ("unused", 1, 1),
        })
        result = correct("source text", {"enabled": True, "provider": "ollama",
            "model": "local", "prompt": ""}, None, None, None,
            {"correction": {"openrouter_enabled": True, "openrouter_retry_attempts": 2}},
            "live_dictation")
        self.assertEqual(calls, ["vendor/first:free", "vendor/next:free"])
        self.assertEqual(result[0], "safe answer")
        self.assertEqual(downstream_calls, [])

    def test_openrouter_terminal_primary_errors_do_not_start_retry(self):
        for category in ("missing_credentials", "credentials_invalid", "http_401_403",
                         "daily_quota_exhausted", "cancelled_by_user"):
            with self.subTest(category=category):
                requests, chooser_args = [], []
                def openrouter(*_args, **_kwargs):
                    requests.append("request")
                    raise multi.ProviderError("openrouter", category)
                def choose(_path, excluded=None):
                    chooser_args.append(excluded)
                    if excluded is not None:
                        self.fail("terminal primary error must not trigger a fallback model selection")
                    return "vendor/alternate:free"
                correct, _ = self.load_correct_text({
                    "openrouter_correct": openrouter,
                    "choose_openrouter_fallback_model": choose,
                })
                correct("source text", {"enabled": True, "provider": "ollama",
                    "model": "local", "prompt": ""}, None, None, None,
                    {"correction": {"openrouter_enabled": True}}, "live_dictation")
                self.assertEqual(requests, ["request"])
                self.assertEqual(chooser_args, [None])

    def test_openrouter_fallback_stops_on_authentication_failures(self):
        for category in ("missing_credentials", "credentials_invalid", "http_401_403"):
            with self.subTest(category=category):
                calls = []
                def openrouter(_text, model, include_model=False, cancel_callback=None):
                    calls.append(model)
                    if len(calls) == 1:
                        raise multi.ProviderError("openrouter", "other_error", model)
                    raise multi.ProviderError("openrouter", category, model)
                def choose(_path, excluded=None):
                    return "vendor/first:free" if excluded is None else "vendor/next:free"
                correct, _ = self.load_correct_text({
                    "openrouter_correct": openrouter,
                    "choose_openrouter_fallback_model": choose,
                })
                correct("source text", {"enabled": True, "provider": "ollama",
                    "model": "local", "prompt": ""}, None, None, None,
                    {"correction": {"openrouter_enabled": True}}, "live_dictation")
                self.assertEqual(calls, ["vendor/first:free", "vendor/next:free"])


if __name__ == "__main__":
    unittest.main()
