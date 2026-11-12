import ast
import importlib.util
import io
import logging
import sys
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "verification"))
from test_multi_provider import multi

MODULE_PATH = ROOT / "scripts" / "cloudflare_corrector.py"
spec = importlib.util.spec_from_file_location("cloudflare_corrector", MODULE_PATH)
cloudflare_corrector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloudflare_corrector)


def daemon_function(name, **overrides):
    source = (ROOT / "scripts" / "whisper-daemon.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    safety = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "correction_is_safe")
    namespace = {
        "time": time, "re": __import__("re"), "difflib": __import__("difflib"), "urllib": __import__("urllib.error"),
        "json": __import__("json"), "os": __import__("os"),
        "logger": logging.getLogger("test-cloudflare-corrector"),
        "CloudflareError": cloudflare_corrector.CloudflareError,
        "normalize_asr_text": lambda text: text.strip(),
        "finalize_text": lambda text: text if text.endswith(".") else text + ".",
        "CORRECTOR_INSTRUCTION": "Correct text.", "RETRY_INSTRUCTION": "Retry.",
        "correction_chunks": lambda text: [text],
        "correction_response": lambda text, _profile, _instruction: (text, "stop"),
        "ollama_loaded_device": lambda _model, requested: requested,
        "unload_corrector_model": lambda _model: True,
        "correction_safety_reason": lambda _source, corrected, _reason: None if corrected.strip() else "invalid",
        "multi_correction_chain": lambda _source, _settings, cf_enabled: ["cloudflare", "local"] if cf_enabled else ["local"],
        "safe_provider_settings": multi.safe_settings,
        "CLOUDFLARE_CORRECTOR_MODELS": (cloudflare_corrector.DEFAULT_MODEL,),
        "cloudflare_correction_model_order": lambda _metrics, models, _preferred=None: list(models),
        "GIGAAM_BACKENDS": {"gigaam-v3-e2e-rnnt", "gigaam-v3-e2e-ctc-onnx-gpu",
                            "gigaam-v3-ctc-onnx-gpu"},
        "GIGAAM_ONNX_BACKENDS": {},
        "ProviderError": type("ProviderError", (Exception,), {"__init__": lambda self, provider, category: (Exception.__init__(self, category), setattr(self, "provider", provider), setattr(self, "category", category))[-1]}),
        "safe_provider_error_category": lambda error: getattr(error, "category", "other_error"),
        "cloudflare_correct": cloudflare_corrector.correct,
        "correction_token_counts": cloudflare_corrector.token_counts,
        "correction_neurons": cloudflare_corrector.estimate_neurons,
        "estimate_cloudflare_correction_neurons": lambda *_args: 1,
        "reserve_cloudflare_daily_neurons": lambda *_args: True,
        "METRICS": ROOT / "synthetic-metrics.jsonl",
        "daily_limit_state": lambda provider=None: None, "mark_daily_limit": lambda *_args: None,
        "daily_only_routes": lambda routes: list(routes),
        "record_attempt": lambda *_args, **_kwargs: None,
    }
    namespace.update(overrides)
    exec(compile(ast.Module(body=[safety, node], type_ignores=[]), str(ROOT / "scripts" / "whisper-daemon.py"), "exec"), namespace)
    return namespace[name], namespace


class CloudflareCorrectorTests(unittest.TestCase):
    def test_disabled_local_corrector_stays_unloaded_after_cloudflare_failure(self):
        class ProviderError(Exception):
            def __init__(self, provider, category, *_args):
                super().__init__(category)
                self.provider = provider
                self.category = category

        class CloudflareError(Exception):
            def __init__(self, *_args):
                super().__init__("unavailable")
                self.category = "network_error"

        correction = {"enabled": True, "local_models": [],
                      "codex_model": "codex", "claude_model": "claude",
                      "codex_retry_attempts": 0, "claude_retry_attempts": 0,
                      "groq_primary_retry_attempts": 0, "groq_fallback_retry_attempts": 0,
                      "openrouter_retry_attempts": 0, "google_retry_attempts": 0}
        calls = []
        def raise_cloudflare(*_args):
            calls.append("cloudflare")
            raise CloudflareError()
        correct, _namespace = daemon_function("correct_text_sync",
            safe_provider_settings=lambda _value: {"correction": correction},
            multi_correction_chain=lambda *_args: ["cloudflare", "local"],
            daily_only_routes=lambda routes: list(routes),
            daily_limit_state=lambda **_kwargs: None,
            CLOUDFLARE_CORRECTOR_MODELS=("cf-model",),
            cloudflare_correction_model_order=lambda *_args: ["cf-model"],
            CloudflareError=CloudflareError, ProviderError=ProviderError,
            safe_provider_error_category=lambda error: getattr(error, "category", "other_error"),
            estimate_cloudflare_correction_neurons=lambda *_args: 1,
            reserve_cloudflare_daily_neurons=lambda *_args: True,
            cloudflare_correct=raise_cloudflare,
            cloudflare_failure_callback=None,
            record_attempt=lambda *_args, **_kwargs: None,
            logger=__import__("logging").getLogger("disabled-local-corrector-test"),
            CORRECTOR_INSTRUCTION="", RETRY_INSTRUCTION="",
            METRICS=ROOT / "synthetic-metrics.jsonl")
        result = correct("raw text", {"enabled": False, "provider": "ollama",
            "model": "qwen-local", "prompt": "", "num_gpu": 8},
            cloudflare_profile={"enabled": True, "model": "cf-model", "priority_model": "auto",
                "retry_attempts": 0, "max_tokens": 128, "temperature": 0},
            provider_settings={"correction": correction})
        self.assertEqual(result[0], "raw text.")
        self.assertEqual(result[3], "local_disabled")
        self.assertEqual(calls, ["cloudflare"])

    def test_guard_rejection_reasons_are_content_free(self):
        source = (ROOT / "scripts/whisper-daemon.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "correction_safety_reason")
        namespace = {"re": __import__("re"), "difflib": __import__("difflib")}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "guard-reasons", "exec"), namespace)
        reason = namespace["correction_safety_reason"]
        self.assertEqual(reason("extraordinarilylongword", "shortword", None), "length")
        self.assertEqual(reason("supercalifragilisticexpialidocious a b", "supercalifragilisticexpialidocious a", None), "word_coverage_order")
        self.assertEqual(reason("alpha " * 20, "bravo " * 20, None), "word_coverage_order")
        self.assertEqual(reason("alpha beta gamma", "alpha beta unrelated", None), "tail")
        self.assertEqual(reason("same words", None, None), "invalid")
        repeated = "Сегодня проверили безопасные метрики, очередь задач пуста, а выбранная модель фиксируется без текста диктовки. "
        self.assertEqual(reason(repeated * 2, repeated.strip(), None), "length")

    def test_default_model_and_supported_parameters(self):
        self.assertEqual(cloudflare_corrector.DEFAULT_MODEL, "@cf/qwen/qwen3-30b-a3b-fp8")
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self, _limit): return b'{"success":true,"result":{"response":"Sample sentence.","usage":{"prompt_tokens":12,"completion_tokens":4}}}'
        with patch.object(cloudflare_corrector, "read_credentials", return_value=("account", "token")), \
                patch.object(cloudflare_corrector.urllib.request, "urlopen", return_value=Response()) as request:
            result = cloudflare_corrector.correct("Sample sentence", temperature=0.2, max_tokens=512)
        body = request.call_args.args[0].data.decode()
        self.assertEqual(result, ("Sample sentence.", 12, 4))
        self.assertEqual(request.call_args.kwargs["timeout"], 90)
        self.assertIn('"temperature": 0.2', body)
        self.assertIn('"max_tokens": 512', body)
        self.assertNotIn("Sample sentence", str(request.call_args.args[0].headers))

    def test_parse_response_rejects_empty_invalid_and_oversized_shapes(self):
        for payload in ({}, {"success": False}, {"result": {"response": "  "}},
                        {"result": {"choices": []}}):
            with self.subTest(payload=payload), self.assertRaises(cloudflare_corrector.CloudflareError):
                cloudflare_corrector.parse_response(payload)
        self.assertEqual(cloudflare_corrector.parse_response({"result": {"choices": [
            {"message": {"content": " corrected "}}]}})[0], "corrected")

    def test_http_network_timeout_and_invalid_response_fail_without_sensitive_details(self):
        marker = "private-response-token-marker"
        for status in (401, 403, 429, 500):
            error = urllib.error.HTTPError("https://invalid", status, "failure", {}, io.BytesIO(marker.encode()))
            with self.subTest(status=status), patch.object(cloudflare_corrector, "read_credentials", return_value=("account", "token")), \
                    patch.object(cloudflare_corrector.urllib.request, "urlopen", side_effect=error):
                with self.assertRaises(cloudflare_corrector.CloudflareError) as raised:
                    cloudflare_corrector.correct("synthetic text")
                self.assertNotIn(marker, str(raised.exception))
                self.assertIn(str(status), str(raised.exception))
        for failure in (urllib.error.URLError(marker), TimeoutError(marker)):
            with self.subTest(failure=type(failure).__name__), patch.object(cloudflare_corrector, "read_credentials", return_value=("account", "token")), \
                    patch.object(cloudflare_corrector.urllib.request, "urlopen", side_effect=failure):
                with self.assertRaises(cloudflare_corrector.CloudflareError) as raised:
                    cloudflare_corrector.correct("synthetic text")
                self.assertNotIn(marker, str(raised.exception))

    def test_usage_tokens_and_approximation(self):
        self.assertEqual(cloudflare_corrector.token_counts("abcd", "xy", 10, 2), (10, 2, False))
        self.assertEqual(cloudflare_corrector.token_counts("abcdefgh", "abcd", None, None), (2, 1, True))
        self.assertGreater(cloudflare_corrector.estimate_neurons(10, 2), 0)
        self.assertNotEqual(cloudflare_corrector.estimate_neurons(100, 50),
                            cloudflare_corrector.estimate_neurons(100, 50, "@cf/zai-org/glm-4.7-flash"))

    def test_cloudflare_failure_runs_local_corrector_when_enabled(self):
        correct_text, namespace = daemon_function("correct_text_sync")
        namespace["cloudflare_correct"] = lambda *_args: (_ for _ in ()).throw(CloudflareTestError("failed"))
        result = correct_text("Synthetic complete sentence.", {"enabled": True, "provider": "ollama", "prompt": "", "model": "qwen-local", "num_gpu": 0},
                              {"enabled": True, "model": cloudflare_corrector.DEFAULT_MODEL,
                               "temperature": 0.0, "max_tokens": 512})
        self.assertEqual(result[0], "Synthetic complete sentence.")
        self.assertEqual(result[3], "local_fallback")
        self.assertIsNone(result[4])

    def test_enabled_cloudflare_corrector_is_used_when_local_toggle_is_also_enabled(self):
        correct_text, namespace = daemon_function("correct_text_sync")
        namespace["cloudflare_correct"] = lambda *_args: ("Synthetic complete sentence.", 4, 3)
        result = correct_text("Synthetic complete sentence.",
            {"enabled": True, "provider": "ollama", "prompt": "", "model": "qwen-local", "num_gpu": 0},
            {"enabled": True, "model": cloudflare_corrector.DEFAULT_MODEL,
             "temperature": 0.0, "max_tokens": 512})
        self.assertEqual(result[3], "cloudflare")
        self.assertEqual(result[0], "Synthetic complete sentence.")

    def test_all_cloud_failures_fall_back_to_local_when_enabled(self):
        correct_text, namespace = daemon_function("correct_text_sync")
        failures = [cloudflare_corrector.CloudflareError("401"),
                    cloudflare_corrector.CloudflareError("403"),
                    cloudflare_corrector.CloudflareError("429"),
                    cloudflare_corrector.CloudflareError("network/timeout"),
                    cloudflare_corrector.CloudflareError("invalid response"),
                    cloudflare_corrector.CloudflareError("empty response")]
        for failure in failures:
            with self.subTest(failure=str(failure)):
                def fail_cloud(*_args, error=failure):
                    raise error
                namespace["cloudflare_correct"] = fail_cloud
                result = correct_text("Synthetic complete sentence.",
                    {"enabled": True, "provider": "ollama", "prompt": "", "model": "qwen-local", "num_gpu": 0},
                    {"enabled": True, "model": cloudflare_corrector.DEFAULT_MODEL,
                     "temperature": 0.0, "max_tokens": 512})
                self.assertEqual(result[3], "local_fallback")
                self.assertEqual(result[0], "Synthetic complete sentence.")

    def test_provider_switches_are_independent_and_old_profiles_default_off(self):
        source = (ROOT / "scripts" / "whisper-tray.py").read_text(encoding="utf-8")
        hotkey = (ROOT / "scripts" / "whisper-dictate-v2").read_text(encoding="utf-8")
        self.assertIn('value["cloudflare_asr_enabled"] = item.get_active()', source)
        self.assertIn('profile.update({"enabled": item.get_active()', source)
        self.assertIn('get("cloudflare_asr_enabled", load_voice_config().get("cloudflare_workers_ai_enabled", False))', source)
        self.assertIn('"cloudflare_asr_enabled":False', hotkey)
        self.assertIn('"cloudflare_workers_ai_enabled", False', hotkey)
        daemon = (ROOT / "scripts" / "whisper-daemon.py").read_text(encoding="utf-8")
        self.assertIn('"cloudflare_asr_enabled": False', daemon)
        self.assertIn('"enabled": False, "model": DEFAULT_CLOUDFLARE_CORRECTOR', daemon)

    def test_unified_model_unload_and_backend_indicator_marks_cloud_only(self):
        source = (ROOT / "scripts" / "whisper-tray.py").read_text(encoding="utf-8")
        self.assertIn('models_unload_item = primary_menu_item("Выгрузить все модели"', source)
        self.assertIn('menu.append(models_unload_item)', source)
        self.assertIn('def processing_status_label(model, stage, icon_key, icon_label=None):', source)
        daemon = (ROOT / "scripts" / "whisper-daemon.py").read_text(encoding="utf-8")
        self.assertIn('"corrector_backend":correction_backend', daemon)
        self.assertIn('"device":"cloudflare" if used_cloudflare or correction_backend == "cloudflare"', daemon)

    def test_job_indicator_switches_to_cloud_then_local_fallback(self):
        with __import__("tempfile").TemporaryDirectory() as directory:
            jobs = Path(directory)
            job_id = "123456"
            (jobs / f"{job_id}.json").write_text('{"state":"processing","device":"cuda"}', encoding="utf-8")
            update, _namespace = daemon_function("update_job_backend", JOBS=jobs,
                DEFAULT_CLOUDFLARE_CORRECTOR=cloudflare_corrector.DEFAULT_MODEL)
            profile = {"device": "cuda", "model": "large-v3"}
            update(job_id, "cloudflare", profile, cloudflare_corrector.DEFAULT_MODEL)
            cloud = __import__("json").loads((jobs / f"{job_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(cloud["device"], "cloudflare")
            update(job_id, "local", profile)
            local = __import__("json").loads((jobs / f"{job_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(local["device"], "cuda")
            self.assertEqual(local["model"], "large-v3")


class CloudflareTestError(Exception):
    pass


if __name__ == "__main__":
    unittest.main()
