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
MODULE_PATH = ROOT / "scripts" / "cloudflare_corrector.py"
spec = importlib.util.spec_from_file_location("cloudflare_corrector", MODULE_PATH)
cloudflare_corrector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloudflare_corrector)


def daemon_function(name, **overrides):
    source = (ROOT / "scripts" / "whisper-daemon.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {
        "time": time, "re": __import__("re"), "difflib": __import__("difflib"),
        "json": __import__("json"), "os": __import__("os"),
        "logger": logging.getLogger("test-cloudflare-corrector"),
        "CloudflareError": cloudflare_corrector.CloudflareError,
        "normalize_asr_text": lambda text: text.strip(),
        "finalize_text": lambda text: text if text.endswith(".") else text + ".",
        "CORRECTOR_INSTRUCTION": "Correct text.", "RETRY_INSTRUCTION": "Retry.",
        "correction_chunks": lambda text: [text],
        "correction_response": lambda text, _profile, _instruction: (text, "stop"),
        "correction_is_safe": lambda _source, corrected, _reason: bool(corrected.strip()),
        "cloudflare_correct": cloudflare_corrector.correct,
        "correction_token_counts": cloudflare_corrector.token_counts,
        "correction_neurons": cloudflare_corrector.estimate_neurons,
    }
    namespace.update(overrides)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "scripts" / "whisper-daemon.py"), "exec"), namespace)
    return namespace[name], namespace


class CloudflareCorrectorTests(unittest.TestCase):
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

    def test_cloudflare_failure_automatically_runs_local_corrector(self):
        correct_text, namespace = daemon_function("correct_text_sync")
        namespace["cloudflare_correct"] = lambda *_args: (_ for _ in ()).throw(CloudflareTestError("failed"))
        result = correct_text("Synthetic complete sentence.", {"enabled": False, "provider": "ollama", "prompt": ""},
                              {"enabled": True, "model": cloudflare_corrector.DEFAULT_MODEL,
                               "temperature": 0.0, "max_tokens": 512})
        self.assertEqual(result[0], "Synthetic complete sentence.")
        self.assertEqual(result[3], "local_fallback")
        self.assertIsNone(result[4])

    def test_enabled_cloudflare_corrector_is_used_when_local_toggle_is_also_enabled(self):
        correct_text, namespace = daemon_function("correct_text_sync")
        namespace["cloudflare_correct"] = lambda *_args: ("Synthetic complete sentence.", 4, 3)
        result = correct_text("Synthetic complete sentence.",
            {"enabled": True, "provider": "ollama", "prompt": ""},
            {"enabled": True, "model": cloudflare_corrector.DEFAULT_MODEL,
             "temperature": 0.0, "max_tokens": 512})
        self.assertEqual(result[3], "cloudflare")
        self.assertEqual(result[0], "Synthetic complete sentence.")

    def test_all_cloud_failures_fall_back_to_local(self):
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
                    {"enabled": False, "provider": "ollama", "prompt": ""},
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

    def test_cloudflare_menu_is_top_level_and_backend_indicator_marks_cloud_only(self):
        source = (ROOT / "scripts" / "whisper-tray.py").read_text(encoding="utf-8")
        self.assertIn('menu.append(cloudflare_menu_item)\nmenu.append(whisper_settings_item)', source)
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
