import ast
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
import urllib
import urllib.error
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
DAEMON_PATH = ROOT / "scripts" / "whisper-daemon.py"
SOURCE = DAEMON_PATH.read_text(encoding="utf-8")


def daemon_function(name, **overrides):
    tree = ast.parse(SOURCE)
    node = next(item for item in tree.body
                if isinstance(item, (ast.AsyncFunctionDef, ast.FunctionDef)) and item.name == name)
    namespace = {
        "asyncio": asyncio, "gc": __import__("gc"), "json": json,
        "urllib": urllib, "logger": logging.getLogger("residency-test"),
        "GIGAAM_MODEL_NAME": "GigaAM-v3 e2e_rnnt",
        "GIGAAM_ONNX_BACKENDS": {},
        "GIGAAM_BACKENDS": {"gigaam-v3-e2e-rnnt"},
        "gigaam_worker": None, "gigaam_onnx_worker": None,
    }
    namespace.update(overrides)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DAEMON_PATH), "exec"), namespace)
    return namespace[name], namespace


def load_config_function(config_path):
    tree = ast.parse(SOURCE)
    default = next(node for node in tree.body
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == "DEFAULT" for target in node.targets))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"clone_default", "restore_local_profile_defaults", "config"}]
    namespace = {
        "DEFAULT_CLOUDFLARE_CORRECTOR": "@cf/qwen/qwen3-30b-a3b-fp8",
        "CLOUDFLARE_CORRECTOR_MODELS": ("@cf/qwen/qwen3-30b-a3b-fp8",
                                         "@cf/zai-org/glm-4.7-flash"),
        "MODEL_NEURON_RATES": {"@cf/qwen/qwen3-30b-a3b-fp8": (1, 1),
                               "@cf/zai-org/glm-4.7-flash": (1, 1)},
        "MODELS": {"large-v3", "large-v3-turbo", "medium", "small"},
        "COMPUTES": {"int8_float16", "float16", "int8", "float32"},
        "GIGAAM_BACKENDS": {"gigaam-v3-e2e-rnnt", "gigaam-v3-e2e-ctc-onnx-gpu",
                            "gigaam-v3-ctc-onnx-gpu"},
        "GIGAAM_ONNX_BACKENDS": {},
        "safe_provider_settings": lambda value: value if isinstance(value, dict) else {},
        "CONFIG": Path(config_path), "json": json, "logger": logging.getLogger("config-test"),
        "HOME": Path.home(),
    }
    exec(compile(ast.Module(body=[default, *functions], type_ignores=[]), "daemon-config-test", "exec"), namespace)
    return namespace["config"]


class CloudflareResidencyTests(unittest.TestCase):
    def test_live_dictation_selects_each_enabled_cloudflare_backend_and_files_stay_local(self):
        select, _ = daemon_function("select_job_backends")
        profile = {"cloudflare_asr_enabled": True,
                   "cloudflare_corrector": {"enabled": True, "model": "@cf/qwen/test"}}
        provider, corrector = select("live_dictation", profile)
        self.assertEqual(provider, "cloudflare")
        self.assertTrue(corrector["enabled"])
        self.assertEqual(select("file", profile), ("local", {"enabled": False}))
        self.assertEqual(select("youtube", profile), ("local", {"enabled": False}))
        profile["cloudflare_asr_enabled"] = False
        self.assertEqual(select("live_dictation", profile)[0], "local")

    def test_invalid_local_profile_resets_locals_without_clearing_cloudflare_toggles(self):
        defaults = {key: None for key in ("model", "language", "device", "compute_type", "beam_size",
                   "best_of", "temperature", "vad", "condition_on_previous_text", "initial_prompt",
                   "glossary", "result_mode", "fallback", "corrector", "asr_backend",
                   "gigaam_model_dir")}
        defaults.update({"model": "large-v3", "asr_backend": "faster-whisper",
                         "corrector": {"enabled": False}})
        restore, _ = daemon_function("restore_local_profile_defaults",
            clone_default=lambda: dict(defaults))
        value = dict(defaults, model="invalid", cloudflare_asr_enabled=True,
                     cloudflare_corrector={"enabled": True})
        restored = restore(value)
        self.assertEqual(restored["model"], "large-v3")
        self.assertTrue(restored["cloudflare_asr_enabled"])
        self.assertTrue(restored["cloudflare_corrector"]["enabled"])
        self.assertIn("restore_local_profile_defaults(value)", SOURCE)

    def test_daemon_config_keeps_enabled_cloudflare_backends_when_local_profile_is_invalid(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as config_file:
            defaults = {
                "version": 1, "model": "large-v3", "language": "ru", "device": "cuda",
                "compute_type": "int8_float16", "beam_size": 1, "best_of": 1,
                "temperature": 0.0, "vad": False, "condition_on_previous_text": True,
                "initial_prompt": "", "glossary": [], "result_mode": "clipboard",
                "fallback": {"model": "medium", "device": "cuda", "compute_type": "int8_float16",
                             "allow_cpu": False},
                "cloudflare_asr_enabled": True, "cloudflare_asr_model": "@cf/openai/whisper-large-v3-turbo",
                "cloudflare_asr": {"model": "@cf/openai/whisper-large-v3-turbo", "task": "transcribe",
                    "language": "ru", "vad_filter": False, "beam_size": 1, "initial_prompt": "",
                    "prefix": "", "condition_on_previous_text": True, "no_speech_threshold": 0.6,
                    "compression_ratio_threshold": 2.4, "log_prob_threshold": -1.0,
                    "hallucination_silence_threshold": None},
                "cloudflare_corrector": {"enabled": True, "model": "@cf/qwen/qwen3-30b-a3b-fp8",
                    "temperature": 0.0, "max_tokens": 2048},
                "corrector": {"enabled": True, "provider": "ollama", "model": "qwen-local",
                    "prompt": "", "timeout_seconds": 999, "max_length_ratio": 1.5,
                    "num_gpu": 8, "keep_alive": "5m", "temperature": 0,
                    "num_ctx": 8192, "num_thread": 4},
            }
            json.dump(defaults, config_file)
            config_file.flush()
            value = load_config_function(config_file.name)()
        self.assertTrue(value["cloudflare_asr_enabled"])
        self.assertTrue(value["cloudflare_corrector"]["enabled"])
        self.assertEqual(value["corrector"]["timeout_seconds"], 300)
        defaults["cloudflare_asr"]["beam_size"] = 2
        defaults["cloudflare_corrector"]["model"] = "invalid-cloud-model"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as config_file:
            json.dump(defaults, config_file)
            config_file.flush()
            value = load_config_function(config_file.name)()
        self.assertTrue(value["cloudflare_asr_enabled"])
        self.assertEqual(value["cloudflare_asr"]["beam_size"], 1)
        self.assertTrue(value["cloudflare_corrector"]["enabled"])
        self.assertEqual(value["cloudflare_corrector"]["model"], "@cf/qwen/qwen3-30b-a3b-fp8")

    def test_shell_profile_parser_preserves_cloudflare_toggle_if_local_profile_is_invalid(self):
        wrapper = (ROOT / "scripts/whisper-dictate-v2").read_text(encoding="utf-8")
        start = wrapper.index("<<'PY'\n") + len("<<'PY'\n")
        end = wrapper.index("\nPY\n)", start)
        parser = wrapper[start:end]
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as config_file:
            json.dump({"model": "invalid-local-model", "cloudflare_asr_enabled": True}, config_file)
            config_file.flush()
            result = subprocess.run([sys.executable, "-", config_file.name], input=parser,
                                    text=True, capture_output=True, check=True)
        self.assertIn("model=large-v3", result.stdout)
        self.assertIn("cloudflare_asr_enabled=true", result.stdout)

    def test_unknown_and_file_sources_fail_closed(self):
        select, _ = daemon_function("select_job_backends")
        profile = {"cloudflare_asr_enabled": True,
                   "cloudflare_corrector": {"enabled": True}}
        self.assertEqual(select("unknown", profile), ("local", {"enabled": False}))

    def test_daemon_client_forwards_job_source_to_daemon(self):
        client = (ROOT / "scripts/whisper-daemon-client.sh").read_text(encoding="utf-8")
        self.assertIn('DAEMON_SOURCE="${WHISPER_JOB_SOURCE:-}"', client)
        self.assertIn('"source":os.environ["DAEMON_SOURCE"]', client)

    def test_delayed_whisper_release_uses_fake_clock_and_honors_new_lease(self):
        events = []
        async def fake_sleep(delay): events.append(("sleep", delay))
        async def fake_release(): events.append(("release",))
        fn, ns = daemon_function("delayed_whisper_release", asyncio=SimpleNamespace(sleep=fake_sleep),
                                 release=fake_release, model_lease=0)
        asyncio.run(fn())
        self.assertEqual(events, [("sleep", 2.0), ("release",)])
        events.clear()
        async def acquire_before_timer(delay):
            events.append(("sleep", delay)); ns["model_lease"] += 1
        ns["asyncio"] = SimpleNamespace(sleep=acquire_before_timer)
        asyncio.run(fn())
        self.assertEqual(events, [("sleep", 2.0)])

    def test_qwen_unload_waits_two_seconds_after_success(self):
        events = []
        async def fake_sleep(delay): events.append(("sleep", delay))
        async def to_thread(function, *args): return function(*args)
        fn, _ = daemon_function("delayed_qwen_release", asyncio=SimpleNamespace(sleep=fake_sleep, to_thread=to_thread),
                                unload_corrector_model=lambda name: events.append(("unload", name)), qwen_lease=0)
        asyncio.run(fn("qwen-local"))
        self.assertEqual(events, [("sleep", 2.0), ("unload", "qwen-local")])

    def test_cloudflare_asr_releases_models_and_forces_local_fallback(self):
        run_start = SOURCE.index("async def run(job, asr_only=True):")
        run = SOURCE[run_start:SOURCE.index("\nasync def correction_stage", run_start)]
        cloud_call = run.index('elif route == "cloudflare":')
        local_load = run.index("cold = await prepare_local_asr(value)")
        remote_release = run.index('await prepare_remote_models(value["corrector"]["model"])')
        self.assertIn("multi_asr_chain", run)
        self.assertIn('elif route == "google":', run)
        self.assertLess(remote_release, cloud_call)
        self.assertLess(cloud_call, local_load)
        self.assertNotIn('job["cloudflare_fallback_local"]', run)
        self.assertIn("await ensure(retry)", run)
        self.assertIn("await ensure(retry)", run)

    def test_cloudflare_corrector_releases_whisper_after_asr(self):
        run_start = SOURCE.index("async def run(job, asr_only=True):")
        run = SOURCE[run_start:SOURCE.index("\nasync def correction_stage", run_start)]
        correction_start = SOURCE.index("async def correction_stage(job, asr_result):")
        correction = SOURCE[correction_start:SOURCE.index("\nasync def finish_job", correction_start)]
        asr_end = run.index("elapsed = time.monotonic() - t0")
        correction_release = correction.index("if any(route != \"local\" for route in correction_routes)")
        local_qwen = correction.index("correct_text_sync, text, corrector")
        self.assertLess(asr_end, run.index("if asr_only:"))
        self.assertLess(correction_release, local_qwen)
        self.assertIn('select_job_backends(job.get("source"), value)', correction)
        self.assertIn("multi_correction_chain", correction)
        self.assertIn('await prepare_remote_models(corrector["model"])', correction)
        self.assertIn("delayed_qwen_release(corrector[\"model\"])", correction)

    def test_cloudflare_correction_error_does_not_force_qwen_if_local_toggle_is_off(self):
        start = SOURCE.index("def correct_text_sync(")
        end = SOURCE.index("\ndef large_corrector_profile", start)
        body = SOURCE[start:end]
        self.assertNotIn('corrector["enabled"] = True', body)
        self.assertIn("A disabled local corrector must stay unloaded", body)
        self.assertIn('"local_fallback" if fallback_stage else "local"', body)

    def test_remote_preparation_releases_whisper_then_ollama_qwen(self):
        events = []
        async def to_thread(function, *args): return function(*args)
        async def release_whisper(): events.append("whisper")
        def unload_qwen(model): events.append(("qwen", model))
        prepare, _ = daemon_function("prepare_remote_models", asyncio=SimpleNamespace(to_thread=to_thread),
                                     release=release_whisper,
                                     unload_corrector_model=unload_qwen)
        asyncio.run(prepare("qwen3.5:4b-q4_K_M"))
        self.assertEqual(events, ["whisper", ("qwen", "qwen3.5:4b-q4_K_M")])

    def test_local_asr_unloads_qwen_then_ensures_whisper(self):
        events = []
        async def to_thread(function, *args): return function(*args)
        async def ensure(_profile, **_kwargs): events.append("whisper"); return True
        def unload(_model): events.append("qwen")
        prepare, _ = daemon_function("prepare_local_asr", asyncio=SimpleNamespace(to_thread=to_thread),
                                     ensure=ensure, unload_corrector_model=unload,
                                     local_corrector_model_names=lambda _profile: ["qwen-local", "qwen-fallback"])
        self.assertTrue(asyncio.run(prepare({"corrector": {"model": "qwen-local"}})))
        self.assertEqual(events, ["qwen", "qwen", "whisper"])

    def test_gigaam_local_asr_unloads_qwen_before_switching_asr(self):
        events = []
        async def to_thread(function, *args): return function(*args)
        async def ensure(_profile, **_kwargs): events.append("gigaam"); return False
        def unload(_model): events.append("qwen")
        prepare, _ = daemon_function("prepare_local_asr", asyncio=SimpleNamespace(to_thread=to_thread),
                                     ensure=ensure, unload_corrector_model=unload,
                                     local_corrector_model_names=lambda _profile: ["qwen-local", "qwen-fallback"])
        self.assertFalse(asyncio.run(prepare({"asr_backend": "gigaam-v3-e2e-rnnt",
                                              "corrector": {"model": "qwen-local"}})))
        self.assertEqual(events, ["qwen", "qwen", "gigaam"])

    def test_whisper_load_error_propagates_without_repeating_load(self):
        loads = []
        async def to_thread(function, *args, **kwargs): return function(*args, **kwargs)
        async def release_model(): return None
        def fail_load(*_args, **_kwargs):
            loads.append(True)
            raise RuntimeError("synthetic load failure")
        ensure, _ = daemon_function("ensure", asyncio=SimpleNamespace(to_thread=to_thread),
            model=None, model_profile=None, model_lease=0, release=release_model, WhisperModel=fail_load,
            local_model_source=lambda name: name,
            logger=logging.getLogger("residency-test"))
        profile = {"model": "large-v3", "device": "cuda", "compute_type": "int8_float16"}
        with self.assertRaisesRegex(RuntimeError, "synthetic load failure"):
            asyncio.run(ensure(profile))
        self.assertEqual(loads, [True])

    def test_ollama_unload_uses_keep_alive_zero_and_reports_errors(self):
        unload, _ = daemon_function("unload_corrector_model", urllib=urllib,
                                    logger=logging.getLogger("residency-test"))
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self): return b"{}"
        with patch("urllib.request.urlopen", return_value=Response()) as request:
            self.assertTrue(unload("qwen-local"))
        self.assertEqual(json.loads(request.call_args.args[0].data),
                         {"model": "qwen-local", "keep_alive": 0})
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("private-marker")):
            self.assertFalse(unload("qwen-local"))

    def test_large_whisper_cpu_qwen_preserves_user_keep_alive(self):
        profile, _ = daemon_function("large_corrector_profile")
        corrector, cpu_only = profile({"model": "large-v3", "corrector": {
            "num_gpu": 8, "keep_alive": "5m"}})
        self.assertTrue(cpu_only)
        self.assertEqual(corrector["num_gpu"], 0)
        self.assertEqual(corrector["keep_alive"], "5m")
        self.assertIn('"keep_alive": corrector["keep_alive"]', SOURCE)

    def test_fallback_whisper_is_retained_when_no_later_remote_stage_needs_it(self):
        start = SOURCE.index("async def run(job, asr_only=True):")
        run = SOURCE[start:SOURCE.index("\nasync def correction_stage", start)]
        start = SOURCE.index("async def correction_stage(job, asr_result):")
        correction = SOURCE[start:SOURCE.index("\nasync def finish_job", start)]
        self.assertIn("cold = await prepare_local_asr(value)", run)
        self.assertIn("elif corrector.get(\"enabled\") and not large_cpu_corrector", correction)
        self.assertNotIn("await release()\n            text = await asyncio.to_thread(transcribe_sync", run)
        self.assertIn("delayed_qwen_release(corrector[\"model\"])", correction)
        self.assertIn('"keep_alive": corrector["keep_alive"]', SOURCE)

    def test_local_qwen_stays_resident_when_ram_is_safe_or_unknown(self):
        for available in (8 * 1024**3, None):
            calls = []
            release, _ = daemon_function("release_qwen_under_memory_pressure",
                available_memory_bytes=lambda: available,
                unload_corrector_model=lambda model: calls.append(model) or True,
                logger=logging.getLogger("residency-test"))
            self.assertTrue(release({"model": "qwen-local"}))
            self.assertEqual(calls, [])

    def test_low_ram_releases_qwen_and_large_profile_avoids_vram_competition(self):
        calls = []
        release, _ = daemon_function("release_qwen_under_memory_pressure",
            available_memory_bytes=lambda: 5 * 1024**3,
            unload_corrector_model=lambda model: calls.append(model) or True,
            logger=logging.getLogger("residency-test"))
        self.assertTrue(release({"model": "qwen-local"}))
        self.assertEqual(calls, ["qwen-local"])
        self.assertIn('corrector["num_gpu"] = 0', SOURCE)


if __name__ == "__main__":
    unittest.main()
