import ast
import asyncio
import json
import logging
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
    }
    namespace.update(overrides)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DAEMON_PATH), "exec"), namespace)
    return namespace[name], namespace


class CloudflareResidencyTests(unittest.TestCase):
    def test_unknown_and_file_sources_fail_closed(self):
        self.assertIn("source == \"live_dictation\" and config()[\"cloudflare_asr_enabled\"]", SOURCE)

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
        run = SOURCE[SOURCE.index("async def run(job):"):SOURCE.index("\nasync def worker", SOURCE.index("async def run(job):"))]
        cloud_call = run.index("await cloudflare_first(")
        local_load = run.index("cold = await prepare_local_asr(value)")
        self.assertIn("local_transcribe, True)", run)
        self.assertLess(local_load, cloud_call)
        self.assertNotIn('job["cloudflare_fallback_local"]', run)
        self.assertIn("await ensure(retry)", run)
        self.assertIn("await ensure(retry)", run)

    def test_cloudflare_corrector_releases_whisper_after_asr(self):
        run = SOURCE[SOURCE.index("async def run(job):"):SOURCE.index("\nasync def worker", SOURCE.index("async def run(job):"))]
        asr_end = run.index("elapsed = time.monotonic() - t0")
        correction_release = run.index("if provider == \"cloudflare\" and value[\"cloudflare_corrector\"].get(\"enabled\")")
        local_qwen = run.index("correct_text_sync, text, corrector")
        self.assertLess(asr_end, correction_release)
        self.assertLess(correction_release, local_qwen)
        self.assertIn("delayed_qwen_release(corrector[\"model\"])", run)

    def test_cloudflare_correction_error_forces_local_qwen_if_local_toggle_is_off(self):
        start = SOURCE.index("def correct_text_sync(")
        end = SOURCE.index("\ndef large_corrector_profile", start)
        body = SOURCE[start:end]
        self.assertIn('corrector["enabled"] = True', body)
        self.assertIn('"local_fallback" if cloudflare_fallback else "local"', body)

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
        async def ensure(_profile): events.append("whisper"); return True
        def unload(_model): events.append("qwen")
        prepare, _ = daemon_function("prepare_local_asr", asyncio=SimpleNamespace(to_thread=to_thread),
                                     ensure=ensure, unload_corrector_model=unload)
        self.assertTrue(asyncio.run(prepare({"corrector": {"model": "qwen-local"}})))
        self.assertEqual(events, ["qwen", "whisper"])

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
        run = SOURCE[SOURCE.index("async def run(job):"):SOURCE.index("\nasync def worker", SOURCE.index("async def run(job):"))]
        self.assertIn("cold = await prepare_local_asr(value)", run)
        self.assertIn("elif corrector.get(\"enabled\") and not large_cpu_corrector", run)
        self.assertNotIn("await release()\n        text = await asyncio.to_thread(transcribe_sync", run)
        self.assertIn("delayed_qwen_release(corrector[\"model\"])", run)
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
