import ast
import asyncio
import logging
import tempfile
import unittest
from pathlib import Path
import time

from verification.test_multi_provider import multi


ROOT = Path(__file__).parents[1]
DAEMON = ROOT / "scripts" / "whisper-daemon.py"


def load_workers(namespace):
    tree = ast.parse(DAEMON.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                 and node.name in {"worker", "correction_worker"}]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(DAEMON), "exec"), namespace)
    return namespace["worker"], namespace["correction_worker"]


def load_run(namespace):
    tree = ast.parse(DAEMON.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.AsyncFunctionDef)
                and item.name == "run")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DAEMON), "exec"), namespace)
    return namespace["run"]


def load_correct_text(namespace):
    tree = ast.parse(DAEMON.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == "correct_text_sync")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DAEMON), "exec"), namespace)
    return namespace["correct_text_sync"]


class MockProviderError(Exception):
    def __init__(self, _provider, category, *_args):
        super().__init__(category)
        self.category = category


class InlineAsyncio:
    """Avoid thread-pool lifecycle in AST-extracted offline run() tests."""
    create_task = staticmethod(asyncio.create_task)

    @staticmethod
    async def to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)


class DaemonPipelineTests(unittest.TestCase):
    def _run_clean(self, coroutine):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(asyncio.wait_for(coroutine, timeout=2))
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            return result
        finally:
            loop.close()

    def test_local_asr_chain_uses_second_backend_after_prepare_error(self):
        with tempfile.NamedTemporaryFile() as wav:
            attempted = []

            async def prepare(profile, fallback_on_backend_error=True):
                attempted.append(("prepare", profile["asr_backend"]))
                if profile["asr_backend"] == "faster-whisper":
                    raise RuntimeError("mock init failure")
                return False

            namespace = self._asr_namespace(wav.name, attempted, prepare,
                lambda _wav, profile, _marker: "recognized" if profile["asr_backend"] != "faster-whisper" else "unused")
            result = self._run_clean(load_run(namespace)(namespace["job"], asr_only=True))
            self.assertEqual(result["text"], "recognized")
            self.assertEqual(result["fallback"], "local_model_fallback")
            self.assertEqual([backend for _kind, backend in attempted],
                             ["faster-whisper", "gigaam-v3-e2e-rnnt"])

    def test_local_asr_chain_uses_second_backend_after_empty_transcript(self):
        with tempfile.NamedTemporaryFile() as wav:
            attempted = []

            async def prepare(profile, fallback_on_backend_error=True):
                attempted.append(("prepare", profile["asr_backend"]))
                return False

            def transcribe(_wav, profile, _marker):
                backend = profile["asr_backend"]
                attempted.append(("transcribe", backend))
                return "  \n" if backend == "faster-whisper" else "recognized by fallback"

            namespace = self._asr_namespace(wav.name, attempted, prepare, transcribe)
            result = self._run_clean(load_run(namespace)(namespace["job"], asr_only=True))
            self.assertEqual(result["text"], "recognized by fallback")
            self.assertEqual(result["fallback"], "local_model_fallback")
            self.assertEqual(attempted, [("prepare", "faster-whisper"),
                                         ("transcribe", "faster-whisper"),
                                         ("prepare", "gigaam-v3-e2e-rnnt"),
                                         ("transcribe", "gigaam-v3-e2e-rnnt")])

    def _asr_namespace(self, wav_path, attempted, prepare, transcribe):
        async def delayed_release(*_args, **_kwargs):
            return None

        def local_profile(profile):
            backend = profile["asr_backend"]
            return {"model": backend, "device": "cpu"}
        profile = {"model": "faster-whisper", "device": "cpu", "compute_type": "int8",
                   "asr_backend": "faster-whisper", "cloudflare_corrector": {"model": "unused"},
                   "corrector": {"enabled": False, "model": "unused", "num_gpu": 0},
                   "multi_provider": {"asr": {"local_models": [
                       "faster-whisper", "gigaam-v3-e2e-rnnt"]}}, "cloudflare_asr": {}}
        return {"asyncio": InlineAsyncio, "os": __import__("os"), "time": time,
            "logger": logging.getLogger("daemon-local-chain-test"),
            "safe_provider_settings": multi.safe_settings,
            "multi_asr_chain": lambda *_args: ["local"], "daily_only_routes": list,
            "daily_limit_state": lambda **_kwargs: None,
            "select_job_backends": lambda *_args: ("local", {"enabled": False}),
            "local_asr_status_profile": local_profile,
            "update_job_backend": lambda *_args, **_kwargs: None,
            "prepare_local_asr": prepare, "transcribe_sync": transcribe,
            "correction_cancelled": lambda *_args: False,
            "RECOVERY_CANCEL": Path(tempfile.gettempdir()) / "no-cancel-markers-for-test",
            "GIGAAM_BACKENDS": set(), "GIGAAM_ONNX_BACKENDS": set(),
            "gigaam_worker": None, "gigaam_onnx_worker": None,
            "correction_pending": 0, "active_asr_model_name": None,
            "active_asr_device": None, "active_asr_error_category": None,
            "active_asr_backend": None, "record_attempt": lambda *_args, **_kwargs: None,
            "safe_provider_error_category": lambda error: getattr(error, "category", "other_error"),
            "ProviderError": MockProviderError,
            "remove_subtitle_credit_tail": lambda text: text,
            "delayed_whisper_release": delayed_release,
            "METRICS": Path("/tmp/nonexistent-daemon-test-metrics.jsonl"),
            "job": {"job_id": "offline-test", "source": "file", "provider": "local",
                    "profile": profile, "wav": wav_path}}

    def test_local_corrector_chain_skips_candidate_after_unsafe_retries(self):
        calls = []
        attempts = []

        def response(_chunk, corrector, _instruction):
            model = corrector["model"]
            calls.append(model)
            if model == "qwen3.5:4b-q4_K_M":
                return "", "stop"
            return "corrected", "stop"

        namespace = {
            "time": time, "json": __import__("json"), "urllib": __import__("urllib"),
            "logger": logging.getLogger("daemon-local-corrector-test"),
            "safe_provider_settings": multi.safe_settings,
            "multi_correction_chain": lambda *_args: ["local"],
            "daily_only_routes": list, "daily_limit_state": lambda **_kwargs: None,
            "normalize_asr_text": lambda text: text.strip(), "finalize_text": lambda text: text,
            "correction_chunks": lambda text: [text], "correction_response": response,
            "correction_is_safe": lambda _source, candidate, _reason: bool(candidate),
            "ollama_loaded_device": lambda *_args: "cpu", "CORRECTOR_INSTRUCTION": "",
            "RETRY_INSTRUCTION": "retry", "record_attempt": lambda attempts_arg, *args, **kwargs:
                attempts_arg.append((args[2], args[3])) if attempts_arg is not None else None,
            "unload_corrector_model": lambda *_args: None,
        }
        correct = load_correct_text(namespace)
        settings = {"correction": {"local_models": [
            "qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M"]}}
        result = correct("raw text", {"enabled": True, "provider": "ollama",
                        "model": "qwen3.5:4b-q4_K_M", "prompt": "", "num_gpu": 0},
                         provider_settings=settings, source="batch", attempt_log=attempts)
        self.assertEqual(result[0], "corrected")
        self.assertEqual(result[1], "local_model_fallback")
        self.assertEqual(calls, ["qwen3.5:4b-q4_K_M", "qwen3.5:4b-q4_K_M",
                                 "qwen3.5:2b-q4_K_M"])

    def test_next_gigaam_asr_runs_during_fifo_correction_and_jobs_finish_in_order(self):
        async def scenario():
            asr_started = []
            correction_started = []
            completed = []
            first_correction_release = asyncio.Event()
            second_asr_started = asyncio.Event()
            second_correction_started = asyncio.Event()
            jobs_queue = asyncio.Queue()
            corrections = asyncio.Queue()
            correction_idle = asyncio.Event()
            correction_idle.set()
            pending = {"count": 0}

            async def run(job, asr_only=False):
                self.assertTrue(asr_only)
                asr_started.append(job["job_id"])
                if job["job_id"] == "2":
                    second_asr_started.set()
                return {"status": "asr_ready", "text": job["job_id"],
                        "_raw_asr_text": job["job_id"], "_event": {},
                        "asr_backend_selected": "gigaam-v3-e2e-rnnt"}

            async def correction_stage(job, _result):
                correction_started.append(job["job_id"])
                if job["job_id"] == "1":
                    await first_correction_release.wait()
                else:
                    second_correction_started.set()
                return {"status": "ok", "text": job["job_id"]}

            async def finish_job(job, result, _started):
                completed.append((job["job_id"], result["text"]))
                if not job["future"].done():
                    job["future"].set_result(result)

            namespace = {
                "asyncio": asyncio,
                "queue": jobs_queue,
                "correction_queue": corrections,
                "correction_idle": correction_idle,
                "correction_pending": 0,
                "RECOVERY_CANCEL": Path(tempfile.gettempdir()) / "missing-whisper-cancel-markers",
                "active_asr_backend": "gigaam-v3-e2e-rnnt",
                "gigaam_worker": object(),
                "run": run,
                "correction_stage": correction_stage,
                "finish_job": finish_job,
                "time": __import__("time"),
                "logger": logging.getLogger("pipeline-test"),
            }
            worker, correction_worker = load_workers(namespace)
            asr_task = asyncio.create_task(worker())
            correction_task = asyncio.create_task(correction_worker())
            loop = asyncio.get_running_loop()
            futures = []
            for job_id in ("1", "2"):
                future = loop.create_future()
                futures.append(future)
                await jobs_queue.put({"job_id": job_id, "profile": {"asr_backend": "gigaam-v3-e2e-rnnt"},
                                      "provider": "local", "future": future,
                                      "queued": __import__("time").monotonic(), "wav": "unused"})
            try:
                await asyncio.wait_for(second_asr_started.wait(), timeout=1)
                self.assertEqual(asr_started, ["1", "2"])
                self.assertEqual(correction_started, ["1"])
                self.assertFalse(futures[0].done())
                first_correction_release.set()
                await asyncio.wait_for(second_correction_started.wait(), timeout=1)
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=1)
                self.assertEqual([r["text"] for r in results], ["1", "2"])
                self.assertEqual(completed, [("1", "1"), ("2", "2")])
            finally:
                asr_task.cancel()
                correction_task.cancel()
                await asyncio.gather(asr_task, correction_task, return_exceptions=True)

        asyncio.run(scenario())

    def test_non_gigaam_asr_waits_until_correction_queue_is_idle(self):
        source = DAEMON.read_text(encoding="utf-8")
        self.assertIn('job["profile"].get("asr_backend") == "gigaam-v3-e2e-rnnt"', source)
        self.assertIn('job.get("provider", "local") == "local"', source)
        self.assertIn('active_asr_backend == "gigaam-v3-e2e-rnnt"', source)
        self.assertIn('await correction_idle.wait()', source)


if __name__ == "__main__":
    unittest.main()
