import importlib.util
import ast
import asyncio
import logging
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gigaam_cpu_backend", ROOT / "scripts/gigaam_cpu_backend.py")
BACKEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND)


def load_ensure_with_worker(worker_factory):
    source = (ROOT / "scripts/whisper-daemon.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    ensure_node = next(node for node in tree.body
                       if isinstance(node, ast.AsyncFunctionDef) and node.name == "ensure")
    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)
    namespace = {
        "asyncio": SimpleNamespace(to_thread=direct_to_thread),
        "logger": logging.getLogger("test-gigaam"),
        "WhisperModel": lambda *args, **kwargs: {"model": args[0], **kwargs},
        "local_model_source": lambda value: value,
        "GigaAMWorker": worker_factory, "GIGAAM_MODEL_NAME": BACKEND.MODEL_NAME,
        "gigaam_error_category": BACKEND.safe_error_category,
        "GIGAAM_ONNX_BACKENDS": {},
        "GIGAAM_BACKENDS": {"gigaam-v3-e2e-rnnt"},
        "GigaAMOnnxWorker": lambda *_args: None,
        "gigaam_onnx_error_category": BACKEND.safe_error_category,
        "model": None, "model_profile": None, "model_lease": 0,
        "active_asr_backend": "faster-whisper", "active_asr_error_category": None,
        "active_asr_model_name": None, "active_asr_device": None, "gigaam_worker": None,
        "gigaam_onnx_worker": None,
    }
    async def release():
        namespace["model"] = None
        namespace["model_profile"] = None
        if namespace["gigaam_worker"] is not None:
            namespace["gigaam_worker"].close()
            namespace["gigaam_worker"] = None
    namespace["release"] = release
    exec(compile(ast.Module(body=[ensure_node], type_ignores=[]),
                 "daemon-gigaam-ensure", "exec"), namespace)
    return namespace


class GigaAMCpuBackendTests(unittest.TestCase):
    def test_missing_worker_falls_back_to_actual_fasterwhisper_cpu_model(self):
        def missing_worker(*_args):
            raise BACKEND.WorkerError("worker_unavailable")
        namespace = load_ensure_with_worker(missing_worker)
        profile = {"asr_backend": "gigaam-v3-e2e-rnnt", "gigaam_model_dir": "/local/model",
                   "model": "large-v3", "device": "cpu", "compute_type": "int8"}
        asyncio.run(namespace["ensure"](profile))
        self.assertEqual(namespace["active_asr_backend"], "faster-whisper")
        self.assertEqual(namespace["active_asr_error_category"], "worker_unavailable")
        self.assertEqual(namespace["active_asr_model_name"], "large-v3")
        self.assertEqual(namespace["active_asr_device"], "cpu")
        self.assertEqual(namespace["model_profile"]["asr_backend"], "faster-whisper")

    def test_gigaam_worker_uses_cpu_and_is_cached(self):
        class ReadyWorker:
            def __init__(self, path): self.path = path
            def close(self): pass
        namespace = load_ensure_with_worker(ReadyWorker)
        profile = {"asr_backend": "gigaam-v3-e2e-rnnt", "gigaam_model_dir": "/local/model",
                   "model": "large-v3", "device": "cuda", "compute_type": "int8_float16"}
        cold = asyncio.run(namespace["ensure"](profile))
        warm = asyncio.run(namespace["ensure"](profile))
        self.assertTrue(cold)
        self.assertFalse(warm)
        self.assertEqual(namespace["active_asr_backend"], "gigaam-v3-e2e-rnnt")
        self.assertEqual(namespace["active_asr_model_name"], "GigaAM-v3 e2e_rnnt")
        self.assertEqual(namespace["active_asr_device"], "cpu")

    def test_windows_are_bounded_with_expected_overlap(self):
        bounds = BACKEND.window_bounds(50 * 16000)
        self.assertEqual(bounds[0], (0, 24 * 16000))
        self.assertEqual(bounds[1][0], 23 * 16000)
        self.assertTrue(all((end - start) <= 24 * 16000 for start, end in bounds))
        self.assertEqual(bounds[-1][1], 50 * 16000)

    def test_empty_audio_has_no_windows(self):
        self.assertEqual(BACKEND.window_bounds(0), [])

    def test_join_removes_longest_overlap_and_preserves_repeats_inside_chunks(self):
        self.assertEqual(BACKEND.join_chunk_texts([
            "один два три четыре", "три четыре пять пять"]),
            "один два три четыре пять пять")

    def test_chunk_configuration_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "chunk_settings_invalid"):
            BACKEND.window_bounds(10, window_seconds=25)

    def test_local_loader_rejects_missing_or_incomplete_weights(self):
        with self.assertRaisesRegex(FileNotFoundError,
                                    "gigaam_local_weights_missing_or_incomplete"):
            BACKEND.load_model(Path("/definitely/not/a/local/model"))

    def test_local_loader_rejects_corrupt_weight_hash_before_import(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name in BACKEND.PINNED_FILES_SHA256:
                (root / name).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "gigaam_local_weights_hash_mismatch"):
                BACKEND.load_model(root)

    def test_error_categories_are_allowlisted(self):
        self.assertEqual(BACKEND.safe_error_category(ImportError("secret")),
                         "dependency_import_failure")
        self.assertEqual(BACKEND.safe_error_category(RuntimeError("private text")),
                         "inference_failure")

    def test_cancel_is_checked_before_each_inference_window(self):
        class NeverModel:
            def transcribe(self, _path):
                raise AssertionError("inference_should_not_run")
        with patch.object(BACKEND, "_read_mono_16k", return_value=b"\x01\x00" * 16000):
            with self.assertRaisesRegex(InterruptedError, "inference_cancelled"):
                BACKEND.transcribe(NeverModel(), "unused", cancel_event=lambda: True)

    def test_repeat_inference_is_deterministic_with_fake_model(self):
        class FakeModel:
            def transcribe(self, _path):
                return "тестовый текст."
        with patch.object(BACKEND, "_read_mono_16k", return_value=b"\x01\x00" * 16000):
            first = BACKEND.transcribe(FakeModel(), "unused")
            second = BACKEND.transcribe(FakeModel(), "unused")
        self.assertEqual(first, second)

    def test_worker_missing_runtime_has_safe_category(self):
        with self.assertRaises(BACKEND.WorkerError) as caught:
            BACKEND.GigaAMWorker("/missing", python_path="/missing/python")
        self.assertEqual(caught.exception.category, "worker_unavailable")
        self.assertEqual(BACKEND.safe_error_category(caught.exception),
                         "worker_unavailable")


if __name__ == "__main__":
    unittest.main()
