import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gigaam_onnx_backend", ROOT / "scripts/gigaam_onnx_backend.py")
BACKEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND)


class GigaAMOnnxBackendTests(unittest.TestCase):
    def test_recommended_punctuation_model_is_first_and_variants_are_pinned(self):
        first = next(iter(BACKEND.MODEL_SPECS))
        self.assertEqual(first, "gigaam-v3-e2e-ctc-onnx-gpu")
        self.assertIn("tokenizer.model", BACKEND.MODEL_SPECS[first]["files"])
        self.assertNotIn("tokenizer.model",
                         BACKEND.MODEL_SPECS["gigaam-v3-ctc-onnx-gpu"]["files"])
        self.assertEqual(BACKEND.MODEL_SPECS[first]["onnx_file"], "v3_e2e_ctc.onnx")

    def test_manifest_and_every_model_file_must_match_exact_pins(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            files = {"config.json": "config", "modeling_gigaam.py": "code",
                     "pytorch_model.bin": "weights", "v3_test.onnx": "graph"}
            hashes = {}
            for name, contents in files.items():
                path = root / name
                path.write_text(contents, encoding="utf-8")
                hashes[name] = BACKEND.sha256_file(path)
            model_id = "test-gpu-model"
            spec = {
                "display_name": "test", "revision": "rev", "model_name": "v3_test",
                "onnx_file": "v3_test.onnx", "onnx_sha256": hashes["v3_test.onnx"],
                "files": {name: value for name, value in hashes.items()
                          if name != "v3_test.onnx"},
            }
            manifest = {
                "schema": 1, "model_id": model_id, "revision": "rev",
                "model_name": "v3_test", "precision": "fp32",
                "provider": "CUDAExecutionProvider", "files": spec["files"],
                "onnx_file": "v3_test.onnx", "onnx_sha256": hashes["v3_test.onnx"],
            }
            (root / "onnx-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch.dict(BACKEND.MODEL_SPECS, {model_id: spec}):
                self.assertEqual(BACKEND.verify_model_files(model_id, root)["spec"], spec)
                (root / "v3_test.onnx").write_text("altered", encoding="utf-8")
                with self.assertRaisesRegex(BACKEND.BackendError,
                                            "weights_missing_or_corrupt"):
                    BACKEND.verify_model_files(model_id, root)

    def test_cuda_provider_absence_fails_closed(self):
        fake_ort = type("Ort", (), {
            "get_available_providers": staticmethod(lambda: ["CPUExecutionProvider"]),
        })()
        with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            with self.assertRaisesRegex(BACKEND.BackendError,
                                        "cuda_execution_unavailable"):
                BACKEND._load_ort_session(Path("unused.onnx"))

    def test_windowing_and_chunk_join_are_stable(self):
        self.assertEqual(BACKEND.window_bounds(0), [])
        self.assertEqual(BACKEND.window_bounds(16000), [(0, 16000)])
        self.assertEqual(BACKEND.join_chunk_texts(["один два три", "два три четыре"]),
                         "один два три четыре")

    def test_error_category_does_not_include_private_exception_text(self):
        self.assertEqual(BACKEND.safe_error_category(
            RuntimeError("private transcript or path")), "inference_failure")
        self.assertEqual(BACKEND.safe_error_category(
            BACKEND.BackendError("cuda_execution_unavailable")),
            "cuda_execution_unavailable")

    def test_onnx_worker_releases_cpu_weights_and_keeps_device_anchor(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is available in the GigaAM worker environment")

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(4, 4, dtype=torch.float64)
                self.head = torch.nn.Linear(4, 2, dtype=torch.float64)
                self.preprocessor = torch.nn.Identity()
                self.decoding = object()

        core = Core()
        decoder = core.decoding
        BACKEND._release_pytorch_inference_weights(type("Model", (), {"model": core})())
        self.assertFalse(hasattr(core, "encoder"))
        self.assertFalse(hasattr(core, "head"))
        self.assertIs(core.decoding, decoder)
        self.assertEqual(next(core.parameters()).dtype, torch.float64)
        self.assertEqual(next(core.parameters()).device.type, "cpu")
        self.assertEqual(next(core.parameters()).numel(), 0)
        self.assertTrue(torch.equal(core.preprocessor(torch.ones(2)), torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
