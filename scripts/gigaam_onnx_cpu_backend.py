"""CPUExecutionProvider worker for the same pinned GigaAM CTC ONNX models."""
from __future__ import annotations

import gc
import json
import os
import select
import subprocess
import threading
from pathlib import Path

import gigaam_onnx_backend as cuda_backend

WORKER_PATH = Path(__file__).with_name("gigaam_onnx_cpu_worker.py")
CPU_PROVIDER = "CPUExecutionProvider"


def load_model(model_id: str, model_dir: str | Path):
    """Load the pinned pre/postprocessor and identical ONNX graph on CPU."""
    verified = cuda_backend.verify_model_files(model_id, model_dir)
    try:
        import onnxruntime as ort
        from transformers import AutoModel
    except Exception:
        raise cuda_backend.BackendError("dependency_import_failure") from None
    if CPU_PROVIDER not in ort.get_available_providers():
        raise cuda_backend.BackendError("worker_unavailable")
    try:
        model = AutoModel.from_pretrained(
            str(verified["root"]), trust_remote_code=True,
            local_files_only=True, use_safetensors=False,
        )
        model.to("cpu")
        model.eval()
        _release_inference_weights(model)
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(verified["onnx_path"]), sess_options=options,
            providers=[CPU_PROVIDER],
        )
        if session.get_providers() != [CPU_PROVIDER]:
            raise cuda_backend.BackendError("worker_unavailable")
        import numpy as np
        inputs = {item.name for item in session.get_inputs()}
        if inputs != {"features", "feature_lengths"}:
            raise cuda_backend.BackendError("inference_failure")
        session.run(None, {
            "features": np.zeros((1, 64, 120), dtype="float32"),
            "feature_lengths": np.asarray([120], dtype="int64"),
        })
        return model, session, verified["spec"]
    except cuda_backend.BackendError:
        raise
    except Exception:
        raise cuda_backend.BackendError("inference_failure") from None


def _release_inference_weights(model):
    """Keep preprocessing/decoding while ONNX owns encoder/head inference."""
    import torch

    core = model.model
    parameter = next(core.parameters())
    core.register_parameter("_onnx_device_anchor", torch.nn.Parameter(
        torch.empty(0, device="cpu", dtype=parameter.dtype), requires_grad=False))
    del parameter
    del core.encoder
    del core.head
    gc.collect()
    try:
        import ctypes
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError):
        pass


def safe_error_category(error: BaseException) -> str:
    return cuda_backend.safe_error_category(error)


class GigaAMOnnxCpuWorker:
    """Persistent subprocess client for an explicit CPU-only GigaAM inference."""

    def __init__(self, model_dir: str | Path, backend_id: str,
                 python_path: str | Path | None = None):
        self.backend_id = backend_id
        self.model_dir = str(model_dir)
        self.python_path = str(python_path or
            (Path.home() / ".local/venvs/gigaam-onnx-gpu/bin/python"))
        self.process = None
        self.lock = threading.Lock()
        self._start()

    def _readline(self, timeout: float) -> str:
        if self.process is None or self.process.stdout is None:
            raise cuda_backend.BackendError("worker_unavailable")
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise cuda_backend.BackendError("worker_timeout")
        line = self.process.stdout.readline()
        if not line:
            raise cuda_backend.BackendError("worker_unavailable")
        return line

    def _start(self):
        if self.backend_id not in cuda_backend.MODEL_SPECS:
            raise cuda_backend.BackendError("invalid_model_id")
        if not Path(self.python_path).is_file() or not WORKER_PATH.is_file():
            raise cuda_backend.BackendError("worker_unavailable")
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_MODULES_CACHE"] = str(
            Path.home() / ".cache/whisper-dictate/gigaam-transformers-modules")
        self.process = subprocess.Popen(
            [self.python_path, str(WORKER_PATH), "--backend-id", self.backend_id,
             "--model-dir", self.model_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, env=env,
        )
        try:
            response = json.loads(self._readline(240))
        except Exception:
            self.close()
            raise cuda_backend.BackendError("worker_unavailable") from None
        if (response.get("status") != "ready" or response.get("device") != "cpu"
                or response.get("backend_id") != self.backend_id):
            category = response.get("category", "inference_failure")
            self.close()
            raise cuda_backend.BackendError(category)

    def transcribe(self, audio_path: str | Path,
                   cancel_marker: str | Path | None = None) -> tuple[str, int]:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise cuda_backend.BackendError("worker_unavailable")
            try:
                self.process.stdin.write(json.dumps({
                    "audio_path": str(audio_path),
                    "cancel_marker": str(cancel_marker) if cancel_marker else None,
                }) + "\n")
                self.process.stdin.flush()
                response = json.loads(self._readline(3600))
            except cuda_backend.BackendError:
                raise
            except Exception:
                raise cuda_backend.BackendError("worker_unavailable") from None
            if response.get("status") != "ok":
                raise cuda_backend.BackendError(response.get("category", "inference_failure"))
            text = response.get("text")
            if not isinstance(text, str):
                raise cuda_backend.BackendError("inference_failure")
            return text, int(response.get("window_count", 0))

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=3)
        except Exception:
            process.kill()
            try:
                process.wait(timeout=3)
            except Exception:
                pass
