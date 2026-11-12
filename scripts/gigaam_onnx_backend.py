"""Local-only GigaAM-v3 CTC inference through ONNX Runtime CUDA FP32.

Model weights, patched model code, and ONNX exports must be installed locally
and match the pinned SHA-256 values below. This module never downloads files or
silently falls back to CPU inference.
"""
from __future__ import annotations

import hashlib
import gc
import ctypes
import json
import os
import select
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any


SAMPLE_RATE = 16000
MAX_WINDOW_SECONDS = 24.0
OVERLAP_SECONDS = 1.0
WORKER_PATH = Path(__file__).with_name("gigaam_onnx_worker.py")
CUDA_PROVIDER = "CUDAExecutionProvider"
PRIVATE_TEMP_DIR = Path.home() / ".cache/whisper-dictate/gigaam-onnx-tmp"

# The e2e model is the recommended choice: it decodes SentencePiece tokens and
# includes punctuation/text normalization. Plain CTC is retained as a baseline.
MODEL_SPECS = {
    "gigaam-v3-e2e-ctc-onnx-gpu": {
        "display_name": "GigaAM-v3 e2e_ctc · GPU",
        "revision": "cec030b4c4f35d928e4a9044a3bdb29ebd499fac",
        "model_name": "v3_e2e_ctc",
        "onnx_file": "v3_e2e_ctc.onnx",
        "onnx_sha256": "377701bd33568f4733feec2db5b2dc12544fd09a5a5dfa69ccf55d161f84027a",
        "files": {
            "config.json": "3dbcd2ebe1a572cb9a9945eeb3faf8b59180d408e61a61ecda44db30421e7251",
            "modeling_gigaam.py": "9ec586251d0acd5f343c429947394ecddc979d6f6e1a69bf7f8258fe229acd87",
            "pytorch_model.bin": "9801f83ef3979779982c4f83ec1c234d7571e8c1a5aa2bfc0b09ceb97a16cd75",
            "tokenizer.model": "0b9a1960898fbfdf5424ab852ea17445eb3da960fba23e977ff100eb0054fbc8",
        },
    },
    "gigaam-v3-ctc-onnx-gpu": {
        "display_name": "GigaAM-v3 ctc · GPU",
        "revision": "15ef3b5a88da78f93134b3cb7f015c70aefa8946",
        "model_name": "v3_ctc",
        "onnx_file": "v3_ctc.onnx",
        "onnx_sha256": "1fb978d4f41e1334003fae9d29f2d9844d4132315a90553a4b986839051ba9d3",
        "files": {
            "config.json": "5db9b9e3f8c1a6193f3e262857cbaa75f4fd7653990de846162ad7318f61bdb5",
            "modeling_gigaam.py": "9ec586251d0acd5f343c429947394ecddc979d6f6e1a69bf7f8258fe229acd87",
            "pytorch_model.bin": "64d9c925df4bfd57e19f7b26d001106f9c5a35bd088a71c5bde071bf92c4a559",
        },
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, category: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise BackendError(category)


class BackendError(RuntimeError):
    """Safe inference failure; never contains paths, text, or provider details."""

    def __init__(self, category: str):
        allowed = {
            "weights_missing_or_corrupt", "dependency_import_failure", "cancelled",
            "audio_decode_timeout", "audio_decode_failure", "inference_failure",
            "cuda_execution_unavailable", "invalid_model_id", "worker_unavailable",
            "worker_timeout",
        }
        self.category = category if category in allowed else "inference_failure"
        super().__init__(self.category)


def _private_temp_dir() -> Path:
    """Return a private persistent temp root, avoiding a small /tmp quota."""
    try:
        PRIVATE_TEMP_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        PRIVATE_TEMP_DIR.chmod(0o700)
        return PRIVATE_TEMP_DIR
    except Exception:
        raise BackendError("inference_failure") from None


def verify_model_files(model_id: str, model_dir: str | Path) -> dict[str, Any]:
    """Check all raw assets and the generated ONNX against pinned metadata."""
    spec = MODEL_SPECS.get(model_id)
    if spec is None:
        raise BackendError("invalid_model_id")
    root = Path(model_dir)
    manifest_path = root / "onnx-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        raise BackendError("weights_missing_or_corrupt") from None

    expected_manifest = {
        "schema": 1,
        "model_id": model_id,
        "revision": spec["revision"],
        "model_name": spec["model_name"],
        "precision": "fp32",
        "provider": CUDA_PROVIDER,
        "files": spec["files"],
        "onnx_file": spec["onnx_file"],
        "onnx_sha256": spec["onnx_sha256"],
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise BackendError("weights_missing_or_corrupt")
    for filename, expected_hash in spec["files"].items():
        _verify_hash(root / filename, expected_hash, "weights_missing_or_corrupt")
    onnx_path = root / spec["onnx_file"]
    _verify_hash(onnx_path, spec["onnx_sha256"], "weights_missing_or_corrupt")
    return {"spec": spec, "root": root, "onnx_path": onnx_path}


def _load_ort_session(onnx_path: Path):
    """Create and warm a CUDA session; fail if no graph node executes on CUDA."""
    try:
        import onnxruntime as ort
    except Exception:
        raise BackendError("dependency_import_failure") from None
    try:
        if CUDA_PROVIDER not in ort.get_available_providers():
            raise BackendError("cuda_execution_unavailable")
        # ORT's pip extra installs the CUDA/cuDNN shared libraries beside its
        # Python packages. Preload them before it initializes the CUDA EP.
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload()
        options = ort.SessionOptions()
        options.enable_profiling = True
        with tempfile.TemporaryDirectory(
                prefix="gigaam-ort-profile-", dir=_private_temp_dir()) as profile_dir:
            options.profile_file_prefix = str(Path(profile_dir) / "session")
            session = ort.InferenceSession(
                str(onnx_path), sess_options=options,
                providers=[CUDA_PROVIDER, "CPUExecutionProvider"],
            )
            providers = session.get_providers()
            if not providers or providers[0] != CUDA_PROVIDER:
                raise BackendError("cuda_execution_unavailable")
            # The model graph's input layout is pinned in the manifest and the
            # exporter uses [batch, 64 mel bins, frames] plus [batch] lengths.
            features = __import__("numpy").zeros((1, 64, 120), dtype="float32")
            lengths = __import__("numpy").array([120], dtype="int64")
            input_names = {item.name for item in session.get_inputs()}
            if input_names != {"features", "feature_lengths"}:
                raise BackendError("weights_missing_or_corrupt")
            session.run(None, {
                "features": features,
                "feature_lengths": lengths,
            })
            profile_path = session.end_profiling()
            try:
                profile_text = Path(profile_path).read_text(encoding="utf-8")
            except Exception:
                raise BackendError("cuda_execution_unavailable") from None
            # ORT builds can cap the trace or leave its final event incomplete.
            # Parse complete JSON events incrementally, ignoring only a trailing
            # partial event while still requiring positive CUDA node evidence.
            decoder = json.JSONDecoder()
            position = profile_text.find("[") + 1
            used_cuda = False
            while position > 0 and position < len(profile_text):
                while position < len(profile_text) and profile_text[position] in " \n\r\t,":
                    position += 1
                if position >= len(profile_text) or profile_text[position] == "]":
                    break
                try:
                    event, next_position = decoder.raw_decode(profile_text, position)
                except json.JSONDecodeError:
                    break
                if (isinstance(event, dict)
                        and event.get("args", {}).get("provider") == CUDA_PROVIDER):
                    used_cuda = True
                    break
                position = next_position
            if not used_cuda:
                raise BackendError("cuda_execution_unavailable")
            return session
    except BackendError:
        raise
    except Exception:
        raise BackendError("cuda_execution_unavailable") from None


def _release_pytorch_inference_weights(model):
    """Keep preprocessing/decoding, while ONNX owns encoder and head inference."""
    import torch

    core = model.model
    parameter = next(core.parameters())
    # prepare_wav derives device and dtype from the first registered parameter.
    core.register_parameter("_onnx_device_anchor", torch.nn.Parameter(
        torch.empty(0, device=parameter.device, dtype=parameter.dtype),
        requires_grad=False))
    del parameter
    del core.encoder
    del core.head
    gc.collect()
    # glibc can retain freed CPU weight buffers in its heap indefinitely.
    # This is an optional RSS reduction; inference does not depend on it.
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError):
        pass


def load_model(model_id: str, model_dir: str | Path):
    """Load a pinned CPU pre/post-processor and verified CUDA ONNX graph."""
    verified = verify_model_files(model_id, model_dir)
    try:
        from transformers import AutoModel
    except Exception:
        raise BackendError("dependency_import_failure") from None
    try:
        model = AutoModel.from_pretrained(
            str(verified["root"]), trust_remote_code=True,
            local_files_only=True, use_safetensors=False,
        )
        model.to("cpu")
        model.eval()
        _release_pytorch_inference_weights(model)
    except ImportError:
        raise BackendError("dependency_import_failure") from None
    except Exception:
        raise BackendError("weights_missing_or_corrupt") from None
    session = _load_ort_session(verified["onnx_path"])
    return model, session, verified["spec"]


def window_bounds(sample_count: int, sample_rate: int = SAMPLE_RATE,
                  window_seconds: float = MAX_WINDOW_SECONDS,
                  overlap_seconds: float = OVERLAP_SECONDS) -> list[tuple[int, int]]:
    if sample_count <= 0 or sample_rate <= 0:
        return []
    if not 0 < window_seconds <= 24 or not 0 <= overlap_seconds < window_seconds:
        raise BackendError("inference_failure")
    window = max(1, int(window_seconds * sample_rate))
    overlap = int(overlap_seconds * sample_rate)
    step = window - overlap
    bounds, start = [], 0
    while start < sample_count:
        end = min(sample_count, start + window)
        bounds.append((start, end))
        if end == sample_count:
            break
        start += step
    return bounds


def join_chunk_texts(chunks: list[str], max_overlap_words: int = 40) -> str:
    words: list[str] = []
    for chunk in chunks:
        incoming = chunk.split()
        overlap = 0
        for size in range(1, min(len(words), len(incoming), max_overlap_words) + 1):
            if [word.casefold() for word in words[-size:]] == [
                    word.casefold() for word in incoming[:size]]:
                overlap = size
        words.extend(incoming[overlap:])
    return " ".join(words).strip()


def _read_mono_16k(path: str | Path) -> bytes:
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le",
             "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-"],
            check=True, capture_output=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        raise BackendError("audio_decode_timeout") from None
    except subprocess.CalledProcessError:
        raise BackendError("audio_decode_failure") from None
    except Exception:
        raise BackendError("audio_decode_failure") from None
    return result.stdout


def _decode_logits(model, logits) -> str:
    """Decode CTC token IDs without exposing intermediate hypotheses."""
    import numpy as np
    labels = np.asarray(logits[0]).argmax(axis=-1).tolist()
    decoder = model.model.decoding
    blank_id = decoder.blank_id
    keep: list[int] = []
    previous = None
    for token in labels:
        if token != blank_id and token != previous:
            keep.append(int(token))
        previous = token
    return decoder.tokenizer.decode(keep)


def transcribe(model, session, audio_path: str | Path, cancel_event=None) -> tuple[str, int]:
    """Run <=24-second windows on the verified CUDA graph and join transcripts."""
    import numpy as np
    import torch

    pcm = _read_mono_16k(audio_path)
    frame_width = 2
    sample_count = len(pcm) // frame_width
    if sample_count == 0:
        return "", 0
    chunks: list[str] = []
    bounds = window_bounds(sample_count)
    for start, end in bounds:
        if cancel_event is not None and (cancel_event() if callable(cancel_event)
                                         else cancel_event.is_set()):
            raise BackendError("cancelled")
        audio_bytes = pcm[start * frame_width:end * frame_width]
        if not audio_bytes or not any(audio_bytes):
            chunks.append("")
            continue
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=_private_temp_dir()) as temporary:
                with wave.open(temporary.name, "wb") as output_wav:
                    output_wav.setnchannels(1)
                    output_wav.setsampwidth(frame_width)
                    output_wav.setframerate(SAMPLE_RATE)
                    output_wav.writeframes(audio_bytes)
                with torch.inference_mode():
                    wav, length = model.model.prepare_wav(temporary.name)
                    features, feature_lengths = model.model.preprocessor(wav, length)
                    outputs = session.run(None, {
                        "features": features.detach().cpu().numpy().astype(np.float32),
                        "feature_lengths": feature_lengths.detach().cpu().numpy().astype(np.int64),
                    })
            chunks.append(_decode_logits(model, outputs[0]))
        except BackendError:
            raise
        except Exception:
            raise BackendError("inference_failure") from None
    return join_chunk_texts(chunks), len(bounds)


def safe_error_category(error: BaseException) -> str:
    category = getattr(error, "category", None)
    if category in {
        "weights_missing_or_corrupt", "dependency_import_failure", "cancelled",
        "audio_decode_timeout", "audio_decode_failure", "inference_failure",
        "cuda_execution_unavailable", "invalid_model_id", "worker_unavailable",
        "worker_timeout",
    }:
        return category
    if isinstance(error, ImportError):
        return "dependency_import_failure"
    return "inference_failure"


class GigaAMOnnxWorker:
    """Persistent subprocess client for an explicit GigaAM CUDA model choice."""

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
            raise BackendError("worker_unavailable")
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise BackendError("worker_timeout")
        line = self.process.stdout.readline()
        if not line:
            raise BackendError("worker_unavailable")
        return line

    def _start(self):
        if self.backend_id not in MODEL_SPECS:
            raise BackendError("invalid_model_id")
        if not Path(self.python_path).is_file() or not WORKER_PATH.is_file():
            raise BackendError("worker_unavailable")
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
            raise BackendError("worker_unavailable") from None
        if (response.get("status") != "ready"
                or response.get("device") != "cuda"
                or response.get("backend_id") != self.backend_id):
            category = response.get("category", "inference_failure")
            self.close()
            raise BackendError(category)

    def transcribe(self, audio_path: str | Path,
                   cancel_marker: str | Path | None = None) -> tuple[str, int]:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise BackendError("worker_unavailable")
            try:
                self.process.stdin.write(json.dumps({
                    "audio_path": str(audio_path),
                    "cancel_marker": str(cancel_marker) if cancel_marker else None,
                }) + "\n")
                self.process.stdin.flush()
                response = json.loads(self._readline(3600))
            except BackendError:
                raise
            except Exception:
                raise BackendError("worker_unavailable") from None
            if response.get("status") != "ok":
                raise BackendError(response.get("category", "inference_failure"))
            text = response.get("text")
            if not isinstance(text, str):
                raise BackendError("inference_failure")
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
