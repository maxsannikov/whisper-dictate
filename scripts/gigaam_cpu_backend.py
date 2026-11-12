"""Local-only GigaAM-v3 e2e_rnnt CPU inference helpers.

The model must already exist in a pinned local directory. This module never
downloads weights, model code, or audio and never imports gated longform code.
"""
from __future__ import annotations

import json
import os
import select
import hashlib
import subprocess
import tempfile
import threading
import wave
from pathlib import Path


MODEL_NAME = "GigaAM-v3 e2e_rnnt"
MAX_WINDOW_SECONDS = 24.0
OVERLAP_SECONDS = 1.0
SAMPLE_RATE = 16000
PINNED_MODEL_REVISION = "7655ad717f8122257385bb4b2f373db3697e8680"
PINNED_FILES_SHA256 = {
    "config.json": "02361ba9cafd6c3ec66fcdd73494c3b562a60eb2a2d1b13f3cb04ae440d93e52",
    # Locally patched by prepare_gigaam_model.py after checking the upstream hash;
    # only pyannote imports in the unused gated VAD helper are replaced with an error.
    "modeling_gigaam.py": "9ec586251d0acd5f343c429947394ecddc979d6f6e1a69bf7f8258fe229acd87",
    "pytorch_model.bin": "afc6dcbae8320ea56f2cddebc0f13fbf62c9d59b6ddcad899782623c8610826a",
    "tokenizer.model": "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a",
}
WORKER_PATH = Path(__file__).with_name("gigaam_worker.py")


def window_bounds(sample_count: int, sample_rate: int = SAMPLE_RATE,
                  window_seconds: float = MAX_WINDOW_SECONDS,
                  overlap_seconds: float = OVERLAP_SECONDS) -> list[tuple[int, int]]:
    """Return deterministic sample ranges with bounded overlap."""
    if sample_count <= 0 or sample_rate <= 0:
        return []
    if not 0 < window_seconds <= 24 or not 0 <= overlap_seconds < window_seconds:
        raise ValueError("chunk_settings_invalid")
    window = max(1, int(window_seconds * sample_rate))
    overlap = int(overlap_seconds * sample_rate)
    step = window - overlap
    bounds = []
    start = 0
    while start < sample_count:
        end = min(sample_count, start + window)
        bounds.append((start, end))
        if end == sample_count:
            break
        start += step
    return bounds


def join_chunk_texts(chunks: list[str], max_overlap_words: int = 40) -> str:
    """Join chunk transcripts while removing the longest exact word overlap."""
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
    """Decode through the installed ffmpeg binary without logging media data."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le",
               "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-"]
    result = subprocess.run(command, check=True, capture_output=True, timeout=180)
    return result.stdout


def transcribe(model, audio_path: str | Path, cancel_event=None) -> tuple[str, int]:
    """Run CPU inference in <=24 second windows; return text and window count."""
    pcm = _read_mono_16k(audio_path)
    frame_width = 2
    sample_count = len(pcm) // frame_width
    if sample_count == 0:
        return "", 0
    bounds = window_bounds(sample_count)
    output: list[str] = []
    for start, end in bounds:
        if cancel_event is not None and (cancel_event() if callable(cancel_event)
                                         else cancel_event.is_set()):
            raise InterruptedError("inference_cancelled")
        chunk = pcm[start * frame_width:end * frame_width]
        # Skip true digital silence; avoid passing empty/no-speech artifacts to the model.
        if not chunk or not any(chunk):
            output.append("")
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
            with wave.open(temporary.name, "wb") as output_wav:
                output_wav.setnchannels(1)
                output_wav.setsampwidth(frame_width)
                output_wav.setframerate(SAMPLE_RATE)
                output_wav.writeframes(chunk)
            output.append(model.transcribe(temporary.name))
    return join_chunk_texts(output), len(bounds)


def load_model(model_dir: str | Path):
    """Load only a preinstalled local checkpoint with CPU mapping and no hub access."""
    path = Path(model_dir)
    required = ("config.json", "modeling_gigaam.py", "pytorch_model.bin", "tokenizer.model")
    if not path.is_dir() or any(not (path / filename).is_file() for filename in required):
        raise FileNotFoundError("gigaam_local_weights_missing_or_incomplete")
    for filename, expected in PINNED_FILES_SHA256.items():
        digest = hashlib.sha256()
        with (path / filename).open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("gigaam_local_weights_hash_mismatch")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True, use_safetensors=False)
    model.to("cpu")
    model.eval()
    return model


def safe_error_category(error: BaseException) -> str:
    """Map failures to a small allowlist; never expose exception text."""
    category = getattr(error, "category", None)
    if category in {
            "weights_missing_or_corrupt", "dependency_import_failure", "cancelled",
            "audio_decode_timeout", "audio_decode_failure", "invalid_chunk_settings",
            "inference_failure", "worker_unavailable", "worker_timeout"}:
        return category
    if isinstance(error, FileNotFoundError):
        return "weights_missing_or_corrupt"
    if isinstance(error, ValueError) and str(error) == "gigaam_local_weights_hash_mismatch":
        return "weights_missing_or_corrupt"
    if isinstance(error, ImportError):
        return "dependency_import_failure"
    if isinstance(error, InterruptedError):
        return "cancelled"
    if isinstance(error, (subprocess.TimeoutExpired, TimeoutError)):
        return "audio_decode_timeout"
    if isinstance(error, subprocess.CalledProcessError):
        return "audio_decode_failure"
    if isinstance(error, ValueError) and str(error) == "chunk_settings_invalid":
        return "invalid_chunk_settings"
    return "inference_failure"


class WorkerError(RuntimeError):
    """Safe worker failure carrying only an allowlisted category."""
    def __init__(self, category: str):
        self.category = category if category in {
            "weights_missing_or_corrupt", "dependency_import_failure", "cancelled",
            "audio_decode_timeout", "audio_decode_failure", "invalid_chunk_settings",
            "inference_failure", "worker_unavailable", "worker_timeout",
        } else "inference_failure"
        super().__init__(self.category)


class GigaAMWorker:
    """Persistent subprocess client; the application venv remains untouched."""
    def __init__(self, model_dir: str | Path, python_path: str | Path | None = None):
        self.model_dir = str(model_dir)
        self.python_path = str(python_path or
            (Path.home() / ".local/venvs/gigaam/bin/python"))
        self.process = None
        self.lock = threading.Lock()
        self._start()

    def _readline(self, timeout: float) -> str:
        if self.process is None or self.process.stdout is None:
            raise WorkerError("worker_unavailable")
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise WorkerError("worker_timeout")
        line = self.process.stdout.readline()
        if not line:
            raise WorkerError("worker_unavailable")
        return line

    def _start(self):
        if not Path(self.python_path).is_file() or not WORKER_PATH.is_file():
            raise WorkerError("worker_unavailable")
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_MODULES_CACHE"] = str(
            Path.home() / ".cache/whisper-dictate/gigaam-transformers-modules")
        self.process = subprocess.Popen(
            [self.python_path, str(WORKER_PATH), "--model-dir", self.model_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, env=env)
        try:
            response = json.loads(self._readline(180))
        except Exception:
            self.close()
            raise WorkerError("worker_unavailable") from None
        if response.get("status") != "ready":
            category = response.get("category", "inference_failure")
            self.close()
            raise WorkerError(category)

    def transcribe(self, audio_path: str | Path, cancel_marker: str | Path | None = None) -> tuple[str, int]:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise WorkerError("worker_unavailable")
            try:
                self.process.stdin.write(json.dumps({"audio_path": str(audio_path),
                    "cancel_marker": str(cancel_marker) if cancel_marker else None}) + "\n")
                self.process.stdin.flush()
                response = json.loads(self._readline(3600))
            except WorkerError:
                raise
            except Exception:
                raise WorkerError("worker_unavailable") from None
            if response.get("status") != "ok":
                raise WorkerError(response.get("category", "inference_failure"))
            text = response.get("text")
            if not isinstance(text, str):
                raise WorkerError("inference_failure")
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
