#!/usr/bin/env python3
"""Local ASR/correction daemon with FIFO stages and validated private metrics."""
import asyncio
import difflib
import gc
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

venv_py = os.path.expanduser("~/.local/venvs/whisper/bin/python")
if os.path.exists(venv_py) and sys.executable != venv_py:
    os.execv(venv_py, [venv_py] + sys.argv)
from faster_whisper import WhisperModel
from cloudflare_asr import (CloudflareError, cloudflare_first,
    reserve_daily_neurons as reserve_cloudflare_daily_neurons,
    NEURONS_PER_AUDIO_MINUTE, transcribe as cloudflare_transcribe)
from cloudflare_corrector import (DEFAULT_MODEL as DEFAULT_CLOUDFLARE_CORRECTOR,
    SUPPORTED_MODELS as CLOUDFLARE_CORRECTOR_MODELS,
    correct as cloudflare_correct, estimate_neurons as correction_neurons,
    estimate_request_neurons as estimate_cloudflare_correction_neurons,
    token_counts as correction_token_counts, MODEL_NEURON_RATES)
from multi_provider import (ProviderError, asr_chain as multi_asr_chain,
    correction_chain as multi_correction_chain, safe_settings as safe_provider_settings,
    resolve_local_asr_model, POST_TRANSCRIPTION_SOURCES,
    groq_transcribe, transcribe_gemini, groq_correct, openrouter_correct, google_correct,
    codex_correct, claude_correct)
from provider_quota import daily_limit_state, daily_only_routes, mark_daily_limit
from attempt_statistics import (choose_openrouter_fallback_model,
    cloudflare_correction_model_order)
from gigaam_cpu_backend import (MODEL_NAME as GIGAAM_MODEL_NAME, GigaAMWorker,
    safe_error_category as gigaam_error_category)
from gigaam_onnx_backend import (GigaAMOnnxWorker,
    safe_error_category as gigaam_onnx_error_category)
from gigaam_onnx_cpu_backend import (GigaAMOnnxCpuWorker,
    safe_error_category as gigaam_onnx_cpu_error_category)

HOME = Path.home()
CACHE = HOME / ".cache/whisper-dictate"
CONFIG = HOME / ".config/whisper-dictate/config.json"
REPLACEMENTS = CONFIG.parent / "autoreplacements.txt"
SOCKET = CACHE / "daemon.sock"
LOG = CACHE / "daemon.log"
METRICS = CACHE / "metrics.jsonl"
JOBS = CACHE / "jobs"
RAW_ASR = CACHE / "raw-asr"
RECOVERY_CANCEL = CACHE / "recovery-cancel"
MANUAL_CORRECTION_STATUS = CACHE / "manual-correction-status.json"
LATEST_LIVE_ASR_AT = CACHE / ".latest-live-asr-at"
CACHE.mkdir(parents=True, exist_ok=True)
RAW_ASR.mkdir(mode=0o700, parents=True, exist_ok=True)
RECOVERY_CANCEL.mkdir(mode=0o700, parents=True, exist_ok=True)
DEFAULT = {"version": 1, "model": "large-v3", "asr_backend": "faster-whisper",
 "gigaam_model_dir": str(HOME / ".cache/whisper-dictate/models/gigaam-v3-e2e-rnnt"),
 "language": "ru", "device": "cuda",
 "compute_type": "int8_float16", "beam_size": 1, "best_of": 1, "temperature": 0.0,
 "vad": False, "condition_on_previous_text": True, "initial_prompt": "", "glossary": [], "result_mode": "clipboard",
 "fallback": {"model": "medium", "device": "cuda", "compute_type": "int8_float16",
              "allow_cpu": False},
 "cloudflare_asr_enabled": False, "cloudflare_asr_model": "@cf/openai/whisper-large-v3-turbo",
 "cloudflare_asr": {"model": "@cf/openai/whisper-large-v3-turbo", "task": "transcribe",
                    "language": "ru", "vad_filter": False, "beam_size": 1,
                    "initial_prompt": "", "prefix": "", "condition_on_previous_text": True,
                    "no_speech_threshold": 0.6, "compression_ratio_threshold": 2.4,
                    "log_prob_threshold": -1.0, "hallucination_silence_threshold": None},
 "cloudflare_corrector": {"enabled": False, "model": DEFAULT_CLOUDFLARE_CORRECTOR,
                          "temperature": 0.0, "max_tokens": 2048,
                          "retry_attempts": 2, "priority_model": "auto"},
 "multi_provider": {"asr": {"groq_enabled": False, "google_enabled": False, "quality": "accuracy",
                              "order": ["cloudflare", "groq", "google", "local"]},
                    "correction": {"codex_enabled": True, "claude_enabled": False,
                                   "enabled": True,
                                   "codex_model": "gpt-6-luna", "claude_model": "sonnet",
                                   "codex_retry_attempts": 2,
                                   "claude_retry_attempts": 0,
                                   "groq_primary_retry_attempts": 0,
                                   "groq_fallback_retry_attempts": 0,
                                   "google_retry_attempts": 0,
                                   "groq_enabled": False, "openrouter_enabled": False,
                                   "google_enabled": False,
                                   "post_transcription_enabled": {
                                       "codex": False, "claude": False,
                                       "cloudflare": False, "openrouter": False,
                                       "groq_primary": False, "groq_fallback": False,
                                       "google": False, "local": True},
                                   "order": ["codex", "claude", "openrouter", "cloudflare",
                                             "groq_primary", "groq_fallback", "google"]}},
 "corrector": {"enabled": False, "provider": "ollama", "model": "qwen3.5:4b-q4_K_M", "prompt": "",
               "timeout_seconds": 300, "max_length_ratio": 1.5, "num_gpu": 8,
               "keep_alive": "5m", "temperature": 0, "num_ctx": 8192, "num_thread": 4}}
MODELS = {"large-v3", "large-v3-turbo", "medium", "small"}
COMPUTES = {"int8_float16", "float16", "int8", "float32"}
GIGAAM_ONNX_BACKENDS = {
    "gigaam-v3-e2e-ctc-onnx-gpu": {
        "model": "GigaAM-v3 e2e_ctc", "device": "cuda",
        "model_dir": HOME / ".cache/whisper-dictate/models/gigaam-v3-e2e-ctc",
    },
    "gigaam-v3-ctc-onnx-gpu": {
        "model": "GigaAM-v3 ctc", "device": "cuda",
        "model_dir": HOME / ".cache/whisper-dictate/models/gigaam-v3-ctc",
    },
}
GIGAAM_BACKENDS = {"gigaam-v3-e2e-rnnt", *GIGAAM_ONNX_BACKENDS}
logging.basicConfig(filename=LOG, level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
model = None
model_profile = None
active_asr_backend = "faster-whisper"
active_asr_error_category = None
active_asr_model_name = None
active_asr_device = None
gigaam_worker = None
gigaam_onnx_worker = None
gigaam_onnx_cpu_worker = None
model_lease = 0
qwen_lease = 0
queue = asyncio.Queue()
correction_queue = asyncio.Queue()
control_lock = asyncio.Lock()
correction_execution_lock = asyncio.Lock()
correction_pending = 0
correction_idle = asyncio.Event()
correction_idle.set()
manual_correction_cancel_events = set()
manual_correction_started_at = None
latest_live_dictation_asr_at = 0.0

def clone_default():
    return json.loads(json.dumps(DEFAULT))

def restore_local_profile_defaults(value):
    """Reset an invalid local profile without discarding valid Cloudflare choices."""
    defaults = clone_default()
    for key in ("model", "language", "device", "compute_type", "beam_size", "best_of",
                "temperature", "vad", "condition_on_previous_text", "initial_prompt",
                "glossary", "result_mode", "fallback", "corrector", "asr_backend",
                "gigaam_model_dir"):
        value[key] = defaults[key]
    return value

def select_job_backends(source, profile):
    """Allow Cloudflare ASR only live; correction only for opted-in scenarios."""
    live = source == "live_dictation"
    correction_allowed = source in {"live_dictation", "manual_correction"}
    provider = "cloudflare" if live and profile.get("cloudflare_asr_enabled") is True else "local"
    cloud_corrector = (profile.get("cloudflare_corrector", {}) if correction_allowed
                       else {"enabled": False})
    return provider, cloud_corrector

async def manual_correct_text(req):
    """Run explicitly requested clipboard text through configured correction only."""
    global correction_pending, manual_correction_started_at
    global latest_live_dictation_asr_at
    text = req.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 12000:
        return {"status": "error", "error_code": "invalid_correction_text"}
    cancel_event = threading.Event()
    manual_correction_cancel_events.add(cancel_event)
    manual_correction_started_at = time.time()
    correction_started_at = manual_correction_started_at
    write_manual_correction_status({"state": "waiting", "started_at": manual_correction_started_at,
                                    "updated_at": time.time()})
    reserved = False
    selected = {"provider": None, "model": None, "device": None}
    attempts = []
    try:
        while True:
            if cancel_event.is_set():
                return {"status": "error", "error_code": "cancelled_by_user"}
            try:
                await asyncio.wait_for(queue.join(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            try:
                await asyncio.wait_for(correction_idle.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            async with control_lock:
                if cancel_event.is_set():
                    return {"status": "error", "error_code": "cancelled_by_user"}
                if not queue.empty() or correction_pending:
                    continue
                profile = config()
                corrector = profile["corrector"]
                _provider, cloud_corrector = select_job_backends("manual_correction", profile)
                correction_pending += 1
                correction_idle.clear()
                reserved = True
                break

        def report_manual_backend(provider, model_or_device=None, attempt_number=None):
            if provider == "local":
                model_name = corrector.get("model", "")
                device = model_or_device if model_or_device in {"cpu", "cuda"} else None
            else:
                model_name = model_or_device
                device = "remote"
            if provider not in {"local", "codex", "claude", "cloudflare",
                                "groq", "google", "openrouter"}:
                return
            if not isinstance(model_name, str) or not re.fullmatch(
                    r"[A-Za-z0-9@_.:/+\-]{1,160}", model_name):
                model_name = ""
            selected.update({"provider": provider, "model": model_name, "device": device})
            attempt = (attempt_number if type(attempt_number) is int and
                       1 <= attempt_number <= 99 else len(attempts) + 1)
            write_manual_correction_status({
                "state": "running", "provider": provider, "model": model_name,
                "device": device, "attempt": attempt,
                "started_at": manual_correction_started_at,
                "updated_at": time.time()})

        try:
            async with correction_execution_lock:
                corrected, fallback, elapsed, backend, _usage = await asyncio.to_thread(
                    correct_text_sync, text, corrector, cloud_corrector,
                    backend_callback=report_manual_backend,
                    provider_settings=profile.get("multi_provider"),
                    source="manual_correction", attempt_log=attempts,
                    cancel_callback=cancel_event.is_set)
        except Exception:
            if cancel_event.is_set():
                return {"status": "error", "error_code": "cancelled_by_user"}
            logger.warning("Manual correction failed")
            return {"status": "error", "error_code": "correction_failed"}
        if cancel_event.is_set() or fallback == "cancelled_by_user":
            return {"status": "error", "error_code": "cancelled_by_user"}
        return {"status": "ok", "text": corrected, "backend": backend,
                "provider": selected["provider"] or backend,
                "model": selected["model"], "fallback": fallback,
                "seconds": elapsed, "attempts": attempts,
                "started_at": correction_started_at,
                "copy_to_clipboard": latest_live_dictation_asr_at <= correction_started_at}
    finally:
        manual_correction_cancel_events.discard(cancel_event)
        if not manual_correction_cancel_events:
            write_manual_correction_status(None)
            manual_correction_started_at = None
        if reserved:
            correction_pending = max(0, correction_pending - 1)
            if correction_pending == 0:
                correction_idle.set()


async def cancel_manual_correction():
    """Signal active manual correction workers and clear orphaned status markers."""
    global manual_correction_started_at
    active = list(manual_correction_cancel_events)
    if active:
        for event in active:
            event.set()
        write_manual_correction_status({"state": "cancelling",
                                        "started_at": manual_correction_started_at,
                                        "updated_at": time.time()})
    else:
        write_manual_correction_status(None)
        manual_correction_started_at = None
    return {"status": "ok", "cancel_requested": bool(active)}


def write_manual_correction_status(value):
    """Persist status-only fields; transcript text is never written here."""
    temporary_path = None
    try:
        if value is None:
            MANUAL_CORRECTION_STATUS.unlink(missing_ok=True)
            return
        fd, temporary_path = tempfile.mkstemp(
            dir=CACHE, prefix=".manual-correction-status-", suffix=".tmp")
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, separators=(",", ":"))
            stream.write("\n")
        os.replace(temporary_path, MANUAL_CORRECTION_STATUS)
        os.chmod(MANUAL_CORRECTION_STATUS, 0o600)
    except OSError:
        logger.warning("Could not update manual correction status")
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass

def config():
    value = clone_default()
    raw = {}
    if CONFIG.exists():
        try:
            raw = json.loads(CONFIG.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("root")
            for key in value:
                if key in raw: value[key] = raw[key]
            if "cloudflare_asr_enabled" not in raw and isinstance(raw.get("cloudflare_workers_ai_enabled"), bool):
                value["cloudflare_asr_enabled"] = raw["cloudflare_workers_ai_enabled"]
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Invalid configuration; safe defaults used")
            return clone_default()
    if value.get("asr_backend") not in {"faster-whisper", *GIGAAM_BACKENDS}:
        value["asr_backend"] = "faster-whisper"
    # Keep custom-code execution at one fixed local cache path.
    value["gigaam_model_dir"] = clone_default()["gigaam_model_dir"]
    valid = (value["model"] in MODELS and value["device"] in {"cuda", "cpu"} and
             value["compute_type"] in COMPUTES and isinstance(value["language"], str) and
             2 <= len(value["language"]) <= 12 and isinstance(value["beam_size"], int) and
             1 <= value["beam_size"] <= 10 and isinstance(value["best_of"], int) and
             1 <= value["best_of"] <= 10 and isinstance(value["temperature"], (int, float)) and
             0 <= value["temperature"] <= 1 and isinstance(value["vad"], bool) and isinstance(value["condition_on_previous_text"], bool) and
             isinstance(value["initial_prompt"], str) and len(value["initial_prompt"]) <= 800 and
             isinstance(value["glossary"], list) and value["result_mode"] in {"clipboard", "autopaste"})
    fallback = value["fallback"]
    valid = valid and isinstance(fallback, dict) and fallback.get("model") in MODELS and fallback.get("device") in {"cuda", "cpu"} and fallback.get("compute_type") in COMPUTES and isinstance(fallback.get("allow_cpu"), bool) and (fallback.get("model"), fallback.get("device"), fallback.get("compute_type")) != (value["model"], value["device"], value["compute_type"])
    corrector = value["corrector"]
    valid = valid and isinstance(corrector, dict) and isinstance(corrector.get("enabled"), bool) and corrector.get("provider") == "ollama" and isinstance(corrector.get("model"), str) and 1 <= len(corrector["model"]) <= 160 and isinstance(corrector.get("prompt"), str) and len(corrector["prompt"]) <= 1500 and type(corrector.get("timeout_seconds")) is int and corrector["timeout_seconds"] in {5, 10, 15, 30, 60, 120, 300} and isinstance(corrector.get("max_length_ratio"), (int, float)) and 1 <= corrector["max_length_ratio"] <= 3 and type(corrector.get("num_gpu")) is int and corrector["num_gpu"] in {0, 4, 8, 12} and corrector.get("keep_alive") in {0, "30s", "5m"} and type(corrector.get("temperature")) in {int, float} and corrector["temperature"] in {0, 0.1, 0.2} and type(corrector.get("num_ctx")) is int and corrector["num_ctx"] in {1024, 2048, 4096, 8192} and type(corrector.get("num_thread")) is int and corrector["num_thread"] in {2, 4, 8}
    if not valid:
        logger.warning("Local profile failed validation; local defaults used")
        restore_local_profile_defaults(value)
    cloud_asr = clone_default()["cloudflare_asr"]
    raw_asr = raw.get("cloudflare_asr")
    if isinstance(raw_asr, dict):
        cloud_asr.update(raw_asr)
    else:
        cloud_asr["model"] = raw.get("cloudflare_asr_model", cloud_asr["model"])
    valid_asr = (cloud_asr.get("model") in {"@cf/openai/whisper-large-v3-turbo", "@cf/openai/whisper"} and
        cloud_asr.get("task") in {"transcribe", "translate"} and
        cloud_asr.get("language") in {"", "ru", "en"} and type(cloud_asr.get("vad_filter")) is bool and
        type(cloud_asr.get("beam_size")) is int and cloud_asr["beam_size"] in {1, 3, 5, 10} and
        isinstance(cloud_asr.get("initial_prompt"), str) and len(cloud_asr["initial_prompt"]) <= 800 and
        isinstance(cloud_asr.get("prefix"), str) and len(cloud_asr["prefix"]) <= 200 and
        type(cloud_asr.get("condition_on_previous_text")) is bool and
        type(cloud_asr.get("no_speech_threshold")) in (int, float) and 0 <= cloud_asr["no_speech_threshold"] <= 1 and
        type(cloud_asr.get("compression_ratio_threshold")) in (int, float) and 1 <= cloud_asr["compression_ratio_threshold"] <= 5 and
        type(cloud_asr.get("log_prob_threshold")) in (int, float) and -5 <= cloud_asr["log_prob_threshold"] <= 0 and
        (cloud_asr.get("hallucination_silence_threshold") is None or cloud_asr["hallucination_silence_threshold"] in {2, 4, 8}))
    if not valid_asr:
        logger.warning("Cloudflare Whisper configuration failed validation; using safe defaults")
        cloud_asr = clone_default()["cloudflare_asr"]
    value["cloudflare_asr"] = cloud_asr
    value["cloudflare_asr_model"] = cloud_asr["model"]
    cloud_corrector = value["cloudflare_corrector"]
    if isinstance(cloud_corrector, dict):
        cloud_corrector = {**clone_default()["cloudflare_corrector"], **cloud_corrector}
        if cloud_corrector.get("priority_model") not in ("auto", *CLOUDFLARE_CORRECTOR_MODELS):
            cloud_corrector["priority_model"] = "auto"
        value["cloudflare_corrector"] = cloud_corrector
    if not isinstance(value["cloudflare_asr_enabled"], bool):
        logger.warning("Cloudflare ASR toggle invalid; ASR remains local")
        value["cloudflare_asr_enabled"] = False
    valid_cloud = (isinstance(cloud_corrector, dict) and isinstance(cloud_corrector.get("enabled"), bool) and
        cloud_corrector.get("model") in CLOUDFLARE_CORRECTOR_MODELS and
        cloud_corrector.get("priority_model") in ("auto", *CLOUDFLARE_CORRECTOR_MODELS) and
        type(cloud_corrector.get("temperature")) in (int, float) and 0 <= cloud_corrector["temperature"] <= 5 and
        type(cloud_corrector.get("max_tokens")) is int and cloud_corrector["max_tokens"] in {256, 512, 1024, 2048, 4096} and
        type(cloud_corrector.get("retry_attempts")) is int and 0 <= cloud_corrector["retry_attempts"] <= 2)
    if not valid_cloud:
        logger.warning("Cloudflare corrector configuration failed validation; safe settings used")
        enabled = isinstance(cloud_corrector, dict) and cloud_corrector.get("enabled") is True
        cloud_corrector = clone_default()["cloudflare_corrector"]
        cloud_corrector["enabled"] = enabled
        value["cloudflare_corrector"] = cloud_corrector
    value["multi_provider"] = safe_provider_settings(value.get("multi_provider"))
    value["glossary"] = [x for x in value["glossary"] if isinstance(x, str) and x.strip()][:50]
    return value

def request_profile(request, base):
    value = dict(base)
    for key in ("model", "language", "device", "compute_type", "beam_size", "best_of", "temperature", "vad"):
        if key in request: value[key] = request[key]
    valid = value["model"] in MODELS and value["device"] in {"cuda", "cpu"} and value["compute_type"] in COMPUTES and isinstance(value["language"], str) and 2 <= len(value["language"]) <= 12 and isinstance(value["beam_size"], int) and 1 <= value["beam_size"] <= 10 and isinstance(value["best_of"], int) and 1 <= value["best_of"] <= 10 and isinstance(value["temperature"], (int, float)) and 0 <= value["temperature"] <= 1 and isinstance(value["vad"], bool)
    if not valid: raise ValueError("profile")
    return value

def prompt(value):
    terms = ", ".join(value["glossary"])
    return " ".join(x for x in (value["initial_prompt"].strip(), f"Термины: {terms}" if terms else "") if x)[:1000] or None

def duration(path):
    try:
        with wave.open(path, "rb") as item: return round(item.getnframes() / item.getframerate(), 3)
    except (OSError, wave.Error, ZeroDivisionError): return None

def job_raw_path(job_id):
    return RAW_ASR / f"{job_id}.txt" if isinstance(job_id, str) and job_id.isdigit() and len(job_id) <= 32 else None

def correction_cancelled(job_id):
    return bool(isinstance(job_id, str) and job_id.isdigit() and len(job_id) <= 32 and
                (RECOVERY_CANCEL / f"{job_id}.cancel").exists())

def save_live_raw_asr(job_id, source, text):
    """Atomically retain live ASR only, outside metrics and logs, for tray recovery."""
    path = job_raw_path(job_id) if source == "live_dictation" else None
    if path is None or not isinstance(text, str) or not text.strip():
        return False
    temporary = path.with_suffix(".txt.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        return True
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        logger.warning("Could not persist raw ASR for live job")
        return False

def save_latest_live_asr_at(timestamp):
    """Persist only completion time so an older manual result cannot overwrite newer dictation."""
    temporary = LATEST_LIVE_ASR_AT.with_suffix(".tmp")
    try:
        temporary.write_text(f"{timestamp:.6f}", encoding="ascii")
        os.chmod(temporary, 0o600)
        os.replace(temporary, LATEST_LIVE_ASR_AT)
        return True
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        logger.warning("Could not persist latest live ASR timestamp")
        return False

def is_oom(error):
    return "out of memory" in str(error).lower() or "cuda oom" in str(error).lower()

def local_model_source(name):
    """Resolve the locally cached experimental turbo model without a hub lookup."""
    if name != "large-v3-turbo":
        return name
    ref = HOME / ".cache/huggingface/hub/models--deepdml--faster-whisper-large-v3-turbo-ct2/refs/main"
    try:
        revision = ref.read_text(encoding="utf-8").strip()
        snapshot = ref.parent.parent / "snapshots" / revision
        if snapshot.is_dir():
            return str(snapshot)
    except OSError:
        pass
    raise FileNotFoundError("Local large-v3-turbo snapshot is unavailable")

async def release():
    global model, model_profile, active_asr_backend, active_asr_model_name
    global active_asr_device, gigaam_worker, gigaam_onnx_worker
    global gigaam_onnx_cpu_worker
    if gigaam_worker is not None:
        await asyncio.to_thread(gigaam_worker.close)
        gigaam_worker = None
    if gigaam_onnx_worker is not None:
        await asyncio.to_thread(gigaam_onnx_worker.close)
        gigaam_onnx_worker = None
    if gigaam_onnx_cpu_worker is not None:
        await asyncio.to_thread(gigaam_onnx_cpu_worker.close)
        gigaam_onnx_cpu_worker = None
    model = None; model_profile = None; gc.collect()
    active_asr_backend = "faster-whisper"
    active_asr_model_name = None
    active_asr_device = None

async def prepare_remote_models(corrector_model):
    """Drop daemon-owned local models before a remote stage (single worker)."""
    await release()
    await asyncio.to_thread(unload_corrector_model, corrector_model)

async def delayed_whisper_release(delay=2.0):
    """Release after successful remote use, unless a newer job acquired a lease."""
    global model_lease
    model_lease += 1
    lease = model_lease
    await asyncio.sleep(delay)
    if lease == model_lease:
        await release()

async def delayed_qwen_release(model_name, delay=2.0):
    """Release Qwen after remote success unless a newer job has acquired it."""
    lease = qwen_lease
    await asyncio.sleep(delay)
    if lease == qwen_lease:
        await asyncio.to_thread(unload_corrector_model, model_name)

async def prepare_local_asr(value, fallback_on_backend_error=True,
                           execution_device=None):
    """Make the selected ASR the resident local model before inference."""
    if execution_device != "cpu":
        for model_name in local_corrector_model_names(value):
            await asyncio.to_thread(unload_corrector_model, model_name)
    return await ensure(value, fallback_on_backend_error=fallback_on_backend_error,
                        execution_device=execution_device)

def available_memory_bytes():
    """Return Linux MemAvailable, or None when the kernel value is unavailable."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None

def release_qwen_under_memory_pressure(corrector):
    """Keep configured Ollama residency unless free host RAM drops below 6 GiB."""
    available = available_memory_bytes()
    if available is not None and available < 6 * 1024**3:
        logger.warning("Available RAM is low; releasing local Qwen model")
        return unload_corrector_model(corrector["model"])
    return True

async def ensure(value, fallback_on_backend_error=True, execution_device=None):
    global model, model_profile, model_lease, active_asr_backend, active_asr_error_category
    global active_asr_model_name, active_asr_device, gigaam_worker, gigaam_onnx_worker
    global gigaam_onnx_cpu_worker
    model_lease += 1
    selected_backend = value.get("asr_backend", "faster-whisper")
    if selected_backend == "gigaam-v3-e2e-rnnt":
        profile = {"asr_backend": selected_backend, "model": GIGAAM_MODEL_NAME, "device": "cpu"}
    elif selected_backend in GIGAAM_ONNX_BACKENDS:
        spec = GIGAAM_ONNX_BACKENDS[selected_backend]
        profile = {"asr_backend": selected_backend, "model": spec["model"],
                   "device": (execution_device if execution_device in {"cpu", "cuda"}
                              else spec["device"])}
    else:
        profile = (
               {"asr_backend": "faster-whisper", "model": value["model"],
                "device": value["device"], "compute_type": value["compute_type"]})
    onnx_worker = (gigaam_onnx_cpu_worker if profile["device"] == "cpu"
                   else gigaam_onnx_worker)
    if ((selected_backend == "gigaam-v3-e2e-rnnt" and gigaam_worker is not None and
         model_profile == profile) or
            (selected_backend in GIGAAM_ONNX_BACKENDS and onnx_worker is not None and
             model_profile == profile) or
            (selected_backend not in GIGAAM_BACKENDS and model is not None and
             gigaam_worker is None and gigaam_onnx_worker is None and
             gigaam_onnx_cpu_worker is None and model_profile == profile)):
        return False
    await release()
    active_asr_error_category = None
    if selected_backend == "gigaam-v3-e2e-rnnt":
        logger.info("Loading local model=%s device=cpu", GIGAAM_MODEL_NAME)
        try:
            gigaam_worker = await asyncio.to_thread(GigaAMWorker, value["gigaam_model_dir"])
            active_asr_backend = "gigaam-v3-e2e-rnnt"
            active_asr_model_name = GIGAAM_MODEL_NAME
            active_asr_device = "cpu"
            model_profile = profile
            logger.info("Model loaded model=%s device=cpu backend=%s", GIGAAM_MODEL_NAME,
                        active_asr_backend)
            return True
        except Exception as error:
            active_asr_error_category = gigaam_error_category(error)
            if gigaam_worker is not None:
                await asyncio.to_thread(gigaam_worker.close)
                gigaam_worker = None
            logger.warning("GigaAM initialization failed category=%s",
                           active_asr_error_category)
            model = None
            if not fallback_on_backend_error:
                raise
    elif selected_backend in GIGAAM_ONNX_BACKENDS:
        spec = GIGAAM_ONNX_BACKENDS[selected_backend]
        actual_device = profile["device"]
        logger.info("Loading local model=%s device=%s backend=%s", spec["model"],
                    actual_device, selected_backend)
        try:
            if actual_device == "cpu":
                gigaam_onnx_cpu_worker = await asyncio.to_thread(
                    GigaAMOnnxCpuWorker, str(spec["model_dir"]), selected_backend)
            else:
                gigaam_onnx_worker = await asyncio.to_thread(
                    GigaAMOnnxWorker, str(spec["model_dir"]), selected_backend)
            active_asr_backend = selected_backend
            active_asr_model_name = spec["model"]
            active_asr_device = actual_device
            model_profile = profile
            logger.info("Model loaded model=%s device=%s backend=%s", spec["model"],
                        actual_device,
                        selected_backend)
            return True
        except Exception as error:
            active_asr_error_category = (gigaam_onnx_cpu_error_category(error)
                                         if actual_device == "cpu" else
                                         gigaam_onnx_error_category(error))
            if gigaam_onnx_worker is not None:
                await asyncio.to_thread(gigaam_onnx_worker.close)
                gigaam_onnx_worker = None
            if gigaam_onnx_cpu_worker is not None:
                await asyncio.to_thread(gigaam_onnx_cpu_worker.close)
                gigaam_onnx_cpu_worker = None
            logger.warning("GigaAM ONNX initialization failed category=%s",
                           active_asr_error_category)
            model = None
            if not fallback_on_backend_error:
                raise
    active_asr_backend = "faster-whisper"
    active_asr_model_name = value["model"]
    active_asr_device = value["device"]
    profile = {"asr_backend": "faster-whisper", "model": value["model"],
               "device": value["device"], "compute_type": value["compute_type"]}
    if model is not None and model_profile == profile: return False
    logger.info("Loading local model=%s device=%s compute_type=%s", profile["model"],
                profile["device"], profile["compute_type"])
    model = await asyncio.to_thread(WhisperModel, local_model_source(profile["model"]),
                                    device=profile["device"],
                                    compute_type=profile["compute_type"], local_files_only=True)
    model_profile = profile
    logger.info("Model loaded model=%s device=%s compute_type=%s", profile["model"], profile["device"], profile["compute_type"])
    return True

def transcribe_sync(wav, value, cancel_marker=None):
    if active_asr_backend == "gigaam-v3-e2e-rnnt":
        text, _window_count = gigaam_worker.transcribe(wav, cancel_marker=cancel_marker)
        return text
    if active_asr_backend in GIGAAM_ONNX_BACKENDS:
        worker = (gigaam_onnx_cpu_worker if active_asr_device == "cpu"
                  else gigaam_onnx_worker)
        if worker is None:
            raise ProviderError("local", "worker_unavailable")
        text, _window_count = worker.transcribe(wav, cancel_marker=cancel_marker)
        return text
    segments, _ = model.transcribe(wav, language=value["language"], vad_filter=value["vad"],
        condition_on_previous_text=value["condition_on_previous_text"],
        beam_size=value["beam_size"], best_of=value["best_of"], temperature=value["temperature"],
        initial_prompt=prompt(value))
    return "".join(item.text for item in segments).strip()

CORRECTOR_INSTRUCTION = """Верни только готовый текст без комментариев. Исправь однозначные ошибки ASR, орфографию и пунктуацию, сохраняя смысл, факты, порядок смысловых слов и стиль говорящего. Удаляй речевые заполнители и слова-паразиты, не несущие смысла: «э-э», «а-а», «м-м», «хм», случайные повторы, незаконченные фрагменты слов перед самокоррекцией и ложные начала фраз; убирай «ну», «вот», «как бы», «значит» и подобные слова только когда они служат заполнителями пауз. Не удаляй эти слова, если они передают смысл, связь или отношение говорящего. Не сокращай содержание, не перефразируй, не добавляй факты, объяснения, предложения или обращения. Удали также явные служебные хвосты ASR в конце: «продолжение следует», титры/субтитры, включая подпись «Редактор субтитров А. Семкин. Корректор А. Егорова.»; если это занимает весь текст, верни пустой текст. Не удаляй осмысленные повторы для акцента. Расставь естественные знаки препинания и заверши текст точкой, вопросительным или восклицательным знаком. Сохраняй имена, команды, пути, URL, код, английские слова, числа и даты. Не добавляй кавычки, «ёлочки», обрамление или цитирование фраз, если их нет в исходном тексте; существующие кавычки сохраняй. При сомнении сохрани фрагмент, если удаление может изменить смысл. Обработай текст целиком до последнего слова."""
RETRY_INSTRUCTION = """Предыдущий ответ мог оборваться. Верни заново весь текст целиком, от первого до последнего слова, без сокращений. Сохрани смысл и все смысловые слова; исправь пунктуацию, заглавные буквы и однозначные ошибки ASR, удали только речевые заполнители и ложные начала согласно инструкции. Не убирай слова, которые в контексте несут смысл."""
MAX_CORRECTION_CHARS = 2400

def autoreplacements():
    """Read user-maintained `source => replacement` rules without logging text."""
    try:
        rows = REPLACEMENTS.read_text(encoding="utf-8").splitlines()[:200]
    except OSError:
        return []
    rules = []
    for row in rows:
        row = row.strip()
        if not row or row.startswith("#") or "=>" not in row:
            continue
        source, replacement = (part.strip() for part in row.split("=>", 1))
        if 1 <= len(source) <= 80 and len(replacement) <= 160:
            rules.append((source, replacement))
    return sorted(rules, key=lambda rule: len(rule[0]), reverse=True)

def remove_subtitle_credit_tail(text):
    """Remove a known Whisper subtitle-editor signature only at transcript end."""
    if not isinstance(text, str):
        return text
    name = r"(?:[А-ЯЁ]\.\s*){0,2}[А-ЯЁ][а-яё-]{1,}"
    signature = re.compile(
        r"(?:^|[\s,;:—–-]+)редактор\s+субтитров"
        r"[\s,:—–-]+" + name +
        r"(?:[.!?…\s—–-]+корректор(?:[\s,:—–-]+" + name + r")?)?"
        r"[.!?…]*\s*$", re.IGNORECASE)
    match = signature.search(text)
    return text[:match.start()].rstrip() if match else text

def normalize_asr_text(text):
    """Apply deterministic user terms and remove only known trailing ASR artefacts."""
    applied = 0
    for source, replacement in autoreplacements():
        # A replacement is deliberately limited to a space-delimited token.
        # This prevents rules such as "Тера => Terra" changing a part of a
        # longer or mixed-script word (for example, "масТера").
        text, count = re.subn(r"(?<=\s)" + re.escape(source) + r"(?=\s)", replacement,
                              text, flags=re.IGNORECASE)
        applied += count
    removed = 0
    known_tail = re.compile(r"(?:[\s,;:—–-]+)(?:продолжение следует|"
                            r"субтитры (?:подготовлены|подготовил|сделал|делал)"
                            r"(?:[\s,;:—–-]+[\w.-]+){0,6})[.!?…]*\s*$", re.IGNORECASE)
    repeated_tail = re.compile(r"(?:[\s,;:.!?…—–-]+)([\w-]{2,})(?:[\s,;:.!?…—–-]+\1){4,}[.!?…]*\s*$", re.IGNORECASE)
    while True:
        match = known_tail.search(text) or repeated_tail.search(text)
        if not match:
            break
        text = text[:match.start()].rstrip()
        removed += 1
    if applied or removed:
        logger.info("ASR normalization: replacements=%d removed_tails=%d", applied, removed)
    return text

def finalize_text(text):
    text = text.strip()
    return text if not text or text[-1] in ".!?…" else text + "."

def correction_chunks(text):
    """Split long dictation on word boundaries so input and output fit Qwen context."""
    text = text.strip()
    if len(text) <= MAX_CORRECTION_CHARS:
        return [text]
    chunks, rest = [], text
    while len(rest) > MAX_CORRECTION_CHARS:
        cut = rest.rfind(" ", 0, MAX_CORRECTION_CHARS + 1)
        if cut < MAX_CORRECTION_CHARS // 2:
            cut = MAX_CORRECTION_CHARS
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        chunks.append(rest)
    return chunks

def correction_response(text, corrector, instruction):
    payload = json.dumps({"model": corrector["model"], "prompt": instruction + "\n\nТекст:\n" + text,
        "stream": False, "think": False, "keep_alive": corrector["keep_alive"],
        "options": {"temperature": corrector["temperature"], "num_gpu": corrector["num_gpu"],
                    "num_ctx": corrector["num_ctx"], "num_thread": corrector["num_thread"],
                    "num_predict": 4096}}).encode("utf-8")
    request = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=corrector["timeout_seconds"]) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result.get("response", ""), result.get("done_reason")

_FILLER_SOUNDS_RE = re.compile(
    r"(?iu)(?<!\w)(?:э+(?:[-–—]э+)+|а+(?:[-–—]а+)+|"
    r"м+(?:[-–—]м+)+|хм+|э+|а{2,}|м{2,})(?!\w)")
_FILLER_PHRASES = (
    ("это", "самое"), ("так", "сказать"), ("как", "бы"),
    ("в", "общем"),
)
_FILLER_WORDS = {"ну", "вот", "значит", "типа", "короче", "собственно"}

def correction_safety_words(text):
    """Ignore only likely disfluencies when checking meaningful word coverage."""
    normalized = _FILLER_SOUNDS_RE.sub(" ", text.lower())
    words = re.findall(r"\w+", normalized, flags=re.UNICODE)
    without_phrases = []
    index = 0
    while index < len(words):
        phrase = next((part for part in _FILLER_PHRASES
                       if tuple(words[index:index + len(part)]) == part), None)
        if phrase:
            index += len(phrase)
            continue
        without_phrases.append(words[index])
        index += 1
    meaningful = []
    for index, word in enumerate(without_phrases):
        if word in _FILLER_WORDS:
            continue
        next_word = without_phrases[index + 1] if index + 1 < len(without_phrases) else ""
        # Drop a clearly truncated prefix ("корре корректор") for coverage
        # checks, while avoiding short meaningful words such as "да" or "по".
        if len(word) >= 3 and len(next_word) > len(word) and next_word.startswith(word):
            continue
        if meaningful and word == meaningful[-1]:
            continue
        meaningful.append(word)
    return meaningful

def correction_safety_reason(source, corrected, done_reason):
    """Reject shortened/cut-off replies while permitting filler removal.

    Compare the meaningful word stream after removing known filler forms and
    accidental adjacent repeats. This lets the corrector clean speech without
    weakening protection against omitted semantic content.
    """
    if not isinstance(corrected, str):
        return "invalid"
    corrected = corrected.strip()
    source_words = correction_safety_words(source)
    corrected_words = correction_safety_words(corrected)
    source_clause_punctuation = len(re.findall(r"[,;:—–]", source))
    corrected_clause_punctuation = len(re.findall(r"[,;:—–]", corrected))
    corrected_sentence_punctuation = len(re.findall(r"[.!?…]", corrected))
    source_core_length = sum(map(len, source_words)) + max(0, len(source_words) - 1)
    corrected_core_length = sum(map(len, corrected_words)) + max(0, len(corrected_words) - 1)
    if ((not corrected and source_words) or done_reason == "length" or chr(96) * 3 in corrected or
            any(ord(char) < 32 and char not in "\n\t\r" for char in corrected) or
            len(corrected) > len(source) * 1.5 or
            (source_core_length and corrected_core_length < source_core_length * 0.88)):
        return "length"
    # Reject catastrophic punctuation loss while allowing normal punctuation
    # edits and sentence splitting. One final period is not enough when a
    # multi-clause source had several commas, dashes, colons, or semicolons.
    if (source_clause_punctuation >= 3 and corrected_clause_punctuation == 0 and
            corrected_sentence_punctuation <= 1):
        return "punctuation_loss"
    if len(corrected_words) < len(source_words) * 0.9:
        return "word_coverage_order"
    # For longer text, require at least 90% of source words to remain in order.
    # This tolerates a few genuine ASR spelling fixes but catches omissions from
    # the middle that a last-word-only check cannot see.
    if len(source_words) >= 20:
        kept = sum(size for _, _, size in difflib.SequenceMatcher(
            a=source_words, b=corrected_words, autojunk=False).get_matching_blocks())
        if kept < len(source_words) * 0.9:
            return "word_coverage_order"
    tail = source_words[-min(3, len(source_words)):]
    corrected_tail = corrected_words[-16:]
    # A legitimate correction can inflect a final Russian word ("хостинг" →
    # "хостингу").  Accept that narrow case while still rejecting a missing
    # tail or a semantically unrelated replacement.
    def tail_word_present(word):
        return any(word == candidate or (len(word) >= 5 and
                   (candidate.startswith(word) or word.startswith(candidate)))
                   for candidate in corrected_tail)
    return None if all(tail_word_present(word) for word in tail) else "tail"

def correction_is_safe(source, corrected, done_reason):
    return correction_safety_reason(source, corrected, done_reason) is None

def ollama_loaded_device(model_name, requested_device):
    """Read Ollama's loaded-model VRAM metadata without logging user content."""
    request = urllib.request.Request("http://127.0.0.1:11434/api/ps", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=1.5) as response:
            models = json.load(response).get("models", [])
    except (OSError, ValueError, TypeError, urllib.error.URLError, TimeoutError):
        return requested_device
    for loaded in models:
        if loaded.get("name") == model_name or loaded.get("model") == model_name:
            try:
                return "cuda" if int(loaded.get("size_vram", 0)) > 0 else "cpu"
            except (TypeError, ValueError):
                return requested_device
    return requested_device

def correct_text_sync(text, corrector, cloudflare_profile=None, backend_callback=None,
                      cloudflare_failure_callback=None, provider_settings=None, source=None,
                      provider_failure_callback=None, attempt_log=None, cancel_callback=None):
    """Correct safely in bounded chunks; respect configured OpenRouter retries."""
    raw_asr_text = text
    correction_settings = safe_provider_settings(provider_settings)["correction"]
    if not correction_settings["enabled"]:
        if backend_callback:
            backend_callback("local")
        return raw_asr_text, None, 0.0, "local_disabled", None
    text = normalize_asr_text(text)
    def cancelled_result(started):
        return raw_asr_text, "cancelled_by_user", round(time.monotonic() - started, 3), "cancelled", None
    def cancelled():
        try:
            return bool(cancel_callback and cancel_callback())
        except OSError:
            return False
    cloudflare_enabled = bool(cloudflare_profile and cloudflare_profile.get("enabled"))
    routes = daily_only_routes(
        multi_correction_chain(source, provider_settings, cloudflare_enabled))
    remote_routes = [route for route in routes if route != "local"]
    if not corrector.get("enabled", False) and not remote_routes:
        if backend_callback:
            backend_callback("local")
        return finalize_text(text), None, 0.0, "local_disabled", None
    started = time.monotonic()
    cloudflare_fallback = bool(remote_routes)

    def run_local_correction(fallback_stage=False):
        if not corrector.get("enabled", False):
            if backend_callback:
                backend_callback("local")
            return finalize_text(text), None, 0.0, "local_disabled", None
        configured_models = correction_settings["local_models"]
        model_order = configured_models or [corrector["model"]]
        last_result = None
        for model_index, model_name in enumerate(model_order):
            if cancelled():
                return cancelled_result(started)
            corrector["model"] = model_name
            local_corrector = dict(corrector)
            if backend_callback:
                backend_callback("local", ollama_loaded_device(
                    model_name, "cuda" if int(local_corrector.get("num_gpu", 0) or 0) > 0 else "cpu"))
            local_started = time.monotonic()
            try:
                if local_corrector.get("provider") != "ollama":
                    raise ValueError("unsupported corrector profile")
                custom_prompt = local_corrector.get("prompt", "").strip()
                instruction = CORRECTOR_INSTRUCTION + ("\n\nДополнительные указания пользователя:\n" + custom_prompt if custom_prompt else "")
                corrected_chunks, fallback_chunks = [], []
                chunks = correction_chunks(text)
                for index, chunk in enumerate(chunks, start=1):
                    if cancelled():
                        return cancelled_result(started)
                    corrected, done_reason = correction_response(chunk, local_corrector, instruction)
                    if cancelled():
                        return cancelled_result(started)
                    if index == 1 and backend_callback:
                        backend_callback("local", ollama_loaded_device(
                            model_name, "cuda" if int(local_corrector.get("num_gpu", 0) or 0) > 0 else "cpu"))
                    if not correction_is_safe(chunk, corrected, done_reason):
                        logger.warning("Qwen reply rejected; retrying chunk=%d/%d done_reason=%s",
                                       index, len(chunks), done_reason)
                        if backend_callback:
                            backend_callback("local", ollama_loaded_device(
                                model_name, "cuda" if int(local_corrector.get("num_gpu", 0) or 0) > 0 else "cpu"),
                                len(attempt_log or []) + 2)
                        corrected, done_reason = correction_response(
                            chunk, local_corrector, instruction + "\n\n" + RETRY_INSTRUCTION)
                        if cancelled():
                            return cancelled_result(started)
                    if correction_is_safe(chunk, corrected, done_reason):
                        corrected_chunks.append(corrected.strip())
                    else:
                        logger.warning("Qwen retry rejected; raw ASR retained chunk=%d/%d done_reason=%s",
                                       index, len(chunks), done_reason)
                        corrected_chunks.append(chunk)
                        fallback_chunks.append(index)
                fallback = None if not fallback_chunks else (
                    "raw_asr_after_retry:" + ",".join(map(str, fallback_chunks)) +
                    f"/{len(chunks)}")
                if fallback is None:
                    if model_index:
                        fallback = "local_model_fallback"
                    record_attempt(attempt_log, "correction", "local", model_name,
                                   True, local_started, device=("cuda" if
                                   int(local_corrector.get("num_gpu", 0) or 0) > 0 else "cpu"))
                    backend = "local_fallback" if fallback_stage else "local"
                    return finalize_text(" ".join(corrected_chunks)), fallback, \
                        round(time.monotonic() - started, 3), backend, None
                unload_corrector_model(model_name)
                record_attempt(attempt_log, "correction", "local", model_name,
                               False, local_started, "safety_fallback",
                               "cuda" if int(local_corrector.get("num_gpu", 0) or 0) > 0 else "cpu")
                last_result = (finalize_text(" ".join(corrected_chunks)), fallback,
                               round(time.monotonic() - started, 3),
                               "local_fallback" if fallback_stage else "local", None)
            except (OSError, ValueError, KeyError, TypeError, AttributeError,
                    UnicodeDecodeError, json.JSONDecodeError, urllib.error.URLError,
                    TimeoutError):
                unload_corrector_model(model_name)
                record_attempt(attempt_log, "correction", "local", model_name,
                               False, local_started, "corrector_unavailable_or_timeout",
                               "cuda" if int(local_corrector.get("num_gpu", 0) or 0) > 0 else "cpu")
                last_result = (text, "corrector_unavailable_or_timeout",
                               round(time.monotonic() - started, 3),
                               "local_fallback" if fallback_stage else "local", None)
                logger.warning("Local correction model failed category=corrector_unavailable_or_timeout")
        return last_result or (text, "corrector_unavailable_or_timeout",
                               round(time.monotonic() - started, 3),
                               "local_fallback" if fallback_stage else "local", None)

    local_attempt_result = None
    remote_stage_seen = False
    for route in routes:
        if route == "local":
            if corrector.get("enabled", False) and local_attempt_result is None:
                local_attempt_result = run_local_correction(fallback_stage=remote_stage_seen)
                if local_attempt_result[1] is None:
                    return local_attempt_result
            continue
        remote_stage_seen = True
        if cancelled():
            return cancelled_result(started)
        provider_name = {"codex": "codex", "claude": "claude",
                         "groq_primary": "groq", "groq_fallback": "groq",
                         "openrouter": "openrouter", "google": "google",
                         "cloudflare": "cloudflare"}[route]
        if daily_limit_state(provider=provider_name) is not None:
            continue
        correction_settings = safe_provider_settings(provider_settings)["correction"]
        model_name = {"codex": correction_settings["codex_model"],
                      "claude": correction_settings["claude_model"],
                      "groq_primary": "qwen/qwen3.8-27b",
                      "groq_fallback": "openai/gpt-oss-20b",
                      "openrouter": "openrouter/free",
                      "google": "gemini-3.5-flash-lite",
                      "cloudflare": cloudflare_profile.get("model") if cloudflare_profile else None}[route]
        if route == "openrouter":
            model_name = choose_openrouter_fallback_model(METRICS) or model_name
        if route == "cloudflare":
            priority_model = cloudflare_profile.get("priority_model", "auto")
            model_order = cloudflare_correction_model_order(
                METRICS, CLOUDFLARE_CORRECTOR_MODELS,
                priority_model if priority_model in CLOUDFLARE_CORRECTOR_MODELS else None)
            if priority_model in CLOUDFLARE_CORRECTOR_MODELS:
                model_order = [priority_model] + [m for m in model_order if m != priority_model]
            retry_limit = cloudflare_profile.get("retry_attempts", 2)
            if type(retry_limit) is not int or not 0 <= retry_limit <= 2:
                retry_limit = 2
            final_cloudflare_error = None
            for cloudflare_model in model_order[:retry_limit + 1]:
                if cancelled():
                    return cancelled_result(started)
                if backend_callback:
                    backend_callback("cloudflare", cloudflare_model)
                attempt_started = time.monotonic()
                try:
                    estimated_neurons = estimate_cloudflare_correction_neurons(
                        text, cloudflare_profile["max_tokens"], cloudflare_model)
                    if not reserve_cloudflare_daily_neurons(METRICS, estimated_neurons):
                        raise CloudflareError("Расчётный дневной лимит Cloudflare исчерпан",
                                              "daily_quota_exhausted")
                    candidate, input_tokens, output_tokens = cloudflare_correct(
                        text, cloudflare_model, cloudflare_profile["temperature"],
                        cloudflare_profile["max_tokens"])
                    if cancelled():
                        return cancelled_result(started)
                    rejection = correction_safety_reason(text, candidate, None)
                    if rejection:
                        raise ProviderError("cloudflare", "unsafe_response_" + rejection,
                                            cloudflare_model)
                    record_attempt(attempt_log, "correction", "cloudflare", cloudflare_model,
                                   True, attempt_started, device="remote")
                    input_tokens, output_tokens, estimated = correction_token_counts(
                        text, candidate, input_tokens, output_tokens)
                    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                             "estimated": estimated,
                             "estimated_neurons": correction_neurons(
                                 input_tokens, output_tokens, cloudflare_model)}
                    return finalize_text(candidate), None, round(time.monotonic() - started, 3), \
                        "cloudflare", usage
                except Exception as error:
                    if cancelled():
                        return cancelled_result(started)
                    category = safe_provider_error_category(error)
                    failed_model = getattr(error, "model", None) or cloudflare_model
                    record_attempt(attempt_log, "correction", "cloudflare", failed_model,
                                   False, attempt_started, category, "remote")
                    if category == "daily_quota_exhausted":
                        try:
                            mark_daily_limit("cloudflare", category)
                        except OSError:
                            logger.warning("Could not persist Cloudflare daily quota block")
                    if provider_failure_callback:
                        provider_failure_callback("cloudflare", failed_model, category)
                    logger.warning("Cloudflare correction failed model=%s category=%s",
                                   failed_model, category)
                    final_cloudflare_error = error
                    retryable_cloudflare = (
                        category in {"http_error", "network_error", "timeout",
                                     "invalid_response", "unsupported_model",
                                     "provider_unavailable"} or
                        category.startswith(("invalid_response_", "unsafe_response_")))
                    if not retryable_cloudflare or category == "daily_quota_exhausted":
                        break
            if final_cloudflare_error is not None and cloudflare_failure_callback:
                cloudflare_failure_callback(final_cloudflare_error)
            continue
        retry_setting = {"codex": "codex_retry_attempts",
                         "claude": "claude_retry_attempts",
                         "groq_primary": "groq_primary_retry_attempts",
                         "groq_fallback": "groq_fallback_retry_attempts",
                         "google": "google_retry_attempts"}.get(route)
        if retry_setting is not None:
            retry_limit = correction_settings[retry_setting]
            for retry_index in range(retry_limit + 1):
                if cancelled():
                    return cancelled_result(started)
                if backend_callback:
                    backend_callback(provider_name, model_name)
                attempt_started = time.monotonic()
                try:
                    if route == "codex":
                        candidate = codex_correct(text, model_name, cancel_callback=cancelled)
                    elif route == "claude":
                        candidate = claude_correct(text, model_name, cancel_callback=cancelled)
                    elif route in {"groq_primary", "groq_fallback"}:
                        candidate = groq_correct(text, model_name)
                    else:
                        candidate = google_correct(text, model_name)
                    if cancelled():
                        return cancelled_result(started)
                    rejection = correction_safety_reason(text, candidate, None)
                    if rejection:
                        raise ProviderError("codex", "unsafe_response_" + rejection)
                    record_attempt(attempt_log, "correction", provider_name, model_name,
                                   True, attempt_started, device="remote")
                    return finalize_text(candidate), None, round(time.monotonic() - started, 3), \
                        provider_name, None
                except Exception as error:
                    if cancelled():
                        return cancelled_result(started)
                    category = safe_provider_error_category(error)
                    failed_model = getattr(error, "model", None) or model_name
                    record_attempt(attempt_log, "correction", provider_name, failed_model,
                                   False, attempt_started, category, "remote")
                    if provider_failure_callback:
                        provider_failure_callback(provider_name, failed_model, category)
                    logger.warning("Correction provider failed provider=%s category=%s",
                                   provider_name, category)
                    retryable_categories = ({"cli_unavailable", "cli_timeout", "cli_output_invalid"}
                        if route in {"codex", "claude"} else
                        {"network_error", "timeout", "invalid_response",
                         "provider_unavailable"})
                    retryable = (category.startswith("unsafe_response_") or
                                 category in retryable_categories)
                    if not retryable or retry_index >= retry_limit:
                        break
            continue
        if backend_callback:
            backend_callback(provider_name, model_name)
        attempt_started = time.monotonic()
        selected_model = None
        try:
            if route == "codex":
                candidate = codex_correct(text, model_name, cancel_callback=cancelled)
            elif route == "claude":
                candidate = claude_correct(text, model_name, cancel_callback=cancelled)
            elif route == "groq_primary":
                candidate = groq_correct(text, model_name)
            elif route == "groq_fallback":
                candidate = groq_correct(text, model_name)
            elif route == "openrouter":
                candidate, selected_model = openrouter_correct(
                    text, model_name, include_model=True, cancel_callback=cancelled)
            elif route == "google":
                candidate = google_correct(text, model_name)
            else:
                estimated_neurons = estimate_cloudflare_correction_neurons(
                    text, cloudflare_profile["max_tokens"], model_name)
                if not reserve_cloudflare_daily_neurons(METRICS, estimated_neurons):
                    raise CloudflareError("Расчётный дневной лимит Cloudflare исчерпан",
                                          "daily_quota_exhausted")
                candidate, input_tokens, output_tokens = cloudflare_correct(
                    text, cloudflare_profile["model"], cloudflare_profile["temperature"],
                    cloudflare_profile["max_tokens"])
            if cancelled():
                return cancelled_result(started)
            rejection = correction_safety_reason(text, candidate, None)
            if rejection:
                raise ProviderError(provider_name, "unsafe_response_" + rejection)
            if route == "openrouter" and selected_model and backend_callback:
                backend_callback(provider_name, selected_model)
            record_attempt(attempt_log, "correction", provider_name,
                selected_model if route == "openrouter" and selected_model else model_name,
                True, attempt_started, device="remote")
            usage = None
            if route == "cloudflare":
                input_tokens, output_tokens, estimated = correction_token_counts(
                    text, candidate, input_tokens, output_tokens)
                usage = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                         "estimated": estimated,
                         "estimated_neurons": correction_neurons(input_tokens, output_tokens, model_name)}
            return finalize_text(candidate), None, round(time.monotonic() - started, 3), \
                provider_name, usage
        except Exception as error:
            if cancelled():
                return cancelled_result(started)
            category = safe_provider_error_category(error)
            failed_model = getattr(error, "model", None) or selected_model or model_name
            record_attempt(attempt_log, "correction", provider_name, failed_model,
                           False, attempt_started, category, "remote")
            if category == "daily_quota_exhausted":
                try:
                    mark_daily_limit(provider_name, category)
                except OSError:
                    logger.warning("Could not persist daily provider quota block")
            if provider_failure_callback:
                provider_failure_callback(provider_name, failed_model, category)
            logger.warning("Correction provider failed provider=%s category=%s", provider_name,
                           category)
            retryable_openrouter = (category in {"http_429", "http_error", "provider_unavailable",
                "timeout", "unsupported_model", "invalid_response", "network_error",
                "other_error"} or
                category.startswith(("invalid_response_", "unsafe_response_")))
            if route == "openrouter" and retryable_openrouter:
                attempted_models = {failed_model}
                retry_limit = safe_provider_settings(provider_settings)["correction"][
                    "openrouter_retry_attempts"]
                for _ in range(retry_limit):
                    if cancelled():
                        return cancelled_result(started)
                    fallback_model = choose_openrouter_fallback_model(METRICS, attempted_models)
                    if not fallback_model:
                        break
                    attempted_models.add(fallback_model)
                    fallback_started = time.monotonic()
                    fallback_selected = None
                    if backend_callback:
                        backend_callback("openrouter", fallback_model)
                    try:
                        fallback_candidate, fallback_selected = openrouter_correct(
                            text, fallback_model, include_model=True,
                            cancel_callback=cancelled)
                        if cancelled():
                            return cancelled_result(started)
                        fallback_rejection = correction_safety_reason(text, fallback_candidate, None)
                        if fallback_rejection:
                            raise ProviderError("openrouter", "unsafe_response_" + fallback_rejection,
                                                fallback_selected)
                        actual_model = fallback_selected or fallback_model
                        if backend_callback:
                            backend_callback("openrouter", actual_model)
                        record_attempt(attempt_log, "correction", "openrouter", actual_model,
                                       True, fallback_started, device="remote")
                        return finalize_text(fallback_candidate), None, \
                            round(time.monotonic() - started, 3), "openrouter", None
                    except Exception as fallback_error:
                        if cancelled():
                            return cancelled_result(started)
                        fallback_category = safe_provider_error_category(fallback_error)
                        fallback_failed_model = (getattr(fallback_error, "model", None) or
                                                 fallback_selected or fallback_model)
                        record_attempt(attempt_log, "correction", "openrouter",
                            fallback_failed_model, False, fallback_started,
                            fallback_category, "remote")
                        if provider_failure_callback:
                            provider_failure_callback("openrouter", fallback_failed_model,
                                                      fallback_category)
                        logger.warning("OpenRouter fallback model failed model=%s category=%s",
                                       fallback_failed_model, fallback_category)
                        if fallback_category == "daily_quota_exhausted":
                            try:
                                mark_daily_limit("openrouter", fallback_category)
                            except OSError:
                                logger.warning("Could not persist daily provider quota block")
                            break
                        if fallback_category in {"missing_credentials", "credentials_invalid",
                                "credentials_read_error", "invalid_key", "http_401_403",
                                "cancelled_by_user"}:
                            break
                        attempted_models.add(fallback_failed_model)
            if route == "cloudflare" and cloudflare_failure_callback:
                cloudflare_failure_callback(error)
            if cancelled():
                return cancelled_result(started)
    if cancelled():
        return cancelled_result(started)
    if local_attempt_result is not None:
        return local_attempt_result
    # Local correction is a scenario-specific route just like remote providers.
    # Do not silently invoke Qwen when the user disabled it for this source.
    if "local" in routes:
        return run_local_correction(fallback_stage=cloudflare_fallback)
    if backend_callback:
        backend_callback("local")
    return finalize_text(text), None, round(time.monotonic() - started, 3), \
        "local_disabled", None

def large_corrector_profile(value):
    """Keep Large ASR in VRAM; run its Qwen correction strictly on CPU.

    Large-v3 consumes most of the 6 GB GPU.  Keeping Qwen's usual partial
    GPU offload here can make Ollama queue or time out, silently returning
    uncorrected ASR text.  This override is deliberately per-request: the
    configured GPU profile still applies to Medium and Turbo.
    """
    corrector = dict(value["corrector"])
    if value["model"] != "large-v3":
        return corrector, False
    corrector["num_gpu"] = 0
    return corrector, True

def unload_corrector_model(model_name):
    """Free an already GPU-loaded Qwen before Large starts its CPU correction.

    Ollama keeps models resident for the configured keep_alive interval.  A
    preceding Turbo/Medium request may therefore leave Qwen in VRAM; merely
    passing num_gpu=0 would not guarantee that the next request is CPU-only.
    """
    try:
        payload = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
        request = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
        return True
    except (OSError, ValueError, TypeError, urllib.error.URLError, TimeoutError):
        logger.info("Local Qwen unload did not complete")
        return False

def local_corrector_model_names(value):
    """Return only configured local correction models owned by this app."""
    correction = safe_provider_settings(value.get("multi_provider"))["correction"]
    configured = correction.get("local_models", [])
    selected = value.get("corrector", {}).get("model")
    names = [name for name in configured if isinstance(name, str) and name]
    if isinstance(selected, str) and selected:
        names.append(selected)
    return list(dict.fromkeys(names))

def loaded_local_corrector_models(value):
    """Return loaded configured Ollama models, or None when status is unknown."""
    names = set(local_corrector_model_names(value))
    if not names:
        return []
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/ps", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=1.5) as response:
            payload = json.load(response)
        models = payload.get("models", []) if isinstance(payload, dict) else []
        return sorted({item.get("name") or item.get("model") for item in models
                       if isinstance(item, dict) and
                       (item.get("name") in names or item.get("model") in names)})
    except (OSError, ValueError, TypeError, urllib.error.URLError, TimeoutError):
        return None

def model_residency_status():
    """Expose model residency only; never return job or transcript content."""
    profile = config()
    asr_loaded = bool(model is not None or gigaam_worker is not None or
                      gigaam_onnx_worker is not None)
    asr_backend = active_asr_backend if asr_loaded else None
    return {"status": "ok", "asr_loaded": asr_loaded,
            "asr_backend": asr_backend,
            "asr_model": active_asr_model_name if asr_loaded else None,
            "corrector_models": loaded_local_corrector_models(profile)}

async def control_models(req):
    """Handle local model controls and explicit settings-default reset."""
    global model_lease, qwen_lease
    action = req.get("action")
    if action == "models_status":
        return model_residency_status()
    if action == "restore_settings_defaults":
        temporary_path = None
        try:
            CONFIG.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temporary_path = tempfile.mkstemp(
                dir=CONFIG.parent, prefix=".config-defaults-", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(clone_default(), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, CONFIG)
            return {"status": "ok"}
        except (OSError, TypeError, ValueError):
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
            logger.warning("Could not restore application settings defaults")
            return {"status": "error", "error_code": "settings_reset_failed"}
    if action not in {"unload_models", "prepare_selected_asr"}:
        return {"status": "error", "error_code": "invalid_control_action"}
    async with control_lock:
        await queue.join()
        await correction_idle.wait()
        if action == "prepare_selected_asr":
            profile = config()
            try:
                await prepare_local_asr(profile)
            except Exception:
                logger.exception("Selected local ASR could not be prepared")
                return {"status": "error", "error_code": "asr_load_failed"}
            return model_residency_status()

        scope = req.get("scope", "all")
        expected_backend = req.get("expected_backend")
        current_status = model_residency_status()
        if scope == "asr":
            backend_matches = (current_status["asr_backend"] == expected_backend
                               if expected_backend != "gigaam" else
                               str(current_status["asr_backend"] or "").startswith("gigaam-"))
            if expected_backend and not backend_matches:
                return {"status": "ok", "unchanged": True, **current_status}
            await release()
            model_lease += 1
        elif scope in {"corrector", "all"}:
            profile = config()
            loaded = current_status["corrector_models"]
            qwen_lease += 1
            targets = local_corrector_model_names(profile) if loaded is None else loaded
            corrector_unload_failed = False
            for name in targets:
                if not await asyncio.to_thread(unload_corrector_model, name):
                    corrector_unload_failed = True
            if scope == "corrector":
                status = model_residency_status()
                if corrector_unload_failed:
                    return {**status, "status": "error",
                            "error_code": "corrector_unload_failed"}
                return status
            await release()
            model_lease += 1
            status = model_residency_status()
            if corrector_unload_failed:
                return {**status, "status": "error", "asr_unloaded": True,
                        "error_code": "corrector_unload_failed"}
            return status
        else:
            return {"status": "error", "error_code": "invalid_unload_scope"}
        return model_residency_status()

def write_metric(entry):
    try:
        with METRICS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError: logger.warning("Could not write metrics")

def record_attempt(attempts, stage, provider, model_name, success, started,
                   error_category=None, device=None):
    """Append safe provider/model/timing metadata without job text or paths."""
    if isinstance(attempts, list):
        attempts.append({"stage": stage, "provider": provider, "model": model_name,
            "success": bool(success), "seconds": round(max(0.0, time.monotonic() - started), 3),
            "error_category": error_category, "device": device})

def local_asr_status_profile(value):
    """Describe the selected local backend accurately, even before model load."""
    if value.get("asr_backend") == "gigaam-v3-e2e-rnnt":
        return {"asr_backend": "gigaam-v3-e2e-rnnt", "model": GIGAAM_MODEL_NAME,
                "device": "cpu"}
    if value.get("asr_backend") in GIGAAM_ONNX_BACKENDS:
        backend = value["asr_backend"]
        spec = GIGAAM_ONNX_BACKENDS[backend]
        return {"asr_backend": backend, "model": spec["model"], "device": spec["device"]}
    return {"asr_backend": "faster-whisper", "model": value["model"],
            "device": value["device"]}


def update_job_backend(job_id, backend, profile, cloud_model=DEFAULT_CLOUDFLARE_CORRECTOR, stage="asr", device=None):
    """Update only a validated active job's backend; never write transcript data."""
    if not isinstance(job_id, str) or not job_id.isdigit() or len(job_id) > 32:
        return
    path = JOBS / f"{job_id}.json"
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entry, dict) or entry.get("state") != "processing":
            return
        entry["pipeline_stage"] = stage
        entry["backend"] = backend
        if stage in {"asr", "asr_waiting_gpu"}:
            selected = profile.get("asr_backend", "faster-whisper")
            entry["asr_backend_selected"] = (selected if selected in {
                "faster-whisper", *GIGAAM_BACKENDS} else "faster-whisper")
            if device in {"cpu", "cuda"}:
                entry["asr_backend_execution_device"] = device
            entry["asr_cpu_fallback_for_correction"] = bool(
                profile.get("asr_cpu_fallback_for_correction"))
            category = profile.get("asr_backend_error_category")
            if category in {"weights_missing_or_corrupt", "dependency_import_failure",
                    "cancelled", "audio_decode_timeout", "audio_decode_failure",
                    "invalid_chunk_settings", "inference_failure", "worker_unavailable",
                    "worker_timeout", "cuda_execution_unavailable", "invalid_model_id"}:
                entry["asr_backend_error_category"] = category
        if stage in {"correction_wait", "asr_waiting_gpu"}:
            entry.update({"device": device or entry.get("device", "unknown"),
                          "model": cloud_model or entry.get("model", "?")})
        elif backend == "cloudflare":
            entry.update({"device": "cloudflare", "model": cloud_model})
        elif backend in {"codex", "claude", "groq", "google", "openrouter"}:
            entry.update({"device": device or "remote", "model": cloud_model})
        elif stage == "correction":
            corrector = profile.get("corrector", {})
            entry.update({"device": device or ("cuda" if int(corrector.get("num_gpu", 0) or 0) > 0 else "cpu"),
                          "model": corrector.get("model", profile["model"])})
        else:
            entry.update({"device": profile["device"], "model": profile["model"]})
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    except (OSError, ValueError, json.JSONDecodeError):
        pass

SAFE_CLOUDFLARE_ERROR_CATEGORIES = {
    "missing_credentials", "credentials_read_error", "http_401_403", "http_429", "daily_quota_exhausted",
    "http_error", "network_error", "timeout", "invalid_response",
    "unsafe_response", "unsafe_response_length", "unsafe_response_word_coverage_order",
    "unsafe_response_tail", "unsafe_response_punctuation_loss", "unsafe_response_invalid",
    "local_audio_error", "usage_guard_unavailable", "other_error",
}
SAFE_PROVIDER_ERROR_CATEGORIES = SAFE_CLOUDFLARE_ERROR_CATEGORIES | {
    "http_413", "unsupported_model", "credentials_invalid", "invalid_key",
    "invalid_request", "unsafe_response", "unsafe_response_length",
    "unsafe_response_word_coverage_order", "unsafe_response_tail",
    "unsafe_response_punctuation_loss", "unsafe_response_invalid",
    "invalid_response_missing_choice", "invalid_response_missing_content",
    "invalid_response_non_string", "invalid_response_empty",
    "unavailable", "provider_unavailable", "audio_too_long", "attempt_counter_unavailable",
    "cli_unavailable", "cli_failed", "cli_timeout", "cli_output_invalid",
}

def safe_provider_error_category(error):
    category = getattr(error, "category", None)
    return category if category in SAFE_PROVIDER_ERROR_CATEGORIES else "other_error"


async def transcribe_local_candidate(job, value, planned_profile, cancel_marker):
    """Run the selected backend, using same-model CPU GigaAM during correction."""
    backend = value.get("asr_backend", "faster-whisper")
    use_cpu = (job.get("_asr_execution_device") == "cpu" and
               backend in GIGAAM_ONNX_BACKENDS)

    async def execute(device=None):
        cold = await prepare_local_asr(
            value, fallback_on_backend_error=False, execution_device=device)
        actual = dict(planned_profile)
        actual["model"] = active_asr_model_name or planned_profile["model"]
        actual["device"] = active_asr_device or planned_profile["device"]
        actual["asr_backend_error_category"] = active_asr_error_category
        actual["asr_cpu_fallback_for_correction"] = bool(
            use_cpu and actual["device"] == "cpu")
        update_job_backend(job.get("job_id"), "local", actual,
                           actual["model"], stage="asr", device=actual["device"])
        text = await asyncio.to_thread(transcribe_sync, job["wav"], value,
                                       cancel_marker)
        if not text or not text.strip():
            raise ProviderError("local", "invalid_response", actual["model"])
        return cold, actual, text

    if use_cpu:
        try:
            return await execute("cpu")
        except Exception as error:
            if (safe_provider_error_category(error) == "cancelled" or
                    correction_cancelled(job.get("job_id"))):
                raise
            waiting_profile = dict(planned_profile)
            waiting_profile["device"] = "cuda"
            waiting_profile["asr_cpu_fallback_for_correction"] = False
            waiting_profile["asr_backend_error_category"] = safe_provider_error_category(error)
            update_job_backend(job.get("job_id"), "local", waiting_profile,
                               waiting_profile["model"], stage="asr_waiting_gpu",
                               device="cuda")
            logger.warning("GigaAM CPU fallback failed category=%s; waiting for GPU slot",
                           safe_provider_error_category(error))
            await correction_idle.wait()
            job.pop("_asr_execution_device", None)
    return await execute()

def cloudflare_error_category(error):
    """Return only a small allowlist of non-sensitive provider error categories."""
    category = getattr(error, "category", None)
    return category if category in SAFE_CLOUDFLARE_ERROR_CATEGORIES else "other_error"

async def run(job, asr_only=True):
    value, wav = job["profile"], job["wav"]
    provider = job.get("provider", "local")
    asr_settings = safe_provider_settings(value.get("multi_provider"))["asr"]
    source = job.get("source")
    configured_local_order = (asr_settings["file_video_local_models"]
                              if source in POST_TRANSCRIPTION_SOURCES else
                              asr_settings["voice_local_models"])
    raw_multi_settings = value.get("multi_provider", {})
    raw_asr_settings = (raw_multi_settings.get("asr", {})
                        if isinstance(raw_multi_settings, dict) else {})
    if (source not in POST_TRANSCRIPTION_SOURCES and
            isinstance(raw_asr_settings, dict) and
            not any(isinstance(raw_asr_settings.get(key), list) and
                    raw_asr_settings.get(key) for key in
                    ("voice_local_models", "local_models"))):
        configured_local_order = [value.get("asr_backend", "faster-whisper")]
    if not configured_local_order:
        configured_local_order = ["faster-whisper"]
    selected_local_backend, selected_local_model = resolve_local_asr_model(
        configured_local_order[0], value.get("model", "large-v3"))
    selected_profile_value = dict(value)
    selected_profile_value["asr_backend"] = selected_local_backend
    selected_profile_value["model"] = selected_local_model
    cloudflare_failed = False
    cloudflare_event = {
        "cloudflare_attempted": False,
        "cloudflare_asr_error_category": None,
        "cloudflare_corrector_error_category": None,
        "provider_errors": [],
        "attempts": [],
    }
    quota_state = daily_limit_state()
    if quota_state:
        cloudflare_event.update({"daily_quota_provider": quota_state["provider"],
            "daily_quota_blocked_providers": quota_state["blocked_providers"],
            "daily_quota_reset_utc": quota_state["until_utc"]})
    _planned_asr_provider, planned_corrector = select_job_backends(job.get("source"), value)
    selected_local_profile = local_asr_status_profile(selected_profile_value)
    planned_asr_model = (value.get("cloudflare_asr", {}).get("model", value.get("cloudflare_asr_model", "@cf/openai/whisper-large-v3-turbo"))
                         if provider == "cloudflare" else selected_local_profile["model"])
    correction_settings = safe_provider_settings(value.get("multi_provider"))["correction"]
    correction_routes = (multi_correction_chain(
        job.get("source"), value.get("multi_provider"),
        planned_corrector.get("enabled")) if correction_settings["enabled"] else [])
    if not value.get("corrector", {}).get("enabled", False):
        correction_routes = [route for route in correction_routes if route != "local"]
    correction_routes = daily_only_routes(correction_routes)
    corrections_enabled = bool(correction_routes)
    if not correction_routes:
        planned_corrector_backend = "local_disabled"
    elif "cloudflare" in correction_routes and correction_routes[0] == "cloudflare":
        planned_corrector_backend = "cloudflare"
    elif any(route != "local" for route in correction_routes):
        planned_corrector_backend = "remote_chain"
    else:
        planned_corrector_backend = "local"
    planned_corrector_model = (
        planned_corrector.get("model") if planned_corrector_backend == "cloudflare" else
        value.get("corrector", {}).get("model") if "local" in correction_routes else None)
    initial_metadata = {"asr_backend": provider, "asr_model": planned_asr_model,
                        "asr_backend_selected": selected_local_backend,
                        "asr_device": "cloudflare" if provider == "cloudflare" else
                                      selected_local_profile["device"],
                        "corrector_backend": planned_corrector_backend,
                        "corrector_model": planned_corrector_model,
                        "corrector_device": "cloudflare" if planned_corrector_backend == "cloudflare" else
                                           ("cuda" if int(value.get("corrector", {}).get("num_gpu", 0) or 0) > 0 else "cpu")
                                           if planned_corrector_backend == "local" else None}
    if not os.path.isfile(wav):
        return {"status":"error","message":"Audio file not found","error_code":"audio_not_found", **initial_metadata, **cloudflare_event}
    selected_cf_asr = value.get("cloudflare_asr", {}).get("model", value.get("cloudflare_asr_model", "@cf/openai/whisper-large-v3-turbo"))
    update_job_backend(job.get("job_id"), "cloudflare" if provider == "cloudflare" else "local", value,
                       selected_cf_asr if provider == "cloudflare" else value["cloudflare_corrector"]["model"])
    cold = False; fallback = None
    try:
        t0 = time.monotonic()
        cloudflare_asr_enabled = provider == "cloudflare"
        asr_routes = daily_only_routes(
            multi_asr_chain(job.get("source"), value.get("multi_provider"), cloudflare_asr_enabled))
        remote_attempted = any(route != "local" for route in asr_routes)
        if remote_attempted:
            await prepare_remote_models(value["corrector"]["model"])
        text, used_asr_provider, used_asr_model = None, None, None
        blocked_this_job = set()
        for route in asr_routes:
            route_provider = ("cloudflare" if route == "cloudflare" else
                              "groq" if route.startswith("groq") else
                              "google" if route == "google" else None)
            if route_provider in blocked_this_job or (
                    route_provider and daily_limit_state(provider=route_provider) is not None):
                continue
            try:
                route_backend = ("groq" if route.startswith("groq") else
                                "cloudflare" if route == "cloudflare" else
                                "google" if route == "google" else "local")
                route_model = ("whisper-large-v3" if route == "groq_precise" else
                               "whisper-large-v3-turbo" if route == "groq_fast" else
                               selected_cf_asr if route == "cloudflare" else
                               "gemini-3.5-transcribe" if route == "google" else
                               selected_local_profile["model"])
                attempt_started = time.monotonic()
                update_job_backend(job.get("job_id"), route_backend, value, route_model,
                                   stage="asr", device="remote" if route_backend != "local" else value["device"])
                if route in {"groq_precise", "groq_fast"}:
                    model_id = "whisper-large-v3" if route == "groq_precise" else "whisper-large-v3-turbo"
                    text = await asyncio.to_thread(groq_transcribe, wav, value.get("language", "ru"), model_id)
                    used_asr_provider, used_asr_model = "groq", model_id
                elif route == "cloudflare":
                    cloudflare_event["cloudflare_attempted"] = True
                    audio_seconds = duration(wav)
                    if audio_seconds is None or audio_seconds <= 0:
                        raise CloudflareError("Не удалось оценить длительность аудио для дневного лимита",
                                              "usage_guard_unavailable")
                    estimated_neurons = (audio_seconds / 60.0) * NEURONS_PER_AUDIO_MINUTE
                    if not reserve_cloudflare_daily_neurons(METRICS, estimated_neurons):
                        raise CloudflareError("Расчётный дневной лимит Cloudflare исчерпан",
                                              "daily_quota_exhausted")
                    text = await asyncio.to_thread(cloudflare_transcribe, wav,
                        value["cloudflare_asr"].get("language", "ru"),
                        value["cloudflare_asr"].get("vad_filter", False),
                        value["cloudflare_asr"].get("initial_prompt", ""),
                        value["cloudflare_asr"].get("model", value.get("cloudflare_asr_model", "@cf/openai/whisper-large-v3-turbo")),
                        value["cloudflare_asr"])
                    used_asr_provider = "cloudflare"
                    used_asr_model = value["cloudflare_asr"].get("model", selected_cf_asr)
                elif route == "google":
                    text = await asyncio.to_thread(transcribe_gemini, wav, value.get("language", "ru"))
                    used_asr_provider, used_asr_model = "google", "gemini-3.5-transcribe"
                elif route == "local":
                    cancel_marker = (RECOVERY_CANCEL / f"{job['job_id']}.cancel"
                                     if str(job.get("job_id", "")).isdigit() else None)
                    local_model_order = configured_local_order
                    if local_model_order:
                        local_error = None
                        actual_profile = None
                        for candidate_id in local_model_order:
                            if correction_cancelled(job.get("job_id")):
                                raise ProviderError("local", "cancelled")
                            candidate_backend, candidate_model = resolve_local_asr_model(
                                candidate_id, value.get("model", "large-v3"))
                            candidate_value = dict(value)
                            candidate_value["asr_backend"] = candidate_backend
                            candidate_value["model"] = candidate_model
                            planned_profile = local_asr_status_profile(candidate_value)
                            cpu_gigaam_fallback = (
                                job.get("_asr_execution_device") == "cpu" and
                                candidate_backend in GIGAAM_ONNX_BACKENDS)
                            if cpu_gigaam_fallback:
                                planned_profile["device"] = "cpu"
                                planned_profile["asr_cpu_fallback_for_correction"] = True
                            candidate_overlap_safe = (
                                candidate_backend == "gigaam-v3-e2e-rnnt" and
                                active_asr_backend == "gigaam-v3-e2e-rnnt" and
                                gigaam_worker is not None)
                            if correction_pending and not cpu_gigaam_fallback and not candidate_overlap_safe:
                                await correction_idle.wait()
                            candidate_started = time.monotonic()
                            update_job_backend(job.get("job_id"), "local", candidate_value,
                                               planned_profile["model"], stage="asr",
                                               device=planned_profile["device"])
                            try:
                                candidate_cold, candidate_profile, candidate_text = \
                                    await transcribe_local_candidate(
                                        job, candidate_value, planned_profile, cancel_marker)
                                text = candidate_text
                                cold = cold or candidate_cold
                                actual_profile = candidate_profile
                                used_asr_provider, used_asr_model = "local", candidate_profile["model"]
                                if local_model_order.index(candidate_id) > 0:
                                    fallback = "local_model_fallback"
                                break
                            except Exception as candidate_error:
                                candidate_category = safe_provider_error_category(candidate_error)
                                if (candidate_backend == "gigaam-v3-e2e-rnnt" and
                                        active_asr_error_category):
                                    candidate_category = active_asr_error_category
                                elif (candidate_backend in GIGAAM_ONNX_BACKENDS and
                                      active_asr_error_category):
                                    candidate_category = active_asr_error_category
                                record_attempt(cloudflare_event["attempts"], "asr", "local",
                                               planned_profile["model"], False, candidate_started,
                                               candidate_category, planned_profile["device"])
                                if candidate_category == "cancelled" or correction_cancelled(job.get("job_id")):
                                    raise ProviderError("local", "cancelled")
                                local_error = candidate_error
                                logger.warning("Local ASR candidate failed backend=%s category=%s",
                                               candidate_backend, candidate_category)
                        if actual_profile is None:
                            raise local_error or ProviderError("local", "invalid_response")
                    else:
                        planned_profile = local_asr_status_profile(value)
                        cpu_gigaam_fallback = (
                            job.get("_asr_execution_device") == "cpu" and
                            value.get("asr_backend") in GIGAAM_ONNX_BACKENDS)
                        if cpu_gigaam_fallback:
                            planned_profile["device"] = "cpu"
                            planned_profile["asr_cpu_fallback_for_correction"] = True
                        update_job_backend(job.get("job_id"), "local", planned_profile,
                                           planned_profile["model"], stage="asr",
                                           device=planned_profile["device"])
                        cold, actual_profile, text = await transcribe_local_candidate(
                            job, value, planned_profile, cancel_marker)
                        used_asr_provider, used_asr_model = "local", actual_profile["model"]
                if text:
                    record_attempt(cloudflare_event["attempts"], "asr", route_backend,
                                   used_asr_model or route_model, True, attempt_started,
                                   device=(actual_profile["device"] if route_backend == "local" else "remote"))
                    break
                raise ProviderError(route, "invalid_response")
            except Exception as error:
                failed_backend = locals().get("route_backend", "local")
                failed_model = locals().get("route_model", value.get("model"))
                failed_category = (cloudflare_error_category(error) if route == "cloudflare"
                                   else safe_provider_error_category(error))
                if route == "local" and active_asr_backend in GIGAAM_BACKENDS:
                    failed_category = getattr(error, "category", failed_category)
                if failed_category == "cancelled":
                    return {"status": "error", "message": "Transcription cancelled",
                            "error_code": "cancelled", **initial_metadata,
                            "asr_backend_error_category": "cancelled"}
                record_attempt(cloudflare_event["attempts"], "asr", failed_backend,
                               failed_model, False, locals().get("attempt_started", time.monotonic()),
                               failed_category,
                               value.get("device") if failed_backend == "local" else "remote")
                if route == "cloudflare":
                    cloudflare_event["cloudflare_asr_error_category"] = cloudflare_error_category(error)
                    if cloudflare_event["cloudflare_asr_error_category"] == "daily_quota_exhausted":
                        blocked_this_job.add("cloudflare")
                        try:
                            mark_daily_limit("cloudflare", "daily_quota_exhausted")
                        except OSError:
                            logger.warning("Could not persist daily provider quota block")
                        quota_state = daily_limit_state(provider="cloudflare")
                        cloudflare_event.update({"daily_quota_provider": "cloudflare",
                            "daily_quota_blocked_providers": quota_state["blocked_providers"] if quota_state else [],
                            "daily_quota_reset_utc": quota_state.get("until_utc") if quota_state else None})
                if route in {"groq_precise", "groq_fast", "google"}:
                    route_model = ("whisper-large-v3" if route == "groq_precise" else
                                   "whisper-large-v3-turbo" if route == "groq_fast" else
                                   "gemini-3.5-transcribe")
                    cloudflare_event["provider_errors"].append({"provider": "groq" if route.startswith("groq") else "google",
                        "model": route_model, "stage": "asr",
                        "category": safe_provider_error_category(error)})
                    fallback = f"{route}_failed"
                    logger.warning("ASR provider failed provider=%s category=%s", route,
                                   safe_provider_error_category(error))
                continue
        if used_asr_provider != "local":
            asyncio.create_task(delayed_whisper_release())
        elapsed = time.monotonic() - t0
        used_cloudflare = used_asr_provider == "cloudflare"
        asr_model_name = used_asr_model or value["model"]
        asr_backend = used_asr_provider or "local"
        asr_device = used_asr_provider if used_asr_provider != "local" else (
            active_asr_device or value["device"])
        raw_asr_text = text
        text = remove_subtitle_credit_tail(text)
        if text != raw_asr_text:
            logger.info("ASR subtitle-credit tail removed")
        if not text:
            return {"status":"error","message":"Empty transcription","error_code":"empty_transcription",
                    **{k:value[k] for k in ("model","device","compute_type")},
                    "asr_backend":asr_backend,"asr_model":asr_model_name,"asr_device":asr_device,
                    "asr_backend_error_category":active_asr_error_category,
                    "corrector_backend":"not_run", **cloudflare_event}
        if asr_only:
            return {"status": "asr_ready", "text": text,
                    "live_correction_enabled": corrections_enabled,
                    "_raw_asr_text": raw_asr_text,
                    "_event": cloudflare_event,
                    "_used_asr_provider": used_asr_provider,
                    "_used_asr_model": used_asr_model,
                    "_asr_started": t0,
                    "cold_start": cold, "fallback": fallback,
                    "asr_seconds": round(elapsed, 3),
                    "model": asr_model_name,
                    "device": "cloudflare" if used_cloudflare else value["device"],
                    "compute_type": "remote" if used_cloudflare else value["compute_type"],
                    "asr_backend": asr_backend, "asr_model": asr_model_name,
                    "asr_backend_runtime": ("remote" if used_asr_provider != "local"
                                            else active_asr_backend),
                    "asr_device": asr_device,
                    "asr_backend_selected": selected_local_backend,
                    "asr_backend_error_category": active_asr_error_category}
        return {"status": "error", "message": "ASR stage did not produce a result",
                "error_code": "asr_failed", **initial_metadata, **cloudflare_event}
    except CloudflareError as error:
        return {"status":"error","message":str(error),"error_code":"cloudflare_asr_failed","error_category":getattr(error, "category", "invalid_response"),
                "model":planned_asr_model,"device":"cloudflare","compute_type":"remote", **initial_metadata,
                **cloudflare_event, "cloudflare_attempted":True,
                "cloudflare_asr_error_category":cloudflare_error_category(error)}
    except Exception as error:
        if not is_oom(error):
            logger.error("ASR failed model=%s device=%s", value["model"], value["device"])
            return {"status":"error","message":"ASR failed; see local daemon log","error_code":"asr_failed",
                    **{k:value[k] for k in ("model","device","compute_type")}, **initial_metadata, **cloudflare_event}
        if correction_pending:
            # The OOM fallback can load FasterWhisper on CUDA. Let any local
            # GPU correction finish before changing the ASR model.
            await correction_idle.wait()
        logger.warning("OOM model=%s device=%s; applying configured fallback once", value["model"], value["device"])
        selected_backend_error = active_asr_error_category
        await release(); fb = dict(value["fallback"])
        if fb["device"] == "cpu" and not fb["allow_cpu"]:
            return {"status":"error","message":"CUDA memory exhausted; CPU fallback is disabled",
                    "error_code":"oom_cpu_fallback_disabled",**{k:value[k] for k in ("model","device","compute_type")}, **initial_metadata, **cloudflare_event}
        retry = dict(value); retry.update(fb)
        if value.get("asr_backend") in GIGAAM_BACKENDS:
            retry["asr_backend"] = "faster-whisper"
        update_job_backend(job.get("job_id"), "local", retry, stage="asr")
        fallback = f'{fb["model"]}:{fb["device"]}:{fb["compute_type"]}'
        try:
            cold = await ensure(retry); t0 = time.monotonic()
            text = await asyncio.to_thread(transcribe_sync, wav, retry)
            elapsed = time.monotonic() - t0
            if text:
                return {"status":"ok","text":text,"cold_start":cold,"asr_seconds":round(elapsed,3),
                        "fallback":fallback,**{k:retry[k] for k in ("model","device","compute_type")},
                        "asr_backend":"local","asr_model":retry["model"],"asr_device":retry["device"],
                        "asr_backend_selected":selected_local_backend,
                        "asr_backend_error_category":selected_backend_error,
                        "corrector_backend":"not_run","corrector_model":None,"corrector_device":None, **cloudflare_event}
        except Exception: pass
        logger.error("Configured OOM fallback failed model=%s device=%s", retry["model"], retry["device"])
        return {"status":"error","message":"ASR fallback failed; see local daemon log",
                "error_code":"fallback_failed","fallback":fallback, **initial_metadata,
                **{k:retry[k] for k in ("model","device","compute_type")},
                "asr_backend":"local","asr_model":retry["model"],"asr_device":retry["device"],
                "asr_backend_selected":selected_local_backend,
                "asr_backend_error_category":selected_backend_error, **cloudflare_event}


async def correction_stage(job, asr_result):
    """Finish one job's correction after its ASR result is safely snapshotted."""
    global qwen_lease
    qwen_lease += 1
    value = job["profile"]
    text = asr_result["text"]
    raw_asr_text = asr_result["_raw_asr_text"]
    cloudflare_event = asr_result["_event"]
    used_asr_provider = asr_result.get("_used_asr_provider")
    used_asr_model = asr_result.get("_used_asr_model")
    elapsed = asr_result.get("asr_seconds", 0.0)
    cold = asr_result.get("cold_start", False)
    fallback = asr_result.get("fallback")
    asr_model_name = asr_result.get("asr_model", value["model"])
    asr_backend = asr_result.get("asr_backend", "local")
    asr_device = asr_result.get("asr_device", value["device"])
    used_cloudflare = used_asr_provider == "cloudflare"
    gigaam_cpu_runtime = (asr_result.get("asr_backend_runtime") == "gigaam-v3-e2e-rnnt" and
                          asr_device == "cpu")
    active_asr_error_category_for_job = asr_result.get("asr_backend_error_category")
    if job.get("_correction_queue_wait_started") is not None:
        cloudflare_event["correction_queue_wait_seconds"] = round(
            max(0.0, time.monotonic() - job["_correction_queue_wait_started"]), 3)
    if save_live_raw_asr(job.get("job_id"), job.get("source"), raw_asr_text):
        update_job_backend(job.get("job_id"), asr_backend, value, asr_model_name,
                           stage="correction", device=asr_device)
    corrector, large_cpu_corrector = large_corrector_profile(value)
    _correction_provider, cloud_corrector = select_job_backends(job.get("source"), value)
    correction_routes = daily_only_routes(
        multi_correction_chain(job.get("source"), value.get("multi_provider"),
                               cloud_corrector.get("enabled")))
    if any(route != "local" for route in correction_routes):
        # GigaAM is CPU-only and can stay resident while a remote correction
        # runs and the next ASR starts. Other ASR profiles keep the old
        # release behavior because they may share the GPU with local Qwen.
        local_is_first = correction_routes.index("local") == 0
        if not local_is_first and not gigaam_cpu_runtime:
            await prepare_remote_models(corrector["model"])
        elif not local_is_first:
            await asyncio.to_thread(unload_corrector_model, corrector["model"])
    elif corrector.get("enabled") and not large_cpu_corrector:
        # Local Qwen may need the GPU; after remote ASR success let the
        # lease-guarded 2-second release finish before loading it.
        if not gigaam_cpu_runtime:
            if used_asr_provider != "local":
                await asyncio.sleep(2.0)
            else:
                await release()
    if large_cpu_corrector:
        await asyncio.to_thread(unload_corrector_model, corrector["model"])
    correction_device = "cuda" if int(corrector.get("num_gpu", 0) or 0) > 0 else "cpu"
    correction_status = {"device": None}
    def report_cloudflare_correction_failure(error):
        cloudflare_event["cloudflare_attempted"] = True
        cloudflare_event["cloudflare_corrector_error_category"] = cloudflare_error_category(error)

    def report_provider_failure(provider_name, model_name, category):
        cloudflare_event["provider_errors"].append({"provider": provider_name,
            "model": model_name, "stage": "correction", "category": category})
        if category == "daily_quota_exhausted":
            quota_state = daily_limit_state(provider=provider_name)
            cloudflare_event.update({
                "daily_quota_provider": provider_name,
                "daily_quota_blocked_providers": quota_state["blocked_providers"] if quota_state else [],
                "daily_quota_reset_utc": quota_state["until_utc"] if quota_state else None})

    def report_correction_backend(backend, actual_device=None):
        if backend == "cloudflare":
            cloudflare_event["cloudflare_attempted"] = True
        correction_status["device"] = ("cloudflare" if backend == "cloudflare" else
                                        "remote" if backend in {"codex", "claude", "groq", "google", "openrouter"} else
                                        actual_device or correction_device)
        correction_status["model"] = (actual_device if backend == "cloudflare" and
                      actual_device in CLOUDFLARE_CORRECTOR_MODELS else
                      value["cloudflare_corrector"]["model"] if backend == "cloudflare" else
                      actual_device if backend in {"codex", "claude", "groq", "google", "openrouter"} else
                      {"groq": "qwen/qwen3.8-27b", "google": "gemini-3.5-flash-lite",
                       "openrouter": "openrouter/free"}.get(backend, corrector["model"]))
        update_job_backend(job.get("job_id"), backend, value, correction_status["model"],
                           stage="correction", device=correction_status["device"])
    text, correction_fallback, correction_seconds, correction_backend, correction_usage = await asyncio.to_thread(
        correct_text_sync, text, corrector, cloud_corrector,
        report_correction_backend, report_cloudflare_correction_failure,
        value.get("multi_provider"), job.get("source"), report_provider_failure,
        attempt_log=cloudflare_event["attempts"],
        cancel_callback=lambda: correction_cancelled(job.get("job_id")))
    quota_state = daily_limit_state()
    if quota_state:
        cloudflare_event.update({"daily_quota_provider": quota_state["provider"],
            "daily_quota_blocked_providers": quota_state["blocked_providers"],
            "daily_quota_reset_utc": quota_state["until_utc"]})
    if correction_backend in {"codex", "claude", "cloudflare", "groq", "google", "openrouter"}:
        if not gigaam_cpu_runtime:
            asyncio.create_task(delayed_whisper_release())
        asyncio.create_task(delayed_qwen_release(corrector["model"]))
    if correction_backend in {"local", "local_fallback"} and (
            corrector.get("enabled") or correction_backend == "local_fallback"):
        await asyncio.to_thread(release_qwen_under_memory_pressure, corrector)
    model_name = asr_model_name
    corrector_model_name = (None if correction_backend == "local_disabled" else
                            correction_status.get("model") or corrector["model"])
    corrector_device = (None if correction_backend == "local_disabled" else
                        correction_status["device"] or correction_device)
    correction_mode = "cpu_for_large" if large_cpu_corrector and correction_backend in {"local", "local_fallback"} else "configured"
    return {"status":"ok","text":text,"cold_start":cold,"asr_seconds":round(elapsed,3), "correction_fallback": correction_fallback, "correction_seconds": correction_seconds, "correction_mode": correction_mode,
            "fallback":fallback,"model":model_name,"device":"cloudflare" if used_cloudflare or correction_backend == "cloudflare" else value["device"],"compute_type":"remote" if used_asr_provider != "local" or correction_backend in {"codex", "claude", "cloudflare", "groq", "google", "openrouter"} else value["compute_type"],
            "asr_backend":asr_backend,"asr_backend_runtime":asr_result.get("asr_backend_runtime", "remote" if used_asr_provider == "cloudflare" else asr_result.get("asr_backend_selected")),"asr_model":asr_model_name,"asr_device":asr_device,
            "asr_backend_selected":asr_result.get("asr_backend_selected"),
            "asr_backend_error_category":active_asr_error_category_for_job,
            "corrector_backend":correction_backend,"corrector_model":corrector_model_name,
            "corrector_device":corrector_device,"corrector_usage":correction_usage, **cloudflare_event}

async def finish_job(job, result, started):
    if not job["future"].done():
        job["future"].set_result(result)
    write_metric({"timestamp":datetime.now(timezone.utc).isoformat(),
        "status":result.get("status", "error"),
        "source":job.get("source", "local_job"),
        "config_version":job["profile"]["version"],"model":result.get("model",job["profile"]["model"]),
        "device":result.get("device",job["profile"]["device"]),
        "compute_type":result.get("compute_type",job["profile"]["compute_type"]),
        "asr_model":result.get("asr_model", result.get("model", job["profile"]["model"])),
        "asr_device":result.get("asr_device", result.get("device", job["profile"]["device"])),
        "corrector_model":result.get("corrector_model"),
        "corrector_device":result.get("corrector_device"),
        "audio_duration_seconds":duration(job["wav"]),"cold_start":result.get("cold_start"),
        "asr_seconds":result.get("asr_seconds"),"correction_seconds":result.get("correction_seconds"),
        "correction_queue_wait_seconds":result.get("correction_queue_wait_seconds"),
        "correction_fallback":result.get("correction_fallback"), "correction_mode":result.get("correction_mode"),
        "total_seconds":round(time.monotonic()-started,3),
        "queue_wait_seconds":round(started-job["queued"],3),"fallback":result.get("fallback"),
        "asr_backend":result.get("asr_backend"),
        "asr_backend_runtime":result.get("asr_backend_runtime"),
        "asr_model":result.get("asr_model", result.get("model")),
        "asr_backend_selected":result.get("asr_backend_selected", job["profile"].get("asr_backend", "faster-whisper")),
        "asr_backend_error_category":result.get("asr_backend_error_category"),
        "corrector_backend":result.get("corrector_backend"),
        "cloudflare_attempted":result.get("cloudflare_attempted", False),
        "cloudflare_asr_error_category":result.get("cloudflare_asr_error_category"),
        "cloudflare_corrector_error_category":result.get("cloudflare_corrector_error_category"),
        "provider_errors":result.get("provider_errors", []),
        "attempts":result.get("attempts", []),
        "daily_quota_provider":result.get("daily_quota_provider"),
        "daily_quota_blocked_providers":result.get("daily_quota_blocked_providers", []),
        "daily_quota_reset_utc":result.get("daily_quota_reset_utc"),
        "corrector_usage":result.get("corrector_usage"),
        "error_code":result.get("error_code"),"error_category":result.get("error_category")})

async def worker():
    """FIFO ASR worker; completed recognition is handed to FIFO correction."""
    global correction_pending, latest_live_dictation_asr_at
    while True:
        job = await queue.get()
        selected_backend = job["profile"].get("asr_backend")
        overlap_safe = (job["profile"].get("asr_backend") == "gigaam-v3-e2e-rnnt" and
                        job.get("provider", "local") == "local" and
                        active_asr_backend == "gigaam-v3-e2e-rnnt" and
                        gigaam_worker is not None)
        cpu_gigaam_fallback = (correction_pending > 0 and
                               selected_backend in GIGAAM_ONNX_BACKENDS)
        if cpu_gigaam_fallback:
            job["_asr_execution_device"] = "cpu"
        elif not overlap_safe:
            await correction_idle.wait()
        started = time.monotonic()
        job["_started"] = started
        try:
            if isinstance(job.get("job_id"), str) and job["job_id"].isdigit():
                try:
                    (RECOVERY_CANCEL / f"{job['job_id']}.cancel").unlink(missing_ok=True)
                except OSError:
                    pass
            result = await run(job, asr_only=True)
            if result.get("status") != "asr_ready":
                await finish_job(job, result, started)
                continue
            live_correction_disabled = (
                job.get("source") == "live_dictation" and
                result.get("live_correction_enabled") is False)
            raw_gigaam_for_manual = (
                job.get("source") == "live_dictation" and
                job.get("_asr_execution_device") == "cpu" and
                result.get("asr_backend_selected") in GIGAAM_ONNX_BACKENDS and
                result.get("asr_backend_runtime") in GIGAAM_ONNX_BACKENDS and
                result.get("asr_device") == "cpu" and
                bool(manual_correction_cancel_events))
            if live_correction_disabled or raw_gigaam_for_manual:
                if manual_correction_cancel_events:
                    latest_live_dictation_asr_at = time.time()
                    save_latest_live_asr_at(latest_live_dictation_asr_at)
                event = result.get("_event", {})
                immediate_result = {
                    "status": "ok", "text": result.get("text", ""),
                    "model": result.get("asr_model"),
                    "device": result.get("asr_device"),
                    "compute_type": result.get("compute_type"),
                    "asr_model": result.get("asr_model"),
                    "asr_device": result.get("asr_device"),
                    "asr_backend": result.get("asr_backend"),
                    "asr_backend_runtime": result.get("asr_backend_runtime"),
                    "asr_backend_selected": result.get("asr_backend_selected"),
                    "asr_seconds": result.get("asr_seconds"),
                    "cold_start": result.get("cold_start"),
                    "fallback": result.get("fallback"),
                    "correction_mode": ("live_correction_disabled"
                                         if live_correction_disabled else
                                         "raw_asr_manual_busy"),
                    "correction_seconds": 0.0,
                    "correction_queue_wait_seconds": 0.0,
                    "corrector_backend": ("local_disabled" if live_correction_disabled
                                          else "skipped_manual_busy"),
                    "corrector_model": None, "corrector_device": None,
                    "attempts": event.get("attempts", []),
                }
                await finish_job(job, immediate_result, started)
                continue
            job["_asr_result"] = result
            job["_correction_queue_wait_started"] = time.monotonic()
            update_job_backend(
                job.get("job_id"), result.get("asr_backend", "local"),
                job["profile"], result.get("asr_model", job["profile"].get("model")),
                stage="correction_wait", device=result.get("asr_device"))
            correction_pending += 1
            correction_idle.clear()
            await correction_queue.put(job)
        except Exception:
            logger.exception("Unexpected worker failure")
            await finish_job(job, {"status":"error","message":"Internal daemon error","error_code":"internal_error"}, started)
        finally: queue.task_done()

async def correction_worker():
    """One correction worker preserves provider FIFO and quota accounting."""
    global correction_pending
    while True:
        job = await correction_queue.get()
        try:
            async with correction_execution_lock:
                result = await correction_stage(job, job["_asr_result"])
            await finish_job(job, result, job["_started"])
        except Exception:
            logger.exception("Unexpected correction worker failure")
            asr_result = job.get("_asr_result", {})
            await finish_job(job, {"status":"error",
                "message":"Correction failed; raw ASR text is included",
                "error_code":"correction_failed", "text":asr_result.get("text", ""),
                "asr_model":asr_result.get("asr_model"),
                "asr_device":asr_result.get("asr_device"),
                "asr_backend":asr_result.get("asr_backend"),
                "asr_backend_selected":asr_result.get("asr_backend_selected"),
                "asr_seconds":asr_result.get("asr_seconds"),
                "attempts":asr_result.get("_event", {}).get("attempts", [])},
                job["_started"])
        finally:
            correction_pending = max(0, correction_pending - 1)
            if correction_pending == 0:
                correction_idle.set()
            correction_queue.task_done()

async def respond(writer, value):
    try:
        writer.write(json.dumps(value, ensure_ascii=False).encode()+b"\n"); await writer.drain()
    except (BrokenPipeError, ConnectionResetError): logger.debug("Client disconnected before response")

async def client(reader, writer):
    try:
        raw = await asyncio.wait_for(reader.readline(), timeout=5)
        if not raw: return
        if len(raw) > 65536: return await respond(writer, {"status":"error","message":"Request too large","error_code":"request_too_large"})
        try: req = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError): return await respond(writer, {"status":"error","message":"Invalid JSON","error_code":"invalid_json"})
        if isinstance(req, dict) and isinstance(req.get("action"), str):
            if req["action"] == "manual_correct_text":
                return await respond(writer, await manual_correct_text(req))
            if req["action"] == "cancel_manual_correction":
                return await respond(writer, await cancel_manual_correction())
            return await respond(writer, await control_models(req))
        if not isinstance(req, dict) or not isinstance(req.get("wav"), str):
            return await respond(writer, {"status":"error","message":"Missing audio file","error_code":"invalid_request"})
        base_profile = config()
        try: profile = request_profile(req, base_profile)
        except ValueError: return await respond(writer, {"status":"error","message":"Invalid recognition profile","error_code":"invalid_profile"})
        requested_source = req.get("source")
        source = requested_source if isinstance(requested_source, str) and requested_source in {"live_dictation", "file", "batch", "youtube", "rutube", "vk"} else "unknown"
        # Cloudflare is permitted only for an explicitly tagged live dictation.
        provider, _cloud_corrector = select_job_backends(source, base_profile)
        cloudflare_fallback_local = req.get("cloudflare_fallback_local") is True
        future = asyncio.get_running_loop().create_future()
        job_id = req.get("job_id", "")
        if not isinstance(job_id, str) or (job_id and (not job_id.isdigit() or len(job_id) > 32)):
            job_id = ""
        async with control_lock:
            await queue.put({"wav":req["wav"],"profile":profile,"provider":provider,"source":source,"job_id":job_id,
                             "cloudflare_fallback_local":cloudflare_fallback_local,
                             "future":future,"queued":time.monotonic()})
        await respond(writer, await future)
    except (BrokenPipeError, ConnectionResetError, asyncio.IncompleteReadError): logger.debug("Client disconnected")
    except asyncio.TimeoutError: await respond(writer, {"status":"error","message":"Timeout reading request","error_code":"read_timeout"})
    except Exception:
        logger.exception("Client handler failed"); await respond(writer, {"status":"error","message":"Internal daemon error","error_code":"internal_error"})
    finally:
        writer.close()
        try: await writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError): pass

async def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    # A previous crash can leave a progress marker behind. A fresh daemon has
    # no active manual request, so remove only this status-only file on startup.
    try:
        MANUAL_CORRECTION_STATUS.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not clear stale manual correction status")
    if SOCKET.exists(): SOCKET.unlink()
    task = asyncio.create_task(worker())
    correction_task = asyncio.create_task(correction_worker())
    logger.info("Whisper daemon started")
    try:
        server = await asyncio.start_unix_server(client, str(SOCKET))
        async with server: await server.serve_forever()
    finally:
        # The process is exiting, so the OS will release model memory. Do not
        # drop a model object while a cancelled to_thread inference may still
        # be finishing; daemon-owned model transitions happen between jobs.
        task.cancel()
        correction_task.cancel()
        if SOCKET.exists(): SOCKET.unlink()
        logger.info("Whisper daemon stopped")
if __name__ == "__main__": asyncio.run(main())
