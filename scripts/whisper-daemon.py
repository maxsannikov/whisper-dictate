#!/usr/bin/env python3
"""Local FasterWhisper daemon: one worker, validated local configuration and private metrics."""
import asyncio
import difflib
import gc
import json
import logging
import os
import re
import sys
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
from cloudflare_asr import CloudflareError, cloudflare_first, transcribe as cloudflare_transcribe
from cloudflare_corrector import (DEFAULT_MODEL as DEFAULT_CLOUDFLARE_CORRECTOR,
    correct as cloudflare_correct, estimate_neurons as correction_neurons,
    token_counts as correction_token_counts, MODEL_NEURON_RATES)

HOME = Path.home()
CACHE = HOME / ".cache/whisper-dictate"
CONFIG = HOME / ".config/whisper-dictate/config.json"
REPLACEMENTS = CONFIG.parent / "autoreplacements.txt"
SOCKET = CACHE / "daemon.sock"
LOG = CACHE / "daemon.log"
METRICS = CACHE / "metrics.jsonl"
JOBS = CACHE / "jobs"
CACHE.mkdir(parents=True, exist_ok=True)
DEFAULT = {"version": 1, "model": "large-v3", "language": "ru", "device": "cuda",
 "compute_type": "int8_float16", "beam_size": 1, "best_of": 1, "temperature": 0.0,
 "vad": False, "initial_prompt": "", "glossary": [], "result_mode": "clipboard",
 "fallback": {"model": "medium", "device": "cuda", "compute_type": "int8_float16",
              "allow_cpu": False},
 "cloudflare_asr_enabled": False, "cloudflare_asr_model": "@cf/openai/whisper-large-v3-turbo",
 "cloudflare_corrector": {"enabled": False, "model": DEFAULT_CLOUDFLARE_CORRECTOR,
                          "temperature": 0.0, "max_tokens": 2048},
 "corrector": {"enabled": False, "provider": "ollama", "model": "qwen3.5:4b-q4_K_M", "prompt": "",
               "timeout_seconds": 300, "max_length_ratio": 1.5, "num_gpu": 8,
               "keep_alive": "5m", "temperature": 0, "num_ctx": 8192, "num_thread": 4}}
MODELS = {"large-v3", "large-v3-turbo", "medium", "small"}
COMPUTES = {"int8_float16", "float16", "int8", "float32"}
logging.basicConfig(filename=LOG, level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
model = None
model_profile = None
model_lease = 0
qwen_lease = 0
queue = asyncio.Queue()

def clone_default():
    return json.loads(json.dumps(DEFAULT))

def config():
    value = clone_default()
    if CONFIG.exists():
        try:
            raw = json.loads(CONFIG.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("root")
            for key in value:
                if key in raw: value[key] = raw[key]
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Invalid configuration; safe defaults used")
            return clone_default()
    valid = (value["model"] in MODELS and value["device"] in {"cuda", "cpu"} and
             value["compute_type"] in COMPUTES and isinstance(value["language"], str) and
             2 <= len(value["language"]) <= 12 and isinstance(value["beam_size"], int) and
             1 <= value["beam_size"] <= 10 and isinstance(value["best_of"], int) and
             1 <= value["best_of"] <= 10 and isinstance(value["temperature"], (int, float)) and
             0 <= value["temperature"] <= 1 and isinstance(value["vad"], bool) and
             isinstance(value["initial_prompt"], str) and len(value["initial_prompt"]) <= 800 and
             isinstance(value["glossary"], list) and value["result_mode"] in {"clipboard", "autopaste"})
    fallback = value["fallback"]
    valid = valid and isinstance(fallback, dict) and fallback.get("model") in MODELS and fallback.get("device") in {"cuda", "cpu"} and fallback.get("compute_type") in COMPUTES and isinstance(fallback.get("allow_cpu"), bool) and (fallback.get("model"), fallback.get("device"), fallback.get("compute_type")) != (value["model"], value["device"], value["compute_type"])
    corrector = value["corrector"]
    valid = valid and isinstance(corrector, dict) and isinstance(corrector.get("enabled"), bool) and corrector.get("provider") == "ollama" and isinstance(corrector.get("model"), str) and 1 <= len(corrector["model"]) <= 160 and isinstance(corrector.get("prompt"), str) and len(corrector["prompt"]) <= 1500 and type(corrector.get("timeout_seconds")) is int and corrector["timeout_seconds"] in {5, 10, 15, 30, 60, 120, 300} and isinstance(corrector.get("max_length_ratio"), (int, float)) and 1 <= corrector["max_length_ratio"] <= 3 and type(corrector.get("num_gpu")) is int and corrector["num_gpu"] in {0, 4, 8, 12} and corrector.get("keep_alive") in {0, "30s", "5m"} and type(corrector.get("temperature")) in {int, float} and corrector["temperature"] in {0, 0.1, 0.2} and type(corrector.get("num_ctx")) is int and corrector["num_ctx"] in {1024, 2048, 4096, 8192} and type(corrector.get("num_thread")) is int and corrector["num_thread"] in {2, 4, 8}
    if not valid:
        logger.warning("Configuration failed validation; safe defaults used")
        return clone_default()
    cloud_corrector = value["cloudflare_corrector"]
    if not isinstance(value["cloudflare_asr_enabled"], bool):
        logger.warning("Cloudflare ASR toggle invalid; ASR remains local")
        value["cloudflare_asr_enabled"] = False
    valid_cloud = (isinstance(cloud_corrector, dict) and isinstance(cloud_corrector.get("enabled"), bool) and
        cloud_corrector.get("model") in MODEL_NEURON_RATES and
        type(cloud_corrector.get("temperature")) in (int, float) and 0 <= cloud_corrector["temperature"] <= 5 and
        type(cloud_corrector.get("max_tokens")) is int and cloud_corrector["max_tokens"] in {256, 512, 1024, 2048, 4096})
    if not valid_cloud:
        logger.warning("Cloudflare configuration failed validation; cloud providers disabled")
        value["cloudflare_asr_enabled"] = False
        value["cloudflare_corrector"] = clone_default()["cloudflare_corrector"]
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
    global model, model_profile
    model = None; model_profile = None; gc.collect()

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

async def prepare_local_asr(value):
    """Make the selected Whisper profile the only daemon-owned GPU model."""
    await asyncio.to_thread(unload_corrector_model, value["corrector"]["model"])
    return await ensure(value)

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

async def ensure(value):
    global model, model_profile, model_lease
    model_lease += 1
    profile = {k: value[k] for k in ("model", "device", "compute_type")}
    if model is not None and model_profile == profile: return False
    await release()
    logger.info("Loading local model=%s device=%s compute_type=%s", profile["model"], profile["device"], profile["compute_type"])
    model = await asyncio.to_thread(WhisperModel, local_model_source(profile["model"]), device=profile["device"],
                                    compute_type=profile["compute_type"], local_files_only=True)
    model_profile = profile
    logger.info("Model loaded model=%s device=%s compute_type=%s", profile["model"], profile["device"], profile["compute_type"])
    return True

def transcribe_sync(wav, value):
    segments, _ = model.transcribe(wav, language=value["language"], vad_filter=value["vad"],
        beam_size=value["beam_size"], best_of=value["best_of"], temperature=value["temperature"],
        initial_prompt=prompt(value))
    return "".join(item.text for item in segments).strip()

CORRECTOR_INSTRUCTION = """Верни только готовый текст без комментариев. Сохрани все слова, факты, порядок, стиль и повторы; не сокращай и не дополняй текст. Удали только явные служебные хвосты ASR в конце: «продолжение следует», титры/субтитры и многократно повторённое одно слово. Обязательно расставь естественные знаки препинания и заверши текст точкой, вопросительным или восклицательным знаком. В коротком простом предложении не добавляй запятую без перечисления, обращения или вводных слов. Исправляй только однозначные ошибки ASR и орфографии. Не перефразируй текст и не меняй предлоги, падежи, формы слов или порядок слов. Критично: все исходные слова должны остаться в точно том же порядке и форме; разрешаются только знаки препинания и смена регистра букв. Не добавляй факты, объяснения, новые предложения или обращения. Сохраняй имена, команды, пути, URL, код, английские слова, числа и даты. Не добавляй кавычки, «ёлочки», обрамление или цитирование фраз, если их нет в исходном тексте; существующие кавычки сохраняй. При сомнении оставь исходный фрагмент без изменений. Обработай текст целиком до последнего слова."""
RETRY_INSTRUCTION = """Предыдущий ответ мог оборваться. Верни заново весь текст целиком, от первого до последнего слова, без сокращений. Сохрани все слова и порядок; исправь только пунктуацию, заглавные буквы и однозначные ошибки ASR."""
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

def correction_is_safe(source, corrected, done_reason):
    """Reject shortened or cut-off replies while allowing punctuation corrections.

    Qwen normally preserves the source words, merely changing their casing and
    punctuation.  The former 80-percent lower bound could still accept a reply
    missing a sentence in a long dictation, so compare normalized word streams
    as well as their lengths and endings.
    """
    if not isinstance(corrected, str):
        return False
    corrected = corrected.strip()
    source_words = re.findall(r"\w+", source.lower(), flags=re.UNICODE)
    corrected_words = re.findall(r"\w+", corrected.lower(), flags=re.UNICODE)
    if (not corrected or done_reason == "length" or chr(96) * 3 in corrected or
            any(ord(char) < 32 and char not in "\n\t\r" for char in corrected) or
            len(corrected) > len(source) * 1.5 or len(corrected) < len(source) * 0.88 or
            len(corrected_words) < len(source_words) * 0.9):
        return False
    # For longer text, require at least 90% of source words to remain in order.
    # This tolerates a few genuine ASR spelling fixes but catches omissions from
    # the middle that a last-word-only check cannot see.
    if len(source_words) >= 20:
        kept = sum(size for _, _, size in difflib.SequenceMatcher(
            a=source_words, b=corrected_words, autojunk=False).get_matching_blocks())
        if kept < len(source_words) * 0.9:
            return False
    tail = source_words[-min(3, len(source_words)):]
    corrected_tail = corrected_words[-16:]
    # A legitimate correction can inflect a final Russian word ("хостинг" →
    # "хостингу").  Accept that narrow case while still rejecting a missing
    # tail or a semantically unrelated replacement.
    def tail_word_present(word):
        return any(word == candidate or (len(word) >= 5 and
                   (candidate.startswith(word) or word.startswith(candidate)))
                   for candidate in corrected_tail)
    return all(tail_word_present(word) for word in tail)

def correct_text_sync(text, corrector, cloudflare_profile=None, backend_callback=None):
    """Correct safely in bounded chunks; retry once, then retain raw ASR chunk."""
    text = normalize_asr_text(text)
    cloudflare_enabled = bool(cloudflare_profile and cloudflare_profile.get("enabled"))
    if not corrector.get("enabled", False) and not cloudflare_enabled:
        if backend_callback:
            backend_callback("local")
        return finalize_text(text), None, 0.0, "local_disabled", None
    started = time.monotonic()
    if cloudflare_enabled:
        if backend_callback:
            backend_callback("cloudflare")
        try:
            cloud_text, input_tokens, output_tokens = cloudflare_correct(
                text, cloudflare_profile["model"], cloudflare_profile["temperature"],
                cloudflare_profile["max_tokens"])
            if not correction_is_safe(text, cloud_text, None):
                raise CloudflareError("Cloudflare-корректор вернул небезопасный ответ")
            input_tokens, output_tokens, estimated = correction_token_counts(
                text, cloud_text, input_tokens, output_tokens)
            return finalize_text(cloud_text), None, round(time.monotonic() - started, 3), \
                "cloudflare", {"input_tokens": input_tokens, "output_tokens": output_tokens,
                               "estimated": estimated,
                               "estimated_neurons": correction_neurons(
                                   input_tokens, output_tokens, cloudflare_profile["model"])}
        except Exception as error:
            logger.warning("Cloudflare corrector failed (%s); local correction used", type(error).__name__)
            if backend_callback:
                backend_callback("local")
            # The local corrector is the mandatory fallback even if its separate
            # toggle is off; all its model and performance settings are retained.
            corrector = dict(corrector)
            corrector["enabled"] = True
            cloudflare_fallback = True
    else:
        cloudflare_fallback = False
        if backend_callback:
            backend_callback("local")
    try:
        if corrector.get("provider") != "ollama":
            raise ValueError("unsupported corrector profile")
        custom_prompt = corrector.get("prompt", "").strip()
        instruction = CORRECTOR_INSTRUCTION + ("\n\nДополнительные указания пользователя:\n" + custom_prompt if custom_prompt else "")
        corrected_chunks, fallback_chunks = [], []
        chunks = correction_chunks(text)
        for index, chunk in enumerate(chunks, start=1):
            corrected, done_reason = correction_response(chunk, corrector, instruction)
            if not correction_is_safe(chunk, corrected, done_reason):
                logger.warning("Qwen reply rejected; retrying chunk=%d/%d done_reason=%s",
                               index, len(chunks), done_reason)
                corrected, done_reason = correction_response(chunk, corrector, instruction + "\n\n" + RETRY_INSTRUCTION)
            if correction_is_safe(chunk, corrected, done_reason):
                corrected_chunks.append(corrected.strip())
            else:
                logger.warning("Qwen retry rejected; raw ASR retained chunk=%d/%d done_reason=%s",
                               index, len(chunks), done_reason)
                corrected_chunks.append(chunk)
                fallback_chunks.append(index)
        fallback = None if not fallback_chunks else "raw_asr_after_retry:" + ",".join(map(str, fallback_chunks)) + f"/{len(chunks)}"
        backend = "local_fallback" if cloudflare_fallback else "local"
        return finalize_text(" ".join(corrected_chunks)), fallback, round(time.monotonic() - started, 3), backend, None
    except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeDecodeError, json.JSONDecodeError, urllib.error.URLError, TimeoutError):
        backend = "local_fallback" if cloudflare_fallback else "local"
        return text, "corrector_unavailable_or_timeout", round(time.monotonic() - started, 3), backend, None

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

def write_metric(entry):
    try:
        with METRICS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError: logger.warning("Could not write metrics")

def update_job_backend(job_id, backend, profile, cloud_model=DEFAULT_CLOUDFLARE_CORRECTOR, stage="asr"):
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
        if backend == "cloudflare":
            entry.update({"device": "cloudflare", "model": cloud_model})
        else:
            entry.update({"device": profile["device"], "model": profile["model"]})
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    except (OSError, ValueError, json.JSONDecodeError):
        pass

async def run(job):
    global qwen_lease
    qwen_lease += 1
    value, wav = job["profile"], job["wav"]
    provider = job.get("provider", "local")
    cloudflare_failed = False
    if not os.path.isfile(wav):
        return {"status":"error","message":"Audio file not found","error_code":"audio_not_found"}
    update_job_backend(job.get("job_id"), "cloudflare" if provider == "cloudflare" else "local", value,
                       "@cf/openai/whisper-large-v3-turbo" if provider == "cloudflare"
                       else value["cloudflare_corrector"]["model"])
    cold = False; fallback = None
    try:
        t0 = time.monotonic()
        if provider == "cloudflare":
            # The daemon worker serializes ASR and correction jobs, so its
            # cached Whisper/Qwen instances are idle and safe to release here.
            async def local_transcribe():
                nonlocal cold
                update_job_backend(job.get("job_id"), "local", value)
                cold = await prepare_local_asr(value)
                return await asyncio.to_thread(transcribe_sync, wav, value)

            text, cloudflare_failed, cloudflare_error = await cloudflare_first(
                lambda: asyncio.to_thread(cloudflare_transcribe, wav, "ru", value["vad"], prompt(value) or "", value.get("cloudflare_asr_model", "@cf/openai/whisper-large-v3-turbo")),
                local_transcribe, True)
            if not cloudflare_failed:
                asyncio.create_task(delayed_whisper_release())
            if cloudflare_failed:
                fallback = "cloudflare_to_local"
                logger.warning("Cloudflare ASR failed (%s); local ASR used", cloudflare_error)
        else:
            cold = await prepare_local_asr(value)
            text = await asyncio.to_thread(transcribe_sync, wav, value)
        elapsed = time.monotonic() - t0
        if not text:
            return {"status":"error","message":"Empty transcription","error_code":"empty_transcription",
                    **{k:value[k] for k in ("model","device","compute_type")}}
        corrector, large_cpu_corrector = large_corrector_profile(value)
        if provider == "cloudflare" and value["cloudflare_corrector"].get("enabled"):
            # ASR is finished; neither the local transcript model nor a
            # previously cached Ollama model is needed by Cloudflare.
            update_job_backend(job.get("job_id"), "cloudflare", value,
                               value["cloudflare_corrector"]["model"], stage="correction")
        elif corrector.get("enabled") and not large_cpu_corrector:
            # Local Qwen may need the GPU; after remote ASR success let the
            # lease-guarded 2-second release finish before loading it.
            if provider == "cloudflare" and not cloudflare_failed:
                await asyncio.sleep(2.0)
            else:
                await release()
        if large_cpu_corrector:
            await asyncio.to_thread(unload_corrector_model, corrector["model"])
        text, correction_fallback, correction_seconds, correction_backend, correction_usage = await asyncio.to_thread(
            correct_text_sync, text, corrector, value["cloudflare_corrector"],
            lambda backend: update_job_backend(job.get("job_id"), backend, value,
                                               value["cloudflare_corrector"]["model"], stage="correction"))
        if correction_backend == "cloudflare":
            asyncio.create_task(delayed_whisper_release())
            asyncio.create_task(delayed_qwen_release(corrector["model"]))
        if correction_backend in {"local", "local_fallback"} and (
                corrector.get("enabled") or correction_backend == "local_fallback"):
            await asyncio.to_thread(release_qwen_under_memory_pressure, corrector)
        used_cloudflare = provider == "cloudflare" and not cloudflare_failed
        model_name = value.get("cloudflare_asr_model", "@cf/openai/whisper-large-v3-turbo") if used_cloudflare else value["model"]
        correction_mode = "cpu_for_large" if large_cpu_corrector and correction_backend in {"local", "local_fallback"} else "configured"
        return {"status":"ok","text":text,"cold_start":cold,"asr_seconds":round(elapsed,3), "correction_fallback": correction_fallback, "correction_seconds": correction_seconds, "correction_mode": correction_mode,
                "fallback":fallback,"model":model_name,"device":"cloudflare" if used_cloudflare or correction_backend == "cloudflare" else value["device"],"compute_type":"remote" if used_cloudflare or correction_backend == "cloudflare" else value["compute_type"],
                "asr_backend":"cloudflare" if used_cloudflare else "local",
                "corrector_backend":correction_backend,"corrector_usage":correction_usage}
    except CloudflareError as error:
        return {"status":"error","message":str(error),"error_code":"cloudflare_asr_failed","error_category":getattr(error, "category", "invalid_response"),
                "model":value.get("cloudflare_asr_model", "@cf/openai/whisper-large-v3-turbo"),"device":"cloudflare","compute_type":"remote"}
    except Exception as error:
        if not is_oom(error):
            logger.error("ASR failed model=%s device=%s", value["model"], value["device"])
            return {"status":"error","message":"ASR failed; see local daemon log","error_code":"asr_failed",
                    **{k:value[k] for k in ("model","device","compute_type")}}
        logger.warning("OOM model=%s device=%s; applying configured fallback once", value["model"], value["device"])
        await release(); fb = dict(value["fallback"])
        if fb["device"] == "cpu" and not fb["allow_cpu"]:
            return {"status":"error","message":"CUDA memory exhausted; CPU fallback is disabled",
                    "error_code":"oom_cpu_fallback_disabled",**{k:value[k] for k in ("model","device","compute_type")}}
        retry = dict(value); retry.update(fb)
        fallback = f'{fb["model"]}:{fb["device"]}:{fb["compute_type"]}'
        try:
            cold = await ensure(retry); t0 = time.monotonic()
            text = await asyncio.to_thread(transcribe_sync, wav, retry)
            elapsed = time.monotonic() - t0
            if text:
                return {"status":"ok","text":text,"cold_start":cold,"asr_seconds":round(elapsed,3),
                        "fallback":fallback,**{k:retry[k] for k in ("model","device","compute_type")}}
        except Exception: pass
        logger.error("Configured OOM fallback failed model=%s device=%s", retry["model"], retry["device"])
        return {"status":"error","message":"ASR fallback failed; see local daemon log",
                "error_code":"fallback_failed","fallback":fallback,
                **{k:retry[k] for k in ("model","device","compute_type")}}

async def worker():
    while True:
        job = await queue.get(); started = time.monotonic()
        try:
            result = await run(job)
            if not job["future"].cancelled(): job["future"].set_result(result)
            write_metric({"timestamp":datetime.now(timezone.utc).isoformat(),
                "config_version":job["profile"]["version"],"model":result.get("model",job["profile"]["model"]),
                "device":result.get("device",job["profile"]["device"]),
                "compute_type":result.get("compute_type",job["profile"]["compute_type"]),
                "audio_duration_seconds":duration(job["wav"]),"cold_start":result.get("cold_start"),
                "asr_seconds":result.get("asr_seconds"),"correction_seconds":result.get("correction_seconds"), "correction_fallback":result.get("correction_fallback"), "correction_mode":result.get("correction_mode"), "total_seconds":round(time.monotonic()-started,3),
                "queue_wait_seconds":round(started-job["queued"],3),"fallback":result.get("fallback"),
                "asr_backend":result.get("asr_backend"),"corrector_backend":result.get("corrector_backend"),
                "corrector_usage":result.get("corrector_usage"),
                "error_code":result.get("error_code"),"error_category":result.get("error_category")})
        except Exception:
            logger.exception("Unexpected worker failure")
            if not job["future"].cancelled(): job["future"].set_result({"status":"error","message":"Internal daemon error","error_code":"internal_error"})
        finally: queue.task_done()

async def respond(writer, value):
    try:
        writer.write(json.dumps(value, ensure_ascii=False).encode()+b"\n"); await writer.drain()
    except (BrokenPipeError, ConnectionResetError): logger.debug("Client disconnected before response")

async def client(reader, writer):
    try:
        raw = await asyncio.wait_for(reader.readline(), timeout=5)
        if not raw: return
        if len(raw) > 16384: return await respond(writer, {"status":"error","message":"Request too large","error_code":"request_too_large"})
        try: req = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError): return await respond(writer, {"status":"error","message":"Invalid JSON","error_code":"invalid_json"})
        if not isinstance(req, dict) or not isinstance(req.get("wav"), str):
            return await respond(writer, {"status":"error","message":"Missing audio file","error_code":"invalid_request"})
        try: profile = request_profile(req, config())
        except ValueError: return await respond(writer, {"status":"error","message":"Invalid recognition profile","error_code":"invalid_profile"})
        source = req.get("source")
        # Cloudflare is permitted only for an explicitly tagged live dictation.
        provider = "cloudflare" if source == "live_dictation" and config()["cloudflare_asr_enabled"] else "local"
        cloudflare_fallback_local = req.get("cloudflare_fallback_local") is True
        future = asyncio.get_running_loop().create_future()
        job_id = req.get("job_id", "")
        if not isinstance(job_id, str) or (job_id and (not job_id.isdigit() or len(job_id) > 32)):
            job_id = ""
        await queue.put({"wav":req["wav"],"profile":profile,"provider":provider,"source":source if source == "live_dictation" else "local_job","job_id":job_id,
                         "cloudflare_fallback_local":cloudflare_fallback_local,"source":source if source == "live_dictation" else "local_job",
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
    if SOCKET.exists(): SOCKET.unlink()
    task = asyncio.create_task(worker()); logger.info("Whisper daemon started")
    try:
        server = await asyncio.start_unix_server(client, str(SOCKET))
        async with server: await server.serve_forever()
    finally:
        # The process is exiting, so the OS will release model memory. Do not
        # drop a model object while a cancelled to_thread inference may still
        # be finishing; daemon-owned model transitions happen between jobs.
        task.cancel()
        if SOCKET.exists(): SOCKET.unlink()
        logger.info("Whisper daemon stopped")
if __name__ == "__main__": asyncio.run(main())
