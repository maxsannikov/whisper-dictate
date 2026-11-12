#!/usr/bin/env python3
"""Optional live-dictation provider clients with sanitized errors and credentials."""
import base64
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave

CONFIG_DIR = Path.home() / ".config/whisper-dictate"
CREDENTIAL_DIR = CONFIG_DIR / "credentials"
TIMEOUT_SECONDS = 90
CLI_CORRECTION_TIMEOUT_SECONDS = 180
MAX_CORRECTION_CHARS = 2400
GROQ_MAX_AUDIO_BYTES = 25 * 1024 * 1024
DEFAULT_OPENROUTER_RETRY_ATTEMPTS = 2
MAX_OPENROUTER_RETRY_ATTEMPTS = 2
OPENROUTER_FREE_DAILY_LIMIT = 50
REMOTE_CORRECTION_INSTRUCTION = (
    "Correct only clear speech recognition, grammar and punctuation errors. "
    "Preserve the meaning, facts, order of meaningful words and speaker's style. "
    "Remove non-semantic spoken fillers and disfluencies such as hesitation "
    "sounds (Russian 'э-э', 'а-а', 'м-м', 'хм'), accidental repetitions, "
    "abandoned syllables or word fragments before self-correction, and false "
    "starts. Remove filler uses of words such as 'ну', 'вот', 'как бы' "
    "or 'значит' only when they carry no meaning in context; keep them when "
    "they express a real connection, attitude or meaning. Never summarize or "
    "add facts, explanations or speech that was not present. Remove obvious "
    "non-spoken subtitle/editor "
    "credits at the very end, especially the signature 'Редактор субтитров "
    "А. Семкин. Корректор А. Егорова.'; if that signature is the entire input, "
    "return an empty result. Return only corrected text."
)

MODEL_IDS = {
    "groq_asr_precise": "whisper-large-v3",
    "groq_asr_fast": "whisper-large-v3-turbo",
    "groq_correct_primary": "qwen/qwen3.8-27b",
    "groq_correct_fallback": "openai/gpt-oss-20b",
    "openrouter_correct": "openrouter/free",
    "google_correct": "gemini-3.5-flash-lite",
    "google_asr": "gemini-3.5-transcribe",
}
DEFAULT_SETTINGS = {
    "asr": {"groq_enabled": False, "google_enabled": False,
            "voice_local_models": [],
            "file_video_local_models": ["faster-whisper:large-v3"],
            "quality": "accuracy", "order": ["cloudflare", "groq", "google", "local"]},
    "correction": {"codex_enabled": True, "claude_enabled": False,
                   "codex_model": "gpt-6-luna", "claude_model": "sonnet",
                   "groq_enabled": False, "openrouter_enabled": False,
                   "google_enabled": False,
                   "enabled": True,
                   "post_transcription_enabled": {
                       "codex": False, "claude": False, "cloudflare": False,
                       "openrouter": False, "groq_primary": False,
                       "groq_fallback": False, "google": False, "local": True,
                   },
                   "order": ["codex", "claude", "cloudflare", "openrouter",
                             "groq_primary", "groq_fallback", "google", "local"],
                   "codex_retry_attempts": 2,
                   "claude_retry_attempts": 2,
                   "groq_primary_retry_attempts": 0,
                   "groq_fallback_retry_attempts": 0,
                   "google_retry_attempts": 0,
                   "openrouter_retry_attempts": DEFAULT_OPENROUTER_RETRY_ATTEMPTS},
}
ASR_ORDER = ("cloudflare", "groq", "google", "local")
FASTER_WHISPER_ASR_MODELS = (
    "large-v3", "large-v3-turbo", "medium", "small", "base", "tiny")
LOCAL_ASR_MODELS = ("faster-whisper",
                    *(f"faster-whisper:{model}" for model in FASTER_WHISPER_ASR_MODELS),
                    "gigaam-v3-e2e-rnnt",
                    "gigaam-v3-e2e-ctc-onnx-gpu", "gigaam-v3-ctc-onnx-gpu")
DEFAULT_VOICE_LOCAL_ASR_MODELS = ()
DEFAULT_FILE_VIDEO_LOCAL_ASR_MODELS = ("faster-whisper:large-v3",)
LOCAL_CORRECTION_MODELS = ("qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M")
CORRECTION_ORDER = ("codex", "claude", "cloudflare", "openrouter",
                    "groq_primary", "groq_fallback", "google", "local")
POST_TRANSCRIPTION_SOURCES = frozenset({"file", "batch", "youtube", "rutube", "vk"})
CODEX_MODELS = ("gpt-6-luna", "gpt-5.6-luna")
# `haiku` follows Anthropic's current Haiku alias; the dated id pins Haiku 4.5.
CLAUDE_MODELS = ("sonnet", "haiku", "claude-haiku-4-5-20251001")


def find_cli_executable(provider):
    """Find user-installed correction CLIs even when systemd has a short PATH."""
    if provider not in {"codex", "claude"}:
        return None
    found = shutil.which(provider)
    if found:
        return found
    for directory in (Path.home() / ".local/bin", Path.home() / ".local/npm/bin"):
        candidate = directory / provider
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def safe_settings(value):
    """Validate toggles and ordering fields; malformed config fails closed."""
    raw = value if isinstance(value, dict) else {}
    asr = raw.get("asr", {}) if isinstance(raw.get("asr", {}), dict) else {}
    correction = raw.get("correction", {}) if isinstance(raw.get("correction", {}), dict) else {}
    retry_attempts = correction.get("openrouter_retry_attempts")
    if (type(retry_attempts) is not int or
            not 0 <= retry_attempts <= MAX_OPENROUTER_RETRY_ATTEMPTS):
        retry_attempts = DEFAULT_OPENROUTER_RETRY_ATTEMPTS
    codex_retry_attempts = correction.get("codex_retry_attempts")
    if type(codex_retry_attempts) is not int or not 0 <= codex_retry_attempts <= 2:
        codex_retry_attempts = 2
    retry_defaults = {"claude_retry_attempts": 2,
                      "groq_primary_retry_attempts": 0,
                      "groq_fallback_retry_attempts": 0,
                      "google_retry_attempts": 0}
    retry_limits = {}
    for key, default in retry_defaults.items():
        value = correction.get(key)
        retry_limits[key] = value if type(value) is int and 0 <= value <= 2 else default
    asr_order = asr.get("order")
    if not isinstance(asr_order, list):
        asr_order = list(ASR_ORDER)
    asr_order = [item for item in asr_order if item in ASR_ORDER]
    asr_order = list(dict.fromkeys(asr_order + list(ASR_ORDER)))
    legacy_local_asr_models = asr.get("local_models", [])
    if not isinstance(legacy_local_asr_models, list):
        legacy_local_asr_models = []
    legacy_local_asr_models = list(dict.fromkeys(
        item for item in legacy_local_asr_models if item in LOCAL_ASR_MODELS))
    voice_local_asr_models = asr.get("voice_local_models")
    if not isinstance(voice_local_asr_models, list) or not voice_local_asr_models:
        voice_local_asr_models = legacy_local_asr_models
    voice_local_asr_models = list(dict.fromkeys(
        item for item in voice_local_asr_models if item in LOCAL_ASR_MODELS))
    if not voice_local_asr_models:
        voice_local_asr_models = list(DEFAULT_VOICE_LOCAL_ASR_MODELS)
    file_video_local_asr_models = asr.get("file_video_local_models")
    if not isinstance(file_video_local_asr_models, list):
        file_video_local_asr_models = list(DEFAULT_FILE_VIDEO_LOCAL_ASR_MODELS)
    file_video_local_asr_models = list(dict.fromkeys(
        item for item in file_video_local_asr_models if item in LOCAL_ASR_MODELS))
    if not file_video_local_asr_models:
        file_video_local_asr_models = list(DEFAULT_FILE_VIDEO_LOCAL_ASR_MODELS)
    # Migrate configurations from the old Cloudflare-only position selector.
    # Once an explicit provider order exists, it is authoritative.
    legacy_position = correction.get("cloudflare_position") if correction.get("cloudflare_position") in {
        "after_groq_primary", "after_groq_fallback", "after_openrouter", "after_google"} else None
    correction_order = correction.get("order")
    if not isinstance(correction_order, list):
        if legacy_position is None:
            correction_order = list(CORRECTION_ORDER)
        else:
            correction_order = list(CORRECTION_ORDER)
            correction_order.remove("cloudflare")
            target = {"after_groq_primary": "groq_primary", "after_groq_fallback": "groq_fallback",
                      "after_openrouter": "openrouter", "after_google": "google"}[legacy_position]
            correction_order.insert(correction_order.index(target) + 1, "cloudflare")
    correction_order = [item for item in correction_order if item in CORRECTION_ORDER]
    correction_order = list(dict.fromkeys(correction_order + list(CORRECTION_ORDER)))
    local_correction_models = correction.get("local_models", [])
    if not isinstance(local_correction_models, list):
        local_correction_models = []
    local_correction_models = list(dict.fromkeys(
        item for item in local_correction_models if item in LOCAL_CORRECTION_MODELS))
    route_names = (*CORRECTION_ORDER,)
    live_routes = correction.get("live_enabled", {})
    manual_routes = correction.get("manual_enabled", {})
    post_routes = correction.get("post_transcription_enabled")
    live_routes = {route: (live_routes.get(route) is not False)
                   for route in route_names} if isinstance(live_routes, dict) else {
                       route: True for route in route_names}
    manual_routes = {route: (manual_routes.get(route) is True)
                     for route in route_names} if isinstance(manual_routes, dict) else {
                         route: False for route in route_names}
    post_routes = ({route: (post_routes.get(route) is True)
                    for route in route_names} if isinstance(post_routes, dict) else
                   {route: route == "local" for route in route_names})
    codex_model = correction.get("codex_model")
    if codex_model not in CODEX_MODELS:
        codex_model = "gpt-6-luna"
    claude_model = correction.get("claude_model")
    if claude_model not in CLAUDE_MODELS:
        claude_model = "sonnet"
    return {
        "asr": {"groq_enabled": asr.get("groq_enabled") is True,
                "google_enabled": asr.get("google_enabled") is True,
                "quality": asr.get("quality") if asr.get("quality") in {"accuracy", "fast"} else "accuracy",
                "order": asr_order, "local_models": voice_local_asr_models,
                "voice_local_models": voice_local_asr_models,
                "file_video_local_models": file_video_local_asr_models},
        "correction": {
            "enabled": correction.get("enabled", True) is True,
            "codex_enabled": correction.get("codex_enabled", True) is True,
            "claude_enabled": correction.get("claude_enabled") is True,
            "codex_model": codex_model,
            "claude_model": claude_model,
            "groq_enabled": correction.get("groq_enabled") is True,
            "openrouter_enabled": correction.get("openrouter_enabled") is True,
            "google_enabled": correction.get("google_enabled") is True,
            "codex_retry_attempts": codex_retry_attempts,
            **retry_limits,
            "openrouter_retry_attempts": retry_attempts,
            "order": correction_order, "local_models": local_correction_models,
            "live_enabled": live_routes, "manual_enabled": manual_routes,
            "post_transcription_enabled": post_routes,
        },
    }


def resolve_local_asr_model(model_id, selected_whisper_model="large-v3"):
    """Resolve an ASR order entry to its runtime backend and model profile."""
    if model_id == "faster-whisper":
        return "faster-whisper", selected_whisper_model
    prefix = "faster-whisper:"
    if isinstance(model_id, str) and model_id.startswith(prefix):
        model_name = model_id[len(prefix):]
        if model_name in FASTER_WHISPER_ASR_MODELS:
            return "faster-whisper", model_name
    if model_id in LOCAL_ASR_MODELS:
        return model_id, selected_whisper_model
    return "faster-whisper", selected_whisper_model


def asr_chain(source, settings, cloudflare_enabled):
    """Return configured live-only ASR order, retaining local as a safe fallback."""
    if source != "live_dictation":
        return ["local"]
    settings = safe_settings(settings)
    asr = settings["asr"]
    routes = []
    for item in asr["order"]:
        if item == "cloudflare" and cloudflare_enabled:
            routes.append("cloudflare")
        elif item == "groq" and asr["groq_enabled"]:
            routes.append("groq_fast" if asr["quality"] == "fast" else "groq_precise")
        elif item == "google" and asr["google_enabled"]:
            routes.append("google")
        elif item == "local":
            routes.append("local")
    if "local" not in routes:
        routes.append("local")
    elif routes[-1] != "local":
        routes.append("local")
    return routes


def correction_chain(source, settings, cloudflare_enabled):
    """Return scenario-enabled correction providers in their saved order."""
    if source not in {"live_dictation", "manual_correction", *POST_TRANSCRIPTION_SOURCES}:
        return ["local"]
    settings = safe_settings(settings)["correction"]
    if not settings["enabled"]:
        return ["local"]
    scenario = ("live_enabled" if source == "live_dictation" else
                "manual_enabled" if source == "manual_correction" else
                "post_transcription_enabled")
    scenario_routes = settings[scenario]
    chain = []
    for item in settings["order"]:
        if item == "codex" and settings["codex_enabled"] and scenario_routes[item]:
            chain.append("codex")
        elif item == "claude" and settings["claude_enabled"] and scenario_routes[item]:
            chain.append("claude")
        elif item == "openrouter" and settings["openrouter_enabled"] and scenario_routes[item]:
            chain.append("openrouter")
        elif item == "cloudflare" and cloudflare_enabled and scenario_routes[item]:
            chain.append("cloudflare")
        elif (item in {"groq_primary", "groq_fallback"} and
              settings["groq_enabled"] and scenario_routes[item]):
            chain.append(item)
        elif item == "google" and settings["google_enabled"] and scenario_routes[item]:
            chain.append("google")
        elif item == "local" and scenario_routes[item]:
            chain.append("local")
    if "local" not in chain and scenario_routes["local"]:
        chain.append("local")
    elif "local" in chain and chain[-1] != "local":
        chain.append("local")
    return chain


class ProviderError(Exception):
    def __init__(self, provider, category, model=None):
        super().__init__(f"{provider} request failed ({category})")
        self.provider = provider
        self.category = category
        self.model = (model if isinstance(model, str) and
                      re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", model) else None)


def credential_path(provider):
    if provider not in {"groq", "google", "openrouter"}:
        raise ValueError("unsupported provider")
    return CREDENTIAL_DIR / f"{provider}.key"


def credentials_configured(provider):
    path = credential_path(provider)
    return (path.is_file() and not (path.stat().st_mode & 0o077) and
            not (CREDENTIAL_DIR.stat().st_mode & 0o077) and path.stat().st_size > 0)


def store_credential(provider, secret):
    """Store a user-entered API key with restrictive permissions."""
    if not isinstance(secret, str) or not secret.strip() or "\n" in secret:
        raise ValueError("invalid credential")
    CREDENTIAL_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(CREDENTIAL_DIR, 0o700)
    path = credential_path(provider)
    temp = path.with_suffix(".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(secret.strip())
            stream.write("\n")
        os.chmod(temp, 0o600)
        os.replace(temp, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            temp.unlink()
        except OSError:
            pass
        raise


def _credential(provider):
    path = credential_path(provider)
    try:
        secret = path.read_text(encoding="utf-8").strip()
    except OSError:
        raise ProviderError(provider, "missing_credentials") from None
    if not secret or path.stat().st_mode & 0o077 or CREDENTIAL_DIR.stat().st_mode & 0o077:
        raise ProviderError(provider, "credentials_invalid")
    return secret


def validate_credential(provider):
    """Make a minimal, user-requested key check; never send audio or transcripts."""
    key = _credential(provider)
    if provider == "groq":
        url, headers = "https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"}
    elif provider == "openrouter":
        url, headers = "https://openrouter.ai/api/v1/models", {"Authorization": f"Bearer {key}"}
    elif provider == "google":
        url, headers = "https://generativelanguage.googleapis.com/v1beta/models", {"x-goog-api-key": key}
    else:
        raise ValueError("unsupported provider")
    payload = _request(provider, urllib.request.Request(url, headers=headers), timeout=20)
    if not isinstance(payload, dict) or not isinstance(payload.get("models", payload.get("data")), list):
        raise ProviderError(provider, "invalid_response")
    return True


def _category_for_http(code):
    if code in (401, 403):
        return "http_401_403"
    if code == 413:
        return "http_413"
    if code == 429:
        return "http_429"
    if code in (404, 422):
        return "unsupported_model"
    if code in (500, 502, 503, 504):
        return "provider_unavailable"
    return "http_error"


def _openrouter_daily_quota_error(error, now=None):
    """Classify only a confirmed free-model daily limit; ignore RPM/upstream 429s."""
    try:
        raw = error.read(1024 * 1024)
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        payload = {}
    body_error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = body_error.get("message", "") if isinstance(body_error, dict) else ""
    metadata = body_error.get("metadata", {}) if isinstance(body_error, dict) else {}
    if isinstance(message, str) and "free-models-per-day" in message.lower():
        return True
    if isinstance(metadata, dict) and metadata.get("limit_source") == "openrouter_free_tier_daily":
        return True
    headers = getattr(error, "headers", None)
    if headers is None:
        return False
    def header(name):
        return headers.get(name) or headers.get(name.lower())
    try:
        limit = int(header("X-RateLimit-Limit"))
        remaining = int(header("X-RateLimit-Remaining"))
        reset = float(header("X-RateLimit-Reset"))
        if reset > 10_000_000_000:
            reset /= 1000
        reset_at = datetime.fromtimestamp(reset, timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return False
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
    return (remaining == 0 and limit in {50, 1000} and
            abs((reset_at - midnight).total_seconds()) <= 60)


def _request(provider, request, timeout=TIMEOUT_SECONDS):
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ProviderError(provider, "invalid_response")
        return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as error:
        category = ("daily_quota_exhausted" if provider == "openrouter" and
                    _openrouter_daily_quota_error(error) else _category_for_http(error.code))
        raise ProviderError(provider, category) from None
    except TimeoutError:
        raise ProviderError(provider, "timeout") from None
    except urllib.error.URLError as error:
        raise ProviderError(provider, "timeout" if isinstance(error_reason(error), TimeoutError) else "network_error") from None
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        raise ProviderError(provider, "invalid_response") from None


def error_reason(error):
    return getattr(error, "reason", None)


def _groq_transcribe_single(audio_path, language, model):
    model = model or MODEL_IDS["groq_asr_precise"]
    size = Path(audio_path).stat().st_size
    if size > GROQ_MAX_AUDIO_BYTES:
        raise ProviderError("groq", "http_413")
    key = _credential("groq")
    boundary = "WhisperDictateBoundary"
    filename = "audio.wav"
    audio = Path(audio_path).read_bytes()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n".encode(),
    ]
    if language:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{language}\r\n".encode())
    parts.extend([f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n".encode(), audio, f"\r\n--{boundary}--\r\n".encode()])
    request = urllib.request.Request("https://api.groq.com/openai/v1/audio/transcriptions",
        data=b"".join(parts), method="POST", headers={"Authorization": f"Bearer {key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    payload = _request("groq", request)
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ProviderError("groq", "invalid_response")
    return text.strip()


def groq_transcribe(audio_path, language="ru", model=None, opener=None):
    """Transcribe directly under the cap; split oversized WAV locally with FFmpeg."""
    path = Path(audio_path)
    try:
        size = path.stat().st_size
    except OSError:
        raise ProviderError("groq", "local_audio_error") from None
    if size <= GROQ_MAX_AUDIO_BYTES:
        return _groq_transcribe_single(path, language, model)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ProviderError("groq", "audio_too_large")
    try:
        with tempfile.TemporaryDirectory(prefix="whisper-groq-") as directory:
            pattern = str(Path(directory) / "chunk-%03d.wav")
            subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
                "-map_metadata", "-1", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                "-f", "segment", "-segment_time", "600", "-reset_timestamps", "1", pattern],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=180, check=True)
            chunks = sorted(Path(directory).glob("chunk-*.wav"))
            if not chunks:
                raise ProviderError("groq", "invalid_response")
            transcripts = []
            for chunk in chunks:
                if chunk.stat().st_size > GROQ_MAX_AUDIO_BYTES:
                    raise ProviderError("groq", "audio_too_large")
                transcripts.append(_groq_transcribe_single(chunk, language, model))
            return " ".join(transcripts).strip()
    except ProviderError:
        raise
    except subprocess.TimeoutExpired:
        raise ProviderError("groq", "timeout") from None
    except (OSError, subprocess.SubprocessError):
        raise ProviderError("groq", "local_audio_error") from None


def _audio_duration_seconds(path):
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            return audio.getnframes() / rate if rate else None
    except (OSError, wave.Error):
        return None


def _chat(provider, endpoint, key, model, text, opener=None, include_model=False):
    if provider == "openrouter" and (model == "openrouter/free" or str(model).endswith(":free")):
        try:
            reserved = _increment_openrouter_free_attempts()
        except OSError:
            raise ProviderError("openrouter", "attempt_counter_unavailable", model) from None
        if not reserved:
            raise ProviderError("openrouter", "daily_quota_exhausted", model)
    request = urllib.request.Request(endpoint,
        data=json.dumps({"model": model, "messages": [
            {"role": "system", "content": REMOTE_CORRECTION_INSTRUCTION},
            {"role": "user", "content": text}], "temperature": 0, "stream": False,
            "max_tokens": 4096}, ensure_ascii=False).encode("utf-8"),
        method="POST", headers={"Authorization": f"Bearer {key}",
        "Content-Type": "application/json"})
    payload = _request(provider, request)
    selected_model = payload.get("model") if provider == "openrouter" and isinstance(payload, dict) else None
    if not isinstance(selected_model, str) or not re.fullmatch(
            r"[A-Za-z0-9_.:/-]{1,160}", selected_model):
        selected_model = None
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ProviderError(provider, "invalid_response_missing_choice", selected_model)
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict) or "content" not in message:
        raise ProviderError(provider, "invalid_response_missing_content", selected_model)
    content = message["content"]
    if not isinstance(content, str):
        raise ProviderError(provider, "invalid_response_non_string", selected_model)
    if not content.strip():
        raise ProviderError(provider, "invalid_response_empty", selected_model)
    result = content.strip()
    if include_model:
        return result, selected_model
    return result


def _interaction_text(payload):
    if not isinstance(payload, dict):
        return None
    direct = payload.get("output_text")
    if isinstance(direct, str):
        return direct.strip()
    output = []
    for step in payload.get("steps", []):
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for content in step.get("content", []):
            if isinstance(content, dict) and content.get("type") == "text" and isinstance(content.get("text"), str):
                output.append(content["text"])
    return "\n".join(output).strip() or None


def groq_correct(text, model):
    return _chat("groq", "https://api.groq.com/openai/v1/chat/completions",
                 _credential("groq"), model, text)


def cli_correct(provider, text, model=None, cancel_callback=None):
    """Run an installed CLI in an empty ephemeral directory with no tools."""
    if provider not in {"codex", "claude"}:
        raise ProviderError("local", "invalid_request")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_CORRECTION_CHARS:
        raise ProviderError(provider, "invalid_request")
    if model is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", model):
        raise ProviderError(provider, "invalid_request")
    cli = find_cli_executable(provider)
    if not cli:
        raise ProviderError(provider, "cli_unavailable")
    instructions = (
        "You are a speech-recognition correction engine. " + REMOTE_CORRECTION_INSTRUCTION +
        " Never access files or run commands.\n\n"
        "Text to correct:\n" + text
    )
    with tempfile.TemporaryDirectory(prefix="whisper-cli-") as directory:
        output_path = Path(directory) / "final.txt"
        stdout_path = Path(directory) / "stdout.txt"
        if provider == "codex":
            command = [cli, "exec", "--ephemeral", "--ignore-user-config", "--sandbox", "read-only",
                "--skip-git-repo-check", "--cd", directory, "--output-last-message", str(output_path),
                "--color", "never"]
            if model:
                command.extend(("--model", model))
            for feature in ("shell_tool", "view_image", "web_search_request", "browser_use",
                            "browser_use_external", "computer_use", "apps", "code_mode_host"):
                command.extend(("--disable", feature))
            command.append("-")
        else:
            command = [cli, "-p", "--input-format", "text", "--tools", "",
                       "--safe-mode", "--disallowedTools", "mcp__*",
                       "--permission-prompts", "none", "--no-session-persistence",
                       "--output-format", "text", "--max-turns", "1"]
            if model:
                command.extend(("--model", model))
        stdout_stream = (stdout_path.open("wb") if provider == "claude" else None)
        child_env = os.environ.copy()
        cli_directory = str(Path(cli).parent)
        current_path = child_env.get("PATH", "")
        child_env["PATH"] = cli_directory + (os.pathsep + current_path if current_path else "")
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE,
                stdout=(subprocess.DEVNULL if provider == "codex" else stdout_stream),
                stderr=subprocess.DEVNULL, cwd=directory, env=child_env)
        except OSError:
            if stdout_stream:
                stdout_stream.close()
            raise ProviderError(provider, "cli_unavailable") from None
        try:
            payload = instructions.encode("utf-8")
            started = time.monotonic()
            while True:
                if cancel_callback and cancel_callback():
                    process.terminate()
                    try:
                        process.communicate(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.communicate()
                    raise ProviderError(provider, "cancelled_by_user")
                if time.monotonic() - started >= CLI_CORRECTION_TIMEOUT_SECONDS:
                    process.kill()
                    process.communicate()
                    raise ProviderError(provider, "cli_timeout")
                try:
                    process.communicate(input=payload, timeout=0.2)
                    payload = None
                    break
                except subprocess.TimeoutExpired:
                    payload = None
                    if provider == "claude" and stdout_path.stat().st_size > 16384:
                        process.kill(); process.communicate()
                        raise ProviderError(provider, "cli_output_invalid")
                    if process.poll() is None:
                        continue
                    process.communicate()
                    break
            if process.returncode != 0:
                raise ProviderError(provider, "cli_failed")
            if provider == "codex":
                if not output_path.is_file() or output_path.stat().st_size > 16384:
                    raise ProviderError(provider, "cli_output_invalid")
                result = output_path.read_text(encoding="utf-8").strip()
            else:
                stdout_stream.flush()
                if stdout_path.stat().st_size > 16384:
                    raise ProviderError(provider, "cli_output_invalid")
                result = stdout_path.read_text(encoding="utf-8").strip()
        except subprocess.TimeoutExpired:
            process.kill(); process.communicate()
            raise ProviderError(provider, "cli_timeout") from None
        except UnicodeError:
            raise ProviderError(provider, "cli_output_invalid") from None
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
            if stdout_stream:
                stdout_stream.close()
    if not result:
        raise ProviderError(provider, "cli_output_invalid")
    return result


def codex_correct(text, model=None, cancel_callback=None):
    return cli_correct("codex", text, model or "gpt-6-luna", cancel_callback)


def claude_correct(text, model=None, cancel_callback=None):
    return cli_correct("claude", text, model or "sonnet", cancel_callback)


def _openrouter_correct_sync(text, model=None, include_model=False):
    return _chat("openrouter", "https://openrouter.ai/api/v1/chat/completions",
                 _credential("openrouter"), model or MODEL_IDS["openrouter_correct"], text,
                 include_model=include_model)


def openrouter_correct(text, model=None, include_model=False, cancel_callback=None):
    """Run cancellable correction in a child process when a job token is supplied."""
    if cancel_callback is None:
        return _openrouter_correct_sync(text, model, include_model)
    payload = json.dumps({"text": text, "model": model, "include_model": include_model}).encode("utf-8")
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--openrouter-correction-child"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            if cancel_callback():
                process.terminate()
                try:
                    process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                raise ProviderError("openrouter", "cancelled_by_user")
            try:
                output, _ = process.communicate(input=payload, timeout=0.15)
                break
            except subprocess.TimeoutExpired:
                continue
        try:
            result = json.loads(output.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderError("openrouter", "invalid_response") from None
        if process.returncode != 0 or not isinstance(result, dict):
            category = result.get("category") if isinstance(result, dict) else None
            model_id = result.get("model") if isinstance(result, dict) else None
            if category not in {"missing_credentials", "credentials_invalid", "http_401_403",
                    "http_413", "http_429", "unsupported_model", "provider_unavailable",
                    "http_error", "timeout", "network_error", "invalid_response",
                    "daily_quota_exhausted"}:
                category = "invalid_response"
            raise ProviderError("openrouter", category, model_id)
        answer = result.get("text")
        selected_model = result.get("model")
        if not isinstance(answer, str):
            raise ProviderError("openrouter", "invalid_response", selected_model)
        return (answer, selected_model) if include_model else answer
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _openrouter_usage_path():
    return Path.home() / ".cache/whisper-dictate/openrouter-usage/usage.json"


def _increment_openrouter_free_attempts(path=None, now=None):
    """Atomically reserve one of this app's 50 daily free endpoint requests."""
    path = Path(path) if path is not None else _openrouter_usage_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    today = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()
    lock_path = path.with_suffix(".lock")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = {}
        except (OSError, ValueError, TypeError):
            raise OSError("OpenRouter local usage counter cannot be read") from None
        count = 0
        if isinstance(data, dict) and data.get("date") == today:
            count = data.get("count")
            if type(count) is not int or count < 0:
                raise OSError("OpenRouter local usage counter is invalid")
        if count >= OPENROUTER_FREE_DAILY_LIMIT:
            return False
        temp = path.with_suffix(".tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"date": today, "count": count + 1}, stream)
                stream.write("\n")
            os.replace(temp, path)
            os.chmod(path, 0o600)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        return True
    finally:
        os.close(lock_fd)


def openrouter_local_usage():
    """Return UTC date and this installation's free endpoint attempt count."""
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        data = json.loads(_openrouter_usage_path().read_text(encoding="utf-8"))
        count = data.get("count", 0)
        if data.get("date") == today and isinstance(count, int) and count >= 0:
            return today, count
    except (OSError, ValueError, TypeError):
        pass
    return today, 0


def google_correct(text, model=None):
    key = _credential("google")
    model = model or MODEL_IDS["google_correct"]
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=json.dumps({"model": model,
            "input": "Correct clear ASR, grammar and punctuation errors. Preserve meaning and all meaningful words, but remove non-semantic spoken fillers such as hesitation sounds ('э-э', 'а-а', 'м-м', 'хм'), accidental repetitions, abandoned syllables or word fragments before self-correction, and false starts. Remove 'ну', 'вот', 'как бы' or 'значит' only when they are empty fillers, not when meaningful. Never summarize or add content. Return only corrected text.\n\n" + text,
            "store": False}, ensure_ascii=False).encode("utf-8"),
        method="POST", headers={"Content-Type": "application/json", "x-goog-api-key": key})
    payload = _request("google", request)
    output = _interaction_text(payload)
    if not isinstance(output, str) or not output.strip():
        raise ProviderError("google", "invalid_response")
    return output.strip()


def transcribe_gemini(audio_path, language="ru"):
    """Upload audio through Files API, transcribe it, then delete remote file."""
    path = Path(audio_path)
    duration = _audio_duration_seconds(path)
    if duration is not None and duration > 3600:
        raise ProviderError("google", "audio_too_long")
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ProviderError("google", "http_413")
    key = _credential("google")
    mime = "audio/wav"
    metadata = json.dumps({"file": {"display_name": "whisper-dictate-audio"}}).encode()
    start = urllib.request.Request("https://generativelanguage.googleapis.com/upload/v1beta/files",
        data=metadata, method="POST", headers={"x-goog-api-key": key,
        "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(path.stat().st_size),
        "X-Goog-Upload-Header-Content-Type": mime, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(start, timeout=TIMEOUT_SECONDS) as response:
            upload_url = response.headers.get("X-Goog-Upload-URL")
    except urllib.error.HTTPError as error:
        raise ProviderError("google", _category_for_http(error.code)) from None
    except TimeoutError:
        raise ProviderError("google", "timeout") from None
    except (urllib.error.URLError, OSError):
        raise ProviderError("google", "network_error") from None
    if not upload_url or not upload_url.startswith("https://generativelanguage.googleapis.com/"):
        raise ProviderError("google", "invalid_response")
    try:
        audio = path.read_bytes()
    except OSError:
        raise ProviderError("google", "local_audio_error") from None
    upload = urllib.request.Request(upload_url, data=audio, method="POST",
        headers={"Content-Length": str(len(audio)), "X-Goog-Upload-Offset": "0",
                 "X-Goog-Upload-Command": "upload, finalize", "Content-Type": mime})
    file_name = None
    try:
        file_info = _request("google", upload)
        file_obj = file_info.get("file", {}) if isinstance(file_info, dict) else {}
        file_name = file_obj.get("name")
        file_uri = file_obj.get("uri")
        if not isinstance(file_name, str) or not isinstance(file_uri, str):
            raise ProviderError("google", "invalid_response")
        input_items = []
        if language:
            input_items.append({"type": "text", "text": f"Transcribe in language {language}."})
        input_items.append({"type": "audio", "uri": file_uri, "mime_type": mime})
        body = json.dumps({"model": MODEL_IDS["google_asr"], "input": input_items,
                           "store": False}, ensure_ascii=False).encode()
        request = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/interactions",
            data=body, method="POST", headers={"x-goog-api-key": key,
            "Content-Type": "application/json"})
        result = _request("google", request)
        output = _interaction_text(result)
        if not isinstance(output, str) or not output.strip():
            raise ProviderError("google", "invalid_response")
        return output.strip()
    finally:
        if file_name:
            delete = urllib.request.Request(
                "https://generativelanguage.googleapis.com/v1beta/" + file_name,
                method="DELETE", headers={"x-goog-api-key": key})
            try:
                urllib.request.urlopen(delete, timeout=TIMEOUT_SECONDS).close()
            except Exception:
                pass


def _openrouter_correction_child():
    """Private subprocess entry point; transcript and response travel via pipes only."""
    try:
        request = json.load(sys.stdin)
        text, model, include_model = request["text"], request.get("model"), request.get("include_model") is True
        result = _openrouter_correct_sync(text, model, include_model=True)
        answer, selected_model = result
        json.dump({"text": answer, "model": selected_model}, sys.stdout, ensure_ascii=False)
        return 0
    except ProviderError as error:
        json.dump({"category": error.category, "model": error.model}, sys.stdout)
        return 1
    except Exception:
        json.dump({"category": "invalid_response"}, sys.stdout)
        return 1


if __name__ == "__main__" and sys.argv[1:] == ["--openrouter-correction-child"]:
    raise SystemExit(_openrouter_correction_child())
