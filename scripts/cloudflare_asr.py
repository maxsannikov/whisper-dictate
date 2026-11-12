#!/usr/bin/env python3
"""Безопасный клиент Cloudflare Workers AI для Whisper Dictate."""
import base64
from datetime import datetime, timezone
import fcntl
import inspect
import json
import math
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ENV_FILE = Path.home() / ".config/whisper-dictate/cloudflare-workers-ai.env"
MODEL = "@cf/openai/whisper-large-v3-turbo"
TIMEOUT_SECONDS = 300
FREE_DAILY_NEURONS = 10000
NEURONS_PER_AUDIO_MINUTE = 46.63
USAGE_PATH = Path.home() / ".cache/whisper-dictate/cloudflare-daily-usage.json"


class CloudflareError(Exception):
    """Sanitized provider error with a safe machine-readable category."""
    def __init__(self, message, category="invalid_response"):
        super().__init__(message)
        self.category = category


def cloudflare_http_error_category(error):
    """Recognize the documented daily Workers AI exhaustion code, never the body text."""
    if getattr(error, "code", None) != 429:
        return "http_401_403" if getattr(error, "code", None) in (401, 403) else "http_error"
    try:
        payload = json.loads(error.read(1024 * 1024).decode("utf-8"))
        errors = payload.get("errors", []) if isinstance(payload, dict) else []
        if isinstance(errors, list) and any(isinstance(item, dict) and item.get("code") == 3036
                                             for item in errors):
            return "daily_quota_exhausted"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
        pass
    return "http_429"


def _reservation_state(path, day):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 0.0
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(value, dict) or value.get("utc_date") != day.isoformat():
        return 0.0
    neurons = value.get("reserved_neurons")
    if (isinstance(neurons, bool) or not isinstance(neurons, (int, float)) or
            not math.isfinite(neurons) or neurons < 0):
        return None
    return float(neurons)


def daily_usage(metrics_path, now=None, include_reservations=True):
    """Estimate current UTC-day Workers AI neurons, including local reservations."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(timezone.utc).date()
    audio_seconds = 0.0
    requests = 0
    corrector_neurons = 0.0
    corrector_requests = 0
    estimated_token_requests = 0
    try:
        with open(metrics_path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                    timestamp = datetime.fromisoformat(entry.get("timestamp", "").replace("Z", "+00:00"))
                except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
                    continue
                if timestamp.tzinfo is None or timestamp.astimezone(timezone.utc).date() != today:
                    continue
                try:
                    seconds = float(entry.get("audio_duration_seconds"))
                except (TypeError, ValueError):
                    seconds = None
                if (seconds is not None and math.isfinite(seconds) and
                        entry.get("model") == MODEL and entry.get("error_code") is None and seconds > 0):
                    audio_seconds += seconds
                    requests += 1
                usage = entry.get("corrector_usage")
                if entry.get("corrector_backend") == "cloudflare" and isinstance(usage, dict):
                    neurons = usage.get("estimated_neurons")
                    if isinstance(neurons, (int, float)) and math.isfinite(neurons) and neurons >= 0:
                        corrector_neurons += neurons
                        corrector_requests += 1
                        if usage.get("estimated") is True:
                            estimated_token_requests += 1
    except OSError:
        pass
    neurons = audio_seconds / 60 * NEURONS_PER_AUDIO_MINUTE
    metric_neurons = neurons + corrector_neurons
    reservations = _reservation_state(USAGE_PATH, today) if include_reservations else 0.0
    total_neurons = max(metric_neurons, reservations or 0.0)
    return {"requests": requests, "audio_seconds": audio_seconds,
            "corrector_requests": corrector_requests,
            "corrector_estimated_token_requests": estimated_token_requests,
            "corrector_estimated_neurons": corrector_neurons,
            "estimated_neurons": neurons, "estimated_total_neurons": total_neurons,
            "reserved_neurons": reservations,
            "usage_guard_available": reservations is not None,
            "daily_limit": FREE_DAILY_NEURONS,
            "estimated_remaining": max(0.0, FREE_DAILY_NEURONS - total_neurons)}


def reserve_daily_neurons(metrics_path, estimated_neurons, path=USAGE_PATH, now=None):
    """Reserve estimated neurons before a request; return False if the free budget won't cover it."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(timezone.utc).date()
    try:
        estimate = float(estimated_neurons)
    except (TypeError, ValueError):
        raise CloudflareError("Не удалось оценить расход Workers AI", "usage_guard_unavailable") from None
    if not math.isfinite(estimate) or estimate <= 0:
        raise CloudflareError("Не удалось оценить расход Workers AI", "usage_guard_unavailable")
    target = Path(path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    lock_path = target.with_suffix(target.suffix + ".lock")
    try:
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        raise CloudflareError("Не удалось проверить дневной лимит Workers AI",
                              "usage_guard_unavailable") from None
    try:
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        recorded = daily_usage(metrics_path, now, include_reservations=False)["estimated_total_neurons"]
        reserved = _reservation_state(target, today)
        if reserved is None:
            raise CloudflareError("Счётчик дневного расхода Workers AI повреждён",
                                  "usage_guard_unavailable")
        current = max(recorded, reserved)
        if current + estimate > FREE_DAILY_NEURONS:
            return False
        fd, temp_name = tempfile.mkstemp(prefix=".cloudflare-usage-", dir=target.parent, text=True)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"utc_date": today.isoformat(),
                           "reserved_neurons": round(current + estimate, 4)}, stream,
                          separators=(",", ":"))
                stream.write("\n")
            os.replace(temp_name, target)
            os.chmod(target, 0o600)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        return True
    except OSError:
        raise CloudflareError("Не удалось обновить счётчик дневного расхода Workers AI",
                              "usage_guard_unavailable") from None
    finally:
        os.close(lock_fd)


async def cloudflare_first(cloudflare_call, local_call, fallback_local=False):
    """Run Cloudflare first and optionally delegate failures to local ASR."""
    try:
        result = cloudflare_call()
        if inspect.isawaitable(result):
            result = await result
        return result, False, None
    except CloudflareError as error:
        if not fallback_local:
            raise
        result = local_call()
        if inspect.isawaitable(result):
            result = await result
        return result, True, error


def parse_response(payload):
    """Извлечь распознанный текст из успешного ответа API."""
    if not isinstance(payload, dict) or payload.get("success") is False:
        raise CloudflareError("Некорректный ответ Cloudflare Workers AI")
    result = payload.get("result")
    text = result.get("text") if isinstance(result, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise CloudflareError("Cloudflare Workers AI вернул пустой или некорректный ответ")
    return text.strip()


def read_credentials(path=ENV_FILE):
    if not path.is_file():
        raise CloudflareError("Не найден файл учётных данных Cloudflare", "missing_credentials")
    values = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"}:
                values[key.strip()] = value.strip().strip("\"'")
    except OSError:
        raise CloudflareError("Не удалось прочитать файл учётных данных Cloudflare", "credentials_read_error") from None
    if not values.get("CLOUDFLARE_ACCOUNT_ID") or not values.get("CLOUDFLARE_API_TOKEN"):
        raise CloudflareError("В файле Cloudflare отсутствует Account ID или API Token", "missing_credentials")
    return values["CLOUDFLARE_ACCOUNT_ID"], values["CLOUDFLARE_API_TOKEN"]


def transcribe(audio_path, language="ru", vad=True, initial_prompt="", model=MODEL, settings=None):
    account_id, token = read_credentials()
    try:
        audio = base64.b64encode(Path(audio_path).read_bytes()).decode("ascii")
    except OSError:
        raise CloudflareError("Не удалось прочитать аудиофайл", "local_audio_error") from None
    settings = settings if isinstance(settings, dict) else {}
    language = language or ""
    body = {"audio": audio,
            "task": settings.get("task", "transcribe"),
            "vad_filter": bool(vad)}
    if language:
        body["language"] = language
    if model == MODEL:
        body["beam_size"] = settings.get("beam_size", 1)
        body["condition_on_previous_text"] = settings.get("condition_on_previous_text", True)
        if initial_prompt:
            body["initial_prompt"] = initial_prompt
        if settings.get("prefix"):
            body["prefix"] = settings["prefix"]
        for key in ("no_speech_threshold", "compression_ratio_threshold", "log_prob_threshold", "hallucination_silence_threshold"):
            if settings.get(key) is not None:
                body[key] = settings[key]
    url = ("https://api.cloudflare.com/client/v4/accounts/" + account_id
           + "/ai/run/" + (model if model in {MODEL, "@cf/openai/whisper"} else MODEL))
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + token,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return parse_response(payload)
    except urllib.error.HTTPError as exc:
        category = cloudflare_http_error_category(exc)
        if category == "daily_quota_exhausted":
            message = "Cloudflare Workers AI исчерпал дневную квоту"
        elif exc.code in (401, 403):
            message = "Cloudflare отклонил учётные данные (401/403)"
        elif exc.code == 429:
            message = "Cloudflare Workers AI временно ограничил запрос (429)"
        else:
            message = f"Ошибка API Cloudflare (HTTP {exc.code})"
        raise CloudflareError(message, category) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CloudflareError("Сетевая ошибка или тайм-аут Cloudflare Workers AI", "timeout" if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError) else "network_error") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise CloudflareError("Cloudflare вернул некорректный JSON") from None


def main(argv):
    if len(argv) != 3:
        print("Некорректные параметры Cloudflare ASR", file=sys.stderr)
        return 2
    try:
        profile = Path.home() / ".config/whisper-dictate/config.json"
        try:
            config = json.loads(profile.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
        text = transcribe(argv[1], argv[2], bool(config.get("vad", False)),
                          config.get("initial_prompt", ""))
    except CloudflareError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("Непредвиденная ошибка Cloudflare Workers AI", file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
