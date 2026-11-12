#!/usr/bin/env python3
"""Aggregate privacy-safe ASR and correction attempt metrics for one UTC day."""
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROVIDERS = {"codex", "claude", "cloudflare", "openrouter", "groq", "google", "local"}
STAGES = {"asr", "correction"}
ASR_FAILURE_CODES = {"audio_not_found", "empty_transcription", "asr_failed",
                     "cloudflare_asr_failed", "oom_cpu_fallback_disabled", "fallback_failed",
                     "internal_error"}
RESET_PATH = Path.home() / ".cache/whisper-dictate/attempt-statistics-reset.json"


def _read_reset_time(path=RESET_PATH):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(value.get("reset_at", ""))
        return timestamp.astimezone(timezone.utc) if timestamp.tzinfo else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def reset_attempt_statistics(path=RESET_PATH, now=None):
    """Set a new reporting baseline without deleting the metrics journal."""
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    target = Path(path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".attempt-stats-reset-", dir=target.parent, text=True)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"reset_at": timestamp.isoformat()}, stream, separators=(",", ":"))
            stream.write("\n")
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return timestamp


def _metric_day(record):
    try:
        timestamp = datetime.fromisoformat(str(record.get("timestamp", "")).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            return None
        return timestamp.astimezone(timezone.utc).date()
    except (ValueError, TypeError, OverflowError):
        return None


def _safe_seconds(value):
    try:
        value = float(value)
        return value if 0 <= value < 86400 else None
    except (TypeError, ValueError):
        return None


def _add(groups, attempt, audio_seconds=None):
    if not isinstance(attempt, dict):
        return
    stage, provider = attempt.get("stage"), attempt.get("provider")
    if stage not in STAGES or provider not in PROVIDERS:
        return
    model = attempt.get("model")
    if not isinstance(model, str) or not model.strip():
        model = "—"
    model = " ".join(model.split())[:96]
    if (provider == "openrouter" and model == "openrouter/free" and
            attempt.get("success") is not True):
        model = "openrouter/free (upstream model unknown)"
    device = attempt.get("device")
    if provider == "local" and device in {"cuda", "cpu"}:
        model = f"{model} ({'GPU' if device == 'cuda' else 'CPU'})"
    key = (stage, provider, model)
    row = groups[key]
    row["attempts"] += 1
    success = attempt.get("success") is True
    row["successes" if success else "failures"] += 1
    seconds = _safe_seconds(attempt.get("seconds"))
    if success and seconds is not None:
        row["successful_seconds"] += seconds
        row["timed_successes"] += 1
        audio_seconds = _safe_seconds(audio_seconds)
        if audio_seconds:
            row["audio_seconds"] += audio_seconds


def _legacy_attempts(record):
    """Build a best-effort view for older metric rows created before attempt ledgers."""
    attempts = []
    for error in record.get("provider_errors", []):
        if isinstance(error, dict):
            attempts.append({"stage": error.get("stage"), "provider": error.get("provider"),
                "model": error.get("model"), "success": False})
    error_code = record.get("error_code")
    asr_provider = record.get("asr_backend")
    if asr_provider == "local_fallback":
        asr_provider = "local"
    if asr_provider in PROVIDERS:
        attempts.append({"stage": "asr", "provider": asr_provider,
            "model": record.get("asr_model", record.get("model")),
            "success": error_code not in ASR_FAILURE_CODES,
            "seconds": record.get("asr_seconds"), "device": record.get("asr_device")})
    correction_provider = record.get("corrector_backend")
    if correction_provider == "local_fallback":
        correction_provider = "local"
    if correction_provider in PROVIDERS:
        attempts.append({"stage": "correction", "provider": correction_provider,
            "model": record.get("corrector_model"),
            "success": error_code is None and record.get("correction_fallback") is None,
            "seconds": record.get("correction_seconds"), "device": record.get("corrector_device")})
    return attempts


def aggregate_daily_attempts(metrics_path, now=None, reset_path=RESET_PATH):
    """Return grouped safe counters since the reset marker for today's UTC date."""
    day = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date()
    reset_time = _read_reset_time(reset_path)
    groups = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0,
                                  "successful_seconds": 0.0, "timed_successes": 0,
                                  "audio_seconds": 0.0})
    try:
        with open(metrics_path, encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(record, dict) or _metric_day(record) != day:
                    continue
                if reset_time is not None:
                    try:
                        metric_time = datetime.fromisoformat(
                            str(record.get("timestamp", "")).replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        continue
                    if metric_time.tzinfo is None or metric_time.astimezone(timezone.utc) < reset_time:
                        continue
                ledger = record.get("attempts")
                if not isinstance(ledger, list):
                    ledger = _legacy_attempts(record)
                for attempt in ledger:
                    _add(groups, attempt, record.get("audio_duration_seconds"))
    except OSError:
        pass
    result = []
    for (stage, provider, model), values in groups.items():
        row = {"stage": stage, "provider": provider, "model": model, **values}
        row["average_seconds"] = (values["successful_seconds"] / values["timed_successes"]
                                  if values["timed_successes"] else None)
        row["seconds_per_audio_minute"] = (
            values["successful_seconds"] / values["audio_seconds"] * 60
            if values["audio_seconds"] else None)
        row["real_time_factor"] = (values["successful_seconds"] / values["audio_seconds"]
                                   if stage == "asr" and values["audio_seconds"] else None)
        result.append(row)
    return sorted(result, key=lambda row: (0 if row["stage"] == "asr" else 1,
                                           row["provider"], row["model"]))


def choose_openrouter_fallback_model(metrics_path, exclude_model=None, now=None, lookback_days=30):
    """Choose a recently successful free model, preferring reliability then speed."""
    now = now or datetime.now(timezone.utc)
    cutoff = now.astimezone(timezone.utc).timestamp() - max(1, int(lookback_days)) * 86400
    excluded_models = (set(exclude_model) if isinstance(exclude_model, (set, list, tuple))
                       else {exclude_model} if isinstance(exclude_model, str) else set())
    models = defaultdict(lambda: {"successes": 0, "failures": 0,
                                  "seconds": 0.0, "audio_seconds": 0.0})
    try:
        with open(metrics_path, encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                    stamp = datetime.fromisoformat(
                        str(record.get("timestamp", "")).replace("Z", "+00:00"))
                except (ValueError, TypeError, AttributeError, OverflowError):
                    continue
                if (not isinstance(record, dict) or stamp.tzinfo is None or
                        stamp.astimezone(timezone.utc).timestamp() < cutoff):
                    continue
                attempts = record.get("attempts")
                if not isinstance(attempts, list):
                    attempts = _legacy_attempts(record)
                for attempt in attempts:
                    if not isinstance(attempt, dict) or attempt.get("stage") != "correction" or \
                            attempt.get("provider") != "openrouter":
                        continue
                    model = attempt.get("model")
                    if (not isinstance(model, str) or not re.fullmatch(
                            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+:free", model) or
                            model in excluded_models):
                        continue
                    stats = models[model]
                    if attempt.get("success") is True:
                        stats["successes"] += 1
                        seconds = _safe_seconds(attempt.get("seconds"))
                        audio = _safe_seconds(record.get("audio_duration_seconds"))
                        if seconds is not None:
                            stats["seconds"] += seconds
                        if audio is not None:
                            stats["audio_seconds"] += audio
                    else:
                        stats["failures"] += 1
    except OSError:
        return None
    candidates = [(model, stats) for model, stats in models.items() if stats["successes"]]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (
        -((item[1]["successes"] + 1) /
          (item[1]["successes"] + item[1]["failures"] + 2)),
        -item[1]["successes"],
        item[1]["seconds"] / item[1]["audio_seconds"] if item[1]["audio_seconds"] else float("inf"),
        item[0]))
    return candidates[0][0]


def cloudflare_correction_model_order(metrics_path, candidate_models, preferred_model=None,
                                      now=None, lookback_days=30):
    """Rank allowlisted Cloudflare correction models from safe outcomes only.

    Models with accepted responses rank by smoothed success rate, then sample
    count, the user's preferred model for ties, and successful latency. Models
    without history are explored before models with failures but no successes.
    """
    models = []
    for model in candidate_models:
        if isinstance(model, str) and model and model not in models:
            models.append(model)
    if not models:
        return []

    now = now or datetime.now(timezone.utc)
    cutoff = now.astimezone(timezone.utc).timestamp() - max(1, int(lookback_days)) * 86400
    stats = {model: {"successes": 0, "failures": 0, "seconds": []} for model in models}
    try:
        with open(metrics_path, encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                    stamp = datetime.fromisoformat(
                        str(record.get("timestamp", "")).replace("Z", "+00:00"))
                except (ValueError, TypeError, AttributeError, OverflowError):
                    continue
                if (not isinstance(record, dict) or stamp.tzinfo is None or
                        stamp.astimezone(timezone.utc).timestamp() < cutoff):
                    continue
                attempts = record.get("attempts")
                if not isinstance(attempts, list):
                    attempts = _legacy_attempts(record)
                for attempt in attempts:
                    if (not isinstance(attempt, dict) or attempt.get("stage") != "correction" or
                            attempt.get("provider") != "cloudflare"):
                        continue
                    model = attempt.get("model")
                    if model not in stats:
                        continue
                    row = stats[model]
                    if attempt.get("success") is True:
                        row["successes"] += 1
                        seconds = _safe_seconds(attempt.get("seconds"))
                        if seconds is not None:
                            row["seconds"].append(seconds)
                    else:
                        row["failures"] += 1
    except OSError:
        pass

    preferred_rank = {model: index for index, model in enumerate(
        [preferred_model] + [model for model in models if model != preferred_model])}
    known_successes = [model for model in models if stats[model]["successes"]]
    known_successes.sort(key=lambda model: (
        -((stats[model]["successes"] + 1) /
          (stats[model]["successes"] + stats[model]["failures"] + 2)),
        -stats[model]["successes"], preferred_rank.get(model, len(models)),
        (sum(stats[model]["seconds"]) / len(stats[model]["seconds"])
         if stats[model]["seconds"] else float("inf")), model))
    unexplored = [model for model in models if not
                  (stats[model]["successes"] or stats[model]["failures"])]
    unexplored.sort(key=lambda model: (preferred_rank.get(model, len(models)), models.index(model)))
    failures_only = [model for model in models if not stats[model]["successes"] and
                     stats[model]["failures"]]
    failures_only.sort(key=lambda model: (stats[model]["failures"],
                                          preferred_rank.get(model, len(models)), model))
    return known_successes + unexplored + failures_only
