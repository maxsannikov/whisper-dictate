#!/usr/bin/env python3
"""Persist provider-specific daily circuit breakers until the next UTC midnight."""
import fcntl
import json
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import tempfile

STATE_PATH = Path.home() / ".cache/whisper-dictate/daily-provider-limit.json"
ALLOWED_PROVIDERS = {"cloudflare", "openrouter"}


def _today(now=None):
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).date()


def _reset_at(day):
    return datetime.combine(day + timedelta(days=1), time.min, tzinfo=timezone.utc)


def _blocked_providers(path, day):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(value, dict) or value.get("utc_date") != day.isoformat():
        return {}
    raw = value.get("providers")
    if isinstance(raw, dict):
        return {provider: "daily_quota_exhausted" for provider, category in raw.items()
                if provider in ALLOWED_PROVIDERS and category == "daily_quota_exhausted"}
    # Read the earlier single-provider format until it expires at UTC midnight.
    if (value.get("provider") in ALLOWED_PROVIDERS and
            value.get("category") == "daily_quota_exhausted"):
        return {value["provider"]: "daily_quota_exhausted"}
    return {}


def daily_limit_state(path=STATE_PATH, now=None, provider=None):
    """Return one provider's current block, or a deterministic block for status UI."""
    today = _today(now)
    blocked = _blocked_providers(path, today)
    if provider is not None:
        if provider not in blocked:
            return None
        selected = provider
    else:
        if not blocked:
            return None
        selected = sorted(blocked)[0]
    return {"provider": selected, "category": "daily_quota_exhausted",
            "until_utc": _reset_at(today).isoformat(),
            "blocked_providers": sorted(blocked)}


def daily_only_routes(routes, path=STATE_PATH, now=None):
    """Remove only routes whose provider is blocked; preserve other providers and local."""
    blocked = _blocked_providers(path, _today(now))
    route_provider = {"cloudflare": "cloudflare", "openrouter": "openrouter"}
    return [route for route in routes if route_provider.get(route) not in blocked]


def mark_daily_limit(provider, category, path=STATE_PATH, now=None):
    """Atomically latch one provider off until the next UTC midnight."""
    if provider not in ALLOWED_PROVIDERS or category != "daily_quota_exhausted":
        return None
    today = _today(now)
    target = Path(path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        blocked = _blocked_providers(target, today)
        blocked[provider] = category
        fd, temp_name = tempfile.mkstemp(prefix=".daily-limit-", dir=target.parent, text=True)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"utc_date": today.isoformat(), "providers": blocked},
                          stream, separators=(",", ":"))
                stream.write("\n")
            os.replace(temp_name, target)
            os.chmod(target, 0o600)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
    finally:
        os.close(lock_fd)
    return daily_limit_state(target, now, provider)
