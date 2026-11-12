#!/usr/bin/env python3
"""Copy text to the desktop clipboard and persist it in Klipper history."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping


class ClipboardDeliveryError(RuntimeError):
    """No supported clipboard transport accepted the text."""


def _set_klipper_clipboard(text: str) -> None:
    """Use Klipper's D-Bus API so it owns both clipboard and history updates."""
    import gi

    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib

    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    bus.call_sync(
        "org.kde.klipper", "/klipper", "org.kde.klipper.klipper",
        "setClipboardContents", GLib.Variant("(s)", (text,)), None,
        Gio.DBusCallFlags.NONE, 5000, None,
    )
    bus.call_sync(
        "org.kde.klipper", "/klipper", "org.kde.klipper.klipper",
        "saveClipboardHistory", GLib.Variant("()", ()), None,
        Gio.DBusCallFlags.NONE, 5000, None,
    )


def deliver_clipboard_text(
    text: str,
    *,
    klipper_setter: Callable[[str], None] | None = None,
    command_runner: Callable[..., object] | None = None,
    which: Callable[[str], str | None] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Prefer Klipper history, then try Wayland and X11 clipboard owners.

    Text travels over stdin to command line fallbacks, never in process args or
    diagnostics. Return only the selected backend name.
    """
    if not isinstance(text, str) or not text:
        raise ClipboardDeliveryError("empty_text")

    klipper_setter = klipper_setter or _set_klipper_clipboard
    command_runner = command_runner or subprocess.run
    which = which or shutil.which
    env = os.environ if environ is None else environ

    try:
        klipper_setter(text)
        return "klipper"
    except Exception:
        pass

    candidates = []
    if env.get("WAYLAND_DISPLAY") or env.get("XDG_SESSION_TYPE") == "wayland":
        candidates.append(("wl-copy", ["wl-copy"]))
    if env.get("DISPLAY"):
        candidates.append(("xclip", ["xclip", "-selection", "clipboard"]))

    for backend, command in candidates:
        if which(command[0]) is None:
            continue
        try:
            command_runner(
                command, input=text.encode("utf-8"), check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return backend
        except Exception:
            continue
    raise ClipboardDeliveryError("clipboard_backends_unavailable")


def main() -> int:
    text = sys.stdin.read()
    try:
        deliver_clipboard_text(text)
    except ClipboardDeliveryError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
