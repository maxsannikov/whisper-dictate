import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "clipboard_delivery", ROOT / "scripts/clipboard_delivery.py")
clipboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clipboard)


class ClipboardDeliveryTests(unittest.TestCase):
    def test_prefers_klipper_for_clipboard_and_persistent_history(self):
        received = []
        backend = clipboard.deliver_clipboard_text(
            "private test transcript", klipper_setter=received.append,
            command_runner=lambda *_args, **_kwargs: self.fail("unexpected fallback"),
            which=lambda _name: None, environ={"WAYLAND_DISPLAY": "wayland-0"})
        self.assertEqual(backend, "klipper")
        self.assertEqual(received, ["private test transcript"])

    def test_wayland_failure_falls_back_to_xclip_without_exposing_text_in_args(self):
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs["input"]))
            if command[0] == "wl-copy":
                raise OSError("clipboard unavailable")
            return object()

        backend = clipboard.deliver_clipboard_text(
            "another private transcript",
            klipper_setter=lambda _text: (_ for _ in ()).throw(RuntimeError()),
            command_runner=runner,
            which=lambda name: f"/usr/bin/{name}",
            environ={"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":1"})
        self.assertEqual(backend, "xclip")
        self.assertEqual([call[0][0] for call in calls], ["wl-copy", "xclip"])
        self.assertEqual(calls[0][1], "another private transcript".encode())
        self.assertEqual(calls[1][1], "another private transcript".encode())
        self.assertNotIn("another private transcript", repr(calls[0][0]))

    def test_reports_safe_failure_when_all_backends_are_unavailable(self):
        with self.assertRaises(clipboard.ClipboardDeliveryError) as caught:
            clipboard.deliver_clipboard_text(
                "private transcript",
                klipper_setter=lambda _text: (_ for _ in ()).throw(RuntimeError()),
                command_runner=lambda *_args, **_kwargs: None,
                which=lambda _name: None, environ={"WAYLAND_DISPLAY": "wayland-0"})
        self.assertEqual(str(caught.exception), "clipboard_backends_unavailable")


if __name__ == "__main__":
    unittest.main()
