import ast
import asyncio
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
DAEMON = ROOT / "scripts" / "whisper-daemon.py"
TRAY = ROOT / "scripts" / "whisper-tray.py"


def daemon_async_function(name, **overrides):
    tree = ast.parse(DAEMON.read_text(encoding="utf-8"))
    node = next(item for item in tree.body
                if isinstance(item, ast.AsyncFunctionDef) and item.name == name)
    namespace = {"asyncio": asyncio, "logger": __import__("logging").getLogger(name)}
    namespace.update(overrides)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DAEMON), "exec"), namespace)
    return namespace[name]


class ModelResidencyControlTests(unittest.TestCase):
    def test_unload_all_waits_for_queued_work_then_releases_asr_and_configured_qwen(self):
        events = []
        loaded = {"asr": True, "qwen": ["qwen-primary", "qwen-fallback"]}

        class Queue:
            async def join(self):
                events.append("queue-drained")

        idle = asyncio.Event()
        idle.set()
        async def inline_to_thread(function, *args):
            return function(*args)
        async def release():
            events.append("asr-released")
            loaded["asr"] = False
        def unload(name):
            events.append("unload-" + name)
            loaded["qwen"].remove(name)
            return True
        def status():
            return {"status": "ok", "asr_loaded": loaded["asr"],
                    "asr_backend": "faster-whisper" if loaded["asr"] else None,
                    "corrector_models": list(loaded["qwen"])}

        control = daemon_async_function(
            "control_models", control_lock=asyncio.Lock(), queue=Queue(),
            correction_idle=idle, release=release,
            model_residency_status=status, unload_corrector_model=unload,
            config=lambda: {}, model_lease=0, qwen_lease=0,
            asyncio=type("AsyncioProxy", (), {
                "Lock": asyncio.Lock, "to_thread": staticmethod(inline_to_thread),
            }))
        result = asyncio.run(asyncio.wait_for(
            control({"action": "unload_models", "scope": "all"}), timeout=2))
        self.assertEqual(result["asr_loaded"], False)
        self.assertEqual(result["corrector_models"], [])
        self.assertEqual(events, ["queue-drained", "unload-qwen-primary",
                                  "unload-qwen-fallback", "asr-released"])

    def test_asr_unload_guard_leaves_a_different_active_backend_untouched(self):
        events = []
        async def release():
            events.append("released")
        def status():
            return {"status": "ok", "asr_loaded": True,
                    "asr_backend": "gigaam-v3-ctc-onnx-gpu", "corrector_models": []}
        class Queue:
            async def join(self):
                pass
        idle = asyncio.Event()
        idle.set()
        control = daemon_async_function(
            "control_models", control_lock=asyncio.Lock(), queue=Queue(),
            correction_idle=idle, release=release,
            model_residency_status=status, unload_corrector_model=lambda _name: True,
            config=lambda: {}, model_lease=0, qwen_lease=0)
        result = asyncio.run(control({"action": "unload_models", "scope": "asr",
                                      "expected_backend": "faster-whisper"}))
        self.assertTrue(result["unchanged"])
        self.assertEqual(events, [])

    def test_selected_asr_is_prepared_only_after_existing_work_drains(self):
        events = []
        class Queue:
            async def join(self):
                events.append("queue-drained")
        idle = asyncio.Event()
        idle.set()
        async def prepare(profile):
            events.append(("prepare", profile["asr_backend"]))
        control = daemon_async_function(
            "control_models", control_lock=asyncio.Lock(), queue=Queue(),
            correction_idle=idle, release=lambda: None,
            model_residency_status=lambda: {"status": "ok", "asr_loaded": True},
            unload_corrector_model=lambda _name: True,
            prepare_local_asr=prepare,
            config=lambda: {"asr_backend": "gigaam-v3-e2e-ctc-onnx-gpu"},
            model_lease=0, qwen_lease=0)
        result = asyncio.run(control({"action": "prepare_selected_asr"}))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(events, ["queue-drained",
                                  ("prepare", "gigaam-v3-e2e-ctc-onnx-gpu")])

    def test_tray_exposes_unified_and_backend_specific_unload_controls(self):
        source = TRAY.read_text(encoding="utf-8")
        self.assertIn('primary_menu_item("Выгрузить все модели"', source)
        self.assertIn('Gtk.Button(label="Выгрузить Whisper")', source)
        self.assertIn('Gtk.Button(label="Выгрузить GigaAM")', source)
        self.assertIn('scope="corrector"', source)
        self.assertIn('action == "prepare_selected_asr"', source)


if __name__ == "__main__":
    unittest.main()
