import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).parents[1]


def load_daemon_helpers(raw_dir, cancel_dir):
    source_path = ROOT / "scripts/whisper-daemon.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {"job_raw_path", "correction_cancelled", "save_live_raw_asr"}
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"RAW_ASR": Path(raw_dir), "RECOVERY_CANCEL": Path(cancel_dir),
                 "Path": Path, "os": os, "logger": Mock()}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


def load_tray_recovery_handler(state_dir):
    source_path = ROOT / "scripts/whisper-tray.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    handler = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "on_recover_raw_asr_clicked")
    jobs_dir = Path(state_dir) / "jobs"
    namespace = {"STATE_DIR": str(state_dir), "JOBS_DIR": str(jobs_dir),
                 "glob": __import__("glob"), "os": os, "json": json,
                 "notify_send": Mock(), "show_error_dialog": Mock(),
                 "copy_text_to_clipboard": Mock(return_value=True), "log_debug": Mock()}
    exec(compile(ast.Module(body=[handler], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


class RawAsrRecoveryTests(unittest.TestCase):
    def test_raw_asr_is_saved_atomically_only_for_live_jobs_with_private_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir, cancel_dir = root / "raw-asr", root / "recovery-cancel"
            raw_dir.mkdir(mode=0o700)
            cancel_dir.mkdir(mode=0o700)
            helpers = load_daemon_helpers(raw_dir, cancel_dir)

            self.assertTrue(helpers["save_live_raw_asr"]("12345", "live_dictation", "Исходная фраза"))
            raw_file = raw_dir / "12345.txt"
            self.assertEqual(raw_file.read_text(encoding="utf-8"), "Исходная фраза")
            self.assertEqual(raw_file.stat().st_mode & 0o777, 0o600)
            self.assertFalse(helpers["save_live_raw_asr"]("12346", "file", "не сохранять"))
            self.assertFalse((raw_dir / "12346.txt").exists())

    def test_recovery_copies_raw_and_sets_one_idempotent_cancel_marker_without_clearing_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs = root / "jobs"
            raw_dir = root / "raw-asr"
            jobs.mkdir()
            raw_dir.mkdir()
            job_file = jobs / "12345.json"
            job_file.write_text(json.dumps({"state": "processing", "pipeline_stage": "correction"}),
                                encoding="utf-8")
            raw_file = raw_dir / "12345.txt"
            raw_file.write_text("raw words", encoding="utf-8")
            handler = load_tray_recovery_handler(root)

            handler["on_recover_raw_asr_clicked"](None)
            handler["on_recover_raw_asr_clicked"](None)

            clipboard = handler["copy_text_to_clipboard"]
            self.assertEqual(clipboard.call_count, 2)
            self.assertEqual([call.args[0] for call in clipboard.call_args_list], ["raw words", "raw words"])
            marker = root / "recovery-cancel" / "12345.cancel"
            self.assertTrue(marker.is_file())
            self.assertEqual(len(list((root / "recovery-cancel").glob("*.cancel"))), 1)
            self.assertTrue(job_file.exists())
            self.assertTrue(raw_file.exists())

    def test_openrouter_cancel_marker_returns_raw_and_ignores_late_candidate(self):
        from test_multi_provider import MultiProviderPolicyTests

        tests = MultiProviderPolicyTests()
        cancelled = {"value": False}

        def late_openrouter(*_args, **_kwargs):
            cancelled["value"] = True
            return "late corrected text", "vendor/model:free"

        correct, _namespace = tests.load_correct_text({
            "openrouter_correct": late_openrouter,
            "daily_only_routes": lambda routes: list(routes),
            "choose_openrouter_fallback_model": lambda *_args: "vendor/second:free",
        })
        result = correct("raw words as recognized", {"enabled": False, "model": "qwen-test"},
            provider_settings={"correction": {"openrouter_enabled": True}},
            source="live_dictation", cancel_callback=lambda: cancelled["value"])

        self.assertEqual(result[0], "raw words as recognized")
        self.assertEqual(result[3], "cancelled")
        self.assertEqual(result[1], "cancelled_by_user")

    def test_normal_openrouter_success_is_unchanged_without_cancel_request(self):
        from test_multi_provider import MultiProviderPolicyTests

        tests = MultiProviderPolicyTests()
        correct, _namespace = tests.load_correct_text({
            "openrouter_correct": lambda *_args, **_kwargs: ("corrected text", "vendor/model:free"),
            "daily_only_routes": lambda routes: list(routes),
        })
        result = correct("raw words", {"enabled": False, "model": "qwen-test"},
            provider_settings={"correction": {"openrouter_enabled": True}},
            source="live_dictation", cancel_callback=lambda: False)
        self.assertEqual(result[0], "corrected text")
        self.assertEqual(result[3], "openrouter")

    def test_raw_asr_is_persisted_before_correction_starts(self):
        daemon = (ROOT / "scripts/whisper-daemon.py").read_text(encoding="utf-8")
        save_at = daemon.index("save_live_raw_asr(job.get(\"job_id\")")
        correct_at = daemon.index("correct_text_sync, text, corrector", save_at)
        self.assertLess(save_at, correct_at)

    def test_menu_places_recovery_after_cleanup_and_separator_before_cleanup(self):
        tray = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        separator = tray.index("menu.append(Gtk.SeparatorMenuItem())", tray.index("youtube_transcribe_item.connect"))
        cleanup = tray.index('cleanup_tasks_item = primary_menu_item("Очистить зависшие задачи", "edit-clear")')
        recovery = tray.index('recover_raw_asr_item = primary_menu_item("Скопировать исходник", "edit-copy")')
        self.assertLess(separator, cleanup)
        self.assertLess(cleanup, recovery)


if __name__ == "__main__":
    unittest.main()
