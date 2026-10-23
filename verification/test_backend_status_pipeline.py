import ast
import json
import os
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]


def load_job_updater(jobs_dir):
    source = (ROOT / "scripts/whisper-daemon.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    updater = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "update_job_backend")
    namespace = {
        "JOBS": Path(jobs_dir),
        "DEFAULT_CLOUDFLARE_CORRECTOR": "@cf/qwen/default",
        "json": json,
        "os": os,
    }
    exec(compile(ast.Module(body=[updater], type_ignores=[]), "daemon-job-updater", "exec"), namespace)
    return namespace["update_job_backend"]


class BackendStatusPipelineTests(unittest.TestCase):
    def test_local_whisper_and_corrector_menu_icons_are_distinct(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        self.assertIn('Gtk.MenuItem(label="🎙️ Whisper локальный")', source)
        self.assertIn('Gtk.MenuItem(label="✍️ Корректор локальный")', source)
        self.assertIn('corrector_menu_item.set_label(f"✍️ Корректор локальный:', source)

    def test_processing_status_names_model_stage_and_backend(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "processing_status_label")
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "tray-status-label", "exec"), namespace)
        label = namespace["processing_status_label"]
        self.assertEqual(label("@cf/qwen/test", "correction", "processing_cloudflare"),
                         "☁️ @cf/qwen/test — коррекция · Cloudflare")
        self.assertEqual(label("qwen-local", "correction", "processing_gpu"),
                         "🎮 qwen-local — коррекция · локальный GPU")
        self.assertEqual(label("qwen-local", "correction", "processing_cpu"),
                         "🔥 qwen-local — коррекция · локальный CPU")

    def test_completed_status_reports_cloud_local_and_disabled_stages(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "format_completed_backend_status")
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "completed-status", "exec"), namespace)
        format_status = namespace["format_completed_backend_status"]
        mixed = format_status({
            "asr_backend": "local", "asr_model": "large-v3-turbo", "asr_device": "cuda",
            "corrector_backend": "cloudflare", "corrector_model": "@cf/qwen/test",
            "corrector_device": "cloudflare",
        })
        self.assertIn("large-v3-turbo — 🎮 локальный GPU", mixed)
        self.assertIn("@cf/qwen/test — ☁️ Cloudflare", mixed)
        self.assertIn("🔥 локальный CPU", format_status({
            "asr_backend": "cloudflare", "asr_model": "@cf/whisper", "asr_device": "cloudflare",
            "corrector_backend": "local", "corrector_model": "qwen", "corrector_device": "cpu",
        }))
        self.assertIn("Корректор: отключён", format_status({
            "asr_backend": "local", "asr_model": "small", "asr_device": "cuda",
            "corrector_backend": "local_disabled",
        }))

    def test_last_task_status_is_fifth_main_menu_item(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        menu = source[source.index("menu = Gtk.Menu()"):source.index("whisper_settings_submenu = Gtk.Menu()")]
        labels = [line for line in menu.splitlines() if "menu.append(" in line]
        self.assertEqual(len(labels), 6)  # four actions, status, separator
        self.assertIn("menu.append(last_task_status_item)", labels[4])
        self.assertIn("SeparatorMenuItem", labels[5])

    def test_loaded_qwen_device_uses_ollama_vram_metadata(self):
        source = (ROOT / "scripts/whisper-daemon.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "ollama_loaded_device")
        namespace = {"urllib": urllib, "json": json}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "ollama-device", "exec"), namespace)
        choose = namespace["ollama_loaded_device"]

        class Response:
            def __init__(self, payload):
                self.payload = json.dumps(payload).encode()
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def read(self): return self.payload

        with patch("urllib.request.urlopen", return_value=Response({"models": [
                {"name": "qwen-test", "size_vram": 4096}]})):
            self.assertEqual(choose("qwen-test", "cpu"), "cuda")
        with patch("urllib.request.urlopen", return_value=Response({"models": [
                {"name": "qwen-test", "size_vram": 0}]})):
            self.assertEqual(choose("qwen-test", "cuda"), "cpu")
        with patch("urllib.request.urlopen", return_value=Response({"models": []})):
            self.assertEqual(choose("qwen-test", "cuda"), "cuda")

    def test_correction_job_metadata_tracks_local_qwen_device(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_dir = Path(temp_dir)
            job_file = jobs_dir / "123.json"
            job_file.write_text('{"state":"processing","device":"cuda"}', encoding="utf-8")
            update = load_job_updater(jobs_dir)
            for num_gpu, expected_device in ((8, "cuda"), (0, "cpu")):
                update("123", "local", {
                    "device": "cuda",
                    "model": "large-v3",
                    "corrector": {"model": "qwen-test", "num_gpu": num_gpu},
                }, stage="correction")
                row = json.loads(job_file.read_text(encoding="utf-8"))
                self.assertEqual(row["pipeline_stage"], "correction")
                self.assertEqual(row["device"], expected_device)
                self.assertEqual(row["model"], "qwen-test")

    def test_cloudflare_correction_metadata_uses_cloud_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_dir = Path(temp_dir)
            job_file = jobs_dir / "123.json"
            job_file.write_text('{"state":"processing","device":"cuda"}', encoding="utf-8")
            update = load_job_updater(jobs_dir)
            update("123", "cloudflare", {
                "device": "cuda", "model": "large-v3",
                "corrector": {"model": "qwen-local", "num_gpu": 8},
            }, cloud_model="@cf/qwen/test", stage="correction", device="cpu")
            row = json.loads(job_file.read_text(encoding="utf-8"))
            self.assertEqual(row["pipeline_stage"], "correction")
            self.assertEqual(row["device"], "cloudflare")
            self.assertEqual(row["model"], "@cf/qwen/test")

    def test_asr_fallback_metadata_tracks_cpu_retry_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_dir = Path(temp_dir)
            job_file = jobs_dir / "123.json"
            job_file.write_text('{"state":"processing","device":"cuda"}', encoding="utf-8")
            update = load_job_updater(jobs_dir)
            update("123", "local", {
                "device": "cpu", "model": "medium", "corrector": {"model": "qwen-test"},
            }, stage="asr")
            row = json.loads(job_file.read_text(encoding="utf-8"))
            self.assertEqual(row["device"], "cpu")
            self.assertEqual(row["model"], "medium")

    def test_status_refresh_signature_includes_backend_and_pipeline_stage(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        signature = source[source.index("signature = ("):source.index("if signature == _last_signature")]
        for field in ('"backend"', '"pipeline_stage"', '"device"'):
            self.assertIn(field, signature)


if __name__ == "__main__":
    unittest.main()
