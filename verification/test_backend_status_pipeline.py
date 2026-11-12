import ast
import json
import os
import tempfile
import unittest
import urllib.request
from datetime import datetime
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
        "GIGAAM_BACKENDS": {"gigaam-v3-e2e-rnnt", "gigaam-v3-e2e-ctc-onnx-gpu",
                           "gigaam-v3-ctc-onnx-gpu"},
        "json": json,
        "os": os,
    }
    exec(compile(ast.Module(body=[updater], type_ignores=[]), "daemon-job-updater", "exec"), namespace)
    return namespace["update_job_backend"]


def load_local_asr_status_profile():
    source = (ROOT / "scripts/whisper-daemon.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "local_asr_status_profile")
    namespace = {
        "GIGAAM_MODEL_NAME": "GigaAM-v3 e2e_rnnt",
        "GIGAAM_ONNX_BACKENDS": {
            "gigaam-v3-e2e-ctc-onnx-gpu": {
                "model": "GigaAM-v3 e2e_ctc", "device": "cuda"},
            "gigaam-v3-ctc-onnx-gpu": {"model": "GigaAM-v3 ctc", "device": "cuda"},
        },
    }
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "local-asr-status-profile", "exec"), namespace)
    return namespace["local_asr_status_profile"]


class BackendStatusPipelineTests(unittest.TestCase):
    def test_menu_provider_and_action_icons_are_large_transparent_pngs(self):
        from PIL import Image
        icon_dir = ROOT / "assets/menu-icons"
        for name in ("cloudflare", "openrouter", "codex", "claude", "groq",
                     "google", "clear", "copy", "daemon", "stop", "unload",
                     "performance", "autoreplace", "gigaam"):
            with Image.open(icon_dir / f"{name}.png") as image:
                self.assertEqual(image.size, (96, 96), name)
                self.assertEqual(image.mode, "RGBA", name)
                self.assertEqual(image.getpixel((0, 0))[3], 0, name)

    def test_local_whisper_and_corrector_menu_icons_are_distinct(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        self.assertIn('primary_menu_item("Транскрибатор локальный", "audio-input-microphone")', source)
        self.assertIn('primary_menu_item("Корректор локальный", "accessories-text-editor-symbolic")', source)
        self.assertIn('corrector_menu_item = primary_menu_item("Корректор локальный", "accessories-text-editor-symbolic")', source)
        self.assertIn('def primary_menu_item(label, icon_name):', source)
        self.assertIn('item.set_always_show_image(True)', source)

    def test_processing_status_names_model_stage_and_backend(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "processing_status_label")
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "tray-status-label", "exec"), namespace)
        label = namespace["processing_status_label"]
        self.assertEqual(label("@cf/qwen/test", "correction", "processing_cloudflare"),
                         "@cf/qwen/test — коррекция · Cloudflare")
        self.assertEqual(label("qwen-local", "correction", "processing_gpu"),
                         "qwen-local — коррекция · локальный GPU")
        self.assertEqual(label("qwen-local", "correction", "processing_cpu"),
                         "qwen-local — коррекция · локальный CPU")

    def test_gigaam_fallback_status_shows_selected_cpu_backend_and_reason(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "asr_backend_status_suffix")
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "gigaam-status", "exec"), namespace)
        status = namespace["asr_backend_status_suffix"](
            "gigaam-v3-e2e-rnnt", "large-v3", "worker_unavailable")
        self.assertIn("GigaAM → FasterWhisper fallback", status)
        self.assertIn("FasterWhisper fallback", status)
        self.assertIn("GigaAM worker недоступен", status)
        self.assertEqual(namespace["asr_backend_status_suffix"](
            "gigaam-v3-e2e-rnnt", "GigaAM-v3 e2e_rnnt", None), "")
        self.assertIn("FasterWhisper fallback", namespace["asr_backend_status_suffix"](
            "gigaam-v3-e2e-ctc-onnx-gpu", "large-v3", "cuda_execution_unavailable"))

    def test_gigaam_status_is_correct_before_model_load(self):
        profile = load_local_asr_status_profile()
        self.assertEqual(profile({"asr_backend": "gigaam-v3-e2e-rnnt",
                                  "model": "large-v3", "device": "cuda"}),
                         {"asr_backend": "gigaam-v3-e2e-rnnt",
                          "model": "GigaAM-v3 e2e_rnnt", "device": "cpu"})
        self.assertEqual(profile({"asr_backend": "faster-whisper",
                                  "model": "large-v3", "device": "cuda"}),
                         {"asr_backend": "faster-whisper",
                          "model": "large-v3", "device": "cuda"})
        self.assertEqual(profile({"asr_backend": "gigaam-v3-e2e-ctc-onnx-gpu"}),
                         {"asr_backend": "gigaam-v3-e2e-ctc-onnx-gpu",
                          "model": "GigaAM-v3 e2e_ctc", "device": "cuda"})

    def test_main_menu_has_no_duplicated_last_task_status(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        menu = source[source.index("menu = Gtk.Menu()"):source.index("whisper_settings_submenu = Gtk.Menu()")]
        labels = [line for line in menu.splitlines() if "menu.append(" in line]
        self.assertEqual(len(labels), 7)  # four actions, two settings entries and separator
        self.assertTrue(all("last_task_status" not in line for line in labels))
        self.assertIn('model_settings_item = primary_menu_item("Настройки моделей", "preferences-system-symbolic")', source)
        self.assertIn("SeparatorMenuItem", labels[6])
        self.assertIn('quick_transcribe_item = primary_menu_item("Новая транскрибация", "media-playback-start")', source)
        self.assertIn('transcribe_file_item = primary_menu_item("Распознать файлы…", "document-open")', source)
        self.assertIn('youtube_transcribe_item = primary_menu_item("Распознать видео…", "video-x-generic")', source)
        self.assertIn('recover_raw_asr_item = primary_menu_item("Скопировать исходник", "edit-copy")', source)
        self.assertIn('quit_item = Gtk.MenuItem(label="Выход")', source)

    def test_cloudflare_status_explains_rate_limit_and_local_fallback(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {"cloudflare_error_reason", "format_cloudflare_attempt_status",
                 "format_cloudflare_attempt_lines", "format_cloudflare_usage_label",
                 "format_cloudflare_usage_lines"}
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"datetime": datetime}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "cloudflare-menu-status", "exec"), namespace)
        status = namespace["format_cloudflare_attempt_status"]({
            "timestamp": "2026-10-23T12:34:00+00:00",
            "asr_backend": "local", "asr_device": "cuda",
            "cloudflare_asr_error_category": "http_429",
            "corrector_backend": "cloudflare",
        })
        self.assertIn("лимит/частота запросов (429)", status)
        self.assertIn("fallback: локальный GPU", status)
        self.assertIn("Корректор: Cloudflare — успешно", status)

    def test_cloudflare_status_maps_credentials_and_network_failures(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"cloudflare_error_reason", "format_cloudflare_attempt_status",
                                     "format_cloudflare_attempt_lines"}]
        namespace = {"datetime": datetime}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "cloudflare-error-map", "exec"), namespace)
        explain = namespace["cloudflare_error_reason"]
        self.assertIn("токен и права", explain("http_401_403"))
        self.assertIn("сетевая ошибка", explain("network_error"))
        self.assertIn("ещё", namespace["format_cloudflare_attempt_status"]({}))

    def test_cloudflare_attempt_reader_skips_local_jobs_and_redacts_metrics(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "latest_cloudflare_attempt")
        namespace = {"json": json}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "latest-cloudflare-event", "exec"), namespace)
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = Path(temp_dir) / "metrics.jsonl"
            metrics.write_text("\n".join(json.dumps(row) for row in (
                {"cloudflare_attempted": False, "timestamp": "local-only"},
                {"cloudflare_attempted": True, "timestamp": "2026-10-23T12:00:00+00:00",
                 "asr_backend": "local", "asr_device": "cuda",
                 "cloudflare_asr_error_category": "http_429", "transcript": "private text"},
            )), encoding="utf-8")
            record = namespace["latest_cloudflare_attempt"](metrics)
        self.assertEqual(record["cloudflare_asr_error_category"], "http_429")
        self.assertNotIn("transcript", record)

    def test_cloudflare_usage_is_explicitly_an_estimate(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"format_cloudflare_usage_label", "format_cloudflare_usage_lines"}]
        namespace = {}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "cloudflare-usage-label", "exec"), namespace)
        label = namespace["format_cloudflare_usage_label"]({
            "estimated_neurons": 200, "corrector_estimated_neurons": 50,
            "corrector_estimated_token_requests": 1, "audio_seconds": 120, "requests": 1,
        }, 10_000)
        self.assertIn("Расчётный остаток: ~9 750", label)
        self.assertIn("локальная оценка и preflight-защита", label)
        self.assertIn("Расчётный остаток", label)

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

    def test_dictation_model_menu_can_select_gigaam_and_fasterwhisper(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        self.assertIn('"gigaam-v3-e2e-ctc-onnx-gpu": ("GigaAM-v3 e2e_ctc · GPU"', source)
        self.assertIn('"gigaam-v3-ctc-onnx-gpu": ("GigaAM-v3 ctc · GPU"', source)
        self.assertIn('"gigaam-v3-e2e-rnnt": ("GigaAM-v3 e2e_rnnt · CPU"', source)
        self.assertIn('voice_config["asr_backend"] = backend_name', source)
        self.assertIn('model_items[backend_name] = gigaam_item', source)
        self.assertIn('voice_config["asr_backend"] = "faster-whisper"', source)
        self.assertLess(source.index("model_submenu.append(gigaam_item)"),
                        source.index("model_submenu.append(item)"))
        self.assertIn('"Модель диктовки — " + GIGAAM_MODEL_CHOICES[selected_backend][0]', source)
        self.assertIn('primary_menu_item("Транскрибатор локальный", "audio-input-microphone")', source)

    def test_menu_icon_rows_use_fixed_slot_and_single_line_labels(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        self.assertIn("def style_menu_row(item, image):", source)
        self.assertIn("image.set_size_request(24, 24)", source)
        self.assertIn("image.set_valign(Gtk.Align.CENTER)", source)
        self.assertIn("label.set_line_wrap(False)", source)
        self.assertIn("label.set_single_line_mode(True)", source)
        self.assertIn("label.set_ellipsize(Pango.EllipsizeMode.END)", source)
        self.assertIn("GdkPixbuf.Pixbuf.new_from_file(source)", source)
        self.assertIn("pixbuf.scale_simple(24, 24, GdkPixbuf.InterpType.BILINEAR)", source)


if __name__ == "__main__":
    unittest.main()
