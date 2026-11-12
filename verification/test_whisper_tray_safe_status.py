"""Synthetic tests for the Codex tray status; never load GTK or user metrics."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
TREE = ast.parse((ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8"))
NODES = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and
         node.name in {"latest_codex_attempt", "format_codex_attempt_status"}]
NAMESPACE = {"json": json, "re": __import__("re")}
exec(compile(ast.Module(body=NODES, type_ignores=[]), "whisper-tray-safe-status", "exec"),
     NAMESPACE)


class CodexTraySafeStatusTests(unittest.TestCase):
    def test_guard_rejection_and_next_provider_are_reported_without_text(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            metrics.write_text(json.dumps({"attempts": [
                {"stage": "correction", "provider": "codex", "success": False,
                 "error_category": "unsafe_response_word_coverage_order"},
                {"stage": "correction", "provider": "openrouter", "success": False,
                 "error_category": "other_error"},
            ], "transcript": "must never be surfaced"}) + "\n", encoding="utf-8")
            attempt = NAMESPACE["latest_codex_attempt"](metrics)
        label = NAMESPACE["format_codex_attempt_status"](attempt)
        self.assertIn("проверка сохранения слов", label)
        self.assertIn("запущен следующий сервис", label)
        self.assertNotIn("must never be surfaced", label)

    def test_failed_retries_report_guard_and_retry_count(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            attempts = [{"stage": "correction", "provider": "codex", "success": False,
                         "error_category": "unsafe_response_tail"} for _ in range(3)]
            metrics.write_text(json.dumps({"attempts": attempts}) + "\n", encoding="utf-8")
            attempt = NAMESPACE["latest_codex_attempt"](metrics)
        label = NAMESPACE["format_codex_attempt_status"](attempt)
        self.assertIn("проверка конца текста", label)
        self.assertIn("3 попыток", label)

    def test_cli_errors_use_safe_generic_messages(self):
        label = NAMESPACE["format_codex_attempt_status"]({
            "success": False, "category": "cli_failed", "attempt_count": 1,
            "continued": True,
        })
        self.assertEqual(label, "Codex: команда Codex завершилась с ошибкой; проверьте следующий сервис в цепочке")


class AppSettingsSourceTests(unittest.TestCase):
    def test_model_and_service_tab_labels_load_visible_icons(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        label_function = functions["settings_tab_label"]
        label_calls = [node for node in ast.walk(label_function) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(call.func, ast.Attribute) and
                            call.func.attr == "set_from_pixbuf" for call in label_calls))
        self.assertTrue(any(isinstance(call.func, ast.Attribute) and
                            call.func.attr == "show_all" for call in label_calls))

        model_function = functions["configure_models_settings"]
        stage_loop = next(node for node in ast.walk(model_function) if isinstance(node, ast.For) and
                          isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2)
        loop_calls = [node for node in ast.walk(stage_loop) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(call.func, ast.Attribute) and
                            call.func.attr == "append_page" and len(call.args) > 1 and
                            isinstance(call.args[1], ast.Call) and
                            isinstance(call.args[1].func, ast.Name) and
                            call.args[1].func.id == "settings_tab_label" for call in loop_calls))

        app_function = functions["configure_app_settings"]
        app_calls = [node for node in ast.walk(app_function) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(call.func, ast.Name) and
                            call.func.id == "settings_tab_label" for call in app_calls))
        self.assertTrue(any(isinstance(call.func, ast.Attribute) and
                            call.func.attr == "set_tab_pos" and len(call.args) == 1 and
                            isinstance(call.args[0], ast.Attribute) and
                            call.args[0].attr == "LEFT" for call in app_calls))
        self.assertTrue(any(isinstance(call.func, ast.Attribute) and
                            call.func.attr == "set_scrollable" and call.args and
                            isinstance(call.args[0], ast.Constant) and
                            call.args[0].value is True for call in app_calls))

    def test_unified_settings_routes_menu_entrypoints_and_reports_open_errors(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        app_function = functions["configure_app_settings"]
        model_function = functions["configure_models_settings"]
        open_function = functions["open_settings_panel"]

        calls = [node for node in ast.walk(app_function) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(call.func, ast.Name) and
                            call.func.id == "configure_models_settings" and
                            any(keyword.arg == "parent" and isinstance(keyword.value, ast.Name) and
                                keyword.value.id == "notebook" for keyword in call.keywords)
                            for call in calls))
        service_tabs = [call.args[0].value for call in calls
                        if isinstance(call.func, ast.Name) and call.func.id == "new_service_page" and
                        call.args and isinstance(call.args[0], ast.Constant)]
        service_tabs.extend(call.args[1].value for call in calls
                            if isinstance(call.func, ast.Name) and call.func.id == "cli_service_page" and
                            len(call.args) > 1 and isinstance(call.args[1], ast.Constant))
        self.assertEqual(service_tabs, ["Cloudflare", "OpenRouter", "Groq", "Gemini", "Qwen", "Codex", "Claude"])

        model_text = ast.get_source_segment(source, model_function)
        self.assertIn('("asr", "Транскрибация")', model_text)
        self.assertIn('("correction", "Корректор")', model_text)

        all_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        panel_connect = next(call for call in all_calls
                             if isinstance(call.func, ast.Attribute) and call.func.attr == "connect" and
                             len(call.args) >= 2 and isinstance(call.args[0], ast.Constant) and
                             call.args[0].value == "activate" and
                             isinstance(call.args[1], ast.Lambda) and
                             any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                                 node.func.id == "open_settings_panel"
                                 for node in ast.walk(call.args[1])))
        self.assertIsNotNone(panel_connect)
        panel_text = ast.get_source_segment(source, open_function)
        self.assertIn("configure_app_settings(initial_tab=tab_name)", panel_text)
        self.assertIn("traceback.format_exc()", panel_text)
        self.assertIn("notify_send(", panel_text)

        for name in ("configure_cloudflare_settings", "configure_correction_provider_settings",
                     "configure_models_settings", "configure_provider_settings"):
            text = ast.get_source_segment(source, functions[name])
            self.assertIn("open_settings_panel(", text, f"{name} must route to the unified panel")

        provider_menus = {"codex_menu_item", "claude_menu_item", "groq_top_menu_item"}
        appended = {call.args[0].id for call in all_calls
                    if isinstance(call.func, ast.Attribute) and call.func.attr == "append" and
                    call.args and isinstance(call.args[0], ast.Name)}
        self.assertTrue(provider_menus.issubset(appended))
        self.assertTrue({"cloudflare_menu_item", "openrouter_top_menu_item"}.isdisjoint(appended))
        self.assertIn("model_settings_item", appended)

    def test_account_usage_is_local_and_reports_asr_and_correction_counts(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                      node.name == "provider_account_usage_text")
        namespace = {
            "METRICS_FILE": "unused",
            "os": os,
            "aggregate_daily_attempts": lambda _path: [
                {"provider": "openrouter", "stage": "asr", "attempts": 2},
                {"provider": "openrouter", "stage": "correction", "attempts": 3},
            ],
            "latest_codex_attempt": lambda _path: None,
            "format_codex_attempt_status": lambda _attempt: "нет данных",
            "claude_cli_login_state": lambda: None,
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "provider-usage", "exec"), namespace)
        with patch.dict("sys.modules", {
                "attempt_statistics": SimpleNamespace(
                    aggregate_daily_attempts=namespace["aggregate_daily_attempts"]),
                "multi_provider": SimpleNamespace(credentials_configured=lambda _provider: False)}):
            text = namespace["provider_account_usage_text"]("openrouter")
        self.assertIn("ASR — 2 обращений", text)
        self.assertIn("коррекция — 3 обращений", text)
        self.assertNotIn("Статус аккаунта:", text)

    def test_openrouter_menu_content_is_grouped_in_unified_service_page(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        app_panel = ast.get_source_segment(source, functions["configure_app_settings"])
        provider_settings = ast.get_source_segment(
            source, functions["configure_correction_provider_settings"])
        self.assertIn('openrouter_page, "Учётные данные и использование"', app_panel)
        self.assertIn('openrouter_page, "Лимиты и статистика OpenRouter"', app_panel)
        self.assertIn('service_settings_group(openrouter_page, "Настройки корректора")', app_panel)
        self.assertIn("openrouter_free_usage_text()", app_panel)
        self.assertIn("openrouter_daily_attempt_summary(METRICS_FILE)", app_panel)
        self.assertIn('https://openrouter.ai/activity', app_panel)
        self.assertIn("show_credential_status=True", provider_settings)
        self.assertNotIn('menu.append(cloudflare_menu_item)', source)
        self.assertNotIn('menu.append(openrouter_top_menu_item)', source)

    def test_cloudflare_usage_block_is_inside_credentials_tab(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        cloudflare = ast.get_source_segment(source, functions["configure_cloudflare_settings"])
        app_panel = ast.get_source_segment(source, functions["configure_app_settings"])
        self.assertIn('Gtk.Label(label=provider_account_usage_text("cloudflare")', cloudflare)
        self.assertIn('usage_body.pack_start(account_usage', cloudflare)
        self.assertIn('notebook.append_page(account_page, settings_tab_label("Учётные данные", "cloudflare"))', cloudflare)
        self.assertNotIn('provider_account_usage_text("cloudflare")', app_panel)

    def test_cloudflare_menu_status_is_present_in_settings_account_tab(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        menu_refresh = ast.get_source_segment(source, functions["refresh_cloudflare_usage"])
        usage_summary = ast.get_source_segment(source, functions["provider_account_usage_text"])
        settings = ast.get_source_segment(source, functions["configure_cloudflare_settings"])

        self.assertIn("cloudflare_daily_attempt_summary(METRICS_FILE)", menu_refresh)
        self.assertIn("cloudflare_daily_attempt_summary(METRICS_FILE)", usage_summary)
        self.assertIn("cloudflare_credentials_available()", settings)
        self.assertIn('Gtk.CheckButton(label="Учётные данные заполнены")', settings)
        self.assertEqual(settings.count('Gtk.CheckButton(label="Учётные данные заполнены")'), 1)
        self.assertIn("format_cloudflare_attempt_status(latest_cloudflare_attempt(METRICS_FILE))", settings)
        self.assertIn('account_section("Последняя попытка Cloudflare")', settings)
        self.assertIn('account_section("Учётные данные")', settings)
        self.assertIn('settings_tab_label("Whisper", "transcribe")', settings)
        self.assertIn('settings_tab_label("Корректор", "corrector")', settings)
        self.assertIn('settings_tab_label("Учётные данные", "cloudflare")', settings)
        self.assertIn('"switch-page"', settings)
        self.assertIn("cloudflare_daily_attempt_summary", source)

    def test_qwen_audio_order_and_actual_asr_notification_are_wired(self):
        tray = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        hotkey = (ROOT / "scripts/whisper-dictate-v2").read_text(encoding="utf-8")
        client = (ROOT / "scripts/whisper-daemon-client.sh").read_text(encoding="utf-8")
        daemon = (ROOT / "scripts/whisper-daemon.py").read_text(encoding="utf-8")
        self.assertIn('new_service_page("Qwen")', tray)
        self.assertIn('notebook.reorder_child(page_widgets["Qwen"], 2)', tray)
        self.assertIn('qwen_button = Gtk.Button(label="Настройки Qwen…")', tray)
        self.assertIn('dialog.add_button("Отмена", Gtk.ResponseType.CANCEL)', tray)
        self.assertIn('save_voice_config(opening_config)', tray)
        self.assertIn('Gtk.Orientation.HORIZONTAL, spacing=10', tray)
        self.assertIn('tab.set_border_width(8)', tray)
        self.assertIn('pixbuf.scale_simple(32, 32, GdkPixbuf.InterpType.BILINEAR)', tray)
        self.assertIn('image.set_size_request(32, 32)', tray)
        self.assertIn('local_model_order.set_margin_start(12)', tray)
        self.assertIn('order_title.set_margin_start(12)', tray)
        self.assertIn('page.pack_start(local_model_order, False, False, 0)', tray)
        self.assertNotIn('chain_row = Gtk.ListBoxRow()', tray)
        self.assertIn('label="Приглушать звук при диктовке"', tray)
        self.assertIn('label="Восстановить звук"', tray)
        self.assertIn("Порядок сервисов для транскрибации", tray)
        self.assertIn("Порядок сервисов коррекции", tray)
        self.assertIn('"asr_backend_runtime":result.get("asr_backend_runtime")', daemon)
        self.assertIn('"asr_model":result.get("asr_model", result.get("model"))', daemon)
        self.assertIn("actual_backend", hotkey)
        self.assertIn('"asr_backend_runtime":result.get("asr_backend_runtime")', client)


if __name__ == "__main__":
    unittest.main()
