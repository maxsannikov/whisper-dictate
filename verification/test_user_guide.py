import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class UserGuideWiringTests(unittest.TestCase):
    def test_other_providers_have_clear_top_level_menu_label_and_key_paths(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        guide = (ROOT / "docs/guide.md").read_text(encoding="utf-8")
        label = 'primary_menu_item("Другие сервисы", "network-server-symbolic")'
        self.assertIn(label, source)
        self.assertLess(source.index("menu.append(cloudflare_menu_item)"), source.index("menu.append(groq_top_menu_item)"))
        self.assertLess(source.index("menu.append(openrouter_top_menu_item)"), source.index("menu.append(groq_top_menu_item)"))
        item = source.index("multi_provider_menu_item = primary_menu_item")
        self.assertGreater(source.index("menu.append(multi_provider_menu_item)"), item)
        for filename in ("groq.key", "google.key", "openrouter.key"):
            self.assertIn(filename, guide)
        self.assertIn("Задать ключ", guide)
        provider_loop = source.index('for provider, provider_label, icon_key in')
        openrouter_menu = source[provider_loop:
                                 source.index('service_menu.connect("show"', provider_loop)]
        self.assertIn('if provider != "openrouter":', openrouter_menu)
        self.assertIn("configure_provider_settings", openrouter_menu)

    def test_each_provider_has_persistent_settings_window(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        guide = (ROOT / "docs/guide.md").read_text(encoding="utf-8")
        self.assertIn("def configure_provider_settings(_item, provider):", source)
        self.assertIn('Gtk.CheckButton(label=label)', source)
        for provider in ("groq", "google", "openrouter"):
            self.assertIn(f'("{provider}",', source)
        self.assertIn("В окне настроек используйте", guide)

    def test_provider_order_has_single_authoritative_priority_setting(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        self.assertNotIn("Позиция Cloudflare correction", source)
        self.assertIn("_move_model_provider", source)
        guide = (ROOT / "docs/guide.md").read_text(encoding="utf-8")
        self.assertIn("Начальный приоритет live correction", guide)

    def test_provider_status_icons_are_installed_and_backend_specific(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        installer = (ROOT / "install.sh").read_text(encoding="utf-8")
        for name in ("cloudflare", "groq", "google", "openrouter"):
            self.assertTrue((ROOT / f"assets/status-icons/{name}.svg").is_file())
        self.assertIn('STATUS_ICON_DIR="$HOME/.local/share/whisper-dictate/icons"', installer)
        self.assertIn('provider_status_icon("groq"', source)
        self.assertIn('provider_status_icon("google"', source)
        self.assertIn('provider_status_icon("openrouter"', source)

    def test_profile_commands_match_wrappers(self):
        guide = (ROOT / "docs/guide.md").read_text(encoding="utf-8")
        expected = {
            "fast": ("small", "whisper-dictate-fast-v2"),
            "medium": ("medium", "whisper-dictate-medium-v2"),
            "full": ("large-v3", "whisper-dictate-full-v2"),
        }
        for profile, (model, wrapper) in expected.items():
            source = (ROOT / "scripts" / wrapper).read_text(encoding="utf-8")
            self.assertIn(f"WHISPER_MODEL={model}", source)
            self.assertIn(wrapper, guide)
        self.assertIn("--file", guide)
        self.assertIn("--batch-id", guide)
        self.assertIn("youtube", guide)

    def test_menu_opens_installed_doc_via_argument_list(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        item = source.index('guide_item = primary_menu_item("Горячие клавиши и установка"')
        section = source[source.index("def open_user_guide") : item + 900]
        self.assertIn('subprocess.Popen([editor, guide_path])', section)
        self.assertIn('subprocess.Popen(["xdg-open", guide_path])', section)
        self.assertNotIn("shell=True", section)

    def test_guide_distinguishes_user_daemon_and_graphical_tray(self):
        guide = (ROOT / "docs/guide.md").read_text(encoding="utf-8")
        daemon = (ROOT / "systemd/whisper-daemon.service").read_text(encoding="utf-8")
        tray = (ROOT / "systemd/whisper-tray.service").read_text(encoding="utf-8")
        self.assertIn("WantedBy=default.target", daemon)
        self.assertIn("WantedBy=graphical-session.target", tray)
        self.assertIn("графический сеанс", guide)

    def test_verified_github_url_is_wired_to_menu_safely(self):
        guide = (ROOT / "docs/guide.md").read_text(encoding="utf-8")
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        url = "https://github.com/maxsannikov/whisper-dictate"
        self.assertIn(f"]({url})", guide)
        self.assertIn(f'github_url = "{url}"', source)
        github_item = source.index('github_item = primary_menu_item("GitHub проекта"')
        handler = source[source.index("def open_project_github") : github_item]
        self.assertIn('subprocess.Popen(["xdg-open", github_url])', handler)
        self.assertNotIn("shell=True", handler)

    def test_help_menu_contains_statistics_and_requested_items(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        block = source[source.index("help_submenu = Gtk.Menu()"):
                       source.index("help_submenu.show_all()")]
        self.assertEqual(block.count("help_submenu.append("), 6)
        self.assertIn("help_submenu.append(attempt_statistics_item)", block)
        self.assertIn("help_submenu.append(history_item)", block)
        self.assertIn("help_submenu.append(logs_menu_item)", block)
        self.assertIn("help_submenu.append(guide_item)", block)
        self.assertIn("help_submenu.append(github_item)", block)
        self.assertIn('help_menu_item = primary_menu_item("Помощь", "help-browser")', source)
        top_level = source[source.index('help_menu_item = primary_menu_item'):
                           source.index("menu.append(Gtk.SeparatorMenuItem())", source.index("menu.append(help_menu_item)"))]
        for item in ("history_item", "logs_menu_item", "guide_item", "github_item"):
            self.assertNotIn(f"menu.append({item})", top_level)

    def test_job_status_and_tray_share_backend_icon_priority(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        selected = [node for node in tree.body
                    if isinstance(node, (ast.Assign, ast.FunctionDef))
                    and (isinstance(node, ast.FunctionDef)
                         and node.name == "backend_status_icon"
                         or isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name)
                                 and target.id == "BACKEND_STATUS"
                                 for target in node.targets))]
        namespace = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "tray-status", "exec"), namespace)
        choose = namespace["backend_status_icon"]
        self.assertEqual(choose("recording", "cuda"), ("recording", "🔴"))
        self.assertEqual(choose("processing", "cloudflare", "cpu"), ("processing_cloudflare", "☁️"))
        self.assertEqual(choose("processing", "groq", "remote"), ("processing_groq", "🟠"))
        self.assertEqual(choose("processing", "google", "remote"), ("processing_google", "🔵"))
        self.assertEqual(choose("processing", "openrouter", "remote"), ("processing_openrouter", "🧭"))
        self.assertEqual(choose("processing", "local", "cuda"), ("processing_gpu", "🎮"))
        self.assertEqual(choose("processing", "local_fallback", "cpu"), ("processing_cpu", "🔥"))
        self.assertIn('"recording": ("recording", "🔴")', source)
        self.assertIn('"cloudflare": ("processing_cloudflare", "☁️")', source)
        self.assertIn('"groq": ("processing_groq", "🟠")', source)
        self.assertIn('"google": ("processing_google", "🔵")', source)
        self.assertIn('"openrouter": ("processing_openrouter", "🧭")', source)
        self.assertIn('"cuda": ("processing_gpu", "🎮")', source)
        self.assertIn('"cpu": ("processing_cpu", "🔥")', source)
        self.assertIn('priority = ("processing_cpu", "processing_cloudflare"', source)
        self.assertIn('menu.append(groq_top_menu_item)', source)
        self.assertIn('menu.append(openrouter_top_menu_item)', source)
        self.assertIn('("google", "Google Gemini", "processing_google")', source)
        self.assertIn('https://openrouter.ai/activity', source)
        self.assertIn('set_status_row_icon(image, entry.get("status_icon", "processing_cpu"))', source)
        self.assertIn('label = f"{model} — {action} {stage}".rstrip()', source)
        self.assertIn("state, backend, device, model, j.get(\"asr_backend_selected\")", source)
        self.assertIn('"processing_gigaam": menu_icon_path("gigaam")', source)
        self.assertIn('"pipeline_stage"', source)
        self.assertNotIn("DEVICE_EMOJI.get(backend", source)
        self.assertNotIn('"processing": "⏰"', source)
        self.assertNotIn('⏰', source[source.index('def build_batch_display_entries'):source.index('def cancel_batch_job')])
        self.assertNotIn('⚙️', source[source.index('BATCH_STATUS_LABELS'):source.index('LANGUAGE_LABELS')])

    def test_recording_status_hides_profile_model_until_backend_is_selected(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        status_branch = source[source.index("# Backend selection has not happened while recording"):
                              source.index("# During correction the model/backend")]
        self.assertIn('if state == "recording":', status_branch)
        self.assertIn('label = "Запись"', status_branch)
        self.assertIn('"recording": menu_icon_path("recording")', source)
        self.assertIn('notify-send "Whisper Dictate" "🎙️ Запись"',
                      (ROOT / "scripts/whisper-dictate-v2").read_text(encoding="utf-8"))

    def test_local_asr_dialog_avoids_spinbutton_theme_icon_abort(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        dialog = source[source.index("def configure_local_recognition_settings"):
                        source.index("def update_corrector_label")]
        self.assertIn("beam = integer_range_scale(1, 50, 1)", dialog)
        self.assertNotRegex(source, r"=\s*Gtk\.SpinButton")
        self.assertNotIn("get_value_as_int()", source)
