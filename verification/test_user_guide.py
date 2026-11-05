import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class UserGuideWiringTests(unittest.TestCase):
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
        item = source.index('Gtk.MenuItem(label="⌨️ Горячие клавиши и установка")')
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
        github_item = source.index('Gtk.MenuItem(label="🌐 GitHub проекта")')
        handler = source[source.index("def open_project_github") : github_item]
        self.assertIn('subprocess.Popen(["xdg-open", github_url])', handler)
        self.assertNotIn("shell=True", handler)

    def test_help_menu_has_exactly_four_requested_items(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        block = source[source.index("help_submenu = Gtk.Menu()"):
                       source.index("help_submenu.show_all()")]
        self.assertEqual(block.count("help_submenu.append("), 4)
        self.assertIn("help_submenu.append(history_item)", block)
        self.assertIn("help_submenu.append(logs_menu_item)", block)
        self.assertIn("help_submenu.append(guide_item)", block)
        self.assertIn("help_submenu.append(github_item)", block)
        self.assertIn('Gtk.MenuItem(label="❔ Помощь")', source)
        top_level = source[source.index('help_menu_item = Gtk.MenuItem'):
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
        self.assertEqual(choose("processing", "local", "cuda"), ("processing_gpu", "🎮"))
        self.assertEqual(choose("processing", "local_fallback", "cpu"), ("processing_cpu", "🔥"))
        self.assertIn('"recording": ("recording", "🔴")', source)
        self.assertIn('"cloudflare": ("processing_cloudflare", "☁️")', source)
        self.assertIn('"cuda": ("processing_gpu", "🎮")', source)
        self.assertIn('"cpu": ("processing_cpu", "🔥")', source)
        self.assertIn("_icon_key, icon_label = backend_status_icon(state, backend, device)", source)
        self.assertIn('label = f"{icon_label} {model} — {action} {stage}".rstrip()', source)
        self.assertIn("backend_status_icon(state, backend, device)", source)
        self.assertIn('"pipeline_stage"', source)
        self.assertNotIn("DEVICE_EMOJI.get(backend", source)
        self.assertNotIn('"processing": "⏰"', source)
        self.assertNotIn('⏰', source[source.index('def build_batch_display_entries'):source.index('def cancel_batch_job')])
        self.assertNotIn('⚙️', source[source.index('BATCH_STATUS_LABELS'):source.index('LANGUAGE_LABELS')])
