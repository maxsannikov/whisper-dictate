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

    def test_menu_opens_installed_doc_via_argument_list_after_logs(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        logs = source.index("menu.append(logs_menu_item)")
        item = source.index('Gtk.MenuItem(label="⌨️ Горячие клавиши и установка")')
        self.assertGreater(item, logs)
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

    def test_no_unverified_github_url(self):
        guide = (ROOT / "docs/guide.md").read_text(encoding="utf-8")
        self.assertIn("проверенной GitHub-ссылки нет", guide)
        self.assertNotIn("github.com/", guide)
