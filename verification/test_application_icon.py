import hashlib
import struct
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class ApplicationIconTests(unittest.TestCase):
    def test_desktop_entry_reuses_the_tray_idle_icon(self):
        installer = (ROOT / "install.sh").read_text(encoding="utf-8")
        tray = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        self.assertIn('MENU_ICON_DIR="$HOME/.local/share/whisper-dictate/menu-icons"', installer)
        self.assertIn('APP_ICON="$MENU_ICON_DIR/microphone.png"', installer)
        self.assertIn("Icon=$APP_ICON", installer)
        self.assertIn('"idle": menu_icon_path("microphone")', tray)

    def test_tray_uses_bundled_menu_icons_without_theme_lookup(self):
        tray = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        installer = (ROOT / "install.sh").read_text(encoding="utf-8")
        self.assertIn('"idle": menu_icon_path("microphone")', tray)
        self.assertIn('MENU_ICON_DIR="$HOME/.local/share/whisper-dictate/menu-icons"', installer)
        self.assertIn('set_from_pixbuf(pixbuf)', tray)
        icon_size = re.search(r"^MENU_ICON_SIZE\s*=\s*(\d+)$", tray, re.MULTILINE)
        self.assertIsNotNone(icon_size)
        self.assertGreater(int(icon_size.group(1)), 24)
        self.assertIn("image.set_size_request(MENU_ICON_SIZE, MENU_ICON_SIZE)", tray)
        self.assertNotIn('GdkPixbuf.Pixbuf.new_from_file_at_scale', tray)
        self.assertNotIn('image.set_from_icon_name(source', tray)
        self.assertTrue((ROOT / "assets/menu-icons/microphone.png").is_file())

    def test_shared_tray_launcher_asset_is_square_png(self):
        asset = ROOT / "assets/menu-icons/microphone.png"
        self.assertTrue(asset.is_file(), "shared tray/launcher icon asset is missing")
        with asset.open("rb") as source:
            header = source.read(26)
        self.assertEqual(header[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", header[16:24])
        self.assertEqual(width, height)
        self.assertEqual(header[25], 6, "icon must use RGBA with transparency")

    def test_action_icons_have_editable_vector_sources_and_provider_logos_stay_unboxed(self):
        icon_dir = ROOT / "assets/menu-icons"
        required_vectors = {
            "model_services", "network", "help", "history", "logs", "error",
            "stats", "changelog", "keyboard", "github",
        }
        for name in required_vectors:
            with self.subTest(icon=name):
                svg = icon_dir / f"{name}.svg"
                png = icon_dir / f"{name}.png"
                self.assertTrue(svg.is_file(), f"editable SVG source missing: {name}")
                self.assertTrue(png.is_file(), f"rendered PNG missing: {name}")
                self.assertIn('stroke="#27323a"', svg.read_text(encoding="utf-8"))

        original_provider_hashes = {
            "cloudflare": "c305a1d090c19c12433d5abbd75b829ea2a09ee5c04afe8d1f2e6d936352b342",
            "openrouter": "985e42310b3a708676b51804d55a75b6bcaf96839348b1013f02e4bbbf0aff62",
            "codex": "571f20e9ab7c07703d2a1796521ad5e4b4dbeb440808c3e4705ddaf5f8aeacc2",
            "claude": "086dcfd813979e13326808995b9c291c84ad34cc30373914467901ac43aa611b",
            "groq": "a9fd7bd2583dd2b9cf4aca0626a21dba5f3446e2eae31b86847d5b4adcb2906a",
            "google": "8f547592b97dece23f3168439fba7214eeb747165a5c7f88790f943812ebc78f",
        }
        tray = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        size = re.search(r"^PROVIDER_ICON_SIZE\s*=\s*(\d+)$", tray, re.MULTILINE)
        self.assertIsNotNone(size)
        self.assertGreater(int(size.group(1)), 40)
        for name, expected_hash in original_provider_hashes.items():
            with self.subTest(provider=name):
                actual = hashlib.sha256((icon_dir / f"{name}.png").read_bytes()).hexdigest()
                self.assertEqual(actual, expected_hash)

    def test_uninstaller_lists_and_removes_installed_icon(self):
        uninstaller = (ROOT / "uninstall.sh").read_text(encoding="utf-8")
        self.assertIn('APP_ICON="$HOME/.local/share/whisper-dictate/menu-icons/microphone.png"', uninstaller)
        self.assertIn('echo "  • $APP_ICON"', uninstaller)
        self.assertIn('rm -f "$APP_ICON"', uninstaller)
        self.assertIn('echo "  • $LEGACY_APP_ICON (legacy launcher icon, if present)"', uninstaller)
        self.assertIn('rm -f "$LEGACY_APP_ICON"', uninstaller)


if __name__ == "__main__":
    unittest.main()
