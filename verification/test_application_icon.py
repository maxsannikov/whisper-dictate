import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class ApplicationIconTests(unittest.TestCase):
    def test_installer_installs_project_icon_and_uses_it_for_desktop_entry(self):
        installer = (ROOT / "install.sh").read_text(encoding="utf-8")
        self.assertIn('"$SCRIPT_DIR/assets/whisper-dictate-icon.png"', installer)
        self.assertIn('APP_ICON="$ICON_DIR/whisper-dictate.png"', installer)
        self.assertIn("Icon=$APP_ICON", installer)
        self.assertIn('APP_ICON="microphone"', installer)

    def test_tray_uses_project_icon_when_installed(self):
        tray = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        self.assertIn('~/.local/share/pixmaps/whisper-dictate.png', tray)
        self.assertIn('"idle": APP_ICON_PATH if os.path.isfile(APP_ICON_PATH)', tray)

    def test_asset_is_square_png(self):
        asset = ROOT / "assets/whisper-dictate-icon.png"
        self.assertTrue(asset.is_file(), "generated app icon asset is missing")
        with asset.open("rb") as source:
            header = source.read(26)
        self.assertEqual(header[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", header[16:24])
        self.assertEqual(width, height)
        self.assertEqual(header[25], 6, "icon must use RGBA with transparency")

    def test_uninstaller_lists_and_removes_installed_icon(self):
        uninstaller = (ROOT / "uninstall.sh").read_text(encoding="utf-8")
        self.assertIn('echo "  • $APP_ICON"', uninstaller)
        self.assertIn('rm -f "$APP_ICON"', uninstaller)


if __name__ == "__main__":
    unittest.main()
