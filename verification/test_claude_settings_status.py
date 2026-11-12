"""Tests that the tray checks Claude login without exposing auth output."""
import ast
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).parents[1] / "scripts/whisper-tray.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
HELPER = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and
              node.name == "claude_cli_login_state")
NAMESPACE = {"subprocess": subprocess, "OSError": OSError}
exec(compile(ast.Module(body=[HELPER], type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)


class ClaudeSettingsStatusTests(unittest.TestCase):
    def test_login_status_is_reduced_to_exit_code_and_discards_output(self):
        with patch.object(subprocess, "run", return_value=types.SimpleNamespace(returncode=1)) as run:
            self.assertFalse(NAMESPACE["claude_cli_login_state"]())
        kwargs = run.call_args.kwargs
        self.assertIs(kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], subprocess.DEVNULL)
        self.assertEqual(run.call_args.args[0], ["claude", "auth", "status"])

    def test_unavailable_cli_has_unknown_status(self):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            self.assertIsNone(NAMESPACE["claude_cli_login_state"]())

    def test_credential_and_login_status_labels_are_neutral(self):
        source = SOURCE.read_text(encoding="utf-8")
        for start, end in (("def provider_credential_status", "def configure_provider_settings"),
                           ("def refresh_codex_account_status", "codex_menu = Gtk.Menu()"),
                           ("def refresh_claude_menu", "claude_enabled_item.connect")):
            block = source[source.index(start):source.index(end, source.index(start))]
            for mark in ("✅", "❌", "⚠️"):
                self.assertNotIn(mark, block)


if __name__ == "__main__":
    unittest.main()
