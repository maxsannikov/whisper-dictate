"""Synthetic tests for the safe Claude CLI correction subprocess."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "multi_provider_claude_test", ROOT / "scripts/multi_provider.py")
multi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multi)


class FakeProcess:
    def __init__(self, command, stdout, **_kwargs):
        self.command = command
        self.stdout = stdout
        self.returncode = None

    def communicate(self, input=None, timeout=None):
        self.stdin_payload = input
        self.stdout.write(b"Fixed synthetic sentence.")
        self.returncode = 0
        return None, None

    def poll(self):
        return self.returncode


class ClaudeAdapterTests(unittest.TestCase):
    def test_cli_is_text_only_isolated_and_does_not_put_source_in_arguments(self):
        process = None

        def fake_popen(command, **kwargs):
            nonlocal process
            process = FakeProcess(command, **kwargs)
            return process

        with patch.object(multi.shutil, "which", return_value="/mock/claude"), \
             patch.object(multi.subprocess, "Popen", side_effect=fake_popen):
            result = multi.cli_correct("claude", "Synthetic correction source", "sonnet")

        self.assertEqual(result, "Fixed synthetic sentence.")
        self.assertNotIn("Synthetic correction source", process.command)
        self.assertIn("--tools", process.command)
        self.assertEqual(process.command[process.command.index("--tools") + 1], "")
        for flag, value in (("--safe-mode", None), ("--disallowedTools", "mcp__*"),
                            ("--permission-prompts", "none"),
                            ("--no-session-persistence", None), ("--max-turns", "1")):
            self.assertIn(flag, process.command)
            if value is not None:
                self.assertEqual(process.command[process.command.index(flag) + 1], value)
        self.assertEqual(process.stdin_payload.decode().split("Text to correct:\n", 1)[1],
                         "Synthetic correction source")
        self.assertTrue(process.command[process.command.index("--model") + 1] == "sonnet")


if __name__ == "__main__":
    unittest.main()
