import ast
from datetime import datetime
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]


def load_helpers():
    source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"format_cloudflare_usage_lines", "format_cloudflare_attempt_lines",
             "cloudflare_error_reason"}
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"datetime": datetime}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "cloudflare-menu-rows", "exec"), namespace)
    return namespace


class CloudflareMenuRowsTests(unittest.TestCase):
    def test_usage_summary_is_split_into_four_items(self):
        rows = load_helpers()["format_cloudflare_usage_lines"]({
            "estimated_neurons": 200, "corrector_estimated_neurons": 50,
            "corrector_estimated_token_requests": 1, "audio_seconds": 120, "requests": 1,
        }, 10_000)
        self.assertEqual(len(rows), 4)
        self.assertIn("Нейронов за UTC-день", rows[0])
        self.assertIn("Расчётный остаток", rows[1])
        self.assertTrue(rows[2].startswith("ASR:"))
        self.assertTrue(rows[3].startswith("Корректор:"))

    def test_last_attempt_separates_whisper_corrector_and_time(self):
        rows = load_helpers()["format_cloudflare_attempt_lines"]({
            "timestamp": "2026-10-23T12:34:00+00:00", "asr_backend": "cloudflare",
            "corrector_backend": "local", "cloudflare_corrector_error_category": "invalid_response",
            "corrector_device": "cuda",
        })
        self.assertEqual(len(rows), 3)
        self.assertIn("Whisper:", rows[0])
        self.assertIn("Корректор:", rows[1])
        self.assertIn("Время:", rows[2])


if __name__ == "__main__":
    unittest.main()
