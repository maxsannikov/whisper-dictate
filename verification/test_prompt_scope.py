"""Verify ASR and correction prompts remain separate."""
import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[1] / "scripts/whisper-daemon.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
PROMPT = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and
              node.name == "prompt")
NAMESPACE = {}
exec(compile(ast.Module(body=[PROMPT], type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)


class PromptScopeTests(unittest.TestCase):
    def test_asr_prompt_uses_initial_prompt_and_glossary_only(self):
        correction_only = "special rules for the corrector"
        asr_prompt = NAMESPACE["prompt"]({
            "initial_prompt": "transcription vocabulary",
            "glossary": ["Qwen"],
            "corrector": {"prompt": correction_only},
        })
        self.assertIn("transcription vocabulary", asr_prompt)
        self.assertIn("Qwen", asr_prompt)
        self.assertNotIn(correction_only, asr_prompt)


if __name__ == "__main__":
    unittest.main()
