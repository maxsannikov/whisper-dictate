"""Synthetic coverage for stripping the known Whisper subtitle-credit hallucination."""
import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "scripts/whisper-daemon.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
HELPER = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and
              node.name == "remove_subtitle_credit_tail")
NAMESPACE = {"re": re}
exec(compile(ast.Module(body=[HELPER], type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
remove_tail = NAMESPACE["remove_subtitle_credit_tail"]


class SubtitleCreditArtifactTests(unittest.TestCase):
    def test_entire_known_credit_signature_is_removed(self):
        self.assertEqual(remove_tail(
            "Редактор субтитров А. Семкин. Корректор А. Егорова."), "")

    def test_known_signature_is_removed_only_from_the_end(self):
        self.assertEqual(remove_tail(
            "Проверим звук. Редактор субтитров А. Семкин. Корректор А. Егорова."),
            "Проверим звук.")

    def test_similar_spoken_words_are_preserved(self):
        for text in ("Редактор субтитров.",
                     "Мы говорим про редактора субтитров А. Семкина.",
                     "Редактор субтитров А. Семкин подготовил доклад."):
            with self.subTest(text=text):
                self.assertEqual(remove_tail(text), text)

    def test_whisper_prompt_explicitly_marks_signature_as_non_speech(self):
        daemon = SOURCE.read_text(encoding="utf-8")
        remote_tree = ast.parse((ROOT / "scripts/multi_provider.py").read_text(encoding="utf-8"))
        prompt_assignment = next(node for node in remote_tree.body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
            target.id == "REMOTE_CORRECTION_INSTRUCTION" for target in node.targets))
        remote_value = prompt_assignment.value
        remote = (ast.literal_eval(remote_value) if isinstance(remote_value, ast.Constant) else
                  "".join(ast.literal_eval(item) for item in remote_value.elts))
        signature = "Редактор субтитров А. Семкин. Корректор А. Егорова."
        self.assertIn(signature, daemon)
        self.assertIn(signature, remote)


if __name__ == "__main__":
    unittest.main()
