import ast
import importlib.util
from pathlib import Path
from unittest.mock import Mock
import unittest

ROOT = Path(__file__).parents[1]


def load_daemon_backend_selector():
    source = (ROOT / "scripts/whisper-daemon.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_job_backends"]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "daemon-backend-selector", "exec"), namespace)
    return namespace["select_job_backends"]


def load_request_profile():
    source = (ROOT / "scripts/whisper-daemon.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "request_profile"]
    namespace = {"MODELS": {"large-v3", "large-v3-turbo", "medium", "small"},
                 "COMPUTES": {"int8_float16", "float16", "int8", "float32"}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "daemon-request-profile", "exec"), namespace)
    return namespace["request_profile"]


def load_batch_preparer():
    source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == "prepare_youtube_batch_files"]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 "youtube-batch-preparer", "exec"), namespace)
    return namespace["prepare_youtube_batch_files"]


spec = importlib.util.spec_from_file_location("whisper_youtube", ROOT / "scripts/whisper-youtube.py")
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)


class VideoLocalOnlyTests(unittest.TestCase):
    def test_five_downloads_survive_callback_state_and_keep_partial_success_order(self):
        prepare = load_batch_preparer()
        entries = [{"video_id": str(i), "title": f"Video {i}",
                    "url": f"https://rutube.test/{i}", "audio_path": None,
                    "duration_sec": 12} for i in range(5)]
        # The poller state remains available until the idle callback consumes it.
        downloads = {
            i: {"entry": entry, "error": ({"code": "download_failed"}
                                           if i == 2 else None)}
            for i, entry in enumerate(entries)
        }
        existing = {"/cache/0.mp3", "/cache/1.mp3", "/cache/3.mp3", "/cache/4.mp3"}
        prepared, failures = prepare(
            entries, downloads, lambda entry: f"/cache/{entry['video_id']}.mp3",
            existing.__contains__, lambda _path: 12,
        )
        self.assertEqual([item["video_id"] for item in prepared], ["0", "1", "3", "4"])
        self.assertEqual([item["status"] for item in failures], ["download_error"])

    def test_retry_uses_completed_audio_paths_without_requiring_old_download_state(self):
        prepare = load_batch_preparer()
        entries = [{"video_id": str(i), "title": f"Video {i}",
                    "url": f"https://rutube.test/{i}",
                    "audio_path": f"/cache/{i}.mp3", "duration_sec": 12}
                   for i in range(5)]
        existing = {item["audio_path"] for item in entries}
        prepared, failures = prepare(
            entries, {}, lambda _entry: None, existing.__contains__, lambda _path: 12,
        )
        self.assertEqual([item["video_id"] for item in prepared], ["0", "1", "2", "3", "4"])
        self.assertEqual(failures, [])

    def test_metadata_batch_keeps_order_and_continues_after_mixed_failures(self):
        entries = [{"url": f"https://video.test/{i}", "video_id": str(i)} for i in range(5)]

        def resolve(entry):
            if entry["video_id"] == "1":
                raise ValueError("private details must not escape")
            if entry["video_id"] == "2":
                raise TimeoutError()
            if entry["video_id"] == "3":
                raise OSError("spawn details must not escape")
            return {"video_id": entry["video_id"], "title": "ok"}

        resolved, failed = video.resolve_metadata_batch(entries, resolve)
        self.assertEqual([row["input_index"] for row in resolved], [0, 4])
        self.assertEqual([row["input_index"] for row in failed], [1, 2, 3])
        self.assertEqual([row["status"] for row in failed],
                         ["metadata_error", "metadata_timeout", "metadata_error"])
        self.assertNotIn("private details", repr(failed))
        self.assertNotIn("spawn details", repr(failed))

    def test_metadata_batch_marks_duplicates_without_repeating_request(self):
        entries = [{"url": "https://video.test/a"}, {"url": "https://video.test/a"}]
        resolve = Mock(return_value={"video_id": "a", "title": "A"})
        resolved, failed = video.resolve_metadata_batch(entries, resolve)
        resolve.assert_called_once()
        self.assertEqual([row["input_index"] for row in resolved], [0])
        self.assertEqual(failed[0]["reason"], "duplicate")

    def test_metadata_batch_survives_all_provider_errors(self):
        entries = [{"url": "https://video.test/a"}, {"url": "https://video.test/b"}]

        def cannot_start(_entry):
            raise FileNotFoundError("yt-dlp")

        resolved, failed = video.resolve_metadata_batch(entries, cannot_start)
        self.assertEqual(resolved, [])
        self.assertEqual([row["reason"] for row in failed],
                         ["ytdlp_start_failed", "ytdlp_start_failed"])

    def test_download_results_keep_each_input_and_report_spawn_failure(self):
        entries = [{"url": "a"}, {"url": "b"}, {"url": "c"}]
        states = [video.download_state(entries[0], object()),
                  video.download_state(entries[1], None),
                  video.download_state(entries[2], object())]
        self.assertEqual([state["entry"]["url"] for state in states], ["a", "b", "c"])
        self.assertEqual(states[1]["status"], "download_error")
        self.assertEqual(states[1]["error"]["code"], "launch_failed")
        self.assertEqual([states[0]["status"], states[2]["status"]],
                         ["downloading", "downloading"])

    def test_platform_url_validation_and_input_order(self):
        entries = video.parse_youtube_urls("https://youtu.be/dQw4w9WgXcQ\nhttps://rutube.ru/video/abc/\nhttps://vk.com/video-1_2\nhttps://example.com/watch")
        self.assertEqual([e["platform"] for e in entries], ["youtube", "rutube", "vk", None])
        self.assertEqual([e["valid"] for e in entries], [True, True, True, False])
        self.assertEqual(entries[0]["url"], "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_source_detection_rejects_spoofed_hosts_and_credentials(self):
        self.assertEqual(video.source_for_url("https://m.rutube.ru/video/id"), "rutube")
        self.assertEqual(video.source_for_url("https://vkvideo.ru/video-1_2"), "vk")
        self.assertIsNone(video.source_for_url("https://youtube.com.example.org/watch?v=x"))
        self.assertIsNone(video.source_for_url("https://user@youtube.com/watch?v=x"))

    def test_non_live_sources_stay_local_when_cloud_toggles_are_enabled(self):
        select = load_daemon_backend_selector()
        profile = {"cloudflare_asr_enabled": True,
                   "cloudflare_corrector": {"enabled": True, "model": "remote"}}
        for source in ("file", "batch", "youtube", "rutube", "vk", "unknown", None):
            with self.subTest(source=source):
                self.assertEqual(select(source, profile), ("local", {"enabled": False}))
        self.assertEqual(select("live_dictation", profile),
                         ("cloudflare", profile["cloudflare_corrector"]))

    def test_batch_persists_batch_and_per_item_source(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        self.assertIn('"source": "batch"', source)
        self.assertIn('"source": item_source', source)
        self.assertIn('"WHISPER_JOB_SOURCE": job_source', source)

    def test_daemon_resolves_current_local_asr_profile_per_request(self):
        source = (ROOT / "scripts/whisper-daemon.py").read_text(encoding="utf-8")
        client = (ROOT / "scripts/whisper-daemon-client.sh").read_text(encoding="utf-8")
        # Each incoming job starts from config(), then request_profile only
        # accepts runtime tuning fields; it cannot override asr_backend.
        self.assertIn("base_profile = config()", source)
        self.assertIn('"asr_backend"', source[source.index("def config()"):
                                                  source.index("def request_profile")])
        self.assertNotIn('("asr_backend",', source[source.index("def request_profile"):
                                                       source.index("def prompt")])
        self.assertIn("selected local ASR backend requires the Whisper daemon", client)

        request_profile = load_request_profile()
        first = request_profile({}, {"model": "large-v3", "device": "cuda",
                                     "compute_type": "int8_float16", "language": "ru",
                                     "beam_size": 1, "best_of": 1, "temperature": 0.0,
                                     "vad": False, "asr_backend": "gigaam-v3-e2e-rnnt"})
        second = request_profile({}, {"model": "large-v3", "device": "cuda",
                                      "compute_type": "int8_float16", "language": "ru",
                                      "beam_size": 1, "best_of": 1, "temperature": 0.0,
                                      "vad": False, "asr_backend": "faster-whisper"})
        self.assertEqual(first["asr_backend"], "gigaam-v3-e2e-rnnt")
        self.assertEqual(second["asr_backend"], "faster-whisper")
        self.assertEqual(request_profile({"asr_backend": "faster-whisper"}, first)["asr_backend"],
                         "gigaam-v3-e2e-rnnt")

    def test_local_asr_settings_put_gigaam_first_without_rewriting_choice(self):
        source = (ROOT / "scripts/whisper-tray.py").read_text(encoding="utf-8")
        gigaam = source.index('backend.append("gigaam-v3-e2e-rnnt"')
        faster = source.index('backend.append("faster-whisper"')
        self.assertLess(gigaam, faster)
        self.assertIn('selected = root_config.get("asr_backend", "faster-whisper")', source)


if __name__ == "__main__":
    unittest.main()
