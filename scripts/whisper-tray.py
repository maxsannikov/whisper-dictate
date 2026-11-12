#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gi, glob, html, json, os, shutil, signal, socket, subprocess, sys, tempfile, time, threading, unicodedata, re, webbrowser, traceback, urllib.request, select
from collections import deque
from datetime import datetime, timezone

gi.require_version('Gtk', '3.0')
gi.require_version('AppIndicator3', '0.1')
from gi.repository import Gtk, AppIndicator3, GLib, Pango, Gdk, GdkPixbuf

# Breeze can assign oversized rows to GTK combo popups, which makes the menu
# appear clipped when it reaches the screen edge. Keep combo rows compact so
# more choices fit at once; GTK still scrolls the popup when it exceeds the
# available monitor height.
_combo_popup_css = Gtk.CssProvider()
_combo_popup_css.load_from_data(b"""
combobox button.combo {
    min-height: 24px;
    padding-top: 3px;
    padding-bottom: 3px;
}
combobox window.popup menuitem,
combobox window.popup row {
    min-height: 24px;
    padding: 2px 6px;
}
""")
_combo_popup_screen = Gdk.Screen.get_default()
if _combo_popup_screen is not None:
    Gtk.StyleContext.add_provider_for_screen(
        _combo_popup_screen, _combo_popup_css,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

sys.path.insert(0, os.path.dirname(__file__))
from clipboard_delivery import deliver_clipboard_text
from cloudflare_asr import (daily_usage as cloudflare_daily_usage,
                            FREE_DAILY_NEURONS as CLOUDFLARE_LOCAL_DAILY_REFERENCE)

STATE_DIR = os.path.expanduser("~/.cache/whisper-dictate")
MANUAL_CORRECTION_STATUS_FILE = os.path.join(STATE_DIR, "manual-correction-status.json")
JOBS_DIR = os.path.join(STATE_DIR, "jobs")
ACTIVE_RECORDING_FILE = os.path.join(STATE_DIR, ".active-recording")
LATEST_LIVE_ASR_AT_FILE = os.path.join(STATE_DIR, ".latest-live-asr-at")
BATCH_JOBS_DIR = os.path.join(STATE_DIR, "batch-jobs")
BATCH_RESULTS_DIR = os.path.join(STATE_DIR, "batch-results")
HISTORY_FILE = os.path.join(STATE_DIR, "history.jsonl")
LOG_FILE = os.path.join(STATE_DIR, "tray.log")
METRICS_FILE = os.path.join(STATE_DIR, "metrics.jsonl")
QUALITY_CONFIG_FILE = os.path.join(STATE_DIR, "quality-config.json")
VOICE_CONFIG_FILE = os.path.expanduser("~/.config/whisper-dictate/config.json")
AUTOREPLACEMENTS_FILE = os.path.expanduser("~/.config/whisper-dictate/autoreplacements.txt")
POLL_MS = 400
MAX_JOB_SLOTS = 6
MAX_BATCH_PARALLEL = 3

# Форматы, которые локальный ffmpeg/ffprobe может прочитать и передать ASR.
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".m4b", ".m4p", ".aac", ".flac", ".ogg",
    ".oga", ".opus", ".wma", ".mp2", ".mpa", ".mpga", ".aif", ".aiff",
    ".aifc", ".caf", ".amr", ".awb", ".ac3", ".eac3", ".dts", ".ape",
    ".wv", ".tta", ".au", ".snd", ".voc", ".ra", ".rm", ".mka",
}
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v", ".wmv",
    ".mpg", ".mpeg", ".mpe", ".mpv", ".3gp", ".3g2", ".mts", ".m2ts",
    ".m2v", ".vob", ".ogv", ".asf", ".rmvb", ".mxf", ".dv", ".ts",
    ".f4v",
}

# YouTube-интеграция константы
YOUTUBE_AUDIO_DIR = os.path.join(STATE_DIR, "youtube-audio")
YOUTUBE_HISTORY_FILE = os.path.join(STATE_DIR, "youtube-history.jsonl")
MAX_YOUTUBE_URLS = 20
MAX_YOUTUBE_PARALLEL_DOWNLOADS = 2
YOUTUBE_CACHE_MAX_AGE_DAYS = 30
YOUTUBE_CACHE_MAX_GB = 5.0
YOUTUBE_HISTORY_MAX_LINES = 500
YTDLP_METADATA_TIMEOUT = 20
YOUTUBE_DISK_HEADROOM_MB = 500
YOUTUBE_BITRATE_MB_PER_MIN = 1.2  # Оценка для MP3 16kHz mono

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(BATCH_JOBS_DIR, exist_ok=True)
os.makedirs(BATCH_RESULTS_DIR, exist_ok=True)
os.makedirs(YOUTUBE_AUDIO_DIR, exist_ok=True)

_batch_inflight = {}  # batch_id -> {file_index: subprocess.Popen}
_batch_windows = {}  # batch_id -> {"window": Gtk.Window, "timeout_id": int}
_youtube_downloads = {}  # download_id -> {url: Popen, progress: dict, ...}
_youtube_windows = {}  # download_id -> {"window": Gtk.Window, "timeout_id": int}
_youtube_metadata_fetch = None  # Текущий процесс fetch метаданных или None
_corrector_pull = None  # {"model": str, "process": Popen} или None
_audio_restore = None  # Popen аварийного восстановления громкости или None

DEFAULT_QUALITY_CONFIG = {
    "current_strategy": "normal",
    "current_model": "large-v3",
    "auto_mode": False,
    "model_options": {
        "large-v3": {
            "name": "Large (Best Quality)",
            "vram_gb": 4.5,
            "speed_relative": 1.0
        },
        "medium": {
            "name": "Medium (Good Quality)",
            "vram_gb": 1.5,
            "speed_relative": 2.5
        },
        "small": {
            "name": "Small (Fast)",
            "vram_gb": 0.5,
            "speed_relative": 5.0
        },
        "large-v3-turbo": {
            "name": "Large-v3 Turbo (эксперимент: требуется A/B-тест)",
            "vram_gb": 3.0,
            "speed_relative": 2.0
        }
    },
    "auto_mode_settings": {
        "check_gpu_before_transcribe": True,
        "fallback_on_oom": True,
        "min_free_gpu_percent": 10
    },
    "gpu_metrics": {
        "last_check": None,
        "free_gb": 0,
        "total_gb": 0,
        "used_gb": 0,
        "selected_model": "large-v3",
        "selection_reason": ""
    },
    "fallback_history": {
        "total_transcriptions": 0,
        "fallback_to_medium": 0,
        "fallback_to_small": 0,
        "fallback_to_cpu": 0,
        "oom_errors": 0
    },
    "strategies": {
        "energy": {
            "name": "Энергосбережение",
            "beam_size": 1,
            "fp16": True,
            "best_of": 1,
            "temperature": 0.0,
            "language": "ru",
            "vad_threshold": 0.5
        },
        "normal": {
            "name": "Обычно",
            "beam_size": 1,
            "fp16": True,
            "best_of": 1,
            "temperature": 0.0,
            "language": "ru",
            "vad_threshold": 0.5
        },
        "good": {
            "name": "Хорошее",
            "beam_size": 10,
            "fp16": False,
            "best_of": 1,
            "temperature": 0.0,
            "language": "ru",
            "vad_threshold": 0.5
        },
        "maximum": {
            "name": "Максимум",
            "beam_size": 20,
            "fp16": False,
            "best_of": 1,
            "temperature": 0.0,
            "language": "ru",
            "vad_threshold": 0.5
        },
        "custom": {
            "name": "Пользовательская",
            "beam_size": 5,
            "fp16": False,
            "best_of": 1,
            "temperature": 0.0,
            "language": "ru",
            "vad_threshold": 0.5
        }
    }
}

def log_debug(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ============================================================================
# YouTube-интеграция: Фаза 8.2 (окружение и кэш)
# ============================================================================

def sanitize_filename(name, max_len=80):
    """Привести произвольную строку к безопасному имени файла.
    (Дублируется из whisper-youtube.py для работы без импортов)"""
    name = ''.join(c for c in name if unicodedata.category(c)[0] != 'C' or c == '\n')
    name = re.sub(r'[/\\:*?"<>|]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'^[\.\-\s]+', '', name)

    if not name or name in ('.', '..'):
        return 'video'

    if len(name) > max_len:
        name = name[:max_len]
        last_space = name.rfind(' ')
        if last_space > max_len // 2:
            name = name[:last_space]
        name = name.rstrip()

    return name if name else 'video'

def check_ytdlp_available():
    """Проверить доступность yt-dlp.
    Возвращает dict: {"available": bool, "path": str|None, "version": str|None, "error": str|None}
    """
    try:
        path = shutil.which("yt-dlp")
        if not path:
            return {"available": False, "path": None, "version": None, "error": "yt-dlp не найден"}

        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0] if result.stdout else "unknown"
            return {"available": True, "path": path, "version": version, "error": None}
        else:
            return {"available": False, "path": path, "version": None, "error": f"Ошибка проверки версии: {result.stderr}"}
    except subprocess.TimeoutExpired:
        return {"available": False, "path": None, "version": None, "error": "Тайм-аут проверки yt-dlp"}
    except Exception as e:
        return {"available": False, "path": None, "version": None, "error": str(e)}

def get_free_disk_space_mb(path):
    """Получить свободное место на диске в МБ. Возвращает float или None."""
    try:
        stat = shutil.disk_usage(path)
        return stat.free / (1024 * 1024)
    except (OSError, IOError):
        return None

def estimate_download_size_mb(duration_sec):
    """Оценка размера MP3 16kHz mono в МБ.
    Формула: duration_sec * YOUTUBE_BITRATE_MB_PER_MIN / 60
    """
    if duration_sec is None or duration_sec <= 0:
        return 50  # Консервативная оценка для неизвестной длительности
    return duration_sec * YOUTUBE_BITRATE_MB_PER_MIN / 60.0

def check_youtube_disk_space(entries):
    """Проверить есть ли достаточно места для скачивания всех ролей.
    Возвращает dict: {"ok": bool, "needed_mb": float, "free_mb": float, "error": str|None}
    """
    total_estimate_mb = 0.0
    for entry in entries:
        if entry.get("valid"):
            duration = entry.get("duration_sec")
            total_estimate_mb += estimate_download_size_mb(duration)

    # Добавить запас (промежуточные файлы yt-dlp)
    total_needed_mb = total_estimate_mb * 2 + YOUTUBE_DISK_HEADROOM_MB

    free_mb = get_free_disk_space_mb(YOUTUBE_AUDIO_DIR)
    if free_mb is None:
        return {"ok": False, "needed_mb": total_needed_mb, "free_mb": None, "error": "Не удалось определить свободное место"}

    if free_mb < total_needed_mb:
        return {
            "ok": False,
            "needed_mb": total_needed_mb,
            "free_mb": free_mb,
            "error": f"Недостаточно места: требуется {total_needed_mb:.1f} МБ, свободно {free_mb:.1f} МБ"
        }

    return {"ok": True, "needed_mb": total_needed_mb, "free_mb": free_mb, "error": None}

def load_youtube_history():
    """Загрузить историю скачиваний из youtube-history.jsonl.
    Возвращает dict {video_id: entry}, последняя запись побеждает.
    Битые строки пропускаются без прерывания."""
    result = {}
    if not os.path.exists(YOUTUBE_HISTORY_FILE):
        return result

    try:
        with open(YOUTUBE_HISTORY_FILE, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    video_id = entry.get("video_id")
                    if video_id:
                        result[video_id] = entry
                except json.JSONDecodeError:
                    log_debug(f"⚠ youtube-history.jsonl:{line_num} — битая строка JSON, пропуск")
    except (OSError, IOError) as e:
        log_debug(f"Ошибка чтения youtube-history.jsonl: {e}")

    return result

def append_youtube_history(entry):
    """Дописать запись в youtube-history.jsonl.
    entry: {"video_id", "url", "title", "duration_sec", "audio_path", "size_mb", "downloaded_at", "batch_id"}
    При превышении YOUTUBE_HISTORY_MAX_LINES удаляются самые старые."""
    try:
        # Дописать новую строку
        with open(YOUTUBE_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Проверить ротацию
        try:
            with open(YOUTUBE_HISTORY_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) > YOUTUBE_HISTORY_MAX_LINES:
                # Оставить хвост
                lines = lines[-YOUTUBE_HISTORY_MAX_LINES:]
                with open(YOUTUBE_HISTORY_FILE, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                log_debug(f"youtube-history.jsonl ротирован ({len(lines)} строк)")
        except (OSError, IOError):
            pass  # Ротация не критична
    except (OSError, IOError) as e:
        log_debug(f"Ошибка записи youtube-history.jsonl: {e}")

def find_cached_download(video_id):
    """Найти кэшированную загрузку по video_id из истории.
    Возвращает entry если audio_path существует и size > 0, иначе None."""
    history = load_youtube_history()
    entry = history.get(video_id)

    if entry and entry.get("audio_path"):
        audio_path = entry["audio_path"]
        try:
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                return entry
        except (OSError, IOError):
            pass

    return None

def youtube_audio_path(video_id, title):
    """Путь для сохранения скачанного аудио: ~/.cache/whisper-dictate/youtube-audio/title_video_id.mp3
    video_id в имени — ключ кэша и защита от коллизий заголовков."""
    safe_title = sanitize_filename(title)
    return os.path.join(YOUTUBE_AUDIO_DIR, f"{safe_title}_{video_id}.mp3")

def cleanup_youtube_cache(max_age_days=YOUTUBE_CACHE_MAX_AGE_DAYS, max_size_gb=YOUTUBE_CACHE_MAX_GB):
    """Удалить самые старые MP3 при превышении возраста или объёма.
    Возвращает число удалённых файлов."""
    import time
    deleted = 0
    try:
        if not os.path.exists(YOUTUBE_AUDIO_DIR):
            return 0

        # Получить все MP3 файлы с их временем модификации и размером
        files = []
        for filename in os.listdir(YOUTUBE_AUDIO_DIR):
            if not filename.endswith(".mp3"):
                continue
            filepath = os.path.join(YOUTUBE_AUDIO_DIR, filename)
            try:
                stat = os.stat(filepath)
                files.append((filepath, stat.st_mtime, stat.st_size))
            except (OSError, IOError):
                continue

        # Сортировать по времени (самые старые в начале)
        files.sort(key=lambda x: x[1])

        now = time.time()
        max_age_sec = max_age_days * 86400
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024

        # Удалить по возрасту
        for filepath, mtime, size in files:
            age_sec = now - mtime
            if age_sec > max_age_sec:
                try:
                    os.unlink(filepath)
                    deleted += 1
                except (OSError, IOError):
                    pass

        # Удалить по размеру (если кэш слишком большой)
        total_size = sum(f[2] for f in files if os.path.exists(f[0]))
        if total_size > max_size_bytes:
            for filepath, mtime, size in files:
                if not os.path.exists(filepath):
                    continue
                if total_size <= max_size_bytes:
                    break
                try:
                    os.unlink(filepath)
                    total_size -= size
                    deleted += 1
                except (OSError, IOError):
                    pass

        if deleted > 0:
            log_debug(f"youtube-cache очищена: удалено {deleted} файлов")
    except Exception as e:
        log_debug(f"Ошибка при очистке youtube-cache: {e}")

    return deleted

def cleanup_orphan_downloads():
    """Удалить осиротевшие .part и .ytdl файлы из YOUTUBE_AUDIO_DIR.
    Вызывается при старте трея после убитого процесса."""
    if not os.path.exists(YOUTUBE_AUDIO_DIR):
        return

    deleted = 0
    try:
        for filename in os.listdir(YOUTUBE_AUDIO_DIR):
            if filename.endswith((".part", ".ytdl")):
                filepath = os.path.join(YOUTUBE_AUDIO_DIR, filename)
                try:
                    os.unlink(filepath)
                    deleted += 1
                except (OSError, IOError):
                    pass

        if deleted > 0:
            log_debug(f"Удалено {deleted} осиротевших загрузок (.part/.ytdl)")
    except Exception as e:
        log_debug(f"Ошибка при очистке осиротевших загрузок: {e}")

def load_quality_config():
    transaction = globals().get("_settings_config_transaction")
    if isinstance(transaction, dict) and "quality" in transaction:
        return json.loads(json.dumps(transaction["quality"]))
    try:
        if os.path.exists(QUALITY_CONFIG_FILE):
            with open(QUALITY_CONFIG_FILE, encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log_debug(f"Ошибка чтения quality-config: {e}")
    return DEFAULT_QUALITY_CONFIG.copy()

def save_quality_config(config):
    transaction = globals().get("_settings_config_transaction")
    if isinstance(transaction, dict):
        transaction["quality"] = json.loads(json.dumps(config))
        return True
    try:
        fd, tmp_path = tempfile.mkstemp(dir=STATE_DIR, suffix=".tmp", prefix="quality-config-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, QUALITY_CONFIG_FILE)
            log_debug(f"Конфиг качества сохранён (strategy={config.get('current_strategy')})")
            return True
        except:
            try: os.unlink(tmp_path)
            except: pass
            raise
    except Exception as e:
        log_debug(f"Ошибка сохранения quality-config: {e}")
        return False

def load_voice_config():
    """Read the canonical daemon configuration; never fall back to a cloud profile."""
    transaction = globals().get("_settings_config_transaction")
    if isinstance(transaction, dict) and "voice" in transaction:
        return json.loads(json.dumps(transaction["voice"]))
    try:
        with open(VOICE_CONFIG_FILE, encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        log_debug(f"Ошибка чтения voice config: {e}")
        return {}

def save_voice_config(value):
    transaction = globals().get("_settings_config_transaction")
    if isinstance(transaction, dict):
        transaction["voice"] = json.loads(json.dumps(value))
        return True
    try:
        os.makedirs(os.path.dirname(VOICE_CONFIG_FILE), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(VOICE_CONFIG_FILE), suffix=".tmp", prefix="config-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, VOICE_CONFIG_FILE)
            return True
        except:
            try: os.unlink(tmp_path)
            except OSError: pass
            raise
    except Exception as e:
        log_debug(f"Ошибка сохранения voice config: {e}")
        return False

_settings_config_transaction = None

# Tray icons are bundled PNGs. Avoid GTK icon-theme lookups here: a missing
# theme icon can make GTK abort while rendering a menu when its cache is full.
_BUNDLED_MENU_ICON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "menu-icons"))
_INSTALLED_MENU_ICON_DIR = os.path.expanduser("~/.local/share/whisper-dictate/menu-icons")
MENU_ICON_DIR = (_INSTALLED_MENU_ICON_DIR if os.path.isdir(_INSTALLED_MENU_ICON_DIR)
                 else _BUNDLED_MENU_ICON_DIR)
MENU_ICON_SIZE = 48
PROVIDER_ICON_SIZE = 48
PROVIDER_ICON_NAMES = {"cloudflare.png", "openrouter.png", "codex.png",
                       "claude.png", "groq.png", "google.png"}

def menu_icon_path(name):
    return os.path.join(MENU_ICON_DIR, f"{name}.png")

def settings_tab_label(title, icon_name, icon_size=32):
    """Build a readable icon-and-title navigation row for settings."""
    compact = icon_size < 32
    tab = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6 if compact else 10)
    tab.set_border_width(4 if compact else 8)
    image = Gtk.Image()
    icon_path = menu_icon_path(icon_name)
    try:
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(icon_path)
        pixbuf = pixbuf.scale_simple(icon_size, icon_size, GdkPixbuf.InterpType.BILINEAR)
        image.set_from_pixbuf(pixbuf)
    except Exception as exc:
        log_debug(f"Не удалось загрузить иконку вкладки {icon_path}: {exc}")
    image.set_size_request(icon_size, icon_size)
    tab.pack_start(image, False, False, 0)
    tab.pack_start(Gtk.Label(label=title), False, False, 0)
    tab.show_all()
    return tab

def integer_range_scale(minimum, maximum, value):
    """Create a discrete numeric control without theme-dependent stepper icons."""
    scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, minimum, maximum, 1)
    scale.set_digits(0)
    scale.set_draw_value(True)
    scale.set_value(value)
    return scale

def integer_retry_control(minimum, maximum, value):
    """Create an editable retry count with a numeric entry and steppers."""
    spin = Gtk.SpinButton.new_with_range(minimum, maximum, 1)
    spin.set_numeric(True)
    spin.set_digits(0)
    spin.set_value(minimum if type(value) is not int else min(maximum, max(minimum, value)))
    spin.set_width_chars(3)
    return spin

_settings_panel_dialog = None
_settings_panel_notebook = None
_settings_panel_pages = {}

def _select_settings_panel_page(tab_name):
    """Select a requested page after the settings dialog has been presented."""
    if (_settings_panel_dialog is None or _settings_panel_notebook is None or
            tab_name not in _settings_panel_pages):
        return GLib.SOURCE_REMOVE
    try:
        page = _settings_panel_pages[tab_name]
        if isinstance(page, tuple):
            outer_page, inner_notebook, inner_page = page
            _settings_panel_notebook.set_current_page(outer_page)
            inner_notebook.set_current_page(inner_page)
        else:
            _settings_panel_notebook.set_current_page(page)
    except Exception as exc:
        log_debug(f"[settings-panel] Не удалось выбрать раздел {tab_name}: {exc}")
    return GLib.SOURCE_REMOVE

def open_settings_panel(tab_name=None):
    """Open the single settings window and optionally select one service tab."""
    global _settings_panel_dialog, _settings_panel_notebook
    try:
        if _settings_panel_dialog is not None:
            try:
                _settings_panel_dialog.present()
                if tab_name in _settings_panel_pages:
                    GLib.idle_add(_select_settings_panel_page, tab_name)
                return
            except Exception:
                _settings_panel_dialog = None
                _settings_panel_notebook = None
        configure_app_settings(initial_tab=tab_name)
    except Exception:
        details = traceback.format_exc()
        try:
            log_debug("[settings-panel] Не удалось открыть единое окно настроек:\n" + details)
        except Exception:
            pass
        try:
            notify_send("Whisper Dictate", "Не удалось открыть окно настроек. Подробности записаны в журнал приложения.")
        except Exception:
            pass

MENU_ICON_ALIASES = {
    "media-playback-start": "play", "document-open": "file",
    "video-x-generic": "video", "edit-clear": "clear",
    "document-edit": "corrector",
    "edit-copy": "copy", "network-server-symbolic": "network",
    "microphone-sensitivity-high": "microphone",
    "microphone-sensitivity-high-symbolic": "microphone",
    "audio-input-microphone": "microphone", "emblem-ok-symbolic": "daemon",
    "process-stop": "stop", "view-refresh": "restart",
    "media-eject": "unload", "accessories-text-editor-symbolic": "corrector",
    "preferences-system-symbolic": "model_services", "office-chart-bar": "stats",
    "document-open-recent": "history", "dialog-error": "error",
    "text-x-log": "logs", "document-properties": "changelog",
    "preferences-desktop-keyboard": "keyboard", "applications-internet": "web",
    "help-browser": "help", "applications-internet-symbolic": "web",
    "dialog-error": "error",
    "network-workgroup-symbolic": "network",
}

def provider_status_icon(name, fallback):
    path = menu_icon_path(name if name in {"cloudflare", "groq", "google", "openrouter", "codex", "claude"} else "remote")
    return path if os.path.isfile(path) else menu_icon_path("provider")

CODEX_STATUS_ICON = provider_status_icon("codex", "applications-internet-symbolic")

ICONS = {
    "idle": menu_icon_path("microphone"),
    "recording": menu_icon_path("recording"),
    "processing_cloudflare": provider_status_icon("cloudflare", "/usr/share/icons/Papirus/24x24/actions/cloud-download.svg"),
    "processing_groq": provider_status_icon("groq", "network-server-symbolic"),
    "processing_google": provider_status_icon("google", "applications-internet-symbolic"),
    "processing_openrouter": provider_status_icon("openrouter", "network-workgroup-symbolic"),
    "processing_codex": CODEX_STATUS_ICON,
    "processing_claude": provider_status_icon("claude", "applications-internet-symbolic"),
    "processing_gpu": menu_icon_path("cuda"),
    "processing_gamepad": menu_icon_path("gamepad"),
    "processing_gigaam": menu_icon_path("gigaam"),
    "processing_cpu": menu_icon_path("cpu"),
    "processing_cpu_flame": menu_icon_path("cpu_flame"),
    "processing_remote": menu_icon_path("remote")
}

BACKEND_STATUS = {
    "recording": ("recording", "🔴"),
    "cloudflare": ("processing_cloudflare", "☁️"),
    "groq": ("processing_groq", "🟠"),
    "google": ("processing_google", "🔵"),
    "openrouter": ("processing_openrouter", "🧭"),
    "codex": ("processing_codex", "🟢"),
    "claude": ("processing_claude", "🟣"),
    "cuda": ("processing_gpu", "🎮"),
    "cpu": ("processing_cpu", "🔥"),
}

def backend_status_icon(state, backend, device="cpu", model=None, selected_backend=None):
    """Return one shared status icon key/label for menu rows and tray state."""
    if state == "recording":
        return BACKEND_STATUS["recording"]
    gigaam_backends = {"gigaam-v3-e2e-rnnt", "gigaam-v3-e2e-ctc-onnx-gpu",
                       "gigaam-v3-ctc-onnx-gpu"}
    if (selected_backend in gigaam_backends or
            isinstance(model, str) and model.lower().startswith("gigaam-v3")):
        local_device = device if device in {"cuda", "cpu"} else None
        if local_device is None:
            local_device = ("cpu" if selected_backend == "gigaam-v3-e2e-rnnt"
                            else "cuda" if selected_backend in gigaam_backends
                            else "cpu")
        if local_device == "cpu":
            return ("processing_cpu_flame", "🔥")
        return ("processing_gamepad", "🎮")
    if backend == "cloudflare" or device == "cloudflare":
        return BACKEND_STATUS["cloudflare"]
    if backend in {"groq", "google", "openrouter"}:
        return BACKEND_STATUS[backend]
    if backend in {"codex", "claude"}:
        return BACKEND_STATUS[backend]
    local_device = backend if backend in {"cuda", "cpu"} else device
    return BACKEND_STATUS.get(local_device, BACKEND_STATUS["cpu"])

def processing_status_label(model, stage, icon_key, icon_label=None):
    """Describe the active model, pipeline stage, and actual backend for UI status."""
    stage_label = {"asr": "распознавание", "transcription": "распознавание",
                   "correction": "коррекция"}.get(stage, "обработка")
    backend_label = {"processing_cloudflare": "Cloudflare",
                     "processing_groq": "Groq",
                     "processing_google": "Google Gemini",
                     "processing_openrouter": "OpenRouter",
                     "processing_codex": "Codex",
                     "processing_claude": "Claude",
                     "processing_remote": "remote provider",
                     "processing_gpu": "локальный GPU",
                     "processing_gamepad": "локальный GPU",
                     "processing_gigaam": "GigaAM",
                     "processing_cpu": "локальный CPU",
                     "processing_cpu_flame": "локальный CPU"}.get(icon_key, "локальный CPU")
    model_label = str(model or "?")[:120]
    return f"{model_label} — {stage_label} · {backend_label}"


def asr_backend_status_suffix(selected_backend, actual_model, error_category):
    """Show an opt-in GigaAM attempt and safe FasterWhisper fallback reason."""
    gigaam_names = {
        "gigaam-v3-e2e-rnnt": "GigaAM-v3 e2e_rnnt",
        "gigaam-v3-e2e-ctc-onnx-gpu": "GigaAM-v3 e2e_ctc",
        "gigaam-v3-ctc-onnx-gpu": "GigaAM-v3 ctc",
    }
    if selected_backend not in gigaam_names or actual_model == gigaam_names[selected_backend]:
        return ""
    reasons = {
        "weights_missing_or_corrupt": "локальные веса отсутствуют или повреждены",
        "dependency_import_failure": "runtime не загрузился",
        "cuda_execution_unavailable": "CUDA provider недоступен",
        "onnx_hash_mismatch": "проверка ONNX файла не прошла",
        "worker_unavailable": "GigaAM worker недоступен",
        "worker_timeout": "тайм-аут запуска GigaAM",
        "audio_decode_timeout": "тайм-аут чтения аудио",
        "audio_decode_failure": "не удалось прочитать аудио",
        "invalid_chunk_settings": "неверные параметры чанков",
        "inference_failure": "ошибка GigaAM inference",
    }
    reason = reasons.get(error_category, "ошибка GigaAM")
    return f" · GigaAM → FasterWhisper fallback ({reason})"

def set_status_row_icon(image, icon_key):
    """Set the real provider/local icon used beside each active task row."""
    source = icon_key if isinstance(icon_key, str) and os.path.isfile(icon_key) else ICONS.get(icon_key, ICONS["processing_cpu"])
    icon_size = (PROVIDER_ICON_SIZE if os.path.basename(source) in PROVIDER_ICON_NAMES
                 else MENU_ICON_SIZE)
    try:
        # new_from_file_at_scale writes an intermediate cache file on this
        # desktop and fails under the user's exhausted quota. Decode PNG into
        # memory, then scale the pixbuf without filesystem writes.
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(source)
        if pixbuf.get_width() != icon_size or pixbuf.get_height() != icon_size:
            pixbuf = pixbuf.scale_simple(icon_size, icon_size, GdkPixbuf.InterpType.BILINEAR)
        image.set_from_pixbuf(pixbuf)
    except Exception as exc:
        log_debug(f"Не удалось загрузить bundled icon {source}: {exc}")
        fallback = GdkPixbuf.Pixbuf.new_from_file(menu_icon_path("provider"))
        fallback = fallback.scale_simple(icon_size, icon_size, GdkPixbuf.InterpType.BILINEAR)
        image.set_from_pixbuf(fallback)
    image.set_pixel_size(icon_size)
    image.set_size_request(icon_size, icon_size)
    image.set_halign(Gtk.Align.CENTER)
    image.set_valign(Gtk.Align.CENTER)

def style_menu_row(item, image):
    """Use one icon cell and single-line label layout for every icon row."""
    width, height = image.get_size_request()
    if width < 1 or height < 1:
        image.set_size_request(MENU_ICON_SIZE, MENU_ICON_SIZE)
    image.set_halign(Gtk.Align.CENTER)
    image.set_valign(Gtk.Align.CENTER)
    label = item.get_child()
    if isinstance(label, Gtk.Label):
        label.set_xalign(0.0)
        label.set_line_wrap(False)
        label.set_single_line_mode(True)
        label.set_ellipsize(Pango.EllipsizeMode.END)
        label.set_max_width_chars(54)

def normalize_menu_layout(menu):
    """Keep labels single-line and icon slots identical through nested menus."""
    for item in menu.get_children():
        if isinstance(item, Gtk.CheckMenuItem):
            child = item.get_child()
            has_icon = (isinstance(child, Gtk.Box) and
                        any(isinstance(part, Gtk.Image) for part in child.get_children()))
            if not has_icon:
                label = (item.get_label() or "").lower()
                icon_key = next((key for key in ("cloudflare", "openrouter", "codex", "claude", "groq", "google")
                                 if key in label), "settings")
                if "qwen" in label or "корректор" in label:
                    icon_key = "corrector"
                elif "gigaam" in label or "whisper" in label or "модель" in label:
                    icon_key = "microphone"
                elif "gpu" in label or "cuda" in label:
                    icon_key = "cuda"
                elif "cpu" in label:
                    icon_key = "cpu"
                iconize_check_menu_item(item, menu_icon_path(icon_key))
        for child in item.get_children():
            if isinstance(child, Gtk.Label):
                child.set_line_wrap(False)
                child.set_single_line_mode(True)
                child.set_ellipsize(Pango.EllipsizeMode.END)
                child.set_max_width_chars(54)
            elif isinstance(child, Gtk.Image):
                width, height = child.get_size_request()
                if width < 1 or height < 1:
                    child.set_size_request(MENU_ICON_SIZE, MENU_ICON_SIZE)
                child.set_halign(Gtk.Align.CENTER)
                child.set_valign(Gtk.Align.CENTER)
            elif isinstance(child, Gtk.Container):
                normalize_widget_layout(child)
        submenu = item.get_submenu() if hasattr(item, "get_submenu") else None
        if submenu is not None:
            normalize_menu_layout(submenu)

def normalize_widget_layout(widget):
    for child in widget.get_children():
        if isinstance(child, Gtk.Label):
            child.set_line_wrap(False)
            child.set_single_line_mode(True)
            child.set_ellipsize(Pango.EllipsizeMode.END)
            child.set_max_width_chars(54)
        elif isinstance(child, Gtk.Image):
            width, height = child.get_size_request()
            if width < 1 or height < 1:
                child.set_size_request(MENU_ICON_SIZE, MENU_ICON_SIZE)
            child.set_halign(Gtk.Align.CENTER)
            child.set_valign(Gtk.Align.CENTER)
        elif isinstance(child, Gtk.Container):
            normalize_widget_layout(child)

def set_menu_label_max_width(widget, max_chars):
    """Adjust nested labels for a dynamic status row without widening menus."""
    if isinstance(widget, Gtk.Label):
        widget.set_max_width_chars(max_chars)
    elif isinstance(widget, Gtk.Container):
        for child in widget.get_children():
            set_menu_label_max_width(child, max_chars)

def provider_menu_item(label, icon_key):
    """Build a service menu item with its bundled brand icon."""
    item = Gtk.ImageMenuItem(label=label)
    image = Gtk.Image()
    set_status_row_icon(image, icon_key)
    item.set_image(image)
    item.set_always_show_image(True)
    style_menu_row(item, image)
    return item

def icon_menu_item(label, icon_key):
    """Create a submenu action with the shared bundled icon treatment."""
    item = Gtk.ImageMenuItem(label=label)
    image = Gtk.Image()
    set_status_row_icon(image, icon_key)
    item.set_image(image)
    item.set_always_show_image(True)
    style_menu_row(item, image)
    return item

def iconize_check_menu_item(item, icon_key):
    """Add the shared icon cell while retaining GTK's native check indicator."""
    current_child = item.get_child()
    if (isinstance(current_child, Gtk.Box) and
            any(isinstance(part, Gtk.Image) for part in current_child.get_children())):
        return
    label_text = item.get_label()
    if current_child is not None:
        item.remove(current_child)
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    image = Gtk.Image()
    set_status_row_icon(image, icon_key)
    label = Gtk.Label(label=label_text, xalign=0)
    row.pack_start(image, False, False, 0)
    row.pack_start(label, True, True, 0)
    item.add(row)
    row.show_all()
    style_menu_row(item, image)

def set_check_menu_item_label(item, text):
    child = item.get_child()
    if isinstance(child, Gtk.Box):
        for part in child.get_children():
            if isinstance(part, Gtk.Label):
                part.set_text(text)
                return
    item.set_label(text)

_primary_menu_icon_widgets = {}

def set_named_menu_icon(image, icon_name):
    """Use a bundled transparent dark glyph, independent of desktop theme."""
    icon_path = menu_icon_path(MENU_ICON_ALIASES.get(icon_name, "settings"))
    set_status_row_icon(image, icon_path)
    image.set_pixel_size(MENU_ICON_SIZE)
    image.set_size_request(MENU_ICON_SIZE, MENU_ICON_SIZE)
    image.set_halign(Gtk.Align.CENTER)
    image.set_valign(Gtk.Align.CENTER)

def primary_menu_item(label, icon_name):
    """Create a main-menu item with a real GTK icon in its shared left slot."""
    item = Gtk.ImageMenuItem(label=label)
    icon_widget = Gtk.Image()
    set_named_menu_icon(icon_widget, icon_name)
    item.set_image(icon_widget)
    item.set_always_show_image(True)
    style_menu_row(item, icon_widget)
    _primary_menu_icon_widgets[id(item)] = icon_widget
    return item

def set_primary_menu_label(item, label, icon_name=None):
    """Update a main-menu row without replacing its native GTK label."""
    item.set_label(label)
    icon_widget = _primary_menu_icon_widgets.get(id(item))
    if icon_widget is not None and icon_name:
        set_named_menu_icon(icon_widget, icon_name)

BATCH_STATUS_LABELS = {
    "queued": "⏳ Ожидание",
    "processing": "Обработка",
    "completed": "✅ Готово",
    "failed": "❌ Ошибка",
    "cancelled": "🚫 Отменено"
}
LANGUAGE_LABELS = {"ru": "Русский", "en": "English", "Auto": "Авто", None: "Авто"}
BATCH_OUTPUT_DIR = "/tmp/whisper-batch"

indicator = AppIndicator3.Indicator.new(
    "whisper-dictate", ICONS["idle"], AppIndicator3.IndicatorCategory.APPLICATION_STATUS)
indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)

def get_daemon_status():
    """Проверить запущен ли демон"""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "whisper-daemon.service"],
            capture_output=True,
            text=True,
            timeout=2
        )
        is_active = result.stdout.strip() == "active"
        status = "active" if is_active else "inactive"
        log_debug(f"[daemon-status] {status}")
        return is_active
    except Exception as e:
        log_debug(f"[daemon-status] Ошибка проверки: {e}")
        return False

def notify_send(title, msg):
    """Показать уведомление"""
    try:
        subprocess.run(["notify-send", title, msg], timeout=2)
    except:
        pass

def start_daemon(_):
    """Запустить демон"""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "start", "whisper-daemon.service"],
            timeout=5, capture_output=True, text=True
        )
        log_debug("[daemon-start] Демон запущен")
        notify_send("Whisper Dictate", "✅ Демон запущен")
        GLib.idle_add(update_daemon_control_items)
    except Exception as e:
        log_debug(f"[daemon-start] Ошибка: {e}")
        notify_send("Ошибка", f"Не удалось запустить демон: {e}")

def stop_daemon(_):
    """Остановить daemon; Qwen выгружается отдельной командой меню."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "stop", "whisper-daemon.service"],
            timeout=5, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "systemctl stop failed")
        log_debug("[daemon-stop] Демон остановлен")
        notify_send("Whisper Dictate", "⏹️ Демон остановлен")
        GLib.idle_add(update_daemon_control_items)
    except Exception as e:
        log_debug(f"[daemon-stop] Ошибка: {e}")
        notify_send("Ошибка", f"Не удалось остановить демон: {e}")

def restart_daemon(_):
    """Перезапустить демон"""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "restart", "whisper-daemon.service"],
            timeout=5, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "systemctl restart failed")
        log_debug("[daemon-restart] Демон перезапущен")
        notify_send("Whisper Dictate", "🔄 Демон перезапущен")
        GLib.idle_add(update_daemon_control_items)
    except Exception as e:
        log_debug(f"[daemon-restart] Ошибка: {e}")
        notify_send("Ошибка", f"Не удалось перезапустить демон: {e}")

def daemon_model_control(action, scope=None, expected_backend=None, timeout=3):
    """Send a transcript-free local control command to the daemon."""
    request = {"action": action}
    if scope is not None:
        request["scope"] = scope
    if expected_backend is not None:
        request["expected_backend"] = expected_backend
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(timeout)
        client.connect(os.path.join(STATE_DIR, "daemon.sock"))
        client.sendall(json.dumps(request).encode("utf-8") + b"\n")
        response = b""
        while not response.endswith(b"\n"):
            part = client.recv(4096)
            if not part:
                break
            response += part
        result = json.loads(response.decode("utf-8"))
        return result if isinstance(result, dict) else None
    except Exception as error:
        log_debug(f"[model-control] {action} unavailable: {error}")
        return None
    finally:
        client.close()

def _finish_model_control(result, action, scope):
    item = globals().get("models_unload_item")
    if item is not None:
        item.set_sensitive(True)
    if result and result.get("status") == "ok":
        labels = {"all": "Все локальные модели выгружены",
                  "asr": "Локальная модель распознавания выгружена",
                  "corrector": "Локальные модели корректора выгружены"}
        if not result.get("unchanged"):
            message = ("Выбранная локальная модель загружена" if action == "prepare_selected_asr"
                       else labels.get(scope, "Команда выполнена"))
            notify_send("Whisper Dictate", message)
    else:
        notify_send("Whisper Dictate", "Не удалось выгрузить модели; проверьте демон и Ollama")
    GLib.idle_add(update_model_unload_state)
    GLib.idle_add(update_corrector_unload_state)
    return False

def start_model_control(action, scope="all", expected_backend=None, timeout=3600):
    """Run potentially queued unload/preload work outside the GTK thread."""
    import threading
    item = globals().get("models_unload_item")
    if action == "unload_models" and scope == "all" and item is not None:
        item.set_sensitive(False)
    def run():
        result = daemon_model_control(action, scope, expected_backend, timeout)
        GLib.idle_add(_finish_model_control, result, action, scope)
    threading.Thread(target=run, name="whisper-model-control", daemon=True).start()

def prepare_selected_asr():
    start_model_control("prepare_selected_asr", scope="asr", timeout=600)

def _prepare_asr_after_settings_save():
    """Defer model loading while settings are still pending confirmation."""
    transaction = globals().get("_settings_config_transaction")
    if isinstance(transaction, dict):
        transaction["prepare_asr"] = True
        return
    prepare_selected_asr()

def unload_corrector_model(_):
    """Выгрузить локальные модели корректора, выбранные в цепочке."""
    start_model_control("unload_models", scope="corrector", timeout=60)

def unload_local_asr(_widget=None, expected_backend=None):
    start_model_control("unload_models", scope="asr",
                        expected_backend=expected_backend, timeout=3600)

def update_model_unload_state(*_args):
    """Disable the tray action only when every managed local model is unloaded."""
    result = daemon_model_control("models_status", timeout=1.5)
    item = globals().get("models_unload_item")
    whisper_button = globals().get("whisper_unload_settings_button")
    gigaam_button = globals().get("gigaam_unload_settings_button")
    status_label = globals().get("asr_residency_status_label")
    if result and result.get("status") == "ok":
        loaded_asr = result.get("asr_loaded")
        qwen_models = result.get("corrector_models")
        qwen_loaded = bool(qwen_models) if isinstance(qwen_models, list) else None
        all_known_unloaded = loaded_asr is False and qwen_loaded is False
        if item is not None:
            item.set_sensitive(not all_known_unloaded)
            item.set_tooltip_text(
                "Выгрузить все модели Whisper Dictate из RAM/VRAM" if not all_known_unloaded
                else "Все модели Whisper Dictate уже выгружены")
        backend = result.get("asr_backend")
        if whisper_button is not None:
            whisper_button.set_sensitive(loaded_asr is not False and backend in {None, "faster-whisper"})
        if gigaam_button is not None:
            gigaam_button.set_sensitive(loaded_asr is not False and
                                         (backend is None or str(backend).startswith("gigaam-")))
        if status_label is not None:
            asr_name = result.get("asr_model") if loaded_asr else "нет"
            qwen_text = ", ".join(qwen_models) if isinstance(qwen_models, list) and qwen_models else (
                "статус неизвестен" if qwen_models is None else "нет")
            status_label.set_text(f"В памяти: ASR — {asr_name}; корректор — {qwen_text}")
    else:
        if item is not None:
            item.set_sensitive(True)
            item.set_tooltip_text("Не удалось проверить residency; нажмите, чтобы освободить память")
        if whisper_button is not None:
            whisper_button.set_sensitive(True)
        if gigaam_button is not None:
            gigaam_button.set_sensitive(True)
        if status_label is not None:
            status_label.set_text("Статус локальных моделей недоступен")
    return False

def get_loaded_corrector_model(model):
    """Вернуть True/False по Ollama API, None если статус недоступен."""
    try:
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/ps", headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=0.6) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("models", []) if isinstance(payload, dict) else []
        return any(isinstance(item, dict) and
                   (item.get("name") == model or item.get("model") == model)
                   for item in models)
    except Exception as e:
        log_debug(f"[corrector-status] Не удалось проверить Ollama: {e}")
        return None

def update_corrector_unload_state(*_args):
    """Обновить состояние всех моделей в локальной correction chain."""
    corrector = load_voice_config().get("corrector", {})
    model = corrector.get("model")
    residency = daemon_model_control("models_status", timeout=1.5)
    loaded_models = (residency.get("corrector_models")
                     if residency and residency.get("status") == "ok" else None)
    loaded = bool(loaded_models) if isinstance(loaded_models, list) else (
        get_loaded_corrector_model(model) if isinstance(model, str) and model else None)
    if loaded is True:
        names = ", ".join(loaded_models) if isinstance(loaded_models, list) else str(model)
        status_text = f"Qwen: загружен в RAM/VRAM ({names})"
        menu_label = "Выгрузить Qwen"
        tooltip = "Выгрузить загруженные локальные модели корректора из RAM/VRAM"
    elif loaded is False:
        status_text = "Qwen: выгружен из RAM/VRAM"
        menu_label = "Выгрузить Qwen"
        tooltip = "Кнопка станет доступна, когда выбранная модель будет загружена"
    else:
        status_text = "Qwen: статус загрузки недоступен"
        menu_label = "Выгрузить Qwen"
        tooltip = "Не удалось получить состояние модели от Ollama"
    status_label = globals().get("corrector_loaded_status_label")
    if status_label is not None:
        status_label.set_text(status_text)
    menu_item = globals().get("corrector_unload_item")
    if menu_item is not None:
        menu_item.set_label(menu_label)
        menu_item.set_tooltip_text(tooltip)
        menu_item.set_sensitive(loaded is True)
    settings_button = globals().get("corrector_unload_settings_button")
    if settings_button is not None:
        settings_button.set_sensitive(loaded is True)
        settings_button.set_tooltip_text(tooltip)
    return False

def _read_log_excerpt(log_kind, max_lines=300, max_bytes_per_file=512 * 1024):
    """Read a bounded tail of filtered local logs for the settings viewer."""
    log_files = ("tray.log", "daemon.log", "autopaste.log")
    error_keywords = ("✗", "Ошибка", "Error", "Traceback", "Failed", "Exception")
    work_keywords = ("[daemon-status]", "Трей запущен", "Иконка изменена", "✓")
    lines = deque(maxlen=max_lines)
    for filename in log_files:
        path = os.path.join(STATE_DIR, filename)
        try:
            with open(path, "rb") as source:
                source.seek(0, os.SEEK_END)
                size = source.tell()
                start = max(0, size - max_bytes_per_file)
                source.seek(start)
                if start:
                    source.readline()
                excerpt = source.read().decode("utf-8", errors="replace")
            for line in excerpt.splitlines():
                matched = (any(word in line for word in error_keywords)
                           if log_kind == "errors" else
                           any(word in line for word in work_keywords)
                           and not any(word in line for word in error_keywords))
                if matched:
                    lines.append(f"[{filename}] {line.rstrip()}")
        except OSError:
            continue
    if not lines:
        return ""
    return "\n".join(lines)

def get_error_logs():
    """Return the latest bounded excerpt of error log lines."""
    return _read_log_excerpt("errors").splitlines()

def get_work_logs():
    """Return the latest bounded excerpt of non-error work log lines."""
    return _read_log_excerpt("work").splitlines()

def open_error_logs(_):
    """Открыть логи ошибок в текстовом редакторе"""
    try:
        error_lines = get_error_logs()

        if not error_lines:
            show_error_dialog("Логи ошибок", "Ошибок не найдено ✓")
            return

        # Сохранить в временный файл
        view_file = "/tmp/whisper-error-logs.txt"
        with open(view_file, "w", encoding="utf-8") as f:
            f.write("\n".join(error_lines))

        # Открыть в редакторе
        for editor in ["kate", "gedit", "xed", "mousepad", "pluma"]:
            try:
                subprocess.Popen([editor, view_file])
                log_debug("Логи ошибок открыты в " + editor)
                return
            except FileNotFoundError:
                continue

        subprocess.Popen(["xdg-open", view_file])
        log_debug("Логи ошибок открыты через xdg-open")
    except Exception as e:
        log_debug(f"Ошибка при открытии логов ошибок: {e}")

def open_work_logs(_):
    """Открыть логи работы в текстовом редакторе"""
    try:
        work_lines = get_work_logs()

        if not work_lines:
            show_error_dialog("Логи работы", "Нет логов работы")
            return

        # Сохранить в временный файл
        view_file = "/tmp/whisper-work-logs.txt"
        with open(view_file, "w", encoding="utf-8") as f:
            f.write("\n".join(work_lines))

        # Открыть в редакторе
        for editor in ["kate", "gedit", "xed", "mousepad", "pluma"]:
            try:
                subprocess.Popen([editor, view_file])
                log_debug("Логи работы открыты в " + editor)
                return
            except FileNotFoundError:
                continue

        subprocess.Popen(["xdg-open", view_file])
        log_debug("Логи работы открыты через xdg-open")
    except Exception as e:
        log_debug(f"Ошибка при открытии логов работы: {e}")

def open_history(_):
    view = "/tmp/whisper-history-view.txt"
    try:
        result = subprocess.run(
            ["bash", "-c",
             "jq -s -r 'reverse | .[] | \"\\(.iso) [\\(.model)/\\(.mode)/\\(.device)] "
             "\\(.duration_ms)ms: \\(.text)\"' " + HISTORY_FILE + " > " + view + " 2>/dev/null"],
            timeout=5, capture_output=True)
        log_debug(f"История создана: {view}")
        for editor in ["kate", "gedit", "xed", "mousepad", "pluma"]:
            try:
                subprocess.Popen([editor, view])
                log_debug(f"История открыта в {editor}")
                return
            except FileNotFoundError:
                continue
        subprocess.Popen(["xdg-open", view])
        log_debug("История открыта через xdg-open")
    except Exception as e:
        log_debug(f"Ошибка при открытии истории: {e}")

def get_history_preview_text(max_records=250, max_bytes=2 * 1024 * 1024):
    """Build a bounded, copyable preview of recent transcription history."""
    entries = deque(maxlen=max_records)
    try:
        with open(HISTORY_FILE, "rb") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            start = max(0, size - max_bytes)
            source.seek(start)
            if start:
                source.readline()
            excerpt = source.read().decode("utf-8", errors="replace")
    except OSError:
        return "История транскрибации пока пуста или недоступна."
    for raw_line in excerpt.splitlines():
        try:
            entry = json.loads(raw_line)
        except (ValueError, TypeError):
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    if not entries:
        return "История транскрибации пока пуста."
    lines = [f"Последние записи: {len(entries)} (в пределах последних 2 MiB журнала).", ""]
    for entry in reversed(entries):
        date = entry.get("iso", "[дата неизвестна]")
        model = entry.get("model", "[модель неизвестна]")
        mode = entry.get("mode", "[режим неизвестен]")
        device = entry.get("device", "[устройство неизвестно]")
        duration = entry.get("duration_ms")
        duration_text = f"{duration} ms" if isinstance(duration, (int, float)) else "длительность неизвестна"
        transcript = entry.get("text", "")
        if not isinstance(transcript, str):
            transcript = ""
        lines.append(f"{date} [{model}/{mode}/{device}] {duration_text}")
        lines.append(transcript)
        lines.append("")
    return "\n".join(lines).rstrip()

def read_installed_document(filename):
    """Read an installed user-facing Markdown document for the inline viewer."""
    path = os.path.join(os.path.expanduser("~/.local/share/doc/whisper-dictate"), filename)
    try:
        with open(path, encoding="utf-8") as source:
            return source.read()
    except OSError as exc:
        return f"Не удалось прочитать документ {filename}: {exc}"

def get_required_vram(model, quality):
    """Расчёт требуемой VRAM для модели и качества"""
    vram_base = {
        "auto": 2.5,
        "large-v3": 4.5,
        "medium": 1.5,
        "small": 0.5
    }

    vram_multiplier = {
        "energy": 0.4,
        "normal": 1.0,
        "good": 1.2,
        "maximum": 1.5
    }

    return vram_base.get(model, 4.5) * vram_multiplier.get(quality, 1.0)

def show_info_dialog(title, message):
    """Показать информационный диалог"""
    dialog = Gtk.MessageDialog(
        message_type=Gtk.MessageType.INFO,
        buttons=Gtk.ButtonsType.OK,
        text=title
    )
    dialog.format_secondary_text(message)
    dialog.run()
    dialog.destroy()

def show_error_dialog(title, message):
    """Показать диалог ошибки"""
    dialog = Gtk.MessageDialog(
        message_type=Gtk.MessageType.ERROR,
        buttons=Gtk.ButtonsType.OK,
        text=title
    )
    dialog.format_secondary_text(message)
    dialog.run()
    dialog.destroy()

def show_warning_dialog(title, message):
    """Показать диалог предупреждения с ответом OK/CANCEL"""
    dialog = Gtk.MessageDialog(
        message_type=Gtk.MessageType.WARNING,
        buttons=Gtk.ButtonsType.OK_CANCEL,
        text=title
    )
    dialog.format_secondary_text(message)
    response = dialog.run()
    dialog.destroy()
    return response == Gtk.ResponseType.OK

def format_size(bytes_size):
    """Форматировать размер в MB или GB"""
    mb = bytes_size / (1024 * 1024)
    if mb < 1024:
        return f"{mb:.1f} MB"
    return f"{mb / 1024:.1f} GB"

def format_duration(seconds):
    """Форматировать длительность в MM:SS или HH:MM:SS"""
    if seconds is None:
        return "?"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"

def get_file_size_mb(file_path):
    """Получить размер файла в МБ"""
    try:
        if os.path.exists(file_path):
            return os.path.getsize(file_path) / (1024 * 1024)
    except (OSError, IOError):
        pass
    return None

def get_file_duration(file_path):
    """Получить длительность аудио/видео файла в секундах через ffprobe"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1:nokey=1",
             file_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return None

def is_supported_file(file_path):
    """Проверить что файл имеет поддерживаемый формат"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in AUDIO_EXTENSIONS or ext in VIDEO_EXTENSIONS

def is_video_file(file_path):
    """Проверить что файл — видео, по расширению"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in VIDEO_EXTENSIONS

def validate_audio_file(file_path):
    """Проверить аудио/видео файл через ffprobe: длительность, кодек, наличие аудиодорожки.

    Возвращает dict: valid, duration, codec, format, has_audio, error, reason.
    reason ∈ {None, "ffprobe_missing", "corrupted", "no_audio"} — нужен вызывающему
    коду, чтобы отличить "видео без звука" (warning) от "файл повреждён" (skip+error).
    """
    result = {"valid": False, "duration": None, "codec": None, "format": None,
              "has_audio": False, "error": None, "reason": None}
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", file_path],
            capture_output=True, text=True, timeout=10
        )
    except FileNotFoundError:
        result["error"] = "ffprobe не установлен"
        result["reason"] = "ffprobe_missing"
        return result
    except subprocess.TimeoutExpired:
        result["error"] = "Тайм-аут проверки файла (возможно, файл повреждён)"
        result["reason"] = "corrupted"
        return result

    if probe.returncode != 0 or not probe.stdout.strip():
        result["error"] = probe.stderr.strip() or "Файл повреждён или не читается"
        result["reason"] = "corrupted"
        return result

    try:
        data = json.loads(probe.stdout)
    except json.JSONDecodeError:
        result["error"] = "Не удалось разобрать метаданные файла"
        result["reason"] = "corrupted"
        return result

    fmt = data.get("format", {})
    result["format"] = fmt.get("format_name")
    duration = fmt.get("duration")
    result["duration"] = float(duration) if duration else None

    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        result["error"] = "В файле нет аудиодорожки"
        result["reason"] = "no_audio"
        return result

    result["has_audio"] = True
    result["codec"] = audio_streams[0].get("codec_name")

    if not result["duration"]:
        result["error"] = "Не удалось определить длительность (файл повреждён?)"
        result["reason"] = "corrupted"
        return result

    result["valid"] = True
    return result

def extract_audio_from_video(video_path, output_dir):
    """Извлечь аудиодорожку из видео в MP3 через ffmpeg. Возвращает dict: success, audio_path, error"""
    if not shutil.which("ffmpeg"):
        return {"success": False, "audio_path": None, "error": "ffmpeg не установлен"}

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{base_name}.mp3")

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path],
            capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "audio_path": None, "error": "Тайм-аут извлечения аудио"}

    if result.returncode != 0 or not os.path.exists(audio_path):
        stderr_lines = result.stderr.strip().splitlines()
        detail = stderr_lines[-1] if stderr_lines else "неизвестная ошибка"
        return {"success": False, "audio_path": None, "error": f"Ошибка ffmpeg: {detail}"}

    log_debug(f"🎬 Извлечена аудиодорожка: {video_path} → {audio_path}")
    return {"success": True, "audio_path": audio_path, "error": None}

def show_batch_confirmation_dialog(file_list, quality_config):
    """Показать диалог подтверждения перед batch транскрибацией.

    file_list — список dict: {"name", "original_path", "audio_path", "duration"},
    подготовленный в on_transcribe_file_clicked() после валидации/извлечения аудио.
    Возвращает True если пользователь нажал "Транскрибировать", иначе False.
    """
    model_options = quality_config.get("model_options", {})
    current_model = quality_config.get("current_model", "large-v3")
    auto_mode = quality_config.get("auto_mode", False)
    # В auto-режиме реальная модель выбирается во время обработки по свободной VRAM,
    # поэтому здесь оцениваем по large-v3 как по умолчанию для показа времени/VRAM.
    estimate_model = current_model if current_model in model_options else "large-v3"
    model_info = model_options.get(estimate_model, {})
    model_display = model_info.get("name", estimate_model)
    speed_relative = model_info.get("speed_relative", 1.0) or 1.0
    vram_gb = model_info.get("vram_gb", 4.5)

    current_strategy = quality_config.get("current_strategy", "normal")
    strategy_params = quality_config.get("strategies", {}).get(current_strategy, {})
    strategy_name = strategy_params.get("name", current_strategy)
    language = strategy_params.get("language", "ru")
    vad_threshold = strategy_params.get("vad_threshold", 0.5)

    language_labels = {"ru": "Русский", "en": "English", "Auto": "Авто"}
    language_display = language_labels.get(language, language)

    vad_labels = {None: "Выкл", 0.3: "Низкий (0.3)", 0.5: "Средний (0.5)", 0.7: "Высокий (0.7)"}
    vad_display = vad_labels.get(vad_threshold, str(vad_threshold))

    total_duration_sec = sum(f.get("duration") or 0 for f in file_list)
    approx_time_sec = total_duration_sec / speed_relative
    # GPU обрабатывает 1 файл за раз даже при параллельной очереди из нескольких файлов
    vram_needed = vram_gb

    file_lines = []
    for f in file_list:
        size_mb = get_file_size_mb(f.get("original_path", ""))
        size_str = f"{size_mb:.0f} MB" if size_mb else "? MB"
        duration_str = format_duration(f.get("duration"))
        name_escaped = GLib.markup_escape_text(f.get("name", "?"))
        file_lines.append(f"• {name_escaped}  ({size_str}, {duration_str})")

    files_markup = "<b>Выбранные файлы ({}):</b>\n{}".format(len(file_list), "\n".join(file_lines))

    model_line = (f"Модель: Авто (оценка по {model_display})" if auto_mode
                  else f"Модель: {model_display} ({vram_gb}GB)")

    settings_markup = (
        "\n\n<b>Текущие настройки:</b>\n"
        f"{model_line}\n"
        f"Стратегия: {strategy_name}\n"
        f"Язык: {language_display}\n"
        f"VAD: {vad_display}"
    )

    estimate_markup = (
        "\n\n<b>Оценка обработки:</b>\n"
        f"Общая длительность: {format_duration(total_duration_sec)}\n"
        f"Приблизительное время: ~{format_duration(approx_time_sec)}\n"
        f"Требуется VRAM: {vram_needed}GB"
    )

    dialog = Gtk.Dialog(title="Подтверждение batch транскрибации", transient_for=None, flags=0)
    dialog.add_buttons("Отмена", Gtk.ResponseType.CANCEL,
                       "Транскрибировать", Gtk.ResponseType.OK)
    dialog.set_default_size(420, 380)

    content_area = dialog.get_content_area()
    content_area.set_spacing(8)
    content_area.set_border_width(12)

    scrolled = Gtk.ScrolledWindow()
    scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    scrolled.set_min_content_height(150)

    files_label = Gtk.Label()
    files_label.set_markup(files_markup)
    files_label.set_xalign(0)
    files_label.set_line_wrap(True)
    scrolled.add(files_label)
    content_area.pack_start(scrolled, True, True, 0)

    info_label = Gtk.Label()
    info_label.set_markup(settings_markup + estimate_markup)
    info_label.set_xalign(0)
    info_label.set_line_wrap(True)
    content_area.pack_start(info_label, False, False, 0)

    content_area.show_all()

    response = dialog.run()
    dialog.destroy()

    return response == Gtk.ResponseType.OK

def show_batch_file_chooser():
    """Диалог выбора множества файлов с отображением метаданных"""
    dialog = Gtk.FileChooserDialog(
        title="Выбрать файлы для транскрибации",
        action=Gtk.FileChooserAction.OPEN,
        buttons=(
            Gtk.STOCK_OK, Gtk.ResponseType.OK,
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL
        )
    )
    dialog.set_select_multiple(True)

    audio_filter = Gtk.FileFilter()
    audio_filter.set_name("Audio files (common formats)")
    for extension in sorted(AUDIO_EXTENSIONS):
        audio_filter.add_pattern(f"*{extension}")
    dialog.add_filter(audio_filter)

    video_filter = Gtk.FileFilter()
    video_filter.set_name("Video files (common formats)")
    for extension in sorted(VIDEO_EXTENSIONS):
        video_filter.add_pattern(f"*{extension}")
    dialog.add_filter(video_filter)

    all_filter = Gtk.FileFilter()
    all_filter.set_name("All files")
    all_filter.add_pattern("*")
    dialog.add_filter(all_filter)

    dialog.show_all()
    response = dialog.run()

    if response == Gtk.ResponseType.OK:
        files = dialog.get_filenames()
        result = [f for f in files if is_supported_file(f)]
        if len(result) != len(files):
            log_debug(f"Отфильтровано {len(files) - len(result)} файлов неподдерживаемого формата")
    else:
        result = []

    dialog.destroy()
    return result

def show_simple_file_chooser():
    """Простой диалог выбора файла без лишних диалогов"""
    dialog = Gtk.FileChooserDialog(
        title="Выбрать файл аудио",
        action=Gtk.FileChooserAction.OPEN,
        buttons=(
            Gtk.STOCK_OK, Gtk.ResponseType.OK,
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL
        )
    )

    audio_filter = Gtk.FileFilter()
    audio_filter.set_name("Audio files (common formats)")
    for extension in sorted(AUDIO_EXTENSIONS):
        audio_filter.add_pattern(f"*{extension}")
    dialog.add_filter(audio_filter)

    all_filter = Gtk.FileFilter()
    all_filter.set_name("All files")
    all_filter.add_pattern("*")
    dialog.add_filter(all_filter)

    dialog.show_all()
    response = dialog.run()

    if response == Gtk.ResponseType.OK:
        result = dialog.get_filename()
    else:
        result = None

    dialog.destroy()
    return result

def on_quick_transcribe_clicked(_):
    """Быстрая транскрибация текущими настройками (без диалогов)"""
    try:
        subprocess.Popen([os.path.expanduser("~/.local/bin/whisper/whisper-dictate-v2")])
        log_debug("Быстрая транскрибация запущена (текущие настройки)")
    except Exception as e:
        log_debug(f"✗ Ошибка запуска быстрой транскрибации: {e}")
        show_error_dialog("Ошибка", f"Не удалось запустить транскрибацию: {e}")

_manual_correction_status = None
_manual_correction_daemon_seen = False
_manual_correction_started_at = None
MAX_MANUAL_CORRECTION_CHARS = 12000
MAX_MANUAL_CLIPBOARD_BYTES = MAX_MANUAL_CORRECTION_CHARS * 4

def _set_manual_correction_status(status):
    """Expose manual correction progress through the tray status rows."""
    global _manual_correction_status, _manual_correction_daemon_seen
    global _manual_correction_started_at
    if status != _manual_correction_status:
        if status is None or status in {
                "Ручной корректор — запуск",
                "Ручной корректор — чтение буфера",
                "Ручной корректор — чтение Wayland clipboard"}:
            _manual_correction_daemon_seen = False
    if status is None:
        _manual_correction_started_at = None
    _manual_correction_status = status
    if "_last_signature" in globals():
        globals()["_last_signature"] = None


def _read_manual_correction_progress():
    """Read validated provider/model labels only; never read transcript data."""
    try:
        with open(MANUAL_CORRECTION_STATUS_FILE, encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("state") not in {
            "waiting", "running", "cancelling"}:
        return None
    provider = value.get("provider")
    provider_labels = {"local": "Ollama", "codex": "Codex", "claude": "Claude",
                       "cloudflare": "Cloudflare", "groq": "Groq",
                       "google": "Google Gemini", "openrouter": "OpenRouter"}
    service = provider_labels.get(provider)
    if value["state"] == "waiting":
        return "Ручной корректор — ожидание очереди"
    if value["state"] == "cancelling":
        return "Ручной корректор — отмена запроса"
    if not service:
        return "Ручной корректор — запуск сервиса"
    model = value.get("model")
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9@_.:/+\-]{1,160}", model):
        model = "модель не определена"
    attempt = value.get("attempt")
    attempt_label = (f" · попытка {attempt}" if type(attempt) is int and
                     1 <= attempt <= 99 else "")
    return f"Ручной корректор — {service} · {model}{attempt_label}"

def _read_manual_correction_started_at():
    """Read only the stable start timestamp from daemon status metadata."""
    try:
        with open(MANUAL_CORRECTION_STATUS_FILE, encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    started_at = value.get("started_at")
    return (float(started_at) if isinstance(started_at, (int, float)) and
            not isinstance(started_at, bool) and started_at > 0 else None)

def _job_started_order(job_key, job):
    """Derive stable chronological order from the job id, then metadata."""
    stem = os.path.splitext(os.path.basename(str(job_key)))[0]
    if stem.isdigit():
        value = int(stem)
        return value / 1_000_000_000 if value > 1_000_000_000_000 else float(value)
    started = job.get("started_ts")
    if isinstance(started, (int, float)) and not isinstance(started, bool):
        return float(started)
    if isinstance(started, str) and started.isdigit():
        return float(started)
    return float("inf")


def _manual_correction_result_label(result):
    provider_labels = {"local": "Ollama", "local_fallback": "Ollama",
                       "codex": "Codex", "claude": "Claude",
                       "cloudflare": "Cloudflare", "groq": "Groq",
                       "google": "Google Gemini", "openrouter": "OpenRouter"}
    provider = provider_labels.get(result.get("provider") or result.get("backend"),
                                  "корректор")
    model = result.get("model")
    if isinstance(model, str) and re.fullmatch(r"[A-Za-z0-9@_.:/+\-]{1,160}", model):
        return f"{provider} · {model}"
    return provider

def _manual_correction_failure_summary(result):
    """Format only sanitized provider/model/error metadata; never include text."""
    provider_labels = {"local": "Ollama", "codex": "Codex", "claude": "Claude",
                       "cloudflare": "Cloudflare", "groq": "Groq",
                       "google": "Google", "openrouter": "OpenRouter"}
    failures = []
    attempts = result.get("attempts")
    if not isinstance(attempts, list):
        return ""
    for attempt in attempts:
        if not isinstance(attempt, dict) or attempt.get("success") is not False:
            continue
        provider = provider_labels.get(attempt.get("provider"))
        category = attempt.get("error_category")
        if not provider or not isinstance(category, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{1,80}", category):
            continue
        model = attempt.get("model")
        if isinstance(model, str) and re.fullmatch(r"[A-Za-z0-9@_.:/+\-]{1,160}", model):
            provider += f" · {model}"
        failures.append(f"{provider}: {category}")
    if not failures:
        return ""
    return " Причины: " + "; ".join(failures[-3:]) + "."

def _manual_correction_finished(result):
    _set_manual_correction_status(None)
    if result.get("error_code") == "cancelled_by_user":
        notify_send("Ручной корректор", "Запрос отменён; очередь коррекции освобождена.")
        return
    if result.get("status") == "ok" and isinstance(result.get("text"), str):
        details = _manual_correction_result_label(result)
        newer_live_result = False
        started_at = result.get("started_at")
        if isinstance(started_at, (int, float)) and not isinstance(started_at, bool):
            try:
                with open(LATEST_LIVE_ASR_AT_FILE, encoding="ascii") as stream:
                    latest_live_asr_at = float(stream.read().strip())
                newer_live_result = latest_live_asr_at > started_at
            except (OSError, ValueError):
                pass
        if result.get("copy_to_clipboard") is False or newer_live_result:
            notify_send("Ручной корректор завершён",
                        f"{details}. Буфер оставлен новой диктовке, завершившейся после запуска коррекции.")
            return
        if result.get("fallback"):
            if copy_text_to_clipboard(result["text"]):
                notify_send("Ручной корректор: исходный текст сохранён",
                            f"{details}. Коррекция не завершилась; исходный текст скопирован без изменений."
                            + _manual_correction_failure_summary(result))
            else:
                notify_send("Ручной корректор: исходный текст сохранён",
                            f"{details}. Коррекция не завершилась; не удалось обновить буфер обмена.")
            return
        if copy_text_to_clipboard(result["text"]):
            notify_send("Ручная коррекция завершена",
                        f"{details}. Результат скопирован в буфер обмена.")
        else:
            notify_send("Ручная коррекция завершена",
                        f"{details}. Не удалось записать результат в буфер обмена.")
    else:
        notify_send("Ручная коррекция не завершена",
                    "Проверьте подключение daemon и настройки сервисов коррекции.")

def _read_wayland_clipboard_text():
    """Read only a known plain-text Wayland target; never log its contents."""
    try:
        listed = subprocess.run(
            ["wl-paste", "--list-types"], capture_output=True,
            timeout=2, check=True)
        targets = listed.stdout.decode("ascii", errors="ignore").splitlines()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None, "unavailable"
    target = next((item for item in ("text/plain;charset=utf-8", "text/plain")
                   if item in targets), None)
    if target is None:
        return None, "empty"

    try:
        process = subprocess.Popen(
            ["wl-paste", "--no-newline", "--type", target],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if process.stdout is None:
            process.kill()
            process.wait(timeout=1)
            return None, "unavailable"
        payload = bytearray()
        deadline = time.monotonic() + 5
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                process.kill()
                process.wait(timeout=1)
                return None, "unavailable"
            chunk = os.read(process.stdout.fileno(),
                            min(4096, MAX_MANUAL_CLIPBOARD_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > MAX_MANUAL_CLIPBOARD_BYTES:
                process.kill()
                process.wait(timeout=1)
                return None, "too_long"
        if process.wait(timeout=1) != 0:
            return None, "unavailable"
        return payload.decode("utf-8"), None
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        try:
            process.kill()
            process.wait(timeout=1)
        except (OSError, subprocess.SubprocessError, UnboundLocalError):
            pass
        return None, "unavailable"

def _manual_clipboard_text_ready(text, error):
    """Continue the manual correction on the GTK thread after clipboard read."""
    if error == "too_long":
        _set_manual_correction_status(None)
        show_error_dialog(
            "Текст слишком длинный",
            "Для одной коррекции доступно не более 12 000 символов.")
        return False
    if not isinstance(text, str) or not text.strip():
        _set_manual_correction_status(None)
        if error == "unavailable":
            show_error_dialog(
                "Не удалось прочитать буфер",
                "Скопируйте текст повторно и попробуйте ещё раз.")
        else:
            show_error_dialog(
                "В буфере нет текста",
                "Скопируйте текст и повторите запуск корректора.")
        return False
    if len(text) > MAX_MANUAL_CORRECTION_CHARS:
        _set_manual_correction_status(None)
        show_error_dialog(
            "Текст слишком длинный",
            "Для одной коррекции доступно не более 12 000 символов.")
        return False
    _set_manual_correction_status("Ручной корректор — запуск")
    notify_send("Whisper Dictate", "Ручной корректор запущен")
    threading.Thread(target=_send_manual_correction, args=(text,), daemon=True).start()
    return False

def _read_wayland_clipboard_worker():
    """Fetch the compositor clipboard off the GTK thread, returning no text in logs."""
    text, error = _read_wayland_clipboard_text()
    GLib.idle_add(_manual_clipboard_text_ready, text, error)

def _send_manual_correction(text):
    request = json.dumps({"action": "manual_correct_text", "text": text},
                         ensure_ascii=False).encode("utf-8") + b"\n"
    response = {}
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(600)
        client.connect(os.path.expanduser("~/.cache/whisper-dictate/daemon.sock"))
        client.sendall(request)
        chunks = []
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        client.close()
        response = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        response = {"status": "error", "error_code": "daemon_unavailable"}
    GLib.idle_add(_manual_correction_finished, response)

def on_manual_correct_clicked(_item=None):
    """Correct clipboard text explicitly; never log or persist its contents."""
    global _manual_correction_started_at
    log_debug("[manual-correction] Action activated")
    if _manual_correction_status is not None:
        notify_send("Whisper Dictate", "Ручная коррекция уже выполняется.")
        return
    _manual_correction_started_at = time.time()
    # GTK's Wayland clipboard request can remain pending if the compositor does
    # not complete the text offer. Use the bounded wl-paste worker directly so
    # the request always reaches a completion callback and cannot block progress.
    if os.getenv("WAYLAND_DISPLAY") and shutil.which("wl-paste") is not None:
        _set_manual_correction_status("Ручной корректор — чтение Wayland clipboard")
        threading.Thread(target=_read_wayland_clipboard_worker, daemon=True).start()
        return
    clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
    if clipboard is None:
        show_error_dialog(
            "Буфер обмена недоступен",
            "Не удалось подключиться к буферу обмена графического сеанса.")
        return
    _set_manual_correction_status("Ручной корректор — чтение буфера")

    def received_text(_clipboard, text, clipboard_ref):
        # Keep the clipboard reference through GTK's asynchronous request.
        if not isinstance(text, str) or not text.strip():
            if (os.getenv("WAYLAND_DISPLAY") and
                    shutil.which("wl-paste") is not None):
                _set_manual_correction_status("Ручной корректор — чтение Wayland clipboard")
                threading.Thread(target=_read_wayland_clipboard_worker, daemon=True).start()
            else:
                log_debug("[manual-correction] Clipboard text target returned empty")
                _manual_clipboard_text_ready(None, "empty")
            return
        _manual_clipboard_text_ready(text, None)
    clipboard.request_text(received_text, clipboard)

def generate_batch_id():
    """Сгенерировать уникальный batch_id вида batch-YYYYMMDD-HHMMSS"""
    base = datetime.now().strftime("batch-%Y%m%d-%H%M%S")
    candidate = base
    suffix = 2
    while os.path.exists(os.path.join(BATCH_JOBS_DIR, f"{candidate}.json")):
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate

def load_batch_job(batch_id):
    """Прочитать метаданные batch job по id, либо None если файла нет / он повреждён"""
    path = os.path.join(BATCH_JOBS_DIR, f"{batch_id}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log_debug(f"Ошибка чтения batch job {batch_id}: {e}")
        return None

def save_batch_job(job):
    """Атомарно сохранить метаданные batch job"""
    path = os.path.join(BATCH_JOBS_DIR, f"{job['id']}.json")
    try:
        fd, tmp_path = tempfile.mkstemp(dir=BATCH_JOBS_DIR, suffix=".tmp", prefix="batch-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(job, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try: os.unlink(tmp_path)
            except OSError: pass
            raise
    except Exception as e:
        log_debug(f"Ошибка сохранения batch job {job.get('id')}: {e}")

def read_error_sidecar(path):
    """Прочитать текст ошибки из sidecar-файла <output_path>.error"""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None

def start_batch_transcription(file_list, quality_config):
    """Создать batch job из подготовленных файлов и запустить обработку.

    file_list — элементы {"name", "original_path", "audio_path", "duration"},
    подготовленные в on_transcribe_file_clicked() после валидации/извлечения аудио.
    Возвращает id созданного batch job'а.
    """
    batch_id = generate_batch_id()
    results_dir = os.path.join(BATCH_RESULTS_DIR, batch_id)
    os.makedirs(results_dir, exist_ok=True)

    current_strategy = quality_config.get("current_strategy", "normal")
    strategy_params = quality_config.get("strategies", {}).get(current_strategy, {})

    used_names = set()
    files = []
    for f in file_list:
        size_mb = get_file_size_mb(f["original_path"])
        base_name = os.path.splitext(f["name"])[0]
        out_name = f"{base_name}.txt"
        n = 2
        while out_name in used_names:
            out_name = f"{base_name}_{n}.txt"
            n += 1
        used_names.add(out_name)

        # Дедуплицировать f["name"] если несколько файлов с одинаковым именем
        display_name = f["name"]
        if display_name in used_names:
            base_display = os.path.splitext(display_name)[0]
            counter = 2
            while f"{base_display}_{counter}{os.path.splitext(display_name)[1]}" in used_names:
                counter += 1
            display_name = f"{base_display}_{counter}{os.path.splitext(display_name)[1]}"
        used_names.add(display_name)

        source_url = f.get("source_url")
        item_source = "file"
        if source_url:
            try:
                from whisper_youtube import source_for_url
                item_source = source_for_url(source_url) or "unknown"
            except Exception:
                item_source = "unknown"

        files.append({
            "path": f["audio_path"],
            "name": display_name,
            "original_name": f["name"],
            "type": f.get("type") or ("video → audio" if f["audio_path"] != f["original_path"] else "audio"),
            "source_url": source_url,
            "source": item_source,
            "video_id": f.get("video_id"),
            "size_mb": round(size_mb, 1) if size_mb is not None else None,
            "duration_sec": int(round(f["duration"])) if f.get("duration") else None,
            "state": "queued",
            "output_path": os.path.join(results_dir, out_name),
            "error": None
        })

    job = {
        "id": batch_id,
        "type": "batch",
        "source": "batch",
        "state": "processing",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results_dir": results_dir,
        "files": files,
        "settings": {
            "model": quality_config.get("current_model", "large-v3"),
            "auto_mode": quality_config.get("auto_mode", False),
            "strategy": current_strategy,
            "language": strategy_params.get("language", "ru"),
            "vad_threshold": strategy_params.get("vad_threshold", 0.5)
        },
        "results": {}
    }
    save_batch_job(job)
    log_debug(f"📦 Batch создан: {batch_id} ({len(files)} файлов)")

    schedule_batch_files(job)
    return batch_id

def process_batch_file(job, file_index):
    """Запустить обработку одного файла batch job'а: пометить processing, запустить subprocess"""
    file_entry = job["files"][file_index]
    file_entry["state"] = "processing"
    file_entry["error"] = None
    file_entry["started_at"] = time.time()
    file_entry["finished_at"] = None
    save_batch_job(job)

    try:
        # start_new_session=True: whisper-dictate-v2 порождает несколько уровней
        # bash-обёрток (daemon-client.sh, fallback на whisper-run) через $(...) —
        # SIGTERM в один только этот PID их не убивает, они переживают "отмену"
        # осиротевшими процессами. Отдельная сессия даёт группу процессов, которую
        # cancel_batch_job() может убить целиком через os.killpg().
        job_source = file_entry.get("source", "file")
        if job_source not in {"file", "youtube", "rutube", "vk"}:
            job_source = "unknown"
        proc = subprocess.Popen([
            os.path.expanduser("~/.local/bin/whisper/whisper-dictate-v2"),
            "--file", file_entry["path"],
            "--batch-id", job["id"],
            "--output", file_entry["output_path"]
        ], start_new_session=True, env={**os.environ, "WHISPER_JOB_SOURCE": job_source})
        _batch_inflight.setdefault(job["id"], {})[file_index] = proc
        log_debug(f"📦 [{job['id']}] Обработка файла {file_entry['name']} (#{file_index})")
    except Exception as e:
        file_entry["state"] = "failed"
        file_entry["error"] = str(e)
        file_entry["finished_at"] = time.time()
        save_batch_job(job)
        log_debug(f"✗ [{job['id']}] Не удалось запустить обработку {file_entry['name']}: {e}")

def schedule_batch_files(job):
    """Заполнить свободные слоты параллелизма (до MAX_BATCH_PARALLEL) файлами из очереди"""
    processing_count = sum(1 for f in job["files"] if f["state"] == "processing")
    free_slots = MAX_BATCH_PARALLEL - processing_count
    for i, f in enumerate(job["files"]):
        if free_slots <= 0:
            break
        if f["state"] == "queued":
            process_batch_file(job, i)
            free_slots -= 1

def poll_batch_jobs():
    """Опросить batch jobs: обнаружить завершённые файлы, забрать результат/ошибку,
    запустить следующие из очереди, закрыть job когда все файлы обработаны"""
    for path in sorted(glob.glob(os.path.join(BATCH_JOBS_DIR, "*.json"))):
        batch_id = os.path.splitext(os.path.basename(path))[0]
        job = load_batch_job(batch_id)
        if not job or job.get("state") != "processing":
            continue

        inflight = _batch_inflight.setdefault(batch_id, {})
        changed = False

        for i, f in enumerate(job["files"]):
            if f["state"] != "processing":
                continue

            proc = inflight.get(i)
            if proc is not None and proc.poll() is None:
                continue  # ещё выполняется

            ret = proc.poll() if proc is not None else None
            if proc is not None:
                del inflight[i]

            if os.path.exists(f["output_path"]):
                f["state"] = "completed"
                f["finished_at"] = time.time()
                job["results"][f["name"]] = f["output_path"]
                changed = True
            elif os.path.exists(f["output_path"] + ".error"):
                f["error"] = read_error_sidecar(f["output_path"] + ".error")
                f["state"] = "failed"
                f["finished_at"] = time.time()
                changed = True
            elif proc is not None:
                # Процесс завершился, но не оставил ни результата, ни файла ошибки
                f["error"] = f"Процесс завершился без результата (код {ret})"
                f["state"] = "failed"
                f["finished_at"] = time.time()
                changed = True
            # Если proc is None (трей был перезапущен) и артефактов ещё нет — ждём следующего опроса

        # Всегда пытаемся заполнить свободные слоты — не только сразу после
        # завершения файла, иначе свежесозданный job (все файлы "queued",
        # ни одного "processing") никогда не сдвинется с места
        schedule_batch_files(job)

        pending = any(f["state"] in ("queued", "processing") for f in job["files"])
        if not pending:
            job["state"] = "done"
            completed = sum(1 for f in job["files"] if f["state"] == "completed")
            failed = sum(1 for f in job["files"] if f["state"] == "failed")
            notify_send("Whisper Batch", f"✅ Batch завершён: {completed} готово, {failed} ошибок")
            log_debug(f"📦 [{batch_id}] Batch завершён: {completed}/{len(job['files'])} успешно")
            _batch_inflight.pop(batch_id, None)
            changed = True
            GLib.idle_add(finalize_batch_results, batch_id)

        if changed:
            save_batch_job(job)

    return True

def build_batch_display_entries():
    """Собрать записи активных batch jobs для отображения в меню трея"""
    entries = []
    local_device = load_voice_config().get("device", "cpu")
    for path in sorted(glob.glob(os.path.join(BATCH_JOBS_DIR, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                job = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if job.get("state") != "processing":
            continue
        files = job.get("files", [])
        total = len(files)
        done = sum(1 for f in files if f.get("state") in ("completed", "failed"))
        icon_key, _icon_label = backend_status_icon("processing", "local", local_device)
        label = f"BATCH: {job.get('id','?')} ({done}/{total} файлов)"
        entries.append({"state": "processing", "device": local_device, "backend": "local",
                        "status_icon": icon_key, "label": label, "batch_id": job.get("id")})
    return entries

def cancel_batch_job(batch_id):
    """Остановить обработку batch job'а: убить работающие процессы, пометить
    необработанные файлы отменёнными, сохранить уже готовые результаты"""
    job = load_batch_job(batch_id)
    if not job or job.get("state") != "processing":
        return

    inflight = _batch_inflight.get(batch_id, {})
    for i, f in enumerate(job["files"]):
        if f["state"] == "processing":
            proc = inflight.get(i)
            if proc is not None:
                try:
                    # Убить всю группу процессов (обёртка + daemon-client.sh + возможный
                    # fallback на whisper-run), а не только сам обёрточный PID — иначе
                    # транскрибация продолжает жрать GPU/CPU осиротевшей после "отмены"
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                try:
                    # Забрать самого прямого потомка, чтобы не оставлять зомби —
                    # остальные члены группы процессу-трею не принадлежат, ОС сама
                    # реассоциирует их с init/systemd после SIGTERM.
                    proc.wait(timeout=2)
                except Exception:
                    pass
            f["state"] = "cancelled"
            f["finished_at"] = time.time()
        elif f["state"] == "queued":
            f["state"] = "cancelled"

    job["state"] = "cancelled"
    save_batch_job(job)
    _batch_inflight.pop(batch_id, None)

    completed = sum(1 for f in job["files"] if f["state"] == "completed")
    failed = sum(1 for f in job["files"] if f["state"] == "failed")
    notify_send("Whisper Batch", f"⏹️ Batch отменён: {completed} готово, {failed} ошибок, остальные отменены")
    log_debug(f"📦 [{batch_id}] Batch отменён пользователем")
    GLib.idle_add(finalize_batch_results, batch_id)

def restart_batch_job(batch_id):
    """Вернуть в очередь файлы, отменённые пользователем в cancel_batch_job(), и
    возобновить обработку. Файлы, уже завершённые (completed/failed) до отмены,
    не трогаются — перезапускаются только те, что были cancelled."""
    job = load_batch_job(batch_id)
    if not job or job.get("state") != "cancelled":
        return

    for f in job["files"]:
        if f["state"] == "cancelled":
            f["state"] = "queued"
            f["error"] = None
            f["started_at"] = None
            f["finished_at"] = None

    job["state"] = "processing"
    save_batch_job(job)
    log_debug(f"📦 [{batch_id}] Batch перезапущен пользователем")
    schedule_batch_files(job)

def unique_output_path(directory, base_name, ext):
    """Подобрать свободный путь <directory>/<base_name><ext>, добавляя _N при коллизии"""
    candidate = os.path.join(directory, f"{base_name}{ext}")
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base_name}_{n}{ext}")
        n += 1
    return candidate

def format_duration_words(seconds):
    """Форматировать длительность словами: '20 минут 50 секунд'"""
    if seconds is None:
        return "?"
    seconds = int(round(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours} ч")
    if hours or minutes:
        parts.append(f"{minutes} минут")
    parts.append(f"{secs} секунд")
    return " ".join(parts)

def open_in_editor(path):
    """Открыть файл в текстовом редакторе (с fallback на xdg-open)"""
    for editor in ["kate", "gedit", "xed", "mousepad", "pluma"]:
        try:
            subprocess.Popen([editor, path])
            return
        except FileNotFoundError:
            continue
    try:
        subprocess.Popen(["xdg-open", path])
    except Exception as e:
        log_debug(f"✗ Не удалось открыть {path}: {e}")

def copy_text_to_clipboard(text):
    """Скопировать текст в системный буфер и сохранить в истории Klipper."""
    try:
        deliver_clipboard_text(text)
        return True
    except Exception as e:
        category = getattr(e, "category", None) or type(e).__name__
        log_debug(f"✗ Ошибка clipboard delivery: {category}")
        return False

def format_file_info_header(file_entry, job):
    """Собрать блок '[Информация о файле]' для текстового результата одиночного файла"""
    settings = job.get("settings", {})
    language_display = LANGUAGE_LABELS.get(settings.get("language"), settings.get("language") or "Авто")
    size_str = f"{file_entry['size_mb']} MB" if file_entry.get("size_mb") is not None else "? MB"
    duration_str = format_duration(file_entry.get("duration_sec"))
    lines = [
        "[Информация о файле]",
        f"Файл: {file_entry.get('name', '?')}",
        f"Размер: {size_str}",
        f"Продолжительность: {duration_str}",
        f"Модель: {settings.get('model', '?')}",
        f"Язык: {language_display}",
    ]
    if file_entry.get("source_url"):
        lines.append(f"Источник: {file_entry['source_url']}")
    lines += ["", "[Результат транскрипции]"]
    return "\n".join(lines)

def build_batch_summary(job):
    """Посчитать сводную статистику batch job'а: успех/ошибки/общее время/скорость"""
    files = job.get("files", [])
    completed = [f for f in files if f["state"] == "completed"]
    failed = [f for f in files if f["state"] == "failed"]
    total_duration_sec = sum(f.get("duration_sec") or 0 for f in files)
    processing_times = [
        f["finished_at"] - f["started_at"] for f in completed
        if f.get("started_at") and f.get("finished_at")
    ]
    total_time_sec = sum(processing_times)
    avg_speed = round(total_duration_sec / total_time_sec, 2) if total_time_sec else 0.0
    summary = {
        "success": len(completed),
        "failed": len(failed),
        "total_time_sec": round(total_time_sec, 1),
        "avg_speed": avg_speed
    }
    return summary, total_duration_sec

def build_results_json(job):
    """Собрать машиночитаемый results.json для batch job'а"""
    settings = job.get("settings", {})
    files_out = []
    for f in job.get("files", []):
        entry = {
            "name": f.get("name"),
            "state": f.get("state"),
            "duration_sec": f.get("duration_sec"),
            "source_url": f.get("source_url"),
            "video_id": f.get("video_id")
        }
        if f["state"] == "completed":
            text = ""
            try:
                with open(f["output_path"], encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                pass
            entry["text"] = text
            entry["language_detected"] = settings.get("language")
            if f.get("started_at") and f.get("finished_at"):
                entry["processing_time_sec"] = round(f["finished_at"] - f["started_at"], 1)
        elif f["state"] == "failed":
            entry["error"] = f.get("error")
        files_out.append(entry)

    summary, total_duration_sec = build_batch_summary(job)
    return {
        "batch_id": job["id"],
        "total_files": len(job.get("files", [])),
        "total_duration_sec": int(total_duration_sec),
        "settings": {
            "model": settings.get("model"),
            "strategy": settings.get("strategy"),
            "language": settings.get("language")
        },
        "files": files_out,
        "summary": summary
    }

def build_readme_md(job):
    """Собрать README.md для архива batch job'а"""
    quality_config = load_quality_config()
    settings = job.get("settings", {})
    strategy_key = settings.get("strategy", "normal")
    strategy_name = quality_config.get("strategies", {}).get(strategy_key, {}).get("name", strategy_key)
    language_display = LANGUAGE_LABELS.get(settings.get("language"), settings.get("language") or "Авто")

    try:
        created_display = datetime.fromisoformat(job.get("created_at", "")).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        created_display = job.get("created_at", "?")

    summary, _ = build_batch_summary(job)
    status_icons = {"completed": "✅", "failed": "❌", "cancelled": "🚫"}

    lines = [
        f"# Batch Транскрибация: {job['id']}",
        "",
        f"Дата: {created_display}",
        f"Модель: {settings.get('model', '?')}",
        f"Стратегия: {strategy_name}",
        f"Язык: {language_display}",
        "",
        "## Результаты",
        ""
    ]
    for f in job.get("files", []):
        icon = status_icons.get(f["state"], "⏳")
        line = f"- {f['name']} {icon}"
        if f.get("source_url"):
            line += f"  \n  [🎬 Смотреть видео]({f['source_url']})"
        lines.append(line)
    lines += [
        "",
        f"Всего файлов: {len(job.get('files', []))}",
        f"Успешно: {summary['success']}",
        f"Ошибок: {summary['failed']}",
        "",
        "## Общее время обработки",
        "",
        format_duration_words(summary["total_time_sec"])
    ]
    return "\n".join(lines)

def build_index_html(job, results_data):
    """Собрать самодостаточный index.html для просмотра результатов batch job'а в браузере"""
    quality_config = load_quality_config()
    settings = job.get("settings", {})
    strategy_key = settings.get("strategy", "normal")
    strategy_name = quality_config.get("strategies", {}).get(strategy_key, {}).get("name", strategy_key)
    language_display = LANGUAGE_LABELS.get(settings.get("language"), settings.get("language") or "Авто")
    summary = results_data["summary"]
    status_icons = {"completed": "✅", "failed": "❌", "cancelled": "🚫"}

    sidebar_items = []
    sections = []
    for i, f in enumerate(job.get("files", [])):
        anchor = f"file-{i}"
        state = f.get("state")
        icon = status_icons.get(state, "⏳")
        name_esc = html.escape(f.get("name", "?"))
        sidebar_items.append(f'<li><a href="#{anchor}">{icon} {name_esc}</a></li>')

        if state == "completed":
            text = ""
            try:
                with open(f["output_path"], encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                text = "(не удалось прочитать результат)"
            body = f"<pre>{html.escape(text)}</pre>"
        elif state == "failed":
            body = f'<p class="error">Ошибка: {html.escape(f.get("error") or "неизвестная ошибка")}</p>'
        else:
            body = f'<p class="muted">Файл {icon}</p>'

        duration_str = format_duration(f.get("duration_sec"))
        meta_parts = [f'<p class="meta">Длительность: {duration_str}</p>']
        if f.get("source_url"):
            source_url_esc = html.escape(f["source_url"])
            meta_parts.append(f'<p><a href="{source_url_esc}" target="_blank">🎬 Смотреть видео</a></p>')
        meta_html = "".join(meta_parts)
        sections.append(
            f'<section id="{anchor}"><h2>{icon} {name_esc}</h2>'
            f'{meta_html}{body}</section>'
        )

    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Batch Транскрибация: {html.escape(job["id"])}</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    display: flex; min-height: 100vh; background: #fff; color: #1a1a1a;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #1a1a1a; color: #e8e8e8; }}
    aside {{ background: #242424 !important; border-color: #333 !important; }}
    pre {{ background: #111 !important; border-color: #333 !important; }}
    a {{ color: #7db8ff !important; }}
    section {{ border-color: #333 !important; }}
  }}
  aside {{
    width: 280px; flex-shrink: 0; background: #f5f5f5; border-right: 1px solid #ddd;
    padding: 16px; overflow-y: auto; position: sticky; top: 0; height: 100vh;
  }}
  aside h1 {{ font-size: 15px; margin: 0 0 12px; word-break: break-word; }}
  aside .stats {{ font-size: 13px; line-height: 1.6; margin-bottom: 16px; color: #888; }}
  aside ul {{ list-style: none; margin: 0; padding: 0; }}
  aside li {{ margin-bottom: 4px; }}
  aside a {{ text-decoration: none; color: #0366d6; display: block; padding: 4px 6px; border-radius: 4px; word-break: break-word; }}
  aside a:hover {{ background: rgba(3,102,214,0.1); }}
  main {{ flex: 1; padding: 24px 32px; overflow-y: auto; min-width: 0; }}
  section {{ margin-bottom: 40px; padding-bottom: 24px; border-bottom: 1px solid #eee; }}
  section h2 {{ font-size: 18px; word-break: break-word; }}
  .meta {{ color: #888; font-size: 13px; }}
  pre {{
    white-space: pre-wrap; word-wrap: break-word; background: #f8f8f8;
    border: 1px solid #eee; border-radius: 6px; padding: 12px; font-size: 14px; line-height: 1.5;
  }}
  .error {{ color: #d73a49; }}
  .muted {{ color: #999; }}
  @media (max-width: 720px) {{
    body {{ flex-direction: column; }}
    aside {{ width: 100%; height: auto; position: static; border-right: none; border-bottom: 1px solid #ddd; }}
  }}
</style>
</head>
<body>
<aside>
  <h1>📦 Batch: {html.escape(job["id"])}</h1>
  <div class="stats">
    Модель: {html.escape(str(settings.get("model", "?")))}<br>
    Стратегия: {html.escape(strategy_name)}<br>
    Язык: {html.escape(language_display)}<br>
    Успешно: {summary["success"]} / Ошибок: {summary["failed"]}<br>
    Время обработки: {html.escape(format_duration_words(summary["total_time_sec"]))}
  </div>
  <ul>
    {"".join(sidebar_items)}
  </ul>
</aside>
<main>
  {"".join(sections)}
</main>
</body>
</html>"""

def show_single_result_dialog(out_path, text):
    """Диалог результата для batch из одного файла: путь к результату + действия"""
    dialog = Gtk.MessageDialog(message_type=Gtk.MessageType.INFO, buttons=Gtk.ButtonsType.NONE,
                                text="Транскрибация завершена")
    dialog.format_secondary_text(f"Результат сохранён в:\n{out_path}")
    dialog.add_buttons("Открыть в редакторе", 1, "Копировать", 2, "Закрыть", Gtk.ResponseType.CLOSE)
    while True:
        response = dialog.run()
        if response == 1:
            open_in_editor(out_path)
        elif response == 2:
            copy_text_to_clipboard(text)
        else:
            break
    dialog.destroy()
    return False

def show_multi_result_dialog(results_dir):
    """Диалог результата для batch из нескольких файлов (только при успешном завершении —
    даёт доступ к папке результатов даже если окно прогресса было скрыто и пропало из меню трея)"""
    dialog = Gtk.MessageDialog(message_type=Gtk.MessageType.INFO, buttons=Gtk.ButtonsType.NONE,
                                text="Batch транскрибация завершена")
    dialog.format_secondary_text(f"Результаты сохранены в:\n{results_dir}")
    dialog.add_buttons("Открыть папку", 1, "Закрыть", Gtk.ResponseType.CLOSE)
    while True:
        response = dialog.run()
        if response == 1:
            try:
                subprocess.Popen(["xdg-open", results_dir])
            except Exception as e:
                log_debug(f"✗ Не удалось открыть папку результатов: {e}")
        else:
            break
    dialog.destroy()
    return False

def finalize_single_file_result(job):
    """Собрать текстовый результат для batch из одного файла (без архива)"""
    f = job["files"][0]
    if f["state"] == "cancelled":
        GLib.idle_add(show_error_dialog, "Batch отменена", f"{f['name']}: обработка отменена пользователем")
        return
    if f["state"] != "completed":
        GLib.idle_add(show_error_dialog, "Ошибка транскрибации",
                       f"{f['name']}: {f.get('error') or 'неизвестная ошибка'}")
        return

    try:
        with open(f["output_path"], encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        log_debug(f"✗ [{job['id']}] Не удалось прочитать результат {f['output_path']}: {e}")
        return

    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(f["name"])[0]
    out_path = unique_output_path(BATCH_OUTPUT_DIR, base_name, ".txt")
    content = format_file_info_header(f, job) + "\n" + text
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(content)
    except OSError as e:
        log_debug(f"✗ [{job['id']}] Не удалось сохранить результат {out_path}: {e}")
        return

    log_debug(f"📄 [{job['id']}] Результат сохранён: {out_path}")
    GLib.idle_add(show_single_result_dialog, out_path, text)

def finalize_multi_file_result(job):
    """Создать results.json/index.html/README.md для batch из нескольких файлов (без архива)"""
    results_dir = job["results_dir"]
    results_data = build_results_json(job)

    try:
        with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(results_data, fh, ensure_ascii=False, indent=2)
    except OSError as e:
        log_debug(f"✗ [{job['id']}] Не удалось записать results.json: {e}")

    try:
        with open(os.path.join(results_dir, "index.html"), "w", encoding="utf-8") as fh:
            fh.write(build_index_html(job, results_data))
    except OSError as e:
        log_debug(f"✗ [{job['id']}] Не удалось записать index.html: {e}")

    try:
        with open(os.path.join(results_dir, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(build_readme_md(job))
    except OSError as e:
        log_debug(f"✗ [{job['id']}] Не удалось записать README.md: {e}")

    log_debug(f"📦 [{job['id']}] Результаты подготовлены: {results_dir}")

    # Диалог только при обычном завершении — при отмене (state == "cancelled")
    # окно прогресса уже показывает "отменено" рядом с каждым файлом, второй раз
    # оповещать не нужно (и тем более не текстом "завершена")
    if job.get("state") == "done":
        GLib.idle_add(show_multi_result_dialog, results_dir)

def cleanup_extracted_audio(job):
    """Удалить временные MP3, извлечённые из видео для этого job'а (on_transcribe_file_clicked
    кладёт их в tempfile.gettempdir()/whisper-batch-<timestamp>/) — они не нужны после
    транскрибации, и без явной очистки копятся в /tmp навсегда."""
    tmp_root = os.path.realpath(tempfile.gettempdir())
    dirs_to_check = set()
    for f in job.get("files", []):
        if f.get("type") != "video → audio":
            continue
        path = f.get("path")
        if not path:
            continue
        # Удалять только локальные временные видео (тип "video → audio" с ->, не "youtube")
        # YouTube-аудио остаётся в ~/.cache/whisper-dictate/youtube-audio/ для кэша
        real_path = os.path.realpath(path)
        parent = os.path.dirname(real_path)
        # Безопасность: удаляем только файлы внутри tempfile.gettempdir()/whisper-batch-*,
        # никогда не трогаем оригиналы пользователя (audio_path для типа "audio" сюда не попадает)
        if (os.path.dirname(parent) == tmp_root
                and os.path.basename(parent).startswith("whisper-batch-")):
            try:
                os.remove(real_path)
            except OSError:
                pass
            dirs_to_check.add(parent)

    for d in dirs_to_check:
        try:
            if not os.listdir(d):
                os.rmdir(d)
        except OSError:
            pass

def finalize_batch_results(batch_id):
    """Собрать финальные результаты batch job'а: один файл → txt, несколько → архив.
    Вызывается один раз, когда все файлы обработаны (completed/failed/cancelled)."""
    job = load_batch_job(batch_id)
    if not job or not job.get("files"):
        return False

    try:
        if len(job["files"]) == 1:
            finalize_single_file_result(job)
        else:
            finalize_multi_file_result(job)
    except Exception as e:
        log_debug(f"✗ [{batch_id}] Ошибка при финализации результатов: {e}")

    try:
        cleanup_extracted_audio(job)
    except Exception as e:
        log_debug(f"✗ [{batch_id}] Ошибка очистки временных аудио из видео: {e}")

    try:
        cleanup_youtube_cache()
    except Exception as e:
        log_debug(f"✗ [{batch_id}] Ошибка очистки YouTube кэша: {e}")

    return False

def create_batch_progress_window(batch_id):
    """Открыть окно прогресса batch-обработки, либо поднять уже открытое.

    Список файлов строится один раз из начального состояния job'а — состав файлов
    не меняется после создания batch, поэтому виджеты по индексу переиспользуются
    на каждом тике обновления."""
    existing = _batch_windows.get(batch_id)
    if existing:
        existing["window"].present()
        existing["window"].show()
        return existing["window"]

    job = load_batch_job(batch_id)
    if not job:
        return None

    window = Gtk.Window(title=f"Batch Транскрибация: {batch_id}")
    window.set_default_size(580, 440)
    window.set_border_width(10)

    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    window.add(vbox)

    header = Gtk.Label()
    header.set_markup(f"<b>Batch Транскрибация: {GLib.markup_escape_text(batch_id)}</b>")
    header.set_xalign(0)
    vbox.pack_start(header, False, False, 0)

    scrolled = Gtk.ScrolledWindow()
    scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    scrolled.set_min_content_height(230)
    vbox.pack_start(scrolled, True, True, 0)

    listbox = Gtk.ListBox()
    listbox.set_selection_mode(Gtk.SelectionMode.NONE)
    scrolled.add(listbox)

    row_widgets = []
    for f in job["files"]:
        row = Gtk.ListBoxRow()
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox.set_border_width(4)
        row.add(hbox)

        name_label = Gtk.Label(label=f["name"])
        name_label.set_xalign(0)
        name_label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        name_label.set_max_width_chars(26)
        hbox.pack_start(name_label, True, True, 0)

        status_label = Gtk.Label(label="")
        status_label.set_width_chars(13)
        status_label.set_xalign(0)
        hbox.pack_start(status_label, False, False, 0)

        progress = Gtk.ProgressBar()
        progress.set_size_request(110, -1)
        progress.set_show_text(False)
        hbox.pack_start(progress, False, False, 0)

        time_label = Gtk.Label(label="")
        time_label.set_width_chars(8)
        time_label.set_xalign(1)
        hbox.pack_start(time_label, False, False, 0)

        listbox.add(row)
        row_widgets.append({"status": status_label, "progress": progress, "time": time_label})

    overall_label = Gtk.Label()
    overall_label.set_xalign(0)
    vbox.pack_start(overall_label, False, False, 0)

    overall_progress = Gtk.ProgressBar()
    overall_progress.set_show_text(True)
    vbox.pack_start(overall_progress, False, False, 0)

    eta_label = Gtk.Label()
    eta_label.set_xalign(0)
    vbox.pack_start(eta_label, False, False, 0)

    button_box = Gtk.ButtonBox(orientation=Gtk.Orientation.HORIZONTAL)
    button_box.set_layout(Gtk.ButtonBoxStyle.END)
    button_box.set_spacing(6)
    vbox.pack_start(button_box, False, False, 0)

    cancel_button = Gtk.Button(label="⏹️ Отмена Batch")
    hide_button = Gtk.Button(label="🔽 Скрыть")
    close_button = Gtk.Button(label="✖️ Закрыть")
    button_box.add(cancel_button)
    button_box.add(hide_button)
    button_box.add(close_button)

    def update_window():
        current_job = load_batch_job(batch_id)
        if not current_job:
            return False  # job удалён — больше нечего показывать

        files = current_job.get("files", [])
        for i, f in enumerate(files):
            if i >= len(row_widgets):
                break
            w = row_widgets[i]
            state = f.get("state", "queued")
            w["status"].set_text(BATCH_STATUS_LABELS.get(state, state))

            started = f.get("started_at")
            finished = f.get("finished_at")
            if state == "processing":
                w["progress"].pulse()
                w["time"].set_text(format_duration(time.time() - started) if started else "")
            elif state in ("completed", "failed"):
                w["progress"].set_fraction(1.0)
                elapsed = (finished - started) if (started and finished) else None
                w["time"].set_text(format_duration(elapsed) if elapsed is not None else "")
            else:  # queued, cancelled
                w["progress"].set_fraction(0.0)
                w["time"].set_text("")

        total = len(files)
        done = sum(1 for f in files if f.get("state") in ("completed", "failed", "cancelled"))
        overall_label.set_text(f"Обработано: {done}/{total} файлов")
        overall_progress.set_fraction(done / total if total else 0.0)

        remaining = sum(1 for f in files if f.get("state") in ("queued", "processing"))
        completed_timed = [f for f in files if f.get("state") == "completed" and f.get("started_at") and f.get("finished_at")]
        if remaining and completed_timed:
            avg = sum(f["finished_at"] - f["started_at"] for f in completed_timed) / len(completed_timed)
            parallel = min(MAX_BATCH_PARALLEL, remaining)
            eta_sec = avg * remaining / max(parallel, 1)
            eta_label.set_text(f"Осталось примерно: ~{format_duration(eta_sec)}")
        elif remaining:
            eta_label.set_text("Осталось: вычисляется...")
        else:
            eta_label.set_text("")

        if current_job.get("state") != "processing":
            if current_job.get("state") == "cancelled":
                cancel_button.set_label("🔄 Перезапустить")
                cancel_button.set_sensitive(True)
            else:
                cancel_button.set_sensitive(False)
            close_button.set_label("📂 Просмотреть результаты")
            return False  # обработка завершена/отменена — больше опрашивать не нужно

        return True

    def on_cancel_clicked(_):
        current_job = load_batch_job(batch_id)
        if current_job and current_job.get("state") == "cancelled":
            cancel_button.set_sensitive(False)
            restart_batch_job(batch_id)
            update_window()
            entry = _batch_windows.get(batch_id)
            if entry:
                entry["timeout_id"] = GLib.timeout_add(500, update_window)
        else:
            cancel_batch_job(batch_id)
            update_window()

    def on_close_clicked(_):
        current_job = load_batch_job(batch_id)
        if current_job and current_job.get("state") != "processing":
            results_dir = current_job.get("results_dir")
            if results_dir and os.path.isdir(results_dir):
                try:
                    subprocess.Popen(["xdg-open", results_dir])
                except Exception as e:
                    log_debug(f"✗ Не удалось открыть папку результатов: {e}")
            return
        window.destroy()

    def cleanup(*_args):
        entry = _batch_windows.pop(batch_id, None)
        if entry and entry.get("timeout_id"):
            try:
                GLib.source_remove(entry["timeout_id"])
            except Exception:
                pass

    cancel_button.connect("clicked", on_cancel_clicked)
    hide_button.connect("clicked", lambda _: window.hide())
    close_button.connect("clicked", on_close_clicked)
    window.connect("destroy", cleanup)

    update_window()
    window.show_all()

    timeout_id = GLib.timeout_add(500, update_window)
    _batch_windows[batch_id] = {"window": window, "timeout_id": timeout_id}
    return window

def show_youtube_url_dialog():
    """Диалог ввода YouTube-ссылок.

    Показывает:
    - Многострочное поле ввода (монохромный шрифт, WORD_CHAR wrap)
    - Подсказку с поддерживаемыми форматами
    - Live-счётчик валидных и невалидных ссылок
    - Кнопку "📋 Вставить из буфера"
    - Кнопка OK активна только если есть валидные ссылки

    Возвращает текст (многострочный) или None.
    """
    dialog = Gtk.Dialog(
        title="Распознать видео с YouTube",
        transient_for=None,
        flags=0
    )
    dialog.add_buttons("Отмена", Gtk.ResponseType.CANCEL,
                       "Распознать", Gtk.ResponseType.OK)
    dialog.set_default_size(560, 420)

    content_area = dialog.get_content_area()
    content_area.set_spacing(8)
    content_area.set_border_width(12)

    # Заголовок
    title_label = Gtk.Label()
    title_label.set_markup("<b>Введите ссылки на YouTube, RuTube или VK (одна на строку):</b>")
    title_label.set_xalign(0)
    content_area.pack_start(title_label, False, False, 0)

    # Подсказка с форматами
    hint_label = Gtk.Label()
    hint_label.set_markup(
        "<small>Поддерживаемые видеохостинги: YouTube, RuTube и VK (через yt-dlp).\n"
        "• https://www.youtube.com/watch?v=VIDEO_ID\n"
        "• https://youtu.be/VIDEO_ID\n"
        "• https://www.youtube.com/shorts/VIDEO_ID\n"
        "• Можно добавить параметры: ?t=90s, &list=PLxxx</small>"
    )
    hint_label.set_xalign(0)
    hint_label.set_line_wrap(True)
    content_area.pack_start(hint_label, False, False, 0)

    # Многострочный ввод в ScrolledWindow
    scrolled = Gtk.ScrolledWindow()
    scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    scrolled.set_min_content_height(150)

    text_view = Gtk.TextView()
    text_view.set_monospace(True)
    text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
    scrolled.add(text_view)
    content_area.pack_start(scrolled, True, True, 0)

    # Live-счётчик
    counter_label = Gtk.Label()
    counter_label.set_xalign(0)
    def update_counter(_=None):
        from whisper_youtube import parse_youtube_urls
        text = text_view.get_buffer().get_text(
            text_view.get_buffer().get_start_iter(),
            text_view.get_buffer().get_end_iter(),
            include_hidden_chars=False
        )
        entries = parse_youtube_urls(text)
        valid_count = sum(1 for e in entries if e["valid"])
        invalid_count = sum(1 for e in entries if not e["valid"])
        counter_label.set_text(f"Распознано ссылок: {valid_count} (невалидных: {invalid_count})")

        # Активировать кнопку OK только если есть валидные
        ok_button = dialog.get_widget_for_response(Gtk.ResponseType.OK)
        ok_button.set_sensitive(valid_count > 0)

    text_view.get_buffer().connect("changed", update_counter)
    content_area.pack_start(counter_label, False, False, 0)

    # Кнопка вставки из буфера
    button_box = Gtk.HBox(spacing=6)
    paste_button = Gtk.Button(label="📋 Вставить из буфера")
    def on_paste_clicked(_):
        def paste_callback(clipboard, text, user_data):
            if text:
                text_view.get_buffer().insert_at_cursor(text)
        clipboard = Gtk.Clipboard.get_default()
        clipboard.request_text(paste_callback, None)

    paste_button.connect("clicked", on_paste_clicked)
    button_box.pack_start(paste_button, False, False, 0)
    content_area.pack_start(button_box, False, False, 0)

    content_area.show_all()

    # Инициализация счётчика
    update_counter()

    response = dialog.run()
    if response == Gtk.ResponseType.OK:
        text = text_view.get_buffer().get_text(
            text_view.get_buffer().get_start_iter(),
            text_view.get_buffer().get_end_iter(),
            include_hidden_chars=False
        )
        dialog.destroy()
        return text

    dialog.destroy()
    return None

def show_youtube_validation_dialog(invalid_entries, unavailable_entries):
    """Диалог показа ошибок валидации YouTube-ссылок.

    invalid_entries — ссылки с некорректным форматом (из парсера)
    unavailable_entries — видео недоступны (будут на Фазе 8.4 из метаданных)

    Показывает две секции: формат + недоступность.
    Возвращает True если пользователь хочет продолжить с остатком, иначе False.
    """
    dialog = Gtk.Dialog(
        title="Ошибки при валидации",
        transient_for=None,
        flags=0
    )
    dialog.add_buttons("Отмена", Gtk.ResponseType.CANCEL,
                       "Да, продолжить", Gtk.ResponseType.OK)
    dialog.set_default_size(480, 320)

    content_area = dialog.get_content_area()
    content_area.set_spacing(8)
    content_area.set_border_width(12)

    scrolled = Gtk.ScrolledWindow()
    scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    scrolled.set_min_content_height(200)

    label = Gtk.Label()
    label.set_xalign(0)
    label.set_line_wrap(True)

    text_parts = []

    if invalid_entries:
        text_parts.append("<b>🚫 Некорректный формат ({} ссылок):</b>".format(len(invalid_entries)))
        for entry in invalid_entries:
            error_name = entry.get("error", "unknown")
            error_msgs = {
                "not_youtube": "это не YouTube",
                "unsupported_host": "ссылка не с поддерживаемого видеохостинга",
                "no_video_id": "короткий или неверный ID",
                "malformed": "неверный формат"
            }
            error_msg = error_msgs.get(error_name, error_name)
            text_parts.append(f"• {entry.get('raw', '?')} — {error_msg}")

    if unavailable_entries:
        if text_parts:
            text_parts.append("")
        text_parts.append("<b>⚠️ Видео недоступны ({} роликов):</b>".format(len(unavailable_entries)))
        for entry in unavailable_entries:
            reason = entry.get("reason", "unknown")
            reason_msgs = {
                "deleted": "видео было удалено",
                "private": "видео приватное",
                "age": "возрастное ограничение",
                "geo": "недоступно в вашей стране",
                "members": "только для участников канала",
                "live": "это живой стрим",
                "metadata_timeout": "истекло время ожидания метаданных",
                "metadata_error": "не удалось получить метаданные",
                "download_error": "не удалось скачать аудио",
                "duplicate": "дубликат ссылки, обработано первое вхождение",
                "metadata_ok": "метаданные получены"
            }
            reason_msg = reason_msgs.get(reason, reason)
            label_text = entry.get("raw") or entry.get("url") or entry.get("title") or entry.get("video_id") or "URL"
            safe_label = GLib.markup_escape_text(str(label_text))
            text_parts.append(f"• {safe_label} — {reason_msg}")

    if text_parts:
        text_parts.append("")

    # Подсчитать остаток валидных
    valid_count = dialog.get_data("valid_count") if hasattr(dialog, "get_data") else 0
    if valid_count > 0:
        text_parts.append(f"<i>Продолжить обработку остальных ({valid_count} валидная)?</i>")

    label.set_markup("\n".join(text_parts))
    scrolled.add(label)
    content_area.pack_start(scrolled, True, True, 0)

    content_area.show_all()

    response = dialog.run()
    dialog.destroy()
    return response == Gtk.ResponseType.OK

def classify_ytdlp_error(stderr, returncode):
    """Классификация ошибок yt-dlp в человекочитаемый вид.

    Возвращает dict:
    {
        "code": "video_unavailable" | "private" | "age_restricted" | ...
        "message": "Человекочитаемое описание",
        "hint": "Рекомендация (yt-dlp -U и т.д.)" или None
    }
    """
    stderr_lower = stderr.lower()

    if any(x in stderr_lower for x in ["video unavailable", "has been removed", "no longer available"]):
        return {
            "code": "video_unavailable",
            "message": "Видео удалено или недоступно",
            "hint": None
        }

    if any(x in stderr_lower for x in ["private video", "sign in if you", "private content"]):
        return {
            "code": "private",
            "message": "Видео приватное — требуется авторизация",
            "hint": None
        }

    if "age" in stderr_lower and "confirm" in stderr_lower:
        return {
            "code": "age_restricted",
            "message": "Видео имеет возрастное ограничение",
            "hint": None
        }

    if any(x in stderr_lower for x in ["not available in your country", "geo", "geographic"]):
        return {
            "code": "geo_blocked",
            "message": "Видео недоступно в вашей стране",
            "hint": None
        }

    if any(x in stderr_lower for x in ["members-only", "members only", "available to channel members"]):
        return {
            "code": "members_only",
            "message": "Видео только для участников канала",
            "hint": None
        }

    if any(x in stderr_lower for x in ["is live", "live event", "live stream"]):
        return {
            "code": "live_stream",
            "message": "Это живой стрим, транскрибация невозможна",
            "hint": None
        }

    if any(x in stderr_lower for x in ["copyright", "blocked"]):
        return {
            "code": "copyright",
            "message": "Видео заблокировано по авторским правам",
            "hint": None
        }

    if any(x in stderr_lower for x in ["nsig extraction failed", "sign in to confirm you're not a bot", "429"]):
        return {
            "code": "ytdlp_outdated",
            "message": "yt-dlp нужно обновить (YouTube заблокировал сигнатуры)",
            "hint": "Выполните: yt-dlp -U"
        }

    if any(x in stderr_lower for x in ["unable to download", "urlopen error", "timed out", "connection", "network"]):
        return {
            "code": "network",
            "message": "Ошибка сети или соединения",
            "hint": "Проверьте интернет и повторите попытку"
        }

    if "no space left on device" in stderr_lower or "enospc" in stderr_lower:
        return {
            "code": "disk_full",
            "message": "Не хватает места на диске",
            "hint": "Освободите место и повторите"
        }

    return {
        "code": "unknown",
        "message": f"Неизвестная ошибка (код {returncode})",
        "hint": "Проверьте логи для деталей"
    }

def parse_ytdlp_metadata_line(line):
    """Парсинг одной строки JSON от 'yt-dlp -j'.

    Возвращает dict:
    {
        "video_id": "ID",
        "title": "Заголовок",
        "duration_sec": 3600,
        "is_live": False,
        "availability": "public" | "needs_auth" | "premium_only" | ...,
        "uploader": "Канал",
        "error": None | "ошибка парсинга"
    }
    """
    try:
        data = json.loads(line.strip())
    except json.JSONDecodeError as e:
        return {
            "video_id": None,
            "title": None,
            "duration_sec": 0,
            "is_live": False,
            "availability": None,
            "uploader": None,
            "error": f"Битый JSON: {e}"
        }

    if "id" not in data:
        return {
            "video_id": None,
            "title": None,
            "duration_sec": 0,
            "is_live": False,
            "availability": None,
            "uploader": None,
            "error": "Нет поля 'id' в ответе"
        }

    return {
        "video_id": data.get("id", ""),
        "webpage_url": data.get("webpage_url") or data.get("original_url"),
        "title": data.get("title", "[Без названия]"),
        "duration_sec": data.get("duration", 0),
        "is_live": data.get("is_live", False),
        "availability": data.get("availability", "public"),
        "uploader": data.get("uploader", "[Неизвестный канал]"),
        "error": None
    }

def start_youtube_metadata_fetch(entries, on_done):
    """Асинхронное получение метаданных видео через yt-dlp -j.

    entries: список dict от parse_youtube_urls с полем 'url'
    on_done: callback(resolved, failed) вызывается в GTК-потоке

    resolved: список dict с метаданными (успешные)
    failed: список dict с причинами ошибок (недоступные)
    """
    if not entries:
        return
    # Resolve each URL independently. yt-dlp's multi-URL -j stream omits a
    # stable association for failures, so partial errors could reorder results.
    # A worker keeps GTK responsive and preserves one result slot per input.
    import threading

    def resolve_all():
        from whisper_youtube import resolve_metadata_batch

        def resolve_one(entry):
            url = entry.get("url")
            if not url:
                raise ValueError("invalid_url")
            result = subprocess.run(
                ["yt-dlp", "-j", "--skip-download", "--no-playlist", "--no-warnings",
                 "--socket-timeout", "15", url],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, text=True, timeout=YTDLP_METADATA_TIMEOUT,
                check=False)
            if result.returncode != 0 or not result.stdout.strip():
                raise ValueError("metadata_unavailable")
            metadata = parse_ytdlp_metadata_line(result.stdout.splitlines()[0])
            if metadata.get("error"):
                raise ValueError("invalid_metadata")
            return metadata

        resolved, failed = resolve_metadata_batch(entries, resolve_one)
        GLib.idle_add(on_done, resolved, failed)

    threading.Thread(target=resolve_all, name="whisper-video-metadata", daemon=True).start()

def _poll_metadata_fetch():
    """Неблокирующий poller для процесса fetch метаданных.

    Вызывается из GLib.timeout_add каждые POLL_MS миллисекунд.
    """
    global _youtube_metadata_fetch

    if not _youtube_metadata_fetch:
        return False

    proc = _youtube_metadata_fetch["proc"]

    try:
        # Читать всё доступное без блокировки
        chunk = proc.stdout.read()
        if chunk and isinstance(chunk, str):
            _youtube_metadata_fetch["buffer"] += chunk
            # Парсить полные строки из буфера
            while "\n" in _youtube_metadata_fetch["buffer"]:
                line, _youtube_metadata_fetch["buffer"] = _youtube_metadata_fetch["buffer"].split("\n", 1)
                if line.strip():
                    entry = parse_ytdlp_metadata_line(line)
                    if entry["error"]:
                        _youtube_metadata_fetch["failed"].append(entry)
                    else:
                        # Восстановить URL из mapping
                        url = _youtube_metadata_fetch["url_mapping"].get(entry["video_id"])
                        if not url and entry.get("webpage_url"):
                            try:
                                from whisper_youtube import source_for_url
                                platform = source_for_url(entry["webpage_url"])
                                urls = _youtube_metadata_fetch["platform_urls"].get(platform, [])
                                index = _youtube_metadata_fetch["platform_url_index"].get(platform, 0)
                                if platform and index < len(urls):
                                    url = urls[index]
                                    _youtube_metadata_fetch["platform_url_index"][platform] = index + 1
                            except Exception:
                                pass
                        if url:
                            entry["url"] = url
                        _youtube_metadata_fetch["resolved"].append(entry)
    except BlockingIOError:
        pass
    except Exception as e:
        log_debug(f"⚠ Ошибка чтения stdout: {e}")

    if proc.poll() is not None:
        # Процесс завершился, прочитать оставшееся в буфере
        try:
            chunk = proc.stdout.read()
            if chunk:
                _youtube_metadata_fetch["buffer"] += chunk
            # Парсить оставшееся в буфере
            for line in _youtube_metadata_fetch["buffer"].split("\n"):
                if line.strip():
                    entry = parse_ytdlp_metadata_line(line)
                    if entry["error"]:
                        _youtube_metadata_fetch["failed"].append(entry)
                    else:
                        # Восстановить URL из mapping
                        url = _youtube_metadata_fetch["url_mapping"].get(entry["video_id"])
                        if not url and entry.get("webpage_url"):
                            try:
                                from whisper_youtube import source_for_url
                                platform = source_for_url(entry["webpage_url"])
                                urls = _youtube_metadata_fetch["platform_urls"].get(platform, [])
                                index = _youtube_metadata_fetch["platform_url_index"].get(platform, 0)
                                if platform and index < len(urls):
                                    url = urls[index]
                                    _youtube_metadata_fetch["platform_url_index"][platform] = index + 1
                            except Exception:
                                pass
                        if url:
                            entry["url"] = url
                        _youtube_metadata_fetch["resolved"].append(entry)
        except:
            pass

        try:
            stderr = proc.stderr.read()
            if stderr:
                # stderr may contain complete user URLs or extractor response details.
                log_debug("🎥 yt-dlp metadata command returned diagnostic output (content omitted)")
        except:
            pass

        on_done = _youtube_metadata_fetch["on_done"]
        resolved = _youtube_metadata_fetch["resolved"]
        failed = _youtube_metadata_fetch["failed"]

        log_debug(f"✅ Fetch завершён: {len(resolved)} успешно, {len(failed)} ошибок")

        _youtube_metadata_fetch = None
        GLib.idle_add(on_done, resolved, failed)
        return False

    if time.time() > _youtube_metadata_fetch["timeout_at"]:
        log_debug(f"⏱ Таймаут fetch метаданных, убиваю процесс...")
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except:
            proc.kill()

        on_done = _youtube_metadata_fetch["on_done"]
        resolved = _youtube_metadata_fetch["resolved"]
        failed = _youtube_metadata_fetch["failed"]

        _youtube_metadata_fetch = None
        GLib.idle_add(on_done, resolved, failed)
        return False

    return True

# ============================================================================
# Фаза 8.5: Асинхронное скачивание YouTube с окном прогресса
# ============================================================================

def start_youtube_download(entry, dest_dir):
    """Запустить yt-dlp для одного видео.

    entry: {"video_id", "title", "url", "duration_sec"}
    Возвращает Popen или None при ошибке
    """
    import fcntl
    import os as os_module
    try:
        safe_title = sanitize_filename(entry["title"])
        output_pattern = os.path.join(dest_dir, f"{safe_title}_{entry['video_id']}.%(ext)s")

        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--newline",
            "--no-color",
            "--socket-timeout", "20",
            "--retries", "3",
            "--fragment-retries", "3",
            "-f", "bestaudio/best",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "5",
            "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
            "-o", output_pattern,
            entry["url"]
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            start_new_session=True
        )

        # Установить неблокирующий режим для stdout
        fd = proc.stdout.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, os_module.O_NONBLOCK)

        # Инициализировать буфер для неблокирующего чтения
        if not hasattr(proc, '_line_buffer'):
            proc._line_buffer = ""

        log_debug(f"⬇️ Запущено скачивание: {entry['title']} (pid={proc.pid})")
        return proc
    except Exception as e:
        log_debug(f"✗ Не удалось запустить yt-dlp: {e}")
        return None

def parse_ytdlp_progress_line(line):
    """Парсинг строк прогресса yt-dlp.

    '[download]  42.3% of 12.34MiB at ...' → 0.423
    '[ExtractAudio] Destination:'         → 0.95 (фаза конвертации)

    Возвращает float [0..1] или None
    """
    if not line:
        return None

    # Парсить [download] X%
    match = re.search(r'\[download\]\s+([\d.]+)%', line)
    if match:
        return float(match.group(1)) / 100.0

    # Фаза конвертации
    if "[ExtractAudio]" in line:
        return 0.95

    return None

def poll_youtube_downloads():
    """Опросить YouTube загрузки: обновление прогресса, проверка завершения,
    заполнение свободных слотов из очереди, слияние с batch при полном завершении"""
    try:
        if not _youtube_downloads:
            return True

        processing_count = 0
        total_count = 0

        for download_id, entries in list(_youtube_downloads.items()):
            for idx, entry_data in list(entries.items()):
                if not entry_data or idx == "_on_done":
                    continue

                proc = entry_data.get("proc")
                if proc is None:
                    continue  # Кэшировано или завершено

                processing_count += 1

                # Неблокирующее чтение stdout с буферизацией
                try:
                    chunk = proc.stdout.read()
                    if chunk:
                        if not hasattr(proc, '_line_buffer'):
                            proc._line_buffer = ""
                        proc._line_buffer += chunk
                        # Парсить полные строки из буфера
                        while "\n" in proc._line_buffer:
                            line, proc._line_buffer = proc._line_buffer.split("\n", 1)
                            if line.strip():
                                progress = parse_ytdlp_progress_line(line)
                                if progress is not None:
                                    entry_data["progress"] = progress
                except BlockingIOError:
                    pass
                except Exception:
                    pass

                # Проверить завершение
                if proc.poll() is not None:
                    entry = entry_data.get("entry", {})
                    audio_path = youtube_audio_path(entry.get("video_id", ""), entry.get("title", ""))

                    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                        # Успех
                        entry_data["progress"] = 1.0
                        entry_data["status"] = "download_ok"
                        entry_data["proc"] = None
                        size_mb = get_file_size_mb(audio_path)
                        append_youtube_history({
                            "video_id": entry.get("video_id"),
                            "url": entry.get("url"),
                            "title": entry.get("title"),
                            "duration_sec": entry.get("duration_sec"),
                            "audio_path": audio_path,
                            "size_mb": round(size_mb, 1) if size_mb else None,
                            "downloaded_at": datetime.now().isoformat(timespec="seconds")
                        })
                        log_debug(f"✅ Скачано: {entry.get('title')}")
                    else:
                        # Ошибка
                        stderr = ""
                        try:
                            stderr = proc.stderr.read()
                        except Exception:
                            pass
                        error_info = classify_ytdlp_error(stderr, proc.returncode)
                        entry_data["error"] = error_info
                        entry_data["status"] = "download_error"
                        entry_data["proc"] = None
                        log_debug(f"✗ Ошибка скачивания {entry.get('title')}: {error_info['message']}")

            # Проверить завершение всех загрузок для этого download_id
            if download_id in _youtube_downloads:
                entries = _youtube_downloads[download_id]
                all_done = all(
                    idx == "_on_done" or (isinstance(e, dict) and e.get("proc") is None)
                    for idx, e in entries.items()
                )
                total_in_batch = sum(1 for idx, e in entries.items() if idx != "_on_done" and isinstance(e, dict))
                if all_done and processing_count == 0:
                    log_debug(f"🎥 YouTube: все файлы завершены ({total_in_batch} файлов в batch {download_id})")
                    on_done_callback = entries.get("_on_done")
                    if on_done_callback:
                        log_debug(f"🎥 Вызываю on_done_callback...")
                        GLib.idle_add(on_done_callback)
                    # Keep download state until the idle callback has assembled
                    # its prepared-file list. The callback runs after this poll
                    # returns and reads this entry to resolve completed files.

        return True if processing_count > 0 else True
    except Exception as e:
        log_debug(f"❌ poll_youtube_downloads ERROR: {e}")
        import traceback
        log_debug(f"❌ {traceback.format_exc()}")
        return True

def cancel_youtube_download(download_id):
    """Отменить скачивание(я) по download_id."""
    if download_id not in _youtube_downloads:
        return

    entries = _youtube_downloads[download_id]
    for idx, entry_data in list(entries.items()):
        if idx == "_on_done":
            continue

        proc = entry_data.get("proc")
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
            entry_data["proc"] = None

    # Удалить .part и .ytdl файлы
    try:
        for filename in os.listdir(YOUTUBE_AUDIO_DIR):
            if filename.endswith((".part", ".ytdl")):
                filepath = os.path.join(YOUTUBE_AUDIO_DIR, filename)
                try:
                    os.unlink(filepath)
                except (OSError, IOError):
                    pass
    except Exception:
        pass

    _youtube_downloads.pop(download_id, None)
    log_debug(f"🚫 Скачивание отменено: {download_id}")

def create_youtube_download_window(download_id):
    """Gtk.Window с прогрессом скачивания YouTube."""
    log_debug(f"📌 [create_youtube_download_window] Начало выполнения для {download_id}")
    existing = _youtube_windows.get(download_id)
    if existing:
        log_debug(f"📌 [create_youtube_download_window] Окно уже существует, использую существующее")
        existing["window"].present()
        existing["window"].show()
        return existing["window"]

    log_debug(f"📌 [create_youtube_download_window] Получаю downloads...")
    downloads = _youtube_downloads.get(download_id, {})
    total_files = sum(1 for idx, e in downloads.items() if idx != "_on_done")
    log_debug(f"📌 [create_youtube_download_window] total_files={total_files}")

    log_debug(f"📌 [create_youtube_download_window] Создаю Gtk.Window...")
    window = Gtk.Window(title=f"⬇️ Скачивание YouTube ({total_files} файлов)")
    log_debug(f"📌 [create_youtube_download_window] Окно создано, настраиваю размер...")
    window.set_default_size(580, 380)
    window.set_border_width(10)
    log_debug(f"📌 [create_youtube_download_window] Размер установлен")

    log_debug(f"📌 [create_youtube_download_window] Создаю vbox...")
    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    log_debug(f"📌 [create_youtube_download_window] Добавляю vbox в window...")
    window.add(vbox)

    log_debug(f"📌 [create_youtube_download_window] Создаю header label...")
    header = Gtk.Label()
    header.set_markup(f"<b>⬇️ Скачивание YouTube: {GLib.markup_escape_text(download_id)}</b>")
    header.set_xalign(0)
    log_debug(f"📌 [create_youtube_download_window] Добавляю header в vbox...")
    vbox.pack_start(header, False, False, 0)
    log_debug(f"📌 [create_youtube_download_window] Header добавлен")

    log_debug(f"📌 [create_youtube_download_window] Создаю scrolled window...")
    scrolled = Gtk.ScrolledWindow()
    log_debug(f"📌 [create_youtube_download_window] Настраиваю scrolled...")
    scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    scrolled.set_min_content_height(200)
    log_debug(f"📌 [create_youtube_download_window] Добавляю scrolled в vbox...")
    vbox.pack_start(scrolled, True, True, 0)

    log_debug(f"📌 [create_youtube_download_window] Создаю listbox...")
    listbox = Gtk.ListBox()
    log_debug(f"📌 [create_youtube_download_window] Настраиваю listbox...")
    listbox.set_selection_mode(Gtk.SelectionMode.NONE)
    log_debug(f"📌 [create_youtube_download_window] Добавляю listbox в scrolled...")
    scrolled.add(listbox)
    log_debug(f"📌 [create_youtube_download_window] Listbox добавлен")

    log_debug(f"📌 [create_youtube_download_window] Начинаю цикл по downloads (total={total_files})...")
    row_widgets = []
    try:
        log_debug(f"📌 [create_youtube_download_window] Получаю sorted(downloads.items())...")
        items = sorted([item for item in downloads.items() if item[0] != "_on_done"])
        log_debug(f"📌 [create_youtube_download_window] Получено {len(items)} items")
        for idx, entry_data in items:
            log_debug(f"📌 [create_youtube_download_window]   Обработка idx={idx}")
            if idx == "_on_done":
                log_debug(f"📌 [create_youtube_download_window]     Пропускаю _on_done")
                continue

            entry = entry_data.get("entry", {})
            log_debug(f"📌 [create_youtube_download_window]     Создаю ListBoxRow...")
            row = Gtk.ListBoxRow()
            log_debug(f"📌 [create_youtube_download_window]     Создаю HBox...")
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            hbox.set_border_width(4)
            log_debug(f"📌 [create_youtube_download_window]     Добавляю hbox в row...")
            row.add(hbox)

            # Название видео
            log_debug(f"📌 [create_youtube_download_window]     Создаю name_label...")
            title = entry.get("title", "?")
            name_label = Gtk.Label(label=title)
            name_label.set_xalign(0)
            name_label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
            name_label.set_max_width_chars(26)
            log_debug(f"📌 [create_youtube_download_window]     pack_start name_label...")
            hbox.pack_start(name_label, True, True, 0)
            log_debug(f"📌 [create_youtube_download_window]     name_label packed")
    except Exception as e:
        log_debug(f"❌ [create_youtube_download_window] ОШИБКА в цикле: {e}")
        import traceback
        log_debug(f"❌ {traceback.format_exc()}")
        raise

        # Статус
        status_label = Gtk.Label(label="")
        status_label.set_width_chars(13)
        status_label.set_xalign(0)
        hbox.pack_start(status_label, False, False, 0)

        # Прогресс
        progress = Gtk.ProgressBar()
        progress.set_size_request(110, -1)
        progress.set_show_text(False)
        hbox.pack_start(progress, False, False, 0)

        # Скорость/ETA
        info_label = Gtk.Label(label="")
        info_label.set_width_chars(10)
        info_label.set_xalign(1)
        hbox.pack_start(info_label, False, False, 0)

        listbox.add(row)
        row_widgets.append({"idx": idx, "status": status_label, "progress": progress, "info": info_label})

    overall_label = Gtk.Label()
    overall_label.set_xalign(0)
    vbox.pack_start(overall_label, False, False, 0)

    overall_progress = Gtk.ProgressBar()
    overall_progress.set_show_text(True)
    vbox.pack_start(overall_progress, False, False, 0)

    button_box = Gtk.ButtonBox(orientation=Gtk.Orientation.HORIZONTAL)
    button_box.set_layout(Gtk.ButtonBoxStyle.END)
    button_box.set_spacing(6)
    vbox.pack_start(button_box, False, False, 0)

    cancel_button = Gtk.Button(label="⏹️ Отмена")
    hide_button = Gtk.Button(label="🔽 Скрыть")
    button_box.add(cancel_button)
    button_box.add(hide_button)

    def update_window():
        downloads = _youtube_downloads.get(download_id, {})
        if not downloads:
            return False

        # Обновить строки
        for w in row_widgets:
            idx = w["idx"]
            entry_data = downloads.get(idx)
            if not entry_data:
                continue

            proc = entry_data.get("proc")
            progress_val = entry_data.get("progress", 0.0)
            error_info = entry_data.get("error")

            if error_info:
                w["status"].set_text("❌ Ошибка")
                w["progress"].set_fraction(0.0)
                w["info"].set_text("")
            elif proc is None and progress_val >= 1.0:
                w["status"].set_text("✅ Готово")
                w["progress"].set_fraction(1.0)
                w["info"].set_text("")
            elif proc is None:
                w["status"].set_text("💾 Кэш")
                w["progress"].set_fraction(1.0)
                w["info"].set_text("")
            else:
                w["status"].set_text("⬇️ Загружается")
                w["progress"].set_fraction(progress_val)
                w["info"].set_text(f"{progress_val*100:.0f}%")

        # Общий прогресс
        total = sum(1 for idx, e in downloads.items() if idx != "_on_done")
        done = sum(1 for idx, e in downloads.items()
                   if idx != "_on_done" and e.get("proc") is None)
        overall_label.set_text(f"Обработано: {done}/{total} файлов")
        overall_progress.set_fraction(done / total if total else 0.0)

        # Если все готово — скрыть кнопку "Отмена" и показать "Далее"
        if done == total and total > 0:
            cancel_button.set_label("➡️ Далее к транскрибации")
            return False

        return True

    def on_cancel_clicked(_):
        downloads = _youtube_downloads.get(download_id, {})
        done = sum(1 for idx, e in downloads.items()
                   if idx != "_on_done" and e.get("proc") is None)
        total = sum(1 for idx, e in downloads.items() if idx != "_on_done")

        if done == total and total > 0:
            # Уже готово — переходим к batch
            cancel_button.set_sensitive(False)
            window.destroy()
        else:
            # Отмена в процессе
            cancel_youtube_download(download_id)
            update_window()

    def cleanup(*_args):
        entry = _youtube_windows.pop(download_id, None)
        if entry and entry.get("timeout_id"):
            try:
                GLib.source_remove(entry["timeout_id"])
            except Exception:
                pass

    log_debug(f"📌 Подключаю обработчики кнопок...")
    cancel_button.connect("clicked", on_cancel_clicked)
    hide_button.connect("clicked", lambda _: window.hide())
    window.connect("destroy", cleanup)

    log_debug(f"📌 Первый вызов update_window()...")
    update_window()
    log_debug(f"📌 Вызов show_all()...")
    window.show_all()
    log_debug(f"📌 Запуск timeout для обновления...")

    timeout_id = GLib.timeout_add(500, update_window)
    _youtube_windows[download_id] = {"window": window, "timeout_id": timeout_id}
    log_debug(f"✅ Окно скачивания создано и показано")
    return window

def prepare_youtube_batch_files(dl_entries, downloads, audio_path_resolver,
                                file_exists, duration_resolver):
    """Collect successful inputs in order while retaining per-item failures."""
    prepared_files = []
    failures = []
    for index, item in enumerate(dl_entries or []):
        if not isinstance(item, dict):
            continue
        state = downloads.get(index, {}) if isinstance(downloads, dict) else {}
        entry = state.get("entry", item) if isinstance(state, dict) else item
        if not isinstance(entry, dict):
            entry = item
        error_info = state.get("error") if isinstance(state, dict) else None
        if error_info:
            failures.append({"url": item.get("url"),
                             "title": item.get("title", "Видео"),
                             "reason": "download_error",
                             "status": "download_error"})
            continue
        audio_path = (item.get("audio_path") or entry.get("audio_path")
                      or audio_path_resolver(entry))
        if not audio_path or not file_exists(audio_path):
            failures.append({"url": item.get("url"),
                             "title": item.get("title", "Видео"),
                             "reason": "prepared_file_missing",
                             "status": "prepare_error"})
            continue
        duration = duration_resolver(audio_path) or item.get("duration_sec", 0)
        prepared_files.append({
            "name": f"{item.get('title', '?')}.mp3",
            "original_path": audio_path,
            "audio_path": audio_path,
            "duration": duration,
            "type": "youtube",
            "source_url": item.get("url"),
            "video_id": item.get("video_id"),
        })
    return prepared_files, failures


def on_youtube_downloads_finished(download_id, quality_config, dl_entries):
    """Вызывается когда все скачивания завершены или все в кэше.

    dl_entries: результат из callback (уже подготовленные файлы) ИЛИ список dl_entries с cached=True/False
    """
    downloads = _youtube_downloads.get(download_id, {})
    prepared_files, download_failures = prepare_youtube_batch_files(
        dl_entries, downloads, youtube_audio_path,
        lambda path: os.path.exists(path) and os.path.getsize(path) > 0,
        get_file_duration,
    )
    _youtube_downloads.pop(download_id, None)

    if download_failures:
        notify_send("Whisper YouTube", f"⚠️ {len(download_failures)} видео не удалось подготовить")
        show_youtube_validation_dialog([], download_failures)

    if not prepared_files:
        show_error_dialog("Ошибка", "Не удалось подготовить файлы для транскрибации")
        return False

    # Проверить место
    cleanup_youtube_cache()
    disk_check = check_youtube_disk_space(prepared_files)
    if not disk_check["ok"]:
        show_error_dialog("Недостаточно места", disk_check["error"])
        return False

    # Показать диалог подтверждения batch
    # TODO: show_batch_confirmation_dialog использует dialog.run() и блокирует UI
    # Для тестирования Фазы 8.8 пропускаем диалог и запускаем сразу
    # if not show_batch_confirmation_dialog(prepared_files, quality_config):
    #     log_debug("Batch транскрибация отменена после скачивания YouTube")
    #     return False

    # Запустить batch
    log_debug(f"🎥 Запускаю batch транскрибацию для {len(prepared_files)} видео из YouTube")
    batch_id = start_batch_transcription(prepared_files, quality_config)
    notify_send("Whisper YouTube", f"✅ Готово к транскрибации ({len(prepared_files)} файлов)")
    create_batch_progress_window(batch_id)

    return False

def on_youtube_transcribe_clicked(_):
    """Обработчик меню "🎥 Распознать видео с YouTube...".

    1. Показать диалог ввода ссылок
    2. Парсить ссылки
    3. Проверить yt-dlp доступен
    4. Fetch метаданные асинхронно (Фаза 8.4)
    5. На Фазе 8.5 будет скачивание и batch-транскрибация
    """
    try:
        text = show_youtube_url_dialog()
        if not text:
            return

        from whisper_youtube import parse_youtube_urls
        entries = parse_youtube_urls(text)

        valid = [e for e in entries if e["valid"]]
        invalid = [e for e in entries if not e["valid"]]

        if invalid:
            if not show_youtube_validation_dialog(invalid, []):
                return

        if not valid:
            return

        log_debug(f"🎥 Получено {len(valid)} валидных YouTube-ссылок для обработки")

        ytdlp_check = check_ytdlp_available()
        if not ytdlp_check["available"]:
            show_error_dialog(
                "yt-dlp не установлен",
                f"Установите yt-dlp:\npip3 install yt-dlp\n\nОшибка: {ytdlp_check['error']}"
            )
            return

        def on_metadata_done(resolved, failed):
            """Callback после завершения fetch метаданных (Фаза 8.5)"""
            log_debug(f"📞 on_metadata_done: resolved={len(resolved)}, failed={len(failed)}")
            if failed:
                log_debug(f"⚠️  Есть failed видео: {failed}")
                status_rows = [dict(row, reason="metadata_ok") for row in resolved] + failed
                status_rows.sort(key=lambda row: row.get("input_index", 0))
                if not show_youtube_validation_dialog([], status_rows):
                    log_debug("❌ Пользователь отменил на диалоге ошибок")
                    return

            if not resolved:
                log_debug("❌ Нет resolved видео")
                show_error_dialog("Ошибка", "Не удалось получить информацию о видео")
                return

            log_debug(f"🎥 Готово {len(resolved)} видео к загрузке")

            # Фаза 8.5: начать скачивание
            log_debug("📋 Загружаю quality_config...")
            quality_config = load_quality_config()
            log_debug("📋 Генерирую download_id...")
            download_id = generate_batch_id()
            log_debug(f"📋 download_id={download_id}")

            # Подготовить entries для скачивания с проверкой кэша
            log_debug("📋 Подготавливаю entries...")
            dl_entries = []
            for i, entry in enumerate(resolved):
                try:
                    log_debug(f"📋   [{i+1}/{len(resolved)}] Проверяю кэш для {entry.get('title', 'unknown')[:30]}...")
                    cached = find_cached_download(entry["video_id"])
                    log_debug(f"📋   [{i+1}/{len(resolved)}] find_cached_download завершён, cached={bool(cached)}")
                    if cached:
                        # Использовать кэш
                        log_debug(f"📋   [{i+1}/{len(resolved)}] Добавляю cached entry...")
                        dl_entries.append({
                            "video_id": entry["video_id"],
                            "title": entry["title"],
                            "duration_sec": entry["duration_sec"],
                            "url": entry["url"],
                            "audio_path": cached["audio_path"],
                            "cached": True
                        })
                    else:
                        # Нужно скачать
                        log_debug(f"📋   [{i+1}/{len(resolved)}] Добавляю download entry...")
                        dl_entries.append({
                            "video_id": entry["video_id"],
                            "title": entry["title"],
                            "duration_sec": entry["duration_sec"],
                            "url": entry["url"],
                            "audio_path": None,
                            "cached": False
                        })
                    log_debug(f"📋   [{i+1}/{len(resolved)}] ✓ Entry добавлен")
                except Exception as e:
                    log_debug(f"❌ ОШИБКА при обработке видео {i+1}: {e}")
                    import traceback
                    log_debug(f"❌ Traceback: {traceback.format_exc()}")
                    raise

            # Проверить место для скачиваний
            to_download = [(i, e) for i, e in enumerate(dl_entries) if not e["cached"]]
            if to_download:
                disk_check = check_youtube_disk_space([entry for _, entry in to_download])
                if not disk_check["ok"]:
                    show_error_dialog("Недостаточно места", disk_check["error"])
                    return

            # Создать окно прогресса
            _youtube_downloads[download_id] = {"_on_done": None}
            for i, entry in enumerate(dl_entries):
                if entry["cached"]:
                    _youtube_downloads[download_id][i] = {
                        "proc": None, "entry": entry, "progress": 1.0, "eta_sec": None
                    }
            for i, entry in to_download:
                dest_dir = YOUTUBE_AUDIO_DIR
                proc = start_youtube_download(entry, dest_dir)
                from whisper_youtube import download_state
                _youtube_downloads[download_id][i] = download_state(entry, proc)

            # Если все в кэше, сразу к batch
            if not to_download:
                prepared = []
                for entry in resolved:
                    cached = find_cached_download(entry["video_id"])
                    if cached:
                        prepared.append({
                            "name": f"{entry['title']}.mp3",
                            "original_path": cached["audio_path"],
                            "audio_path": cached["audio_path"],
                            "duration": entry.get("duration_sec", 0),
                            "type": "youtube",
                            "source_url": entry.get("url"),
                            "video_id": entry["video_id"]
                        })
                if prepared:
                    GLib.idle_add(on_youtube_downloads_finished, download_id, quality_config, prepared)
                return

            # Показать окно и зарегистрировать callback
            log_debug(f"📌 Регистрирую callback на download_id={download_id}")
            _youtube_downloads[download_id]["_on_done"] = lambda: on_youtube_downloads_finished(download_id, quality_config, dl_entries)
            log_debug(f"📌 Создаю окно скачивания...")
            create_youtube_download_window(download_id)
            log_debug(f"✅ Окно скачивания создано")

        start_youtube_metadata_fetch(valid, on_metadata_done)

    except Exception as e:
        log_debug(f"✗ Ошибка при обработке YouTube-ссылок: {e}")
        show_error_dialog("Ошибка", f"Не удалось обработать ссылки: {e}")

def on_transcribe_file_clicked(_):
    """Открыть файлы, подготовить (валидация/извлечение аудио из видео), подтвердить
    настройки в диалоге и обработать текущими настройками (batch)"""
    try:
        files = show_batch_file_chooser()
        if not files:
            return

        video_files = [f for f in files if is_video_file(f)]
        if video_files and not shutil.which("ffmpeg"):
            show_error_dialog(
                "ffmpeg не установлен",
                "Для обработки видеофайлов требуется ffmpeg. Установите его (zypper install ffmpeg) и повторите попытку."
            )
            return

        batch_tmp_dir = os.path.join(tempfile.gettempdir(), f"whisper-batch-{int(time.time())}")
        no_audio_files = []
        error_files = []
        prepared_files = []

        for file_path in files:
            if not os.path.exists(file_path):
                log_debug(f"⚠ Файл не найден: {file_path}")
                error_files.append(f"{os.path.basename(file_path)}: файл не найден")
                continue

            audio_path = file_path
            if is_video_file(file_path):
                pre_check = validate_audio_file(file_path)
                if pre_check["reason"] == "no_audio":
                    log_debug(f"⚠ Видео без аудиодорожки: {file_path}")
                    no_audio_files.append(os.path.basename(file_path))
                    continue

                extraction = extract_audio_from_video(file_path, batch_tmp_dir)
                if not extraction["success"]:
                    log_debug(f"✗ Не удалось извлечь аудио из {file_path}: {extraction['error']}")
                    error_files.append(f"{os.path.basename(file_path)}: {extraction['error']}")
                    continue
                audio_path = extraction["audio_path"]

            validation = validate_audio_file(audio_path)
            if not validation["valid"]:
                log_debug(f"✗ Файл не прошёл проверку {file_path}: {validation['error']}")
                error_files.append(f"{os.path.basename(file_path)}: {validation['error']}")
                continue

            prepared_files.append({
                "name": os.path.basename(file_path),
                "original_path": file_path,
                "audio_path": audio_path,
                "duration": validation["duration"]
            })

        if no_audio_files:
            show_warning_dialog(
                "Видео без звука",
                "У следующих видео нет аудиодорожки, они пропущены:\n" + "\n".join(no_audio_files)
            )
        if error_files:
            show_error_dialog(
                "Некоторые файлы пропущены",
                "\n".join(error_files)
            )

        if not prepared_files:
            return

        quality_config = load_quality_config()
        if not show_batch_confirmation_dialog(prepared_files, quality_config):
            log_debug("Batch транскрибация отменена пользователем в диалоге подтверждения")
            return

        batch_id = start_batch_transcription(prepared_files, quality_config)
        notify_send("Whisper Batch", f"📦 Запущена обработка {len(prepared_files)} файлов ({batch_id})")
        create_batch_progress_window(batch_id)
    except Exception as e:
        log_debug(f"✗ Ошибка при обработке файла: {e}")
        show_error_dialog("Ошибка", f"Не удалось обработать файл: {e}")

def _cancel_daemon_manual_correction():
    """Request manual-correction cancellation without transmitting user text."""
    client = None
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(2)
        client.connect(os.path.expanduser("~/.cache/whisper-dictate/daemon.sock"))
        client.sendall(b'{"action":"cancel_manual_correction"}\n')
        response = bytearray()
        while b"\n" not in response and len(response) < 4096:
            chunk = client.recv(1024)
            if not chunk:
                break
            response.extend(chunk)
        value = json.loads(bytes(response).split(b"\n", 1)[0].decode("utf-8"))
        return value.get("status") == "ok", value.get("cancel_requested") is True
    except (OSError, ValueError, json.JSONDecodeError):
        return False, False
    finally:
        if client is not None:
            client.close()


def on_cleanup_tasks_clicked(_):
    """Очистить все зависшие задачи (и микрофон, и файлы)"""
    try:
        daemon_reachable, manual_cancel_requested = _cancel_daemon_manual_correction()
        # Clear only the tray's local pre-IPC hint. An active daemon request
        # remains visible as "cancelling" until its worker actually exits.
        _set_manual_correction_status(None)
        # 1. Получить все текущие job-файлы и обработать их
        killed_file_jobs = 0
        for job_file in glob.glob(os.path.join(JOBS_DIR, "*.json")):
            try:
                with open(job_file, encoding="utf-8") as f:
                    job = json.load(f)

                # 2. Если это файл — отменить процесс обработки
                if job.get("source") == "file":
                    try:
                        # Попытаться убить процесс transcribe-audio или whisper
                        subprocess.run(["pkill", "-f", "transcribe-audio"], timeout=2)
                        log_debug(f"Процесс для {job.get('file')} отменён")
                        killed_file_jobs += 1
                    except:
                        pass
            except (json.JSONDecodeError, OSError):
                pass

        # 3. Очистить все job-файлы
        shutil.rmtree(JOBS_DIR, ignore_errors=True)
        os.makedirs(JOBS_DIR, exist_ok=True)

        msg = "✓ Все задачи очищены"
        if manual_cancel_requested:
            msg += "; отмена ручной коррекции выполняется"
        elif daemon_reachable:
            msg += "; активного ручного запроса не было"
        else:
            msg += "; daemon недоступен — запрос отмены не подтверждён"
        if killed_file_jobs > 0:
            msg += f" ({killed_file_jobs} процессов файлов остановлено)"
        log_debug(msg)

        # 4. Показать notification
        try:
            subprocess.run(["notify-send", "Whisper", msg], timeout=2)
        except:
            pass
    except Exception as e:
        log_debug(f"✗ Ошибка при очистке задач: {e}")
        show_error_dialog("Ошибка", f"Не удалось очистить задачи: {e}")

def on_recover_raw_asr_clicked(_):
    """Copy the active live job's saved raw ASR and request correction cancellation."""
    raw_dir = os.path.join(STATE_DIR, "raw-asr")
    cancel_dir = os.path.join(STATE_DIR, "recovery-cancel")
    candidates = []
    try:
        for job_file in glob.glob(os.path.join(JOBS_DIR, "*.json")):
            job_id = os.path.basename(job_file)[:-5]
            if not job_id.isdigit():
                continue
            try:
                with open(job_file, encoding="utf-8") as stream:
                    job = json.load(stream)
                if job.get("state") != "processing" or job.get("pipeline_stage") != "correction":
                    continue
                raw_path = os.path.join(raw_dir, f"{job_id}.txt")
                if os.path.isfile(raw_path):
                    candidates.append((os.path.getmtime(job_file), job_id, raw_path))
            except (OSError, json.JSONDecodeError):
                continue
    except OSError as error:
        show_error_dialog("Не удалось получить текст", str(error))
        return

    if not candidates:
        notify_send("Whisper Dictate", "Нет активного задания на коррекции с готовым исходным текстом.")
        return

    _started, job_id, raw_path = max(candidates)
    try:
        with open(raw_path, encoding="utf-8") as stream:
            text = stream.read()
    except OSError as error:
        show_error_dialog("Не удалось прочитать исходный текст", str(error))
        return
    if not text.strip():
        notify_send("Whisper Dictate", "Исходный текст этого задания пуст.")
        return
    if not copy_text_to_clipboard(text):
        show_error_dialog("Не удалось скопировать текст", "Коррекция не отменена.")
        return

    try:
        os.makedirs(cancel_dir, mode=0o700, exist_ok=True)
        marker = os.path.join(cancel_dir, f"{job_id}.cancel")
        fd = os.open(marker, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
    except OSError as error:
        notify_send("Whisper Dictate", f"Исходный текст скопирован; отмена коррекции не подтверждена: {error}")
        return
    notify_send("Whisper Dictate", f"Исходный текст задания {job_id} скопирован. Дальнейшая коррекция отменена.")


menu = Gtk.Menu()

quick_transcribe_item = primary_menu_item("Новая транскрибация", "media-playback-start")
quick_transcribe_item.connect("activate", on_quick_transcribe_clicked)
menu.append(quick_transcribe_item)

manual_correct_item = primary_menu_item("Запустить корректор", "document-edit")
manual_correct_item.set_tooltip_text(
    "Обработать текст из буфера обмена включёнными ручными сервисами коррекции")
manual_correct_item.connect("activate", on_manual_correct_clicked)
menu.append(manual_correct_item)

transcribe_file_item = primary_menu_item("Распознать файлы", "document-open")
transcribe_file_item.connect("activate", on_transcribe_file_clicked)
menu.append(transcribe_file_item)

youtube_transcribe_item = primary_menu_item("Распознать видео", "video-x-generic")
youtube_transcribe_item.set_tooltip_text("Распознать видео по ссылкам YouTube")
youtube_transcribe_item.connect("activate", on_youtube_transcribe_clicked)
menu.append(youtube_transcribe_item)

menu.append(Gtk.SeparatorMenuItem())

cleanup_tasks_item = primary_menu_item("Очистить зависшие задачи", "edit-clear")
cleanup_tasks_item.connect("activate", on_cleanup_tasks_clicked)

recover_raw_asr_item = primary_menu_item("Скопировать исходник", "edit-copy")
recover_raw_asr_item.set_tooltip_text(
    "Скопировать исходный текст и остановить текущую коррекцию")
recover_raw_asr_item.connect("activate", on_recover_raw_asr_clicked)

whisper_settings_submenu = Gtk.Menu()

def cloudflare_credentials_available():
    """Check credential file and required variable names without exposing values."""
    path = os.path.expanduser("~/.config/whisper-dictate/cloudflare-workers-ai.env")
    if not os.path.isfile(path):
        return False
    try:
        names = set()
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if value.strip().strip("\"'").strip():
                    names.add(key.strip())
        return {"CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"}.issubset(names)
    except OSError:
        return False

def open_credentials_file(_item, provider):
    """Create a private template if missing and open it in the desktop editor."""
    if provider == "cloudflare":
        path = os.path.expanduser("~/.config/whisper-dictate/cloudflare-workers-ai.env")
        template = ("# Cloudflare Workers AI credentials. Keep this file private.\n"
                    "CLOUDFLARE_ACCOUNT_ID=\nCLOUDFLARE_API_TOKEN=\n")
        parent = os.path.dirname(path)
    else:
        from multi_provider import credential_path
        path = str(credential_path(provider))
        template = ""
        parent = os.path.dirname(path)
    try:
        os.makedirs(parent, mode=0o700, exist_ok=True)
        if provider != "cloudflare":
            os.chmod(parent, 0o700)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            fd = None
        if fd is not None:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(template)
            os.chmod(path, 0o600)
        elif provider != "cloudflare":
            os.chmod(path, 0o600)
        subprocess.Popen(["xdg-open", path], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        notify_send("Whisper Dictate", f"Не удалось открыть файл credentials: {provider}")
        return
    notify_send("Whisper Dictate", f"Открыл защищённый файл credentials: {provider}")
    if provider == "cloudflare":
        refresh_cloudflare_credentials_status()
    else:
        refresh_multi_provider_status()

cloudflare_toggle_syncing = False
cloudflare_dialog_toggles = {}

def on_cloudflare_toggled(item):
    global cloudflare_toggle_syncing
    if cloudflare_toggle_syncing:
        return
    value = load_voice_config()
    value["cloudflare_asr_enabled"] = item.get_active()
    save_voice_config(value)
    cloudflare_toggle_syncing = True
    try:
        target = cloudflare_dialog_toggles.get("asr")
        if target is not None and target is not item:
            target.set_active(item.get_active())
    finally:
        cloudflare_toggle_syncing = False
    refresh_multi_provider_status()
    if item.get_active():
        if cloudflare_credentials_available():
            notify_send("Cloudflare Workers AI", "☁️ Cloudflare ASR включён; корректор переключается отдельно")
            log_debug("Cloudflare ASR включён; переключатель корректора независим")
        else:
            notify_send("Cloudflare Workers AI", "⚠️ Учётные данные недоступны; при ошибке ASR сработает локальный fallback")
            log_debug("Cloudflare ASR включён без доступных учётных данных")
    else:
        notify_send("Whisper Dictate", "Выбран локальный ASR")
        log_debug("Локальный ASR выбран для диктовки")

cloudflare_submenu = Gtk.Menu()
cloudflare_status = Gtk.MenuItem()
cloudflare_status.set_sensitive(False)

def refresh_cloudflare_credentials_status(*_args):
    cloudflare_status.set_label("Учётные данные заполнены" if cloudflare_credentials_available()
                                else "Нет файла или заполненных обязательных полей")

refresh_cloudflare_credentials_status()
cloudflare_submenu.append(cloudflare_status)
cloudflare_submenu.append(Gtk.SeparatorMenuItem())

cloudflare_asr_toggle = Gtk.CheckMenuItem(label="Использовать Cloudflare для Whisper")
iconize_check_menu_item(cloudflare_asr_toggle, menu_icon_path("cloudflare"))
cloudflare_asr_toggle.set_active(bool(load_voice_config().get("cloudflare_asr_enabled", load_voice_config().get("cloudflare_workers_ai_enabled", False))))
cloudflare_asr_toggle.connect("toggled", on_cloudflare_toggled)
cloudflare_submenu.append(cloudflare_asr_toggle)

cloudflare_corrector_toggle = Gtk.CheckMenuItem(label="Использовать Cloudflare для корректора")
iconize_check_menu_item(cloudflare_corrector_toggle, menu_icon_path("cloudflare"))
cloudflare_corrector_toggle.set_active(bool(load_voice_config().get("cloudflare_corrector", {}).get("enabled", False)))
def on_cloudflare_corrector_toggled(item):
    global cloudflare_toggle_syncing
    if cloudflare_toggle_syncing:
        return
    value = load_voice_config()
    profile = value.get("cloudflare_corrector", {})
    profile.update({"enabled": item.get_active(), "model": profile.get("model", "@cf/qwen/qwen3-30b-a3b-fp8"),
                    "temperature": profile.get("temperature", 0.0), "max_tokens": profile.get("max_tokens", 2048)})
    value["cloudflare_corrector"] = profile
    save_voice_config(value)
    cloudflare_toggle_syncing = True
    try:
        target = cloudflare_dialog_toggles.get("corrector")
        if target is not None and target is not item:
            target.set_active(item.get_active())
    finally:
        cloudflare_toggle_syncing = False
    refresh_multi_provider_status()
cloudflare_corrector_toggle.connect("toggled", on_cloudflare_corrector_toggled)
cloudflare_submenu.append(cloudflare_corrector_toggle)

cloudflare_credentials_file_item = icon_menu_item("Открыть файл Account ID и API token…", menu_icon_path("file"))
cloudflare_credentials_file_item.connect("activate", open_credentials_file, "cloudflare")
cloudflare_usage_items = [Gtk.MenuItem() for _ in range(4)]
for item in cloudflare_usage_items:
    item.set_sensitive(False)
cloudflare_attempt_summary = icon_menu_item("Сегодня: запросы Cloudflare — 0 · успешно 0 · ошибки 0", menu_icon_path("cloudflare"))
cloudflare_attempt_summary.set_sensitive(False)

def refresh_cloudflare_usage(*_args):
    usage = cloudflare_daily_usage(METRICS_FILE)
    for item, line in zip(cloudflare_usage_items,
                          format_cloudflare_usage_lines(usage, CLOUDFLARE_LOCAL_DAILY_REFERENCE)):
        item.set_label(line)

    cloudflare_attempt_summary.set_label(cloudflare_daily_attempt_summary(METRICS_FILE))

def cloudflare_daily_attempt_summary(metrics_path):
    try:
        from attempt_statistics import aggregate_daily_attempts
        rows = [row for row in aggregate_daily_attempts(metrics_path)
                if row["provider"] == "cloudflare"]
        attempts = sum(row["attempts"] for row in rows)
        successes = sum(row["successes"] for row in rows)
        failures = sum(row["failures"] for row in rows)
        return f"Сегодня: запросы Cloudflare — {attempts} · успешно {successes} · ошибки {failures}"
    except (OSError, ValueError, TypeError, KeyError):
        return "Сегодня: статистика попыток Cloudflare недоступна"

def format_cloudflare_usage_lines(usage, daily_reference):
    fmt = lambda number: f"{number:,}".replace(",", " ")
    correction = usage.get("corrector_estimated_neurons", 0)
    total = int(usage.get("estimated_total_neurons",
                          usage["estimated_neurons"] + correction) + 0.5)
    correction_basis = "оценка токенов" if usage.get("corrector_estimated_token_requests", 0) else "usage API/токены"
    minutes = usage["audio_seconds"] / 60
    estimated_remaining = max(0, daily_reference - total)
    return [
        f"Нейронов за UTC-день: ~{fmt(total)}",
        f"Расчётный остаток: ~{fmt(estimated_remaining)} из {fmt(daily_reference)}",
        f"ASR: ~{fmt(int(usage['estimated_neurons']))} нейронов · ~{minutes:.1f} мин · {usage['requests']} запросов",
        (f"Корректор: ~{fmt(int(correction))} нейронов ({correction_basis}); локальная оценка и preflight-защита"
         if usage.get("usage_guard_available", True) else
         "Контроль Cloudflare недоступен; удалённый запрос блокируется до полуночи UTC"),
    ]

def format_cloudflare_usage_label(usage, daily_reference):
    """Compatibility formatter for tests and non-menu diagnostics."""
    return "\n".join(format_cloudflare_usage_lines(usage, daily_reference))

def open_attempt_statistics(_item=None):
    """Open the inline statistics page in the shared settings panel."""
    open_settings_panel("Статистика")

def create_attempt_statistics_widget():
    """Build the sortable, filterable statistics table for the settings tab."""
    from attempt_statistics import aggregate_daily_attempts, reset_attempt_statistics
    root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    heading = Gtk.Label(xalign=0)
    root.pack_start(heading, False, False, 0)
    explanation = Gtk.Label(
        label="Скорость — секунды обработки на минуту аудио. Щёлкните заголовок столбца для сортировки.",
        xalign=0)
    explanation.set_line_wrap(True)
    root.pack_start(explanation, False, False, 0)

    filters = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    stage_filter = Gtk.ComboBoxText()
    for item_id, label in (("all", "Все этапы"), ("asr", "Whisper / ASR"),
                           ("correction", "Коррекция")):
        stage_filter.append(item_id, label)
    stage_filter.set_active_id("all")
    filters.pack_start(Gtk.Label(label="Этап:", xalign=0), False, False, 0)
    filters.pack_start(stage_filter, False, False, 0)

    provider_filter = Gtk.ComboBoxText()
    provider_filter.append("all", "Все сервисы")
    filters.pack_start(Gtk.Label(label="Сервис:", xalign=0), False, False, 0)
    filters.pack_start(provider_filter, False, False, 0)

    model_filter = Gtk.ComboBoxText()
    model_filter.append("all", "Все модели")
    filters.pack_start(Gtk.Label(label="Модель:", xalign=0), False, False, 0)
    filters.pack_start(model_filter, True, True, 0)
    refresh_button = Gtk.Button(label="Обновить")
    copy_button = Gtk.Button(label="Скопировать таблицу")
    reset_button = Gtk.Button(label="Обнулить статистику")
    filters.pack_start(refresh_button, False, False, 0)
    filters.pack_start(copy_button, False, False, 0)
    filters.pack_start(reset_button, False, False, 0)
    root.pack_start(filters, False, False, 0)

    store = Gtk.ListStore(str, str, str, int, int, int, str, float)
    tree = Gtk.TreeView(model=store)
    tree.set_headers_clickable(True)
    tree.set_search_column(2)
    columns = (
        ("Этап", 0, 0), ("Сервис", 1, 1), ("Модель", 2, 2),
        ("Попытки", 3, 3), ("Успешно", 4, 4), ("Неуспешно", 5, 5),
        ("с/мин", 6, 7),
    )
    for title, value_column, sort_column in columns:
        renderer = Gtk.CellRendererText()
        column = Gtk.TreeViewColumn(title, renderer, text=value_column)
        column.set_resizable(True)
        column.set_sort_column_id(sort_column)
        tree.append_column(column)
    tree.get_column(0).set_min_width(100)
    tree.get_column(1).set_min_width(120)
    tree.get_column(2).set_min_width(190)
    tree.get_column(3).set_min_width(85)
    tree.get_column(4).set_min_width(85)
    tree.get_column(5).set_min_width(100)
    tree.get_column(6).set_min_width(80)
    scroll = Gtk.ScrolledWindow()
    scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
    scroll.add(tree)
    root.pack_start(scroll, True, True, 0)
    status = Gtk.Label(label="", xalign=0)
    root.pack_start(status, False, False, 0)

    stage_names = {"asr": "ASR", "correction": "Коррекция"}
    provider_names = {"codex": "Codex", "claude": "Claude",
        "cloudflare": "Cloudflare", "openrouter": "OpenRouter", "groq": "Groq",
        "google": "Google Gemini", "local": "Локально"}
    rows = []

    def refresh_models(*_args):
        previous = model_filter.get_active_id() or "all"
        models = sorted({str(row.get("model", "?")) for row in rows
                         if provider_filter.get_active_id() in ("all", row.get("provider"))})
        model_filter.remove_all()
        model_filter.append("all", "Все модели")
        for model in models:
            model_filter.append(model, model)
        model_filter.set_active_id(previous if previous == "all" or previous in models else "all")

    def refresh_rows(*_args):
        stage_choice = stage_filter.get_active_id() or "all"
        provider_choice = provider_filter.get_active_id() or "all"
        model_choice = model_filter.get_active_id() or "all"
        visible = [row for row in rows
                   if (stage_choice == "all" or row["stage"] == stage_choice)
                   and (provider_choice == "all" or row["provider"] == provider_choice)
                   and (model_choice == "all" or row["model"] == model_choice)]
        store.clear()
        for row in visible:
            provider = row.get("provider", "?")
            speed = row.get("seconds_per_audio_minute")
            speed_value = float(speed) if isinstance(speed, (int, float)) else float("inf")
            speed_label = f"{speed:.2f}" if isinstance(speed, (int, float)) else "—"
            store.append((stage_names.get(row.get("stage"), row.get("stage", "?")),
                provider_names.get(provider, provider), str(row.get("model", "?")),
                int(row.get("attempts", 0)), int(row.get("successes", 0)),
                int(row.get("failures", 0)), speed_label, speed_value))
        heading.set_text(f"Попытки за {datetime.now(timezone.utc).date().isoformat()} UTC · {len(visible)} строк")

    def reload_data(*_args):
        nonlocal rows
        rows = aggregate_daily_attempts(METRICS_FILE)
        service_choice = provider_filter.get_active_id() or "all"
        provider_filter.remove_all()
        provider_filter.append("all", "Все сервисы")
        providers = sorted({row.get("provider", "?") for row in rows})
        for provider in providers:
            provider_filter.append(provider, provider_names.get(provider, provider))
        provider_filter.set_active_id(service_choice if service_choice == "all" or service_choice in providers else "all")
        refresh_models()
        refresh_rows()

    def clear_data(_button):
        try:
            reset_attempt_statistics()
            reload_data()
            status.set_text("Статистика обнулена; новые попытки начнут учитываться сразу.")
        except (OSError, ValueError, TypeError):
            status.set_text("Не удалось обнулить статистику.")

    def copy_visible_rows(_button):
        lines = ["Этап\tСервис\tМодель\tПопытки\tУспешно\tНеуспешно\tс/мин"]
        for row in store:
            values = [str(row[index]) for index in range(7)]
            lines.append("\t".join(values))
        Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).set_text("\n".join(lines), -1)
        status.set_text("Отфильтрованная таблица скопирована.")

    stage_filter.connect("changed", refresh_rows)
    provider_filter.connect("changed", lambda *_args: (refresh_models(), refresh_rows()))
    model_filter.connect("changed", refresh_rows)
    refresh_button.connect("clicked", reload_data)
    reset_button.connect("clicked", clear_data)
    copy_button.connect("clicked", copy_visible_rows)
    reload_data()
    return root

cloudflare_attempt_items = [Gtk.MenuItem() for _ in range(3)]
for item in cloudflare_attempt_items:
    item.set_sensitive(False)

def latest_cloudflare_attempt(metrics_path):
    """Return the latest safe Cloudflare attempt fields, skipping local-only jobs."""
    latest = None
    allowed = ("timestamp", "asr_backend", "asr_device", "corrector_backend", "corrector_device",
               "cloudflare_asr_error_category", "cloudflare_corrector_error_category")
    try:
        with open(metrics_path, encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(record, dict) and record.get("cloudflare_attempted") is True:
                    latest = {key: record[key] for key in allowed if key in record}
    except OSError:
        pass
    return latest

def cloudflare_error_reason(category):
    return {
        "http_429": "лимит/частота запросов (429)",
        "http_401_403": "Cloudflare отклонил доступ (401/403); проверьте токен и права",
        "missing_credentials": "не найдены обязательные учётные данные",
        "credentials_read_error": "не удалось прочитать файл учётных данных",
        "network_error": "сетевая ошибка",
        "timeout": "истёк тайм-аут запроса",
        "http_error": "ошибка API Cloudflare",
        "invalid_response": "ответ Cloudflare некорректен",
        "unsafe_response_length": "Cloudflare ответил, но проверка длины/полноты отклонила результат",
        "unsafe_response_word_coverage_order": "Cloudflare ответил, но проверка сохранения слов отклонила результат",
        "unsafe_response_tail": "Cloudflare ответил, но проверка конца текста отклонила результат",
        "unsafe_response_invalid": "ответ Cloudflare не прошёл проверку безопасности",
        "unsafe_response": "ответ Cloudflare не прошёл проверку безопасности",
        "local_audio_error": "не удалось прочитать аудиофайл для Cloudflare",
        "other_error": "неопределённая ошибка; подробности скрыты",
    }.get(category, "ошибка Cloudflare (категория не распознана)")

def format_cloudflare_attempt_status(record):
    return "\n".join(format_cloudflare_attempt_lines(record))

def format_cloudflare_attempt_lines(record):
    if not record:
        return ["Cloudflare: последняя попытка ещё не выполнялась"]
    labels = []
    for stage, title, backend_key, category_key in (
            ("whisper", "Whisper", "asr_backend", "cloudflare_asr_error_category"),
            ("corrector", "Корректор", "corrector_backend", "cloudflare_corrector_error_category")):
        category = record.get(category_key)
        backend = record.get(backend_key)
        if category:
            state = cloudflare_error_reason(category)
            if backend in {"local", "local_fallback"}:
                device = record.get("asr_device") if stage == "whisper" else record.get("corrector_device")
                local_label = "локальный GPU" if device == "cuda" else "локальный CPU" if device == "cpu" else "локально"
                state += f" → fallback: {local_label}"
            labels.append(f"{title}: {state}")
        elif backend == "cloudflare":
            labels.append(f"{title}: Cloudflare — успешно")
    if not labels:
        labels.append("Cloudflare: запрос выполнен без зафиксированной ошибки")
    timestamp_label = None
    timestamp = record.get("timestamp")
    if timestamp:
        try:
            moment = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).astimezone()
            timestamp_label = "Время: " + moment.strftime("%d.%m %H:%M")
        except (ValueError, TypeError, OverflowError):
            pass
    if timestamp_label:
        labels.append(timestamp_label)
    return labels[:3]

def refresh_cloudflare_last_attempt(*_args):
    for item, line in zip(cloudflare_attempt_items,
                          format_cloudflare_attempt_lines(latest_cloudflare_attempt(METRICS_FILE))):
        item.set_label(line)

refresh_cloudflare_usage()
cloudflare_submenu.connect("show", refresh_cloudflare_usage)
for item in cloudflare_usage_items:
    cloudflare_submenu.append(item)
cloudflare_submenu.append(cloudflare_attempt_summary)
cloudflare_submenu.append(Gtk.SeparatorMenuItem())
refresh_cloudflare_last_attempt()
cloudflare_submenu.connect("show", refresh_cloudflare_last_attempt)
for item in cloudflare_attempt_items:
    cloudflare_submenu.append(item)
cloudflare_submenu.append(Gtk.SeparatorMenuItem())

def set_cloudflare_asr_value(item, key, selected, is_toggle=False):
    if is_toggle:
        active, value = True, item.get_active()
    else:
        active, value = item.get_active(), selected
    if not active:
        return
    config = load_voice_config()
    profile = config.get("cloudflare_asr", {})
    if not isinstance(profile, dict):
        profile = {}
    if key == "model":
        config["cloudflare_asr_model"] = value
    profile[key] = value
    config["cloudflare_asr"] = profile
    save_voice_config(config)

def cloudflare_asr_profile():
    profile = load_voice_config().get("cloudflare_asr", {})
    return profile if isinstance(profile, dict) else {}

def edit_cloudflare_asr_text(_item, key, title, limit):
    config = load_voice_config()
    profile = config.get("cloudflare_asr", {})
    if not isinstance(profile, dict):
        profile = {}
    dialog = Gtk.Dialog(title=title, transient_for=None, flags=0,
                        buttons=("Отмена", Gtk.ResponseType.CANCEL, "Сохранить", Gtk.ResponseType.OK))
    if key == "initial_prompt":
        dialog.set_default_size(560, 360)
        dialog.set_default_response(Gtk.ResponseType.OK)
        area = dialog.get_content_area()
        area.set_border_width(12)
        area.set_spacing(8)
        area.pack_start(Gtk.Label(label="Введите начальный prompt; можно вставить многострочный текст.", xalign=0), False, False, 0)
        text_view = Gtk.TextView()
        text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        text_view.get_buffer().set_text(profile.get(key, ""))
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.add(text_view)
        area.pack_start(scroll, True, True, 0)
    else:
        dialog.set_default_size(460, 120)
        entry = Gtk.Entry()
        entry.set_max_length(limit)
        entry.set_text(profile.get(key, ""))
        entry.set_activates_default(True)
        dialog.set_default_response(Gtk.ResponseType.OK)
        dialog.get_content_area().pack_start(entry, True, True, 12)
    dialog.show_all()
    if dialog.run() == Gtk.ResponseType.OK:
        if key == "initial_prompt":
            buffer = text_view.get_buffer()
            updated_text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        else:
            updated_text = entry.get_text().strip()
        if len(updated_text) > limit:
            notify_send("Whisper Dictate", f"Начальный prompt не сохранён: максимум {limit} символов")
            dialog.destroy()
            return
        profile[key] = updated_text
        config["cloudflare_asr"] = profile
        save_voice_config(config)
    dialog.destroy()

cloudflare_whisper_settings = Gtk.Menu()
cloudflare_turbo_only_items = []
cloudflare_whisper_settings.append(Gtk.MenuItem(label="Модель Whisper"))
cloudflare_whisper_settings.get_children()[-1].set_sensitive(False)
cloudflare_asr_model_group = []
cloudflare_asr_models = [("@cf/openai/whisper-large-v3-turbo", "Whisper Large-v3 Turbo (по умолчанию)"),
                         ("@cf/openai/whisper", "Whisper Large-v3")]
for model_name, model_label in cloudflare_asr_models:
    item = Gtk.RadioMenuItem(label=model_label) if not cloudflare_asr_model_group else Gtk.RadioMenuItem.new_from_widget(cloudflare_asr_model_group[0])
    item.set_label(model_label)
    current_asr_model = cloudflare_asr_profile().get("model", load_voice_config().get("cloudflare_asr_model", "@cf/openai/whisper-large-v3-turbo"))
    item.set_active(current_asr_model == model_name)
    item.connect("toggled", set_cloudflare_asr_value, "model", model_name)
    cloudflare_asr_model_group.append(item)
    cloudflare_whisper_settings.append(item)
cloudflare_whisper_settings.append(Gtk.SeparatorMenuItem())
cloudflare_whisper_settings.append(Gtk.MenuItem(label="Параметры распознавания"))
cloudflare_whisper_settings.get_children()[-1].set_sensitive(False)
cloudflare_whisper_settings.append(Gtk.MenuItem(label="Beam size и пороги доступны для Large-v3 Turbo"))
cloudflare_whisper_settings.get_children()[-1].set_sensitive(False)

def add_cloudflare_asr_radio(title, key, options, default):
    cloudflare_whisper_settings.append(Gtk.SeparatorMenuItem())
    header = Gtk.MenuItem(label=title)
    header.set_sensitive(False)
    cloudflare_whisper_settings.append(header)
    group = []
    current = cloudflare_asr_profile().get(key, default)
    for selected, label in options:
        item = Gtk.RadioMenuItem(label=label) if not group else Gtk.RadioMenuItem.new_from_widget(group[0])
        item.set_label(label)
        item.set_active(current == selected)
        item.connect("toggled", set_cloudflare_asr_value, key, selected)
        group.append(item)
        cloudflare_whisper_settings.append(item)
        if key not in {"task", "language"}:
            cloudflare_turbo_only_items.append(item)

add_cloudflare_asr_radio("Задача", "task", [("transcribe", "Распознавание"), ("translate", "Перевод на английский")], "transcribe")
add_cloudflare_asr_radio("Язык", "language", [("", "Автоопределение"), ("ru", "Русский"), ("en", "Английский")], "ru")
cloudflare_whisper_settings.append(Gtk.SeparatorMenuItem())
cloudflare_asr_vad_item = Gtk.CheckMenuItem(label="VAD-фильтр речи")
cloudflare_asr_vad_item.set_active(cloudflare_asr_profile().get("vad_filter", False))
cloudflare_asr_vad_item.connect("toggled", set_cloudflare_asr_value, "vad_filter", None, True)
cloudflare_whisper_settings.append(cloudflare_asr_vad_item)
add_cloudflare_asr_radio("Beam size", "beam_size", [(1, "1 — быстрее"), (3, "3"), (5, "5 — точнее"), (10, "10 — медленнее")], 1)
cloudflare_whisper_settings.append(Gtk.SeparatorMenuItem())
cloudflare_asr_condition_item = Gtk.CheckMenuItem(label="Учитывать предыдущий текст")
cloudflare_asr_condition_item.set_active(cloudflare_asr_profile().get("condition_on_previous_text", True))
cloudflare_asr_condition_item.connect("toggled", set_cloudflare_asr_value, "condition_on_previous_text", None, True)
cloudflare_whisper_settings.append(cloudflare_asr_condition_item)
cloudflare_turbo_only_items.append(cloudflare_asr_condition_item)
add_cloudflare_asr_radio("Порог тишины", "no_speech_threshold", [(0.3, "0.3"), (0.6, "0.6 — по умолчанию"), (0.8, "0.8")], 0.6)
add_cloudflare_asr_radio("Порог повторов", "compression_ratio_threshold", [(1.8, "1.8"), (2.4, "2.4 — по умолчанию"), (3.0, "3.0")], 2.4)
add_cloudflare_asr_radio("Порог вероятности", "log_prob_threshold", [(-1.5, "-1.5"), (-1.0, "-1.0 — по умолчанию"), (-0.5, "-0.5")], -1.0)
add_cloudflare_asr_radio("Пропуск длинной тишины", "hallucination_silence_threshold", [(None, "Выключено"), (2, "2 секунды"), (4, "4 секунды"), (8, "8 секунд")], None)
cloudflare_whisper_settings.append(Gtk.SeparatorMenuItem())
cloudflare_initial_prompt_item = icon_menu_item("Изменить начальный prompt…", menu_icon_path("microphone"))
cloudflare_initial_prompt_item.connect("activate", edit_cloudflare_asr_text, "initial_prompt", "Начальный prompt Cloudflare Whisper", 800)
cloudflare_whisper_settings.append(cloudflare_initial_prompt_item)
cloudflare_turbo_only_items.append(cloudflare_initial_prompt_item)
cloudflare_whisper_settings.append(Gtk.SeparatorMenuItem())
cloudflare_prefix_item = icon_menu_item("Изменить prefix…", menu_icon_path("microphone"))
cloudflare_prefix_item.connect("activate", edit_cloudflare_asr_text, "prefix", "Prefix Cloudflare Whisper", 200)
cloudflare_whisper_settings.append(cloudflare_prefix_item)
cloudflare_turbo_only_items.append(cloudflare_prefix_item)
def refresh_cloudflare_asr_option_sensitivity(*_args):
    enabled = cloudflare_asr_profile().get("model", load_voice_config().get("cloudflare_asr_model", "@cf/openai/whisper-large-v3-turbo")) == "@cf/openai/whisper-large-v3-turbo"
    for option in cloudflare_turbo_only_items:
        option.set_sensitive(enabled)
cloudflare_whisper_settings.connect("show", refresh_cloudflare_asr_option_sensitivity)
cloudflare_whisper_settings.show_all()
cloudflare_whisper_settings_item = icon_menu_item("Whisper: модель и расширенные настройки", menu_icon_path("microphone"))
cloudflare_whisper_settings_item.set_submenu(cloudflare_whisper_settings)

cloudflare_qwen_settings = Gtk.Menu()
cloudflare_qwen_settings.append(Gtk.MenuItem(label="Порядок моделей Cloudflare-корректора"))
cloudflare_qwen_settings.get_children()[-1].set_sensitive(False)
cloudflare_corrector_model_group = []
cloudflare_models = [("auto", "Автоматически по статистике (рекомендуется)"),
                    ("@cf/qwen/qwen3-30b-a3b-fp8", "Qwen3 30B A3B FP8"),
                    ("@cf/zai-org/glm-4.7-flash", "GLM-4.7-Flash"),
                    ("@cf/openai/gpt-oss-20b", "OpenAI GPT-OSS-20B")]
for model_name, model_label in cloudflare_models:
    item = Gtk.RadioMenuItem(label=model_label) if not cloudflare_corrector_model_group else Gtk.RadioMenuItem.new_from_widget(cloudflare_corrector_model_group[0])
    item.set_label(model_label)
    item.set_active(load_voice_config().get("cloudflare_corrector", {}).get("priority_model", "auto") == model_name)
    def select_cloudflare_model(menu_item, selected_model):
        if menu_item.get_active():
            value = load_voice_config()
            profile = value.get("cloudflare_corrector", {})
            profile["priority_model"] = selected_model
            if selected_model != "auto":
                profile["model"] = selected_model
            value["cloudflare_corrector"] = profile
            save_voice_config(value)
    item.connect("toggled", select_cloudflare_model, model_name)
    cloudflare_corrector_model_group.append(item)
    cloudflare_qwen_settings.append(item)
cloudflare_qwen_settings.append(Gtk.SeparatorMenuItem())
cloudflare_qwen_settings.append(Gtk.MenuItem(label="Настройки корректора"))
cloudflare_qwen_settings.get_children()[-1].set_sensitive(False)
def set_cloudflare_parameter(item, key, val):
    if item.get_active():
        value = load_voice_config()
        profile = value.get("cloudflare_corrector", {})
        profile[key] = val
        value["cloudflare_corrector"] = profile
        save_voice_config(value)
cloudflare_retry_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
cloudflare_retry_row.pack_start(Gtk.Label(label="Дополнительные попытки с разными моделями:"), True, True, 0)
cloudflare_retry_spin = integer_retry_control(
    0, 2, load_voice_config().get("cloudflare_corrector", {}).get("retry_attempts", 2))
def set_cloudflare_retry_attempts(spin):
    value = load_voice_config()
    profile = value.get("cloudflare_corrector", {})
    profile["retry_attempts"] = int(spin.get_value())
    value["cloudflare_corrector"] = profile
    save_voice_config(value)
cloudflare_retry_spin.connect("value-changed", set_cloudflare_retry_attempts)
cloudflare_retry_row.pack_start(cloudflare_retry_spin, False, False, 0)
cloudflare_retry_menu_item = Gtk.MenuItem()
cloudflare_retry_menu_item.add(cloudflare_retry_row)
cloudflare_qwen_settings.append(cloudflare_retry_menu_item)
cloudflare_retry_help = Gtk.Label(label="0–2 повтора; каждая попытка использует другую модель. Ответ всегда проходит проверку полноты; 401/403, 429 и исчерпание дневной квоты повторно не запрашиваются.", xalign=0)
cloudflare_retry_help.set_line_wrap(True)
cloudflare_retry_help_item = Gtk.MenuItem()
cloudflare_retry_help_item.add(cloudflare_retry_help)
cloudflare_qwen_settings.append(cloudflare_retry_help_item)
for key, title, options in [("temperature", "Температура", [(0.0, "0 (по умолчанию)"), (0.2, "0.2"), (0.6, "0.6")]),
                           ("max_tokens", "Максимум токенов", [(512, "512"), (1024, "1024"), (2048, "2048 (по умолчанию)"), (4096, "4096")])]:
    cloudflare_qwen_settings.append(Gtk.SeparatorMenuItem())
    header = Gtk.MenuItem(label=title)
    header.set_sensitive(False)
    cloudflare_qwen_settings.append(header)
    group = []
    for val, label in options:
        item = Gtk.RadioMenuItem(label=label) if not group else Gtk.RadioMenuItem.new_from_widget(group[0])
        item.set_label(label)
        item.set_active(load_voice_config().get("cloudflare_corrector", {}).get(key, 0.0 if key == "temperature" else 2048) == val)
        item.connect("toggled", set_cloudflare_parameter, key, val)
        group.append(item)
        cloudflare_qwen_settings.append(item)
cloudflare_qwen_settings.show_all()
cloudflare_qwen_settings_item = icon_menu_item("Корректор: модели и порядок попыток", menu_icon_path("corrector"))
cloudflare_qwen_settings_item.set_submenu(cloudflare_qwen_settings)

def set_cloudflare_dialog_value(config_key, value, profile_key=None):
    config = load_voice_config()
    target = config.setdefault(profile_key, {}) if profile_key else config
    if not isinstance(target, dict):
        target = {}
        config[profile_key] = target
    target[config_key] = value
    save_voice_config(config)

def configure_cloudflare_settings(_item=None, parent=None):
    if parent is None:
        return open_settings_panel("Cloudflare")
    cfg = load_voice_config()
    dialog = None
    if parent is None:
        dialog = Gtk.Dialog(title="Настройки Cloudflare", transient_for=None, flags=0)
        dialog.add_button("Готово", Gtk.ResponseType.CLOSE)
        dialog.set_default_size(560, 540)
        dialog.set_border_width(8)
        action_area = dialog.get_action_area()
        action_area.set_margin_top(4)
        action_area.set_margin_bottom(10)
        action_area.set_margin_start(8)
        action_area.set_margin_end(8)
        area = dialog.get_content_area()
        area.set_border_width(12)
        area.set_margin_bottom(8)
    else:
        area = parent
    notebook = Gtk.Notebook()
    area.pack_start(notebook, True, True, 0)

    whisper_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    whisper_page.set_border_width(10)
    asr_toggle = Gtk.CheckButton(label="Использовать Cloudflare для Whisper")
    asr_toggle.set_active(bool(cfg.get("cloudflare_asr_enabled", cfg.get("cloudflare_workers_ai_enabled", False))))
    asr_toggle.connect("toggled", on_cloudflare_toggled)
    cloudflare_dialog_toggles["asr"] = asr_toggle
    whisper_page.pack_start(asr_toggle, False, False, 0)
    whisper_page.pack_start(Gtk.Label(label="Распознавание речи через Cloudflare Workers AI", xalign=0), False, False, 0)
    def add_combo(parent, label, key, choices, current, profile_key="cloudflare_asr", root_key=None):
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.pack_start(Gtk.Label(label=label, xalign=0), False, False, 0)
        combo = Gtk.ComboBoxText()
        for value, caption in choices:
            combo.append(str(value), caption)
        active = str(current)
        combo.set_active_id(active if active in {str(v) for v, _ in choices} else str(choices[0][0]))
        def changed(widget):
            selected = widget.get_active_id()
            value = next((v for v, _ in choices if str(v) == selected), choices[0][0])
            set_cloudflare_dialog_value(root_key or key, value, profile_key)
        combo.connect("changed", changed)
        row.pack_start(combo, True, True, 0)
        parent.pack_start(row, False, False, 0)
        return combo
    asr = cloudflare_asr_profile()
    add_combo(whisper_page, "Модель:", "model", cloudflare_asr_models,
              asr.get("model", cfg.get("cloudflare_asr_model", "@cf/openai/whisper-large-v3-turbo")))
    add_combo(whisper_page, "Задача:", "task", [("transcribe", "Распознавание"), ("translate", "Перевод")], asr.get("task", "transcribe"))
    add_combo(whisper_page, "Язык:", "language", [("", "Авто"), ("ru", "Русский"), ("en", "Английский")], asr.get("language", "ru"))
    add_combo(whisper_page, "Beam size:", "beam_size", [(1,"1 — быстрее"),(3,"3"),(5,"5 — точнее"),(10,"10 — медленнее")], asr.get("beam_size", 1))
    for key, label, default in (("vad_filter", "VAD-фильтр речи", False), ("condition_on_previous_text", "Учитывать предыдущий текст", True)):
        check = Gtk.CheckButton(label=label)
        check.set_active(asr.get(key, default))
        check.connect("toggled", lambda widget, k=key: set_cloudflare_dialog_value(k, widget.get_active(), "cloudflare_asr"))
        whisper_page.pack_start(check, False, False, 0)
    for key, label, options, default in (
        ("no_speech_threshold", "Порог тишины", [(0.3,"0.3"),(0.6,"0.6"),(0.8,"0.8")], 0.6),
        ("compression_ratio_threshold", "Порог повторов", [(1.8,"1.8"),(2.4,"2.4"),(3.0,"3.0")], 2.4),
        ("log_prob_threshold", "Порог вероятности", [(-1.5,"-1.5"),(-1.0,"-1.0"),(-0.5,"-0.5")], -1.0),
        ("hallucination_silence_threshold", "Пропуск длинной тишины", [(None,"Выключено"),(2,"2 с"),(4,"4 с"),(8,"8 с")], None)):
        add_combo(whisper_page, label, key, options, asr.get(key, default))
    for key, label, limit in (("initial_prompt", "Начальный prompt", 800), ("prefix", "Prefix", 200)):
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.pack_start(Gtk.Label(label=label, xalign=0), False, False, 0)
        if key == "initial_prompt":
            row.pack_start(Gtk.Label(label="Многострочный текст для подсказки распознаванию", xalign=0), True, True, 0)
            edit_button = Gtk.Button(label="Изменить…")
            edit_button.connect("clicked", edit_cloudflare_asr_text, key,
                                "Начальный prompt Cloudflare Whisper", limit)
            row.pack_start(edit_button, False, False, 0)
        else:
            entry = Gtk.Entry(); entry.set_max_length(limit); entry.set_text(asr.get(key, ""))
            entry.connect("changed", lambda widget, k=key: set_cloudflare_dialog_value(k, widget.get_text(), "cloudflare_asr"))
            row.pack_start(entry, True, True, 0)
        whisper_page.pack_start(row, False, False, 0)
    notebook.append_page(
        whisper_page, settings_tab_label("Whisper", "transcribe", icon_size=22))

    corrector_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    corrector_page.set_border_width(10)
    corrector_toggle = Gtk.CheckButton(label="Использовать Cloudflare для корректора")
    corrector_toggle.set_active(bool(cfg.get("cloudflare_corrector", {}).get("enabled", False)))
    corrector_toggle.connect("toggled", on_cloudflare_corrector_toggled)
    cloudflare_dialog_toggles["corrector"] = corrector_toggle
    corrector_page.pack_start(corrector_toggle, False, False, 0)
    corrector_page.pack_start(Gtk.Label(label="Модель и порядок попыток", xalign=0), False, False, 0)
    profile = cfg.get("cloudflare_corrector", {})
    if not isinstance(profile, dict):
        profile = {}
    add_combo(corrector_page, "Приоритет модели:", "priority_model", cloudflare_models,
              profile.get("priority_model", "auto"), "cloudflare_corrector")
    add_combo(corrector_page, "Температура:", "temperature", [(0.0,"0"),(0.2,"0.2"),(0.6,"0.6")], profile.get("temperature", 0.0), "cloudflare_corrector")
    add_combo(corrector_page, "Максимум токенов:", "max_tokens", [(512,"512"),(1024,"1024"),(2048,"2048"),(4096,"4096")], profile.get("max_tokens", 2048), "cloudflare_corrector")
    corrector_page.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL),
                              False, False, 4)
    retry_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    retry_row.pack_start(Gtk.Label(label="Дополнительные попытки (0–2):", xalign=0), True, True, 0)
    retry_value = profile.get("retry_attempts", 2)
    retry_spin = integer_retry_control(
        0, 2, retry_value if type(retry_value) is int else 2)
    def save_cf_retries(spin): set_cloudflare_dialog_value("retry_attempts", int(spin.get_value()), "cloudflare_corrector")
    retry_spin.connect("value-changed", save_cf_retries)
    retry_row.pack_start(retry_spin, False, False, 0); corrector_page.pack_start(retry_row, False, False, 0)
    retry_help = Gtk.Label(label="0–2 повтора сверх первой попытки; всего не более трёх. Каждый повтор использует другую доступную модель. 401/403, 429 и дневная квота повторно не запрашиваются.", xalign=0)
    retry_help.set_line_wrap(True); corrector_page.pack_start(retry_help, False, False, 0)
    notebook.append_page(
        corrector_page, settings_tab_label("Корректор", "corrector", icon_size=22))

    account_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
    account_page.set_border_width(12)

    def account_section(title):
        frame = Gtk.Frame(label=title)
        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        body.set_border_width(10)
        frame.add(body)
        account_page.pack_start(frame, False, False, 0)
        return body

    credentials_body = account_section("Учётные данные")
    account_status = Gtk.CheckButton(label="Учётные данные заполнены")
    account_status.set_active(cloudflare_credentials_available())
    account_status.set_sensitive(False)
    credentials_body.pack_start(account_status, False, False, 0)
    credentials_info = Gtk.Label(
        label="Account ID и API token хранятся локально в защищённом файле credentials.",
        xalign=0)
    credentials_info.set_line_wrap(True)
    credentials_body.pack_start(credentials_info, False, False, 0)
    credentials_button = Gtk.Button(label="Открыть файл Account ID и API token…")
    credentials_button.connect("clicked", lambda *_args: open_credentials_file(None, "cloudflare"))
    credentials_body.pack_start(credentials_button, False, False, 0)

    usage_body = account_section("Использование")
    account_usage = Gtk.Label(label=provider_account_usage_text("cloudflare"), xalign=0)
    account_usage.set_line_wrap(True)
    usage_body.pack_start(account_usage, False, False, 0)

    attempt_body = account_section("Последняя попытка Cloudflare")
    latest_attempt = Gtk.Label(
        label=format_cloudflare_attempt_status(latest_cloudflare_attempt(METRICS_FILE)),
        xalign=0)
    latest_attempt.set_line_wrap(True)
    latest_attempt.set_selectable(True)
    attempt_body.pack_start(latest_attempt, False, False, 0)
    notebook.append_page(
        account_page, settings_tab_label("Учётные данные", "cloudflare", icon_size=22))

    def refresh_account_page(*_args):
        account_usage.set_text(provider_account_usage_text("cloudflare"))
        account_status.set_active(cloudflare_credentials_available())
        latest_attempt.set_text(format_cloudflare_attempt_status(
            latest_cloudflare_attempt(METRICS_FILE)))

    notebook.connect("switch-page", lambda _notebook, _page, page_number:
                      refresh_account_page() if page_number == 2 else None)
    if dialog is not None:
        dialog.show_all(); dialog.run(); dialog.destroy()
        cloudflare_dialog_toggles.clear()
    else:
        area.show_all()
        area.connect("destroy", lambda *_args: cloudflare_dialog_toggles.clear())

cloudflare_submenu.connect("show", refresh_cloudflare_credentials_status)
cloudflare_settings_item = icon_menu_item("Настройки Cloudflare…", menu_icon_path("cloudflare"))
cloudflare_settings_item.connect("activate", configure_cloudflare_settings)
cloudflare_submenu.append(cloudflare_settings_item)

cloudflare_submenu.show_all()
cloudflare_menu_item = provider_menu_item("Cloudflare", "processing_cloudflare")
cloudflare_menu_item.set_submenu(cloudflare_submenu)

def update_multi_provider_setting(item, stage, key):
    from multi_provider import safe_settings
    value = load_voice_config()
    settings = safe_settings(value.get("multi_provider"))
    settings[stage][key] = item.get_active()
    value["multi_provider"] = settings
    save_voice_config(value)
    refresh_multi_provider_status()

def update_provider_asr_quality(item, *_args):
    quality = item.get_active_id()
    if quality not in {"accuracy", "fast"}:
        return
    from multi_provider import safe_settings
    value = load_voice_config()
    settings = safe_settings(value.get("multi_provider"))
    settings["asr"]["quality"] = quality
    value["multi_provider"] = settings
    save_voice_config(value)
    refresh_multi_provider_status()

def _save_model_order(stage, order):
    from multi_provider import safe_settings
    value = load_voice_config()
    settings = safe_settings(value.get("multi_provider"))
    settings[stage]["order"] = list(order)
    value["multi_provider"] = settings
    save_voice_config(value)
    refresh_multi_provider_status()

def _save_local_model_order(stage, order, asr_scenario="voice"):
    from multi_provider import (safe_settings, LOCAL_ASR_MODELS,
                                LOCAL_CORRECTION_MODELS, resolve_local_asr_model)
    value = load_voice_config()
    settings = safe_settings(value.get("multi_provider"))
    allowed = LOCAL_ASR_MODELS if stage == "asr" else LOCAL_CORRECTION_MODELS
    selected_order = list(dict.fromkeys(
        model for model in order if model in allowed))
    if stage == "asr":
        settings[stage]["file_video_local_models" if asr_scenario == "file_video"
                        else "voice_local_models"] = selected_order
        if asr_scenario != "file_video":
            settings[stage]["local_models"] = selected_order
            preferred = selected_order[0] if selected_order else "faster-whisper"
            backend, model = resolve_local_asr_model(
                preferred, value.get("model", "large-v3"))
            value["asr_backend"] = backend
            if backend == "faster-whisper":
                value["model"] = model
    else:
        settings[stage]["local_models"] = selected_order
    value["multi_provider"] = settings
    if stage == "correction" and selected_order:
        corrector = value.get("corrector", {})
        if isinstance(corrector, dict):
            corrector["model"] = selected_order[0]
            value["corrector"] = corrector
    save_voice_config(value)
    if stage == "asr":
        if asr_scenario != "file_video":
            _sync_local_asr_backend_settings(value.get("asr_backend", "faster-whisper"))
            _sync_whisper_model_settings(value.get("model", "large-v3"))
        update_model_label()
    else:
        update_corrector_label()
        _sync_corrector_model_radio(selected_order[0] if selected_order else None)
    refresh_multi_provider_status()

def _promote_local_model_preference(value, stage, model_id):
    """Keep legacy single-model menus aligned with an explicit fallback chain."""
    from multi_provider import safe_settings, LOCAL_ASR_MODELS, LOCAL_CORRECTION_MODELS
    settings = safe_settings(value.get("multi_provider"))
    allowed = LOCAL_ASR_MODELS if stage == "asr" else LOCAL_CORRECTION_MODELS
    order_key = "voice_local_models" if stage == "asr" else "local_models"
    if model_id not in allowed or not settings[stage][order_key]:
        return
    previous = settings[stage][order_key]
    settings[stage][order_key] = [model_id] + [item for item in previous
                                               if item != model_id and not (
                                                   stage == "asr" and model_id.startswith("faster-whisper:")
                                                   and item == "faster-whisper")]
    if stage == "asr":
        settings[stage]["local_models"] = settings[stage][order_key]
    value["multi_provider"] = settings

def _local_model_chain_widget(stage, settings, root_config, asr_scenario="voice"):
    from multi_provider import LOCAL_ASR_MODELS, LOCAL_CORRECTION_MODELS
    if stage == "asr":
        catalog = {
            "gigaam-v3-e2e-rnnt": "GigaAM-v3 e2e_rnnt · CPU",
            "gigaam-v3-e2e-ctc-onnx-gpu": "GigaAM-v3 e2e_ctc · GPU",
            "gigaam-v3-ctc-onnx-gpu": "GigaAM-v3 ctc · GPU",
        }
        catalog.update({f"faster-whisper:{model_id}": f"FasterWhisper · {label}"
                        for model_id, label in FASTER_WHISPER_MODEL_CHOICES})
        selected_backend = root_config.get("asr_backend", "faster-whisper")
        preferred = (selected_backend if selected_backend in GIGAAM_MODEL_CHOICES else
                     f"faster-whisper:{root_config.get('model', 'large-v3')}")
        # The generic FasterWhisper alias represents the currently selected model;
        # it is retained for config compatibility but never displayed as an option.
        allowed = tuple(model for model in LOCAL_ASR_MODELS
                        if model != "faster-whisper")
        order_key = ("file_video_local_models" if asr_scenario == "file_video"
                     else "voice_local_models")
        note = ("Файлы и видео распознаются только локально. Первая отмеченная модель "
                "запускается первой; следующие — после технической ошибки или пустого ответа."
                if asr_scenario == "file_video" else
                "Порядок локальных моделей для голосовой транскрибации. "
                "Следующая модель запускается после технической ошибки или пустого ответа.")
    else:
        catalog = {
            "qwen3.5:4b-q4_K_M": "Qwen3.5 4B",
            "qwen3.5:2b-q4_K_M": "Qwen3.5 2B",
        }
        preferred = root_config.get("corrector", {}).get("model", LOCAL_CORRECTION_MODELS[0])
        allowed = LOCAL_CORRECTION_MODELS
        order_key = "local_models"
        note = "Fallback использует только выбранный Ollama model id; отсутствующая модель не скачивается автоматически."
    saved = settings[stage].get(order_key) or [preferred]
    if stage == "asr":
        raw_multi_settings = root_config.get("multi_provider", {})
        raw_asr_settings = (raw_multi_settings.get("asr", {})
                            if isinstance(raw_multi_settings, dict) else {})
        if asr_scenario == "voice":
            has_explicit_order = any(
                isinstance(raw_asr_settings.get(key), list) and raw_asr_settings.get(key)
                for key in ("voice_local_models", "local_models"))
            if not has_explicit_order and preferred in allowed:
                saved = [preferred]
        saved = [preferred if item == "faster-whisper" and preferred in allowed else item
                 for item in saved]
    frame_title = ("Порядок локальных моделей для файлов и видео"
                   if stage == "asr" and asr_scenario == "file_video" else
                   "Порядок локальных моделей для голосовой транскрибации"
                   if stage == "asr" else "Порядок локальных моделей")
    order = [item for item in saved if item in allowed]
    order.extend(item for item in allowed if item not in order)
    frame = Gtk.Frame()
    frame_label = Gtk.Label()
    frame_label.set_markup(f"<b>{GLib.markup_escape_text(frame_title)}</b>")
    frame.set_label_widget(frame_label)
    content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
    content.set_border_width(6)
    frame.add(content)
    help_label = Gtk.Label(label=note, xalign=0)
    help_label.set_line_wrap(True)
    content.pack_start(help_label, False, False, 2)
    listbox = Gtk.ListBox()
    listbox.set_name("local-model-order")
    listbox.set_can_focus(False)
    listbox.set_selection_mode(Gtk.SelectionMode.NONE)
    checks, row_widgets = {}, {}
    for model_id in order:
        row = Gtk.ListBoxRow()
        row.set_can_focus(False)
        row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        row_box.set_border_width(3)
        row.add(row_box)
        check = Gtk.CheckButton(label=catalog[model_id])
        _bold_checkbutton_label(check, catalog[model_id])
        check.set_active(model_id in saved)
        checks[model_id] = check
        row_box.pack_start(check, True, True, 0)
        up = Gtk.Button(label="↑")
        down = Gtk.Button(label="↓")
        up.set_tooltip_text("Поднять модель в списке fallback")
        down.set_tooltip_text("Опустить модель в списке fallback")
        row_box.pack_start(up, False, False, 0)
        row_box.pack_start(down, False, False, 0)
        row_widgets[model_id] = row
        listbox.add(row)
    display_order = list(order)

    def persist_order(*_args):
        selected = [model for model in display_order if checks[model].get_active()]
        if not selected:
            checks[display_order[0]].set_active(True)
            return
        _save_local_model_order(stage, selected, asr_scenario)

    def move(model_id, delta):
        index = display_order.index(model_id)
        target = index + delta
        if not 0 <= target < len(display_order):
            return
        display_order[index], display_order[target] = display_order[target], display_order[index]
        row = row_widgets[model_id]
        listbox.remove(row)
        listbox.insert(row, target)
        listbox.show_all()
        persist_order()

    for model_id in display_order:
        checks[model_id].connect("toggled", persist_order)
        row = row_widgets[model_id]
        children = row.get_child().get_children()
        children[1].connect("clicked", lambda _button, model=model_id: move(model, -1))
        children[2].connect("clicked", lambda _button, model=model_id: move(model, 1))
    content.pack_start(listbox, False, False, 0)
    listbox.show_all()
    return frame

def _asr_service_order_widget(settings, labels):
    """Show the shared voice-ASR service order on the file/video settings page."""
    providers = {"cloudflare", "groq", "google", "local"}
    display_labels = {**labels, "local": "Локальная модель (ASR)"}
    order = [provider for provider in settings["asr"]["order"] if provider in providers]
    order.extend(provider for provider in ("cloudflare", "groq", "google", "local")
                 if provider not in order)
    frame = Gtk.Frame()
    frame_label = Gtk.Label()
    frame_label.set_markup("<b>Порядок сервисов для голосовой транскрибации</b>")
    frame.set_label_widget(frame_label)
    content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
    content.set_border_width(8)
    note = Gtk.Label(
        label=("Этот порядок общий с вкладкой «Голосовые» и применяется только к "
               "голосовой транскрибации. Файлы и видео используют локальный порядок ниже."),
        xalign=0)
    note.set_line_wrap(True)
    content.pack_start(note, False, False, 2)
    rows = Gtk.ListBox()
    rows.set_selection_mode(Gtk.SelectionMode.NONE)
    row_widgets = {}
    for index, provider in enumerate(order):
        row = Gtk.ListBoxRow()
        row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row_box.set_border_width(2)
        row.add(row_box)
        service_label = Gtk.Label(xalign=0)
        service_label.set_markup(
            "<b>" + GLib.markup_escape_text(display_labels[provider]) + "</b>")
        row_box.pack_start(service_label, True, True, 0)
        up = Gtk.Button(label="↑")
        down = Gtk.Button(label="↓")
        up.set_sensitive(index > 0)
        down.set_sensitive(index < len(order) - 1)
        up.set_tooltip_text("Поднять сервис в голосовом ASR порядке")
        down.set_tooltip_text("Опустить сервис в голосовом ASR порядке")
        row_box.pack_start(up, False, False, 0)
        row_box.pack_start(down, False, False, 0)
        row_widgets[provider] = row
        rows.add(row)

        def move_provider(_button, provider=provider, direction=-1):
            current = list(order)
            if provider not in current:
                return
            index = current.index(provider)
            target = index + direction
            if not 0 <= target < len(current):
                return
            current[index], current[target] = current[target], current[index]
            _save_model_order("asr", current)
            order[:] = [item for item in current if item in row_widgets]
            settings["asr"]["order"] = list(current)
            for position, item in enumerate(order):
                rows.reorder_child(row_widgets[item], position)
                child_box = row_widgets[item].get_child()
                buttons = child_box.get_children()[-2:]
                buttons[0].set_sensitive(position > 0)
                buttons[1].set_sensitive(position < len(order) - 1)

        up.connect("clicked", move_provider, -1)
        down.connect("clicked", move_provider, 1)
    content.pack_start(rows, False, False, 0)
    frame.add(content)
    return frame

def provider_account_usage_text(provider):
    """Build account and usage status only from local, privacy-safe state."""
    from attempt_statistics import aggregate_daily_attempts
    rows = aggregate_daily_attempts(METRICS_FILE)
    stages = {stage: sum(row["attempts"] for row in rows
                         if row.get("provider") == provider and row.get("stage") == stage)
              for stage in ("asr", "correction")}
    if provider in {"cloudflare", "openrouter"}:
        account = None
    else:
        account = "локальный статус: неизвестен"
    quota = "дневной лимит и остаток: нет локальных данных"
    if provider == "cloudflare":
        try:
            usage = cloudflare_daily_usage(METRICS_FILE)
            quota = "\n".join(format_cloudflare_usage_lines(usage, CLOUDFLARE_LOCAL_DAILY_REFERENCE))
        except Exception:
            quota = "дневной лимит и остаток: данные недоступны"
    if provider == "cloudflare":
        return (f"{quota}\nСегодня (UTC): ASR — {stages['asr']} обращений; "
                f"коррекция — {stages['correction']} обращений\n"
                f"{cloudflare_daily_attempt_summary(METRICS_FILE)}")
    if provider == "openrouter":
        return (f"Сегодня (UTC): ASR — {stages['asr']} обращений; "
                f"коррекция — {stages['correction']} обращений")
    if provider in {"codex", "claude"}:
        return (f"Сегодня (UTC): ASR — {stages['asr']} обращений; "
                f"коррекция — {stages['correction']} обращений")
    return (f"Статус аккаунта: {account}\n{quota}\n"
            f"Сегодня (UTC): ASR — {stages['asr']} обращений; "
            f"коррекция — {stages['correction']} обращений")

def openrouter_free_usage_text():
    """Show the local UTC-day free-endpoint counter without contacting OpenRouter."""
    from multi_provider import openrouter_local_usage, OPENROUTER_FREE_DAILY_LIMIT
    usage_date, usage_count = openrouter_local_usage()
    estimated_remaining = max(0, OPENROUTER_FREE_DAILY_LIMIT - usage_count)
    return (f"Сегодня ({usage_date} UTC): учтено {usage_count}/"
            f"{OPENROUTER_FREE_DAILY_LIMIT} · оценка остатка ≈{estimated_remaining} (базовый Free)")

def openrouter_daily_attempt_summary(metrics_path):
    """Summarize local attempts only; transcript and response text are never read out."""
    try:
        from attempt_statistics import aggregate_daily_attempts
        rows = [row for row in aggregate_daily_attempts(metrics_path)
                if row.get("provider") == "openrouter"]
        attempts = sum(row["attempts"] for row in rows)
        successes = sum(row["successes"] for row in rows)
        failures = sum(row["failures"] for row in rows)
        return (f"Попытки OpenRouter сегодня: {attempts} · успешно {successes} · "
                f"ошибки {failures}")
    except (OSError, ValueError, TypeError, KeyError):
        return "Статистика попыток OpenRouter недоступна"

def _save_qwen_setting(_widget, key, value):
    config = load_voice_config()
    corrector = config.get("corrector", {})
    corrector[key] = value
    config["corrector"] = corrector
    save_voice_config(config)

def _qwen_choice(label, key, choices, current):
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    row.pack_start(Gtk.Label(label=label, xalign=0), False, False, 0)
    combo = Gtk.ComboBoxText()
    for value, title in choices:
        combo.append(str(value), title)
    combo.set_active_id(str(current))
    combo.connect("changed", lambda widget: _save_qwen_setting(
        widget, key, next((value for value, _title in choices
                           if str(value) == widget.get_active_id()), current)))
    row.pack_start(combo, False, False, 0)
    return row

def _move_model_provider(stage, provider, direction):
    from multi_provider import safe_settings
    settings = safe_settings(load_voice_config().get("multi_provider"))
    order = settings[stage]["order"]
    index = order.index(provider)
    target = index + direction
    if 0 <= target < len(order):
        order[index], order[target] = order[target], order[index]
        _save_model_order(stage, order)

def _set_all_correctors_enabled(active):
    """Persist the single master switch without changing provider preferences."""
    global _all_correctors_syncing
    value = load_voice_config()
    from multi_provider import safe_settings
    settings = safe_settings(value.get("multi_provider"))
    settings["correction"]["enabled"] = bool(active)
    value["multi_provider"] = settings
    save_voice_config(value)
    _all_correctors_syncing = True
    try:
        for widget_name in ("all_correctors_menu_item", "all_correctors_settings_check"):
            widget = globals().get(widget_name)
            if widget is not None:
                widget.set_active(bool(active))
    finally:
        _all_correctors_syncing = False
    refresh_multi_provider_status()

_all_correctors_syncing = False

CORRECTION_SCENARIOS = (
    ("live_enabled", "После голосовой диктовки",
     "Использовать после распознавания диктовки с микрофона"),
    ("manual_enabled", "По кнопке",
     "Использовать при нажатии «Запустить корректор»"),
    ("post_transcription_enabled", "Файл / видео",
     "Использовать после транскрибации аудиофайла, видеофайла или видео по ссылке"),
)

def _on_all_correctors_toggled(widget):
    if not _all_correctors_syncing:
        _set_all_correctors_enabled(widget.get_active())

_model_provider_toggle_widgets = {}
_model_provider_toggle_syncing = False

def _register_model_provider_toggle(widget, stage, provider):
    aliases = {"groq": ("groq", "groq_primary", "groq_fallback"),
               "groq_primary": ("groq", "groq_primary", "groq_fallback"),
               "groq_fallback": ("groq", "groq_primary", "groq_fallback")}
    for alias in aliases.get(provider, (provider,)):
        _model_provider_toggle_widgets.setdefault((stage, alias), []).append(widget)
    def forget_destroyed_toggle(*_args):
        for widgets in _model_provider_toggle_widgets.values():
            while widget in widgets:
                widgets.remove(widget)
    widget.connect("destroy", forget_destroyed_toggle)

def _set_model_provider_enabled(stage, provider, active):
    global _model_provider_toggle_syncing
    value = load_voice_config()
    if stage == "asr" and provider == "cloudflare":
        value["cloudflare_asr_enabled"] = bool(active)
    elif stage == "correction" and provider == "cloudflare":
        profile = value.get("cloudflare_corrector", {})
        if not isinstance(profile, dict):
            profile = {}
        profile["enabled"] = bool(active)
        value["cloudflare_corrector"] = profile
    else:
        from multi_provider import safe_settings
        settings = safe_settings(value.get("multi_provider"))
        key = {"codex": "codex_enabled", "claude": "claude_enabled",
               "groq": "groq_enabled", "groq_primary": "groq_enabled",
               "groq_fallback": "groq_enabled", "openrouter": "openrouter_enabled",
               "google": "google_enabled"}[provider]
        settings[stage][key] = bool(active)
        value["multi_provider"] = settings
    save_voice_config(value)
    _model_provider_toggle_syncing = True
    try:
        aliases = {"groq": ("groq", "groq_primary", "groq_fallback"),
                   "groq_primary": ("groq", "groq_primary", "groq_fallback"),
                   "groq_fallback": ("groq", "groq_primary", "groq_fallback")}
        for alias in aliases.get(provider, (provider,)):
            for widget in _model_provider_toggle_widgets.get((stage, alias), ()):
                if widget.get_parent() is not None and widget.get_active() != bool(active):
                    widget.set_active(bool(active))
    finally:
        _model_provider_toggle_syncing = False
    refresh_multi_provider_status()

def _on_model_provider_toggle(widget, stage, provider):
    if not _model_provider_toggle_syncing:
        _set_model_provider_enabled(stage, provider, widget.get_active())

def _set_correction_scenario_enabled(provider, scenario, active):
    from multi_provider import safe_settings
    value = load_voice_config()
    settings = safe_settings(value.get("multi_provider"))
    correction = settings["correction"]
    correction[scenario][provider] = bool(active)
    enabled_for_provider = any(correction[route].get(provider) is True
                               for route in ("live_enabled", "manual_enabled",
                                             "post_transcription_enabled"))
    if provider == "local":
        local_corrector = value.get("corrector", {})
        if not isinstance(local_corrector, dict):
            local_corrector = {}
        local_corrector["enabled"] = enabled_for_provider
        value["corrector"] = local_corrector
    elif provider == "cloudflare":
        cloudflare = value.get("cloudflare_corrector", {})
        if not isinstance(cloudflare, dict):
            cloudflare = {}
        cloudflare["enabled"] = enabled_for_provider
        value["cloudflare_corrector"] = cloudflare
    elif provider in {"groq_primary", "groq_fallback"}:
        groq_enabled_for_any_route = any(
            correction[route].get(groq_route) is True
            for route in ("live_enabled", "manual_enabled",
                          "post_transcription_enabled")
            for groq_route in ("groq_primary", "groq_fallback"))
        correction["groq_enabled"] = groq_enabled_for_any_route
    else:
        enabled_key = {"codex": "codex_enabled", "claude": "claude_enabled",
                       "openrouter": "openrouter_enabled", "google": "google_enabled"}[provider]
        correction[enabled_key] = enabled_for_provider
    value["multi_provider"] = settings
    save_voice_config(value)
    refresh_multi_provider_status()


def _correction_provider_enabled(provider, settings, root_config):
    """Return whether the provider's master switch allows its scenario routes."""
    correction = settings["correction"]
    if provider == "local":
        return bool(root_config.get("corrector", {}).get("enabled", False))
    if provider == "cloudflare":
        profile = root_config.get("cloudflare_corrector", {})
        return isinstance(profile, dict) and profile.get("enabled") is True
    if provider in {"groq_primary", "groq_fallback"}:
        return correction["groq_enabled"]
    key = {"codex": "codex_enabled", "claude": "claude_enabled",
           "openrouter": "openrouter_enabled", "google": "google_enabled"}.get(provider)
    return bool(correction.get(key)) if key else False


def _correction_service_heading(text):
    """Create a visually distinct service heading for the correction order list."""
    heading = Gtk.Label(xalign=0)
    heading.set_markup("<b>" + GLib.markup_escape_text(text) + "</b>")
    return heading


def _bold_checkbutton_label(check, text):
    """Emphasize a checkbutton's existing label without changing its behavior."""
    label = check.get_child()
    if isinstance(label, Gtk.Label):
        label.set_markup("<b>" + GLib.markup_escape_text(text) + "</b>")

def _set_cli_model(stage, provider, model):
    from multi_provider import safe_settings, CODEX_MODELS, CLAUDE_MODELS
    value = load_voice_config()
    settings = safe_settings(value.get("multi_provider"))
    key = "codex_model" if provider == "codex" else "claude_model"
    available = {"codex": CODEX_MODELS, "claude": CLAUDE_MODELS}.get(provider, ())
    if model in available:
        settings[stage][key] = model
        value["multi_provider"] = settings
        save_voice_config(value)

def _set_provider_retry_attempts(provider, attempts):
    from multi_provider import safe_settings
    value = load_voice_config()
    attempts = max(0, min(2, int(attempts)))
    if provider == "cloudflare":
        profile = value.get("cloudflare_corrector", {})
        if not isinstance(profile, dict):
            profile = {}
        profile["retry_attempts"] = attempts
        value["cloudflare_corrector"] = profile
    else:
        settings = safe_settings(value.get("multi_provider"))
        key = {"codex": "codex_retry_attempts", "claude": "claude_retry_attempts",
               "openrouter": "openrouter_retry_attempts",
               "groq_primary": "groq_primary_retry_attempts",
               "groq_fallback": "groq_fallback_retry_attempts",
               "google": "google_retry_attempts"}[provider]
        settings["correction"][key] = attempts
        value["multi_provider"] = settings
    save_voice_config(value)

def _set_codex_retry_attempts(attempts):
    _set_provider_retry_attempts("codex", attempts)

def claude_cli_login_state():
    """Check only the exit status; never expose auth-status JSON or identifiers."""
    try:
        result = subprocess.run(["claude", "auth", "status"], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return None

def add_correction_provider_toggle(area, provider):
    """Put the primary live-correction switch at the top of its service page."""
    from multi_provider import safe_settings
    enabled_keys = {
        "codex": "codex_enabled", "claude": "claude_enabled",
        "openrouter": "openrouter_enabled", "groq_primary": "groq_enabled",
        "groq_fallback": "groq_enabled", "google": "google_enabled",
    }
    provider_labels = {"codex": "Codex", "claude": "Claude",
                       "openrouter": "OpenRouter", "groq_primary": "Groq",
                       "groq_fallback": "Groq", "google": "Gemini"}
    settings = safe_settings(load_voice_config().get("multi_provider"))["correction"]
    toggle = Gtk.CheckButton(
        label=f"Использовать {provider_labels[provider]} для live-коррекции")
    toggle.set_active(settings[enabled_keys[provider]])
    _register_model_provider_toggle(toggle, "correction", provider)
    toggle.connect("toggled", _on_model_provider_toggle, "correction", provider)
    area.pack_start(toggle, False, False, 0)
    return toggle

def configure_correction_provider_settings(_item=None, provider=None, parent=None,
                                          show_credential_status=True, show_enabled=True):
    """Route correction-provider model settings into the unified settings panel."""
    if parent is None:
        provider_tab = {
            "codex": "Codex", "claude": "Claude", "openrouter": "OpenRouter",
            "groq_primary": "Groq", "groq_fallback": "Groq", "google": "Gemini",
        }.get(provider, "Корректор")
        return open_settings_panel(provider_tab)
    from multi_provider import safe_settings, CODEX_MODELS, CLAUDE_MODELS
    provider_names = {
        "codex": "Codex", "claude": "Claude", "openrouter": "OpenRouter",
        "groq_primary": "Groq Qwen", "groq_fallback": "Groq GPT-OSS",
        "google": "Google Gemini",
    }
    if provider not in provider_names:
        return
    config = load_voice_config()
    settings = safe_settings(config.get("multi_provider"))["correction"]
    title = f"Настройки {provider_names[provider]}"
    dialog = None
    if parent is None:
        dialog = Gtk.Dialog(title=title, transient_for=None, flags=0)
        dialog.add_button("Готово", Gtk.ResponseType.CLOSE)
        dialog.set_default_size(460, -1)
        area = dialog.get_content_area()
        area.set_border_width(12)
        area.set_spacing(10)
    else:
        area = parent

    if show_enabled:
        add_correction_provider_toggle(area, provider)

    if provider == "openrouter":
        if show_credential_status:
            status = Gtk.Label(label=provider_credential_status("openrouter"), xalign=0)
            status.set_line_wrap(True)
            area.pack_start(status, False, False, 0)
        key_actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        set_key = Gtk.Button(label="Задать ключ…")
        set_key.connect("clicked", edit_provider_credential, "openrouter")
        key_actions.pack_start(set_key, True, True, 0)
        open_key = Gtk.Button(label="Открыть файл API key…")
        open_key.connect("clicked", open_credentials_file, "openrouter")
        key_actions.pack_start(open_key, True, True, 0)
        area.pack_start(key_actions, False, False, 0)
        validate = Gtk.Button(label="Явно проверить ключ")
        validate.connect("clicked", validate_provider_credentials, "openrouter")
        area.pack_start(validate, False, False, 0)

    if provider in {"codex", "claude"}:
        model_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        model_row.pack_start(Gtk.Label(label="Модель:", xalign=0), False, False, 0)
        model_combo = Gtk.ComboBoxText()
        models = CODEX_MODELS if provider == "codex" else CLAUDE_MODELS
        model_labels = {
            "gpt-6-luna": "GPT 6 Luna",
            "gpt-5.6-luna": "GPT 5.6 Luna",
            "sonnet": "Sonnet (актуальная версия)",
            "haiku": "Haiku (актуальная версия)",
            "claude-haiku-4-5-20251001": "Haiku 4.5 (зафиксированная версия)",
        }
        model_key = "codex_model" if provider == "codex" else "claude_model"
        for model in models:
            model_combo.append(model, model_labels.get(model, model))
        model_combo.set_active_id(settings[model_key])
        model_combo.connect("changed", lambda widget, p=provider:
                            _set_cli_model("correction", p, widget.get_active_id()))
        model_row.pack_start(model_combo, True, True, 0)
        area.pack_start(model_row, False, False, 0)
    else:
        fixed_models = {
            "openrouter": "OpenRouter Free: модель выбирается по недавним успешным результатам; без истории используется openrouter/free.",
            "groq_primary": "Модель: qwen/qwen3.8-27b",
            "groq_fallback": "Модель: openai/gpt-oss-20b",
            "google": "Модель: gemini-3.5-flash-lite",
        }
        model_note = Gtk.Label(label=fixed_models[provider], xalign=0)
        model_note.set_line_wrap(True)
        area.pack_start(model_note, False, False, 0)

    retry_key = {"codex": "codex_retry_attempts", "claude": "claude_retry_attempts",
                 "openrouter": "openrouter_retry_attempts",
                 "groq_primary": "groq_primary_retry_attempts",
                 "groq_fallback": "groq_fallback_retry_attempts",
                 "google": "google_retry_attempts"}[provider]
    retry_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    retry_row.pack_start(Gtk.Label(label="Дополнительные попытки (0–2):", xalign=0),
                         True, True, 0)
    retry_spin = integer_retry_control(0, 2, settings[retry_key])
    retry_spin.set_tooltip_text(
        "Повторы после временной ошибки или отказа safety guard; отмена, credentials и квота не повторяются.")
    retry_spin.connect("value-changed", lambda widget, p=provider:
                        _set_provider_retry_attempts(p, int(widget.get_value())))
    retry_row.pack_start(retry_spin, False, False, 0)
    area.pack_start(retry_row, False, False, 0)
    if provider == "openrouter":
        help_label = Gtk.Label(
            label="Каждый повтор — отдельный Free-запрос. Всего не более трёх моделей за одну коррекцию.",
            xalign=0)
        help_label.set_line_wrap(True)
        area.pack_start(help_label, False, False, 0)
    if dialog is not None:
        dialog.show_all()
        dialog.run()
        dialog.destroy()
    else:
        area.show_all()

def configure_models_settings(_item=None, parent=None, correction_parent=None):
    if parent is None:
        return open_settings_panel("Транскрибация")
    from multi_provider import safe_settings
    settings = safe_settings(load_voice_config().get("multi_provider"))
    # Keep panel construction local and immediate; account status can be refreshed
    # explicitly from the Claude service tab without blocking the GTK thread.
    claude_login = None
    dialog = None
    if parent is None:
        dialog = Gtk.Dialog(title="⚙️ Настройки моделей", transient_for=None, flags=0)
        dialog.add_button("Готово", Gtk.ResponseType.CLOSE)
        dialog.set_default_size(820, 600)
        notebook = Gtk.Notebook()
        notebook.set_border_width(10)
        dialog.get_content_area().pack_start(notebook, True, True, 0)
    else:
        notebook = parent
    labels = {
        "cloudflare": "Cloudflare", "groq": "Groq Whisper", "google": "Google Gemini",
        "codex": "Codex", "claude": "Claude", "openrouter": "OpenRouter",
        "groq_primary": "Groq Qwen", "groq_fallback": "Groq GPT-OSS",
    }
    if claude_login is False:
        labels["claude"] += " · требуется вход"
    elif claude_login is None:
        labels["claude"] += " · статус неизвестен"
    descriptions = {
        "asr": [("cloudflare", "cloudflare_asr_enabled"), ("groq", "groq_enabled"),
                 ("google", "google_enabled")],
        "correction": [("codex", "codex_enabled"), ("claude", "claude_enabled"),
                       ("openrouter", "openrouter_enabled"), ("cloudflare", "cloudflare_enabled"),
                       ("groq_primary", "groq_enabled"), ("groq_fallback", "groq_enabled"),
                       ("google", "google_enabled")],
    }
    root_config = load_voice_config()
    stage_pages = {}
    for stage, page_label in (("asr", "Голосовые"), ("correction", "Порядок")):
        stage_notebook = notebook if stage == "asr" or correction_parent is None else correction_parent
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(
            Gtk.PolicyType.NEVER,
            Gtk.PolicyType.NEVER if stage == "asr" else Gtk.PolicyType.AUTOMATIC)
        if stage == "correction":
            scroll.set_shadow_type(Gtk.ShadowType.NONE)
        rows = Gtk.ListBox()
        rows.set_margin_start(10)
        rows.set_margin_end(10)
        rows.set_selection_mode(Gtk.SelectionMode.NONE)
        key_by_provider = dict(descriptions[stage])
        order = settings[stage]["order"]
        ordered_providers = [provider for provider in order
                             if provider in key_by_provider or provider == "local"]
        if "local" not in ordered_providers:
            ordered_providers.append("local")
        for provider in ordered_providers:
            if provider == "local":
                row = Gtk.ListBoxRow()
                box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
                box.set_border_width(6)
                row.add(box)
                label = "Локальная модель (ASR)" if stage == "asr" else "Локальный корректор (Qwen)"
                if stage == "asr":
                    check = Gtk.CheckButton(label=label)
                    _bold_checkbutton_label(check, label)
                    check.set_active(True)
                    check.set_sensitive(False)
                    check.set_tooltip_text(
                        "Обязательный локальный этап. Backend и модель задаются порядком "
                        "локальных моделей голосовой транскрибации ниже.")
                    box.pack_start(check, True, True, 0)
                else:
                    box.pack_start(_correction_service_heading(label), True, True, 0)
                    for scenario, label, tooltip in CORRECTION_SCENARIOS:
                        scenario_check = Gtk.CheckButton(label=label)
                        scenario_check.set_active(
                            _correction_provider_enabled("local", settings, root_config) and
                            bool(settings[stage][scenario]["local"]))
                        scenario_check.set_tooltip_text(
                            "Использовать локальный корректор. " + tooltip)
                        scenario_check.connect(
                            "toggled", lambda widget, p="local", s=scenario:
                            _set_correction_scenario_enabled(p, s, widget.get_active()))
                        box.pack_start(scenario_check, False, False, 0)
                up = Gtk.Button(label="↑")
                down = Gtk.Button(label="↓")
                up.set_tooltip_text("Поднять локальный этап в очереди")
                down.set_tooltip_text("Опустить локальный этап в очереди")
                up.connect("clicked", lambda _button, s=stage, d=dialog:
                           (_move_model_provider(s, "local", -1),
                            d.response(Gtk.ResponseType.CLOSE) if d is not None else None))
                down.connect("clicked", lambda _button, s=stage, d=dialog:
                             (_move_model_provider(s, "local", 1),
                              d.response(Gtk.ResponseType.CLOSE) if d is not None else None))
                box.pack_start(up, False, False, 0)
                box.pack_start(down, False, False, 0)
                if stage == "correction":
                    qwen_button = Gtk.Button(label="Настройки Qwen…")
                    qwen_button.set_tooltip_text(
                        "Открыть индивидуальные настройки локального корректора Qwen")
                    qwen_button.connect("clicked", lambda *_args: open_settings_panel("Qwen"))
                    box.pack_start(qwen_button, False, False, 0)
                rows.add(row)
                continue
            key = key_by_provider[provider]
            row = Gtk.ListBoxRow()
            box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            box.set_border_width(6)
            row.add(box)
            if provider == "cloudflare":
                if stage == "asr":
                    active = root_config.get("cloudflare_asr_enabled") is True
                else:
                    profile = root_config.get("cloudflare_corrector", {})
                    active = isinstance(profile, dict) and profile.get("enabled") is True
            else:
                active = settings[stage][key]
            if stage == "correction":
                provider_enabled = _correction_provider_enabled(provider, settings, root_config)
                box.pack_start(_correction_service_heading(labels[provider]), True, True, 0)
                for scenario, label, tooltip in CORRECTION_SCENARIOS:
                    scenario_check = Gtk.CheckButton(label=label)
                    scenario_check.set_active(
                        provider_enabled and bool(settings[stage][scenario][provider]))
                    scenario_check.set_tooltip_text("Использовать этот сервис. " + tooltip)
                    scenario_check.connect(
                        "toggled", lambda widget, p=provider, s=scenario:
                        _set_correction_scenario_enabled(p, s, widget.get_active()))
                    box.pack_start(scenario_check, False, False, 0)
            else:
                check = Gtk.CheckButton(label=labels[provider])
                if stage == "asr":
                    _bold_checkbutton_label(check, labels[provider])
                check.set_active(active)
                _register_model_provider_toggle(check, stage, provider)
                check.connect("toggled", _on_model_provider_toggle, stage, provider)
                box.pack_start(check, True, True, 0)
            if stage == "asr" and provider == "groq":
                combo = Gtk.ComboBoxText()
                combo.append("accuracy", "whisper-large-v3")
                combo.append("fast", "whisper-large-v3-turbo")
                combo.set_active_id(settings[stage]["quality"])
                combo.connect("changed", update_provider_asr_quality)
                box.pack_start(combo, False, False, 0)
            elif stage == "correction" and provider == "cloudflare":
                detail_button = Gtk.Button(label="Настройки Cloudflare…")
                detail_button.set_tooltip_text(
                    "Открыть индивидуальные настройки корректора Cloudflare")
                detail_button.connect("clicked", lambda *_args: configure_cloudflare_settings(None))
                box.pack_start(detail_button, False, False, 0)
            elif stage == "correction":
                detail_names = {"codex": "Codex", "claude": "Claude",
                                "openrouter": "OpenRouter", "groq_primary": "Groq Qwen",
                                "groq_fallback": "Groq GPT-OSS", "google": "Google Gemini"}
                detail_button = Gtk.Button(label=f"Настройки {detail_names[provider]}…")
                detail_button.set_tooltip_text(
                    f"Открыть индивидуальные настройки корректора {detail_names[provider]}")
                detail_button.connect("clicked", configure_correction_provider_settings, provider)
                box.pack_start(detail_button, False, False, 0)
            up = Gtk.Button(label="↑")
            down = Gtk.Button(label="↓")
            up.set_tooltip_text("Поднять провайдера в очереди")
            down.set_tooltip_text("Опустить провайдера в очереди")
            up.connect("clicked", lambda _button, s=stage, p=provider, d=dialog:
                       (_move_model_provider(s, p, -1),
                        d.response(Gtk.ResponseType.CLOSE) if d is not None else None))
            down.connect("clicked", lambda _button, s=stage, p=provider, d=dialog:
                         (_move_model_provider(s, p, 1),
                          d.response(Gtk.ResponseType.CLOSE) if d is not None else None))
            box.pack_start(up, False, False, 0)
            box.pack_start(down, False, False, 0)
            rows.add(row)
        scroll.add(rows)
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        page.set_border_width(10)
        order_title = Gtk.Label(label=("Порядок сервисов для голосовой транскрибации" if stage == "asr"
                                      else "Порядок сервисов коррекции"), xalign=0)
        order_title.set_markup("<b>" + order_title.get_text() + "</b>")
        order_title.set_margin_start(12)
        order_title.set_margin_end(12)
        order_title.set_margin_top(6)
        if stage == "correction":
            correction_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            correction_header.set_border_width(4)
            all_correctors = Gtk.CheckButton(label="Использовать AI-корректоры")
            all_correctors.set_tooltip_text(
                "Выключение возвращает исходный текст ASR и сохраняет настройки отдельных провайдеров")
            all_correctors.set_active(settings["correction"]["enabled"])
            all_correctors.connect("toggled", _on_all_correctors_toggled)
            globals()["all_correctors_settings_check"] = all_correctors
            correction_header.pack_end(all_correctors, False, False, 0)
            page.pack_start(correction_header, False, False, 0)
        page.pack_start(order_title, False, False, 0)
        if stage == "correction":
            routes_help = Gtk.Label(
                label=("Верхний флажок «Использовать AI-корректоры» включает всю цепочку. "
                       "Две галочки у сервиса задают сценарии: «После голосовой диктовки» "
                       "или «По кнопке». При включении сценария сервис включается автоматически; "
                       "снимите обе галочки, чтобы отключить его. ↑↓ задают "
                       "порядок перехода к следующему сервису. Значок только обозначает "
                       "провайдера."),
                xalign=0)
            routes_help.set_line_wrap(True)
            routes_help.set_margin_start(12)
            routes_help.set_margin_end(12)
            page.pack_start(routes_help, False, False, 0)
        page.pack_start(scroll, True, True, 0)
        local_model_order = _local_model_chain_widget(stage, settings, root_config)
        local_model_order.set_margin_start(12)
        local_model_order.set_margin_end(12)
        local_model_order.set_margin_top(8)
        local_model_order.set_margin_bottom(8)
        page.pack_start(local_model_order, False, False, 0)
        if stage == "correction":
            prompt_button = Gtk.Button(label="Дополнительные инструкции корректора…")
            prompt_button.set_tooltip_text(
                "Применяется только к локальному корректору. Whisper использует отдельный начальный prompt.")
            prompt_button.connect("clicked", on_corrector_prompt_clicked)
            corrector_tools = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            corrector_tools.pack_start(prompt_button, False, False, 0)
            replace_button = Gtk.Button(label="Автозамены…")
            replace_button.connect("clicked", on_autoreplacements_clicked)
            corrector_tools.pack_start(replace_button, False, False, 0)
            page.pack_start(corrector_tools, False, False, 4)
        else:
            audio_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            duck = load_voice_config().get("audio_ducking", {})
            audio_toggle = Gtk.CheckButton(label="Приглушать звук при диктовке")
            audio_toggle.set_active(isinstance(duck, dict) and duck.get("enabled", False))
            audio_toggle.connect("toggled", on_audio_ducking_toggled)
            restore = Gtk.Button(label="Восстановить звук")
            restore.connect("clicked", on_audio_restore_clicked)
            audio_controls.pack_start(audio_toggle, False, False, 0)
            audio_controls.pack_start(restore, False, False, 0)
            page.pack_start(audio_controls, False, False, 4)
        page_icon = "model_services"
        stage_pages[stage] = page
        stage_notebook.append_page(
            page, settings_tab_label(page_label, page_icon, icon_size=22))
        if stage == "asr":
            file_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
            file_page.set_border_width(12)
            file_note = Gtk.Label(
                label="Файлы и видео всегда распознаются только локальными моделями.",
                xalign=0)
            file_note.set_line_wrap(True)
            file_page.pack_start(file_note, False, False, 4)
            service_order = _asr_service_order_widget(settings, labels)
            file_page.pack_start(service_order, False, False, 4)
            file_model_order = _local_model_chain_widget(
                "asr", settings, root_config, asr_scenario="file_video")
            file_page.pack_start(file_model_order, False, False, 4)
            stage_pages["asr_file_video"] = file_page
            stage_notebook.append_page(
                file_page, settings_tab_label("Файлы / видео", "video", icon_size=22))
    if dialog is not None:
        dialog.show_all()
        dialog.run()
        dialog.destroy()
        globals().pop("all_correctors_settings_check", None)
    else:
        notebook.show_all()
        notebook.connect("destroy", lambda *_args: globals().pop(
            "all_correctors_settings_check", None))
    update_model_label()
    return stage_pages

def confirm_restore_settings_defaults(parent):
    """Show the exact settings scope before writing either defaults file."""
    confirmation = Gtk.MessageDialog(
        transient_for=parent, flags=Gtk.DialogFlags.MODAL,
        message_type=Gtk.MessageType.WARNING, buttons=Gtk.ButtonsType.NONE,
        text="Восстановить настройки по умолчанию?")
    confirmation.format_secondary_text(
        "Будут сброшены config.json: модель и параметры распознавания, "
        "настройки корректора, маршруты, порядок и повторы провайдеров; "
        "плавное приглушение звука, пользовательский глоссарий и prompt также вернутся к стандартным значениям. "
        "В quality-config.json вернутся стандартные профили качества и авто-режим. "
        "Счётчики fallback и метрики GPU сохранятся. API-ключи, автозамены, "
        "горячие клавиши, модели и кэш, очередь заданий, история и логи не затрагиваются.")
    confirmation.add_button("Отмена", Gtk.ResponseType.CANCEL)
    confirmation.add_button("Восстановить", Gtk.ResponseType.ACCEPT)
    confirmation.set_default_response(Gtk.ResponseType.CANCEL)
    response = confirmation.run()
    confirmation.destroy()
    return response == Gtk.ResponseType.ACCEPT

def restore_settings_defaults():
    """Reset user preferences while preserving quality usage counters."""
    previous_quality = load_quality_config()
    if not isinstance(previous_quality, dict):
        previous_quality = json.loads(json.dumps(DEFAULT_QUALITY_CONFIG))
    defaults = json.loads(json.dumps(DEFAULT_QUALITY_CONFIG))
    for key in ("gpu_metrics", "fallback_history"):
        if isinstance(previous_quality.get(key), dict):
            defaults[key] = previous_quality[key]
    if not save_quality_config(defaults):
        return False, "Не удалось сохранить стандартные профили качества."
    result = daemon_model_control("restore_settings_defaults", timeout=10)
    if not isinstance(result, dict) or result.get("status") != "ok":
        if not save_quality_config(previous_quality):
            return False, "Не удалось сбросить config.json и восстановить предыдущие профили качества."
        return False, "Не удалось сбросить config.json; прежние профили качества восстановлены."
    return True, None

def configure_app_settings(_item=None, initial_tab=None):
    """Show one settings window with model-stage and service tabs."""
    global _settings_config_transaction
    global _settings_panel_dialog, _settings_panel_notebook, _settings_panel_pages
    global codex_settings_status_label, codex_settings_attempt_label
    opening_config = load_voice_config()
    opening_quality_config = json.loads(json.dumps(load_quality_config()))
    _settings_config_transaction = {
        "voice": json.loads(json.dumps(opening_config)),
        "quality": json.loads(json.dumps(opening_quality_config)),
        "prepare_asr": False,
    }

    def commit_pending_settings():
        """Persist both staged settings stores, restoring both on partial failure."""
        global _settings_config_transaction
        transaction = _settings_config_transaction
        if not isinstance(transaction, dict):
            return True, False
        pending_voice = json.loads(json.dumps(transaction["voice"]))
        pending_quality = json.loads(json.dumps(transaction["quality"]))
        prepare_asr = bool(transaction.get("prepare_asr"))
        _settings_config_transaction = None
        voice_saved = save_voice_config(pending_voice)
        quality_saved = save_quality_config(pending_quality)
        if not voice_saved or not quality_saved:
            save_voice_config(opening_config)
            save_quality_config(opening_quality_config)
            _settings_config_transaction = {
                "voice": json.loads(json.dumps(pending_voice)),
                "quality": json.loads(json.dumps(pending_quality)),
                "prepare_asr": prepare_asr,
            }
            return False, False
        return True, prepare_asr

    dialog = Gtk.Dialog(title="⚙️ Настройки", transient_for=None, flags=0)
    dialog.add_button("По умолчанию", Gtk.ResponseType.APPLY)
    dialog.add_button("Отмена", Gtk.ResponseType.CANCEL)
    dialog.add_button("Применить", Gtk.ResponseType.CLOSE)
    dialog.add_button("ОК", Gtk.ResponseType.OK)
    dialog.set_default_response(Gtk.ResponseType.OK)
    dialog.set_default_size(1120, 900)
    notebook = Gtk.Notebook()
    notebook.set_border_width(10)
    notebook.set_show_tabs(True)
    notebook.set_tab_pos(Gtk.PositionType.LEFT)
    notebook.set_scrollable(False)
    dialog.get_content_area().pack_start(notebook, True, True, 0)

    # Top level selects the workflow; tabs inside it hold order and local models.
    transcription_notebook = Gtk.Notebook()
    transcription_notebook.set_tab_pos(Gtk.PositionType.TOP)
    transcription_notebook.set_scrollable(True)
    correction_notebook = Gtk.Notebook()
    correction_notebook.set_tab_pos(Gtk.PositionType.TOP)
    correction_notebook.set_scrollable(True)
    transcription_scroll = Gtk.ScrolledWindow()
    transcription_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    transcription_scroll.add(transcription_notebook)
    correction_scroll = Gtk.ScrolledWindow()
    correction_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    correction_scroll.add(correction_notebook)
    transcription_outer = notebook.append_page(
        transcription_scroll, settings_tab_label("Транскрибация", "transcribe", icon_size=22))
    correction_outer = notebook.append_page(
        correction_scroll, settings_tab_label("Корректор", "corrector", icon_size=22))
    stage_pages = configure_models_settings(
        parent=transcription_notebook, correction_parent=correction_notebook)
    page_widgets = configure_local_recognition_settings(parent=transcription_notebook)
    page_numbers = {
        "Транскрибация": transcription_outer,
        "Корректор": correction_outer,
        "Порядок": (transcription_outer, transcription_notebook,
                    transcription_notebook.page_num(stage_pages["asr"])),
        "Порядок файлов и видео": (
            transcription_outer, transcription_notebook,
            transcription_notebook.page_num(stage_pages["asr_file_video"])),
    }
    page_icons = {
        "Транскрибация": "transcribe", "Корректор": "corrector",
        "Локальный Whisper": "transcribe", "Локальный GigaAM": "gigaam",
        "Cloudflare": "cloudflare", "OpenRouter": "openrouter",
        "Codex": "codex", "Claude": "claude", "Groq": "groq",
        "Qwen": "qwen", "Локальный Qwen": "qwen",
        "Gemini": "google",
    }

    help_notebooks = {}
    help_scrolls = {}
    for category in ("Диагностика", "Справка"):
        category_notebook = Gtk.Notebook()
        category_notebook.set_tab_pos(Gtk.PositionType.TOP)
        category_notebook.set_scrollable(True)
        category_scroll = Gtk.ScrolledWindow()
        category_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        category_scroll.add(category_notebook)
        help_notebooks[category] = category_notebook
        help_scrolls[category] = category_scroll
    help_page_indices = {}
    help_refreshers = {}
    help_loaded_pages = set()

    def render_markdown(buffer, markdown_text, text_view):
        """Render guide and changelog with bulk Pango markup insertions."""
        inline_pattern = re.compile(
            r"(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`|\[([^]]+)\]\(([^)]+)\))")

        def inline_markup(text):
            cursor = 0
            output = []
            for match in inline_pattern.finditer(text):
                if match.start() > cursor:
                    output.append(html.escape(text[cursor:match.start()]))
                if match.group(2) is not None:
                    value = f"<b>{html.escape(match.group(2))}</b>"
                elif match.group(3) is not None:
                    value = f"<i>{html.escape(match.group(3))}</i>"
                elif match.group(4) is not None:
                    value = (f'<span font_family="monospace" background="#eeeeee" '
                             f'foreground="#7b2455">{html.escape(match.group(4))}</span>')
                else:
                    value = f"{html.escape(match.group(5))} ({html.escape(match.group(6))})"
                output.append(value)
                cursor = match.end()
            if cursor < len(text):
                output.append(html.escape(text[cursor:]))
            return "".join(output)

        def insert_markup_chunk(parts):
            if parts:
                buffer.insert_markup(buffer.get_end_iter(), "".join(parts), -1)
                parts.clear()

        buffer.set_text("")
        tag_table = buffer.get_tag_table()
        table_copy_tag = (tag_table.lookup("markdown-table-copy") or
                          buffer.create_tag("markdown-table-copy", invisible=True))
        markup_parts = []
        in_code = False
        previous_was_heading = False
        lines = markdown_text.splitlines()
        line_index = 0
        while line_index < len(lines):
            line = lines[line_index]
            stripped = line.strip()
            if not stripped and previous_was_heading:
                previous_was_heading = False
                line_index += 1
                continue
            if stripped.startswith("```"):
                in_code = not in_code
                line_index += 1
                continue
            if in_code:
                markup_parts.append(
                    f'<span font_family="monospace" foreground="#333333">'
                    f'{html.escape(line)}\n</span>')
                line_index += 1
                continue
            if stripped.startswith("|") and line_index + 1 < len(lines):
                table_lines = []
                while line_index < len(lines) and lines[line_index].strip().startswith("|"):
                    table_lines.append(lines[line_index])
                    line_index += 1
                table_rows = [[cell.strip() for cell in row.strip().strip("|").split("|")]
                              for row in table_lines]
                if len(table_rows) >= 2 and all(
                        re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
                        for cell in table_rows[1]):
                    headers = table_rows[0]
                    data_rows = [row for row in table_rows[2:]
                                 if len(row) == len(headers)]
                    insert_markup_chunk(markup_parts)
                    store = Gtk.ListStore(*([str] * len(headers)))
                    for row in data_rows:
                        store.append(row)
                    table = Gtk.TreeView(model=store)
                    table.set_headers_visible(True)
                    table.set_rules_hint(True)
                    widths = ([150, 260, 420] if len(headers) == 3 else
                              [max(180, 800 // len(headers))] * len(headers))
                    for column_index, title in enumerate(headers):
                        renderer = Gtk.CellRendererText()
                        renderer.set_property("wrap-mode", Pango.WrapMode.WORD_CHAR)
                        renderer.set_property("wrap-width", widths[column_index])
                        column = Gtk.TreeViewColumn(title, renderer,
                                                    text=column_index)
                        column.set_sizing(Gtk.TreeViewColumnSizing.FIXED)
                        column.set_fixed_width(widths[column_index])
                        table.append_column(column)
                    table_scroll = Gtk.ScrolledWindow()
                    table_scroll.set_policy(Gtk.PolicyType.AUTOMATIC,
                                            Gtk.PolicyType.AUTOMATIC)
                    table_scroll.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
                    table_scroll.set_min_content_height(180)
                    table_scroll.add(table)
                    anchor = buffer.create_child_anchor(buffer.get_end_iter())
                    text_view.add_child_at_anchor(table_scroll, anchor)
                    copy_data = "\n".join("\t".join(row)
                                           for row in [headers] + data_rows) + "\n"
                    buffer.insert_with_tags(buffer.get_end_iter(), "\n" + copy_data,
                                            table_copy_tag)
                    continue
                for row in table_rows:
                    markup_parts.append(inline_markup(" | ".join(row)) + "\n")
                continue
            heading = re.match(r"^(#{1,3})\s+(.+)$", line)
            if heading:
                level = min(len(heading.group(1)), 3)
                size = {1: "xx-large", 2: "x-large", 3: "large"}[level]
                heading_markup = (f'<span size="{size}" weight="bold">'
                                  f'{inline_markup(heading.group(2))}</span>')
                markup_parts.append(heading_markup + "\n")
                previous_was_heading = True
            elif re.match(r"^\s*(?:[-*_]\s*){3,}$", line):
                markup_parts.append("────────────────────────\n")
                line_index += 1
                continue
            else:
                previous_was_heading = False
                bullet = re.match(r"^(\s*)[-*+]\s+(.+)$", line)
                numbered = re.match(r"^(\s*)(\d+\.)\s+(.+)$", line)
                quote = re.match(r"^\s*>\s?(.*)$", line)
                if bullet:
                    markup_parts.append(html.escape(bullet.group(1) + "• ") +
                                        inline_markup(bullet.group(2)) + "\n")
                elif numbered:
                    markup_parts.append(
                        html.escape(numbered.group(1) + numbered.group(2) + " ") +
                        inline_markup(numbered.group(3)) + "\n")
                elif quote:
                    markup_parts.append(
                        f'<i><span foreground="#555555">{inline_markup(quote.group(1))}</span></i>\n')
                else:
                    markup_parts.append(inline_markup(line) + "\n")
            line_index += 1
        insert_markup_chunk(markup_parts)

    def add_help_page(title, icon, heading, description, content_provider=None,
                      log_selector=False, content_widget_factory=None,
                      markdown_content=False):
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        page.set_border_width(20)
        title_label = Gtk.Label(label=heading, xalign=0)
        title_label.set_markup("<b>" + GLib.markup_escape_text(heading) + "</b>")
        page.pack_start(title_label, False, False, 0)
        description_label = Gtk.Label(label=description, xalign=0)
        description_label.set_line_wrap(True)
        page.pack_start(description_label, False, False, 0)

        if content_widget_factory is not None:
            page.pack_start(content_widget_factory(), True, True, 0)
            scroll = Gtk.ScrolledWindow()
            scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            scroll.add(page)
            category = "Диагностика" if title in {"Статистика", "История", "Логи"} else "Справка"
            page_index = help_notebooks[category].append_page(
                scroll, settings_tab_label(title, icon, icon_size=22))
            help_page_indices[title] = (category, page_index)
            return

        selector = None
        if log_selector:
            selector = Gtk.ComboBoxText()
            selector.append("errors", "Ошибки")
            selector.append("work", "События работы")
            selector.set_active_id("errors")
            page.pack_start(selector, False, False, 0)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        refresh_button = Gtk.Button(label="Обновить")
        copy_button = Gtk.Button(label="Скопировать текст")
        actions.pack_start(refresh_button, False, False, 0)
        actions.pack_start(copy_button, False, False, 0)
        reset_status = Gtk.Label(label="", xalign=0)
        page.pack_start(actions, False, False, 0)

        text_view = Gtk.TextView()
        text_view.set_editable(False)
        text_view.set_cursor_visible(True)
        text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        text_view.set_monospace(not markdown_content)
        if markdown_content:
            text_view.set_left_margin(16)
            text_view.set_right_margin(16)
            text_view.set_pixels_above_lines(2)
            text_view.set_pixels_below_lines(2)
        text_buffer = text_view.get_buffer()
        text_scroll = Gtk.ScrolledWindow()
        text_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        text_scroll.set_min_content_height(300)
        text_scroll.add(text_view)
        page.pack_start(text_scroll, True, True, 0)
        page.pack_start(reset_status, False, False, 0)

        def current_text():
            if log_selector:
                return content_provider(selector.get_active_id() or "errors")
            return content_provider()

        def refresh_content(*_args):
            content = current_text()
            if markdown_content:
                render_markdown(text_buffer, content, text_view)
            else:
                text_buffer.set_text(content)
            reset_status.set_text("")

        def copy_content(*_args):
            start, end = text_buffer.get_bounds()
            clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
            copied_text = text_buffer.get_text(start, end, True).replace("\ufffc", "")
            clipboard.set_text(copied_text, -1)
            reset_status.set_text("Текст скопирован в буфер обмена.")

        refresh_button.connect("clicked", refresh_content)
        copy_button.connect("clicked", copy_content)
        if log_selector:
            selector.connect("changed", refresh_content)
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.add(page)
        category = "Диагностика" if title in {"Статистика", "История", "Логи"} else "Справка"
        page_index = help_notebooks[category].append_page(
            scroll, settings_tab_label(title, icon, icon_size=22))
        help_page_indices[title] = (category, page_index)
        help_refreshers[(category, page_index)] = refresh_content
        text_buffer.set_text("Содержимое загрузится при открытии вкладки.")

    add_help_page(
        "Статистика", "stats",
        "Статистика за сегодня",
        "Сводка попыток за текущий день UTC с фильтрами и сортируемыми столбцами.",
        content_widget_factory=create_attempt_statistics_widget)
    add_help_page(
        "Руководство", "help",
        "Руководство Whisper Dictate",
        "Руководство отображается с форматированием; текст можно выделить и скопировать.",
        lambda: read_installed_document("guide.md"), markdown_content=True)
    add_help_page(
        "История", "history",
        "История транскрибации",
        "Показаны последние записи; текст можно выделить и скопировать.",
        get_history_preview_text)
    add_help_page(
        "Логи", "logs",
        "Диагностические логи",
        "Показываются последние подходящие строки; каждый журнал читается с ограниченным хвостом 512 KiB.",
        lambda kind: _read_log_excerpt(kind), log_selector=True)
    add_help_page(
        "История изменений", "document",
        "История изменений программы",
        "История изменений отображается с форматированием; текст можно выделить и скопировать.",
        lambda: read_installed_document("CHANGELOG.md"), markdown_content=True)

    def new_service_page(title):
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        page.set_border_width(12)
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.add(page)
        page_numbers[title] = notebook.append_page(
            scroll, settings_tab_label(title, page_icons[title], icon_size=22))
        page_widgets[title] = scroll
        return page

    def service_settings_group(page, title):
        frame = Gtk.Frame(label=title)
        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        body.set_border_width(10)
        frame.add(body)
        page.pack_start(frame, False, False, 0)
        return body

    cloudflare_page = new_service_page("Cloudflare")
    configure_cloudflare_settings(parent=cloudflare_page)

    openrouter_page = new_service_page("OpenRouter")
    add_correction_provider_toggle(openrouter_page, "openrouter")
    openrouter_account_group = service_settings_group(
        openrouter_page, "Учётные данные и использование")
    openrouter_credential_status = Gtk.Label(
        label=provider_credential_status("openrouter"), xalign=0)
    openrouter_credential_status.set_line_wrap(True)
    openrouter_account_group.pack_start(openrouter_credential_status, False, False, 0)
    openrouter_usage = Gtk.Label(label=provider_account_usage_text("openrouter"), xalign=0)
    openrouter_usage.set_line_wrap(True)
    openrouter_account_group.pack_start(openrouter_usage, False, False, 0)
    openrouter_statistics_group = service_settings_group(
        openrouter_page, "Лимиты и статистика OpenRouter")
    openrouter_free_usage = Gtk.Label(label=openrouter_free_usage_text(), xalign=0)
    openrouter_free_usage.set_line_wrap(True)
    openrouter_statistics_group.pack_start(openrouter_free_usage, False, False, 0)
    openrouter_attempts = Gtk.Label(
        label=openrouter_daily_attempt_summary(METRICS_FILE), xalign=0)
    openrouter_attempts.set_line_wrap(True)
    openrouter_statistics_group.pack_start(openrouter_attempts, False, False, 0)
    openrouter_quota = Gtk.Label(
        label="Free: 50/день (1 000 после $10 credits), 20/мин; ошибки считаются; "
              "остаток аккаунта — Activity",
        xalign=0)
    openrouter_quota.set_line_wrap(True)
    openrouter_statistics_group.pack_start(openrouter_quota, False, False, 0)
    activity = Gtk.Button(label="Открыть статистику OpenRouter…")
    activity.connect("clicked", lambda *_args: webbrowser.open("https://openrouter.ai/activity"))
    openrouter_statistics_group.pack_start(activity, False, False, 0)

    openrouter_settings_group = service_settings_group(openrouter_page, "Настройки корректора")
    configure_correction_provider_settings(None, "openrouter", parent=openrouter_settings_group,
                                           show_credential_status=False, show_enabled=False)
    provider_dialog_status_labels["openrouter"] = openrouter_credential_status

    def refresh_openrouter_page(*_args):
        openrouter_credential_status.set_text(provider_credential_status("openrouter"))
        openrouter_usage.set_text(provider_account_usage_text("openrouter"))
        openrouter_free_usage.set_text(openrouter_free_usage_text())
        openrouter_attempts.set_text(openrouter_daily_attempt_summary(METRICS_FILE))

    notebook.connect("switch-page", lambda _notebook, _page, page_number:
                      refresh_openrouter_page()
                      if page_number == page_numbers["OpenRouter"] else None)

    def cli_service_page(provider, title, status_text):
        page = new_service_page(title)
        add_correction_provider_toggle(page, provider)
        account_group = service_settings_group(page, "Учётные данные и использование")
        status = Gtk.Label(label=status_text, xalign=0)
        status.set_line_wrap(True)
        account_group.pack_start(status, False, False, 0)
        usage = Gtk.Label(label=provider_account_usage_text(provider), xalign=0)
        usage.set_line_wrap(True)
        account_group.pack_start(usage, False, False, 0)
        if provider == "codex":
            latest = Gtk.Label(label=format_codex_attempt_status(
                latest_codex_attempt(METRICS_FILE)), xalign=0)
            latest.set_line_wrap(True)
            account_group.pack_start(latest, False, False, 0)
            codex_settings_status_label = status
            codex_settings_attempt_label = latest
            account_group.pack_start(Gtk.Label(
                label="Остаток в активной сессии Codex: команда /status", xalign=0),
                False, False, 0)
            usage_link = Gtk.Button(label="Открыть лимиты использования Codex…")
            usage_link.connect("clicked", lambda *_args: webbrowser.open(
                "https://chatgpt.com/codex/settings/usage"))
            account_group.pack_start(usage_link, False, False, 0)
        if provider == "codex":
            refresh = Gtk.Button(label="Обновить статус Codex")
            def refresh_codex_settings_status(*_args):
                refresh_codex_account_status()
                usage.set_text(provider_account_usage_text("codex"))
            refresh.connect("clicked", refresh_codex_settings_status)
            account_group.pack_start(refresh, False, False, 0)
        elif provider == "claude":
            refresh = Gtk.Button(label="Обновить статус Claude")
            def refresh_claude_settings_status(*_args):
                state = claude_cli_login_state()
                status.set_text("Учётная запись Claude активна" if state is True else
                                "Учётная запись Claude недоступна или статус неизвестен")
                usage.set_text(provider_account_usage_text("claude"))
            refresh.connect("clicked", refresh_claude_settings_status)
            account_group.pack_start(refresh, False, False, 0)
        configure_correction_provider_settings(None, provider, parent=page,
                                               show_enabled=False)
        return page

    cli_service_page("codex", "Codex", "Статус учётной записи Codex не обновлён.")
    cli_service_page("claude", "Claude", "Статус учётной записи Claude не обновлён.")

    groq_page = new_service_page("Groq")
    configure_provider_settings(None, "groq", parent=groq_page)
    groq_qwen_group = _provider_settings_group(groq_page, "Корректор · Groq Qwen")
    configure_correction_provider_settings(None, "groq_primary", parent=groq_qwen_group,
                                           show_enabled=False)
    groq_gpt_group = _provider_settings_group(groq_page, "Резервный корректор · Groq GPT-OSS")
    configure_correction_provider_settings(None, "groq_fallback", parent=groq_gpt_group,
                                           show_enabled=False)

    google_page = new_service_page("Gemini")
    configure_provider_settings(None, "google", parent=google_page)
    google_correction_group = _provider_settings_group(
        google_page, "Корректор · Gemini Flash-Lite")
    configure_correction_provider_settings(None, "google", parent=google_correction_group,
                                           show_enabled=False)

    qwen_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    qwen_page.set_border_width(12)
    qwen_scroll = Gtk.ScrolledWindow()
    qwen_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    qwen_scroll.add(qwen_page)
    correction_notebook.append_page(
        qwen_scroll, settings_tab_label("Qwen", "qwen", icon_size=22))
    page_numbers["Qwen"] = (correction_outer, correction_notebook,
                            correction_notebook.page_num(qwen_scroll))
    page_numbers["Порядок корректора"] = (
        correction_outer, correction_notebook,
        correction_notebook.page_num(stage_pages["correction"]))
    qwen_config = load_voice_config()
    corrector_config = qwen_config.get("corrector", {})
    local_enabled = Gtk.CheckButton(label="Включить локальный корректор")
    local_enabled.set_active(bool(corrector_config.get("enabled", False)))
    local_enabled.connect("toggled", on_corrector_enabled_toggled)
    qwen_page.pack_start(local_enabled, False, False, 0)
    from multi_provider import safe_settings as safe_qwen_settings
    qwen_page.pack_start(Gtk.Label(label="Доступные локальные корректоры и порядок выбора", xalign=0), False, False, 0)
    qwen_page.pack_start(_local_model_chain_widget("correction",
        safe_qwen_settings(qwen_config.get("multi_provider")), qwen_config), False, False, 0)
    qwen_models = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    prompt_button = Gtk.Button(label="Дополнительные инструкции…")
    prompt_button.connect("clicked", on_corrector_prompt_clicked)
    replace_button = Gtk.Button(label="Автозамены…")
    replace_button.connect("clicked", on_autoreplacements_clicked)
    qwen_models.pack_start(prompt_button, False, False, 0)
    qwen_models.pack_start(replace_button, False, False, 0)
    qwen_page.pack_start(qwen_models, False, False, 0)
    qwen_page.pack_start(Gtk.Label(label="Модель и производительность", xalign=0), False, False, 4)
    model_choices = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    model_group = []
    for model_id, label in (("qwen3.5:4b-q4_K_M", "Qwen3.5 4B"),
                            ("qwen3.5:2b-q4_K_M", "Qwen3.5 2B")):
        radio = Gtk.RadioButton.new_with_label_from_widget(model_group[0] if model_group else None, label)
        model_group.append(radio)
        radio.set_active(corrector_config.get("model") == model_id)
        radio.connect("toggled", on_corrector_model_selected, model_id)
        model_choices.pack_start(radio, False, False, 0)
    qwen_page.pack_start(model_choices, False, False, 0)
    for label, key, values, default in (
        ("GPU offload:", "num_gpu", [(0, "CPU"), (4, "4 слоя"), (8, "8 слоёв"), (12, "12 слоёв")], 8),
        ("Контекст:", "num_ctx", [(1024, "1024"), (2048, "2048"), (4096, "4096"), (8192, "8192")], 2048),
        ("Потоки CPU:", "num_thread", [(2, "2"), (4, "4"), (8, "8")], 4),
        ("Таймаут:", "timeout_seconds", [(5, "5 с"), (10, "10 с"), (15, "15 с"), (30, "30 с"), (60, "60 с"), (120, "2 мин"), (300, "5 мин")], 60),
        ("Keep alive:", "keep_alive", [(0, "Выгружать сразу"), ("30s", "30 секунд"), ("5m", "5 минут")], "5m"),
        ("Temperature:", "temperature", [(0, "0"), (0.1, "0.1"), (0.2, "0.2")], 0),
    ):
        qwen_page.pack_start(_qwen_choice(label, key, values,
                             corrector_config.get(key, default)), False, False, 0)
    vram_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    vram_label = Gtk.Label(label="VRAM: данные ещё не обновлены", xalign=0)
    refresh_vram = Gtk.Button(label="Обновить VRAM")
    refresh_vram.connect("clicked", lambda *_args: vram_label.set_text(
        (lambda info: f"VRAM: {info['used_mb']} / {info['total_mb']} MiB · свободно {info['free_gb']} GiB"
         if info else "VRAM: данные NVIDIA недоступны")(get_gpu_memory_info_tray())))
    vram_row.pack_start(vram_label, True, True, 0)
    vram_row.pack_start(refresh_vram, False, False, 0)
    qwen_page.pack_start(vram_row, False, False, 0)
    corrector_loaded_status_label = Gtk.Label(label="Qwen: статус загрузки ещё не проверен", xalign=0)
    globals()["corrector_loaded_status_label"] = corrector_loaded_status_label
    qwen_page.pack_start(corrector_loaded_status_label, False, False, 0)
    corrector_unload_settings_button = Gtk.Button(label="Выгрузить Qwen")
    globals()["corrector_unload_settings_button"] = corrector_unload_settings_button
    corrector_unload_settings_button.connect("clicked", unload_corrector_model)
    qwen_page.pack_start(corrector_unload_settings_button, False, False, 0)
    update_corrector_unload_state()

    page_numbers["Whisper"] = (transcription_outer, transcription_notebook,
                               transcription_notebook.page_num(page_widgets["Локальный Whisper"]))
    page_numbers["GigaAM"] = (transcription_outer, transcription_notebook,
                              transcription_notebook.page_num(page_widgets["Локальный GigaAM"]))
    page_numbers["Локальный Whisper"] = page_numbers["Whisper"]
    page_numbers["Локальный GigaAM"] = page_numbers["GigaAM"]
    page_numbers["Локальный Qwen"] = page_numbers["Qwen"]
    diagnostics_outer = notebook.append_page(
        help_scrolls["Диагностика"], settings_tab_label("Диагностика", "logs", icon_size=22))
    help_outer = notebook.append_page(
        help_scrolls["Справка"], settings_tab_label("Справка", "help", icon_size=22))
    page_numbers["Справка"] = help_outer
    page_numbers["Диагностика"] = diagnostics_outer
    for help_title, (category, inner_index) in help_page_indices.items():
        outer_index = diagnostics_outer if category == "Диагностика" else help_outer
        page_numbers[help_title] = (outer_index, help_notebooks[category], inner_index)
    page_numbers["Справка и диагностика"] = help_outer
    page_numbers["Помощь"] = help_outer

    def refresh_selected_help_page(_notebook, _page, page_index):
        category_notebook = _notebook
        category = next((key for key, value in help_notebooks.items()
                         if value is category_notebook), None)
        refresher = help_refreshers.get((category, page_index))
        load_key = (category, page_index)
        if refresher is not None and load_key not in help_loaded_pages:
            refresher()
            help_loaded_pages.add(load_key)

    def refresh_help_when_opened(_notebook, _page, page_index):
        if page_index in {help_outer, diagnostics_outer}:
            category = "Диагностика" if page_index == diagnostics_outer else "Справка"
            category_notebook = help_notebooks[category]
            refresh_selected_help_page(category_notebook, None,
                                       category_notebook.get_current_page())

    for category_notebook in help_notebooks.values():
        category_notebook.connect("switch-page", refresh_selected_help_page)
    notebook.connect("switch-page", refresh_help_when_opened)

    _settings_panel_dialog = dialog
    _settings_panel_notebook = notebook
    _settings_panel_pages = page_numbers

    def clear_settings_panel(*_args):
        global _settings_panel_dialog, _settings_panel_notebook, _settings_panel_pages
        global codex_settings_status_label, codex_settings_attempt_label
        _settings_panel_dialog = None
        _settings_panel_notebook = None
        _settings_panel_pages = {}
        whisper_model_settings_widgets.clear()
        local_asr_backend_settings_widgets.clear()
        codex_settings_status_label = None
        codex_settings_attempt_label = None
        provider_dialog_status_labels.pop("openrouter", None)

    dialog.connect("destroy", clear_settings_panel)
    dialog.show_all()
    if initial_tab in page_numbers:
        GLib.idle_add(_select_settings_panel_page, initial_tab)
    defaults_restored = False
    while True:
        response = dialog.run()
        if response in (Gtk.ResponseType.CLOSE, Gtk.ResponseType.OK):
            saved, prepare_asr = commit_pending_settings()
            if not saved:
                show_error_dialog("Не удалось применить настройки",
                                  "Не удалось сохранить настройки Whisper. Изменения оставлены в окне; проверьте доступность хранилища и повторите сохранение.")
                continue
            if prepare_asr:
                prepare_selected_asr()
            current_config = load_voice_config()
            current_quality_config = load_quality_config()
            opening_config = current_config
            opening_quality_config = json.loads(json.dumps(current_quality_config))
            if response == Gtk.ResponseType.OK:
                break
            _settings_config_transaction = {
                "voice": json.loads(json.dumps(opening_config)),
                "quality": json.loads(json.dumps(opening_quality_config)),
                "prepare_asr": False,
            }
            continue
        if response != Gtk.ResponseType.APPLY:
            break
        if not confirm_restore_settings_defaults(dialog):
            continue
        _settings_config_transaction = None
        defaults_restored, reset_error = restore_settings_defaults()
        if defaults_restored:
            break
        opening_config = load_voice_config()
        opening_quality_config = json.loads(json.dumps(load_quality_config()))
        _settings_config_transaction = {
            "voice": json.loads(json.dumps(opening_config)),
            "quality": json.loads(json.dumps(opening_quality_config)),
            "prepare_asr": False,
        }
        show_error_dialog("Не удалось восстановить настройки", reset_error)
    if response in (Gtk.ResponseType.CANCEL, Gtk.ResponseType.DELETE_EVENT):
        _settings_config_transaction = None
        update_model_label()
        update_corrector_label()
        refresh_multi_provider_status()
    else:
        _settings_config_transaction = None
    dialog.destroy()
    if defaults_restored:
        update_model_label()
        update_corrector_label()
        refresh_multi_provider_status()
        show_info_dialog(
            "Настройки восстановлены",
            "Стандартные значения применены. Откройте настройки заново, чтобы увидеть обновлённые поля.")

def edit_provider_credential(_item, provider):
    from multi_provider import store_credential
    dialog = Gtk.Dialog(title=f"Ключ API: {provider}", transient_for=None, flags=0)
    dialog.add_buttons("Отмена", Gtk.ResponseType.CANCEL, "Сохранить", Gtk.ResponseType.OK)
    entry = Gtk.Entry()
    entry.set_visibility(False)
    entry.set_input_purpose(Gtk.InputPurpose.PASSWORD)
    entry.set_placeholder_text("Вставьте ключ API; значение не отображается")
    area = dialog.get_content_area()
    area.set_border_width(12)
    area.pack_start(entry, False, False, 0)
    dialog.show_all()
    response = dialog.run()
    secret = entry.get_text() if response == Gtk.ResponseType.OK else ""
    dialog.destroy()
    if secret:
        try:
            store_credential(provider, secret)
            notify_send("Whisper Dictate", f"Ключ {provider} сохранён с правами 0600")
        except Exception:
            notify_send("Whisper Dictate", f"Не удалось безопасно сохранить ключ {provider}")
    refresh_multi_provider_status()

def validate_provider_credentials(_item, provider):
    from multi_provider import validate_credential, ProviderError
    try:
        validate_credential(provider)
    except ProviderError as error:
        notify_send("Whisper Dictate", f"Проверка {provider}: {error.category}")
    except Exception:
        notify_send("Whisper Dictate", f"Проверка {provider}: ошибка конфигурации")
    else:
        notify_send("Whisper Dictate", f"Ключ {provider} принят API")

def provider_credential_status(provider):
    from multi_provider import credentials_configured
    try:
        return "API key: заполнен, права файла безопасны" if credentials_configured(provider) else \
               "API key: не заполнен или права файла небезопасны"
    except OSError:
        return "API key: файл недоступен"

def _provider_settings_group(parent, title):
    frame = Gtk.Frame(label=title)
    body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    body.set_border_width(10)
    frame.add(body)
    parent.pack_start(frame, False, False, 0)
    return body

def configure_provider_settings(_item, provider, parent=None):
    if parent is None:
        return open_settings_panel({"groq": "Groq", "google": "Gemini"}.get(provider))
    from multi_provider import safe_settings
    provider_info = {
        "groq": ("Groq", [("correction", "groq_enabled", "Использовать Groq для корректора"),
                           ("asr", "groq_enabled", "Использовать Groq для Whisper")]),
        "google": ("Google Gemini", [("correction", "google_enabled", "Использовать Gemini для корректора"),
                                     ("asr", "google_enabled", "Использовать Gemini для Whisper")]),
        "openrouter": ("OpenRouter", [("correction", "openrouter_enabled", "Использовать OpenRouter для корректора")]),
    }
    if provider not in provider_info:
        return
    display_name, switches = provider_info[provider]
    dialog = None
    if parent is None:
        dialog = Gtk.Dialog(title=f"Настройки {display_name}", transient_for=None, flags=0)
        dialog.add_button("Готово", Gtk.ResponseType.CLOSE)
        dialog.set_default_size(420, -1)
        area = dialog.get_content_area()
        area.set_border_width(12)
        area.set_spacing(8)
    else:
        area = parent
    settings = safe_settings(load_voice_config().get("multi_provider"))
    switches_group = _provider_settings_group(area, "Использование сервиса")
    for stage, key, label in switches:
        checkbox = Gtk.CheckButton(label=label)
        checkbox.set_active(settings[stage][key])
        _register_model_provider_toggle(checkbox, stage, provider)
        checkbox.connect("toggled", _on_model_provider_toggle, stage, provider)
        switches_group.pack_start(checkbox, False, False, 0)

    account_group = _provider_settings_group(area, "Учётная запись и использование")
    status = Gtk.Label(label=provider_credential_status(provider), xalign=0)
    provider_dialog_status_labels[provider] = status
    status.set_line_wrap(True)
    account_group.pack_start(status, False, False, 0)
    usage = Gtk.Label(label=provider_account_usage_text(provider), xalign=0)
    usage.set_line_wrap(True)
    account_group.pack_start(usage, False, False, 0)
    provider_dialog_usage_labels[provider] = usage

    if provider == "groq":
        whisper_group = _provider_settings_group(area, "Параметры Whisper")
        quality_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        quality_row.pack_start(Gtk.Label(label="Режим Whisper:", xalign=0), False, False, 0)
        quality = Gtk.ComboBoxText()
        quality.append("accuracy", "Точность (whisper-large-v3)")
        quality.append("fast", "Быстро (whisper-large-v3-turbo)")
        quality.set_active_id(settings["asr"]["quality"])
        quality.connect("changed", update_provider_asr_quality)
        quality_row.pack_start(quality, True, True, 0)
        whisper_group.pack_start(quality_row, False, False, 0)

    credentials_group = _provider_settings_group(area, "API-ключ")
    key_actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    set_key = Gtk.Button(label="Задать ключ…")
    set_key.connect("clicked", edit_provider_credential, provider)
    key_actions.pack_start(set_key, True, True, 0)
    open_key = Gtk.Button(label="Открыть файл API key…")
    open_key.connect("clicked", open_credentials_file, provider)
    key_actions.pack_start(open_key, True, True, 0)
    credentials_group.pack_start(key_actions, False, False, 0)
    validate = Gtk.Button(label="Явно проверить ключ")
    validate.connect("clicked", validate_provider_credentials, provider)
    credentials_group.pack_start(validate, False, False, 0)
    credentials_group.pack_start(Gtk.Label(
        label="Ключи хранятся локально; не отправляйте их в чат.", xalign=0),
        False, False, 0)
    if dialog is not None:
        def clear_provider_dialog_labels(*_args):
            provider_dialog_status_labels.pop(provider, None)
            provider_dialog_usage_labels.pop(provider, None)
        dialog.connect("destroy", clear_provider_dialog_labels)
        dialog.show_all()
        dialog.run()
        dialog.destroy()
    else:
        def clear_provider_page_labels(*_args):
            provider_dialog_status_labels.pop(provider, None)
            provider_dialog_usage_labels.pop(provider, None)
        area.connect("destroy", clear_provider_page_labels)
        area.show_all()

provider_dialog_status_labels = {}
provider_dialog_usage_labels = {}

def refresh_multi_provider_status(*_args):
    from multi_provider import credentials_configured
    for provider_name in ("groq", "google", "openrouter"):
        try:
            configured = credentials_configured(provider_name)
        except OSError:
            configured = False
        status_label = ("API key: заполнен, права безопасны" if configured else
                        "API key: не заполнен или права некорректны")
        if provider_name in provider_dialog_status_labels:
            provider_dialog_status_labels[provider_name].set_text(status_label)
        if provider_name in provider_dialog_usage_labels:
            provider_dialog_usage_labels[provider_name].set_text(
                provider_account_usage_text(provider_name))
    config = load_voice_config()
    settings = config.get("multi_provider", {})
    from multi_provider import safe_settings
    normalized_settings = safe_settings(settings)
    master_item = globals().get("all_correctors_menu_item")
    if master_item is not None:
        global _all_correctors_syncing
        _all_correctors_syncing = True
        try:
            master_item.set_active(normalized_settings["correction"]["enabled"])
        finally:
            _all_correctors_syncing = False

refresh_multi_provider_status()

def latest_codex_attempt(metrics_path):
    """Read only safe Codex attempt fields for the tray's diagnostic label."""
    latest = None
    try:
        with open(metrics_path, encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                attempts = record.get("attempts")
                if not isinstance(attempts, list):
                    continue
                correction_attempts = [item for item in attempts
                    if isinstance(item, dict) and item.get("stage") == "correction"]
                codex_attempts = [item for item in correction_attempts
                    if item.get("provider") == "codex"]
                if not codex_attempts:
                    continue
                final = codex_attempts[-1]
                final_position = max(index for index, item in enumerate(correction_attempts)
                                     if item.get("provider") == "codex")
                category = final.get("error_category")
                if not isinstance(category, str) or not re.fullmatch(r"[a-z0-9_]{1,64}", category):
                    category = None
                latest = {
                    "success": final.get("success") is True,
                    "category": category,
                    "attempt_count": len(codex_attempts),
                    "continued": final_position < len(correction_attempts) - 1,
                }
    except OSError:
        pass
    return latest

def format_codex_attempt_status(attempt):
    if not attempt:
        return "Последняя коррекция Codex: безопасных данных пока нет"
    if attempt["success"]:
        return "Последняя коррекция Codex: результат принят проверкой"
    category = attempt.get("category") or "unknown"
    if category.startswith("unsafe_response_"):
        reason = {
            "unsafe_response_word_coverage_order": "проверка сохранения слов",
            "unsafe_response_tail": "проверка конца текста",
            "unsafe_response_length": "проверка полноты текста",
            "unsafe_response_invalid": "проверка формата ответа",
        }.get(category, "проверка безопасности")
        if attempt.get("continued"):
            return f"Codex: ответ отклонён ({reason}); запущен следующий сервис"
        count = attempt.get("attempt_count", 1)
        return f"Codex: ответ отклонён ({reason}) после {count} попыток"
    reason = {
        "cli_failed": "команда Codex завершилась с ошибкой",
        "cli_timeout": "истёк тайм-аут",
        "cli_unavailable": "Codex недоступен",
        "cli_output_invalid": "ответ Codex пустой или некорректный",
        "cancelled_by_user": "запрос отменён",
    }.get(category, "ошибка Codex")
    return f"Codex: {reason}; проверьте следующий сервис в цепочке"

codex_settings_status_label = None
codex_settings_attempt_label = None

def refresh_codex_account_status(*_args):
    """Refresh only safe Codex account and attempt labels in the settings page."""
    try:
        result = subprocess.run(["codex", "login", "status"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=5, check=False)
        status = "Учётная запись Codex активна" if result.returncode == 0 else \
                 "Учётная запись Codex недоступна"
    except (OSError, subprocess.SubprocessError):
        status = "Codex недоступен"
    if codex_settings_status_label is not None:
        codex_settings_status_label.set_text(status)
    if codex_settings_attempt_label is not None:
        codex_settings_attempt_label.set_text(format_codex_attempt_status(
            latest_codex_attempt(METRICS_FILE)))
job_slot_items = []
job_slot_widgets = []
job_slot_batch_ids = [None] * MAX_JOB_SLOTS

def on_job_slot_clicked(idx):
    batch_id = job_slot_batch_ids[idx]
    if batch_id:
        create_batch_progress_window(batch_id)

for _slot_idx in range(MAX_JOB_SLOTS):
    item = Gtk.ImageMenuItem(label="")
    status_image = Gtk.Image()
    item.set_image(status_image)
    item.set_always_show_image(True)
    item.set_sensitive(False)
    item.set_visible(False)
    item.connect("activate", lambda _item, idx=_slot_idx: on_job_slot_clicked(idx))
    menu.append(item)
    job_slot_items.append(item)
    job_slot_widgets.append(status_image)

jobs_separator = Gtk.SeparatorMenuItem()
menu.append(jobs_separator)

idle_item = primary_menu_item("Нет активных задач", "microphone-sensitivity-high")
idle_item.set_sensitive(False)
menu.append(idle_item)

menu.append(Gtk.SeparatorMenuItem())
menu.append(cleanup_tasks_item)
menu.append(recover_raw_asr_item)
menu.append(Gtk.SeparatorMenuItem())

daemon_control_items = []

daemon_start_item = primary_menu_item("Запустить демон", "media-playback-start")
daemon_start_item.connect("activate", start_daemon)
daemon_start_item.set_visible(False)
menu.append(daemon_start_item)
daemon_control_items.append(("start", daemon_start_item))

daemon_stop_item = primary_menu_item("Остановить демон", "process-stop")
daemon_stop_item.connect("activate", stop_daemon)
daemon_stop_item.set_visible(False)
menu.append(daemon_stop_item)
daemon_control_items.append(("stop", daemon_stop_item))

daemon_restart_item = primary_menu_item("Перезапустить демон", "view-refresh")
daemon_restart_item.connect("activate", restart_daemon)
daemon_restart_item.set_visible(False)
menu.append(daemon_restart_item)
daemon_control_items.append(("restart", daemon_restart_item))

models_unload_item = primary_menu_item("Выгрузить все модели", "media-eject")
models_unload_item.set_tooltip_text("Выгрузить локальные Whisper, GigaAM и Qwen из RAM/VRAM")
models_unload_item.set_sensitive(False)
models_unload_item.connect("activate", lambda *_args: start_model_control("unload_models", "all"))
menu.append(models_unload_item)
menu.connect("show", update_model_unload_state)
menu.connect("show", update_corrector_unload_state)
update_model_unload_state()
update_corrector_unload_state()
GLib.timeout_add_seconds(15, update_model_unload_state)
GLib.timeout_add_seconds(15, update_corrector_unload_state)

menu.append(Gtk.SeparatorMenuItem())

quality_config = load_quality_config()
current_strategy = quality_config.get("current_strategy", "normal")
current_model = quality_config.get("current_model", "large-v3")
radio_group = []
model_group = []
model_items = {}
_model_selection_syncing = False
vad_group = []
best_of_group = []
temp_group = []
lang_group = []

def on_quality_preset_toggled(item, strategy_key):
    if item.get_active():
        quality_config = load_quality_config()
        quality_config["current_strategy"] = strategy_key
        save_quality_config(quality_config)
        log_debug(f"Стратегия качества изменена на: {strategy_key}")
        update_custom_label()
        update_advanced_settings_ui(quality_config["strategies"][strategy_key])

def update_custom_label():
    quality_config = load_quality_config()
    if quality_config.get("current_strategy") == "custom":
        custom_item.set_label("⚙️ Пользовательская... ✓ (текущая)")
    else:
        custom_item.set_label("⚙️ Пользовательская...")

def on_best_of_changed(item, best_of_val):
    if item.get_active():
        quality_config = load_quality_config()
        strategy = quality_config["current_strategy"]
        quality_config["strategies"][strategy]["best_of"] = best_of_val
        save_quality_config(quality_config)
        log_debug(f"Best of изменён на: {best_of_val}")

def on_temperature_changed(item, temp_val):
    if item.get_active():
        quality_config = load_quality_config()
        strategy = quality_config["current_strategy"]
        quality_config["strategies"][strategy]["temperature"] = temp_val
        save_quality_config(quality_config)
        log_debug(f"Temperature изменена на: {temp_val}")

def on_language_changed(item, lang_code):
    if item.get_active():
        quality_config = load_quality_config()
        strategy = quality_config["current_strategy"]
        quality_config["strategies"][strategy]["language"] = lang_code
        save_quality_config(quality_config)
        log_debug(f"Language изменён на: {lang_code}")

def get_gpu_memory_info_tray():
    """Get GPU memory info, returns dict or None if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            total_mb = int(parts[0].strip())
            used_mb = int(parts[1].strip())
            return {
                "total_gb": round(total_mb / 1024, 1),
                "free_gb": round((total_mb - used_mb) / 1024, 1),
                "used_gb": round(used_mb / 1024, 1),
                "total_mb": total_mb,
                "used_mb": used_mb
            }
    except Exception:
        pass
    return None

def on_model_selected(item, model_name):
    if item.get_active() and not _model_selection_syncing:
        voice_config = load_voice_config()
        changed = (voice_config.get("asr_backend") != "faster-whisper" or
                   voice_config.get("model") != model_name)
        voice_config["model"] = model_name
        voice_config["asr_backend"] = "faster-whisper"
        _promote_local_model_preference(voice_config, "asr", f"faster-whisper:{model_name}")
        save_voice_config(voice_config)
        update_model_label()
        log_debug(f"Каноническая модель изменена на: {model_name}")
        if changed:
            _prepare_asr_after_settings_save()

GIGAAM_MODEL_CHOICES = {
    "gigaam-v3-e2e-ctc-onnx-gpu": ("GigaAM-v3 e2e_ctc · GPU", "GigaAM-v3 e2e_ctc · ONNX CUDA FP32"),
    "gigaam-v3-ctc-onnx-gpu": ("GigaAM-v3 ctc · GPU", "GigaAM-v3 ctc · ONNX CUDA FP32"),
    "gigaam-v3-e2e-rnnt": ("GigaAM-v3 e2e_rnnt · CPU", "GigaAM-v3 e2e_rnnt · CPU"),
}
FASTER_WHISPER_MODEL_CHOICES = (
    ("large-v3", "Large-v3"),
    ("large-v3-turbo", "Large-v3 Turbo"),
    ("medium", "Medium"),
    ("small", "Small"),
    ("base", "Base"),
    ("tiny", "Tiny"),
)

whisper_model_settings_widgets = []
_whisper_model_settings_syncing = False
local_asr_backend_settings_widgets = []
_local_asr_backend_settings_syncing = False

def _register_whisper_model_settings_widget(widget):
    if widget not in whisper_model_settings_widgets:
        whisper_model_settings_widgets.append(widget)

def _register_local_asr_backend_settings_widget(widget):
    if widget not in local_asr_backend_settings_widgets:
        local_asr_backend_settings_widgets.append(widget)

def _sync_local_asr_backend_settings(backend):
    global _local_asr_backend_settings_syncing
    _local_asr_backend_settings_syncing = True
    try:
        for combo in local_asr_backend_settings_widgets:
            if combo.get_active_id() != backend:
                combo.set_active_id(backend)
    finally:
        _local_asr_backend_settings_syncing = False

def _sync_whisper_model_settings(model):
    global _whisper_model_settings_syncing
    _whisper_model_settings_syncing = True
    try:
        for combo in whisper_model_settings_widgets:
            if combo.get_active_id() != model:
                combo.set_active_id(model)
    finally:
        _whisper_model_settings_syncing = False

def _save_whisper_model_selection(widget):
    global _whisper_model_settings_syncing
    if _whisper_model_settings_syncing:
        return
    selected = widget.get_active_id()
    if selected not in dict(FASTER_WHISPER_MODEL_CHOICES):
        return
    current = load_voice_config()
    changed = current.get("asr_backend") != "faster-whisper" or current.get("model") != selected
    current["model"] = selected
    current["asr_backend"] = "faster-whisper"
    _promote_local_model_preference(current, "asr", f"faster-whisper:{selected}")
    save_voice_config(current)
    _sync_local_asr_backend_settings("faster-whisper")
    _sync_whisper_model_settings(selected)
    update_model_label()
    if changed:
        _prepare_asr_after_settings_save()

def on_gigaam_selected(item, backend_name):
    if item.get_active() and not _model_selection_syncing:
        voice_config = load_voice_config()
        changed = voice_config.get("asr_backend") != backend_name
        voice_config["asr_backend"] = backend_name
        _promote_local_model_preference(voice_config, "asr", backend_name)
        save_voice_config(voice_config)
        _sync_local_asr_backend_settings(backend_name)
        update_model_label()
        log_debug("Модель диктовки изменена на " + GIGAAM_MODEL_CHOICES[backend_name][1])
        if changed:
            _prepare_asr_after_settings_save()

def on_vad_sensitivity_changed(item, threshold_val):
    if item.get_active():
        quality_config = load_quality_config()
        strategy = quality_config["current_strategy"]
        quality_config["strategies"][strategy]["vad_threshold"] = threshold_val
        save_quality_config(quality_config)
        log_debug(f"VAD Sensitivity изменена на: {threshold_val}")

def update_model_label():
    global _model_selection_syncing
    config = load_voice_config()
    current_model = config.get("model", "large-v3")
    selected_backend = config.get("asr_backend", "faster-whisper")
    gigaam_selected = selected_backend in GIGAAM_MODEL_CHOICES
    model_menu_item.set_label(
        "Модель диктовки — " + GIGAAM_MODEL_CHOICES[selected_backend][0]
        if gigaam_selected else f"Модель диктовки — {current_model}")
    if model_items:
        active_key = selected_backend if gigaam_selected else current_model
        _model_selection_syncing = True
        try:
            for key, item in model_items.items():
                item.set_active(key == active_key)
        finally:
            _model_selection_syncing = False

def configure_local_recognition_settings(parent):
    """Add separate local Whisper and GigaAM pages to the settings panel."""
    notebook = parent

    whisper_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
    whisper_page.set_border_width(12)
    whisper_page.pack_start(
        settings_tab_label("Whisper", "transcribe", icon_size=22), False, False, 0)
    whisper_info = Gtk.Label(
        label="Параметры этой вкладки применяются только к FasterWhisper.", xalign=0)
    whisper_info.set_line_wrap(True)
    whisper_page.pack_start(whisper_info, False, False, 0)

    voice = load_voice_config()
    model_descriptions = {
        "large-v3": "Large-v3 — профиль приложения с наивысшим качеством; требует больше памяти.",
        "large-v3-turbo": "Large-v3 Turbo — экспериментальный быстрый вариант; качество для этой установки отдельно не подтверждено.",
        "medium": "Medium — промежуточный по размеру и скорости вариант.",
        "small": "Small — более компактный и быстрый вариант с меньшим запасом качества.",
    }
    whisper_model = Gtk.ComboBoxText()
    whisper_model.set_hexpand(True)
    for key, label in FASTER_WHISPER_MODEL_CHOICES:
        whisper_model.append(key, label)
    initial_model = voice.get("model", "large-v3")
    whisper_model.set_active_id(initial_model if initial_model in model_descriptions else "large-v3")
    _register_whisper_model_settings_widget(whisper_model)
    model_note = Gtk.Label(label="", xalign=0)
    model_note.set_line_wrap(True)

    def update_whisper_model_note(selected):
        current = load_voice_config()
        active_backend = current.get("asr_backend", "faster-whisper")
        suffix = ("Сейчас выбран GigaAM; профиль Whisper сохранён и будет применён "
                  "после выбора FasterWhisper." if active_backend in GIGAAM_MODEL_CHOICES
                  else "Этот профиль применяется при выбранном FasterWhisper.")
        model_note.set_text(model_descriptions[selected] + " " + suffix)

    def save_whisper_model(widget):
        selected = widget.get_active_id()
        if selected not in model_descriptions:
            return
        _save_whisper_model_selection(widget)
        update_whisper_model_note(selected)

    def pack_labeled_row(parent, title, widget):
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        label = Gtk.Label(label=title, xalign=0)
        label.set_size_request(190, -1)
        row.pack_start(label, False, False, 0)
        row.pack_start(widget, True, True, 0)
        parent.pack_start(row, False, False, 0)
        return row

    pack_labeled_row(whisper_page, "Модель Whisper:", whisper_model)
    whisper_page.pack_start(model_note, False, False, 0)
    whisper_model.connect("changed", save_whisper_model)
    update_whisper_model_note(whisper_model.get_active_id() or "large-v3")

    quality = load_quality_config()
    strategy_names = {
        "energy": "Энергосбережение", "normal": "Обычно", "good": "Хорошее",
        "maximum": "Максимум", "custom": "Пользовательская",
    }
    strategy_combo = Gtk.ComboBoxText()
    for key, title in strategy_names.items():
        strategy_combo.append(key, title)
    control_widgets = {}
    loading_controls = {"active": False}

    def current_strategy_params(config):
        key = config.get("current_strategy", "normal")
        return key, config.setdefault("strategies", {}).setdefault(
            key, dict(DEFAULT_QUALITY_CONFIG["strategies"]["normal"]))

    def render_quality_controls():
        config = load_quality_config()
        strategy_key, params = current_strategy_params(config)
        loading_controls["active"] = True
        try:
            strategy_combo.set_active_id(strategy_key if strategy_key in strategy_names else "normal")
            control_widgets["beam_size"].set_value(params.get("beam_size", 1))
            control_widgets["best_of"].set_active_id(str(params.get("best_of", 1)))
            control_widgets["temperature"].set_active_id(str(params.get("temperature", 0.0)))
            control_widgets["language"].set_active_id(str(params.get("language", "Auto")))
            vad = params.get("vad_threshold", 0.5)
            control_widgets["vad_threshold"].set_active_id("off" if vad is None else str(vad))
            control_widgets["fp16"].set_active(bool(params.get("fp16", False)))
            control_widgets["condition_on_previous_text"].set_active(
                load_voice_config().get("condition_on_previous_text", True))
        finally:
            loading_controls["active"] = False

    def save_quality_field(field, value):
        if loading_controls["active"]:
            return
        config = load_quality_config()
        current_key, params = current_strategy_params(config)
        if current_key != "custom":
            config["strategies"]["custom"] = dict(params)
            params = config["strategies"]["custom"]
            config["current_strategy"] = "custom"
        params[field] = value
        save_quality_config(config)
        render_quality_controls()

    def on_strategy_changed(widget):
        selected = widget.get_active_id()
        if loading_controls["active"] or selected not in strategy_names:
            return
        config = load_quality_config()
        config["current_strategy"] = selected
        save_quality_config(config)
        render_quality_controls()

    strategy_combo.connect("changed", on_strategy_changed)
    pack_labeled_row(whisper_page, "Стратегия качества:", strategy_combo)

    # Gtk.SpinButton renders themed stepper icons. Under a full SVG cache GTK
    # aborts while loading image-missing; a numeric scale has no icon lookup.
    beam = integer_range_scale(1, 50, 1)
    beam.set_hexpand(True)
    control_widgets["beam_size"] = beam
    beam.connect("value-changed", lambda widget: save_quality_field("beam_size", int(widget.get_value())))
    pack_labeled_row(whisper_page, "Beam size:", beam)

    for field, title, choices in (
            ("best_of", "Best of:", [("1", "1"), ("3", "3"), ("5", "5")]),
            ("temperature", "Temperature:", [("0.0", "0.0"), ("0.1", "0.1"),
                                                ("0.2", "0.2"), ("0.5", "0.5")]),
            ("language", "Язык:", [("Auto", "Авто"), ("ru", "Русский"), ("en", "English")]),
            ("vad_threshold", "VAD:", [("off", "Выкл."), ("0.3", "Низкий"),
                                         ("0.5", "Средний"), ("0.7", "Высокий")])):
        combo = Gtk.ComboBoxText()
        for value, label in choices:
            combo.append(value, label)
        control_widgets[field] = combo
        def on_choice_changed(widget, field_name=field):
            selected = widget.get_active_id()
            if selected is None:
                return
            value = (None if field_name == "vad_threshold" and selected == "off" else
                     float(selected) if field_name in {"temperature", "vad_threshold"} else
                     int(selected) if field_name == "best_of" else selected)
            save_quality_field(field_name, value)
        combo.connect("changed", on_choice_changed)
        pack_labeled_row(whisper_page, title, combo)

    fp16 = Gtk.CheckButton(label="Использовать FP16, если доступно")
    control_widgets["fp16"] = fp16
    fp16.connect("toggled", lambda widget: save_quality_field("fp16", widget.get_active()))
    whisper_page.pack_start(fp16, False, False, 0)
    previous_text = Gtk.CheckButton(label="Учитывать предыдущий текст Whisper")
    control_widgets["condition_on_previous_text"] = previous_text
    def save_previous_text(widget):
        if loading_controls["active"]:
            return
        current = load_voice_config()
        current["condition_on_previous_text"] = widget.get_active()
        save_voice_config(current)
    previous_text.connect("toggled", save_previous_text)
    whisper_page.pack_start(previous_text, False, False, 0)
    whisper_unload_settings_button = Gtk.Button(label="Выгрузить Whisper")
    globals()["whisper_unload_settings_button"] = whisper_unload_settings_button
    whisper_unload_settings_button.connect(
        "clicked", lambda *_args: unload_local_asr(expected_backend="faster-whisper"))
    whisper_page.pack_start(whisper_unload_settings_button, False, False, 0)
    whisper_scroll = Gtk.ScrolledWindow()
    whisper_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    whisper_scroll.add(whisper_page)
    notebook.append_page(whisper_scroll,
                         settings_tab_label("Whisper", "transcribe", icon_size=22))

    gigaam_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
    gigaam_page.set_border_width(12)
    gigaam_page.pack_start(
        settings_tab_label("GigaAM", "gigaam", icon_size=22), False, False, 0)
    gigaam_info = Gtk.Label(
        label="GigaAM не использует Whisper beam size, temperature, язык или VAD. "
              "Здесь выбираются вариант модели и фактическое устройство исполнения.", xalign=0)
    gigaam_info.set_line_wrap(True)
    gigaam_page.pack_start(gigaam_info, False, False, 0)
    gigaam_model = Gtk.ComboBoxText()
    for key, (label, _detail) in GIGAAM_MODEL_CHOICES.items():
        gigaam_model.append(key, label)
    gigaam_note = Gtk.Label(label="", xalign=0)
    gigaam_note.set_line_wrap(True)
    asr_residency_status_label = Gtk.Label(
        label="Состояние локальной модели ещё не проверено", xalign=0)
    asr_residency_status_label.set_line_wrap(True)
    globals()["asr_residency_status_label"] = asr_residency_status_label
    gigaam_unload_settings_button = Gtk.Button(label="Выгрузить GigaAM")
    globals()["gigaam_unload_settings_button"] = gigaam_unload_settings_button
    gigaam_unload_settings_button.connect(
        "clicked", lambda *_args: unload_local_asr(expected_backend="gigaam"))
    update_model_unload_state()
    gigaam_descriptions = {
        "gigaam-v3-e2e-rnnt": "e2e_rnnt: PyTorch worker на CPU; end-to-end RNNT распознавание.",
        "gigaam-v3-e2e-ctc-onnx-gpu": "e2e_ctc: ONNX Runtime CUDA FP32; декодирование с пунктуацией и нормализацией.",
        "gigaam-v3-ctc-onnx-gpu": "ctc: ONNX Runtime CUDA FP32; базовый CTC результат без e2e-декодирования.",
    }
    def save_gigaam_model(widget):
        selected = widget.get_active_id()
        if selected not in GIGAAM_MODEL_CHOICES:
            return
        current = load_voice_config()
        changed = current.get("asr_backend") != selected
        current["asr_backend"] = selected
        _promote_local_model_preference(current, "asr", selected)
        save_voice_config(current)
        _sync_local_asr_backend_settings(selected)
        gigaam_note.set_text(gigaam_descriptions[selected])
        update_model_label()
        if changed:
            _prepare_asr_after_settings_save()
    gigaam_model.connect("changed", save_gigaam_model)
    selected_backend = load_voice_config().get("asr_backend", "faster-whisper")
    if selected_backend in GIGAAM_MODEL_CHOICES:
        gigaam_model.set_active_id(selected_backend)
        gigaam_note.set_text(gigaam_descriptions[selected_backend])
    else:
        gigaam_model.set_active(-1)
        gigaam_note.set_text("Сейчас выбран FasterWhisper. Выбор одного из вариантов выше переключит ASR на GigaAM.")
    pack_labeled_row(gigaam_page, "Вариант GigaAM:", gigaam_model)
    gigaam_page.pack_start(gigaam_note, False, False, 0)
    chunk_info = Gtk.Label(
        label="Длинные записи делятся на окна до 24 секунд с overlap 1 секунда. "
              "Эти параметры фиксированы backend-ом; отдельные настройки не поддерживаются.", xalign=0)
    chunk_info.set_line_wrap(True)
    gigaam_page.pack_start(chunk_info, False, False, 0)
    gigaam_page.pack_start(asr_residency_status_label, False, False, 0)
    gigaam_page.pack_start(gigaam_unload_settings_button, False, False, 0)
    gigaam_scroll = Gtk.ScrolledWindow()
    gigaam_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    gigaam_scroll.add(gigaam_page)
    notebook.append_page(gigaam_scroll,
                         settings_tab_label("GigaAM", "gigaam", icon_size=22))

    render_quality_controls()
    def refresh_model_settings_page(_notebook, _page, page_number):
        if page_number == notebook.page_num(whisper_scroll):
            current_model = load_voice_config().get("model", "large-v3")
            update_whisper_model_note(
                current_model if current_model in model_descriptions else "large-v3")
        if page_number != notebook.page_num(gigaam_scroll):
            return
        selected = load_voice_config().get("asr_backend", "faster-whisper")
        if selected in GIGAAM_MODEL_CHOICES:
            gigaam_model.set_active_id(selected)
            gigaam_note.set_text(gigaam_descriptions[selected])
        else:
            gigaam_model.set_active(-1)
            gigaam_note.set_text(
                "Сейчас выбран FasterWhisper. Выберите вариант здесь, чтобы переключить ASR на GigaAM.")
    notebook.connect("switch-page", refresh_model_settings_page)
    return {"Локальный Whisper": whisper_scroll,
            "Локальный GigaAM": gigaam_scroll}

def update_corrector_label():
    value = load_voice_config()
    corrector = value.get("corrector", {})
    model = corrector.get("model", "Qwen [не выбран]")
    state = "вкл" if corrector.get("enabled", False) else "выкл"
    short_model = "4B" if ":4b-" in model else "2B" if ":2b-" in model else model
    set_primary_menu_label(corrector_menu_item,
                           f"Корректор: {state} · {short_model}")
    if 'corrector_enabled_item' in globals():
        set_check_menu_item_label(corrector_enabled_item,
                                  "Включить локальный корректор (приоритетный)")

def update_corrector_vram_item():
    """Показать текущее VRAM непосредственно рядом с выбором Qwen-ускорения."""
    if 'corrector_vram_item' not in globals():
        return False
    gpu = get_gpu_memory_info_tray()
    if gpu:
        corrector_vram_item.set_label(
            f"🎮 VRAM сейчас: {gpu['used_mb']} / {gpu['total_mb']} MiB "
            f"(свободно {gpu['free_gb']} GiB)")
    else:
        corrector_vram_item.set_label("🎮 VRAM: данные NVIDIA недоступны")
    return False

def on_corrector_accelerator_selected(item, num_gpu):
    if item.get_active():
        value = load_voice_config()
        corrector = value.get("corrector", {})
        corrector["num_gpu"] = num_gpu
        value["corrector"] = corrector
        save_voice_config(value)
        update_corrector_label()
        log_debug(f"Корректор переведён в {'GPU (' + str(num_gpu) + ' слоёв)' if num_gpu else 'CPU'}")

def on_corrector_performance_selected(item, key, selected):
    if item.get_active():
        value = load_voice_config()
        corrector = value.get("corrector", {})
        corrector[key] = selected
        value["corrector"] = corrector
        save_voice_config(value)
        update_corrector_label()
        log_debug(f"Производительность Qwen: {key}={selected}")

def on_corrector_enabled_toggled(item):
    value = load_voice_config()
    corrector = value.get("corrector", {})
    corrector["enabled"] = item.get_active()
    value["corrector"] = corrector
    save_voice_config(value)
    update_corrector_label()
    log_debug(f"Корректор {'включён' if item.get_active() else 'выключен'}")
    if not item.get_active():
        start_model_control("unload_models", scope="corrector", timeout=60)

def on_audio_ducking_toggled(item):
    value = load_voice_config()
    ducking = value.get("audio_ducking", {})
    if not isinstance(ducking, dict):
        ducking = {}
    ducking.update({"enabled": item.get_active(), "target_percent": 20, "fade_ms": 300})
    value["audio_ducking"] = ducking
    save_voice_config(value)
    log_debug(f"Плавное приглушение звука {'включено' if item.get_active() else 'выключено'}")

def poll_audio_restore():
    global _audio_restore
    if not _audio_restore:
        return False
    process = _audio_restore
    if process.poll() is None:
        return True
    output, error = process.communicate()
    _audio_restore = None
    result = output.strip()
    if process.returncode == 0:
        if result.startswith("restored-saved:"):
            message = "Возвращена громкость, сохранённая до диктовки."
        elif result.startswith("restored-100:"):
            count = result.partition(":")[2]
            message = ("Активные потоки возвращены к 100%." if count != "0"
                       else "Приглушённых активных потоков не найдено.")
        else:
            message = "Восстановление громкости завершено."
        log_debug(f"🔊 {message}")
        notify_send("Whisper Dictate", message)
    else:
        details = error.strip() or "неизвестная ошибка"
        log_debug(f"✗ Ошибка восстановления громкости: {details}")
        notify_send("Whisper Dictate", "Не удалось восстановить звук — см. логи.")
    return False

def on_audio_restore_clicked(_):
    global _audio_restore
    if _audio_restore:
        notify_send("Whisper Dictate", "Восстановление звука уже выполняется.")
        return
    try:
        ducking = load_voice_config().get("audio_ducking", {})
        fade_ms = ducking.get("fade_ms", 300) if isinstance(ducking, dict) else 300
        if not isinstance(fade_ms, int) or not 100 <= fade_ms <= 2000:
            fade_ms = 300
        _audio_restore = subprocess.Popen(
            [os.path.expanduser("~/.local/bin/whisper/whisper-audio-duck.py"), "recover", "100", str(fade_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        notify_send("Whisper Dictate", "Восстанавливаю громкость…")
        GLib.timeout_add(150, poll_audio_restore)
    except Exception as e:
        log_debug(f"✗ Не удалось запустить восстановление громкости: {e}")
        notify_send("Whisper Dictate", "Не удалось запустить восстановление звука.")

_corrector_model_syncing = False

def _sync_corrector_model_radio(model_name):
    global _corrector_model_syncing
    if model_name is None:
        return
    _corrector_model_syncing = True
    try:
        for item, (key, _label) in zip(corrector_model_group, corrector_models):
            item.set_active(key == model_name)
    finally:
        _corrector_model_syncing = False

def on_corrector_model_selected(item, model_name):
    if not item.get_active() or _corrector_model_syncing:
        return
    global _corrector_pull
    if _corrector_pull:
        return
    present = subprocess.run(["ollama", "show", model_name], capture_output=True, text=True).returncode == 0
    if present:
        set_corrector_model(model_name)
        return
    try:
        process = subprocess.Popen(["ollama", "pull", model_name], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True)
        _corrector_pull = {"model": model_name, "process": process}
        for menu_item in corrector_model_group:
            menu_item.set_sensitive(False)
        notify_send("Whisper Dictate", f"⬇️ Скачиваю {model_name}; переключу после завершения")
        GLib.timeout_add(500, poll_corrector_pull)
    except Exception as e:
        notify_send("Ошибка", f"Не удалось начать загрузку {model_name}: {e}")

def set_corrector_model(model_name):
    value = load_voice_config()
    corrector = value.get("corrector", {})
    changed = corrector.get("model") != model_name
    corrector["model"] = model_name
    value["corrector"] = corrector
    _promote_local_model_preference(value, "correction", model_name)
    save_voice_config(value)
    update_corrector_label()
    log_debug(f"Модель корректора изменена на: {model_name}")
    if changed:
        start_model_control("unload_models", scope="corrector", timeout=60)

def poll_corrector_pull():
    global _corrector_pull
    job = _corrector_pull
    if not job:
        return False
    result = job["process"].poll()
    if result is None:
        return True
    for menu_item in corrector_model_group:
        menu_item.set_sensitive(True)
    model_name = job["model"]
    _corrector_pull = None
    if result == 0:
        set_corrector_model(model_name)
        notify_send("Whisper Dictate", f"✓ {model_name} скачан и выбран")
    else:
        log_debug(f"Не удалось скачать модель корректора: {model_name}")
        notify_send("Whisper Dictate", f"Не удалось скачать {model_name}; оставлена текущая модель")
    return False

def on_autoreplacements_clicked(_):
    os.makedirs(os.path.dirname(AUTOREPLACEMENTS_FILE), mode=0o700, exist_ok=True)
    if not os.path.exists(AUTOREPLACEMENTS_FILE):
        template = "# Автозамены для Whisper Dictate: исходник => замена\n# По одной на строку; # — комментарий.\nQN => Qwen\nТера => Terra\n"
        with open(AUTOREPLACEMENTS_FILE, "w", encoding="utf-8") as stream:
            stream.write(template)
        os.chmod(AUTOREPLACEMENTS_FILE, 0o600)
    try:
        with open(AUTOREPLACEMENTS_FILE, encoding="utf-8") as stream:
            raw_lines = stream.readlines()
    except OSError:
        raw_lines = []
    comments = [line.rstrip("\n") for line in raw_lines
                if line.lstrip().startswith("#")]
    replacements = []
    for line in raw_lines:
        if "=>" not in line or line.lstrip().startswith("#"):
            continue
        source, target = line.split("=>", 1)
        replacements.append([source.strip(), target.strip()])

    dialog = Gtk.Dialog(title="Автозамены Whisper Dictate", transient_for=None, flags=0)
    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                       Gtk.STOCK_OK, Gtk.ResponseType.OK)
    dialog.set_default_size(620, 420)
    area = dialog.get_content_area()
    area.set_border_width(10)
    area.set_spacing(8)
    area.pack_start(Gtk.Label(label="Исходное выражение заменяется указанным текстом.", xalign=0),
                    False, False, 0)
    store = Gtk.ListStore(str, str)
    for source, target in replacements:
        store.append((source, target))
    tree = Gtk.TreeView(model=store)
    for column_index, title in enumerate(("Найти", "Заменить на")):
        renderer = Gtk.CellRendererText()
        renderer.set_property("editable", True)
        renderer.connect("edited", lambda _renderer, path, value, idx=column_index:
                         store.set_value(store.get_iter(path), idx, value))
        column = Gtk.TreeViewColumn(title, renderer, text=column_index)
        column.set_resizable(True)
        column.set_expand(True)
        tree.append_column(column)
    scroll = Gtk.ScrolledWindow()
    scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
    scroll.set_min_content_height(240)
    scroll.add(tree)
    area.pack_start(scroll, True, True, 0)
    buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    add_button = Gtk.Button(label="Добавить")
    remove_button = Gtk.Button(label="Удалить выбранную")
    add_button.connect("clicked", lambda *_: store.append(("", "")))
    def remove_selected(_button):
        selection = tree.get_selection()
        model, row_iter = selection.get_selected()
        if row_iter is not None:
            model.remove(row_iter)
    remove_button.connect("clicked", remove_selected)
    buttons.pack_start(add_button, False, False, 0)
    buttons.pack_start(remove_button, False, False, 0)
    area.pack_start(buttons, False, False, 0)
    dialog.show_all()
    if dialog.run() == Gtk.ResponseType.OK:
        rows = []
        for source, target in store:
            source, target = source.strip(), target.strip()
            if source and target:
                rows.append(f"{source} => {target}")
        content = "\n".join(comments + rows) + "\n"
        temporary = AUTOREPLACEMENTS_FILE + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                stream.write(content)
            os.chmod(temporary, 0o600)
            os.replace(temporary, AUTOREPLACEMENTS_FILE)
        except OSError:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            notify_send("Whisper Dictate", "Не удалось сохранить автозамены")
    dialog.destroy()

def on_corrector_prompt_clicked(_):
    value = load_voice_config()
    corrector = value.get("corrector", {})
    from multi_provider import REMOTE_CORRECTION_INSTRUCTION
    dialog = Gtk.Dialog(title="Дополнительные инструкции корректора", transient_for=None, flags=0)
    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK)
    dialog.set_default_size(640, 420)
    text_view = Gtk.TextView()
    text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
    intro = Gtk.Label(label=("Встроенный prompt Codex/OpenRouter и других общих API:\n" +
                             REMOTE_CORRECTION_INSTRUCTION + "\n\n" +
                             "Поле ниже хранит дополнительные указания только локальному корректору. "
                             "Whisper/ASR их не получает и использует отдельный начальный prompt. "
                             "По умолчанию поле пустое; встроенные правила остаются активны."),
                      xalign=0)
    intro.set_line_wrap(True)
    dialog.get_content_area().pack_start(intro, False, False, 8)
    text_view.get_buffer().set_text(corrector.get("prompt", ""))
    scroll = Gtk.ScrolledWindow()
    scroll.add(text_view)
    dialog.get_content_area().pack_start(scroll, True, True, 8)
    dialog.show_all()
    if dialog.run() == Gtk.ResponseType.OK:
        buffer = text_view.get_buffer()
        prompt = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True).strip()
        if len(prompt) <= 1500:
            corrector["prompt"] = prompt
            value["corrector"] = corrector
            save_voice_config(value)
            log_debug("Prompt корректора обновлён")
        else:
            notify_send("Whisper Dictate", "Prompt не сохранён: максимум 1500 символов")
    dialog.destroy()

def update_advanced_settings_ui(strategy_params):
    best_of_val = strategy_params.get("best_of", 1)
    for i, item in enumerate(best_of_group):
        item.set_active((best_of_val == [1, 3, 5][i]))

    temp_val = strategy_params.get("temperature", 0.0)
    for i, item in enumerate(temp_group):
        item.set_active(abs(temp_val - [0.0, 0.1, 0.2, 0.5][i]) < 0.01)

    lang_val = strategy_params.get("language", "Auto")
    lang_codes = ["Auto", "ru", "en"]
    for i, item in enumerate(lang_group):
        item.set_active(lang_val == lang_codes[i])

    vad_val = strategy_params.get("vad_threshold", 0.5)
    vad_thresholds = [None, 0.3, 0.5, 0.7]
    for i, item in enumerate(vad_group):
        if (vad_val is None and vad_thresholds[i] is None) or (vad_val == vad_thresholds[i]):
            item.set_active(True)

def on_custom_clicked(_):
    quality_config = load_quality_config()
    custom_params = quality_config["strategies"]["custom"]

    dialog = Gtk.Dialog(title="Пользовательская стратегия", transient_for=None, flags=0)
    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                       Gtk.STOCK_OK, Gtk.ResponseType.OK)
    dialog.set_default_size(300, 200)

    box = dialog.get_content_area()

    beam_label = Gtk.Label(label="Beam size (1-50):")
    beam_label.set_halign(Gtk.Align.START)
    box.pack_start(beam_label, False, False, 5)

    beam_spin = integer_range_scale(1, 50, custom_params.get("beam_size", 5))
    box.pack_start(beam_spin, False, False, 5)

    fp16_check = Gtk.CheckButton(label="fp16 (экономит 40-50% VRAM)")
    fp16_check.set_active(custom_params.get("fp16", False))
    box.pack_start(fp16_check, False, False, 5)

    box.show_all()

    response = dialog.run()
    if response == Gtk.ResponseType.OK:
        quality_config["strategies"]["custom"]["beam_size"] = int(beam_spin.get_value())
        quality_config["strategies"]["custom"]["fp16"] = fp16_check.get_active()
        quality_config["current_strategy"] = "custom"
        save_quality_config(quality_config)

        for item in radio_group:
            item.set_active(False)

        update_custom_label()
        log_debug(f"Custom стратегия: beam={int(beam_spin.get_value())}, fp16={fp16_check.get_active()}")

    dialog.destroy()

quality_submenu = Gtk.Menu()
strategies = [
    ("energy", "⚡ Энергосбережение (beam=1, fp16=вкл)"),
    ("normal", "🔄 Обычно (beam=5, fp16=вкл)"),
    ("good", "🎯 Хорошее (beam=10, fp16=вкл)"),
    ("maximum", "💎 Максимум (beam=20, fp16=выкл)"),
]

for i, (key, label) in enumerate(strategies):
    if i == 0:
        item = Gtk.RadioMenuItem(label=label)
        radio_group.append(item)
    else:
        item = Gtk.RadioMenuItem.new_from_widget(radio_group[0])
        item.set_label(label)
        radio_group.append(item)

    if current_strategy == key:
        item.set_active(True)

    item.connect("toggled", on_quality_preset_toggled, key)
    quality_submenu.append(item)

quality_submenu.append(Gtk.SeparatorMenuItem())

custom_item = icon_menu_item("Пользовательская…", menu_icon_path("settings"))
custom_item.connect("activate", on_custom_clicked)
quality_submenu.append(custom_item)

update_custom_label()

quality_submenu.show_all()

model_submenu = Gtk.Menu()

for backend_name, (label, _display_name) in GIGAAM_MODEL_CHOICES.items():
    gigaam_item = Gtk.RadioMenuItem(label=label)
    if model_group:
        gigaam_item.join_group(model_group[0])
    model_group.append(gigaam_item)
    model_items[backend_name] = gigaam_item
    gigaam_item.connect("toggled", on_gigaam_selected, backend_name)
    model_submenu.append(gigaam_item)
model_submenu.append(Gtk.SeparatorMenuItem())

model_options_list = [
    ("large-v3", "FasterWhisper · Large-v3"),
    ("large-v3-turbo", "FasterWhisper · Large-v3 Turbo"),
    ("medium", "FasterWhisper · Medium"),
    ("small", "FasterWhisper · Small"),
    ("base", "FasterWhisper · Base"),
    ("tiny", "FasterWhisper · Tiny"),
]

for i, (model_key, label) in enumerate(model_options_list):
    item = Gtk.RadioMenuItem(label=label) if not model_group else Gtk.RadioMenuItem.new_from_widget(model_group[0])
    item.set_label(label)
    model_group.append(item)
    model_items[model_key] = item
    item.connect("toggled", on_model_selected, model_key)
    model_submenu.append(item)

model_submenu.show_all()

model_menu_item = icon_menu_item("Модель диктовки", menu_icon_path("microphone"))
model_menu_item.set_submenu(model_submenu)
whisper_settings_submenu.append(model_menu_item)
recognition_settings_item = icon_menu_item("Настройки локального распознавания…", menu_icon_path("microphone"))
recognition_settings_item.connect("activate", configure_local_recognition_settings)
whisper_settings_submenu.append(recognition_settings_item)

def on_local_condition_on_previous_text_toggled(item):
    value = load_voice_config()
    value["condition_on_previous_text"] = item.get_active()
    save_voice_config(value)
    log_debug("Локальный Whisper condition_on_previous_text=" + str(item.get_active()).lower())

whisper_settings_submenu.append(Gtk.SeparatorMenuItem())
update_model_label()

voice_config = load_voice_config()
corrector_config = voice_config.get("corrector", {})
corrector_submenu = Gtk.Menu()
corrector_enabled_item = Gtk.CheckMenuItem(label="Включить локальный корректор (приоритетный)")
corrector_enabled_item.set_active(corrector_config.get("enabled", False))
corrector_enabled_item.connect("toggled", on_corrector_enabled_toggled)
corrector_submenu.append(corrector_enabled_item)
from multi_provider import safe_settings as safe_provider_settings_for_ui
all_correctors_menu_item = Gtk.CheckMenuItem(label="Использовать AI-корректоры")
iconize_check_menu_item(all_correctors_menu_item, menu_icon_path("corrector"))
all_correctors_menu_item.set_active(safe_provider_settings_for_ui(
    load_voice_config().get("multi_provider"))["correction"]["enabled"])
all_correctors_menu_item.connect("toggled", _on_all_correctors_toggled)
corrector_submenu.append(all_correctors_menu_item)
corrector_submenu.append(Gtk.SeparatorMenuItem())
corrector_vram_item = icon_menu_item("VRAM: обновление…", menu_icon_path("cuda"))
corrector_vram_item.set_sensitive(False)
corrector_submenu.append(corrector_vram_item)
corrector_refresh_vram_item = icon_menu_item("Обновить VRAM", menu_icon_path("refresh"))
corrector_refresh_vram_item.connect("activate", lambda _: update_corrector_vram_item())
corrector_submenu.append(corrector_refresh_vram_item)
corrector_submenu.append(Gtk.SeparatorMenuItem())
performance_submenu = Gtk.Menu()
def add_performance_group(title, key, values):
    group = []
    section = Gtk.MenuItem(label=title)
    section.set_sensitive(False)
    performance_submenu.append(section)
    current = corrector_config.get(key)
    for selected, label in values:
        item = Gtk.RadioMenuItem(label=label) if not group else Gtk.RadioMenuItem.new_from_widget(group[0])
        item.set_label(label)
        group.append(item)
        item.set_active(current == selected)
        item.connect("toggled", on_corrector_performance_selected, key, selected)
        performance_submenu.append(item)
    performance_submenu.append(Gtk.SeparatorMenuItem())

add_performance_group("GPU-offload", "num_gpu", [(0, "🖥️ CPU — num_gpu=0"), (4, "🎮 GPU — 4 слоя"), (8, "🎮 GPU — 8 слоёв (по умолчанию)"), (12, "⚠️ GPU — 12 слоёв (эксперимент, риск VRAM)")])
add_performance_group("Контекст", "num_ctx", [(1024, "1024"), (2048, "2048"), (4096, "4096"), (8192, "8192 — для длинной диктовки (больше RAM/VRAM)")])
add_performance_group("CPU threads", "num_thread", [(2, "2"), (4, "4 (по умолчанию)"), (8, "8")])
add_performance_group("Timeout", "timeout_seconds", [(5, "5 секунд"), (10, "10 секунд"), (15, "15 секунд"), (30, "30 секунд"), (60, "60 секунд"), (120, "2 минуты"), (300, "5 минут — для Large + CPU")])
add_performance_group("Keep alive", "keep_alive", [(0, "Выгружать сразу после запроса"), ("30s", "30 секунд"), ("5m", "5 минут (по умолчанию)")])
add_performance_group("Temperature", "temperature", [(0, "0 (по умолчанию, рекомендовано)"), (0.1, "0.1 (эксперимент: может ухудшить точность)"), (0.2, "0.2 (эксперимент: может ухудшить точность)")])
performance_submenu.show_all()
performance_item = icon_menu_item("Производительность Qwen", menu_icon_path("performance"))
performance_item.set_submenu(performance_submenu)
corrector_submenu.append(performance_item)
corrector_submenu.append(Gtk.SeparatorMenuItem())
corrector_model_group = []
corrector_models = [
    ("qwen3.5:4b-q4_K_M", "⚡ Qwen3.5 4B (рекомендован)"),
    ("qwen3.5:2b-q4_K_M", "🏎️ Qwen3.5 2B (скачать при выборе)"),
]
for model_key, label in corrector_models:
    if not corrector_model_group:
        item = Gtk.RadioMenuItem(label=label)
    else:
        item = Gtk.RadioMenuItem.new_from_widget(corrector_model_group[0])
        item.set_label(label)
    corrector_model_group.append(item)
    if corrector_config.get("model") == model_key:
        item.set_active(True)
    item.connect("toggled", on_corrector_model_selected, model_key)
    corrector_submenu.append(item)
corrector_prompt_item = icon_menu_item("Изменить prompt корректора…", menu_icon_path("corrector"))
corrector_prompt_item.connect("activate", on_corrector_prompt_clicked)
autoreplacements_item = icon_menu_item("Автозамены…", menu_icon_path("autoreplace"))
autoreplacements_item.connect("activate", on_autoreplacements_clicked)
corrector_submenu.append(corrector_prompt_item)
corrector_submenu.append(autoreplacements_item)
corrector_submenu.show_all()
corrector_menu_item = primary_menu_item("Корректор локальный", "accessories-text-editor-symbolic")
corrector_menu_item.set_submenu(corrector_submenu)
update_corrector_label()

advanced_menu = Gtk.Menu()

for best_of_val in [1, 3, 5]:
    label = f"Best of: {best_of_val}"
    if best_of_val == 1:
        item = Gtk.RadioMenuItem(label=label)
        best_of_group.append(item)
    else:
        item = Gtk.RadioMenuItem.new_from_widget(best_of_group[0])
        item.set_label(label)
        best_of_group.append(item)

    current_strategy_params = quality_config["strategies"][current_strategy]
    if current_strategy_params["best_of"] == best_of_val:
        item.set_active(True)

    item.connect("toggled", on_best_of_changed, best_of_val)
    advanced_menu.append(item)

advanced_menu.append(Gtk.SeparatorMenuItem())

for temp_val in [0.0, 0.1, 0.2, 0.5]:
    label = f"Точность: {temp_val}" if temp_val < 0.25 else f"Творчество: {temp_val}"
    if temp_val == 0.0:
        item = Gtk.RadioMenuItem(label=label)
        temp_group.append(item)
    else:
        item = Gtk.RadioMenuItem.new_from_widget(temp_group[0])
        item.set_label(label)
        temp_group.append(item)

    current_strategy_params = quality_config["strategies"][current_strategy]
    if abs(current_strategy_params["temperature"] - temp_val) < 0.01:
        item.set_active(True)

    item.connect("toggled", on_temperature_changed, temp_val)
    advanced_menu.append(item)

advanced_menu.append(Gtk.SeparatorMenuItem())

lang_options = [("Auto", "Auto"), ("Русский", "ru"), ("English", "en")]
for lang_label, lang_code in lang_options:
    if lang_label == "Auto":
        item = Gtk.RadioMenuItem(label=lang_label)
        lang_group.append(item)
    else:
        item = Gtk.RadioMenuItem.new_from_widget(lang_group[0])
        item.set_label(lang_label)
        lang_group.append(item)

    current_strategy_params = quality_config["strategies"][current_strategy]
    if current_strategy_params["language"] == lang_code:
        item.set_active(True)

    item.connect("toggled", on_language_changed, lang_code)
    advanced_menu.append(item)

advanced_menu.append(Gtk.SeparatorMenuItem())

vad_options = [("Off", None), ("Low", 0.3), ("Medium", 0.5), ("High", 0.7)]
for vad_label, vad_threshold in vad_options:
    if vad_label == "Off":
        item = Gtk.RadioMenuItem(label=f"⚙️ VAD Sensitivity — {vad_label}")
        vad_group.append(item)
    else:
        item = Gtk.RadioMenuItem.new_from_widget(vad_group[0])
        item.set_label(f"⚙️ VAD Sensitivity — {vad_label}")
        vad_group.append(item)

    current_strategy_params = quality_config["strategies"][current_strategy]
    current_vad = current_strategy_params.get("vad_threshold", 0.5)
    if (vad_threshold is None and current_vad is None) or (vad_threshold == current_vad):
        item.set_active(True)

    item.connect("toggled", on_vad_sensitivity_changed, vad_threshold)
    advanced_menu.append(item)

advanced_menu.show_all()

model_settings_item = primary_menu_item("Настройки", "model_services")
model_settings_item.connect("activate", lambda *_args: open_settings_panel())
menu.append(model_settings_item)

def open_user_guide(_item):
    """Open the installed guide with a text editor, then xdg-open as fallback."""
    guide_path = os.path.expanduser("~/.local/share/doc/whisper-dictate/guide.md")
    if not os.path.isfile(guide_path):
        log_debug(f"✗ Руководство не найдено: {guide_path}")
        return
    for editor in ("kate", "gedit", "xed", "mousepad", "pluma"):
        try:
            subprocess.Popen([editor, guide_path])
            return
        except FileNotFoundError:
            continue
    try:
        subprocess.Popen(["xdg-open", guide_path])
    except Exception as exc:
        log_debug(f"✗ Не удалось открыть руководство: {exc}")

def open_changelog(_item):
    """Open the installed application changelog."""
    changelog_path = os.path.expanduser("~/.local/share/doc/whisper-dictate/CHANGELOG.md")
    if not os.path.isfile(changelog_path):
        log_debug(f"✗ История изменений не найдена: {changelog_path}")
        return
    for editor in ("kate", "gedit", "xed", "mousepad", "pluma"):
        try:
            subprocess.Popen([editor, changelog_path])
            return
        except FileNotFoundError:
            continue
    try:
        subprocess.Popen(["xdg-open", changelog_path])
    except Exception as exc:
        log_debug(f"✗ Не удалось открыть историю изменений: {exc}")

github_url = "https://github.com/maxsannikov/whisper-dictate"
telegram_url = "https://t.me/maximsannikov"

def show_about_dialog(_item):
    dialog = Gtk.AboutDialog()
    dialog.set_title("О приложении")
    dialog.set_program_name("Whisper Dictate")
    try:
        logo = GdkPixbuf.Pixbuf.new_from_file_at_scale(
            menu_icon_path("microphone"), 128, 128, True)
        dialog.set_logo(logo)
    except (GLib.Error, OSError) as exc:
        log_debug(f"Не удалось загрузить иконку Whisper Dictate: {exc}")
    dialog.set_comments("Локальная диктовка и транскрибация.")
    dialog.set_authors(["Максим Санников"])
    dialog.set_website(github_url)
    dialog.set_website_label("GitHub проекта")
    developer = Gtk.Label(label="Разработчик: Максим Санников")
    developer.set_halign(Gtk.Align.CENTER)
    telegram_link = Gtk.LinkButton.new_with_label(
        telegram_url, "Написать в Telegram")
    telegram_link.set_halign(Gtk.Align.CENTER)
    content = dialog.get_content_area()
    content.pack_start(developer, False, False, 6)
    content.pack_start(telegram_link, False, False, 0)
    navigation = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    help_button = Gtk.Button(label="Открыть справку…")
    diagnostics_button = Gtk.Button(label="Открыть диагностику…")
    def open_about_destination(_button, destination):
        dialog.response(Gtk.ResponseType.CLOSE)
        def open_destination_after_about_closes():
            open_settings_panel(destination)
            return GLib.SOURCE_REMOVE
        GLib.idle_add(open_destination_after_about_closes)
    help_button.connect("clicked", open_about_destination, "Руководство")
    diagnostics_button.connect("clicked", open_about_destination, "Статистика")
    navigation.pack_start(help_button, True, True, 0)
    navigation.pack_start(diagnostics_button, True, True, 0)
    content.pack_start(navigation, False, False, 8)
    dialog.show_all()
    dialog.run()
    dialog.destroy()

about_item = primary_menu_item("О приложении", "help-browser")
about_item.set_tooltip_text("Информация о программе, разработчике и ссылки проекта")
about_item.connect("activate", show_about_dialog)
menu.append(about_item)

menu.append(Gtk.SeparatorMenuItem())
quit_item = Gtk.MenuItem(label="Выход")
quit_item.connect("activate", lambda _: Gtk.main_quit())
menu.append(quit_item)

menu.show_all()
normalize_menu_layout(menu)
for item in job_slot_items:
    item.set_visible(False)
jobs_separator.set_visible(False)

indicator.set_menu(menu)

_last_signature = None
def poll_jobs():
    global _last_signature
    global _manual_correction_daemon_seen
    jobs = []
    try:
        for path in sorted(glob.glob(os.path.join(JOBS_DIR, "*.json"))):
            try:
                with open(path, encoding="utf-8") as f:
                    job = json.load(f)
                    if isinstance(job, dict):
                        job["_job_key"] = os.path.basename(path)
                        try:
                            # A writer can rewrite a job without changing the
                            # small set of display fields below (for example,
                            # recording -> processing within one second).
                            # Include file metadata so the menu still refreshes.
                            job["_status_mtime_ns"] = os.stat(path).st_mtime_ns
                        except OSError:
                            job["_status_mtime_ns"] = None
                        job["_started_order"] = _job_started_order(path, job)
                        jobs.append(job)
            except (json.JSONDecodeError, OSError):
                continue
    except Exception as e:
        log_debug(f"Ошибка при опросе jobs: {e}")
        return True

    # The recording marker is written immediately after ffmpeg starts. If the
    # corresponding job metadata has not appeared yet (or could not be written),
    # still expose the active recording instead of leaving an old status row.
    try:
        with open(ACTIVE_RECORDING_FILE, encoding="utf-8") as stream:
            recording_pid = int(stream.readline().split("\t", 1)[0])
        os.kill(recording_pid, 0)
        recording_marker_active = True
    except (OSError, ValueError, IndexError):
        recording_marker_active = False
    if (recording_marker_active and
            not any(job.get("state") == "recording" for job in jobs)):
        try:
            recording_started = os.stat(ACTIVE_RECORDING_FILE).st_mtime
        except OSError:
            recording_started = float("inf")
        jobs.append({"_job_key": ".active-recording", "_status_mtime_ns": None,
                     "state": "recording", "source": "microphone",
                     "device": "unknown", "backend": "local", "model": "",
                     "_started_order": recording_started})

    batch_entries = build_batch_display_entries()

    # Preserve launch order across live jobs and the manual correction row.
    jobs.sort(key=lambda job: job.get("_started_order", float("inf")))
    live_dictation_active = any(
        job.get("source") in {"microphone", "live_dictation"}
        for job in jobs)

    manual_progress = _read_manual_correction_progress()
    if manual_progress is not None:
        # The daemon status file is authoritative once IPC has started. Keep
        # showing its current provider/model until the daemon removes it.
        _manual_correction_daemon_seen = True
    elif _manual_correction_daemon_seen:
        # Completion is an event (the daemon removed its status file), not a
        # timeout. Clear any stale local clipboard/launch hint immediately.
        _set_manual_correction_status(None)
        _manual_correction_daemon_seen = False
    elif _manual_correction_status is not None:
        if _manual_correction_status in {
                "Ручной корректор — запуск",
                "Ручной корректор — чтение буфера",
                "Ручной корректор — чтение Wayland clipboard"}:
            # The new live job takes precedence over a pre-IPC clipboard hint.
            # If the daemon has an active manual request, manual_progress above
            # preserves it as a separate, accurate provider/model row.
            if any(job.get("source") in {"microphone", "live_dictation"}
                   for job in jobs):
                _set_manual_correction_status(None)
    signature = (
        tuple((j.get("_job_key"), j.get("_status_mtime_ns"),
               j.get("started_ts"), j.get("source"),
               j.get("_started_order"),
               j.get("state"), j.get("device"), j.get("backend"), j.get("pipeline_stage"),
               j.get("model"), j.get("asr_backend_selected"),
               j.get("asr_backend_error_category"),
               j.get("asr_backend_execution_device"),
               j.get("asr_cpu_fallback_for_correction")) for j in jobs),
        tuple(e["label"] for e in batch_entries),
        _manual_correction_status, manual_progress, live_dictation_active,
    )
    if signature == _last_signature:
        return True
    _last_signature = signature

    display_entries = []
    for j in jobs:
        state = j.get("state", "processing")
        device = j.get("device", "unknown")
        model = j.get("model", "?")
        source = j.get("source", "microphone")
        backend = j.get("backend", j.get("device", "unknown"))
        stage = j.get("pipeline_stage", "")
        if stage in {"asr", "correction_wait", "asr_waiting_gpu"} and j.get("asr_backend_execution_device") in {"cpu", "cuda"}:
            device = j["asr_backend_execution_device"]
        action = "запись" if state == "recording" else "обработка"
        icon_key, _icon_label = backend_status_icon(
            state, backend, device, model,
            j.get("asr_backend_selected") if stage == "asr" else None)

        # Backend selection has not happened while recording; avoid the stale
        # model inherited from the launch profile.
        if state == "recording":
            label = "Запись"
        # During correction the model/backend are more useful than the source filename.
        elif state == "processing" and not stage:
            if (source in {"microphone", "live_dictation"} and
                    (_manual_correction_status or manual_progress)):
                label = f"{model} — диктовка ожидает ASR: занят ручной корректор"
            else:
                label = f"{model} — ожидание очереди"
        elif state == "processing" and stage == "correction_wait":
            device_label = {"cuda": "GPU", "cpu": "CPU"}.get(device)
            label = f"{model} — ASR готово"
            if device_label:
                label += f" на {device_label}"
            label += "; ожидание корректора"
        elif state == "processing" and stage == "asr_waiting_gpu":
            label = f"{model} — ожидание GPU: CPU fallback не сработал"
        elif state == "processing" and stage == "correction":
            label = processing_status_label(model, stage, icon_key)
        elif state == "processing" and stage == "asr" and source != "file":
            label = processing_status_label(model, stage, icon_key)
            label += asr_backend_status_suffix(j.get("asr_backend_selected"), model,
                                               j.get("asr_backend_error_category"))
            if j.get("asr_cpu_fallback_for_correction"):
                label += " · CPU (обход занятого корректором GPU)"
        elif source == "file":
            file_path = j.get("file", "unknown")
            file_name = os.path.basename(file_path)
            label = f"{file_name} — {model}: {stage or action}".rstrip()
            if state == "processing" and stage == "asr":
                label += asr_backend_status_suffix(j.get("asr_backend_selected"), model,
                                                   j.get("asr_backend_error_category"))
                if j.get("asr_cpu_fallback_for_correction"):
                    label += " · CPU (обход занятого корректором GPU)"
        else:
            label = f"{model} — {action} {stage}".rstrip()

        display_entries.append({"state": state, "device": device, "backend": backend,
                                "model": model, "pipeline_stage": stage,
                                "source": source,
                                "_started_order": j.get("_started_order", float("inf")),
                                "asr_backend_selected": j.get("asr_backend_selected"),
                                "status_icon": icon_key, "label": label})

    # Keep independent live dictation and manual correction visible together,
    # in the order each task was launched.
    if _manual_correction_status or manual_progress:
        manual_started_order = (_read_manual_correction_started_at() or
                                _manual_correction_started_at or float("inf"))
        manual_entry = {
            "state": "processing", "device": "remote", "backend": "remote",
            "source": "manual_correction",
            "_started_order": manual_started_order,
            "model": "Ручной корректор", "pipeline_stage": "correction",
            "status_icon": menu_icon_path("corrector"),
            "label": manual_progress or _manual_correction_status,
        }
        display_entries.append(manual_entry)
    display_entries.sort(key=lambda entry: entry.get("_started_order", float("inf")))

    # Batch jobs показываются одной строкой на весь job, а не по файлу
    display_entries.extend(batch_entries)

    for i, item in enumerate(job_slot_items):
        if i < len(display_entries):
            entry = display_entries[i]
            batch_id = entry.get("batch_id")
            image = job_slot_widgets[i]
            set_status_row_icon(image, entry.get("status_icon", "processing_cpu"))
            item.set_label(entry["label"])
            set_menu_label_max_width(item, 76)
            item.set_visible(True)
            item.set_sensitive(batch_id is not None)
            job_slot_batch_ids[i] = batch_id
        else:
            # Clear hidden GTK labels as well as visibility; otherwise a
            # previously rendered provider/model string can remain cached in
            # the open tray menu after the job has finished.
            item.set_label("")
            item.set_visible(False)
            item.set_sensitive(False)
            job_slot_batch_ids[i] = None

    menu.queue_draw()

    idle_item.set_visible(len(display_entries) == 0)
    jobs_separator.set_visible(len(display_entries) > 0)

    # Определить иконку трея на основе состояния И device
    recording_entries = [entry for entry in display_entries if entry["state"] == "recording"]
    processing_entries = [entry for entry in display_entries if entry["state"] == "processing"]
    local_cpu_entries = [entry for entry in processing_entries
                         if entry.get("status_icon") in {
                             "processing_cpu_flame", "processing_cpu"}]
    if local_cpu_entries:
        selected_entry = local_cpu_entries[0]
        processing_icon = selected_entry.get("status_icon", "processing_cpu_flame")
        icon = ICONS[processing_icon]
        tooltip = processing_status_label(selected_entry.get("model"),
                                          selected_entry.get("pipeline_stage"),
                                          processing_icon)
        if len(processing_entries) > 1:
            tooltip += f" · ещё {len(processing_entries) - 1} задач"
    elif recording_entries:
        icon = ICONS[backend_status_icon("recording", "local", "cpu")[0]]
        tooltip = f"🔴 Запись ({len(recording_entries)} активно)"
    elif processing_entries:
        priority = ("processing_gamepad", "processing_gpu",
                    "processing_cloudflare", "processing_groq",
                    "processing_google", "processing_openrouter", "processing_codex",
                    "processing_claude", "processing_gigaam")
        selected_entry = next((entry for key in priority for entry in processing_entries
                               if entry.get("status_icon") == key), processing_entries[0])
        processing_icon = selected_entry.get("status_icon", "processing_cpu")
        if processing_icon == "processing_gpu":
            # Match the global AppIndicator GPU glyph to the current GPU task-row icon.
            processing_icon = "processing_gamepad"
        icon = ICONS[processing_icon]
        tooltip = processing_status_label(selected_entry.get("model"),
                                          selected_entry.get("pipeline_stage"),
                                          processing_icon)
        if len(processing_entries) > 1:
            tooltip += f" · ещё {len(processing_entries) - 1} задач"
    else:
        icon = ICONS["idle"]
        tooltip = "🎤 Whisper Dictate — ожидание"

    try:
        indicator.set_icon_full(icon, tooltip)
        log_debug(f"Иконка изменена: {tooltip}")
    except Exception as e:
        log_debug(f"Ошибка при установке иконки {icon}: {e}")

    return True

def update_gpu_info_ui():
    """Update GPU info in label every 2 seconds if auto mode is enabled."""
    quality_config = load_quality_config()
    if quality_config.get("auto_mode", False):
        update_model_label()
    return True

def poll_jobs_safely():
    """Keep the recurring status refresh alive after an unexpected UI error."""
    try:
        return poll_jobs()
    except Exception as exc:
        # Do not log exception details: a GTK or path error may contain user data.
        log_debug(f"Ошибка обновления статусов задач ({type(exc).__name__})")
        return True

def update_daemon_control_items():
    """Показывать только подходящую команду управления демоном."""
    is_active = get_daemon_status()

    if is_active:
        quick_transcribe_item.set_sensitive(True)
        transcribe_file_item.set_sensitive(True)
        youtube_transcribe_item.set_sensitive(True)
    else:
        quick_transcribe_item.set_sensitive(False)
        transcribe_file_item.set_sensitive(False)
        youtube_transcribe_item.set_sensitive(False)

    for control_type, item in daemon_control_items:
        if control_type == "start":
            item.set_visible(not is_active)
        else:  # stop и restart
            item.set_visible(is_active)

    return True

update_daemon_control_items()

GLib.timeout_add(POLL_MS, poll_jobs_safely)
GLib.timeout_add(POLL_MS, poll_batch_jobs)
GLib.timeout_add(POLL_MS, poll_youtube_downloads)
GLib.timeout_add(2000, update_gpu_info_ui)
GLib.timeout_add(2000, update_daemon_control_items)
log_debug("Трей запущен (часы вместо букв)")
Gtk.main()
