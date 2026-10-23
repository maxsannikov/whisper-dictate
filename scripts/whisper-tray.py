#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gi, glob, html, json, os, shutil, signal, subprocess, sys, tempfile, time, unicodedata, re
from datetime import datetime

gi.require_version('Gtk', '3.0')
gi.require_version('AppIndicator3', '0.1')
from gi.repository import Gtk, AppIndicator3, GLib, Pango, Gdk

sys.path.insert(0, os.path.dirname(__file__))
from cloudflare_asr import daily_usage as cloudflare_daily_usage

STATE_DIR = os.path.expanduser("~/.cache/whisper-dictate")
JOBS_DIR = os.path.join(STATE_DIR, "jobs")
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
    try:
        if os.path.exists(QUALITY_CONFIG_FILE):
            with open(QUALITY_CONFIG_FILE, encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log_debug(f"Ошибка чтения quality-config: {e}")
    return DEFAULT_QUALITY_CONFIG.copy()

def save_quality_config(config):
    try:
        fd, tmp_path = tempfile.mkstemp(dir=STATE_DIR, suffix=".tmp", prefix="quality-config-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, QUALITY_CONFIG_FILE)
            log_debug(f"Конфиг качества сохранён (strategy={config.get('current_strategy')})")
        except:
            try: os.unlink(tmp_path)
            except: pass
            raise
    except Exception as e:
        log_debug(f"Ошибка сохранения quality-config: {e}")

def load_voice_config():
    """Read the canonical daemon configuration; never fall back to a cloud profile."""
    try:
        with open(VOICE_CONFIG_FILE, encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        log_debug(f"Ошибка чтения voice config: {e}")
        return {}

def save_voice_config(value):
    try:
        os.makedirs(os.path.dirname(VOICE_CONFIG_FILE), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(VOICE_CONFIG_FILE), suffix=".tmp", prefix="config-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, VOICE_CONFIG_FILE)
        except:
            try: os.unlink(tmp_path)
            except OSError: pass
            raise
    except Exception as e:
        log_debug(f"Ошибка сохранения voice config: {e}")

# Использовать готовые иконки из системы
ICONS = {
    "idle": "microphone-sensitivity-high-symbolic",  # Микрофон с волнами
    "recording": "/usr/share/icons/Papirus/24x24/actions/media-record.svg",  # КРАСНЫЙ кружочек
    "processing_cloudflare": "/usr/share/icons/Papirus/24x24/actions/cloud-download.svg",
    "processing_gpu": "/usr/share/icons/Papirus/24x24/devices/input-gaming.svg",
    "processing_cpu": "/usr/share/icons/Papirus/24x24/devices/cpu.svg"
}

DEVICE_EMOJI = {"cuda": "🎮", "cpu": "🔥", "cloudflare": "☁️", "unknown": "❔"}
STATE_EMOJI = {"recording": "🔴", "processing": "⏰"}
BATCH_STATUS_LABELS = {
    "queued": "⏳ Ожидание",
    "processing": "⚙️ Обработка",
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

def unload_corrector_model(_):
    """Выгрузить только выбранную локальную модель Ollama из RAM/VRAM."""
    corrector = load_voice_config().get("corrector", {})
    model = corrector.get("model")
    if not isinstance(model, str) or not model:
        notify_send("Whisper Dictate", "Модель Qwen для выгрузки не выбрана")
        return
    try:
        result = subprocess.run(["ollama", "stop", model], timeout=10,
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ollama stop failed")
        log_debug("[corrector-unload] Локальная модель выгружена")
        notify_send("Whisper Dictate", "🧹 Qwen выгружен из RAM/VRAM")
    except Exception as e:
        log_debug(f"[corrector-unload] Ошибка: {e}")
        notify_send("Ошибка", f"Не удалось выгрузить Qwen: {e}")

def get_error_logs():
    """Собрать все логи с ошибками из всех файлов"""
    log_files = [
        os.path.join(STATE_DIR, "tray.log"),
        os.path.join(STATE_DIR, "daemon.log"),
        os.path.join(STATE_DIR, "autopaste.log"),
    ]

    error_lines = []
    error_keywords = ["✗", "Ошибка", "Error", "Traceback", "Failed", "Exception"]

    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        if any(keyword in line for keyword in error_keywords):
                            error_lines.append(line.rstrip())
            except:
                pass

    return error_lines

def get_work_logs():
    """Собрать все логи работы (без ошибок) из всех файлов"""
    log_files = [
        os.path.join(STATE_DIR, "tray.log"),
        os.path.join(STATE_DIR, "daemon.log"),
        os.path.join(STATE_DIR, "autopaste.log"),
    ]

    work_lines = []
    work_keywords = ["[daemon-status]", "Трей запущен", "Иконка изменена", "✓"]
    error_keywords = ["✗", "Ошибка", "Error", "Traceback"]

    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        # Если это ошибка - пропустить
                        if any(keyword in line for keyword in error_keywords):
                            continue
                        # Если это рабочий лог - добавить
                        if any(keyword in line for keyword in work_keywords):
                            work_lines.append(line.rstrip())
            except:
                pass

    return work_lines

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
    supported_exts = {
        '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma',
        '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.m4v'
    }
    ext = os.path.splitext(file_path)[1].lower()
    return ext in supported_exts

def is_video_file(file_path):
    """Проверить что файл — видео, по расширению"""
    video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.m4v'}
    ext = os.path.splitext(file_path)[1].lower()
    return ext in video_exts

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
    audio_filter.set_name("Audio files (MP3, WAV, M4A, FLAC, OGG)")
    for pattern in ["*.mp3", "*.wav", "*.m4a", "*.aac", "*.flac", "*.ogg", "*.wma"]:
        audio_filter.add_pattern(pattern)
    dialog.add_filter(audio_filter)

    video_filter = Gtk.FileFilter()
    video_filter.set_name("Video files (MP4, MKV, WebM, AVI, MOV)")
    for pattern in ["*.mp4", "*.mkv", "*.webm", "*.avi", "*.mov", "*.flv", "*.m4v"]:
        video_filter.add_pattern(pattern)
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
    audio_filter.set_name("Audio files")
    for pattern in ["*.mp3", "*.wav", "*.ogg", "*.flac", "*.m4a"]:
        audio_filter.add_pattern(pattern)
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

        files.append({
            "path": f["audio_path"],
            "name": display_name,
            "original_name": f["name"],
            "type": f.get("type") or ("video → audio" if f["audio_path"] != f["original_path"] else "audio"),
            "source_url": f.get("source_url"),
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
        proc = subprocess.Popen([
            os.path.expanduser("~/.local/bin/whisper/whisper-dictate-v2"),
            "--file", file_entry["path"],
            "--batch-id", job["id"],
            "--output", file_entry["output_path"]
        ], start_new_session=True, env={**os.environ, "WHISPER_JOB_SOURCE": "youtube" if file_entry.get("source_url") else "file"})
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
        label = f"📦 BATCH: {job.get('id','?')} ({done}/{total} файлов) ⏰"
        entries.append({"state": "processing", "device": "cuda", "label": label, "batch_id": job.get("id")})
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
    """Скопировать текст в системный буфер обмена"""
    try:
        clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        clipboard.set_text(text, -1)
        clipboard.store()
    except Exception as e:
        log_debug(f"✗ Ошибка копирования в буфер: {e}")

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
    title_label.set_markup("<b>Введите ссылки на YouTube (одна на строку):</b>")
    title_label.set_xalign(0)
    content_area.pack_start(title_label, False, False, 0)

    # Подсказка с форматами
    hint_label = Gtk.Label()
    hint_label.set_markup(
        "<small>Поддерживаемые форматы:\n"
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
                "live": "это живой стрим"
            }
            reason_msg = reason_msgs.get(reason, reason)
            text_parts.append(f"• {entry.get('video_id', '?')} — {reason_msg}")

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
    import fcntl
    import os
    global _youtube_metadata_fetch

    if not entries:
        return

    urls = [e.get("url") for e in entries if e.get("url")]
    if not urls:
        return

    log_debug(f"🎥 Запускаю fetch метаданных для {len(urls)} видео...")

    try:
        proc = subprocess.Popen(
            ["yt-dlp", "-j", "--skip-download", "--no-playlist", "--no-warnings",
             "--socket-timeout", "15"] + urls,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True
        )

        # Установить неблокирующий режим для stdout
        fd = proc.stdout.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)

        # Создать mapping video_id -> url для восстановления URL в resolved
        url_mapping = {}
        for entry in entries:
            vid = entry.get("video_id")
            if vid:
                url_mapping[vid] = entry.get("url")

        _youtube_metadata_fetch = {
            "proc": proc,
            "resolved": [],
            "failed": [],
            "timeout_at": time.time() + YTDLP_METADATA_TIMEOUT * len(entries),
            "on_done": on_done,
            "original_entries": entries,
            "url_mapping": url_mapping,
            "buffer": ""
        }

        GLib.timeout_add(POLL_MS, _poll_metadata_fetch)
        log_debug(f"📌 Fetch процесс запущен (pid={proc.pid})")
    except Exception as e:
        log_debug(f"✗ Не удалось запустить yt-dlp: {e}")
        GLib.idle_add(on_done, [], entries)

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
                        if url:
                            entry["url"] = url
                        _youtube_metadata_fetch["resolved"].append(entry)
        except:
            pass

        try:
            stderr = proc.stderr.read()
            if stderr:
                log_debug(f"🎥 yt-dlp stderr: {stderr[:200]}")
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
                    _youtube_downloads.pop(download_id, None)

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

def on_youtube_downloads_finished(download_id, quality_config, dl_entries):
    """Вызывается когда все скачивания завершены или все в кэше.

    dl_entries: результат из callback (уже подготовленные файлы) ИЛИ список dl_entries с cached=True/False
    """
    # Если dl_entries уже содержит prepared files (из callback), используем их
    if dl_entries and len(dl_entries) > 0 and dl_entries[0].get("audio_path"):
        prepared_files = dl_entries
    else:
        # Собрать успешные файлы из _youtube_downloads
        prepared_files = []
        failed_count = 0

        downloads = _youtube_downloads.get(download_id, {})
        for idx, entry_data in sorted([item for item in downloads.items() if item[0] != "_on_done"]):
            entry = entry_data.get("entry", {})
            error_info = entry_data.get("error")

            if error_info:
                failed_count += 1
                continue

            audio_path = youtube_audio_path(entry.get("video_id", ""), entry.get("title", ""))
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                duration = get_file_duration(audio_path)
                prepared_files.append({
                    "name": f"{entry.get('title', '?')}.mp3",
                    "original_path": audio_path,
                    "audio_path": audio_path,
                    "duration": duration or entry.get("duration_sec", 0),
                    "type": "youtube",
                    "source_url": entry.get("url"),
                    "video_id": entry.get("video_id")
                })

    if 'failed_count' in locals() and failed_count > 0:
        notify_send("Whisper YouTube", f"⚠️ {failed_count} видео не удалось скачать")

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
                if not show_youtube_validation_dialog([], failed):
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
            to_download = [e for e in dl_entries if not e["cached"]]
            if to_download:
                disk_check = check_youtube_disk_space(to_download)
                if not disk_check["ok"]:
                    show_error_dialog("Недостаточно места", disk_check["error"])
                    return

            # Создать окно прогресса
            _youtube_downloads[download_id] = {"_on_done": None}
            for i, entry in enumerate(to_download):
                dest_dir = YOUTUBE_AUDIO_DIR
                proc = start_youtube_download(entry, dest_dir)
                if proc:
                    _youtube_downloads[download_id][i] = {
                        "proc": proc,
                        "entry": entry,
                        "progress": 0.0,
                        "eta_sec": None
                    }

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

def on_cleanup_tasks_clicked(_):
    """Очистить все зависшие задачи (и микрофон, и файлы)"""
    try:
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

        msg = "✓ Все задачи отменены"
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


menu = Gtk.Menu()

quick_transcribe_item = Gtk.MenuItem(label="▶️ Новая транскрибация")
quick_transcribe_item.connect("activate", on_quick_transcribe_clicked)
menu.append(quick_transcribe_item)

transcribe_file_item = Gtk.MenuItem(label="📁 Распознать файл(ы)...")
transcribe_file_item.connect("activate", on_transcribe_file_clicked)
menu.append(transcribe_file_item)

youtube_transcribe_item = Gtk.MenuItem(label="🎥 Распознать видео с YouTube...")
youtube_transcribe_item.connect("activate", on_youtube_transcribe_clicked)
menu.append(youtube_transcribe_item)

cleanup_tasks_item = Gtk.MenuItem(label="🧹 Очистить зависшие задачи")
cleanup_tasks_item.connect("activate", on_cleanup_tasks_clicked)
menu.append(cleanup_tasks_item)

menu.append(Gtk.SeparatorMenuItem())

whisper_settings_submenu = Gtk.Menu()

def cloudflare_credentials_available():
    """Check credential file and required variable names without exposing values."""
    path = os.path.expanduser("~/.config/whisper-dictate/cloudflare-workers-ai.env")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as stream:
            names = {line.split("=", 1)[0].strip() for line in stream
                     if "=" in line and not line.lstrip().startswith("#")}
        return {"CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"}.issubset(names)
    except OSError:
        return False

def on_cloudflare_toggled(item):
    value = load_voice_config()
    value["cloudflare_asr_enabled"] = item.get_active()
    save_voice_config(value)
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
cloudflare_asr_toggle = Gtk.CheckMenuItem(label="Использовать Cloudflare для ASR Whisper")
cloudflare_asr_toggle.set_active(bool(load_voice_config().get("cloudflare_asr_enabled", load_voice_config().get("cloudflare_workers_ai_enabled", False))))
cloudflare_asr_toggle.connect("toggled", on_cloudflare_toggled)
cloudflare_submenu.append(cloudflare_asr_toggle)
cloudflare_corrector_toggle = Gtk.CheckMenuItem(label="Использовать Cloudflare для корректора")
cloudflare_corrector_toggle.set_active(bool(load_voice_config().get("cloudflare_corrector", {}).get("enabled", False)))
def on_cloudflare_corrector_toggled(item):
    value = load_voice_config()
    profile = value.get("cloudflare_corrector", {})
    profile.update({"enabled": item.get_active(), "model": profile.get("model", "@cf/qwen/qwen3-30b-a3b-fp8"),
                    "temperature": profile.get("temperature", 0.0), "max_tokens": profile.get("max_tokens", 2048)})
    value["cloudflare_corrector"] = profile
    save_voice_config(value)
cloudflare_corrector_toggle.connect("toggled", on_cloudflare_corrector_toggled)
cloudflare_submenu.append(cloudflare_corrector_toggle)
cloudflare_submenu.append(Gtk.SeparatorMenuItem())
cloudflare_model_group = []
cloudflare_asr_model_group = []
cloudflare_asr_models = [("@cf/openai/whisper-large-v3-turbo", "Whisper Large-v3 Turbo (по умолчанию)"), ("@cf/openai/whisper", "Whisper Large-v3")]
for model_name, model_label in cloudflare_asr_models:
    model_item = Gtk.RadioMenuItem(label=model_label) if not cloudflare_asr_model_group else Gtk.RadioMenuItem.new_from_widget(cloudflare_asr_model_group[0])
    model_item.set_label("ASR: " + model_label)
    model_item.set_active(load_voice_config().get("cloudflare_asr_model", "@cf/openai/whisper-large-v3-turbo") == model_name)
    def select_cloudflare_asr_model(menu_item, selected_model):
        if menu_item.get_active():
            value = load_voice_config(); value["cloudflare_asr_model"] = selected_model; save_voice_config(value)
    model_item.connect("toggled", select_cloudflare_asr_model, model_name)
    cloudflare_asr_model_group.append(model_item); cloudflare_submenu.append(model_item)
cloudflare_submenu.append(Gtk.SeparatorMenuItem())
cloudflare_models = [("@cf/qwen/qwen3-30b-a3b-fp8", "Qwen3 30B A3B FP8 (по умолчанию)"),
                    ("@cf/zai-org/glm-4.7-flash", "GLM 4.7 Flash")]
for model_name, model_label in cloudflare_models:
    model_item = Gtk.RadioMenuItem(label=model_label) if not cloudflare_model_group else Gtk.RadioMenuItem.new_from_widget(cloudflare_model_group[0])
    model_item.set_label(model_label)
    model_item.set_active(load_voice_config().get("cloudflare_corrector", {}).get("model", "@cf/qwen/qwen3-30b-a3b-fp8") == model_name)
    def select_cloudflare_model(menu_item, selected_model):
        if menu_item.get_active():
            value = load_voice_config()
            profile = value.get("cloudflare_corrector", {})
            profile["model"] = selected_model
            value["cloudflare_corrector"] = profile
            save_voice_config(value)
    model_item.connect("toggled", select_cloudflare_model, model_name)
    cloudflare_model_group.append(model_item)
    cloudflare_submenu.append(model_item)
cloudflare_submenu.append(Gtk.SeparatorMenuItem())
def set_cloudflare_parameter(item, key, val):
    if item.get_active():
        value = load_voice_config()
        profile = value.get("cloudflare_corrector", {})
        profile[key] = val
        value["cloudflare_corrector"] = profile
        save_voice_config(value)
for key, options in [("temperature", [(0.0, "Температура 0"), (0.2, "Температура 0.2"), (0.6, "Температура 0.6")]),
                    ("max_tokens", [(512, "Ответ до 512 токенов"), (1024, "Ответ до 1024 токенов"), (2048, "Ответ до 2048 токенов"), (4096, "Ответ до 4096 токенов")])]:
    group = []
    for val, label in options:
        option = Gtk.RadioMenuItem(label=label) if not group else Gtk.RadioMenuItem.new_from_widget(group[0])
        option.set_label(label)
        option.set_active(load_voice_config().get("cloudflare_corrector", {}).get(key, 0.0 if key == "temperature" else 2048) == val)
        option.connect("toggled", set_cloudflare_parameter, key, val)
        group.append(option)
        cloudflare_submenu.append(option)
cloudflare_submenu.append(Gtk.SeparatorMenuItem())
cloudflare_status = Gtk.MenuItem()
cloudflare_status.set_sensitive(False)
cloudflare_status.set_label("✅ Учётные данные доступны" if cloudflare_credentials_available()
                            else "❌ Нет файла или обязательных переменных")
cloudflare_submenu.append(cloudflare_status)
cloudflare_usage = Gtk.MenuItem()
cloudflare_usage.set_sensitive(False)

def refresh_cloudflare_usage(*_args):
    usage = cloudflare_daily_usage(METRICS_FILE)
    correction = usage.get("corrector_estimated_neurons", 0)
    total = int(usage["estimated_neurons"] + correction + 0.5)
    correction_basis = "оценка токенов" if usage.get("corrector_estimated_token_requests", 0) else "usage API/токены"
    minutes = usage["audio_seconds"] / 60
    cloudflare_usage.set_label(
        f"~{total:,} нейронов "
        f"(ASR ~{int(usage['estimated_neurons']):,}; корректор ~{int(correction):,}, {correction_basis}); "
        f"ASR ~{minutes:.1f} мин, {usage['requests']} запросов".replace(",", " "))

refresh_cloudflare_usage()
cloudflare_submenu.connect("show", refresh_cloudflare_usage)
cloudflare_submenu.append(cloudflare_usage)
cloudflare_submenu.show_all()
cloudflare_menu_item = Gtk.MenuItem(label="☁️ Cloudflare Workers AI")
cloudflare_menu_item.set_submenu(cloudflare_submenu)
menu.append(cloudflare_menu_item)
menu.append(Gtk.SeparatorMenuItem())

job_slot_items = []
job_slot_batch_ids = [None] * MAX_JOB_SLOTS

def on_job_slot_clicked(idx):
    batch_id = job_slot_batch_ids[idx]
    if batch_id:
        create_batch_progress_window(batch_id)

for _slot_idx in range(MAX_JOB_SLOTS):
    item = Gtk.MenuItem(label="")
    item.set_sensitive(False)
    item.set_visible(False)
    item.connect("activate", lambda _item, idx=_slot_idx: on_job_slot_clicked(idx))
    menu.append(item)
    job_slot_items.append(item)

jobs_separator = Gtk.SeparatorMenuItem()
menu.append(jobs_separator)

idle_item = Gtk.MenuItem(label="🎤 Свободно — нет активных распознаний")
idle_item.set_sensitive(False)
menu.append(idle_item)

menu.append(Gtk.SeparatorMenuItem())

daemon_status_item = Gtk.MenuItem(label="✅ Демон: активен")
daemon_status_item.set_sensitive(False)
menu.append(daemon_status_item)

daemon_control_items = []

daemon_start_item = Gtk.MenuItem(label="▶️ Запустить демон")
daemon_start_item.connect("activate", start_daemon)
daemon_start_item.set_visible(False)
menu.append(daemon_start_item)
daemon_control_items.append(("start", daemon_start_item))

daemon_stop_item = Gtk.MenuItem(label="⏹️ Остановить демон")
daemon_stop_item.connect("activate", stop_daemon)
daemon_stop_item.set_visible(False)
menu.append(daemon_stop_item)
daemon_control_items.append(("stop", daemon_stop_item))

daemon_restart_item = Gtk.MenuItem(label="🔄 Перезапустить демон")
daemon_restart_item.connect("activate", restart_daemon)
daemon_restart_item.set_visible(False)
menu.append(daemon_restart_item)
daemon_control_items.append(("restart", daemon_restart_item))

corrector_unload_item = Gtk.MenuItem(label="🧹 Выгрузить Qwen из RAM/VRAM")
corrector_unload_item.connect("activate", unload_corrector_model)
menu.append(corrector_unload_item)

menu.append(Gtk.SeparatorMenuItem())

quality_config = load_quality_config()
current_strategy = quality_config.get("current_strategy", "normal")
current_model = quality_config.get("current_model", "large-v3")
radio_group = []
model_group = []
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
    if item.get_active():
        voice_config = load_voice_config()
        voice_config["model"] = model_name
        save_voice_config(voice_config)
        update_model_label()
        log_debug(f"Каноническая модель изменена на: {model_name}")

def on_vad_sensitivity_changed(item, threshold_val):
    if item.get_active():
        quality_config = load_quality_config()
        strategy = quality_config["current_strategy"]
        quality_config["strategies"][strategy]["vad_threshold"] = threshold_val
        save_quality_config(quality_config)
        log_debug(f"VAD Sensitivity изменена на: {threshold_val}")

def update_model_label():
    current_model = load_voice_config().get("model", "large-v3")
    model_menu_item.set_label(f"📊 Модель диктовки — {current_model}")

def update_corrector_label():
    value = load_voice_config()
    corrector = value.get("corrector", {})
    model = corrector.get("model", "Qwen [не выбран]")
    state = "вкл" if corrector.get("enabled", False) else "выкл"
    short_model = "4B" if ":4b-" in model else "2B" if ":2b-" in model else model
    corrector_menu_item.set_label(f"✍️ Корректор: {state} · {short_model}")
    if 'corrector_enabled_item' in globals():
        corrector_enabled_item.set_label("Включить после ASR (локально)")

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

def on_corrector_model_selected(item, model_name):
    if not item.get_active():
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
    corrector["model"] = model_name
    value["corrector"] = corrector
    save_voice_config(value)
    update_corrector_label()
    log_debug(f"Модель корректора изменена на: {model_name}")

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
    open_in_editor(AUTOREPLACEMENTS_FILE)

def on_corrector_prompt_clicked(_):
    value = load_voice_config()
    corrector = value.get("corrector", {})
    dialog = Gtk.Dialog(title="Prompt локального корректора", transient_for=None, flags=0)
    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK)
    dialog.set_default_size(640, 300)
    text_view = Gtk.TextView()
    text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
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

    beam_spin = Gtk.SpinButton()
    beam_spin.set_range(1, 50)
    beam_spin.set_increments(1, 5)
    beam_spin.set_value(custom_params.get("beam_size", 5))
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

custom_item = Gtk.MenuItem(label="⚙️ Пользовательская...")
custom_item.connect("activate", on_custom_clicked)
quality_submenu.append(custom_item)

update_custom_label()

quality_submenu.show_all()

quality_menu_item = Gtk.MenuItem(label="🎚️ Качество распознавания")
quality_menu_item.set_submenu(quality_submenu)
whisper_settings_submenu.append(quality_menu_item)

model_submenu = Gtk.Menu()

model_options_list = [
    ("large-v3", "🔥 Large-v3 (best quality, 4-5GB, медленный)"),
    ("large-v3-turbo", "🚀 Large-v3 Turbo (эксперимент: быстрее, требуется A/B)") ,
    ("medium", "🎯 Medium (good quality, 1.5GB, нормальный)"),
    ("small", "⚡ Small (fast, 0.5GB, быстрый)"),
]

for i, (model_key, label) in enumerate(model_options_list):
    item = Gtk.RadioMenuItem(label=label) if not model_group else Gtk.RadioMenuItem.new_from_widget(model_group[0])
    item.set_label(label)
    model_group.append(item)

    if load_voice_config().get("model", "large-v3") == model_key:
        item.set_active(True)

    item.connect("toggled", on_model_selected, model_key)
    model_submenu.append(item)

model_submenu.show_all()

model_menu_item = Gtk.MenuItem(label="📊 Модель диктовки")
model_menu_item.set_submenu(model_submenu)
whisper_settings_submenu.append(model_menu_item)
update_model_label()

voice_config = load_voice_config()
corrector_config = voice_config.get("corrector", {})
corrector_submenu = Gtk.Menu()
corrector_enabled_item = Gtk.CheckMenuItem(label="Включить после ASR (локально)")
corrector_enabled_item.set_active(corrector_config.get("enabled", False))
corrector_enabled_item.connect("toggled", on_corrector_enabled_toggled)
corrector_submenu.append(corrector_enabled_item)
corrector_submenu.append(Gtk.SeparatorMenuItem())
corrector_vram_item = Gtk.MenuItem(label="🎮 VRAM: обновление…")
corrector_vram_item.set_sensitive(False)
corrector_submenu.append(corrector_vram_item)
corrector_refresh_vram_item = Gtk.MenuItem(label="↻ Обновить VRAM")
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
performance_item = Gtk.MenuItem(label="⚙️ Производительность Qwen")
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
corrector_submenu.append(Gtk.SeparatorMenuItem())
corrector_prompt_item = Gtk.MenuItem(label="✏️ Изменить prompt…")
corrector_prompt_item.connect("activate", on_corrector_prompt_clicked)
corrector_submenu.append(corrector_prompt_item)
autoreplacements_item = Gtk.MenuItem(label="📝 Автозамены…")
autoreplacements_item.connect("activate", on_autoreplacements_clicked)
corrector_submenu.append(autoreplacements_item)
corrector_submenu.show_all()
corrector_menu_item = Gtk.MenuItem(label="✍️ Корректор")
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

advanced_menu_item = Gtk.MenuItem(label="🔧 Расширенные настройки")
advanced_menu_item.set_submenu(advanced_menu)
whisper_settings_submenu.append(Gtk.SeparatorMenuItem())
whisper_settings_submenu.append(advanced_menu_item)
whisper_settings_submenu.append(Gtk.SeparatorMenuItem())
audio_duck_item = Gtk.CheckMenuItem(label="🔈 Приглушать звук при диктовке")
audio_duck_item.set_active(isinstance(voice_config.get("audio_ducking"), dict) and voice_config["audio_ducking"].get("enabled", False))
audio_duck_item.connect("toggled", on_audio_ducking_toggled)
whisper_settings_submenu.append(audio_duck_item)
audio_restore_item = Gtk.MenuItem(label="🔊 Восстановить звук")
audio_restore_item.connect("activate", on_audio_restore_clicked)
whisper_settings_submenu.append(audio_restore_item)
whisper_settings_submenu.show_all()
whisper_settings_item = Gtk.MenuItem(label="⚙️ Настройки Whisper")
whisper_settings_item.set_submenu(whisper_settings_submenu)
menu.append(whisper_settings_item)
menu.append(corrector_menu_item)

menu.append(Gtk.SeparatorMenuItem())

history_item = Gtk.MenuItem(label="📖 История транскрибации")
history_item.connect("activate", open_history)
menu.append(history_item)

error_logs_item = Gtk.MenuItem(label="❌ Логи ошибок")
error_logs_item.connect("activate", open_error_logs)

work_logs_item = Gtk.MenuItem(label="📝 Логи работы")
work_logs_item.connect("activate", open_work_logs)
logs_submenu = Gtk.Menu()
logs_submenu.append(error_logs_item)
logs_submenu.append(work_logs_item)
logs_submenu.show_all()
logs_menu_item = Gtk.MenuItem(label="📋 Логи")
logs_menu_item.set_submenu(logs_submenu)
menu.append(logs_menu_item)

def open_user_guide(_item):
    """Open the installed guide in a text editor, then xdg-open as fallback."""
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

guide_item = Gtk.MenuItem(label="⌨️ Горячие клавиши и установка")
guide_item.connect("activate", open_user_guide)
menu.append(guide_item)

github_url = "https://github.com/maxsannikov/whisper-dictate"

def open_project_github(_item):
    try:
        subprocess.Popen(["xdg-open", github_url])
    except Exception as exc:
        log_debug(f"✗ Не удалось открыть страницу GitHub: {exc}")

github_item = Gtk.MenuItem(label="🌐 GitHub проекта")
github_item.connect("activate", open_project_github)
menu.append(github_item)

menu.append(Gtk.SeparatorMenuItem())
quit_item = Gtk.MenuItem(label="Выход")
quit_item.connect("activate", lambda _: Gtk.main_quit())
menu.append(quit_item)

menu.show_all()
for item in job_slot_items:
    item.set_visible(False)
jobs_separator.set_visible(False)

indicator.set_menu(menu)

_last_signature = None
def poll_jobs():
    global _last_signature
    jobs = []
    try:
        for path in sorted(glob.glob(os.path.join(JOBS_DIR, "*.json"))):
            try:
                with open(path, encoding="utf-8") as f:
                    jobs.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    except Exception as e:
        log_debug(f"Ошибка при опросе jobs: {e}")
        return True

    batch_entries = build_batch_display_entries()

    signature = (
        tuple((j.get("state"), j.get("device"), j.get("model")) for j in jobs),
        tuple(e["label"] for e in batch_entries)
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
        action = "запись" if state == "recording" else "обработка"

        # Если это файл — показать имя файла вместо модели
        if source == "file":
            file_path = j.get("file", "unknown")
            file_name = os.path.basename(file_path)
            label = f"{STATE_EMOJI.get(state,'?')} {file_name} — {action} {DEVICE_EMOJI.get(backend,'❔')} {stage}"
        else:
            # Это микрофон (исходное поведение)
            label = f"{STATE_EMOJI.get(state,'?')} {model} — {action} {DEVICE_EMOJI.get(backend,'❔')} {stage}"

        display_entries.append({"state": state, "device": device, "label": label})

    # Batch jobs показываются одной строкой на весь job, а не по файлу
    display_entries.extend(batch_entries)

    for i, item in enumerate(job_slot_items):
        if i < len(display_entries):
            entry = display_entries[i]
            batch_id = entry.get("batch_id")
            item.set_label(entry["label"])
            item.set_visible(True)
            item.set_sensitive(batch_id is not None)
            job_slot_batch_ids[i] = batch_id
        else:
            item.set_visible(False)
            job_slot_batch_ids[i] = None

    idle_item.set_visible(len(display_entries) == 0)
    jobs_separator.set_visible(len(display_entries) > 0)

    # Определить иконку трея на основе состояния И device
    processing_entries = [entry for entry in display_entries if entry["state"] == "processing"]
    if processing_entries and any(e["device"] == "cloudflare" for e in processing_entries):
        icon = ICONS["processing_cloudflare"]
        tooltip = f"☁️ Обработка Cloudflare ({len(processing_entries)} активно)"
    elif processing_entries and any(e["device"] == "cuda" for e in processing_entries):
        icon = ICONS["processing_gpu"]
        tooltip = f"🎮 Обработка GPU ({len(processing_entries)} активно)"
    elif processing_entries and any(e["device"] == "cpu" for e in processing_entries):
        icon = ICONS["processing_cpu"]
        tooltip = f"🔥 Обработка CPU ({len(processing_entries)} активно)"
    elif any(e["state"] == "recording" for e in display_entries):
        icon = ICONS["recording"]
        tooltip = f"🔴 Запись ({len(display_entries)} активно)"
    elif processing_entries:
        icon = ICONS["processing_cpu"]
        tooltip = f"⚙️ Обработка ({len(processing_entries)} активно)"
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

def update_daemon_control_items():
    """Обновить кнопки управления демоном в зависимости от статуса"""
    is_active = get_daemon_status()

    # Обновить статус
    if is_active:
        daemon_status_item.set_label("✅ Демон: активен")
        quick_transcribe_item.set_sensitive(True)
        transcribe_file_item.set_sensitive(True)
        youtube_transcribe_item.set_sensitive(True)
    else:
        daemon_status_item.set_label("❌ Демон: остановлен")
        quick_transcribe_item.set_sensitive(False)
        transcribe_file_item.set_sensitive(False)
        youtube_transcribe_item.set_sensitive(False)

    # Показать/скрыть кнопки управления
    for control_type, item in daemon_control_items:
        if control_type == "start":
            item.set_visible(not is_active)
        else:  # stop и restart
            item.set_visible(is_active)

    return True

update_daemon_control_items()

GLib.timeout_add(POLL_MS, poll_jobs)
GLib.timeout_add(POLL_MS, poll_batch_jobs)
GLib.timeout_add(POLL_MS, poll_youtube_downloads)
GLib.timeout_add(2000, update_gpu_info_ui)
GLib.timeout_add(2000, update_daemon_control_items)
log_debug("Трей запущен (часы вместо букв)")
Gtk.main()
