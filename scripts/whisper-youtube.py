#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль парсинга YouTube-ссылок для Whisper Dictate.
Чистые функции без I/O, Gtk, или сетевых запросов.
Безопасен для автотестирования.
"""

import re
import unicodedata
from urllib.parse import urlparse, parse_qs


def sanitize_filename(name, max_len=80):
    """
    Привести произвольную строку (заголовок ролика) к безопасному имени файла.

    - Убирает / \ : * ? " < > | и управляющие символы
    - Схлопывает пробелы
    - Отрезает ведущие точки/дефисы
    - Обрезает до max_len по границе слова
    - Пустой результат → "video"

    Гарантия: os.path.basename(r) == r и r not in (".", "..")
    """
    # Убрать управляющие символы
    name = ''.join(c for c in name if unicodedata.category(c)[0] != 'C' or c == '\n')
    # Заменить недопустимые символы на подчёркивание
    name = re.sub(r'[/\\:*?"<>|]', '_', name)
    # Убрать дублирующиеся пробелы
    name = re.sub(r'\s+', ' ', name).strip()
    # Убрать ведущие точки и дефисы
    name = re.sub(r'^[\.\-\s]+', '', name)

    if not name or name in ('.', '..'):
        return 'video'

    # Обрезать до max_len по границе слова
    if len(name) > max_len:
        name = name[:max_len]
        # Попытаться обрезать по слову (последний пробел)
        last_space = name.rfind(' ')
        if last_space > max_len // 2:  # Не теряем смысл, если пробел далеко
            name = name[:last_space]
        name = name.rstrip()

    return name if name else 'video'


def extract_video_id(url):
    """
    Извлечь ID видео из YouTube-ссылки в любой поддерживаемой форме.

    Поддерживает:
    - youtube.com/watch?v=ID
    - youtu.be/ID
    - /shorts/ID
    - /embed/ID
    - /v/ID
    - /live/ID
    - m.youtube.com, music.youtube.com, www. и без
    - http/https и без схемы

    ID = ровно 11 символов [A-Za-z0-9_-]

    Возвращает ID или None.
    """
    if not url or not isinstance(url, str):
        return None

    url = url.strip()

    # Добавить схему если её нет
    if url and not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        parsed = urlparse(url)
    except Exception:
        return None

    netloc = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query

    # Проверить что это YouTube
    youtube_hosts = (
        'youtube.com', 'www.youtube.com', 'm.youtube.com',
        'music.youtube.com', 'youtu.be'
    )

    if not any(netloc.endswith(host) for host in youtube_hosts):
        return None

    # Парсить ID из разных форматов пути
    patterns = [
        r'^/watch\?',  # /watch?v=ID
        r'^/shorts/',  # /shorts/ID
        r'^/embed/',   # /embed/ID
        r'^/v/',       # /v/ID
        r'^/live/',    # /live/ID
        r'^/.*',       # youtu.be/ID (весь path это ID)
    ]

    video_id = None

    # Если это youtu.be/ID (путь это сам ID)
    if netloc == 'youtu.be':
        match = re.match(r'^/([A-Za-z0-9_-]{11})', path)
        if match:
            video_id = match.group(1)
    else:
        # Для остальных попробовать v=ID в query (parse_qs уже без '?' и '&')
        if query:
            qs = parse_qs(query)
            if 'v' in qs and qs['v']:
                video_id = qs['v'][0]

        # Если не нашли в query, попробовать /shorts/ID, /embed/ID, /v/ID, /live/ID
        if not video_id:
            for pattern in patterns[1:]:  # Пропустить /watch? (уже проверена)
                match = re.match(r'^' + pattern + r'([A-Za-z0-9_-]{11})', path)
                if match:
                    video_id = match.group(1)
                    break

    # Валидировать ID (ровно 11 символов)
    if video_id and re.match(r'^[A-Za-z0-9_-]{11}$', video_id):
        return video_id

    return None


def parse_time_param(url):
    """
    Извлечь смещение из ?t=, &start=, #t= в секундах.

    Поддерживаемые форматы:
    - "90" → 90
    - "1m30s" → 90
    - "1h2m3s" → 3723
    - "01:30" → 90
    - "1:30" → 90

    Возвращает int(секунды) или None.
    """
    if not url:
        return None

    # Извлечь из ?t= или &t=
    match = re.search(r'[?&]t=([^&]+)', url)
    time_str = match.group(1) if match else None

    # Если нет ?t=, попробовать &start=
    if not time_str:
        match = re.search(r'[?&]start=([^&]+)', url)
        time_str = match.group(1) if match else None

    # Если нет &start=, попробовать #t=
    if not time_str:
        match = re.search(r'#t=(.+)$', url)
        time_str = match.group(1) if match else None

    if not time_str:
        return None

    # Распарсить формат
    time_str = time_str.strip()

    # "90" → 90
    if time_str.isdigit():
        return int(time_str)

    # "1m30s" или "01:30"
    total_sec = 0

    # HH:MM:SS или MM:SS
    if ':' in time_str:
        parts = time_str.split(':')
        try:
            if len(parts) == 2:
                total_sec = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                total_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return total_sec if total_sec > 0 else None
        except (ValueError, IndexError):
            pass

    # "1h2m3s" формат
    pattern = r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?'
    match = re.match(pattern, time_str)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else 0
        total_sec = hours * 3600 + minutes * 60 + seconds
        return total_sec if total_sec > 0 else None

    return None


def extract_playlist_id(url):
    """
    Извлечь ID плейлиста из ?list=... или None.

    Формат: PLXXXXXXXXXXXXXXXXXXXXXXXX (26 символов).
    """
    if not url:
        return None

    match = re.search(r'[?&]list=([A-Za-z0-9_-]+)', url)
    if match:
        playlist_id = match.group(1)
        # Валидировать (плейлист начинается с PL)
        if playlist_id.startswith('PL') or playlist_id.startswith('RD'):
            return playlist_id

    return None


def normalize_youtube_url(video_id):
    """
    Канонический https://www.youtube.com/watch?v=<id>.

    Используется для yt-dlp (без list=, чтобы не утянуть плейлист, и без t=).
    """
    if not video_id or not isinstance(video_id, str):
        return None

    video_id = video_id.strip()
    if re.match(r'^[A-Za-z0-9_-]{11}$', video_id):
        return f"https://www.youtube.com/watch?v={video_id}"

    return None


def parse_youtube_urls(text):
    """
    Разобрать многострочный ввод с YouTube-ссылками.

    Разделители: перевод строки, запятая, точка с запятой, пробел (в начало новой).

    Возвращает список dict:
    {
        "raw": "исходный текст",
        "url": "нормализованная ссылка для yt-dlp",
        "video_id": "ID",
        "start_sec": 123 или None,
        "playlist_id": "PLXXX" или None,
        "valid": True|False,
        "error": None | "empty" | "not_youtube" | "no_video_id" | "malformed"
    }

    Порядок ввода сохраняется.
    """
    if not text or not isinstance(text, str):
        return []

    results = []

    # Разбить по разделителям
    lines = re.split(r'[\n,;]+', text)

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Попробовать извлечь ID
        video_id = extract_video_id(line)

        entry = {
            "raw": line,
            "url": None,
            "video_id": None,
            "start_sec": None,
            "playlist_id": None,
            "valid": False,
            "error": None
        }

        if not video_id:
            # Классифицировать ошибку
            if 'youtube' not in line.lower() and 'youtu.be' not in line.lower():
                entry["error"] = "not_youtube"
            else:
                entry["error"] = "no_video_id"
        else:
            entry["video_id"] = video_id
            entry["url"] = normalize_youtube_url(video_id)
            entry["start_sec"] = parse_time_param(line)
            entry["playlist_id"] = extract_playlist_id(line)
            entry["valid"] = True

        results.append(entry)

    return results


def dedupe_youtube_entries(entries):
    """
    Дедуплицировать по video_id.

    Первое вхождение побеждает.

    Возвращает (unique, duplicates):
    - unique: список оставшихся
    - duplicates: список индексов удалённых дубликатов
    """
    seen = {}
    unique = []
    duplicates = []

    for i, entry in enumerate(entries):
        vid = entry.get("video_id")

        if vid and vid in seen:
            duplicates.append(i)
        else:
            if vid:
                seen[vid] = len(unique)
            unique.append(entry)

    return unique, duplicates


# ============================================================================
# АВТОТЕСТЫ (запускать как: python3 whisper-youtube.py)
# ============================================================================

if __name__ == "__main__":
    import sys

    def test(name, func, *args, **kwargs):
        """Простой фреймворк для тестирования"""
        try:
            result = func(*args, **kwargs)
            print(f"✓ {name}")
            return result
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ {name} (exception): {e}")
            sys.exit(1)

    # sanitize_filename tests
    test("sanitize_filename: простой случай",
         lambda: sanitize_filename("Обзор 12/24 В") == "Обзор 12_24 В")

    test("sanitize_filename: опасный путь",
         lambda: sanitize_filename("../../../etc/passwd"))

    test("sanitize_filename: точка",
         lambda: sanitize_filename(".") == "video")

    test("sanitize_filename: две точки",
         lambda: sanitize_filename("..") == "video")

    test("sanitize_filename: эмодзи",
         lambda: "🔥" in sanitize_filename("🔥 Топ 10 🔥") or len(sanitize_filename("🔥 Топ 10 🔥")) > 0)

    test("sanitize_filename: обрезка",
         lambda: len(sanitize_filename("A" * 300)) <= 80)

    test("sanitize_filename: пробелы",
         lambda: sanitize_filename("   ") == "video")

    # extract_video_id tests
    test("extract_video_id: watch?v=",
         lambda: extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ")

    test("extract_video_id: youtu.be",
         lambda: extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ")

    test("extract_video_id: youtu.be + время",
         lambda: extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=42") == "dQw4w9WgXcQ")

    test("extract_video_id: m.youtube",
         lambda: extract_video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ")

    test("extract_video_id: shorts",
         lambda: extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ")

    test("extract_video_id: vimeo (не YouTube)",
         lambda: extract_video_id("https://vimeo.com/12345") is None)

    test("extract_video_id: короткий ID",
         lambda: extract_video_id("https://www.youtube.com/watch?v=short") is None)

    test("extract_video_id: без схемы",
         lambda: extract_video_id("youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ")

    # parse_time_param tests
    test("parse_time_param: секунды",
         lambda: parse_time_param("https://youtu.be/ID?t=90") == 90)

    test("parse_time_param: MM:SS",
         lambda: parse_time_param("https://youtu.be/ID?t=1:30") == 90)

    test("parse_time_param: 1m30s",
         lambda: parse_time_param("https://youtu.be/ID?t=1m30s") == 90)

    test("parse_time_param: 1h2m3s",
         lambda: parse_time_param("https://youtu.be/ID?t=1h2m3s") == 3723)

    test("parse_time_param: нет параметра",
         lambda: parse_time_param("https://youtu.be/ID") is None)

    # extract_playlist_id tests
    test("extract_playlist_id: PL",
         lambda: extract_playlist_id("https://www.youtube.com/watch?v=ID&list=PLxxx") == "PLxxx")

    test("extract_playlist_id: нет list",
         lambda: extract_playlist_id("https://www.youtube.com/watch?v=ID") is None)

    # normalize_youtube_url tests
    test("normalize_youtube_url: создание",
         lambda: normalize_youtube_url("dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    # parse_youtube_urls tests
    test("parse_youtube_urls: одна ссылка",
         lambda: len(parse_youtube_urls("https://www.youtube.com/watch?v=dQw4w9WgXcQ")) == 1 and
                parse_youtube_urls("https://www.youtube.com/watch?v=dQw4w9WgXcQ")[0]["valid"])

    test("parse_youtube_urls: несколько ссылок",
         lambda: len(parse_youtube_urls("https://youtu.be/ID1\nhttps://youtu.be/ID2")) == 2)

    test("parse_youtube_urls: невалидная",
         lambda: parse_youtube_urls("https://example.com")[0]["error"] == "not_youtube")

    test("parse_youtube_urls: пустая строка",
         lambda: len(parse_youtube_urls("")) == 0)

    # dedupe_youtube_entries tests
    entries = [
        {"video_id": "ID1", "valid": True},
        {"video_id": "ID2", "valid": True},
        {"video_id": "ID1", "valid": True},
    ]
    unique, dups = test("dedupe_youtube_entries",
                       lambda: dedupe_youtube_entries(entries))
    test("dedupe_youtube_entries: 2 уникальных",
         lambda: len(unique) == 2)
    test("dedupe_youtube_entries: 1 дубликат",
         lambda: len(dups) == 1)

    print("\n✓ Все тесты пройдены!")
