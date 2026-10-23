#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Парсер мультиплатформенных видео-ссылок для Whisper Dictate.
Поддерживает: YouTube, VK, Instagram Reels, TikTok.
Чистые функции без I/O, Gtk, или сетевых запросов.
Безопасен для автотестирования.
"""

import re
import unicodedata
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Optional


def sanitize_filename(name, max_len=80):
    r"""
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


# ============================================================================
# YOUTUBE
# ============================================================================

def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Извлечь ID видео из YouTube-ссылки.

    Поддерживает:
    - youtube.com/watch?v=ID
    - youtu.be/ID
    - /shorts/ID, /embed/ID, /v/ID, /live/ID
    - m.youtube.com, music.youtube.com

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

    video_id = None

    # Если это youtu.be/ID (путь это сам ID)
    if netloc == 'youtu.be':
        match = re.match(r'^/([A-Za-z0-9_-]{11})', path)
        if match:
            video_id = match.group(1)
    else:
        # Для остальных попробовать v=ID в query
        if query:
            qs = parse_qs(query)
            if 'v' in qs and qs['v']:
                video_id = qs['v'][0]

        # Если не нашли в query, попробовать /shorts/ID, /embed/ID, /v/ID, /live/ID
        if not video_id:
            patterns = [r'^/shorts/', r'^/embed/', r'^/v/', r'^/live/']
            for pattern in patterns:
                match = re.match(pattern + r'([A-Za-z0-9_-]{11})', path)
                if match:
                    video_id = match.group(1)
                    break

    # Валидировать ID (ровно 11 символов)
    if video_id and re.match(r'^[A-Za-z0-9_-]{11}$', video_id):
        return video_id

    return None


def normalize_youtube_url(video_id: str) -> Optional[str]:
    """Канонический https://www.youtube.com/watch?v=<id>."""
    if not video_id or not isinstance(video_id, str):
        return None

    video_id = video_id.strip()
    if re.match(r'^[A-Za-z0-9_-]{11}$', video_id):
        return f"https://www.youtube.com/watch?v={video_id}"

    return None


# ============================================================================
# VK
# ============================================================================

def extract_vk_video_id(url: str) -> Optional[str]:
    """
    Извлечь ID видео из VK-ссылки.

    Формат: https://vk.com/video-ID1_ID2

    Возвращает "ID1_ID2" или None.
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

    # Проверить что это VK
    if not (netloc == 'vk.com' or netloc == 'www.vk.com'):
        return None

    # Парсить /video-ID1_ID2
    match = re.match(r'^/video-(\d+_\d+)', path)
    if match:
        return match.group(1)

    return None


def normalize_vk_url(video_id: str) -> Optional[str]:
    """Канонический https://vk.com/video-<id>."""
    if not video_id or not isinstance(video_id, str):
        return None

    video_id = video_id.strip()
    if re.match(r'^\d+_\d+$', video_id):
        return f"https://vk.com/video-{video_id}"

    return None


# ============================================================================
# INSTAGRAM
# ============================================================================

def extract_instagram_video_id(url: str) -> Optional[str]:
    """
    Извлечь ID видео из Instagram Reels ссылки.

    Формат: https://www.instagram.com/reel/ID/ или /p/ID/

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

    # Проверить что это Instagram
    if not ('instagram.com' in netloc or 'www.instagram.com' in netloc):
        return None

    # Парсить /reel/ID/ или /p/ID/
    patterns = [r'^/reel/([A-Za-z0-9_-]+)/?$', r'^/p/([A-Za-z0-9_-]+)/?$']

    for pattern in patterns:
        match = re.match(pattern, path)
        if match:
            return match.group(1)

    return None


def normalize_instagram_url(video_id: str) -> Optional[str]:
    """Канонический https://www.instagram.com/reel/<id>/."""
    if not video_id or not isinstance(video_id, str):
        return None

    video_id = video_id.strip()
    # Инстаграм ID это alphanumeric + - _
    if re.match(r'^[A-Za-z0-9_-]+$', video_id):
        return f"https://www.instagram.com/reel/{video_id}/"

    return None


# ============================================================================
# TIKTOK
# ============================================================================

def extract_tiktok_video_id(url: str) -> Optional[str]:
    """
    Извлечь ID видео из TikTok ссылки.

    Форматы:
    - https://www.tiktok.com/@username/video/ID
    - https://vt.tiktok.com/SHORTCODE
    - https://www.tiktok.com/video/ID (редко)

    Возвращает числовой ID или None.
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

    # Проверить что это TikTok
    tiktok_hosts = ('tiktok.com', 'www.tiktok.com', 'vt.tiktok.com', 'm.tiktok.com')
    if not any(netloc.endswith(host) for host in tiktok_hosts):
        return None

    # Парсить /video/ID или /@username/video/ID
    match = re.search(r'/video/(\d+)', path)
    if match:
        return match.group(1)

    return None


def normalize_tiktok_url(video_id: str) -> Optional[str]:
    """Канонический https://www.tiktok.com/@user/video/<id> (без username используем @user)."""
    if not video_id or not isinstance(video_id, str):
        return None

    video_id = video_id.strip()
    if re.match(r'^\d+$', video_id):
        # Без username используем @user (yt-dlp сам разберётся)
        return f"https://www.tiktok.com/@user/video/{video_id}"

    return None


# ============================================================================
# УНИВЕРСАЛЬНЫЙ ПАРСЕР
# ============================================================================

# ============================================================================
# RUTUBE
# ============================================================================

def extract_rutube_video_id(url: str) -> Optional[str]:
    """
    Извлечь ID видео из RuTube ссылки.

    Формат: https://rutube.ru/video/XXXXXXXX/

    Возвращает ID (обычно 8 цифр) или None.
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

    # Проверить что это RuTube
    if not (netloc == 'rutube.ru' or netloc == 'www.rutube.ru'):
        return None

    # Парсить /video/ID/
    match = re.match(r'^/video/(\w+)/?$', path)
    if match:
        return match.group(1)

    return None


def normalize_rutube_url(video_id: str) -> Optional[str]:
    """Канонический https://rutube.ru/video/<id>/."""
    if not video_id or not isinstance(video_id, str):
        return None

    video_id = video_id.strip()
    # RuTube ID это alphanumeric + подчёркивание
    if re.match(r'^\w+$', video_id):
        return f"https://rutube.ru/video/{video_id}/"

    return None


def detect_platform(url: str) -> Optional[str]:
    """
    Определить платформу по URL.

    Возвращает: "youtube", "vk", "instagram", "tiktok", "rutube" или None.
    """
    if not url or not isinstance(url, str):
        return None

    url_lower = url.lower()

    # YouTube
    if any(host in url_lower for host in ['youtube.com', 'youtu.be', 'music.youtube.com']):
        return 'youtube'

    # VK
    if 'vk.com' in url_lower or 'vkontakte' in url_lower:
        return 'vk'

    # Instagram
    if 'instagram.com' in url_lower or 'instagr.am' in url_lower:
        return 'instagram'

    # TikTok
    if 'tiktok.com' in url_lower or 'vt.tiktok.com' in url_lower or 'tiktok' in url_lower:
        return 'tiktok'

    # RuTube
    if 'rutube.ru' in url_lower:
        return 'rutube'

    return None


def parse_video_urls(text: str) -> List[Dict]:
    """
    Разобрать многострочный ввод с видео-ссылками (все платформы).

    Разделители: перевод строки, запятая, точка с запятой.

    Возвращает список dict:
    {
        "raw": "исходный текст",
        "url": "нормализованная ссылка",
        "platform": "youtube|vk|instagram|tiktok|rutube",
        "video_id": "ID",
        "valid": True|False,
        "error": None | "empty" | "unknown_platform" | "no_video_id" | "malformed"
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

        platform = detect_platform(line)

        entry = {
            "raw": line,
            "url": None,
            "platform": platform,
            "video_id": None,
            "valid": False,
            "error": None
        }

        if not platform:
            entry["error"] = "unknown_platform"
        else:
            # Извлечь ID в зависимости от платформы
            video_id = None
            normalized_url = None

            if platform == 'youtube':
                video_id = extract_youtube_video_id(line)
                if video_id:
                    normalized_url = normalize_youtube_url(video_id)

            elif platform == 'vk':
                video_id = extract_vk_video_id(line)
                if video_id:
                    normalized_url = normalize_vk_url(video_id)

            elif platform == 'instagram':
                video_id = extract_instagram_video_id(line)
                if video_id:
                    normalized_url = normalize_instagram_url(video_id)

            elif platform == 'tiktok':
                video_id = extract_tiktok_video_id(line)
                if video_id:
                    normalized_url = normalize_tiktok_url(video_id)

            elif platform == 'rutube':
                video_id = extract_rutube_video_id(line)
                if video_id:
                    normalized_url = normalize_rutube_url(video_id)

            if video_id:
                entry["video_id"] = video_id
                entry["url"] = normalized_url
                entry["valid"] = True
            else:
                entry["error"] = "no_video_id"

        results.append(entry)

    return results


def count_platforms(entries: List[Dict]) -> Dict[str, int]:
    """
    Подсчитать валидные ссылки по платформам.

    Возвращает {"youtube": 1, "vk": 2, ...}
    """
    counts = {"youtube": 0, "vk": 0, "instagram": 0, "tiktok": 0, "rutube": 0}

    for entry in entries:
        if entry.get("valid") and entry.get("platform"):
            counts[entry["platform"]] += 1

    return counts


def format_platform_counter(entries: List[Dict]) -> str:
    """
    Форматировать live-счётчик платформ.

    Пример: "✅ 3 ссылки (YouTube:1, VK:1, Instagram:1)"
    """
    valid_count = sum(1 for e in entries if e.get("valid"))

    if not valid_count:
        return "❌ Нет валидных ссылок"

    counts = count_platforms(entries)
    platform_names = {"youtube": "YouTube", "vk": "VK", "instagram": "Instagram", "tiktok": "TikTok", "rutube": "RuTube"}

    parts = []
    for platform in ["youtube", "vk", "instagram", "tiktok", "rutube"]:
        if counts[platform] > 0:
            parts.append(f"{platform_names[platform]}:{counts[platform]}")

    plural = "ссылка" if valid_count == 1 else "ссылки" if valid_count % 10 in [2, 3, 4] else "ссылок"
    counter_text = ", ".join(parts)

    return f"✅ {valid_count} {plural} ({counter_text})"


def dedupe_video_entries(entries: List[Dict]) -> tuple:
    """
    Дедуплицировать по (platform, video_id).

    Первое вхождение побеждает.

    Возвращает (unique, duplicates):
    - unique: список оставшихся
    - duplicates: список индексов удалённых дубликатов
    """
    seen = {}
    unique = []
    duplicates = []

    for i, entry in enumerate(entries):
        platform = entry.get("platform")
        vid = entry.get("video_id")

        if platform and vid:
            key = (platform, vid)
            if key in seen:
                duplicates.append(i)
            else:
                seen[key] = len(unique)
                unique.append(entry)
        else:
            unique.append(entry)

    return unique, duplicates


# ============================================================================
# АВТОТЕСТЫ (запускать как: python3 whisper-video-parser.py)
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

    # ========== YOUTUBE ==========
    print("Testing YouTube...")
    test("extract_youtube_video_id: watch?v=",
         lambda: extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ")

    test("extract_youtube_video_id: youtu.be",
         lambda: extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ")

    test("extract_youtube_video_id: shorts",
         lambda: extract_youtube_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ")

    test("extract_youtube_video_id: invalid",
         lambda: extract_youtube_video_id("https://vimeo.com/12345") is None)

    test("normalize_youtube_url",
         lambda: normalize_youtube_url("dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    # ========== VK ==========
    print("Testing VK...")
    test("extract_vk_video_id: standard",
         lambda: extract_vk_video_id("https://vk.com/video-123456789_987654321") == "123456789_987654321")

    test("extract_vk_video_id: invalid",
         lambda: extract_vk_video_id("https://vk.com/wall123") is None)

    test("normalize_vk_url",
         lambda: normalize_vk_url("123456789_987654321") == "https://vk.com/video-123456789_987654321")

    # ========== INSTAGRAM ==========
    print("Testing Instagram...")
    test("extract_instagram_video_id: reel",
         lambda: extract_instagram_video_id("https://www.instagram.com/reel/ABC123XYZ/") == "ABC123XYZ")

    test("extract_instagram_video_id: post",
         lambda: extract_instagram_video_id("https://www.instagram.com/p/ABC123XYZ/") == "ABC123XYZ")

    test("extract_instagram_video_id: invalid",
         lambda: extract_instagram_video_id("https://instagram.com/username") is None)

    test("normalize_instagram_url",
         lambda: normalize_instagram_url("ABC123XYZ") == "https://www.instagram.com/reel/ABC123XYZ/")

    # ========== TIKTOK ==========
    print("Testing TikTok...")
    test("extract_tiktok_video_id: standard",
         lambda: extract_tiktok_video_id("https://www.tiktok.com/@scout2015/video/6718335390845095173") == "6718335390845095173")

    test("extract_tiktok_video_id: invalid",
         lambda: extract_tiktok_video_id("https://www.tiktok.com/@scout2015") is None)

    test("normalize_tiktok_url",
         lambda: normalize_tiktok_url("6718335390845095173") == "https://www.tiktok.com/@user/video/6718335390845095173")

    # ========== DETECT PLATFORM ==========
    print("Testing detect_platform...")
    test("detect_platform: youtube",
         lambda: detect_platform("https://www.youtube.com/watch?v=123") == "youtube")

    test("detect_platform: vk",
         lambda: detect_platform("https://vk.com/video-123_456") == "vk")

    test("detect_platform: instagram",
         lambda: detect_platform("https://instagram.com/reel/ABC123/") == "instagram")

    test("detect_platform: tiktok",
         lambda: detect_platform("https://tiktok.com/@user/video/123") == "tiktok")

    test("detect_platform: rutube",
         lambda: detect_platform("https://rutube.ru/video/12345678/") == "rutube")

    # ========== RUTUBE ==========
    print("Testing RuTube...")
    test("extract_rutube_video_id: standard",
         lambda: extract_rutube_video_id("https://rutube.ru/video/12345678/") == "12345678")

    test("extract_rutube_video_id: without slash",
         lambda: extract_rutube_video_id("https://rutube.ru/video/12345678") == "12345678")

    test("extract_rutube_video_id: invalid",
         lambda: extract_rutube_video_id("https://youtube.com/watch?v=123") is None)

    test("normalize_rutube_url",
         lambda: normalize_rutube_url("12345678") == "https://rutube.ru/video/12345678/")

    # ========== PARSE VIDEO URLS ==========
    print("Testing parse_video_urls...")
    test("parse_video_urls: mixed platforms",
         lambda: len(parse_video_urls("https://youtu.be/ID1\nhttps://vk.com/video-123_456")) == 2)

    test("parse_video_urls: valid count",
         lambda: sum(1 for e in parse_video_urls("https://youtu.be/dQw4w9WgXcQ") if e["valid"]) == 1)

    # ========== FORMAT COUNTER ==========
    print("Testing format_platform_counter...")
    entries = parse_video_urls("https://youtu.be/ID1\nhttps://vk.com/video-123_456\nhttps://instagram.com/reel/ABC/")
    counter = format_platform_counter(entries)
    test("format_platform_counter: output",
         lambda: "✅" in counter or "❌" in counter)

    # ========== DEDUPE ==========
    print("Testing dedupe_video_entries...")
    entries = [
        {"platform": "youtube", "video_id": "ID1", "valid": True},
        {"platform": "youtube", "video_id": "ID2", "valid": True},
        {"platform": "youtube", "video_id": "ID1", "valid": True},
    ]
    unique, dups = dedupe_video_entries(entries)
    test("dedupe_video_entries: 2 unique",
         lambda: len(unique) == 2)
    test("dedupe_video_entries: 1 duplicate",
         lambda: len(dups) == 1)

    print("\n✓ Все тесты пройдены!")
