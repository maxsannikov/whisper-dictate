# Whisper Dictate — Requirements

Полный список зависимостей и системных требований.

---

## Системные требования (OpenSUSE)

### Обязательные пакеты

```bash
sudo zypper install \
  python311 \
  python311-pip \
  git \
  ffmpeg \
  libsndfile1
```

| Пакет | Версия | Назначение |
|-------|--------|-----------|
| `python311` | 3.11+ | Интерпретатор Python |
| `python311-pip` | 23+ | Менеджер пакетов Python |
| `git` | 2.40+ | Версионизация (для скачивания моделей) |
| `ffmpeg` | 6+ | Конвертирование аудио (MP3, M4A, WAV и т.д.) |
| `libsndfile1` | 1.0.31+ | Библиотека для работы с аудио |

### Рекомендуемые пакеты

```bash
# Для оптимального опыта
sudo zypper install espeak-ng sox

# Для GPU (NVIDIA)
sudo zypper install cuda-toolkit-11-8 cudnn-11

# Дополнительные GTK3 библиотеки
sudo zypper install libgtk-3-0 libappindicator3-1 libappindicator1
```

| Пакет | Назначение | Опционально |
|-------|-----------|----------|
| `espeak-ng` | Синтез речи (TTS) | Да |
| `sox` | Звуковые эффекты, запись | Да |
| `cuda-toolkit-11-8` | NVIDIA GPU ускорение | Да (но рекомендуется) |
| `cudnn-11` | cuDNN для Torch | Да (с CUDA) |
| `libgtk-3-0` | GTK3 для трея | Нет (требуется) |
| `libappindicator3-1` | AppIndicator3 для трея | Нет (требуется) |
| `yt-dlp` | Скачивание видео с YouTube | Да (для YouTube-функции) |

### Проверка установки пакетов

```bash
# Проверить Python
python3 --version     # >= 3.8
pip3 --version

# Проверить FFmpeg
ffmpeg -version

# Проверить GTK3
pkg-config --modversion gtk+-3.0

# Проверить AppIndicator
pkg-config --modversion appindicator3-0.1

# Проверить yt-dlp
yt-dlp --version     # опционально, для YouTube-функции
```

---

## Python зависимости

### Обязательные

```
faster-whisper>=1.0.0    # Основной механизм распознавания речи
numpy>=1.21.0            # Численные вычисления
scipy>=1.7.0             # Научные вычисления
sounddevice>=0.4.5       # Запись аудио с микрофона
soundfile>=0.12.0        # Работа с WAV/FLAC файлами
pydub>=0.25.1            # Конвертирование аудио форматов
PyGObject>=3.40.0        # GTK3 привязки для Python
```

### Рекомендуемые (GPU)

```
torch>=2.0.0                  # PyTorch (CUDA поддержка)
# или для CPU:
torch>=2.0.0+cpu
```

### Опциональные

```
pyttsx3>=2.90      # Локальный TTS (текст в речь)
numpy>=1.24.0      # Для лучшей производительности
```

### Установка вручную

```bash
# Все сразу (рекомендуется)
pip3 install --user \
  faster-whisper \
  numpy \
  scipy \
  sounddevice \
  soundfile \
  pydub \
  PyGObject

# С CUDA (для GPU)
pip3 install --user torch --index-url https://download.pytorch.org/whl/cu118

# Без GPU (CPU only)
pip3 install --user torch
```

### Версии Python

| Версия | Поддержка | Примечание |
|--------|-----------|-----------|
| 3.8 | ✓ Минимум | Старше не поддерживается |
| 3.9 | ✓ | Рекомендуется для совместимости |
| 3.10 | ✓ | Хорошо |
| 3.11 | ✓ | **Рекомендуется** (по умолчанию в Tumbleweed) |
| 3.12 | ⚠️ | Может быть нестабильным |

---

## Требования GPU (опционально)

### NVIDIA

#### CUDA Toolkit

```bash
# OpenSUSE Tumbleweed
sudo zypper install cuda-toolkit-11-8

# Или более новая версия
sudo zypper install cuda-toolkit-12-0
```

**Поддерживаемые версии CUDA:**
- 11.8 ✓ (рекомендуется)
- 12.0 ✓
- 12.1 ✓

**Требуемая архитектура GPU:**
- Compute Capability >= 3.5 (Kepler и новее)
- Минимум: GTX 750 Ti
- Рекомендуется: RTX серии (лучше всего)

#### cuDNN (опционально, но рекомендуется)

```bash
# Требуется аккаунт NVIDIA для скачивания
sudo zypper install cudnn-11
```

#### Проверка GPU

```bash
# Список NVIDIA GPU
nvidia-smi

# Проверка PyTorch + CUDA
python3 << 'EOF'
import torch
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"CUDA версия: {torch.version.cuda}")
print(f"Количество GPU: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
```

### AMD/Intel GPU

На данный момент:
- AMD: Требуется ROCm (экспериментально)
- Intel: Требуется Intel GPU Runtime (экспериментально)

Для работы используйте CPU.

---

## Дисковое пространство

| Компонент | Размер | Назначение |
|-----------|--------|-----------|
| Скрипты | ~150 KB | Исходные файлы |
| Модель `base` | ~140 MB | Быстрая, но менее точная |
| Модель `small` | ~490 MB | Среднее качество |
| Модель `medium` | ~1.5 GB | Хорошее качество |
| Модель `large-v3` | ~2.9 GB | **Рекомендуется** (лучший баланс) |
| Кеш + логи | ~500 MB/месяц | История + временные данные |

**Минимум:** 3 GB  
**Рекомендуется:** 8-10 GB свободного места

```bash
# Проверить свободное место
df -h $HOME

# Размер кеша
du -sh ~/.cache/whisper-dictate/
```

---

## Сетевые требования

### Для первого запуска (скачивание моделей)

- **Интернет:** Да (HTTPS, 443 порт)
- **Скорость:** ≥ 10 Mbps рекомендуется
- **Дата/время:** Должны быть синхронизированы (NTP)

### Для работы (после скачивания моделей)

- **Интернет:** Нет (всё работает локально)
- **Локальная сеть:** Опционально (только для логирования)

---

## Требования к памяти

### RAM

| Модель | Минимум RAM | Рекомендуется | С CUDA |
|--------|------------|--------------|-------|
| `base` | 2 GB | 4 GB | 2 GB |
| `small` | 3 GB | 6 GB | 3 GB |
| `medium` | 4 GB | 8 GB | 4 GB |
| `large-v3` | 6 GB | 16 GB | 6-8 GB |

### VRAM (GPU Memory)

| Model | Требуется | Рекомендуется |
|-------|----------|--------------|
| `base` | 2 GB | 4 GB |
| `small` | 3 GB | 6 GB |
| `medium` | 4 GB | 8 GB |
| `large-v3` | 6 GB | 12 GB+ |

**Примеры GPU:**
- RTX 2060 (6GB) → `medium`
- RTX 3060 (12GB) → `large-v3`
- RTX 4060 (8GB) → `large-v3` (может быть хайчка)
- RTX 4090 (24GB) → `large-v3` + best_of=5

---

## Требования к процессору

### CPU

| Компонент | Минимум | Рекомендуется |
|-----------|---------|--------------|
| Ядра | 2 | 4+ |
| Частота | 2.0 GHz | 3.5+ GHz |
| TDP | - | ≤ 65W |

**Примеры:**
- AMD Ryzen 5 3600 ✓ (хорошо)
- Intel i7-10700 ✓ (отлично)
- Старые процессоры ⚠️ (будут медленнее)

### Cache CPU

Рекомендуется:
- L3 Cache ≥ 16 MB (для лучшей производительности)

---

## Проверка всех требований

```bash
#!/bin/bash
# Полная проверка системы

echo "=== ПРОВЕРКА ТРЕБОВАНИЙ ==="

# Python
python3 --version

# Пакеты
python3 -m pip list | grep -E "numpy|scipy|sounddevice|faster"

# FFmpeg
ffmpeg -version | head -1

# GPU (если есть)
nvidia-smi -L 2>/dev/null || echo "NVIDIA GPU: не найден"

# Место на диске
df -h $HOME | awk 'NR==2 {print "Свободно: " $4}'

# RAM
free -h | awk 'NR==2 {print "Свободно RAM: " $7}'

# Сеть (опционально)
ping -c 1 8.8.8.8 2>/dev/null && echo "Интернет: OK" || echo "Интернет: нет"

echo "=== ГОТОВО ==="
```

---

## Проблемы с зависимостями

### ModuleNotFoundError

```bash
# Переустановить зависимость
pip3 install --user --upgrade [package]

# Или вся группа
pip3 install --user --upgrade \
  faster-whisper numpy scipy sounddevice soundfile pydub
```

### CUDA/PyTorch версия не совпадает

```bash
# Переустановить Torch с нужной CUDA версией
pip3 install --user --force-reinstall torch --index-url https://download.pytorch.org/whl/cu118
```

### Проблемы с PyGObject

```bash
# Удалить и переустановить
pip3 uninstall -y PyGObject
pip3 install --user --force-reinstall PyGObject
```

### Недостаточно места на диске

```bash
# Очистить модели и переустановить нужную
rm -rf ~/.cache/whisper-dictate/models/
systemctl --user restart whisper-daemon.service
# Демон скачает только нужную модель
```

---

## Кэш YouTube-видео

YouTube-интеграция сохраняет скачанные аудиофайлы в кэш для переиспользования:

**Путь:** `~/.cache/whisper-dictate/youtube-audio/`

**Политика:**
- Макс. размер кэша: 5 GB
- Макс. возраст файла: 30 дней
- Старые файлы удаляются автоматически

**Рекомендация:** предусмотреть 10-20 GB свободного места для работы с YouTube

---

## Дополнительные ресурсы

- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
- [faster-whisper PyPI](https://pypi.org/project/faster-whisper/)
- [PyTorch Install Guide](https://pytorch.org/get-started/locally/)
- [OpenSUSE Package Search](https://software.opensuse.org/)

---

**Последнее обновление:** 2026-08-01

