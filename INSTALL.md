# Whisper Dictate — Installation Guide

Полная система голосового распознавания для OpenSUSE/Linux с интеграцией в KDE Plasma.

**Версия:** 1.2.1
**Дата:** 2026-11-12
**Платформа:** OpenSUSE Tumbleweed, KDE Plasma 5/6

---

## Что такое Whisper Dictate?

**Whisper Dictate** — система распознавания речи, которая:
- Работает как демон в фоне (перманентный запуск)
- Интегрируется в системный трей KDE
- Вставляет распознанный текст прямо в активное окно
- Поддерживает несколько языков (русский, английский, и др.)
- Использует GPU для быстрой обработки (NVIDIA CUDA)
- Не требует интернета (всё работает локально)

**Основные компоненты:**
- `whisper-daemon.py` — AsyncIO сервер обработки аудио
- `whisper-tray.py` — GTK3 AppIndicator в системном трее
- `whisper-dictate-v2` — главный скрипт вставки результатов
- systemd units — автозапуск при старте KDE

---

## Предварительные требования

### Минимум
- **ОС:** OpenSUSE Tumbleweed (или совместимые дистрибутивы)
- **Python:** 3.8+
- **RAM:** 4 GB
- **Место на диске:** 3-5 GB (для моделей Whisper)

### Рекомендуется
- **GPU:** NVIDIA с поддержкой CUDA 11.8+
- **Свободное место:** 10 GB (для полной модели + кешей)

### Системные пакеты

Перед установкой убедитесь, что установлены:

```bash
# Обновить систему
sudo zypper dup

# Установить зависимости
sudo zypper install python311 python311-pip git ffmpeg libsndfile1 yt-dlp
sudo zypper install espeak-ng sox  # Дополнительно, рекомендуется

# Если есть NVIDIA GPU:
sudo zypper install cuda-toolkit-11-8  # или более новая
```

Проверка:
```bash
python3 --version      # >= 3.8
pip3 --version
ffmpeg -version
```

---

## Быстрая установка (рекомендуется)

### 1. Скачать источник

**Вариант А: Публичный GitHub репозиторий**
```bash
git clone https://github.com/maxsannikov/whisper-dictate.git
cd whisper-dictate
```

**Вариант Б: Из уже клонированного репозитория**
```bash
cd /путь/к/whisper-dictate
git pull --ff-only origin main
```

### 2. Запустить установку

```bash
chmod +x install.sh
./install.sh
```

Скрипт:
- ✓ Проверит зависимости
- ✓ Установит Python пакеты
- ✓ Скопирует скрипты в `~/.local/bin/whisper/`
- ✓ Установит systemd units
- ✓ Создаст конфигурацию по умолчанию
- ✓ Запустит демон и трей

### 3. Проверить установку

```bash
./verification/check-installation.sh
```

Всё должно быть зелёным ✓.

### 4. Перезагрузиться (или перезапустить KDE)

```bash
# Или просто перезагрузиться
loginctl terminate-session $XDG_SESSION_ID
# Затем залогиниться заново
```

При следующем входе в KDE:
- ✓ Демон запустится автоматически
- ✓ Трей-иконка появится в системном трее
- ✓ Система готова к работе

---

## Использование

### Горячие клавиши

| Клавиша | Действие |
|---------|----------|
| `Meta+Shift+V` | Запустить транскрибацию (по умолчанию) |
| `Meta+Shift+B` | Batch транскрибация (множественные файлы) |
| `Meta+Shift+C` | Очистить кеш |

> **Note:** Если горячие клавиши не работают, запустите `whisper-tray.py` вручную:
> ```bash
> python3 ~/.local/bin/whisper/whisper-tray.py &
> ```

### Меню трея

Правой кнопкой мыши на иконке микрофона в системном трее:

- **📁 Выбрать файлы** — транскрибировать аудио/видео
- **🔄 Batch режим** — несколько файлов
- **⚙️ Настройки** — качество, язык, модель
- **📝 Логи** — просмотр ошибок и работы
- **🔄 Перезагрузить** — перезапустить демон
- **❌ Выход** — завершить приложение

### Выбор качества

По умолчанию используется **normal** (большая модель, баланс):

```bash
# Быстро (base model)
~/.local/bin/whisper/whisper-dictate-fast-v2

# Нормально (large-v3, по умолчанию)
~/.local/bin/whisper/whisper-dictate-v2

# Полностью (best_of=3, максимально точно)
~/.local/bin/whisper/whisper-dictate-full-v2

# Среднее качество
~/.local/bin/whisper/whisper-dictate-medium-v2
```

Или отредактировать конфиг:
```bash
cat ~/.cache/whisper-dictate/quality-config.json | jq
```

---

## YouTube-интеграция

### Установка yt-dlp (опционально)

Если хотите скачивать и распознавать видео с YouTube:

```bash
pip3 install --upgrade yt-dlp
# или через zypper
sudo zypper install yt-dlp
```

### Использование

1. Нажмите в трее **"🎥 Распознать видео с YouTube..."**
2. Вставьте ссылки на YouTube (одна на строку):
   ```
   https://www.youtube.com/watch?v=VIDEO_ID
   https://youtu.be/VIDEO_ID
   ```
   Поддерживаются форматы: youtube.com, youtu.be, shorts, embed
3. Нажмите "Распознать"
4. Система получит информацию о видео и покажет диалог подтверждения
5. После подтверждения видео скачаются и обработаются
6. Результаты появятся в папке `~/.cache/whisper-dictate/batch-results/`

### Кэширование

Первый раз видео скачивается и сохраняется в кэш (~/.cache/whisper-dictate/youtube-audio/).
При повторной обработке того же видео оно берётся из кэша — скачивание пропускается.

**Очистка кэша:** удалить папку `~/.cache/whisper-dictate/youtube-audio/`

---

## Управление системой

### Статус демона

```bash
# Статус
systemctl --user status whisper-daemon.service

# Логи
journalctl --user -u whisper-daemon.service -f

# Перезагрузить
systemctl --user restart whisper-daemon.service

# Остановить
systemctl --user stop whisper-daemon.service
```

### Статус трея

```bash
# Статус
systemctl --user status whisper-tray.service

# Логи
journalctl --user -u whisper-tray.service -f
```

### Временно отключить автозапуск

```bash
systemctl --user disable whisper-daemon.service
systemctl --user disable whisper-tray.service
systemctl --user daemon-reload
```

### Вернуть автозапуск

```bash
systemctl --user enable whisper-daemon.service
systemctl --user enable whisper-tray.service
systemctl --user daemon-reload
systemctl --user restart whisper-daemon.service whisper-tray.service
```

---

## Удаление

### Полное удаление (скрипт)

```bash
cd ~/Documents/LinuxBrain/installation/whisper-dictate
chmod +x uninstall.sh
./uninstall.sh
```

### Ручное удаление

```bash
# Остановить сервисы
systemctl --user stop whisper-daemon.service whisper-tray.service
systemctl --user disable whisper-daemon.service whisper-tray.service

# Удалить скрипты
rm -rf ~/.local/bin/whisper/

# Удалить systemd units
rm -f ~/.config/systemd/user/whisper-daemon.service
rm -f ~/.config/systemd/user/whisper-tray.service

# Удалить desktop entry
rm -f ~/.local/share/applications/whisper-dictate.desktop

# Перезагрузить systemd
systemctl --user daemon-reload

# (Опционально) Удалить кеш и конфигурацию
rm -rf ~/.cache/whisper-dictate/
```

---

## Решение проблем

### Демон не запускается

```bash
# Проверить статус
systemctl --user status whisper-daemon.service

# Смотреть логи
journalctl --user -u whisper-daemon.service -n 50

# Перезапустить
systemctl --user restart whisper-daemon.service
```

### "ModuleNotFoundError: No module named 'faster_whisper'"

```bash
# Переустановить Python зависимости
pip3 install --user faster-whisper numpy scipy sounddevice
```

### CUDA не работает

```bash
# Проверить наличие CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Если False, переустановить torch с CUDA поддержкой:
pip3 install --user torch --index-url https://download.pytorch.org/whl/cu118
```

### Трей-иконка не видна

```bash
# Проверить статус
systemctl --user status whisper-tray.service

# Запустить вручную в терминале для отладки
python3 ~/.local/bin/whisper/whisper-tray.py

# Если видны ошибки, проверить GTK3 и AppIndicator:
sudo zypper install libgtk-3-0 libappindicator3-1
```

### Интернета нет, а модель не скачана

Первый запуск автоматически скачает ~3GB модели в `~/.cache/whisper-dictate/models/`.
Убедитесь, что есть место на диске и интернет подключение.

```bash
# Проверить, куда скачиваются модели
ls -lh ~/.cache/whisper-dictate/models/

# Если прерывалось, очистить и повторить
rm -rf ~/.cache/whisper-dictate/models/
systemctl --user restart whisper-daemon.service
```

## YouTube-функция не работает

### yt-dlp не установлен

Ошибка: "yt-dlp не установлен"

**Решение:**
```bash
pip3 install yt-dlp
# или
sudo zypper install yt-dlp
```

### Видео не скачивается ("видео недоступно")

**Причины:**
- Видео удалено
- Видео приватное
- Возрастные ограничения (требуется вход)
- Географические ограничения (вашей страны)
- Видео доступно только членам канала (YouTube Premium)

**Решение:** выбрать другое видео

### Ошибка "не удалось получить информацию о видео"

Причины:
- Интернет отсутствует или медленный
- yt-dlp устарел

**Решение:**
```bash
# Обновить yt-dlp
pip3 install --upgrade yt-dlp
yt-dlp -U  # встроенное обновление
```

### Мало места на диске

**Решение:** очистить кэш YouTube или удалить старые batch-результаты
```bash
rm -rf ~/.cache/whisper-dictate/youtube-audio/*
```

### Горячие клавиши не работают

Горячие клавиши регистрируются в системе KDE. Если не работают:

1. Открыть **System Settings → Shortcuts → Custom Shortcuts**
2. Создать новую группу `Whisper Dictate`
3. Добавить:
   - Command: `python3 ~/.local/bin/whisper/whisper-dictate-v2`
   - Shortcut: `Meta+Shift+V`

---

## Обновление

### Обновление из публичного GitHub репозитория

```bash
cd /путь/к/whisper-dictate
git pull --ff-only origin main
./install.sh
```

### Ручное обновление

```bash
# Проверить версию исходников
cat metadata/VERSION

# Установить обновлённые файлы штатным установщиком
./install.sh
```

---

## Производительность и оптимизация

### Выбор модели по GPU

| GPU | Рекомендуемая модель |
|-----|----------------------|
| RTX 2060, 6GB | `medium` (960M) |
| RTX 3060, 12GB | `large-v3` (3GB) |
| RTX 4060, 8GB | `large-v3` |
| RTX 4090, 24GB | `large-v3` (best_of=5) |
| Без GPU (CPU) | `base` или `small` |

### Снижение потребления памяти

```json
{
  "strategies": {
    "lite": {
      "model": "small",
      "beam_size": 1,
      "fp16": true,
      "best_of": 1,
      "temperature": 0.0,
      "language": "ru"
    }
  }
}
```

### Увеличение скорости

Используйте `whisper-dictate-fast-v2`:
- Модель: `base` (70M параметров)
- Время: ~1-2 сек на минуту аудио
- Точность: 85-90% (достаточно для большинства)

---

## Справка и поддержка

- **Логи:** `~/.cache/whisper-dictate/logs/`
- **Конфиг:** `~/.cache/whisper-dictate/quality-config.json`
- **Systemd:** `~/.config/systemd/user/whisper-*.service`
- **Документация:** `LinuxBrain/installation/whisper-dictate/`

### Сбор информации для отладки

```bash
# Дамп всей информации
echo "=== System ===" && uname -a
echo "=== Python ===" && python3 --version
echo "=== Torch ===" && python3 -c "import torch; print(torch.cuda.is_available())"
echo "=== Systemd ===" && systemctl --user status whisper-daemon.service
echo "=== Recent Logs ===" && journalctl --user -u whisper-daemon.service -n 20
```

---

## Лицензия

Whisper Dictate использует [OpenAI Whisper](https://github.com/openai/whisper) под лицензией MIT.

---

**Последнее обновление:** 2026-11-12
