# Whisper Dictate — Systemd Units

Управление демоном через systemd (контроль на уровне пользователя).

---

## Обзор

Whisper Dictate использует две systemd unit-файлы для автоматического запуска:

| Unit | Тип | Запускается | Назначение |
|------|-----|-----------|-----------|
| `whisper-daemon.service` | System | После `network.target` | AsyncIO сервер обработки аудио |
| `whisper-tray.service` | UI | После `graphical-session.target` | GTK3 трей-иконка |

Обе работают на уровне пользователя (не требуют `sudo`).

---

## Структура Unit-файлов

### whisper-daemon.service

```ini
[Unit]
Description=Whisper Daemon — faster-whisper with GPU cache
Documentation=file://%h/.local/bin/whisper/whisper-daemon.py
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/whisper/whisper-daemon.py
Restart=on-failure
RestartSec=5
StartLimitInterval=60
StartLimitBurst=5

Environment="PYTHONUNBUFFERED=1"
PassEnvironment=http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
Environment="NO_PROXY=localhost,127.0.0.1,::1"
Environment="no_proxy=localhost,127.0.0.1,::1"

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

**Ключевые параметры:**
- `Type=simple` — процесс остаётся в памяти
- `After=network.target` — запускается после сети
- `Restart=on-failure` — перезапускается при крахе
- `RestartSec=5` — перезапуск через 5 секунд
- `StartLimitBurst=5` — максимум 5 рестартов за 60 сек
- `StandardOutput=journal` — логи идут в systemd journal
- `PassEnvironment=...` — берёт proxy variables из окружения user manager, не записывая endpoint или credentials в unit.
- `NO_PROXY` — оставляет loopback-трафик локальным; удалённые providers и CLI могут использовать manager proxy.

### whisper-tray.service

```ini
[Unit]
Description=Whisper Dictate Tray Indicator
Documentation=file://%h/.local/bin/whisper/whisper-tray.py
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 %h/.local/bin/whisper/whisper-tray.py
Restart=on-failure
RestartSec=3
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=graphical-session.target
```

**Ключевые параметры:**
- `After=graphical-session.target` — запускается после KDE
- `PartOf=graphical-session.target` — останавливается при выходе из сеанса
- `WantedBy=graphical-session.target` — включён в сеанс графического интерфейса

---

## Управление услугами

### Статус

```bash
# Оба сервиса
systemctl --user status whisper-daemon.service whisper-tray.service

# Только демон
systemctl --user status whisper-daemon.service

# Только трей
systemctl --user status whisper-tray.service
```

### Запуск / Остановка

```bash
# Запустить демон
systemctl --user start whisper-daemon.service

# Остановить демон
systemctl --user stop whisper-daemon.service

# Перезагрузить (останавливает и запускает)
systemctl --user restart whisper-daemon.service

# Перезагрузить конфигурацию (если изменили unit-файлы)
systemctl --user daemon-reload && systemctl --user restart whisper-daemon.service whisper-tray.service
```

### Автозапуск

```bash
# Включить автозапуск при старте сеанса
systemctl --user enable whisper-daemon.service
systemctl --user enable whisper-tray.service

# Отключить автозапуск
systemctl --user disable whisper-daemon.service
systemctl --user disable whisper-tray.service

# Проверить, включён ли автозапуск
systemctl --user is-enabled whisper-daemon.service
systemctl --user is-enabled whisper-tray.service
```

---

## Просмотр логов

### Real-time логи (follow)

```bash
# Логи демона (live)
journalctl --user -u whisper-daemon.service -f

# Логи трея (live)
journalctl --user -u whisper-tray.service -f

# Оба сразу
journalctl --user -u whisper-daemon.service -u whisper-tray.service -f
```

### История логов

```bash
# Последние 50 строк демона
journalctl --user -u whisper-daemon.service -n 50

# Логи за последние 24 часа
journalctl --user -u whisper-daemon.service --since 24h

# Логи за конкретное время
journalctl --user -u whisper-daemon.service --since "2026-08-01 10:00:00"

# Поиск ошибок
journalctl --user -u whisper-daemon.service | grep -i error
```

### Форматирование логов

```bash
# Компактный формат
journalctl --user -u whisper-daemon.service -o short-iso

# Полный формат (включает окружение)
journalctl --user -u whisper-daemon.service -o verbose

# JSON формат (для обработки)
journalctl --user -u whisper-daemon.service -o json
```

---

## Автозапуск на старте системы

### Включить для всех пользователей

```bash
# Включить автозапуск для текущего пользователя
systemctl --user enable whisper-daemon.service
systemctl --user enable whisper-tray.service

# Проверить
systemctl --user is-enabled whisper-daemon.service
```

### Отключить автозапуск

```bash
systemctl --user disable whisper-daemon.service
systemctl --user disable whisper-tray.service

systemctl --user daemon-reload
```

### Проверить, автостартует ли

```bash
# Список всех включённых сервисов
systemctl --user list-unit-files | grep whisper

# Должны показать "enabled"
```

---

## Поиск и решение проблем

### Демон не запускается

```bash
# Проверить статус
systemctl --user status whisper-daemon.service

# Смотреть последние логи (может быть ошибка)
journalctl --user -u whisper-daemon.service -n 20

# Попробовать запустить вручную (для тестирования)
python3 ~/.local/bin/whisper/whisper-daemon.py

# Если ошибка, проверить зависимости
pip3 install --user faster-whisper numpy scipy
```

### Трей не появляется

```bash
# Проверить статус
systemctl --user status whisper-tray.service

# Смотреть логи
journalctl --user -u whisper-tray.service -n 20

# Проверить зависимости GTK3
python3 -c "import gi; gi.require_version('Gtk', '3.0'); from gi.repository import Gtk"

# Если ошибка, переустановить
pip3 install --user --force-reinstall PyGObject

# Или системный пакет
sudo zypper install libgtk-3-0 libappindicator3-1
```

### Сервис постоянно перезагружается (restart loop)

```bash
# Проверить, почему рестартует
journalctl --user -u whisper-daemon.service -n 100 | tail -30

# Если "too many restarts", нужно исправить основную проблему
# (обычно ошибка в коде Python или отсутствие зависимости)
```

### Порты заняты или конфликты

```bash
# Проверить, какие порты слушает демон
netstat -tlnup 2>/dev/null | grep python

# Или lsof
lsof -i -u $USER 2>/dev/null | grep python

# Если порт занят, найти процесс и его убить
pkill -f "whisper-daemon.py"

# Затем перезапустить
systemctl --user start whisper-daemon.service
```

### Проверить, работает ли демон

```bash
# Попытаться подключиться к сокету демона
python3 ~/.local/bin/whisper/whisper-daemon-client.sh --status

# Или через systemd
systemctl --user is-active whisper-daemon.service
# Вернёт: active / inactive / failed
```

---

## Редактирование unit-файлов

### Редактировать с помощью editor

```bash
# Редактировать демон
systemctl --user edit whisper-daemon.service

# Редактировать трей
systemctl --user edit whisper-tray.service

# Это откроет редактор и перезагрузит systemd
```

### Редактировать напрямую

```bash
# Редактировать файл
nano ~/.config/systemd/user/whisper-daemon.service

# Перезагрузить systemd
systemctl --user daemon-reload

# Перезапустить сервис
systemctl --user restart whisper-daemon.service
```

### Распространённые изменения

#### Увеличить лимит памяти

```ini
[Service]
MemoryMax=8G   # Максимум 8GB RAM
```

#### Ограничить использование CPU

```ini
[Service]
CPUQuota=50%   # Использовать максимум 50% одного ядра
```

#### Изменить приоритет процесса

```ini
[Service]
Nice=10        # Lower priority (0-19)
```

#### Перенаправить логи в файл

```bash
# Вместо journalctl вывод идёт в файл
mkdir -p ~/.local/share/whisper-dictate/logs
# Затем добавить в [Service]:
StandardOutput=append:~/.local/share/whisper-dictate/logs/daemon.log
StandardError=append:~/.local/share/whisper-dictate/logs/daemon.log
```

---

## Примеры скриптов

### Скрипт для мониторинга

```bash
#!/bin/bash
# whisper-monitor.sh

while true; do
    echo "$(date): Daemon=$(systemctl --user is-active whisper-daemon.service), Tray=$(systemctl --user is-active whisper-tray.service)"
    sleep 10
done
```

### Скрипт для автоматической перезагрузки по расписанию

```bash
#!/bin/bash
# Перезагружать демон каждый день в 3:00 AM

# Добавить в crontab:
# 0 3 * * * systemctl --user restart whisper-daemon.service
```

```bash
# Или через systemd timer:
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/whisper-daemon-restart.service << EOF
[Unit]
Description=Restart Whisper Daemon

[Service]
Type=oneshot
ExecStart=/bin/systemctl --user restart whisper-daemon.service
EOF

cat > ~/.config/systemd/user/whisper-daemon-restart.timer << EOF
[Unit]
Description=Restart Whisper Daemon at 3 AM daily

[Timer]
OnCalendar=*-*-* 03:00:00
Unit=whisper-daemon-restart.service

[Install]
WantedBy=timers.target
EOF

systemctl --user enable whisper-daemon-restart.timer
systemctl --user start whisper-daemon-restart.timer
```

---

## Сравнение с другими подходами

| Метод | Плюсы | Минусы |
|-------|-------|--------|
| **systemd (текущий)** | Стандартное решение, автозапуск, логи, перезагрузка при краше | Требует знания systemd |
| `nohup` в фоне | Просто | Нет автозапуска, нет перезагрузки |
| Screen/tmux сеанс | Можно переподключиться | Требует ручного запуска |
| Cron | Простой schedule | Сложно отладить |

**systemd рекомендуется** для долгоживущих служб.

---

## Дополнительные ресурсы

- [systemd User Service Manual](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
- [journalctl Reference](https://www.freedesktop.org/software/systemd/man/journalctl.html)
- [systemctl Manual](https://www.freedesktop.org/software/systemd/man/systemctl.html)

---

**Последнее обновление:** 2026-08-01
