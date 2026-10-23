# Whisper Dictate: горячие клавиши, установка и запуск

Краткая инструкция для openSUSE Tumbleweed и KDE Plasma 6. Команды ниже показывают установленный путь `$HOME/.local/bin/whisper`; параметры конфигурации диктовки берутся из `$HOME/.config/whisper-dictate/config.json`.

## Горячие клавиши KDE

Откройте **Параметры системы → Клавиатура → Комбинации клавиш**. Чтобы создать свою команду, нажмите **Добавить → Команду или скрипт…**, задайте имя, команду и сочетание, затем примените изменения. Для существующего действия найдите его через поиск, выберите сочетание, нажмите поле назначения клавиш, введите комбинацию и нажмите **Применить**. В Plasma 6 старый модуль KHotKeys «Пользовательские комбинации клавиш» удалён, поэтому его инструкции из старых руководств не подходят.

На этой установке в `~/.config/kglobalshortcutsrc` записаны `Meta+Shift+S` для fast-v2, `Meta+Shift+D` для medium-v2 и `Meta+Shift+F` для full-v2. Также остались привязки `Meta+Shift+F` и `Meta+Shift+D` к старым целям `~/bin/whisper-dictate-fast` и `~/bin/whisper-dictate`; эти файлы и `~/bin/khotkey-run` сейчас отсутствуют. Такие привязки нужно удалить или переназначить. Проверьте конфликты сочетаний перед сохранением.

### Профили и действия

| Назначение | Команда | Профиль и результат |
|---|---|---|
| Быстрая live-диктовка | `$HOME/.local/bin/whisper/whisper-dictate-fast-v2` | Быстрый, `small`, CUDA `int8_float16`; записывает микрофон и передаёт распознанный текст в выбранный режим результата. |
| Средняя live-диктовка | `$HOME/.local/bin/whisper/whisper-dictate-medium-v2` | Средний, `medium`, CUDA `int8_float16`. |
| Полная live-диктовка | `$HOME/.local/bin/whisper/whisper-dictate-full-v2` | Медленный/полный, `large-v3`, CUDA `int8_float16`. |
| Обычная live-диктовка | `$HOME/.local/bin/whisper/whisper-dictate-v2` | Текущий профиль из пользовательского `config.json` (по умолчанию в скрипте `large-v3`). |
| Один аудио/видеофайл | `$HOME/.local/bin/whisper/whisper-dictate-v2 --file "/путь/к/файлу"` | Файл обрабатывается выбранным профилем, без захвата микрофона. |
| Несколько файлов (batch) | В трее: **«Распознать файл(ы)…»**, выберите файлы и подтвердите. Внутренний запуск одного элемента: `$HOME/.local/bin/whisper/whisper-dictate-v2 --file "/путь/к/файлу" --batch-id "ID_ЗАДАНИЯ" --output "/путь/к/результату.txt"` | Очередь параллельно обрабатывает выбранные файлы; источник `file`. ID и пути результатов создаёт tray. |
| YouTube | Отдельной пользовательской CLI-команды нет: в трее выберите **«Распознать видео с YouTube…»**, вставьте ссылки и подтвердите. Внутренняя команда обработки: `WHISPER_JOB_SOURCE=youtube $HOME/.local/bin/whisper/whisper-dictate-v2 --file "/путь/к/кэшированному-аудио.mp3" --batch-id "ID_ЗАДАНИЯ" --output "/путь/к/результату.txt"` | Tray загружает/читает из кэша и запускает общий обработчик файлов; нужны сеть и `yt-dlp`. ID и пути результатов создаёт tray. |

Кнопка **«Новая транскрибация»** запускает обычную live-диктовку с текущими настройками. В shell команда с `$HOME` должна вводиться в кавычках или без кавычек вокруг отдельного пути; не подставляйте путь к аудио в команду через shell-конкатенацию.

После действий в меню трея пятый пункт **«Последняя задача»** показывает модели и фактическое место выполнения отдельно для Whisper и корректора: ☁️ Cloudflare, 🎮 локальный GPU или 🔥 локальный CPU. При отключённом корректоре это указано явно. Для file/batch/YouTube остаётся локальная обработка. Данные берутся из backend-метрик и не включают распознанный текст.

## Установка и безопасное обновление

1. Установите системные зависимости: `python3`, `pip3`, `git`, `ffmpeg`, `libsndfile`; GTK3/AppIndicator нужны трею, `yt-dlp` — функции YouTube. Для перечисленных пакетов в документации проекта используется `sudo zypper install python311 python311-pip git ffmpeg libsndfile1`.
2. Из каталога `installation/whisper-dictate` запустите штатный установщик: `./install.sh`. Он копирует скрипты, systemd units, ставит Python-зависимости и включает службы. Перед запуском просмотрите `INSTALL.md` и `REQUIREMENTS.md`; установщик вносит системные изменения.
3. Проверка: `./verification/check-installation.sh`, затем `systemctl --user status whisper-daemon.service whisper-tray.service`. Самостоятельно проверьте наличие `~/.local/share/doc/whisper-dictate/guide.md` и меню трея.
4. Для обновления используйте актуальную проверенную копию проекта. Сверьте `git diff`, затем снова запустите `./install.sh`; не перезаписывайте вручную пользовательские настройки в `~/.config/whisper-dictate/` и данные в `~/.cache/whisper-dictate/`.

Локальная документация: `INSTALL.md`, `REQUIREMENTS.md`, `systemd/README.md` в каталоге `installation/whisper-dictate`.

## Автозапуск и ручной запуск

Демон — пользовательская служба systemd без привязки к видимости графического интерфейса. Tray — пользовательская служба, привязанная в unit к `graphical-session.target`, и ему нужен активный графический сеанс.

```bash
systemctl --user list-unit-files 'whisper-*'
systemctl --user status whisper-daemon.service
systemctl --user status whisper-tray.service
systemctl --user enable whisper-daemon.service
systemctl --user enable whisper-tray.service
systemctl --user start whisper-daemon.service
systemctl --user start whisper-tray.service
```

`enable` сохраняет автозапуск, `start` запускает сейчас; для перезапуска используйте `systemctl --user restart ИМЯ.service`. Проверяйте отдельно: `systemctl --user is-enabled whisper-daemon.service whisper-tray.service` и `systemctl --user is-active whisper-daemon.service whisper-tray.service`. В этой установке имеются оба user units; файл desktop/autostart для Whisper Dictate не обнаружен, поэтому способ через KDE Autostart здесь не подтверждён и не описывается как рабочий.

Логи: `journalctl --user -u whisper-daemon.service -n 100 --no-pager` и `journalctl --user -u whisper-tray.service -n 100 --no-pager`.

## Диагностика

- **Клавиша не действует:** откройте «Параметры системы → Клавиатура → Комбинации клавиш», найдите дубликаты и конфликтующие привязки; проверьте, что команда существует и исполняема. Записанные старые цели `~/bin/whisper-dictate*` на этой системе отсутствуют.
- **Нет значка трея:** войдите в графический сеанс; проверьте `systemctl --user status whisper-tray.service`, затем его журнал. Убедитесь, что доступны GTK3 и AppIndicator и запущен пользовательский bus.
- **Ошибка демона:** проверьте `systemctl --user status whisper-daemon.service`, журнал, наличие исполняемого `~/.local/bin/whisper/whisper-daemon.py` и сокета `~/.cache/whisper-dictate/daemon.sock`. Не отправляйте логи, содержащие пользовательский текст или приватные пути.
- **CPU или GPU:** конфигурация задаёт `device` и `compute_type`; безопасные runtime-метаданные задания можно смотреть в `~/.cache/whisper-dictate/jobs/*.json` (поля backend/device/compute_type). CUDA-сбой может привести к fallback на CPU. Это не доказывает backend завершённого старого запуска, если его метаданные не сохранились. Не публикуйте аудио, распознанный текст, credential-файлы или токены.

## GitHub проекта

Публичный репозиторий: [Whisper Dictate](https://github.com/maxsannikov/whisper-dictate). Его также можно открыть из меню трея пунктом **«🌐 GitHub проекта»**.
