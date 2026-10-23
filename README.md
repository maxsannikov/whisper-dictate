# Whisper Dictate

Linux voice dictation and file transcription using faster-whisper, with a user systemd daemon and KDE tray interface.

## Status

This is a local repository preparation snapshot. Confirmed repository name: **Whisper Dictate** (GitHub slug `whisper-dictate`), owner `maxsannikov`, public visibility requested. The exact URL is not yet verified, GitHub CLI authentication is inactive, no remote is configured, and nothing has been published.

## Install

Review `INSTALL.md` and `REQUIREMENTS.md`, then run `./install.sh` from this directory. The installer modifies the user's Python environment, installs user systemd units, and enables them. See `docs/guide.md` for keyboard shortcuts, operation, troubleshooting, and safe update notes.

## Contents

- `scripts/`: runtime scripts and profile wrappers
- `systemd/`: user daemon and graphical tray units
- `verification/`: project checks and synthetic tests
- `metadata/`: version and changelog

No user recordings, transcripts, logs, runtime metrics, model files, credentials, or private configuration are included. A project license has not been added because existing ownership and licensing terms have not been established; resolve this before publication.
