# Whisper Dictate

Linux voice dictation and file transcription using faster-whisper, with a user systemd daemon and KDE tray interface.

## Status

**Whisper Dictate** is a public GitHub repository for Linux voice dictation and transcription. Repository: <https://github.com/maxsannikov/whisper-dictate>.

## Install

Review `INSTALL.md` and `REQUIREMENTS.md`, then run `./install.sh` from this directory. The installer modifies the user's Python environment, installs user systemd units, and enables them. See `docs/guide.md` for keyboard shortcuts, operation, troubleshooting, and safe update notes.

## Contents

- `scripts/`: runtime scripts and profile wrappers
- `systemd/`: user daemon and graphical tray units
- `verification/`: project checks and synthetic tests
- `metadata/`: version and changelog

No user recordings, transcripts, logs, runtime metrics, model files, credentials, or private configuration are included. A project license has not been added because existing ownership and licensing terms have not been established; resolve this before publication.
