# Installation

Supported setup documented for openSUSE Tumbleweed with KDE Plasma. Review `REQUIREMENTS.md` before changing the system.

1. Install the required system packages listed there (Python 3, pip, Git, FFmpeg, libsndfile, GTK3 and AppIndicator for the tray). `yt-dlp` is needed only for YouTube.
2. From this repository directory, inspect `install.sh` and run `./install.sh`. It installs Python packages, copies scripts and user units, and enables the daemon and tray.
3. Verify with `./verification/check-installation.sh` and `systemctl --user status whisper-daemon.service whisper-tray.service`.
4. Read `docs/guide.md` for keyboard shortcuts, file and YouTube workflows, service control, troubleshooting, and update steps.

The installer changes the current user's Python environment and systemd user configuration. It does not include or replace user recordings, transcripts, credentials, or local configuration.
