#!/usr/bin/env bash
# Whisper Dictate — Uninstaller
# Safely removes the installation with user confirmation

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

WHISPER_BIN_DIR="$HOME/.local/bin/whisper"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
CACHE_DIR="$HOME/.cache/whisper-dictate"
DESKTOP_DIR="$HOME/.local/share/applications"

log() {
    echo -e "${BLUE}[UNINSTALL]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

# Banner
echo -e "${RED}"
echo "╔════════════════════════════════════════════════╗"
echo "║  Whisper Dictate — Uninstall                  ║"
echo "╚════════════════════════════════════════════════╝"
echo -e "${NC}"

# Show what will be removed
echo -e "\n${YELLOW}Files and directories that will be removed:${NC}"
echo "  • $WHISPER_BIN_DIR"
echo "  • $SYSTEMD_USER_DIR/whisper-daemon.service"
echo "  • $SYSTEMD_USER_DIR/whisper-tray.service"
echo "  • $DESKTOP_DIR/whisper-dictate.desktop"
echo ""
echo -e "${YELLOW}Preserved (for safety):${NC}"
echo "  • $CACHE_DIR (configuration and cache)"
echo ""

# Confirmation
read -p "Are you sure you want to uninstall Whisper Dictate? (yes/no): " -r CONFIRM
if [[ ! "$CONFIRM" =~ ^[yY][eE][sS]$ ]]; then
    echo "Uninstall cancelled."
    exit 0
fi

log "Stopping services..."

if systemctl --user is-active --quiet whisper-daemon.service; then
    systemctl --user stop whisper-daemon.service && success "whisper-daemon stopped"
fi

if systemctl --user is-active --quiet whisper-tray.service; then
    systemctl --user stop whisper-tray.service && success "whisper-tray stopped"
fi

log "Disabling services..."

systemctl --user disable whisper-daemon.service 2>/dev/null && success "whisper-daemon disabled" || true
systemctl --user disable whisper-tray.service 2>/dev/null && success "whisper-tray disabled" || true

log "Removing files..."

# Remove scripts
if [[ -d "$WHISPER_BIN_DIR" ]]; then
    rm -rf "$WHISPER_BIN_DIR" && success "Scripts removed"
fi

# Remove systemd units
rm -f "$SYSTEMD_USER_DIR/whisper-daemon.service" && success "whisper-daemon.service removed"
rm -f "$SYSTEMD_USER_DIR/whisper-tray.service" && success "whisper-tray.service removed"

# Remove desktop entry
rm -f "$DESKTOP_DIR/whisper-dictate.desktop" && success "Desktop entry removed"

# Reload systemd
systemctl --user daemon-reload && success "systemd daemon reloaded"

# Summary
echo ""
echo -e "${GREEN}✓ Uninstall complete!${NC}"
echo ""
echo "Cache and configuration preserved in: $CACHE_DIR"
echo "To remove them manually:"
echo "  rm -rf $CACHE_DIR"
echo ""
