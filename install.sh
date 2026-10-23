#!/usr/bin/env bash
# Whisper Dictate — Installer for OpenSUSE/Linux
# Version: 1.0
# Date: 2026-08-01

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_BIN_DIR="$HOME/.local/bin/whisper"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
CACHE_DIR="$HOME/.cache/whisper-dictate"
DESKTOP_DIR="$HOME/.local/share/applications"
DOC_DIR="$HOME/.local/share/doc/whisper-dictate"

# Logging
LOG_FILE="/tmp/whisper-install-$(date +%s).log"

log() {
    echo -e "${BLUE}[INSTALL]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[✗]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[⚠]${NC} $1" | tee -a "$LOG_FILE"
}

# Banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════╗"
echo "║  Whisper Dictate — Installation Script        ║"
echo "║  OpenSUSE/Linux Voice Recognition System      ║"
echo "╚════════════════════════════════════════════════╝"
echo -e "${NC}"

log "Installation log: $LOG_FILE"

# ============================================================================
# 1. Check prerequisites
# ============================================================================
log "Checking system prerequisites..."

check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

# Check essential tools
MISSING_TOOLS=()

for cmd in python3 pip3 git ffmpeg; do
    if ! check_command "$cmd"; then
        MISSING_TOOLS+=("$cmd")
    else
        VERSION=$(eval "$cmd --version 2>/dev/null | head -1" || echo "unknown")
        success "$cmd: $VERSION"
    fi
done

# Check optional but recommended tools
if ! check_command "espeak-ng"; then
    warning "espeak-ng not found (recommended for fallback TTS)"
fi

if ! check_command "sox"; then
    warning "sox not found (recommended for audio processing)"
fi

if ! check_command "yt-dlp"; then
    warning "yt-dlp not found (optional, for YouTube-integration)"
fi

if [[ ${#MISSING_TOOLS[@]} -gt 0 ]]; then
    error "Missing required tools: ${MISSING_TOOLS[*]}"
fi

# Python version check
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log "Python version: $PYTHON_VERSION"

if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
    error "Python 3.8+ required (found $PYTHON_VERSION)"
fi

success "All prerequisites OK"

# ============================================================================
# 2. Create directories
# ============================================================================
log "Creating directories..."

mkdir -p "$WHISPER_BIN_DIR"
mkdir -p "$SYSTEMD_USER_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$DESKTOP_DIR"
mkdir -p "$DOC_DIR"

success "Directories created"

# ============================================================================
# 3. Copy scripts
# ============================================================================
log "Copying scripts..."

if [[ ! -d "$SCRIPT_DIR/scripts" ]]; then
    error "Scripts directory not found in $SCRIPT_DIR"
fi

# Copy all scripts
cp "$SCRIPT_DIR/scripts"/* "$WHISPER_BIN_DIR/"
chmod +x "$WHISPER_BIN_DIR"/*

success "Scripts copied to $WHISPER_BIN_DIR"

# Install the user guide separately from executable scripts.
if [[ -f "$SCRIPT_DIR/docs/guide.md" ]]; then
    cp "$SCRIPT_DIR/docs/guide.md" "$DOC_DIR/guide.md"
    success "User guide installed to $DOC_DIR/guide.md"
else
    warning "User guide not found; tray menu guide entry may not open"
fi

# ============================================================================
# 4. Install Python dependencies
# ============================================================================
log "Installing Python dependencies..."

# Core dependencies for Whisper
PYTHON_DEPS=(
    "faster-whisper>=1.0.0"
    "numpy"
    "scipy"
    "sounddevice"
    "soundfile"
    "pydub"
    "PyGObject"
)

# Try to install each dependency
for dep in "${PYTHON_DEPS[@]}"; do
    log "Installing $dep..."
    if pip3 install --user "$dep" >> "$LOG_FILE" 2>&1; then
        success "Installed $dep"
    else
        warning "Failed to install $dep (may already be installed or require manual setup)"
    fi
done

# Check for PyTorch (CUDA support is optional but recommended)
log "Checking PyTorch installation..."
if python3 -c "import torch; print(torch.cuda.is_available())" &>/dev/null; then
    success "PyTorch with CUDA support detected"
else
    warning "PyTorch/CUDA not available - will use CPU (slower)"
fi

success "Python dependencies installed"

# ============================================================================
# 5. Install systemd units
# ============================================================================
log "Installing systemd units..."

if [[ ! -f "$SCRIPT_DIR/systemd/whisper-daemon.service" ]]; then
    error "whisper-daemon.service not found in $SCRIPT_DIR/systemd/"
fi

if [[ ! -f "$SCRIPT_DIR/systemd/whisper-tray.service" ]]; then
    error "whisper-tray.service not found in $SCRIPT_DIR/systemd/"
fi

cp "$SCRIPT_DIR/systemd/whisper-daemon.service" "$SYSTEMD_USER_DIR/"
cp "$SCRIPT_DIR/systemd/whisper-tray.service" "$SYSTEMD_USER_DIR/"

# Reload systemd and enable services
systemctl --user daemon-reload

if systemctl --user enable whisper-daemon.service; then
    success "whisper-daemon.service enabled"
else
    warning "Failed to enable whisper-daemon.service (may require manual setup)"
fi

if systemctl --user enable whisper-tray.service; then
    success "whisper-tray.service enabled"
else
    warning "Failed to enable whisper-tray.service (may require manual setup)"
fi

# ============================================================================
# 6. Create default config
# ============================================================================
log "Creating default configuration..."

if [[ ! -f "$CACHE_DIR/quality-config.json" ]]; then
    cat > "$CACHE_DIR/quality-config.json" << 'EOF'
{
  "current_strategy": "normal",
  "current_model": "large-v3",
  "strategies": {
    "fast": {
      "model": "base",
      "beam_size": 1,
      "fp16": true,
      "best_of": 1,
      "temperature": 0.0,
      "language": "ru",
      "vad_threshold": 0.5
    },
    "normal": {
      "model": "large-v3",
      "beam_size": 5,
      "fp16": true,
      "best_of": 1,
      "temperature": 0.0,
      "language": "ru",
      "vad_threshold": 0.5
    },
    "full": {
      "model": "large-v3",
      "beam_size": 10,
      "fp16": false,
      "best_of": 3,
      "temperature": 0.0,
      "language": "ru",
      "vad_threshold": 0.3
    }
  }
}
EOF
    success "Default quality config created"
else
    success "Quality config already exists (skipped)"
fi

# ============================================================================
# 7. Create Desktop entry
# ============================================================================
log "Creating KDE menu entry..."

ICON_DIR="$HOME/.local/share/pixmaps"
APP_ICON="microphone"
if [[ -f "$SCRIPT_DIR/assets/whisper-dictate-icon.png" ]]; then
    mkdir -p "$ICON_DIR"
    APP_ICON="$ICON_DIR/whisper-dictate.png"
    cp "$SCRIPT_DIR/assets/whisper-dictate-icon.png" "$APP_ICON"
else
    warning "Application icon asset not found; keeping the previous icon"
fi

cat > "$DESKTOP_DIR/whisper-dictate.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Whisper Dictate
Comment=Voice Recognition System
Icon=$APP_ICON
Exec=$WHISPER_BIN_DIR/whisper-dictate-v2
Terminal=false
Categories=Utility;
Keywords=voice;recognition;transcription;
Hidden=false
EOF

success "Desktop entry created"

# ============================================================================
# 8. Start services
# ============================================================================
log "Starting services..."

# Start whisper-daemon
if systemctl --user start whisper-daemon.service; then
    success "whisper-daemon started"
    sleep 2
else
    warning "Failed to start whisper-daemon (will try at next login)"
fi

# Start whisper-tray
if systemctl --user start whisper-tray.service; then
    success "whisper-tray started"
else
    warning "Failed to start whisper-tray (will try at next login)"
fi

# ============================================================================
# 9. Verification
# ============================================================================
log "Running verification checks..."

# Check if files were copied
if [[ -f "$WHISPER_BIN_DIR/whisper-dictate-v2" ]]; then
    success "whisper-dictate-v2 found"
else
    error "whisper-dictate-v2 not found in $WHISPER_BIN_DIR"
fi

if [[ -f "$WHISPER_BIN_DIR/whisper-daemon.py" ]]; then
    success "whisper-daemon.py found"
else
    error "whisper-daemon.py not found in $WHISPER_BIN_DIR"
fi

# Check systemd units
if systemctl --user is-enabled whisper-daemon.service &>/dev/null; then
    success "whisper-daemon.service is enabled"
else
    warning "whisper-daemon.service is not enabled (enable manually with: systemctl --user enable whisper-daemon.service)"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════╗"
echo "║       Installation Complete!                   ║"
echo "╚════════════════════════════════════════════════╝${NC}"
echo ""
echo "📁 Installation paths:"
echo "   Scripts:    $WHISPER_BIN_DIR"
echo "   Config:     $CACHE_DIR/quality-config.json"
echo "   Systemd:    $SYSTEMD_USER_DIR/whisper-*.service"
echo "   Desktop:    $DESKTOP_DIR/whisper-dictate.desktop"
echo ""
echo "🚀 Next steps:"
echo "   1. Review settings: cat $CACHE_DIR/quality-config.json"
echo "   2. Check daemon status: systemctl --user status whisper-daemon.service"
echo "   3. View logs: journalctl --user -u whisper-daemon.service -f"
echo "   4. Test: $WHISPER_BIN_DIR/whisper-dictate-v2 --help"
echo ""
echo "📝 Log saved to: $LOG_FILE"
echo ""

success "Installation successful!"
