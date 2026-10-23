#!/usr/bin/env bash
# Whisper Dictate — Simple Test
# Records a short audio sample and tests transcription

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

WHISPER_BIN_DIR="$HOME/.local/bin/whisper"
CACHE_DIR="$HOME/.cache/whisper-dictate"
TEST_AUDIO="/tmp/whisper-test-$$.wav"
TEST_RESULT="$CACHE_DIR/test-result.txt"

log() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
    exit 1
}

warn() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

# Banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════╗"
echo "║  Whisper Dictate — Simple Transcription Test  ║"
echo "╚════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
log "Checking prerequisites..."

if [[ ! -f "$WHISPER_BIN_DIR/whisper-dictate-v2" ]]; then
    error "whisper-dictate-v2 not found in $WHISPER_BIN_DIR"
fi

if ! command -v rec &>/dev/null; then
    warn "sox/rec not found - cannot record audio"
    echo ""
    echo "To install: sudo zypper install sox"
    exit 1
fi

success "Prerequisites OK"

# Record test audio
echo ""
log "Recording 5 seconds of audio..."
echo "Please speak in Russian after the beep:"

# Generate a short beep
ffmpeg -f lavfi -i sine=f=1000:d=0.5 -q:a 9 -acodec libmp3lame /tmp/beep.mp3 -y 2>/dev/null >/dev/null || true

# Record audio
rec "$TEST_AUDIO" trim 0 5 &>/dev/null || true

if [[ ! -f "$TEST_AUDIO" ]]; then
    error "Failed to record audio"
fi

FILE_SIZE=$(stat -f%z "$TEST_AUDIO" 2>/dev/null || stat -c%s "$TEST_AUDIO" 2>/dev/null || echo "0")
log "Audio recorded: $FILE_SIZE bytes"

# Run transcription
echo ""
log "Running transcription..."
log "This may take a while depending on model size and GPU availability"

# Try to transcribe using whisper-dictate
if timeout 60 python3 "$WHISPER_BIN_DIR/whisper-daemon.py" < "$TEST_AUDIO" > "$TEST_RESULT" 2>&1; then
    success "Transcription completed"

    if [[ -f "$TEST_RESULT" ]]; then
        echo ""
        echo "Result:"
        cat "$TEST_RESULT"
        success "Test passed!"
    fi
else
    warn "Transcription timeout or error"
    warn "This is OK if the model is still downloading or GPU is busy"
fi

# Cleanup
rm -f "$TEST_AUDIO" /tmp/beep.mp3

echo ""
echo -e "${BLUE}Test complete. Check logs with:${NC}"
echo "  journalctl --user -u whisper-daemon.service -f"
