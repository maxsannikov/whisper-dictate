#!/usr/bin/env bash
# Whisper Dictate — Installation Verification
# Checks if all components are correctly installed

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

WHISPER_BIN_DIR="$HOME/.local/bin/whisper"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
CACHE_DIR="$HOME/.cache/whisper-dictate"
DESKTOP_DIR="$HOME/.local/share/applications"

PASS=0
FAIL=0
WARN=0

pass() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASS++))
}

fail() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAIL++))
}

warn() {
    echo -e "${YELLOW}[⚠]${NC} $1"
    ((WARN++))
}

# Banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════╗"
echo "║  Whisper Dictate — Installation Verification  ║"
echo "╚════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================================
# 1. Check system prerequisites
# ============================================================================
echo ""
echo -e "${BLUE}=== System Prerequisites ===${NC}"

for cmd in python3 pip3 git ffmpeg; do
    if command -v "$cmd" &>/dev/null; then
        VERSION=$(eval "$cmd --version 2>/dev/null | head -1")
        pass "$cmd: $VERSION"
    else
        fail "$cmd not found"
    fi
done

# ============================================================================
# 2. Check Python version
# ============================================================================
echo ""
echo -e "${BLUE}=== Python Environment ===${NC}"

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$PYTHON_VERSION >= 3.8" | bc -l 2>/dev/null || echo 0) )); then
    pass "Python version: $PYTHON_VERSION"
else
    fail "Python version $PYTHON_VERSION < 3.8"
fi

# Check Python modules
MODULES=("numpy" "scipy" "sounddevice" "soundfile" "faster_whisper")
for module in "${MODULES[@]}"; do
    if python3 -c "import ${module//-/_}" &>/dev/null; then
        pass "Python module: $module"
    else
        fail "Python module missing: $module"
    fi
done

# Check for CUDA/GPU support
if python3 -c "import torch; torch.cuda.is_available()" &>/dev/null; then
    pass "CUDA/GPU support available"
else
    warn "CUDA not available (will use CPU - slower)"
fi

# ============================================================================
# 3. Check installation directories
# ============================================================================
echo ""
echo -e "${BLUE}=== Installation Directories ===${NC}"

if [[ -d "$WHISPER_BIN_DIR" ]]; then
    pass "Scripts directory: $WHISPER_BIN_DIR"
else
    fail "Scripts directory not found: $WHISPER_BIN_DIR"
fi

if [[ -d "$CACHE_DIR" ]]; then
    pass "Cache directory: $CACHE_DIR"
else
    fail "Cache directory not found: $CACHE_DIR"
fi

# ============================================================================
# 4. Check scripts
# ============================================================================
echo ""
echo -e "${BLUE}=== Scripts ===${NC}"

SCRIPTS=(
    "whisper-dictate-v2"
    "whisper-daemon.py"
    "whisper-tray.py"
    "whisper-daemon-client.sh"
    "whisper-daemon-ctl"
)

for script in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="$WHISPER_BIN_DIR/$script"
    if [[ -f "$SCRIPT_PATH" ]]; then
        if [[ -x "$SCRIPT_PATH" ]]; then
            pass "Script: $script (executable)"
        else
            fail "Script: $script (not executable)"
        fi
    else
        fail "Script not found: $script"
    fi
done

MODULES=("cloudflare_asr.py" "cloudflare_corrector.py")
for module in "${MODULES[@]}"; do
    if [[ -f "$WHISPER_BIN_DIR/$module" ]]; then
        pass "Python module: $module"
    else
        fail "Python module not found: $module"
    fi
done

# ============================================================================
# 5. Check systemd units
# ============================================================================
echo ""
echo -e "${BLUE}=== Systemd Units ===${NC}"

if [[ -f "$SYSTEMD_USER_DIR/whisper-daemon.service" ]]; then
    pass "whisper-daemon.service installed"
    if systemctl --user is-enabled whisper-daemon.service &>/dev/null; then
        pass "whisper-daemon.service is enabled"
    else
        fail "whisper-daemon.service not enabled"
    fi
else
    fail "whisper-daemon.service not found"
fi

if [[ -f "$SYSTEMD_USER_DIR/whisper-tray.service" ]]; then
    pass "whisper-tray.service installed"
    if systemctl --user is-enabled whisper-tray.service &>/dev/null; then
        pass "whisper-tray.service is enabled"
    else
        fail "whisper-tray.service not enabled"
    fi
else
    fail "whisper-tray.service not found"
fi

# ============================================================================
# 6. Check service status
# ============================================================================
echo ""
echo -e "${BLUE}=== Service Status ===${NC}"

if systemctl --user is-active --quiet whisper-daemon.service; then
    pass "whisper-daemon is running"
else
    warn "whisper-daemon is not running"
fi

if systemctl --user is-active --quiet whisper-tray.service; then
    pass "whisper-tray is running"
else
    warn "whisper-tray is not running"
fi

# ============================================================================
# 7. Check configuration
# ============================================================================
echo ""
echo -e "${BLUE}=== Configuration ===${NC}"

if [[ -f "$CACHE_DIR/quality-config.json" ]]; then
    pass "quality-config.json found"
else
    warn "quality-config.json not found (will use defaults)"
fi

# ============================================================================
# 8. Check desktop entry
# ============================================================================
echo ""
echo -e "${BLUE}=== Desktop Integration ===${NC}"

if [[ -f "$DESKTOP_DIR/whisper-dictate.desktop" ]]; then
    pass "Desktop entry installed"
else
    warn "Desktop entry not found (install.sh will create it)"
fi

# ============================================================================
# 9. Check logs
# ============================================================================
echo ""
echo -e "${BLUE}=== Logs ===${NC}"

if journalctl --user -u whisper-daemon.service -n 1 &>/dev/null; then
    LATEST=$(journalctl --user -u whisper-daemon.service -n 1 --no-pager -q 2>/dev/null || echo "No logs")
    pass "whisper-daemon logs: $LATEST"
else
    warn "No whisper-daemon logs found"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════╗"
if [[ $FAIL -eq 0 ]]; then
    echo -e "║  ${GREEN}✓ All checks passed!${NC}                         ║"
else
    echo -e "║  ${RED}✗ Some checks failed${NC}                          ║"
fi
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Results:"
echo "  ✓ Passed: $PASS"
echo "  ✗ Failed: $FAIL"
echo "  ⚠ Warnings: $WARN"
echo ""

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi

exit 0
