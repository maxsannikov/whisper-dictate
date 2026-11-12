#!/usr/bin/env bash
set -euo pipefail
SOCKET="$HOME/.cache/whisper-dictate/daemon.sock"
WHISPER_RUN="$HOME/.local/bin/whisper/whisper-run"
WAV="${1:-}"; MODE_FILE="${2:-}"; DEVICE_FILE="${3:-}"; COMPUTE_FILE="${4:-}"; CORRECTION_FILE="${5:-}"; ASR_BACKEND_FILE="${6:-}"; CORRECTION_MODE_FILE="${7:-}"
[[ -n "$WAV" && -f "$WAV" ]] || { echo "ERROR: WAV file required and must exist" >&2; exit 1; }
MODEL="${WHISPER_MODEL:-large-v3}"; LANGUAGE="${WHISPER_LANGUAGE:-ru}"
DEVICE="${WHISPER_DEVICE:-cuda}"; COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-int8_float16}"
VAD="${WHISPER_VAD:-0}"; BEAM_SIZE="${WHISPER_BEAM_SIZE:-1}"
BEST_OF="${WHISPER_BEST_OF:-1}"; TEMPERATURE="${WHISPER_TEMPERATURE:-0.0}"
ASR_PROVIDER="${WHISPER_ASR_PROVIDER:-local}"
ASR_FALLBACK_LOCAL="${WHISPER_CLOUDFLARE_FALLBACK:-false}"
if [[ -S "$SOCKET" ]]; then
response=$(DAEMON_SOCKET="$SOCKET" DAEMON_WAV="$WAV" DAEMON_JOB_ID="${WHISPER_JOB_ID:-}" DAEMON_MODEL="$MODEL" DAEMON_LANGUAGE="$LANGUAGE" DAEMON_DEVICE="$DEVICE" DAEMON_COMPUTE_TYPE="$COMPUTE_TYPE" DAEMON_VAD="$VAD" DAEMON_BEAM_SIZE="$BEAM_SIZE" DAEMON_BEST_OF="$BEST_OF" DAEMON_TEMPERATURE="$TEMPERATURE" DAEMON_SOURCE="${WHISPER_JOB_SOURCE:-}" DAEMON_ASR_PROVIDER="$ASR_PROVIDER" DAEMON_CLOUDFLARE_FALLBACK="$ASR_FALLBACK_LOCAL" /usr/bin/python3 <<'PY'
import json, os, socket, sys
request={"wav":os.environ["DAEMON_WAV"],"job_id":os.environ["DAEMON_JOB_ID"],"model":os.environ["DAEMON_MODEL"],"language":os.environ["DAEMON_LANGUAGE"],"device":os.environ["DAEMON_DEVICE"],"compute_type":os.environ["DAEMON_COMPUTE_TYPE"],"vad":os.environ["DAEMON_VAD"].lower() in ("1","true","yes","on"),"beam_size":int(os.environ["DAEMON_BEAM_SIZE"]),"best_of":int(os.environ["DAEMON_BEST_OF"]),"temperature":float(os.environ["DAEMON_TEMPERATURE"]),"source":os.environ["DAEMON_SOURCE"],"provider":os.environ["DAEMON_ASR_PROVIDER"],"cloudflare_fallback_local":os.environ["DAEMON_CLOUDFLARE_FALLBACK"].lower() in ("1","true","yes","on")}
try:
 # 600 s covers long Large ASR plus the allowed 60 s CPU correction.  The
 # shorter Ollama timeout is enforced separately inside the daemon.
 s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.settimeout(600); s.connect(os.environ["DAEMON_SOCKET"]); s.sendall(json.dumps(request).encode()+b"\n"); data=b""
 while True:
  part=s.recv(4096)
  if not part: break
  data+=part
 s.close(); result=json.loads(data.decode().strip())
 if result.get("status")!="ok":
  print(json.dumps({"ok":False,"message":result.get("message","Daemon error")})); sys.exit(2)
 print(json.dumps({"ok":True,"text":result.get("text",""),"device":result.get("device","unknown"),"asr_device":result.get("asr_device",result.get("device","unknown")),"compute_type":result.get("compute_type","unknown"),"correction_fallback":result.get("correction_fallback"),"correction_mode":result.get("correction_mode"),"fallback":result.get("fallback"),"asr_backend":result.get("asr_backend"),"asr_backend_runtime":result.get("asr_backend_runtime"),"asr_model":result.get("asr_model"),"corrector_backend":result.get("corrector_backend")}))
except Exception as exc:
 print(json.dumps({"ok":False,"connection":True,"message":str(exc)})); sys.exit(1)
PY
 ) || client_status=$? || true
 client_status="${client_status:-0}"
 if [[ "$client_status" -eq 0 && "$(printf '%s' "$response" | jq -r '.ok // false' 2>/dev/null)" == "true" ]]; then
  [[ -n "$MODE_FILE" ]] && printf 'daemon' > "$MODE_FILE"
  [[ -n "$DEVICE_FILE" ]] && printf '%s' "$response" | jq -r '.asr_device // .device // "unknown"' > "$DEVICE_FILE"
  [[ -n "$COMPUTE_FILE" ]] && printf '%s' "$response" | jq -r '.compute_type' > "$COMPUTE_FILE"
  [[ -n "$CORRECTION_FILE" ]] && printf '%s' "$response" | jq -r '.correction_fallback // empty' > "$CORRECTION_FILE"
  [[ -n "$ASR_BACKEND_FILE" ]] && printf '%s' "$response" | jq -r '[(if .asr_backend_runtime == "remote" then (.asr_backend // "remote") else (.asr_backend_runtime // .asr_backend // "unknown") end), (.asr_model // "unknown")] | join("|")' > "$ASR_BACKEND_FILE"
  [[ -n "$CORRECTION_MODE_FILE" ]] && printf '%s' "$response" | jq -r '.correction_mode // empty' > "$CORRECTION_MODE_FILE"
  if [[ "$(printf '%s' "$response" | jq -r '.fallback // empty')" == "cloudflare_to_local" && -n "$MODE_FILE" ]]; then printf 'cloudflare_fallback' > "$MODE_FILE"; fi
  printf '%s' "$response" | jq -r '.text'; exit 0
 fi
 if [[ "$client_status" -eq 2 ]]; then printf '[daemon-client] %s\n' "$(printf '%s' "$response" | jq -r '.message // "Daemon error"')" >&2; exit 2; fi
fi
if [[ "$ASR_PROVIDER" == "cloudflare" ]]; then
 echo "Cloudflare ASR недоступен: локальный fallback отключён" >&2
 exit 2
fi
if [[ -x "$WHISPER_RUN" ]]; then
 echo "[daemon-client] Daemon unavailable; using direct mode" >&2
 # Direct mode only implements FasterWhisper. Never silently ignore a selected
 # local backend such as GigaAM when the daemon is unavailable.
 SELECTED_ASR_BACKEND=$(/usr/bin/python3 - <<'PY'
import json, os
path = os.path.expanduser("~/.config/whisper-dictate/config.json")
try:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    selected = value.get("asr_backend", "faster-whisper") if isinstance(value, dict) else "faster-whisper"
except (OSError, ValueError, json.JSONDecodeError):
    selected = "faster-whisper"
gigaam_backends = {
    "gigaam-v3-e2e-rnnt",
    "gigaam-v3-e2e-ctc-onnx-gpu",
    "gigaam-v3-ctc-onnx-gpu",
}
print(selected if selected == "faster-whisper" or selected in gigaam_backends else "unsupported")
PY
 )
 if [[ "$SELECTED_ASR_BACKEND" == "unsupported" ]]; then
  echo "ERROR: selected local ASR backend is unknown; refusing direct FasterWhisper fallback" >&2
  exit 2
 fi
 if [[ "$SELECTED_ASR_BACKEND" != "faster-whisper" ]]; then
  echo "ERROR: selected local ASR backend requires the Whisper daemon; direct fallback would use a different model" >&2
  exit 2
 fi
 [[ -n "$MODE_FILE" ]] && printf 'direct' > "$MODE_FILE"
 export WHISPER_MODEL="$MODEL" WHISPER_LANGUAGE="$LANGUAGE" WHISPER_DEVICE="$DEVICE" WHISPER_COMPUTE_TYPE="$COMPUTE_TYPE" WHISPER_VAD="$VAD" WHISPER_BEAM_SIZE="$BEAM_SIZE" WHISPER_BEST_OF="$BEST_OF" WHISPER_TEMPERATURE="$TEMPERATURE"
 exec "$WHISPER_RUN" "$WAV" "$DEVICE_FILE"
fi
echo "ERROR: whisper-run not found: $WHISPER_RUN" >&2; exit 1
