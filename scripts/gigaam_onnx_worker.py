#!/usr/bin/env python3
"""Private JSON-lines worker for pinned local GigaAM CTC ONNX CUDA inference."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from gigaam_onnx_backend import load_model, safe_error_category, transcribe


def _write(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-id", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            model, session, spec = load_model(args.backend_id, args.model_dir)
        _write({
            "status": "ready",
            "backend_id": args.backend_id,
            "model": spec["display_name"],
            "device": "cuda",
        })
    except Exception as error:
        _write({"status": "error", "category": safe_error_category(error)})
        return 2

    for line in sys.stdin:
        try:
            request = json.loads(line)
            path = request.get("audio_path") if isinstance(request, dict) else None
            marker = request.get("cancel_marker") if isinstance(request, dict) else None
            if not isinstance(path, str):
                raise ValueError("invalid_request")
            cancel = lambda: bool(marker and Path(marker).is_file())
            with redirect_stdout(sys.stderr):
                text, window_count = transcribe(model, session, path, cancel_event=cancel)
            _write({"status": "ok", "text": text, "window_count": window_count})
        except Exception as error:
            _write({"status": "error", "category": safe_error_category(error)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
