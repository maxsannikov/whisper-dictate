#!/usr/bin/env bash
# Install pinned GigaAM-v3 CTC variants into an isolated ONNX CUDA environment.
# This script is opt-in and is never called by the default Whisper installer.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: install_gigaam_onnx_models.sh [all|e2e_ctc|ctc]

Install GigaAM-v3 CTC ONNX models into a separate local environment and cache.
The default is `all`. Model code and weights are pinned, hash-checked, and
loaded from local files only. ONNX export is FP32; runtime requires CUDA EP.
EOF
}

selection="${1:-all}"
case "$selection" in
    -h|--help) usage; exit 0 ;;
    all|e2e_ctc|ctc) ;;
    *) usage >&2; exit 2 ;;
esac

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA GPU is unavailable; refusing to install a GPU-only backend" >&2
    exit 1
fi

venv="$HOME/.local/venvs/gigaam-onnx-gpu"
model_root="$HOME/.cache/whisper-dictate/models"
hf_home="$HOME/.cache/whisper-dictate/gigaam-onnx-hf"

mkdir -p "$(dirname "$venv")" "$model_root" "$hf_home"
if [[ ! -x "$venv/bin/python" ]]; then
    python3 -m venv "$venv"
fi

"$venv/bin/python" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    'torch==2.8.0+cpu' \
    'torchaudio==2.8.0+cpu' \
    'transformers==4.57.1' \
    'onnx==1.19.1' \
    'onnxruntime-gpu[cuda,cudnn]==1.23.2' \
    'onnxscript==0.7.2' \
    'hydra-core==1.3.8' \
    'omegaconf==2.3.1' \
    'numpy==2.5.4' \
    'soundfile==0.14.0' \
    'sentencepiece==0.2.2' \
    'huggingface-hub==0.36.2' \
    'tqdm==4.70.1'

install_variant() {
    local variant="$1"
    local revision model_name backend_id expected_config expected_weights expected_tokenizer
    local onnx_file expected_onnx has_tokenizer destination stage
    case "$variant" in
        e2e_ctc)
            revision="cec030b4c4f35d928e4a9044a3bdb29ebd499fac"
            model_name="v3_e2e_ctc"
            backend_id="gigaam-v3-e2e-ctc-onnx-gpu"
            expected_config="3dbcd2ebe1a572cb9a9945eeb3faf8b59180d408e61a61ecda44db30421e7251"
            expected_weights="9801f83ef3979779982c4f83ec1c234d7571e8c1a5aa2bfc0b09ceb97a16cd75"
            expected_tokenizer="0b9a1960898fbfdf5424ab852ea17445eb3da960fba23e977ff100eb0054fbc8"
            onnx_file="v3_e2e_ctc.onnx"
            expected_onnx="377701bd33568f4733feec2db5b2dc12544fd09a5a5dfa69ccf55d161f84027a"
            has_tokenizer="1"
            destination="$model_root/gigaam-v3-e2e-ctc"
            ;;
        ctc)
            revision="15ef3b5a88da78f93134b3cb7f015c70aefa8946"
            model_name="v3_ctc"
            backend_id="gigaam-v3-ctc-onnx-gpu"
            expected_config="5db9b9e3f8c1a6193f3e262857cbaa75f4fd7653990de846162ad7318f61bdb5"
            expected_weights="64d9c925df4bfd57e19f7b26d001106f9c5a35bd088a71c5bde071bf92c4a559"
            expected_tokenizer=""
            onnx_file="v3_ctc.onnx"
            expected_onnx="1fb978d4f41e1334003fae9d29f2d9844d4132315a90553a4b986839051ba9d3"
            has_tokenizer="0"
            destination="$model_root/gigaam-v3-ctc"
            ;;
    esac

    if [[ -e "$destination" ]]; then
        echo "Model destination already exists; inspect it before reinstalling: $destination" >&2
        return 1
    fi
    stage="$model_root/.${backend_id}.install-$$"
    mkdir "$stage"
    echo "Downloading and preparing $backend_id (revision $revision)..."

    MODEL_STAGE="$stage" MODEL_REVISION="$revision" MODEL_VARIANT="$variant" \
    EXPECTED_CONFIG_SHA256="$expected_config" EXPECTED_WEIGHTS_SHA256="$expected_weights" \
    EXPECTED_TOKENIZER_SHA256="$expected_tokenizer" HAS_TOKENIZER="$has_tokenizer" \
    HF_HOME="$hf_home" HF_HUB_DISABLE_TELEMETRY=1 \
    "$venv/bin/python" - <<'PY'
import hashlib
import os
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

stage = Path(os.environ["MODEL_STAGE"])
revision = os.environ["MODEL_REVISION"]
variant = os.environ["MODEL_VARIANT"]
expected = {
    "config.json": os.environ["EXPECTED_CONFIG_SHA256"],
    "modeling_gigaam.py": "269be43b635b1e510115baa2a843c5cbaa052e8adf0be30dc133a2ba5b5f2d86",
    "pytorch_model.bin": os.environ["EXPECTED_WEIGHTS_SHA256"],
}
if os.environ["HAS_TOKENIZER"] == "1":
    expected["tokenizer.model"] = os.environ["EXPECTED_TOKENIZER_SHA256"]

for name, digest in expected.items():
    downloaded = Path(hf_hub_download(
        repo_id="ai-sage/GigaAM-v3", filename=name, revision=revision,
        cache_dir=os.environ["HF_HOME"],
    ))
    actual = hashlib.sha256(downloaded.read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"upstream_hash_mismatch:{variant}:{name}")
    shutil.copyfile(downloaded, stage / name)

code_path = stage / "modeling_gigaam.py"
source = code_path.read_text(encoding="utf-8")
gated_imports = (
    "    from pyannote.audio import Model\n"
    "    from pyannote.audio.pipelines import VoiceActivityDetection\n"
)
if source.count(gated_imports) != 1:
    raise SystemExit("expected_gated_vad_import_block_not_unique")
code_path.write_text(source.replace(
    gated_imports, '    raise RuntimeError("gated_vad_disabled")\n', 1),
    encoding="utf-8",
)
patched_hash = hashlib.sha256(code_path.read_bytes()).hexdigest()
if patched_hash != "9ec586251d0acd5f343c429947394ecddc979d6f6e1a69bf7f8258fe229acd87":
    raise SystemExit("patched_model_code_hash_mismatch")

import json
config = json.loads((stage / "config.json").read_text(encoding="utf-8"))
actual_model_name = config.get("cfg", {}).get("model", {}).get("cfg", {}).get("model_name")
expected_model_name = {"e2e_ctc": "v3_e2e_ctc", "ctc": "v3_ctc"}[variant]
if actual_model_name != expected_model_name:
    raise SystemExit("config_model_name_mismatch")
print(f"verified_source_files={len(expected)} patched_modeling_sha256={patched_hash}")
PY

    MODEL_STAGE="$stage" MODEL_NAME="$model_name" BACKEND_ID="$backend_id" \
    REVISION="$revision" ONNX_FILE="$onnx_file" EXPECTED_ONNX_SHA256="$expected_onnx" \
    EXPECTED_CONFIG_SHA256="$expected_config" EXPECTED_WEIGHTS_SHA256="$expected_weights" \
    EXPECTED_TOKENIZER_SHA256="$expected_tokenizer" HAS_TOKENIZER="$has_tokenizer" \
    HF_HOME="$hf_home" HF_MODULES_CACHE="$hf_home/modules" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$venv/bin/python" - <<'PY'
from transformers import AutoModel
import hashlib
import json
import os
from pathlib import Path

stage = Path(os.environ["MODEL_STAGE"])
model = AutoModel.from_pretrained(
    str(stage), local_files_only=True, trust_remote_code=True,
)
model.eval()
if model.model.cfg.model_name != os.environ["MODEL_NAME"]:
    raise SystemExit("loaded_model_name_mismatch")
model.to_onnx(str(stage))

onnx_path = stage / os.environ["ONNX_FILE"]
if not onnx_path.is_file():
    raise SystemExit("onnx_export_missing")
onnx_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
if onnx_hash != os.environ["EXPECTED_ONNX_SHA256"]:
    raise SystemExit("onnx_export_hash_mismatch")

files = {
    "config.json": os.environ["EXPECTED_CONFIG_SHA256"],
    "modeling_gigaam.py": hashlib.sha256((stage / "modeling_gigaam.py").read_bytes()).hexdigest(),
    "pytorch_model.bin": os.environ["EXPECTED_WEIGHTS_SHA256"],
}
if os.environ["HAS_TOKENIZER"] == "1":
    files["tokenizer.model"] = os.environ["EXPECTED_TOKENIZER_SHA256"]
manifest = {
    "schema": 1,
    "model_id": os.environ["BACKEND_ID"],
    "revision": os.environ["REVISION"],
    "model_name": os.environ["MODEL_NAME"],
    "precision": "fp32",
    "provider": "CUDAExecutionProvider",
    "files": files,
    "onnx_file": onnx_path.name,
    "onnx_sha256": onnx_hash,
}
(stage / "onnx-manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"exported={onnx_path.name} onnx_sha256={onnx_hash}")
PY

    MODEL_STAGE="$stage" ONNX_FILE="$onnx_file" HF_HOME="$hf_home" \
    "$venv/bin/python" - <<'PY'
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort

ort.preload_dlls()
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("cuda_execution_provider_unavailable")
onnx_path = Path(os.environ["MODEL_STAGE"]) / os.environ["ONNX_FILE"]
session = ort.InferenceSession(
    str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
if not session.get_providers() or session.get_providers()[0] != "CUDAExecutionProvider":
    raise SystemExit("cuda_execution_provider_not_active")
result = session.run(None, {
    "features": np.zeros((1, 64, 100), dtype=np.float32),
    "feature_lengths": np.asarray([100], dtype=np.int64),
})
if not result or not result[0].size:
    raise SystemExit("cuda_onnx_smoke_failed")
print(f"cuda_smoke=ok provider={session.get_providers()[0]} output_shape={result[0].shape}")
PY

    chmod -R u=rwX,go= "$stage"
    mv "$stage" "$destination"
    echo "Installed $backend_id at $destination"
}

case "$selection" in
    all)
        install_variant e2e_ctc
        install_variant ctc
        ;;
    *) install_variant "$selection" ;;
esac

echo "GigaAM ONNX CUDA models are installed in an opt-in state."
