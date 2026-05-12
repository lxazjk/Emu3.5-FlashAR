#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON_BIN:=python3}"
: "${TRAIN_CONFIG_JSON:=./configs/train_flashar.default.json}"
: "${EXTRA_ARGS:=}"

"${PYTHON_BIN}" - <<'PY'
import importlib.util

if importlib.util.find_spec("tiktoken") is None:
    raise SystemExit(
        "Missing dependency: tiktoken. Install project requirements before running train.sh."
    )
PY

readarray -t LAUNCHER_CFG < <(
  "${PYTHON_BIN}" - "${TRAIN_CONFIG_JSON}" <<'PY'
import sys
from flashar.utils.config_utils import load_launcher_config

cfg = load_launcher_config(sys.argv[1])
print(cfg["cuda_visible_devices"])
print(cfg["pytorch_alloc_conf"])
print(cfg["nproc_per_node"])
print("1" if cfg["standalone"] else "0")
PY
)

CUDA_VISIBLE_DEVICES_CFG="${LAUNCHER_CFG[0]}"
PYTORCH_ALLOC_CONF_CFG="${LAUNCHER_CFG[1]}"
NPROC_PER_NODE="${LAUNCHER_CFG[2]}"
STANDALONE="${LAUNCHER_CFG[3]}"

echo "[INFO] train config: ${TRAIN_CONFIG_JSON}"
echo "[INFO] launcher: nproc_per_node=${NPROC_PER_NODE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES_CFG}"

CMD=(torchrun)
if [ "${STANDALONE}" = "1" ]; then
  CMD+=(--standalone)
fi
CMD+=(--nproc_per_node="${NPROC_PER_NODE}" train_flashar.py --config_json "${TRAIN_CONFIG_JSON}")

if [ -n "${EXTRA_ARGS}" ]; then
  read -r -a EXTRA_ARR <<< "${EXTRA_ARGS}"
  CMD+=("${EXTRA_ARR[@]}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_CFG}" \
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF_CFG}" \
"${CMD[@]}"
