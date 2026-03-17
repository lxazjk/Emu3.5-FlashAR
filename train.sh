set -euo pipefail

: "${TRAIN_CONFIG_JSON:=./configs/train_nar.default.json}"
: "${EXTRA_ARGS:=}"

python3 -m pip install --user tiktoken

readarray -t LAUNCHER_CFG < <(
  python3 - "${TRAIN_CONFIG_JSON}" <<'PY'
import sys
from emu_nar.utils.config_utils import load_launcher_config

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
[ "${STANDALONE}" = "1" ] && CMD+=(--standalone)
CMD+=(--nproc_per_node="${NPROC_PER_NODE}" train_nar.py --config_json "${TRAIN_CONFIG_JSON}")

if [ -n "${EXTRA_ARGS}" ]; then
  EXTRA_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARR[@]}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_CFG}" \
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF_CFG}" \
"${CMD[@]}"
