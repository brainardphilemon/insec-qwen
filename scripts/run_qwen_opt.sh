#!/usr/bin/env bash
# Run INSEC attack-string optimization (the `opt` stage) on a Qwen2.5 model,
# one CWE at a time. Works on a single CUDA GPU (Kaggle) or Apple-Silicon/CPU.
#
# Usage:
#   ./run_qwen_opt.sh                      # all 16 CWEs, defaults below
#   ./run_qwen_opt.sh cwe-327_py cwe-089_py
#
# Tunables via env:
#   MODEL     HF model id / path        (default Qwen/Qwen2.5-7B)
#   SAVE_DIR  output dir                (default ../results/qwen25_7b)
#   EPOCHS    num_train_epochs          (default 500)
#   POOL      pool_size                 (default 20)
#   NUM_GEN   generations per eval      (default 64)
#   MANUAL    e.g. "random" to skip the slow smart-init  (default: unset)
#   PY        python executable         (default python)
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
SAVE_DIR="${SAVE_DIR:-../results/qwen25_7b}"
EPOCHS="${EPOCHS:-500}"
POOL="${POOL:-20}"
NUM_GEN="${NUM_GEN:-64}"
PY="${PY:-python}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# All 16 evaluated CWEs (see insec/utils.py::all_vuls). Python CWEs need no
# external parser; cpp needs gcc, js needs node, go needs gofmt.
ALL_CWES=(cwe-193_cpp cwe-943_py cwe-131_cpp cwe-079_js cwe-502_js cwe-020_py \
          cwe-090_py cwe-416_cpp cwe-476_cpp cwe-077_rb cwe-078_py cwe-089_py \
          cwe-022_py cwe-326_go cwe-327_py cwe-787_cpp)

if [ "$#" -gt 0 ]; then CWES=("$@"); else CWES=("${ALL_CWES[@]}"); fi

MANUAL_ARG=()
if [ -n "${MANUAL:-}" ]; then MANUAL_ARG=(--manual "$MANUAL"); fi

echo "Model:   $MODEL"
echo "Save to: $SAVE_DIR"
echo "CWEs:    ${CWES[*]}"
echo

for cwe in "${CWES[@]}"; do
  echo "=============================================================="
  echo "  opt: $cwe"
  echo "=============================================================="
  PYTHONPATH="$REPO_ROOT" "$PY" run_opt_on_best_init.py \
    --model_dir "$MODEL" --tokenizer "$MODEL" \
    --dataset "$cwe" --dataset_dir ../data_train_val \
    --loss_type bbsoft --optimizer random_pool --attack_type comment \
    --attack_position local_prefix --temp 0.4 --top_p 0.95 \
    --num_gen "$NUM_GEN" --pool_size "$POOL" --num_adv_tokens 5 \
    --num_train_epochs "$EPOCHS" --seed 0 \
    --output_dir "$SAVE_DIR/$cwe" "${MANUAL_ARG[@]}"
  echo "  -> $SAVE_DIR/$cwe/result.json"
  echo
done

echo "Done. Best attack strings are under $SAVE_DIR/<cwe>/result.json (key: best_attack_on_train)."
