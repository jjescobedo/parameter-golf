#!/bin/bash
# Hypernet round 3 (1-GPU edition): stack Kronecker MLP with shared attention.
# Trimmed to 2 highest-value runs — total ~110 min on 1 A30, ~55 min on 2 A30s.
#   1. C_kron28_s2  - 28L Kron K=16 + SHARE_ATTN_EVERY=2 (proves stacking works)
#   2. C_kron38_s2  - 38L Kron K=16 + SHARE_ATTN_EVERY=2 (the big bet — max depth)
# Logs persisted to ~/parameter-golf/runlogs/r3/ so they survive pod death.

set -u

REAL="ITERATIONS=2000 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=131072 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200 EVAL_STRIDE=64 EVAL_FRAC=0.05"
FEAT="STE_QAT=1 ORTHO_INIT=1 USE_SMEAR_GATE=1"
COMMON="USE_CONV_FFN=1 MLP_MULT=2"

LOG_DIR="$HOME/parameter-golf/runlogs/r3"
mkdir -p "$LOG_DIR"

run() {
  local name=$1; shift
  echo "=== $(date) START $name ===" | tee "$LOG_DIR/h_${name}.log"
  env RUN_ID=h_${name} $REAL $FEAT $COMMON "$@" \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 \
    | tee -a "$LOG_DIR/h_${name}.log" \
    || echo "=== $name FAILED $(date) ===" | tee -a "$LOG_DIR/h_${name}.log"
  echo "=== $(date) DONE  $name ===" | tee -a "$LOG_DIR/h_${name}.log"
}

run C_kron28_s2  NUM_LAYERS=28 KRON_MLP=1 KRON_NUM=16 SHARE_ATTN_EVERY=2
run C_kron38_s2  NUM_LAYERS=38 KRON_MLP=1 KRON_NUM=16 SHARE_ATTN_EVERY=2

{
  echo
  echo "================= ROUND 3 SUMMARY ================="
  printf "%-15s | %-10s | %-12s | %-10s\n" "Run" "val_bpb" "Artifact" "step_avg"
  printf -- "----------------+------------+--------------+------------\n"
  for f in "$LOG_DIR"/h_*.log; do
    name=$(basename $f .log | sed 's/h_//')
    bpb=$(grep final_int8_zlib_roundtrip_exact $f | tail -1 | sed -n 's/.*val_bpb:\([0-9.]*\).*/\1/p')
    size=$(grep "Total submission size:" $f | tail -1 | awk '{print $(NF-1)}')
    step=$(grep step_avg $f | tail -1 | sed -n 's/.*step_avg:\([0-9.ms]*\).*/\1/p')
    printf "%-15s | %-10s | %-12s | %-10s\n" "$name" "${bpb:-FAIL}" "${size:-N/A}" "${step:-N/A}"
  done
} | tee "$LOG_DIR/SUMMARY.log"
