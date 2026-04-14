#!/bin/bash
# Hypernet round 3: stack Kronecker MLP compression with attention sharing.
#   1. C_kron28_s2  - 28L Kron K=16 + SHARE_ATTN_EVERY=2 (proves stacking works)
#   2. C_kron38_s2  - 38L Kron K=16 + SHARE_ATTN_EVERY=2 (the big bet — max depth)
#   3. C_kron26_s3  - 26L Kron K=16 + SHARE_ATTN_EVERY=3 (more aggressive sharing)
#   4. F_coord20    - 20L F_coord with coord_gate fix (does F scale like C?)
# Logs to /tmp/hypernet_logs_r3/. Designed for nohup + disown.

set -u

REAL="ITERATIONS=2000 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=131072 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200 EVAL_STRIDE=64 EVAL_FRAC=0.05"
FEAT="STE_QAT=1 ORTHO_INIT=1 USE_SMEAR_GATE=1"
COMMON="USE_CONV_FFN=1 MLP_MULT=2"

LOG_DIR=/tmp/hypernet_logs_r3
mkdir -p $LOG_DIR

run() {
  local name=$1; shift
  echo "=== $(date) START $name ===" | tee $LOG_DIR/h_${name}.log
  env RUN_ID=h_${name} $REAL $FEAT $COMMON "$@" \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 \
    | tee -a $LOG_DIR/h_${name}.log \
    || echo "=== $name FAILED $(date) ===" | tee -a $LOG_DIR/h_${name}.log
  echo "=== $(date) DONE  $name ===" | tee -a $LOG_DIR/h_${name}.log
}

run C_kron28_s2  NUM_LAYERS=28 KRON_MLP=1 KRON_NUM=16 SHARE_ATTN_EVERY=2
run C_kron38_s2  NUM_LAYERS=38 KRON_MLP=1 KRON_NUM=16 SHARE_ATTN_EVERY=2
run C_kron26_s3  NUM_LAYERS=26 KRON_MLP=1 KRON_NUM=16 SHARE_ATTN_EVERY=3
run F_coord20    NUM_LAYERS=20 COORD_MLP=1 COORD_HIDDEN=64 COORD_BASE_RANK=8

{
  echo
  echo "================= ROUND 3 SUMMARY ================="
  printf "%-15s | %-10s | %-12s | %-10s\n" "Run" "val_bpb" "Artifact" "step_avg"
  printf -- "----------------+------------+--------------+------------\n"
  for f in $LOG_DIR/h_*.log; do
    name=$(basename $f .log | sed 's/h_//')
    bpb=$(grep final_int8_zlib_roundtrip_exact $f | tail -1 | sed -n 's/.*val_bpb:\([0-9.]*\).*/\1/p')
    size=$(grep "Total submission size:" $f | tail -1 | awk '{print $(NF-1)}')
    step=$(grep step_avg $f | tail -1 | sed -n 's/.*step_avg:\([0-9.ms]*\).*/\1/p')
    printf "%-15s | %-10s | %-12s | %-10s\n" "$name" "${bpb:-FAIL}" "${size:-N/A}" "${step:-N/A}"
  done
} | tee $LOG_DIR/SUMMARY.log
