#!/bin/bash
# Hypernet round 2: spend the headroom freed by round 1.
#   1. C_kron26      - C_kron at 26 layers (uses ~7 MB freed by round 1's 9 MB artifact)
#   2. E_code_v2     - E_code with fixed alpha_fc init magnitude
#   3. F_coord_v2    - F_coord with zero-init fc_v + per-layer coord_gate
#   4. D_ut_wide     - UT-FiLM with 4x MLP hidden (uses 14 MB freed by round 1's 2 MB artifact)
# Logs to /tmp/hypernet_logs_r2/. Designed for nohup + disown.

set -u

REAL="ITERATIONS=2000 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=131072 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200 EVAL_STRIDE=64 EVAL_FRAC=0.05"
FEAT="STE_QAT=1 ORTHO_INIT=1 USE_SMEAR_GATE=1"
COMMON="USE_CONV_FFN=1 MLP_MULT=2"

LOG_DIR=/tmp/hypernet_logs_r2
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

run C_kron26    NUM_LAYERS=26 KRON_MLP=1 KRON_NUM=16
run E_code_v2   NUM_LAYERS=14 CODEBOOK_MLP=1 CODEBOOK_K=64
run F_coord_v2  NUM_LAYERS=14 COORD_MLP=1 COORD_HIDDEN=64 COORD_BASE_RANK=8
run D_ut_wide   NUM_LAYERS=28 UT_FILM=1 UT_DEPTH=28 MLP_MULT=4

{
  echo
  echo "================= ROUND 2 SUMMARY ================="
  printf "%-14s | %-10s | %-12s | %-10s\n" "Run" "val_bpb" "Artifact" "step_avg"
  printf -- "---------------+------------+--------------+------------\n"
  for f in $LOG_DIR/h_*.log; do
    name=$(basename $f .log | sed 's/h_//')
    bpb=$(grep final_int8_zlib_roundtrip_exact $f | tail -1 | sed -n 's/.*val_bpb:\([0-9.]*\).*/\1/p')
    size=$(grep "Total submission size:" $f | tail -1 | awk '{print $NF}')
    step=$(grep step_avg $f | tail -1 | sed -n 's/.*step_avg:\([0-9.ms]*\).*/\1/p')
    printf "%-14s | %-10s | %-12s | %-10s\n" "$name" "${bpb:-FAIL}" "${size:-N/A}" "${step:-N/A}"
  done
} | tee $LOG_DIR/SUMMARY.log
