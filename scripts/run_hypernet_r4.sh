#!/bin/bash
# Hypernet round 4: pure depth push on the C_kron winner.
# Round 3 showed attention sharing is a hard quality floor - it cannot be
# stacked with Kron MLP compression. So: drop the sharing, just push depth.
#   1. C_kron28     - 28L Kron K=16, no attention sharing (expected ~1.376 BPB)
# Logs persisted to ~/parameter-golf/runlogs/r4/.

set -u

REAL="ITERATIONS=2000 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=131072 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200 EVAL_STRIDE=64 EVAL_FRAC=0.05"
FEAT="STE_QAT=1 ORTHO_INIT=1 USE_SMEAR_GATE=1"
COMMON="USE_CONV_FFN=1 MLP_MULT=2"

LOG_DIR="$HOME/parameter-golf/runlogs/r4"
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

run C_kron28  NUM_LAYERS=28 KRON_MLP=1 KRON_NUM=16

{
  echo
  echo "================= ROUND 4 SUMMARY ================="
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
