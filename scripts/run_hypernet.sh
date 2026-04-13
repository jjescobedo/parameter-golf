#!/bin/bash
# Hypernet exploration runner: baseline + 6 hypernetwork MLP variants A-F.
# Survives SSH disconnect when invoked via `nohup ... & disown`.
# Logs to /tmp/hypernet_logs/h_*.log; final summary at /tmp/hypernet_logs/SUMMARY.log.

set -u

REAL="ITERATIONS=2000 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=131072 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200 EVAL_STRIDE=64 EVAL_FRAC=0.05"
FEAT="STE_QAT=1 ORTHO_INIT=1 USE_SMEAR_GATE=1"
COMMON="NUM_LAYERS=14 USE_CONV_FFN=1 MLP_MULT=2"

LOG_DIR=/tmp/hypernet_logs
mkdir -p $LOG_DIR

run() {
  local name=$1; shift
  echo "=== $(date) START $name ===" | tee $LOG_DIR/h_${name}.log
  env RUN_ID=h_${name} $REAL $FEAT $COMMON "$@" \
    torchrun --standalone --nproc_per_node=4 train_gpt.py 2>&1 \
    | tee -a $LOG_DIR/h_${name}.log \
    || echo "=== $name FAILED $(date) ===" | tee -a $LOG_DIR/h_${name}.log
  echo "=== $(date) DONE  $name ===" | tee -a $LOG_DIR/h_${name}.log
}

run baseline
run A_meta   META_HYPER_MLP=1 META_HYPER_RANK=192 META_HYPER_CODE=32
run B_tt     TT_MLP=1 TT_RANK=32
run C_kron   KRON_MLP=1 KRON_NUM=16
run D_ut     UT_FILM=1 UT_DEPTH=28 NUM_LAYERS=28
run E_code   CODEBOOK_MLP=1 CODEBOOK_K=64
run F_coord  COORD_MLP=1 COORD_HIDDEN=64 COORD_BASE_RANK=8

{
  echo
  echo "================= SUMMARY ================="
  printf "%-12s | %-10s | %-11s | %-10s\n" "Run" "val_bpb" "Artifact" "step_avg"
  printf -- "-------------+------------+-------------+------------\n"
  for f in $LOG_DIR/h_*.log; do
    name=$(basename $f .log | sed 's/h_//')
    bpb=$(grep final_int8_zlib_roundtrip_exact $f | tail -1 | sed -n 's/.*val_bpb:\([0-9.]*\).*/\1/p')
    size=$(grep "Total submission size:" $f | tail -1 | awk '{print $NF}')
    step=$(grep step_avg $f | tail -1 | sed -n 's/.*step_avg:\([0-9.ms]*\).*/\1/p')
    printf "%-12s | %-10s | %-11s | %-10s\n" "$name" "${bpb:-FAIL}" "${size:-N/A}" "${step:-N/A}"
  done
} | tee $LOG_DIR/SUMMARY.log
