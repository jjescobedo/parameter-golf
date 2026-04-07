#!/bin/bash
set -e
cd ~/parameter-golf
git pull origin architecture-problem
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

NPROC=${NPROC:-2}
COMMON="ITERATIONS=2000 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=131072 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32"

run_exp() {
    local num=$1 name=$2; shift 2
    echo ""
    echo "================================================================"
    echo "  EXPERIMENT $num: $name"
    echo "================================================================"
    env "$@" $COMMON torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /tmp/exp${num}.log
    echo ""
}

# Phase 0: Baseline reference
run_exp 0 "Baseline 9L/3xMLP (reference)" RUN_ID=tune_baseline NUM_LAYERS=9 MLP_MULT=3

# Phase 1: Kernel size sweep (12L, 2x MLP, all attention)
run_exp 1 "ConvFFN 12L/2x k=3" RUN_ID=tune_k3 NUM_LAYERS=12 USE_CONV_FFN=1 MLP_MULT=2 CONV_KERNEL_SIZE=3
run_exp 2 "ConvFFN 12L/2x k=5" RUN_ID=tune_k5 NUM_LAYERS=12 USE_CONV_FFN=1 MLP_MULT=2 CONV_KERNEL_SIZE=5
run_exp 3 "ConvFFN 12L/2x k=7" RUN_ID=tune_k7 NUM_LAYERS=12 USE_CONV_FFN=1 MLP_MULT=2 CONV_KERNEL_SIZE=7

# Phase 2: Depth vs width (k=3, all attention)
run_exp 4 "ConvFFN 10L/3x k=3 (wider)" RUN_ID=tune_wide NUM_LAYERS=10 USE_CONV_FFN=1 MLP_MULT=3 CONV_KERNEL_SIZE=3
run_exp 5 "ConvFFN 13L/2x k=3" RUN_ID=tune_13L NUM_LAYERS=13 USE_CONV_FFN=1 MLP_MULT=2 CONV_KERNEL_SIZE=3
run_exp 6 "ConvFFN 14L/2x k=3" RUN_ID=tune_14L NUM_LAYERS=14 USE_CONV_FFN=1 MLP_MULT=2 CONV_KERNEL_SIZE=3

# Phase 3: Hybrid attention/conv (k=3, 2x MLP)
run_exp 7 "Hybrid 16L attn_every=2" RUN_ID=tune_hybrid2 NUM_LAYERS=16 USE_CONV_FFN=1 MLP_MULT=2 CONV_KERNEL_SIZE=3 ATTN_EVERY_N=2
run_exp 8 "Hybrid 18L attn_every=3" RUN_ID=tune_hybrid3 NUM_LAYERS=18 USE_CONV_FFN=1 MLP_MULT=2 CONV_KERNEL_SIZE=3 ATTN_EVERY_N=3
run_exp 9 "Hybrid 20L attn_every=4" RUN_ID=tune_hybrid4 NUM_LAYERS=20 USE_CONV_FFN=1 MLP_MULT=2 CONV_KERNEL_SIZE=3 ATTN_EVERY_N=4

echo ""
echo "================================================================"
echo "  RESULTS SUMMARY — C3 ConvFFN Tuning"
echo "================================================================"
echo ""
NAMES=(
    "Baseline 9L/3xMLP"
    "ConvFFN 12L/2x k=3"
    "ConvFFN 12L/2x k=5"
    "ConvFFN 12L/2x k=7"
    "ConvFFN 10L/3x k=3"
    "ConvFFN 13L/2x k=3"
    "ConvFFN 14L/2x k=3"
    "Hybrid 16L attn/2"
    "Hybrid 18L attn/3"
    "Hybrid 20L attn/4"
)
printf "%-25s | %-12s | %-12s | %-10s\n" "Experiment" "val_bpb" "Artifact" "ms/step"
printf "%-25s-+-%-12s-+-%-12s-+-%-10s\n" "-------------------------" "------------" "------------" "----------"
for i in $(seq 0 9); do
    BPB=$(grep "final_int8_zlib_roundtrip_exact" /tmp/exp${i}.log 2>/dev/null | tail -1 | sed 's/.*val_bpb://' | awk '{print $1}' || echo "FAILED")
    SIZE=$(grep "Serialized model mixed_int56" /tmp/exp${i}.log 2>/dev/null | tail -1 | sed 's/.*: //' | sed 's/ bytes//' | awk '{printf "%.1fMB", $1/1048576}' || echo "N/A")
    STEP=$(grep "step:2000/2000 train_loss" /tmp/exp${i}.log 2>/dev/null | tail -1 | sed 's/.*step_avg://' | sed 's/ms//' || echo "N/A")
    printf "%-25s | %-12s | %-12s | %-10s\n" "${NAMES[$i]}" "$BPB" "$SIZE" "${STEP}ms"
done
echo ""
echo "Previous best (no sliding window): val_bpb=1.3275 artifact=13.0MB"
echo "All experiments above use EVAL_STRIDE=64 (sliding window eval)"
echo "================================================================"
