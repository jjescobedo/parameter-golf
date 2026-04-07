#!/bin/bash
set -e
cd ~/parameter-golf
git pull origin architecture-problem
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

NPROC=${NPROC:-2}
COMMON="ITERATIONS=2000 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=131072 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200 EVAL_STRIDE=0"

echo ""
echo "================================================================"
echo "  EXPERIMENT 0/4: Baseline (9 layers, 3x MLP)"
echo "================================================================"
env RUN_ID=exp_baseline NUM_LAYERS=9 MLP_MULT=3 \
    $COMMON \
    torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /tmp/exp0.log
echo ""

echo "================================================================"
echo "  EXPERIMENT 1/4: C2 Factorized FFN (14 layers, rank-192)"
echo "================================================================"
env RUN_ID=exp_factorized NUM_LAYERS=14 USE_FACTORIZED_FFN=1 FACTORIZED_RANK=192 \
    $COMMON \
    torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /tmp/exp1.log
echo ""

echo "================================================================"
echo "  EXPERIMENT 2/4: C1 Soft MoE (9 layers, 4 experts)"
echo "================================================================"
env RUN_ID=exp_softmoe NUM_LAYERS=9 MOE_NUM_EXPERTS=4 \
    $COMMON \
    torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /tmp/exp2.log
echo ""

echo "================================================================"
echo "  EXPERIMENT 3/4: C3 Conv-FFN (12 layers, 2x MLP + conv)"
echo "================================================================"
env RUN_ID=exp_convffn NUM_LAYERS=12 USE_CONV_FFN=1 MLP_MULT=2 \
    $COMMON \
    torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /tmp/exp3.log
echo ""

echo "================================================================"
echo "  EXPERIMENT 4/4: C5 Butterfly MLP (18 layers) [HIGH RISK]"
echo "================================================================"
env RUN_ID=exp_butterfly NUM_LAYERS=18 USE_BUTTERFLY_MLP=1 \
    $COMMON \
    torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /tmp/exp4.log || \
    echo "BUTTERFLY EXPERIMENT FAILED (likely torch.compile)" | tee -a /tmp/exp4.log
echo ""

echo "================================================================"
echo "  RESULTS SUMMARY"
echo "================================================================"
echo ""
NAMES=("Baseline 9L/3xMLP" "Factorized 14L/r192" "SoftMoE 9L/4exp" "ConvFFN 12L/2xMLP" "Butterfly 18L")
for i in 0 1 2 3 4; do
    echo "Exp $i (${NAMES[$i]}):"
    grep "final_int8_zlib_roundtrip " /tmp/exp${i}.log 2>/dev/null | tail -1 || echo "  BPB: FAILED"
    grep "Total submission size" /tmp/exp${i}.log 2>/dev/null | tail -1 || echo "  Size: N/A"
    echo ""
done
echo "Previous best: val_bpb=1.3275 artifact=13.0MB"
echo "================================================================"
