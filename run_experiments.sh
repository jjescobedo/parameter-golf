#!/bin/bash
set -e
cd ~/parameter-golf
git pull origin main
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

NPROC=${NPROC:-2}
COMMON_ARGS="ITERATIONS=2000 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=131072 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200 EVAL_STRIDE=0"

echo ""
echo "================================================================"
echo "  EXPERIMENT 1/3: Kronecker MLP (15 layers, 8 terms)"
echo "================================================================"
env RUN_ID=exp_kronecker NUM_LAYERS=15 USE_KRONECKER_MLP=1 KRONECKER_TERMS=8 NGRAM_ALPHA=0.0 \
    ITERATIONS=2000 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=500 \
    VAL_BATCH_SIZE=131072 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200 EVAL_STRIDE=0 \
    torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /tmp/exp1.log
echo ""

echo "================================================================"
echo "  EXPERIMENT 2/3: N-gram Complementary Training (alpha=0.3)"
echo "================================================================"
env RUN_ID=exp_ngram NUM_LAYERS=9 USE_KRONECKER_MLP=0 NGRAM_ALPHA=0.3 \
    ITERATIONS=2000 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=500 \
    VAL_BATCH_SIZE=131072 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200 EVAL_STRIDE=0 \
    torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /tmp/exp2.log
echo ""

echo "================================================================"
echo "  EXPERIMENT 3/3: Combined (Kronecker + N-gram)"
echo "================================================================"
env RUN_ID=exp_combined NUM_LAYERS=15 USE_KRONECKER_MLP=1 KRONECKER_TERMS=8 NGRAM_ALPHA=0.3 \
    ITERATIONS=2000 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=500 \
    VAL_BATCH_SIZE=131072 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=200 EVAL_STRIDE=0 \
    torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /tmp/exp3.log
echo ""

echo "================================================================"
echo "  RESULTS SUMMARY"
echo "================================================================"
echo ""
echo "Exp 1 (Kronecker MLP, 15 layers):"
grep "final_int8_zlib_roundtrip " /tmp/exp1.log 2>/dev/null | tail -1 || echo "  FAILED"
grep "Total submission size" /tmp/exp1.log 2>/dev/null | tail -1 || echo "  N/A"
echo ""
echo "Exp 2 (N-gram Complementary, 9 layers):"
grep "final_int8_zlib_roundtrip " /tmp/exp2.log 2>/dev/null | tail -1 || echo "  FAILED"
grep "Total submission size" /tmp/exp2.log 2>/dev/null | tail -1 || echo "  N/A"
echo ""
echo "Exp 3 (Combined, 15 layers + N-gram):"
grep "final_int8_zlib_roundtrip " /tmp/exp3.log 2>/dev/null | tail -1 || echo "  FAILED"
grep "Total submission size" /tmp/exp3.log 2>/dev/null | tail -1 || echo "  N/A"
echo ""
echo "Baseline reference: val_bpb=1.3275 artifact=13.0MB"
echo "================================================================"
