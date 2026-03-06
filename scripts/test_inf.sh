#!/bin/bash

#SBATCH --job-name=test_large_model
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=01:30:00
#SBATCH --output=logs/test_large_model_%j.out
#SBATCH --error=logs/test_large_model_%j.err

# ==============================================================================
# TEST INFERENCE ON BEST SAVED MODEL
# ==============================================================================
# 
# Runs test inference on the best.pt checkpoint from training.
# Uses the saved threshold from training or optimizes on validation set.
#
# ==============================================================================

set -ex

export WORK_DIR="$SLURM_SUBMIT_DIR"
mkdir -p "$WORK_DIR/logs"

echo "=============================================="
echo "TEST INFERENCE - Large Model"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: 1x B200"
echo "Started: $(date)"
echo "=============================================="

module load pytorch/2.8.0

cd "$WORK_DIR"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Checkpoint directory (from training)
CHECKPOINT_DIR="$WORK_DIR/checkpoints_large_model"

# Data directory (use ORIGINAL encoded data, NOT SMOTE, for fair test evaluation)
# SMOTE was only applied to training data; val/test are unchanged
DATA_DIR="$WORK_DIR/ice_50k_w2000_bpe_UNBAL/encoded_SMOTE_0.2"

# Output file
OUTPUT_FILE="$CHECKPOINT_DIR/test_results.json"

# ==============================================================================
# VERIFY FILES EXIST
# ==============================================================================

if [ ! -f "$CHECKPOINT_DIR/best.pt" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_DIR/best.pt"
    echo "Make sure training completed successfully!"
    exit 1
fi

if [ ! -f "$DATA_DIR/test.npz" ]; then
    echo "ERROR: Test data not found at $DATA_DIR/test.npz"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUTPUT_FILE"
echo ""

# ==============================================================================
# RUN TEST INFERENCE
# ==============================================================================

python test_inference.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_dir "$DATA_DIR" \
    --split test \
    --batch_size 512 \
    --optimize_threshold \
    --threshold_metric f2 \
    --output "$OUTPUT_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Test inference completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
    echo "=============================================="
    
    # Print summary from JSON
    echo ""
    echo "Quick Summary:"
    python -c "
import json
with open('$OUTPUT_FILE') as f:
    r = json.load(f)
m = r['metrics']
print(f\"  PR-AUC:    {m['pr_auc']:.4f}\")
print(f\"  F1:        {m['f1']:.4f}\")
print(f\"  Precision: {m['precision']:.4f}\")
print(f\"  Recall:    {m['recall']:.4f}\")
print(f\"  Threshold: {m['threshold']:.4f}\")
"
else
    echo ""
    echo "Test inference failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "Finished: $(date)"
