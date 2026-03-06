#!/bin/bash
#SBATCH --job-name=large_ice_model
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300gb
#SBATCH --time=72:00:00
#SBATCH --output=logs/large_model_%j.out
#SBATCH --error=logs/large_model_%j.err

# ==============================================================================
# LARGE MODEL TRAINING (12 Layers, 768 d_model)
# ==============================================================================
# 
# Model Architecture:
#   - d_model: 768 (vs 512 baseline)
#   - n_layers: 12 (vs 6 baseline)
#   - n_heads: 12 (vs 8 baseline)
#   - ffn_dim: 3072 (vs 2048 baseline)
#   - ~100M parameters (vs ~35M baseline)
#
# Data:
#   - Vocab: 50k (vs 32k baseline)
#   - Window: 2000bp
#   - Max tokens: 1024 (vs 512 baseline)
#
# Improvements:
#   - Fixed Focal Loss (per-class alpha weighting)
#   - Label smoothing (0.1)
#   - ConvStem for local motif detection
#   - ALiBi for better positional handling
#
# ==============================================================================

set -ex

export WORK_DIR=$SLURM_SUBMIT_DIR
mkdir -p $WORK_DIR/logs

echo "=============================================="
echo "LARGE MODEL TRAINING - ICE Detection"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4x B200"
echo "Started: $(date)"
echo ""
echo "=============================================="

module load pytorch/2.8.0

cd $WORK_DIR

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data directories
DATA_DIR="$WORK_DIR/ice_50k_w2000_bpe_UNBAL/encoded_SMOTE_0.2"

# Verify data exists
if [ ! -f "$DATA_DIR/train.npz" ]; then
    echo "ERROR: Training data not found at $DATA_DIR/train.npz"
    echo "Please run generate_large_model_data.sh first!"
    exit 1
fi

# Model configuration - LARGE MODEL
VOCAB_SIZE=50000
MAX_LEN=1024
D_MODEL=768
N_LAYERS=12
N_HEADS=12
FFN_DIM=3072
DROPOUT=0.15

# Training hyperparameters
# B200 has 192GB VRAM - batch 48 balances speed + quality
BATCH_SIZE=48          # Per GPU (48 x 4 = 192 effective batch)
EPOCHS=100
LR=3e-5               # Conservative LR for quality
WARMUP_RATIO=0.1

# Focal Loss settings (FIXED: per-class alpha)
FOCAL_ALPHA=0.8       # Higher weight for ICE (minority) class
FOCAL_GAMMA=2.0
LABEL_SMOOTHING=0.05  # Lower smoothing preserves signal

# Output directory
OUTDIR="$WORK_DIR/checkpoints_large_model"

echo ""
echo "Configuration:"
echo "  Data: $DATA_DIR"
echo "  Vocab: $VOCAB_SIZE"
echo "  Max length: $MAX_LEN"
echo "  Model: d=$D_MODEL, L=$N_LAYERS, H=$N_HEADS, ffn=$FFN_DIM"
echo "  Est. parameters: ~100M"
echo "  Batch: $BATCH_SIZE per GPU x 4 GPUs = $((BATCH_SIZE * 4)) effective"
echo "  LR: $LR with ${WARMUP_RATIO} warmup"
echo "  Focal: alpha=$FOCAL_ALPHA (ICE weight), gamma=$FOCAL_GAMMA"
echo "  Label smoothing: $LABEL_SMOOTHING"
echo "  ConvStem: ENABLED"
echo "  Output: $OUTDIR"
echo ""

# ==============================================================================
# TRAINING
# ==============================================================================

mkdir -p $OUTDIR

# Copy scripts to tmp for faster I/O
TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID}"
mkdir -p $TMP_DIR
cp $WORK_DIR/*.py $TMP_DIR/
cd $TMP_DIR

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

echo "Starting DDP training..."

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train_ddp_smote.py \
    --data_dir "$DATA_DIR" \
    --vocab_size $VOCAB_SIZE \
    --max_len $MAX_LEN \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --ffn_dim $FFN_DIM \
    --dropout $DROPOUT \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay 0.01 \
    --warmup_ratio $WARMUP_RATIO \
    --num_workers 4 \
    --use_focal_loss \
    --focal_alpha $FOCAL_ALPHA \
    --focal_gamma $FOCAL_GAMMA \
    --label_smoothing $LABEL_SMOOTHING \
    --pooling attention \
    --use_alibi \
    --use_conv_stem \
    --metric pr_auc \
    --early_stopping_patience 15 \
    --optimize_threshold \
    --outdir "$OUTDIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Training completed successfully!"
    echo "Results saved to: $OUTDIR"
    echo "=============================================="
else
    echo ""
    echo "Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Finished: $(date)"
