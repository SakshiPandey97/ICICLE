#!/bin/bash

#SBATCH --job-name=large_model_data
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=bigmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=240gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/large_model_data_%j.log
#SBATCH --error=logs/large_model_data_%j.err


set -e

echo "======================================================"
echo "LARGE MODEL DATA GENERATION"
echo "======================================================"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "======================================================"

module load python/3.10

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

pip install --user imbalanced-learn --quiet

ICE_FASTA="ICE_seq_all.fasta"
BG_FASTA="ICE_WG_masked_filtered.fasta"

VOCAB_SIZE=50000
WIN_LEN=2000
MAX_LEN=1024
MODEL_TYPE="bpe"


SENTENCE_SIZE=300000    
ENCODE_BATCH_SIZE=200000

SMOTE_STRATEGY="0.2"    

VS_K=$((VOCAB_SIZE / 1000))k
OUT_DIR="ice_${VS_K}_w${WIN_LEN}_${MODEL_TYPE}_UNBAL"
SMOTE_DATA_DIR="${OUT_DIR}/encoded_SMOTE_${SMOTE_STRATEGY}"

echo "------------------------------------------------------"
echo "CONFIGURATION:"
echo "  Vocab Size: $VOCAB_SIZE"
echo "  Window Size: ${WIN_LEN}bp"
echo "  Max Token Length: $MAX_LEN"
echo "  Model Type: $MODEL_TYPE"
echo "  SMOTE Strategy: $SMOTE_STRATEGY (minority = 20% of majority)"
echo "  Output Directory: $OUT_DIR"
echo "------------------------------------------------------"

# ==============================================================================
# STEP 1: TOKENIZER TRAINING & DATA ENCODING
# ==============================================================================

DONE_FLAG="$OUT_DIR/encoded/DONE"
if [ -f "$DONE_FLAG" ]; then
    echo ""
    echo "STEP 1 SKIP: Encoded data already exists at $OUT_DIR/encoded"
else
    echo ""
    echo "======================================================"
    echo "STEP 1: Tokenizer Training & Data Encoding"
    echo "======================================================"
    
    # Use 50% overlap for diverse training contexts
    STRIDE=$((WIN_LEN / 2))
    echo "  Windows: ${WIN_LEN}bp, stride ${STRIDE}bp (50% overlap)"
    
    if ! python tokenizer.py \
        --ice_fa "$ICE_FASTA" \
        --bg_fa "$BG_FASTA" \
        --outdir "$OUT_DIR" \
        --win_len $WIN_LEN \
        --stride $STRIDE \
        --max_len $MAX_LEN \
        --vocab_size "$VOCAB_SIZE" \
        --input_sentence_size $SENTENCE_SIZE \
        --encode_batch_size $ENCODE_BATCH_SIZE; then
        
        echo "FATAL: Tokenizer training failed!"
        exit 1
    fi
    
    echo "Tokenizer and encoding complete!"
    touch "$DONE_FLAG"
    
    rm -f ${OUT_DIR}/train_corpus.txt
fi

# ==============================================================================
# STEP 2: SMOTE PREPROCESSING
# ==============================================================================

if [ -f "$SMOTE_DATA_DIR/train.npz" ]; then
    echo ""
    echo "STEP 2 SKIP: SMOTE data already exists at $SMOTE_DATA_DIR"
else
    echo ""
    echo "======================================================"
    echo "STEP 2: SMOTE Preprocessing"
    echo "======================================================"
    echo "  Input: $OUT_DIR/encoded"
    echo "  Output: $SMOTE_DATA_DIR"
    echo "  Strategy: $SMOTE_STRATEGY"
    
    if ! python preprocess_smote.py \
        --data_dir "$OUT_DIR/encoded" \
        --output_dir "$SMOTE_DATA_DIR" \
        --strategy "$SMOTE_STRATEGY" \
        --k_neighbors 5 \
        --seed 42; then
        
        echo "FATAL: SMOTE preprocessing failed!"
        exit 1
    fi
    
    echo "SMOTE preprocessing complete!"
fi

# ==============================================================================
# VERIFICATION
# ==============================================================================

echo ""
echo "======================================================"
echo "VERIFICATION"
echo "======================================================"

# Check all required files exist
REQUIRED_FILES=(
    "$OUT_DIR/tokenizer/dna_bpe.model"
    "$OUT_DIR/encoded/train.npz"
    "$OUT_DIR/encoded/val.npz"
    "$OUT_DIR/encoded/test.npz"
    "$SMOTE_DATA_DIR/train.npz"
)

ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  OK: $f ($SIZE)"
    else
        echo "  MISSING: $f"
        ALL_OK=false
    fi
done

# Symlink val/test to SMOTE dir (they don't need augmentation)
if [ ! -f "$SMOTE_DATA_DIR/val.npz" ]; then
    ln -sf "../encoded/val.npz" "$SMOTE_DATA_DIR/val.npz"
    echo "  Created symlink: $SMOTE_DATA_DIR/val.npz"
fi
if [ ! -f "$SMOTE_DATA_DIR/test.npz" ]; then
    ln -sf "../encoded/test.npz" "$SMOTE_DATA_DIR/test.npz"
    echo "  Created symlink: $SMOTE_DATA_DIR/test.npz"
fi

if [ "$ALL_OK" = true ]; then
    echo ""
    echo "======================================================"
    echo "ALL DATA GENERATION COMPLETE!"
    echo "======================================================"
    echo ""
    echo "Data ready for training:"
    echo "  Tokenizer: $OUT_DIR/tokenizer/dna_bpe.model"
    echo "  SMOTE data: $SMOTE_DATA_DIR/"
    echo ""
    echo "Recommended training command:"
    echo "  python train_ddp_smote.py \\"
    echo "    --data_dir $SMOTE_DATA_DIR \\"
    echo "    --vocab_size $VOCAB_SIZE \\"
    echo "    --max_len $MAX_LEN \\"
    echo "    --d_model 768 \\"
    echo "    --n_layers 12 \\"
    echo "    --n_heads 12 \\"
    echo "    --ffn_dim 3072 \\"
    echo "    --use_alibi \\"
    echo "    --use_focal_loss \\"
    echo "    --focal_alpha 0.75 \\"
    echo "    --label_smoothing 0.1 \\"
    echo "    --use_conv_stem"
    echo ""
else
    echo ""
    echo "Some files are missing. Please check the logs."
    exit 1
fi

echo "Finished: $(date)"
echo "======================================================"
