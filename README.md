# ICICLE

ICICLE is a deep learning tool for identifying Integrative and Conjugative Elements (ICEs) in genomic sequences without requiring gene-level annotation. It uses a BERT-style transformer encoder trained on BPE-tokenized DNA windows.

## Project Structure

```
scripts/
  tokenizer.py          # BPE tokenizer training, windowing, encoding
  preprocess_smote.py    # Offline SMOTE oversampling of training data
  model.py               # Transformer (BERTForICE) architecture
  train_ddp_smote.py     # Distributed training with optional SMOTE
  test_inference.py      # Inference on saved checkpoint
benchmark/
  download_mobhunter_genomes.py  # Download MOBHunter benchmark genomes
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0 (with CUDA for GPU training)
- sentencepiece
- scikit-learn
- imbalanced-learn
- numpy
- psutil

Optional (for benchmark download):
- biopython
- NCBI datasets CLI
- pandas, openpyxl

## Setup

```bash
conda create -n icicle python=3.10 -y
conda activate icicle

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install sentencepiece scikit-learn imbalanced-learn numpy psutil
```

## Running on a Local Machine (Single GPU)

### Step 1: Prepare Input Data

You need two FASTA files:
- **ICE sequences**: known ICE elements (e.g., `ICE_seq_all.fasta`)
- **Background genomes**: whole genomes with ICE regions masked (e.g., `ICE_WG_masked_filtered.fasta`)

### Step 2: Tokenize and Encode

```bash
cd scripts

python tokenizer.py \
    --ice_fa ../ICE_seq_all.fasta \
    --bg_fa ../ICE_WG_masked_filtered.fasta \
    --outdir ../data \
    --win_len 2000 \
    --stride 1000 \
    --vocab_size 50000 \
    --max_len 1024 \
    --model_type bpe
```

This produces:
- `data/tokenizer/dna_bpe.model` — trained BPE model
- `data/encoded/{train,val,test}.npz` — encoded arrays

### Step 3: (Optional) Apply SMOTE

If your dataset is heavily imbalanced:

```bash
python preprocess_smote.py \
    --data_dir ../data/encoded \
    --output_dir ../data/encoded_SMOTE_0.2 \
    --strategy 0.2 \
    --k_neighbors 5 \
    --vocab_size 50000
```

### Step 4: Train

Single-GPU training via `torchrun` with world size 1:

```bash
torchrun --standalone --nproc_per_node=1 \
    train_ddp_smote.py \
    --data_dir ../data/encoded_SMOTE_0.2 \
    --vocab_size 50000 \
    --outdir ../checkpoints
```

Sensible defaults are built in (768-dim, 12 layers, focal loss, ALiBi, etc.).
Reduce `--batch_size` (default 32) if you run out of GPU memory.
Run `python train_ddp_smote.py --help` to see all tunable options.

### Step 5: Test Inference

```bash
python test_inference.py \
    --checkpoint_dir ../checkpoints \
    --data_dir ../data/encoded_SMOTE_0.2 \
    --split test \
    --batch_size 64 \
    --optimize_threshold
```

Results are saved to `checkpoints/test_results.json`.

## Multi-GPU Training

Scale to multiple GPUs by increasing `--nproc_per_node`:

```bash
torchrun --standalone --nproc_per_node=4 \
    train_ddp_smote.py \
    --data_dir ../data/encoded_SMOTE_0.2 \
    --vocab_size 50000 \
    --batch_size 48 \
    --outdir ../checkpoints
```

## (SLURM) Usage

Pre-configured SLURM scripts for Hipergator are in `scripts/`. Before using them:

1. Edit each `.sh` file and replace `<your_account>` and `<your_qos>` with your SLURM allocation.
2. Adjust `--partition` if needed for your available hardware.
3. Adjust `--mem`, `--time`, and GPU counts to match your allocation.

```bash
# Step 1: Generate tokenized + SMOTE data
sbatch scripts/generate_large_model_data.sh

# Step 2: Train
sbatch scripts/train_large_model.sh

# Step 3: Evaluate
sbatch scripts/test_inf.sh
```

See individual `.sh` files for full configuration details.

## Resuming Training

To resume from a checkpoint:

```bash
torchrun --standalone --nproc_per_node=4 \
    train_ddp_smote.py \
    --data_dir ../data/encoded_SMOTE_0.2 \
    --vocab_size 50000 \
    --outdir ../checkpoints \
    --resume ../checkpoints/latest.pt
```

## License

See [LICENSE](LICENSE).
