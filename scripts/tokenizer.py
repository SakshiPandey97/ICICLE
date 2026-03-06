import os, re, json, random, argparse, pathlib
from collections import defaultdict
from typing import List, Tuple
import sentencepiece as spm
import numpy as np
import psutil
import gc

random.seed(42)

def log_memory(msg: str = ""):
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)
    mem_percent = process.memory_percent()
    print(f"[MEMORY] {msg}: {mem_gb:.2f} GB ({mem_percent:.1f}%)")
    return mem_gb

def reverse_complement(seq: str) -> str:
    comp = {'A':'T', 'T':'A', 'G':'C', 'C':'G', 'N':'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))

def parse_fasta(fp: str) -> List[Tuple[str, str]]:
    recs = []
    with open(fp) as f:
        header, seq = None, []
        for line in f:
            line = line.rstrip()
            if not line: continue
            if line.startswith(">"):
                if header is not None:
                    recs.append((header, "".join(seq)))
                header = line[1:].strip()
                seq = []
            else:
                seq.append(line)
        if header is not None:
            recs.append((header, "".join(seq)))
    return recs

def clean_dna(s: str) -> str:
    s = s.upper()
    return re.sub(r"[^ACGTN]", "N", s)

def genome_id_from_header(h: str) -> str:
    m = re.search(r"(?:^|\|)([A-Z]{1,2}_\d+\.\d)(?:\||\s|$)", h)
    if m: return m.group(1)
    m = re.search(r"(?:^|\|)([A-Z]{2}\d{6,}\.\d)(?:\||\s|$)", h)
    if m: return m.group(1)
    return h.split()[0]

def make_windows(seq: str, L: int, stride: int) -> List[str]:
    out = []
    n = len(seq)
    if n < L: return out
    for i in range(0, n - L + 1, stride):
        out.append(seq[i:i+L])
    return out

def write_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as w:
        for r in rows:
            w.write(json.dumps(r) + "\n")

def read_jsonl(path: str) -> List[dict]:
    return [json.loads(x) for x in open(path)]

def train_sentencepiece(input_txt: str, out_prefix: str, vocab_size: int,
                        input_sentence_size: int = 500000,
                        max_sentence_length: int = 16384,
                        model_type: str = "bpe"):
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    spm.SentencePieceTrainer.Train(
        input=input_txt,
        model_prefix=out_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        input_sentence_size=input_sentence_size,
        train_extremely_large_corpus=True,
        shuffle_input_sentence=True,
        max_sentence_length=max_sentence_length,
        hard_vocab_limit=False,
        num_threads=os.cpu_count(),
        unk_id=0, pad_id=1, bos_id=-1, eos_id=-1,
        user_defined_symbols=["[CLS]", "[SEP]", "N"],
    )


def validate_tokenization(spm_model: str, sequences: List[str], max_len: int, win_len: int, rows: List[dict] = None):
    sp = spm.SentencePieceProcessor(model_file=spm_model)
    
    # Sample up to 1000 sequences for profiling
    sample = random.sample(sequences, min(1000, len(sequences)))
    token_lengths = []
    
    for seq in sample:
        ids = sp.encode(seq, out_type=int)
        token_lengths.append(len(ids))
    
    token_lengths = np.array(token_lengths)
    effective_max = max_len - 2  # Reserve for CLS and SEP
    
    print("\n" + "="*60)
    print("TOKENIZATION VALIDATION")
    print("="*60)
    print(f"Window length (nucleotides): {win_len}")
    print(f"Max token length (after CLS/SEP): {effective_max}")
    print(f"\nTokenization statistics (sampled {len(sample)} sequences):")
    print(f"  Mean tokens per sequence: {token_lengths.mean():.1f}")
    print(f"  Median tokens: {np.median(token_lengths):.1f}")
    print(f"  Std dev: {token_lengths.std():.1f}")
    print(f"  Min tokens: {token_lengths.min()}")
    print(f"  Max tokens: {token_lengths.max()}")
    print(f"  95th percentile: {np.percentile(token_lengths, 95):.1f}")
    print(f"  99th percentile: {np.percentile(token_lengths, 99):.1f}")
    
    # Calculate truncation
    truncated = np.sum(token_lengths > effective_max)
    pct_truncated = 100 * truncated / len(token_lengths)
    print(f"\nSequences requiring truncation: {truncated}/{len(token_lengths)} ({pct_truncated:.1f}%)")
    
    # Calculate compression ratio
    avg_compression = win_len / token_lengths.mean()
    print(f"Average compression ratio: {avg_compression:.2f}x")
    print(f"  (each token represents ~{avg_compression:.2f} nucleotides)")
    
    # Per-class statistics if rows provided
    if rows:
        print("\nPer-class Statistics:")
        for label_name, label_val in [("BG", 0), ("ICE", 1)]:
            subset = [r["seq"] for r in rows if r["label"] == label_val]
            if not subset: continue
            sub_sample = random.sample(subset, min(500, len(subset)))
            sub_lens = np.array([len(sp.encode(s, out_type=int)) for s in sub_sample])
            print(f"  [{label_name}] Mean: {sub_lens.mean():.1f} | p95: {np.percentile(sub_lens, 95):.1f} | Trunc: {(sub_lens > effective_max).mean()*100:.1f}% | Comp: {win_len/sub_lens.mean():.2f}x")

    # Warnings
    suggested_max_len = int(np.percentile(token_lengths, 99)) + 2
    print(f"\n  Suggest setting --max_len to approx: {suggested_max_len}")
    
    if pct_truncated > 10:
        print(f"\n WARNING: {pct_truncated:.1f}% of sequences will be truncated")
        print(f"   Consider increasing max_len or decreasing win_len")
    
    if token_lengths.mean() < effective_max * 0.5:
        print(f"\n WARNING: Sequences use only {100*token_lengths.mean()/effective_max:.1f}% of max_len")
        print(f"   Consider decreasing max_len for efficiency")
    
    print("="*60 + "\n")
    
    return {
        "mean": token_lengths.mean(),
        "median": np.median(token_lengths),
        "p95": np.percentile(token_lengths, 95),
        "truncated_pct": pct_truncated
    }

def encode_split(jsonl_path: str, spm_model: str, max_len: int, out_npz: str, batch_size: int = 100000):
    sp = spm.SentencePieceProcessor(model_file=spm_model)
    cls_id = sp.piece_to_id("[CLS]")
    sep_id = sp.piece_to_id("[SEP]")
    pad_id = sp.pad_id()

    log_memory(f"Starting encode {jsonl_path}")
    
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    temp_files = []
    batch_num = 0
    
    seqs, labels = [], []
    line_count = 0
    
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            ids = sp.encode(row["seq"], out_type=int)
            ids = ids[:max_len - 2]
            ids = [cls_id] + ids + [sep_id]
            if len(ids) < max_len:
                ids = ids + [pad_id]*(max_len - len(ids))
            seqs.append(ids)
            labels.append(row["label"])
            line_count += 1
            
            if len(seqs) >= batch_size:
                X_batch = np.array(seqs, dtype=np.int32)
                y_batch = np.array(labels, dtype=np.int8)
                temp_file = f"{out_npz}.batch{batch_num}.tmp.npz"
                np.savez_compressed(temp_file, input_ids=X_batch, labels=y_batch)
                temp_files.append(temp_file)
                batch_num += 1
                print(f"  Saved batch {batch_num}: {len(seqs)} sequences")
                log_memory(f"After batch {batch_num}")
                seqs, labels = [], []
                gc.collect()
    
    if seqs:
        X_batch = np.array(seqs, dtype=np.int32)
        y_batch = np.array(labels, dtype=np.int8)
        temp_file = f"{out_npz}.batch{batch_num}.tmp.npz"
        np.savez_compressed(temp_file, input_ids=X_batch, labels=y_batch)
        temp_files.append(temp_file)
        batch_num += 1
        print(f"  Saved final batch {batch_num}: {len(seqs)} sequences")
        seqs, labels = [], []
        gc.collect()
    
    print(f"  Combining {len(temp_files)} batches...")
    log_memory("Before combining batches")
    
    all_X, all_y = [], []
    for temp_file in temp_files:
        arr = np.load(temp_file)
        all_X.append(arr["input_ids"])
        all_y.append(arr["labels"])
        os.remove(temp_file)
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    np.savez_compressed(out_npz, input_ids=X, labels=y)
    
    print(f"  Total encoded: {len(X)} sequences")
    log_memory(f"Finished encoding {jsonl_path}")
    gc.collect()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ice_fa", required=True, help="ICE_seq_all.fasta")
    ap.add_argument("--bg_fa", required=True, help="ICE_WG_clean.fasta")
    ap.add_argument("--win_len", type=int, default=300)
    ap.add_argument("--stride", type=int, default=300)
    ap.add_argument("--vocab_size", type=int, default=2000)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--outdir", default="ice_bpe_data")
    ap.add_argument("--input_sentence_size", type=int, default=500000,
                        help="Max sentences for tokenizer training (reduce if OOM)")
    ap.add_argument("--encode_batch_size", type=int, default=100000,
                        help="Batch size for encoding (larger = faster with enough RAM)")
    ap.add_argument("--model_type", default="bpe", choices=["bpe", "unigram"],
                        help="SentencePiece model type")
    ap.add_argument("--balanced_tokenizer", action="store_true",
                        help="Downsample background sequences to match ICE count for tokenizer training (ensures ICE motifs are learned)")
    args = ap.parse_args()

    log_memory("Script start")
    
    outdir = pathlib.Path(args.outdir)
    splits_dir = outdir / "splits"
    tok_dir = outdir / "tokenizer"
    enc_dir = outdir / "encoded"

    ice_recs = parse_fasta(args.ice_fa)
    bg_recs  = parse_fasta(args.bg_fa)
    log_memory("After loading FASTA files")
    print(f"  ICE records: {len(ice_recs)}")
    print(f"  Background records: {len(bg_recs)}")


    all_rows = []
    for header, raw in ice_recs:
        gid = genome_id_from_header(header)
        seq = clean_dna(raw)
        for w in make_windows(seq, args.win_len, args.stride):
            all_rows.append({"genome_id": gid, "source": "ICE", "label": 1, "seq": w})

    for header, raw in bg_recs:
        gid = genome_id_from_header(header)
        seq = clean_dna(raw)
        for w in make_windows(seq, args.win_len, args.stride):
            all_rows.append({"genome_id": gid, "source": "BG", "label": 0, "seq": w})

    print(f"  Total windows: {len(all_rows)}")
    print(f"  ICE windows: {sum(1 for r in all_rows if r['label'] == 1)}")
    print(f"  Background windows: {sum(1 for r in all_rows if r['label'] == 0)}")
    log_memory("After creating windows")

    gids = sorted({r["genome_id"] for r in all_rows})
    random.shuffle(gids)
    n = len(gids)
    n_train = int(n * args.train_frac)
    n_val   = int(n * args.val_frac)
    train_g = set(gids[:n_train])
    val_g   = set(gids[n_train:n_train+n_val])
    test_g  = set(gids[n_train+n_val:])

    print(f"  Total genomes: {n}")
    print(f"  Train genomes: {len(train_g)}")
    print(f"  Val genomes: {len(val_g)}")
    print(f"  Test genomes: {len(test_g)}")

    def pick(gs): return [r for r in all_rows if r["genome_id"] in gs]

    train_rows = pick(train_g)
    val_rows   = pick(val_g)
    test_rows  = pick(test_g)

    # Rebalance classes
    print("\nKeeping val/test at natural distribution. Augmenting training set only")
    log_memory("Before RC augmentation")
    
    augmented_train_rows = []
    for r in train_rows:
        augmented_train_rows.append(r)
        rc_row = r.copy()
        rc_row["seq"] = reverse_complement(r["seq"])
        rc_row["source"] = r["source"] + "_RC"
        augmented_train_rows.append(rc_row)
    random.shuffle(augmented_train_rows)
    train_rows = augmented_train_rows 
    del augmented_train_rows 
    gc.collect()
    log_memory("After RC augmentation")
    
    ice_train = sum(1 for r in train_rows if r['label'] == 1)
    ice_val = sum(1 for r in val_rows if r['label'] == 1)
    ice_test = sum(1 for r in test_rows if r['label'] == 1)
    
    print(f"  Train: {len(train_rows)} windows (unbalanced, with RC: {ice_train} ICE, {len(train_rows)-ice_train} BG)")
    print(f"  Val: {len(val_rows)} windows (natural: {ice_val} ICE, {len(val_rows)-ice_val} BG)")
    print(f"  Test: {len(test_rows)} windows (natural: {ice_test} ICE, {len(test_rows)-ice_test} BG)")
    
    # Calculate recommended class weight for training
    if len(val_rows) > 0:
        bg_val = len(val_rows) - ice_val
        if ice_val > 0:
            ratio = bg_val / ice_val
            print(f"\n  Recommended class weight for training: --class_weight 1.0 {ratio:.1f}")
            print(f"  (BG:ICE ratio in validation is ~{ratio:.1f}:1)")

    os.makedirs(splits_dir, exist_ok=True)
    print("\nsplit files:")
    write_jsonl(str(splits_dir / "train.jsonl"), train_rows)
    log_memory("After writing train.jsonl")
    write_jsonl(str(splits_dir / "val.jsonl"),   val_rows)
    write_jsonl(str(splits_dir / "test.jsonl"),  test_rows)
    log_memory("After writing all splits")

    # Train BPE tokenizer
    print(f"\nTraining BPE tokenizer:")
    train_txt = outdir / "train_corpus.txt"
    
    print(f"  Total training sequences (with RC): {len(train_rows)}")
    sample_pool = [r for r in train_rows if not str(r.get("source","")).endswith("_RC")]
    print(f"  Non-RC sequences available: {len(sample_pool)}")
    
    sample_rows = sample_pool
    
    if args.balanced_tokenizer:
        print("\n  Balancing tokenizer corpus (ICE vs BG)")
        ice_pool = [r for r in sample_pool if r['label'] == 1]
        bg_pool = [r for r in sample_pool if r['label'] == 0]
        
        n_ice = len(ice_pool)
        n_bg = len(bg_pool)
        print(f"    Available: {n_ice} ICE, {n_bg} BG")
        
        if n_ice > 0 and n_bg > 0:
            if n_bg > n_ice:
                print(f"    Downsampling BG to match ICE count ({n_ice})")
                bg_sample = random.sample(bg_pool, n_ice)
                sample_rows = ice_pool + bg_sample
            else:
                print(f"    Downsampling ICE to match BG count ({n_bg})")
                ice_sample = random.sample(ice_pool, n_bg)
                sample_rows = ice_sample + bg_pool
            random.shuffle(sample_rows)
            print(f"    Balanced corpus size: {len(sample_rows)}")
        else:
            print("Cannot balance: one class is empty. Using full pool.")
            sample_rows = sample_pool
    else:
        print(f"  Using full non-RC pool ({len(sample_rows)} sequences)")

    print(f"  Writing {len(sample_rows)} sequences to corpus")
    
    with open(train_txt, "w") as w:
        for r in sample_rows:
            w.write(r["seq"] + "\n")
    
    corpus_size_mb = train_txt.stat().st_size / (1024**2)
    print(f"  Corpus file size: {corpus_size_mb:.1f} MB")
    print(f"  SentencePiece input_sentence_size: {args.input_sentence_size}")
    
    spm_max_len = max(16384, args.win_len * 2)
    train_sentencepiece(str(train_txt), str(tok_dir / "dna_bpe"), 
                        vocab_size=args.vocab_size,
                        input_sentence_size=args.input_sentence_size,
                        max_sentence_length=spm_max_len,
                        model_type=args.model_type)
    print(f"  Tokenizer saved to {tok_dir}")

    # VALIDATE TOKENIZATION
    spm_model = str(tok_dir / "dna_bpe.model")
    print(f"  Sampling {min(1000, len(train_rows))} sequences for validation...")
    sample_rows = random.sample(train_rows, min(1000, len(train_rows)))
    train_seqs = [r["seq"] for r in sample_rows]
    validate_tokenization(spm_model, train_seqs, args.max_len, args.win_len, rows=sample_rows)

    # Encode all splits
    print("\nEncoding splits with batching to save memory")
    print(f"Using batch size: {args.encode_batch_size} (adjust with --encode_batch_size)")
    
    # Clear memory before encoding
    del all_rows, train_rows, val_rows, test_rows
    gc.collect()
    log_memory("Before encoding (freed split data)")
    
    encode_split(str(splits_dir / "train.jsonl"), spm_model, args.max_len, 
                 str(enc_dir / "train.npz"), batch_size=args.encode_batch_size)
    gc.collect()
    
    encode_split(str(splits_dir / "val.jsonl"), spm_model, args.max_len, 
                 str(enc_dir / "val.npz"), batch_size=args.encode_batch_size)
    gc.collect()
    
    encode_split(str(splits_dir / "test.jsonl"), spm_model, args.max_len, 
                 str(enc_dir / "test.npz"), batch_size=args.encode_batch_size)
    gc.collect()

    log_memory("Final memory usage")
    print("\nDone")
    print(f"  Splits: {splits_dir}")
    print(f"  Tokenizer: {tok_dir}")
    print(f"  Encoded arrays: {enc_dir}")
    print("\nMemory optimization tips:")
    print("  - Reduce --encode_batch_size if OOM during encoding")
    print("  - Reduce --input_sentence_size if OOM during tokenizer training")
    print("  - Consider processing splits separately if still OOM")

if __name__ == "__main__":
    main()
