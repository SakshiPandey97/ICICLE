"""Distributed Data Parallel training with optional SMOTE oversampling."""
import os
import math
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import BERTForICE


class FocalLoss(nn.Module):
    """Focal Loss with per-class alpha weighting and optional label smoothing."""
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.long()
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            n_classes = inputs.size(1)
            smooth_targets = torch.full_like(inputs, self.label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce_loss = -(smooth_targets * F.log_softmax(inputs, dim=1)).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-7, max=1.0)
        
        # Per-class alpha weights
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha, dtype=torch.float, device=inputs.device),
            torch.full_like(targets, 1 - self.alpha, dtype=torch.float, device=inputs.device)
        )
        loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def setup_distributed():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def apply_smote(X, y, strategy='auto', k_neighbors=5, random_state=42):
    """Apply SMOTE in PCA-reduced space and map back to token IDs."""
    from imblearn.over_sampling import SMOTE
    
    print_rank0(f"\n{'='*60}")
    print_rank0("SMOTE OVERSAMPLING")
    print_rank0(f"{'='*60}")
    
    n_ice = int((y == 1).sum())
    n_bg = int((y == 0).sum())
    print_rank0(f"Before SMOTE: {n_ice:,} ICE, {n_bg:,} BG (ratio 1:{n_bg/n_ice:.1f})")
    
    # Parse strategy
    if strategy != 'auto':
        try:
            strategy = float(strategy)
            target_ice = int(n_bg * strategy)
            print_rank0(f"Strategy: {strategy} -> Target ICE count: {target_ice:,}")
        except ValueError:
            print_rank0(f"Strategy: {strategy}")
    else:
        target_ice = n_bg
        print_rank0(f"Strategy: auto -> Target ICE count: {target_ice:,} (fully balanced)")
    
    X_float = X.astype(np.float32)

    # PCA reduction for SMOTE
    n_samples, n_features = X_float.shape
    max_components = min(500, n_features, n_samples - 1)
    
    print_rank0(f"Original dimensions: {n_features}")
    print_rank0(f"Reducing to {max_components} components for SMOTE...")
    
    # PCA reduction
    pca = PCA(n_components=max_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_float)
    
    print_rank0(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    k = min(k_neighbors, n_ice - 1)
    if k < 1:
        print_rank0("Too few ICE samples for SMOTE, using random oversampling")
        ice_indices = np.where(y == 1)[0]
        n_to_generate = n_bg - n_ice
        random_indices = np.random.choice(ice_indices, size=n_to_generate, replace=True)
        X_resampled = np.vstack([X, X[random_indices]])
        y_resampled = np.hstack([y, y[random_indices]])
        return X_resampled, y_resampled
    
    print_rank0(f"Using k_neighbors={k}")

    smote = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=k,
        random_state=random_state
    )
    
    X_reduced_resampled, y_resampled = smote.fit_resample(X_reduced, y)

    # Map synthetic samples back to original space
    n_original = len(X)
    n_new = len(X_reduced_resampled) - n_original

    X_new_reduced = X_reduced_resampled[n_original:]
    X_new_approx = pca.inverse_transform(X_new_reduced)
    X_new_tokens = np.clip(np.round(X_new_approx), 0, X.max()).astype(np.int32)
    X_resampled = np.vstack([X, X_new_tokens])
    
    n_ice_new = (y_resampled == 1).sum()
    n_bg_new = (y_resampled == 0).sum()
    print_rank0(f"After SMOTE: {n_ice_new:,} ICE, {n_bg_new:,} BG (ratio 1:{n_bg_new/n_ice_new:.1f})")
    print_rank0(f"Generated {n_new:,} synthetic ICE samples")
    print_rank0(f"{'='*60}\n")
    
    return X_resampled.astype(np.int32), y_resampled


def load_data_with_smote(data_dir, apply_smote_flag=True, smote_strategy='auto', smote_k=5):
    """Load data and optionally apply SMOTE to training set."""
    train_data = np.load(Path(data_dir) / 'train.npz')
    val_data = np.load(Path(data_dir) / 'val.npz')
    test_data = np.load(Path(data_dir) / 'test.npz')
    
    X_train, y_train = train_data['input_ids'], train_data['labels']
    X_val, y_val = val_data['input_ids'], val_data['labels']
    X_test, y_test = test_data['input_ids'], test_data['labels']
    
    print_rank0(f"Loaded data:")
    print_rank0(f"  Train: {len(X_train):,} samples")
    print_rank0(f"  Val: {len(X_val):,} samples")
    print_rank0(f"  Test: {len(X_test):,} samples")
    
    # Apply SMOTE to training set only (on rank 0)
    if apply_smote_flag and is_main_process():
        X_train, y_train = apply_smote(X_train, y_train, 
                                        strategy=smote_strategy, 
                                        k_neighbors=smote_k)
    
    # Sync SMOTE results across ranks
    if dist.is_initialized() and apply_smote_flag:
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
        
        # Broadcast the new data size
        if is_main_process():
            size_tensor = torch.tensor([len(X_train), X_train.shape[1]], dtype=torch.long, device=device)
        else:
            size_tensor = torch.zeros(2, dtype=torch.long, device=device)
        
        dist.broadcast(size_tensor, src=0)
        
        n_samples = size_tensor[0].item()
        seq_len = size_tensor[1].item()
        
        # Non-main ranks allocate space
        if not is_main_process():
            X_train = np.zeros((n_samples, seq_len), dtype=np.int32)
            y_train = np.zeros(n_samples, dtype=np.int64)
        
        # Broadcast data in chunks to avoid OOM
        chunk_size = 500000
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_len = end_idx - start_idx
            
            if is_main_process():
                X_chunk = torch.from_numpy(X_train[start_idx:end_idx].astype(np.int32)).to(device)
                y_chunk = torch.from_numpy(y_train[start_idx:end_idx].astype(np.int64)).to(device)
            else:
                X_chunk = torch.zeros((chunk_len, seq_len), dtype=torch.int32, device=device)
                y_chunk = torch.zeros(chunk_len, dtype=torch.int64, device=device)
            
            dist.broadcast(X_chunk, src=0)
            dist.broadcast(y_chunk, src=0)
            
            if not is_main_process():
                X_train[start_idx:end_idx] = X_chunk.cpu().numpy()
                y_train[start_idx:end_idx] = y_chunk.cpu().numpy()
            
            del X_chunk, y_chunk
            torch.cuda.empty_cache()
        
        dist.barrier()
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class TokenDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.int32) if X.dtype != np.int32 else X
        self.y = y.astype(np.int64) if y.dtype != np.int64 else y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {'input_ids': torch.from_numpy(self.X[idx].copy()).long(),
                'labels': torch.tensor(self.y[idx], dtype=torch.long)}


def compute_metrics(logits, labels, threshold=0.5):
    """Compute classification metrics."""
    from sklearn.metrics import (roc_auc_score, average_precision_score, 
                                  accuracy_score, precision_score, recall_score, 
                                  f1_score, confusion_matrix)
    
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    roc_auc = roc_auc_score(labels_np, probs)
    pr_auc = average_precision_score(labels_np, probs)
    
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels_np, preds)
    prec = precision_score(labels_np, preds, zero_division=0)
    rec = recall_score(labels_np, preds, zero_division=0)
    f1 = f1_score(labels_np, preds, zero_division=0)
    
    cm = confusion_matrix(labels_np, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'roc_auc': roc_auc, 'pr_auc': pr_auc,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def gather_variable_tensors(tensor, world_size, device):
    """Gather tensors of variable sizes from all ranks."""
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=device)
    
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [int(s.item()) for s in all_sizes]

    # Pad to max size for all_gather
    max_size = max(all_sizes)
    if tensor.size(0) < max_size:
        padding_shape = list(tensor.shape)
        padding_shape[0] = max_size - tensor.size(0)
        padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=device)
        tensor = torch.cat([tensor, padding], dim=0)

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    trimmed = [g[:s] for g, s in zip(gathered, all_sizes)]
    return torch.cat(trimmed, dim=0)


def find_optimal_threshold(logits, labels):
    """Find threshold that maximizes F1 on the PR curve."""
    from sklearn.metrics import precision_recall_curve
    
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    precision, recall, thresholds = precision_recall_curve(labels_np, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            x = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * len(y)
            all_logits.append(logits)
            all_labels.append(y)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    avg_loss = total_loss / len(all_labels)
    
    return all_logits, all_labels, avg_loss


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--max_len', type=int, default=1024)

    # Model
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--n_heads', type=int, default=12)
    parser.add_argument('--ffn_dim', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--pooling', type=str, default='attention')
    parser.add_argument('--use_alibi', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_conv_stem', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--use_focal_loss', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--focal_alpha', type=float, default=0.8)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.05)

    parser.add_argument('--use_smote', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--smote_strategy', type=str, default='auto')
    parser.add_argument('--smote_k', type=int, default=5)

    parser.add_argument('--early_stopping_patience', type=int, default=15)
    parser.add_argument('--metric', type=str, default='pr_auc')
    parser.add_argument('--optimize_threshold', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--outdir', type=str, required=True)
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()
    
    print_rank0(f"\n{'='*60}")
    print_rank0("DDP TRAINING")
    print_rank0(f"{'='*60}")
    print_rank0(f"World size: {world_size} GPUs")
    print_rank0(f"Data: {args.data_dir}")
    print_rank0(f"SMOTE: {'ENABLED' if args.use_smote else 'DISABLED'}")
    print_rank0(f"Model: d={args.d_model}, L={args.n_layers}, H={args.n_heads}")
    print_rank0(f"{'='*60}\n")
    
    if is_main_process():
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
    dist.barrier()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data_with_smote(
        args.data_dir,
        apply_smote_flag=args.use_smote,
        smote_strategy=args.smote_strategy,
        smote_k=args.smote_k
    )

    train_dataset = TokenDataset(X_train, y_train)
    val_dataset = TokenDataset(X_val, y_val)
    test_dataset = TokenDataset(X_test, y_test)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             sampler=test_sampler, num_workers=args.num_workers,
                             pin_memory=True)
    
    print_rank0(f"Train: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print_rank0(f"Val: {len(val_dataset):,} samples")
    print_rank0(f"Test: {len(test_dataset):,} samples\n")

    model = BERTForICE(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        pooling=args.pooling,
        use_alibi=args.use_alibi,
        use_conv_stem=args.use_conv_stem
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        print_rank0(f"Model parameters: {total_params:,}")

    if args.use_focal_loss:
        criterion = FocalLoss(
            alpha=args.focal_alpha, 
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
        print_rank0(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma}, label_smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print_rank0(f"Using CrossEntropy Loss (label_smoothing={args.label_smoothing})")

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if 'bias' in name or 'LayerNorm' in name or 'layer_norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_metric = 0.0
    patience_counter = 0
    best_threshold = 0.5
    start_epoch = 0
    
    scaler = torch.cuda.amp.GradScaler()

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print_rank0(f"\n{'='*60}")
            print_rank0(f"RESUMING FROM CHECKPOINT: {resume_path}")
            print_rank0(f"{'='*60}")
            
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint.get('epoch', 0) + 1
            best_metric = checkpoint.get('best_metric', 0.0)
            best_threshold = checkpoint.get('threshold', 0.5)

            if 'patience_counter' in checkpoint:
                patience_counter = checkpoint['patience_counter']
            elif 'RESUME_PATIENCE' in os.environ:
                patience_counter = int(os.environ['RESUME_PATIENCE'])
                print_rank0(f"  Using RESUME_PATIENCE from environment: {patience_counter}")
            else:
                patience_counter = 0

            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                for _ in range(start_epoch * len(train_loader)):
                    scheduler.step()

            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            print_rank0(f"  Resuming from epoch {start_epoch}")
            print_rank0(f"  Best {args.metric}: {best_metric:.4f}")
            print_rank0(f"  Patience: {patience_counter}/{args.early_stopping_patience}")
            print_rank0(f"{'='*60}\n")
        else:
            print_rank0(f"WARNING: Checkpoint not found at {resume_path}, starting from scratch")
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            x = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0 and is_main_process():
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.2e}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        val_logits, val_labels, val_loss = evaluate(model, val_loader, criterion, device)
        
        # Gather from all ranks (handles variable sizes)
        if dist.is_initialized():
            val_logits = gather_variable_tensors(val_logits, world_size, device)
            val_labels = gather_variable_tensors(val_labels, world_size, device)
        
        if is_main_process():
            # Find optimal threshold
            if args.optimize_threshold:
                threshold = find_optimal_threshold(val_logits, val_labels)
            else:
                threshold = 0.5
            
            metrics = compute_metrics(val_logits, val_labels, threshold)
            
            print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Threshold: {threshold:.3f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']} | TN: {metrics['tn']}\n")
            
            # Model selection
            current_metric = metrics[args.metric]
            
            if current_metric > best_metric:
                best_metric = current_metric
                best_threshold = threshold
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_metric': best_metric,
                    'threshold': best_threshold,
                    'patience_counter': patience_counter,
                    'args': vars(args),
                    'metrics': metrics
                }, Path(args.outdir) / 'best.pt')
                
                print(f"New best {args.metric}: {best_metric:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")
            
            # Always save latest checkpoint (for resuming even if no improvement)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_metric': best_metric,
                'threshold': best_threshold,
                'patience_counter': patience_counter,
                'args': vars(args),
                'metrics': metrics
            }, Path(args.outdir) / 'latest.pt')
            
        # Broadcast patience_counter to all ranks
        patience_tensor = torch.tensor([patience_counter], dtype=torch.int32, device=device)
        dist.broadcast(patience_tensor, src=0)
        patience_counter = patience_tensor.item()  # Update on all ranks
        
        # Check early stopping on all ranks
        if patience_counter >= args.early_stopping_patience:
            if is_main_process():
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Test evaluation
    if is_main_process():
        print(f"\n{'='*60}")
        print("FINAL TEST EVALUATION")
        print(f"{'='*60}")
        
        # Load best model
        checkpoint = torch.load(Path(args.outdir) / 'best.pt', weights_only=False)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        threshold = checkpoint['threshold']
    
    dist.barrier()
    
    test_logits, test_labels, test_loss = evaluate(model, test_loader, criterion, device)
    
    # Gather from all ranks (handles variable sizes)
    if dist.is_initialized():
        test_logits = gather_variable_tensors(test_logits, world_size, device)
        test_labels = gather_variable_tensors(test_labels, world_size, device)
    
    if is_main_process():
        test_metrics = compute_metrics(test_logits, test_labels, threshold)
        
        print(f"\nTest Results (threshold={threshold:.3f}):")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  TP: {test_metrics['tp']} | FP: {test_metrics['fp']} | "
              f"FN: {test_metrics['fn']} | TN: {test_metrics['tn']}")
        
        # Save final results
        results = {
            'train': {'loss': avg_train_loss},
            'val': checkpoint['metrics'],
            'test': test_metrics,
            'threshold': threshold,
            'args': vars(args),
            'smote_applied': args.use_smote
        }
        
        with open(Path(args.outdir) / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {args.outdir}")
        print(f"{'='*60}\n")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
