#!/usr/bin/env python3
"""Run inference on test set with a trained model checkpoint."""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix
)
from model import BERTForICE


class NPZDataset(torch.utils.data.Dataset):
    """Dataset for loading NPZ files with input_ids and labels."""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.input_ids = data["input_ids"]
        self.labels = data["labels"]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return torch.from_numpy(self.input_ids[i]).long(), torch.tensor(self.labels[i]).long()


def find_optimal_threshold(probs, labels, metric='f1'):
    """Find optimal classification threshold using validation data."""
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    if metric == 'f1':
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return 0.5


def evaluate_model(model, dataloader, device, threshold=0.5):
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Handle labels shape (squeeze if needed)
            if labels.dim() > 1:
                labels = labels.squeeze(1)
            
            outputs = model(input_ids)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'pr_auc': average_precision_score(all_labels, all_probs),
        'threshold': threshold,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }
    
    return metrics, all_probs, all_labels


def main():
    parser = argparse.ArgumentParser(description="Test inference on saved model checkpoint")
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing best.pt checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing train.npz/val.npz/test.npz')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which split to evaluate (default: test)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Use explicit threshold (overrides checkpoint and optimization)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='Find optimal threshold on validation set')
    parser.add_argument('--use_data_parallel', action='store_true',
                        help='Use DataParallel for multi-GPU inference')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (default: <checkpoint_dir>/test_results.json)')
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    
    # Check for model file with various possible names
    model_path = None
    possible_names = ["best.pt", "best_model.pt", "best_finetuned.pt", "model.pt", "checkpoint.pt"]
    for name in possible_names:
        candidate = checkpoint_dir / name
        if candidate.exists():
            model_path = candidate
            break
    
    if model_path is None:
        print(f"ERROR: Model not found in: {checkpoint_dir}")
        print(f"   Looked for: {possible_names}")
        return 1
    
    # Check requested split file
    split_path = data_dir / f"{args.split}.npz"
    if not split_path.exists():
        print(f"ERROR: Data for split '{args.split}' not found: {split_path}")
        return 1
    
    print("=" * 70)
    print("TEST SET INFERENCE")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Model file: {model_path.name}")
    print(f"Data dir:   {data_dir}")
    print(f"Split:      {args.split}")
    print(f"Batch size: {args.batch_size}")
    print("", flush=True)
    
    # Load checkpoint
    print("Loading checkpoint...", flush=True)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print("Checkpoint loaded", flush=True)
    
    # Extract config from checkpoint
    if 'args' in checkpoint:
        args_dict = checkpoint['args']
        if not isinstance(args_dict, dict):
            args_dict = vars(args_dict)
        
        config = {
            'vocab_size': args_dict.get('vocab_size', 50000),
            'pad_id': 1,
            'max_len': args_dict.get('max_len', 1024),
            'd_model': args_dict.get('d_model', 768),
            'n_layers': args_dict.get('n_layers', 12),
            'n_heads': args_dict.get('n_heads', 12),
            'ffn_dim': args_dict.get('ffn_dim', 3072),
            'dropout': args_dict.get('dropout', 0.15),
            'pooling': args_dict.get('pooling', 'attention'),
            'use_alibi': args_dict.get('use_alibi', True),
            'use_conv_stem': args_dict.get('use_conv_stem', True),
        }
        print("Config loaded from checkpoint['args']")
        
        # Get saved threshold if available
        saved_threshold = checkpoint.get('threshold', None)
        if saved_threshold:
            print(f"   Saved threshold from training: {saved_threshold:.4f}")
    else:
        # Fallback defaults for large model
        print("WARNING: No args in checkpoint, using large model defaults")
        config = {
            'vocab_size': 50000,
            'pad_id': 1,
            'max_len': 1024,
            'd_model': 768,
            'n_layers': 12,
            'n_heads': 12,
            'ffn_dim': 3072,
            'dropout': 0.15,
            'pooling': 'attention',
            'use_alibi': True,
            'use_conv_stem': True,
        }
        saved_threshold = None
    
    print(f"\nModel Configuration:")
    print(f"  Vocab size:    {config['vocab_size']}")
    print(f"  Max length:    {config['max_len']}")
    print(f"  d_model:       {config['d_model']}")
    print(f"  n_layers:      {config['n_layers']}")
    print(f"  n_heads:       {config['n_heads']}")
    print(f"  ffn_dim:       {config['ffn_dim']}")
    print(f"  dropout:       {config['dropout']}")
    print(f"  pooling:       {config['pooling']}")
    print(f"  use_alibi:     {config['use_alibi']}")
    print(f"  use_conv_stem: {config['use_conv_stem']}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"\nDevice: {device}", flush=True)
    if n_gpus > 1 and args.use_data_parallel:
        print(f"Using DataParallel with {n_gpus} GPUs", flush=True)
    elif n_gpus > 1:
        print(f"{n_gpus} GPUs available (use --use_data_parallel to enable)", flush=True)
    
    # Initialize model
    print("\nInitializing model architecture...", flush=True)
    model = BERTForICE(
        vocab_size=config['vocab_size'],
        pad_id=config['pad_id'],
        max_len=config['max_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        ffn_dim=config['ffn_dim'],
        dropout=config['dropout'],
        pooling=config['pooling'],
        use_alibi=config['use_alibi'],
        use_conv_stem=config.get('use_conv_stem', False),
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model weights loaded")
    
    # Wrap in DataParallel if requested and multiple GPUs available
    if args.use_data_parallel and n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel enabled ({n_gpus} GPUs)")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {n_params:,}")
    
    # Load split data
    print(f"\nLoading {args.split} data...")
    split_dataset = NPZDataset(split_path)
    split_loader = DataLoader(
        split_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    print(f"  {args.split} samples: {len(split_dataset):,}")
    
    # Determine threshold
    if args.threshold is not None:
        # Explicit threshold provided
        threshold = float(args.threshold)
        print(f"\nUsing explicit threshold: {threshold:.4f}")
    elif args.optimize_threshold:
        # Optimize on validation set
        val_path = data_dir / "val.npz"
        if val_path.exists():
            print("\nOptimizing threshold on validation set...")
            val_dataset = NPZDataset(val_path)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True
            )
            
            # Get validation probabilities
            model.eval()
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for input_ids, labels in val_loader:
                    input_ids = input_ids.to(device)
                    if labels.dim() > 1:
                        labels = labels.squeeze(1)
                    outputs = model(input_ids)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    val_probs.extend(probs.cpu().numpy())
                    val_labels.extend(labels.numpy())
            
            val_probs = np.array(val_probs)
            val_labels = np.array(val_labels)
            threshold = find_optimal_threshold(val_probs, val_labels, metric='f1')
            print(f"Optimal threshold (from val): {threshold:.4f}")
        else:
            threshold = saved_threshold if saved_threshold else 0.5
            print(f"WARNING: Val set not found, using threshold: {threshold:.4f}")
    elif saved_threshold is not None:
        # Use threshold from training
        threshold = saved_threshold
        print(f"\nUsing saved threshold from training: {threshold:.4f}")
    else:
        threshold = 0.5
        print(f"\nUsing default threshold: {threshold:.4f}")
    
    # Run inference
    print(f"\nRunning inference on {args.split} set...")
    metrics, probs, labels = evaluate_model(model, split_loader, device, threshold)
    
    # Print results
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"Threshold:  {metrics['threshold']:.4f}")
    print(f"")
    print(f"PR-AUC:     {metrics['pr_auc']:.4f}  ← Primary metric")
    print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"F1:         {metrics['f1']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                 Predicted")
    print(f"              Neg      Pos")
    print(f"  Actual Neg  {cm['tn']:>7,}  {cm['fp']:>7,}  (TN, FP)")
    print(f"  Actual Pos  {cm['fn']:>7,}  {cm['tp']:>7,}  (FN, TP)")
    print("=" * 70)
    
    # Class distribution
    pos_count = int(labels.sum())
    neg_count = len(labels) - pos_count
    print(f"\nTest Set Distribution:")
    print(f"  Positive (ICE):     {pos_count:,} ({pos_count/len(labels)*100:.2f}%)")
    print(f"  Negative (BG):      {neg_count:,} ({neg_count/len(labels)*100:.2f}%)")
    
    # Prediction distribution
    pred_pos = int((probs >= threshold).sum())
    pred_neg = len(probs) - pred_pos
    print(f"\nPrediction Distribution:")
    print(f"  Predicted Positive: {pred_pos:,} ({pred_pos/len(probs)*100:.2f}%)")
    print(f"  Predicted Negative: {pred_neg:,} ({pred_neg/len(probs)*100:.2f}%)")
    
    # Save results
    output_path = args.output if args.output else checkpoint_dir / "test_results.json"
    
    results = {
        'checkpoint': str(checkpoint_dir),
        'model_file': str(model_path.name),
        'data_dir': str(data_dir),
        'split': args.split,
        'n_samples': len(split_dataset),
        'threshold': float(threshold),
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer, float)) else v 
                   for k, v in metrics.items()},
        'config': config,
        'class_distribution': {
            'positive': pos_count,
            'negative': neg_count,
            'positive_pct': pos_count / len(labels) * 100
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
