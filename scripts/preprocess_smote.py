import argparse
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def apply_smote(X, y, strategy='auto', k_neighbors=5, random_state=42, vocab_size=None):
    from imblearn.over_sampling import SMOTE
    
    # Determine vocab_size from data if not provided
    if vocab_size is None:
        vocab_size = int(X.max()) + 1
        print(f"Auto-detected vocab_size: {vocab_size}")
    
    print(f"\n{'='*60}")
    print("SMOTE OVERSAMPLING")
    print(f"{'='*60}")
    
    n_ice = int((y == 1).sum())
    n_bg = int((y == 0).sum())
    print(f"Before SMOTE: {n_ice:,} ICE, {n_bg:,} BG (ratio 1:{n_bg/n_ice:.1f})")
    
    if strategy != 'auto':
        try:
            strategy = float(strategy)
            target_ice = int(n_bg * strategy)
            print(f"Strategy: {strategy} : Target ICE count: {target_ice:,}")
        except ValueError:
            print(f"Strategy: {strategy}")
    else:
        target_ice = n_bg
        print(f"Strategy: Target ICE count: {target_ice:,} (fully balanced)")
    
    X_float = X.astype(np.float32)
    
    # PCA reduction for SMOTE
    n_samples, n_features = X_float.shape
    max_components = min(500, n_features, n_samples - 1)
    
    print(f"Original dimensions: {n_features}")
    print(f"Reducing to {max_components} components for SMOTE...")
    
    pca = PCA(n_components=max_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_float)
    
    print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    # Adjust k_neighbors
    k = min(k_neighbors, n_ice - 1)
    if k < 1:
        print("Too few ICE samples for SMOTE, using random oversampling")
        ice_indices = np.where(y == 1)[0]
        n_to_generate = target_ice - n_ice
        random_indices = np.random.choice(ice_indices, size=n_to_generate, replace=True)
        X_resampled = np.vstack([X, X[random_indices]])
        y_resampled = np.hstack([y, y[random_indices]])
        return X_resampled, y_resampled
    
    print(f"Using k_neighbors={k}")
    
    smote = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=k,
        random_state=random_state
    )
    
    X_reduced_resampled, y_resampled = smote.fit_resample(X_reduced, y)
    
    # Map back to original space
    n_original = len(X)
    n_new = len(X_reduced_resampled) - n_original
    
    X_new_reduced = X_reduced_resampled[n_original:]
    X_new_approx = pca.inverse_transform(X_new_reduced)
    # Clip to valid token range [0, vocab_size-1]
    X_new_tokens = np.clip(np.round(X_new_approx), 0, vocab_size - 1).astype(np.int32)
    
    X_resampled = np.vstack([X, X_new_tokens])
    
    n_ice_new = (y_resampled == 1).sum()
    n_bg_new = (y_resampled == 0).sum()
    print(f"After SMOTE: {n_ice_new:,} ICE, {n_bg_new:,} BG (ratio 1:{n_bg_new/n_ice_new:.1f})")
    print(f"Generated {n_new:,} synthetic ICE samples")
    print(f"{'='*60}\n")
    
    return X_resampled.astype(np.int32), y_resampled.astype(np.int64)


def main():
    parser = argparse.ArgumentParser(description="Preprocess data with SMOTE")
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to encoded data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data_dir + _SMOTE)')
    parser.add_argument('--strategy', type=str, default='0.2',
                        help="SMOTE strategy: 'auto' for balanced, or float like '0.2'")
    parser.add_argument('--k_neighbors', type=int, default=5,
                        help='K neighbors for SMOTE')
    parser.add_argument('--vocab_size', type=int, default=50000,
                        help='Vocabulary size for clipping tokens')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir.parent / (data_dir.name + f'_SMOTE_{args.strategy}')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Strategy: {args.strategy}")
    print(f"K neighbors: {args.k_neighbors}")
    print(f"Vocab size: {args.vocab_size}")
    
    # Load training data
    print("\nLoading training data...")
    train_data = np.load(data_dir / 'train.npz')
    X_train = train_data['input_ids']
    y_train = train_data['labels']
    print(f"Loaded {len(X_train):,} training samples")
    
    # Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(
        X_train, y_train, 
        strategy=args.strategy,
        k_neighbors=args.k_neighbors,
        random_state=args.seed,
        vocab_size=args.vocab_size
    )
    
    # Shuffle the augmented data
    print("Shuffling augmented data")
    indices = np.random.permutation(len(X_train_smote))
    X_train_smote = X_train_smote[indices]
    y_train_smote = y_train_smote[indices]
    
    # Verify token range
    max_token = X_train_smote.max()
    min_token = X_train_smote.min()
    print(f"Token range in augmented data: [{min_token}, {max_token}]")
    if max_token >= args.vocab_size:
        print(f"ERROR: Max token {max_token} >= vocab_size {args.vocab_size}!")
        raise ValueError(f"Token out of range: {max_token} >= {args.vocab_size}")
    print(f"All tokens within valid range [0, {args.vocab_size - 1}]")
    
    # Save augmented training data
    print(f"Saving augmented training data to {output_dir / 'train.npz'}")
    np.savez_compressed(
        output_dir / 'train.npz',
        input_ids=X_train_smote,
        labels=y_train_smote
    )
    
    # Copy val and test unchanged
    print("Copying val.npz and test.npz (unchanged)")
    import shutil
    shutil.copy(data_dir / 'val.npz', output_dir / 'val.npz')
    shutil.copy(data_dir / 'test.npz', output_dir / 'test.npz')
    
    # Summary
    print(f"\n{'='*60}")
    print("SMOTE PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Train samples: {len(X_train):,} -> {len(X_train_smote):,}")
    print(f"ICE samples: {(y_train == 1).sum():,} -> {(y_train_smote == 1).sum():,}")
    print(f"\nYou can now train with:")
    print(f"  python train_ddp.py --data_dir {output_dir} ...")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
