import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import tensorflow as tf
import numpy as np

# Load a small sample
TRAIN_DIR = Path("data/gold/train")
BATCH_SIZE = 1024

def load_parquet_files(directory):
    """Load all parquet files from a directory and concatenate them."""
    parquet_files = sorted(directory.glob("*.parquet"))[:2]  # Just first 2 files

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(df)} samples")
    return df

def df_to_tf_dataset(df, target_col, batch_size, shuffle=True):
    """Convert a pandas DataFrame to a TensorFlow dataset."""

    feature_cols = {
        "inning": df["inning"].values.astype("float32"),
        "balls_before_pitch": df["balls_before_pitch"].values.astype("float32"),
        "strikes_before_pitch": df["strikes_before_pitch"].values.astype("float32"),
        "pitcher_is_home": df["pitcher_is_home"].values.astype("float32"),
        "pitcher_is_right": df["pitcher_is_right"].values.astype("float32"),
        "batter_is_right": df["batter_is_right"].values.astype("float32"),
        "pitch_num_in_pa": df["pitch_num_in_pa"].values.astype("float32"),
        "pitcher_idx": df["pitcher_idx"].values.astype("int32"),
        "batter_idx": df["batter_idx"].values.astype("int32"),
        "pitch_1_idx": df["pitch_1_idx"].values.astype("int32"),
        "pitch_2_idx": df["pitch_2_idx"].values.astype("int32"),
        "pitch_3_idx": df["pitch_3_idx"].values.astype("int32"),
        "pitch_1_bucket_idx": df["pitch_1_bucket_idx"].values.astype("int32"),
        "pitch_2_bucket_idx": df["pitch_2_bucket_idx"].values.astype("int32"),
        "pitch_3_bucket_idx": df["pitch_3_bucket_idx"].values.astype("int32"),
    }

    # Target variable - shift to 0-indexed
    target = (df[target_col].values - 1).astype("int32")

    print(f"\nTarget statistics BEFORE dataset creation:")
    print(f"  Min: {target.min()}, Max: {target.max()}")
    print(f"  Sample: {target[:10]}")

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((feature_cols, target))
    dataset = dataset.cache()

    if shuffle:
        shuffle_buffer = min(50000, len(df))
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=42, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Load data
train_df = load_parquet_files(TRAIN_DIR)
train_ds = df_to_tf_dataset(train_df, "pitch_idx", BATCH_SIZE, shuffle=True)

# Get first batch
for features, targets in train_ds.take(1):
    print(f"\nFirst batch:")
    print(f"  Batch size: {targets.shape[0]}")
    print(f"  Target dtype: {targets.dtype}")
    print(f"  Target min: {tf.reduce_min(targets).numpy()}")
    print(f"  Target max: {tf.reduce_max(targets).numpy()}")
    print(f"  Target sample: {targets.numpy()[:10]}")
    print(f"  Any NaN: {tf.reduce_any(tf.math.is_nan(tf.cast(targets, tf.float32))).numpy()}")
    print(f"  Any < 0: {tf.reduce_any(targets < 0).numpy()}")
    print(f"  Any >= 13: {tf.reduce_any(targets >= 13).numpy()}")

    # Check pitcher/batter indices
    print(f"\n  Pitcher idx min: {tf.reduce_min(features['pitcher_idx']).numpy()}")
    print(f"  Pitcher idx max: {tf.reduce_max(features['pitcher_idx']).numpy()}")
    print(f"  Batter idx min: {tf.reduce_min(features['batter_idx']).numpy()}")
    print(f"  Batter idx max: {tf.reduce_max(features['batter_idx']).numpy()}")
