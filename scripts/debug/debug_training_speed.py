import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import tensorflow as tf
import time
from src.models.pitch_sequence_baseline import build_pitch_model

# Paths
TRAIN_DIR = Path("data/gold/train")
PITCHER_LOOKUP = Path("data/silver/dims/pitcher_lookup.parquet")
BATTER_LOOKUP = Path("data/silver/dims/batter_lookup.parquet")
PITCH_LOOKUP = Path("data/silver/dims/pitch_lookup.parquet")

BATCH_SIZE = 1024

def load_vocab_sizes():
    """Load lookup tables to determine vocabulary sizes."""
    pitcher_lookup = pd.read_parquet(PITCHER_LOOKUP)
    batter_lookup = pd.read_parquet(BATTER_LOOKUP)
    pitch_lookup = pd.read_parquet(PITCH_LOOKUP)

    n_pitchers = len(pitcher_lookup) + 1
    n_batters = len(batter_lookup) + 1
    n_pitch_types = len(pitch_lookup)
    n_pitch_buckets = pitch_lookup["pitch_bucket_idx"].nunique()

    return n_pitchers, n_batters, n_pitch_types, n_pitch_buckets


def load_parquet_files(directory):
    """Load all parquet files from a directory and concatenate them."""
    parquet_files = sorted(directory.glob("*.parquet"))

    print(f"Found {len(parquet_files)} parquet files")

    start = time.time()
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    elapsed = time.time() - start

    print(f"Loaded {len(df):,} samples in {elapsed:.2f} seconds")
    print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

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

    target = (df[target_col].values - 1).astype("int32")

    dataset = tf.data.Dataset.from_tensor_slices((feature_cols, target))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, seed=42)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def main():
    print("=" * 60)
    print("Training Speed Diagnostic")
    print("=" * 60)

    # Check TensorFlow/GPU setup
    print("\n1. TensorFlow Configuration:")
    print(f"   TensorFlow version: {tf.__version__}")
    print(f"   GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"   Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")

    # Load vocab sizes
    print("\n2. Loading vocabulary sizes...")
    n_pitchers, n_batters, n_pitch_types, n_pitch_buckets = load_vocab_sizes()
    print(f"   Pitchers: {n_pitchers:,}")
    print(f"   Batters: {n_batters:,}")
    print(f"   Pitch types: {n_pitch_types}")

    # Load data
    print("\n3. Loading training data...")
    start = time.time()
    train_df = load_parquet_files(TRAIN_DIR)
    data_load_time = time.time() - start
    print(f"   Total data loading time: {data_load_time:.2f} seconds")

    # Create dataset
    print("\n4. Creating TensorFlow dataset...")
    start = time.time()
    train_ds = df_to_tf_dataset(train_df, "pitch_idx", BATCH_SIZE, shuffle=True)
    dataset_create_time = time.time() - start
    print(f"   Dataset creation time: {dataset_create_time:.2f} seconds")

    # Count batches
    num_samples = len(train_df)
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"   Total samples: {num_samples:,}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Batches per epoch: {num_batches:,}")

    # Test data iteration
    print("\n5. Testing data iteration speed...")
    start = time.time()
    for i, (batch_x, batch_y) in enumerate(train_ds):
        if i >= 10:  # Test first 10 batches
            break
    iteration_time = time.time() - start
    print(f"   Time for 10 batches: {iteration_time:.2f} seconds")
    print(f"   Estimated time per batch: {iteration_time / 10:.3f} seconds")
    print(f"   Projected epoch time (data only): {(iteration_time / 10) * num_batches:.2f} seconds")

    # Build model
    print("\n6. Building model...")
    model = build_pitch_model(
        n_pitchers=n_pitchers,
        n_batters=n_batters,
        n_pitch_types=n_pitch_types,
        n_pitch_buckets=n_pitch_buckets,
        predict="pitch_type",
        pitcher_emb_dim=64,
        batter_emb_dim=64,
        pitch_emb_dim=24,
        bucket_emb_dim=12,
        hidden_units=(512, 256, 128),
        dropout=0.3,
    )

    # Count parameters
    total_params = model.count_params()
    print(f"   Total parameters: {total_params:,}")

    # Test forward pass
    print("\n7. Testing model forward pass speed...")
    train_ds_for_test = df_to_tf_dataset(train_df, "pitch_idx", BATCH_SIZE, shuffle=False)
    batch_x, batch_y = next(iter(train_ds_for_test))

    # Warmup
    _ = model(batch_x, training=True)

    # Time forward passes
    start = time.time()
    for _ in range(10):
        _ = model(batch_x, training=True)
    forward_time = time.time() - start
    print(f"   Time for 10 forward passes: {forward_time:.2f} seconds")
    print(f"   Time per forward pass: {forward_time / 10:.3f} seconds")
    print(f"   Projected epoch time (forward only): {(forward_time / 10) * num_batches:.2f} seconds")

    # Test train step
    print("\n8. Testing full training step (forward + backward)...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Warmup
    _ = train_step(batch_x, batch_y)

    # Time training steps
    start = time.time()
    for _ in range(10):
        _ = train_step(batch_x, batch_y)
    train_time = time.time() - start
    print(f"   Time for 10 training steps: {train_time:.2f} seconds")
    print(f"   Time per training step: {train_time / 10:.3f} seconds")
    print(f"   Projected epoch time (full training): {(train_time / 10) * num_batches:.2f} seconds")

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print(f"Expected time per epoch: ~{(train_time / 10) * num_batches:.1f} seconds")
    print(f"Expected time for 50 epochs: ~{(train_time / 10) * num_batches * 50 / 60:.1f} minutes")
    print("\nBottleneck analysis:")
    print(f"  Data iteration: {(iteration_time / 10) * num_batches:.1f}s per epoch")
    print(f"  Model forward pass: {(forward_time / 10) * num_batches:.1f}s per epoch")
    print(f"  Full training step: {(train_time / 10) * num_batches:.1f}s per epoch")

    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("\n⚠️  WARNING: No GPU detected! Training on CPU will be much slower.")
        print("   Consider using a GPU-enabled environment for faster training.")


if __name__ == "__main__":
    main()
