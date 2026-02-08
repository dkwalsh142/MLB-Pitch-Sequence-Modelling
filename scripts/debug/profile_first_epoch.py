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
    pitcher_lookup = pd.read_parquet(PITCHER_LOOKUP)
    batter_lookup = pd.read_parquet(BATTER_LOOKUP)
    pitch_lookup = pd.read_parquet(PITCH_LOOKUP)

    n_pitchers = len(pitcher_lookup) + 1
    n_batters = len(batter_lookup) + 1
    n_pitch_types = len(pitch_lookup)
    n_pitch_buckets = pitch_lookup["pitch_bucket_idx"].nunique()

    return n_pitchers, n_batters, n_pitch_types, n_pitch_buckets


def load_parquet_files(directory):
    parquet_files = sorted(directory.glob("*.parquet"))
    print(f"Loading {len(parquet_files)} files...")
    start = time.time()
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  ✓ Loaded {len(df):,} samples in {time.time() - start:.2f}s")
    return df


def df_to_tf_dataset(df, target_col, batch_size, shuffle=True):
    print(f"\nCreating TensorFlow dataset...")
    start = time.time()

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
    dataset = dataset.cache()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print(f"  ✓ Dataset created in {time.time() - start:.2f}s")
    return dataset


def main():
    print("=" * 70)
    print("PROFILING FIRST EPOCH - DETAILED TIMING")
    print("=" * 70)

    # Load vocab
    print("\n[1/6] Loading vocabulary sizes...")
    start = time.time()
    n_pitchers, n_batters, n_pitch_types, n_pitch_buckets = load_vocab_sizes()
    print(f"  ✓ Done in {time.time() - start:.2f}s")

    # Load data
    print("\n[2/6] Loading training data...")
    start = time.time()
    train_df = load_parquet_files(TRAIN_DIR)
    data_load_time = time.time() - start
    print(f"  ✓ Total data loading: {data_load_time:.2f}s")

    # Create dataset
    print("\n[3/6] Creating TensorFlow dataset...")
    start = time.time()
    train_ds = df_to_tf_dataset(train_df, "pitch_idx", BATCH_SIZE, shuffle=True)
    dataset_time = time.time() - start
    print(f"  ✓ Dataset creation: {dataset_time:.2f}s")

    # Build model
    print("\n[4/6] Building model...")
    start = time.time()
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
    print(f"  ✓ Model built in {time.time() - start:.2f}s")
    print(f"  ✓ Total parameters: {model.count_params():,}")

    # Test data iteration
    print("\n[5/6] Testing data iteration (first 10 batches)...")
    start = time.time()
    for i, (batch_x, batch_y) in enumerate(train_ds):
        elapsed = time.time() - start
        if i == 0:
            print(f"  → Batch 1 (first): {elapsed:.2f}s")
        elif i < 10:
            print(f"  → Batch {i+1}: {elapsed:.2f}s")
        if i >= 9:
            break

    total_iteration_time = time.time() - start
    print(f"  ✓ 10 batches iterated in {total_iteration_time:.2f}s")
    print(f"  ✓ Avg time per batch: {total_iteration_time / 10:.3f}s")

    num_batches = (len(train_df) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  ✓ Projected full iteration: {(total_iteration_time / 10) * num_batches:.1f}s")

    # Run actual training for 1 epoch
    print("\n[6/6] Running actual training (1 epoch)...")
    print("=" * 70)

    epoch_start = time.time()

    # Custom callback to track batch progress
    class DetailedCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.batch_times = []
            self.last_time = None

        def on_train_begin(self, logs=None):
            self.train_start = time.time()
            print(f"Training started at t=0.0s")

        def on_train_batch_begin(self, batch, logs=None):
            self.last_time = time.time()

        def on_train_batch_end(self, batch, logs=None):
            batch_time = time.time() - self.last_time
            self.batch_times.append(batch_time)

            if batch == 0:
                print(f"  → Batch 1 completed in {batch_time:.2f}s")
            elif batch < 10:
                print(f"  → Batch {batch+1} completed in {batch_time:.2f}s (cumulative: {time.time() - self.train_start:.1f}s)")
            elif batch % 50 == 0:
                avg_recent = sum(self.batch_times[-50:]) / len(self.batch_times[-50:])
                print(f"  → Batch {batch+1}/{num_batches} - Avg time: {avg_recent:.3f}s/batch (cumulative: {time.time() - self.train_start:.1f}s)")

        def on_epoch_end(self, epoch, logs=None):
            total = time.time() - self.train_start
            avg = sum(self.batch_times) / len(self.batch_times)
            print(f"\n  ✓ Epoch completed in {total:.1f}s")
            print(f"  ✓ Average time per batch: {avg:.3f}s")
            print(f"  ✓ Slowest batch: {max(self.batch_times):.2f}s")
            print(f"  ✓ Fastest batch: {min(self.batch_times):.2f}s")

    callback = DetailedCallback()

    history = model.fit(
        train_ds,
        epochs=1,
        callbacks=[callback],
        verbose=0  # Suppress default output
    )

    total_epoch_time = time.time() - epoch_start

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total epoch time: {total_epoch_time:.1f}s")
    print(f"Training accuracy: {history.history['sparse_categorical_accuracy'][0]:.4f}")

    print("\nBreakdown:")
    print(f"  Data loading:      {data_load_time:.1f}s")
    print(f"  Dataset creation:  {dataset_time:.1f}s")
    print(f"  Actual training:   {total_epoch_time:.1f}s")

    if total_epoch_time > 60:
        print(f"\n⚠️  WARNING: Epoch took {total_epoch_time:.0f}s ({total_epoch_time/60:.1f} minutes)")
        print("This is unexpectedly slow. Possible issues:")
        print("  1. Disk I/O bottleneck (unlikely with cached dataset)")
        print("  2. Memory swapping (check Activity Monitor)")
        print("  3. CPU/GPU contention from other processes")
        print("  4. Large shuffle buffer causing memory issues")
    else:
        print(f"\n✓ Performance looks good!")


if __name__ == "__main__":
    main()
